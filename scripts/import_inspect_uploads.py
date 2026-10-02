#!/usr/bin/env python3
"""
import_inspect_uploads.py — Range les fichiers déposés dans results/inbox/ :
  - les *.eval vers results/inspect_evals/{model_family}/ (structure attendue
    par extract_inspect_summary.py) ;
  - les *.gguf_meta.json (conditions de banc par quant) vers
    results/{model_family}/run_meta/{quant}.json — jamais lu par
    extract_inspect_summary.py, uniquement pour archive/consultation ;
  - bench_config.json (config partagée par le batch) en snapshot vers
    results/{model_family}/run_meta/bench_config.json pour chaque famille
    touchée par le batch.

Pourquoi : le banc distant écrit ses logs sous ./logs/{model}-{quant}.gguf/,
donc chaque upload arrive avec une arborescence par quant (et souvent des
fichiers déjà importés lors d'un upload précédent). Ce script :
  - retrouve tous les *.eval sous le dossier d'upload, quelle que soit la
    profondeur d'imbrication ;
  - lit chaque header.json pour déterminer la famille de modèle (même
    logique que extract_inspect_summary.py — indépendant du nom de dossier
    d'origine) ;
  - déplace chaque fichier vers results/inspect_evals/{family}/ (à plat) ;
  - si un fichier de même nom existe déjà à destination : le supprime de
    l'upload si le contenu est identique (ré-upload), ou le laisse en place
    avec un avertissement si le contenu diffère (collision improbable, l'id
    dans le nom de fichier est aléatoire) ;
  - pour chaque *.gguf_meta.json, détermine family/quant depuis son champ
    "model_file" (même logique de parsing) et fusionne son array "runs"
    (dédupliqué par date) dans le fichier run_meta existant, le cas échéant ;
  - nettoie les dossiers vides restants dans l'upload.

Usage :
    python scripts/import_inspect_uploads.py
    python scripts/import_inspect_uploads.py --extract   # puis lance extract_inspect_summary.py
"""

import argparse
import filecmp
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_inspect_summary import parse_model_string, read_header  # noqa: E402


def find_eval_files(inbox_dir: Path) -> list[Path]:
    return sorted(inbox_dir.rglob("*.eval"))


def remove_empty_dirs(root: Path):
    """Supprime récursivement les dossiers vides sous root (root exclu)."""
    for d in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def import_uploads(inbox_dir: Path, target_dir: Path) -> tuple[int, int, int]:
    imported = skipped_dupe = conflicts = 0

    for eval_path in find_eval_files(inbox_dir):
        try:
            header = read_header(eval_path)
            family, _quant = parse_model_string(header["eval"]["model"])
        except Exception as e:
            print(f"  [ERR] {eval_path}: {e}", file=sys.stderr)
            continue

        dest_dir = target_dir / family
        dest_path = dest_dir / eval_path.name

        if dest_path.exists():
            if filecmp.cmp(eval_path, dest_path, shallow=False):
                eval_path.unlink()
                skipped_dupe += 1
                print(f"  [DUP]  {eval_path.name} déjà présent dans {family}/, supprimé de l'upload")
            else:
                conflicts += 1
                print(
                    f"  [CONFLIT] {eval_path.name} : contenu différent de {dest_path} — laissé en place",
                    file=sys.stderr,
                )
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            eval_path.rename(dest_path)
            imported += 1
            print(f"  [OK]   {eval_path.name} → {family}/")

    remove_empty_dirs(inbox_dir)
    return imported, skipped_dupe, conflicts


def find_meta_files(inbox_dir: Path) -> list[Path]:
    return sorted(inbox_dir.rglob("*.gguf_meta.json"))


def import_meta_files(inbox_dir: Path, results_dir: Path) -> tuple[int, int, set[str]]:
    """Range les *.gguf_meta.json (conditions de banc par quant) vers
    results/{family}/run_meta/{quant}.json. Si le fichier existe déjà
    (ré-upload ou rerun du même quant), fusionne les "runs" en dédupliquant
    par date plutôt que d'écraser.
    """
    imported = merged = 0
    families_touched: set[str] = set()

    for meta_path in find_meta_files(inbox_dir):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            family, quant = parse_model_string(data["model_file"])
        except Exception as e:
            print(f"  [ERR] {meta_path}: {e}", file=sys.stderr)
            continue

        dest_dir = results_dir / family / "run_meta"
        dest_path = dest_dir / f"{quant}.json"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            existing = json.loads(dest_path.read_text(encoding="utf-8"))
            seen_dates = {r.get("date") for r in existing.get("runs", [])}
            new_runs = [r for r in data.get("runs", []) if r.get("date") not in seen_dates]
            existing.setdefault("runs", []).extend(new_runs)
            existing["model_file"] = data.get("model_file", existing.get("model_file"))
            existing["hf_repo"] = data.get("hf_repo", existing.get("hf_repo"))
            dest_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            merged += 1
            print(f"  [MERGE] {meta_path.name} → {family}/run_meta/{quant}.json (+{len(new_runs)} run(s))")
        else:
            dest_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            imported += 1
            print(f"  [OK]   {meta_path.name} → {family}/run_meta/{quant}.json")

        families_touched.add(family)
        meta_path.unlink()

    remove_empty_dirs(inbox_dir)
    return imported, merged, families_touched


def import_bench_config(inbox_dir: Path, results_dir: Path, families: set[str]) -> bool:
    """Snapshot bench_config.json (config partagée par le batch, redondante
    avec les blocs llama_server/inspect déjà inlinés dans chaque
    gguf_meta.json) vers run_meta/bench_config.json pour chaque famille
    touchée par ce batch. Écrase le snapshot précédent — un seul par
    famille, pas d'historique.
    """
    config_path = inbox_dir / "bench_config.json"
    if not config_path.is_file() or not families:
        return False

    content = config_path.read_text(encoding="utf-8")
    for family in sorted(families):
        dest_dir = results_dir / family / "run_meta"
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "bench_config.json").write_text(content, encoding="utf-8")
        print(f"  [OK]   bench_config.json → {family}/run_meta/bench_config.json")

    config_path.unlink()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Importe les .eval déposés dans results/inbox/ vers results/inspect_evals/{model_family}/."
    )
    parser.add_argument(
        "--inbox", "-i",
        type=Path,
        default=Path("./results/inbox"),
        help="Dossier d'upload à traiter (défaut: ./results/inbox)",
    )
    parser.add_argument(
        "--target", "-t",
        type=Path,
        default=Path("./results/inspect_evals"),
        help="Dossier de destination des .eval (défaut: ./results/inspect_evals)",
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=Path,
        default=Path("./results"),
        help="Dossier de destination des run_meta (défaut: ./results)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Lance scripts/extract_inspect_summary.py une fois l'import terminé",
    )
    args = parser.parse_args()

    if not args.inbox.is_dir():
        print(f"Erreur : {args.inbox} n'est pas un dossier.", file=sys.stderr)
        sys.exit(1)

    print(f"Scan de {args.inbox} ...")
    imported, skipped_dupe, conflicts = import_uploads(args.inbox, args.target)

    print(
        f"\nTerminé : {imported} importé(s), {skipped_dupe} doublon(s) ignoré(s), "
        f"{conflicts} conflit(s)."
    )
    if conflicts:
        print("Des conflits nécessitent une vérification manuelle (voir ci-dessus).", file=sys.stderr)

    print(f"\nScan de {args.inbox} (*.gguf_meta.json) ...")
    meta_imported, meta_merged, families = import_meta_files(args.inbox, args.results_dir)
    print(f"Terminé : {meta_imported} nouveau(x), {meta_merged} fusionné(s).")

    if import_bench_config(args.inbox, args.results_dir, families):
        print("bench_config.json archivé.")

    if args.extract:
        print("\nLancement de extract_inspect_summary.py ...")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "extract_inspect_summary.py")],
            check=True,
        )


if __name__ == "__main__":
    main()
