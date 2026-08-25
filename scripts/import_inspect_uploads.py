#!/usr/bin/env python3
"""
import_inspect_uploads.py — Range les .eval déposés dans results/inbox/
vers results/inspect_evals/{model_family}/ (structure attendue par
extract_inspect_summary.py).

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
  - nettoie les dossiers vides restants dans l'upload.

Usage :
    python scripts/import_inspect_uploads.py
    python scripts/import_inspect_uploads.py --extract   # puis lance extract_inspect_summary.py
"""

import argparse
import filecmp
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
        help="Dossier de destination (défaut: ./results/inspect_evals)",
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

    if args.extract:
        print("\nLancement de extract_inspect_summary.py ...")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "extract_inspect_summary.py")],
            check=True,
        )


if __name__ == "__main__":
    main()
