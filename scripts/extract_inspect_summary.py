#!/usr/bin/env python3
"""
extract_inspect_summary.py — Extrait les résultats inspect_ai depuis les fichiers .eval.

Structure attendue :
    results/inspect_evals/{model_family}/{timestamp}_{task}_{id}.eval

Chaque fichier .eval est un ZIP (compression Zstd) contenant header.json avec :
  - eval.model         : nom du modèle + quant (ex: "openai-api/llamacpp/Model-Q4_K_M.gguf")
  - eval.task_display_name : nom du benchmark (ex: "bbeh_mini")
  - results.scores     : liste de scorers avec leurs métriques
  - results.total_samples  : nombre de samples utilisés
  - stats.started_at / stats.completed_at : horodatages pour la durée

Produit :
    - results/{model_family}/summary.json
    - results/all_results.csv  (fusionné avec l'existant si présent)
    - results/all_results.json (fusionné avec l'existant si présent)

Usage :
    python extract_inspect_summary.py
    python extract_inspect_summary.py --inspect-dir results/inspect_evals --output results
"""

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import zipfile_zstd  # noqa: F401 — registers Zstd support in zipfile
except ImportError:
    print(
        "ERREUR : le module 'zipfile_zstd' est requis.\n"
        "Installez-le avec : pip install zipfile_zstd",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Pattern de quantisation GGUF (même logique que extract_summary.py)
# Insensible à la casse : certains repos nomment leurs fichiers "-bf16.gguf".
# ---------------------------------------------------------------------------
QUANT_RE = re.compile(r"^(.*?)[-_]((?:UD[-_])?(?:IQ|Q|MXFP|BF|FP)\S*)$", re.IGNORECASE)


def parse_model_string(model_str: str) -> tuple[str, str]:
    """
    Extrait (model_name, quant) depuis une chaîne de type :
        "openai-api/llamacpp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
        "llamacpp/Qwen3.5-4B-Q4_0.gguf"
        "llamacpp/ornith-1.0-9b-bf16.gguf"

    Retourne ("Qwen3.6-35B-A3B", "UD-IQ1_M") par exemple.
    Le nom du quant est normalisé en majuscules ("bf16" → "BF16").
    """
    # Garder uniquement la dernière partie après le dernier "/"
    basename = model_str.split("/")[-1]
    # Retirer l'extension .gguf
    if basename.lower().endswith(".gguf"):
        basename = basename[:-5]

    m = QUANT_RE.match(basename)
    if m:
        return m.group(1), m.group(2).upper().replace("UD_", "UD-")
    return basename, "unknown"


def read_header(eval_path: Path) -> dict:
    """Lit header.json depuis un fichier .eval (ZIP+Zstd)."""
    with zipfile.ZipFile(eval_path) as zf:
        with zf.open("header.json") as f:
            return json.loads(f.read())


def duration_seconds(started: str, completed: str) -> float:
    """Calcule la durée en secondes entre deux horodatages ISO 8601."""
    dt_start = datetime.fromisoformat(started)
    dt_end = datetime.fromisoformat(completed)
    return round((dt_end - dt_start).total_seconds(), 1)


def extract_scores(header: dict) -> dict:
    """
    Extrait les métriques depuis header['results']['scores'].
    Retourne {"{task}_{scorer}_{metric}": value, ...}
    """
    task = header["eval"]["task_display_name"]
    scores = {}
    for scorer_entry in header.get("results", {}).get("scores", []):
        scorer_name = scorer_entry.get("scorer", scorer_entry.get("name", "scorer"))
        for metric_name, metric_data in scorer_entry.get("metrics", {}).items():
            val = metric_data.get("value")
            if val is not None:
                # Clé normalisée : task_metric (on omet le scorer s'il est redondant)
                key = f"{task}_{metric_name}"
                scores[key] = round(val, 6)
    return scores


def scan_eval_files(inspect_dir: Path) -> list[tuple[Path, str]]:
    """
    Retourne [(eval_path, model_family), ...] pour tous les .eval trouvés.
    model_family = nom du sous-dossier dans inspect_dir.
    """
    found = []
    for family_dir in sorted(inspect_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        for eval_file in sorted(family_dir.glob("*.eval")):
            found.append((eval_file, family_dir.name))
    return found


def build_summaries(eval_files: list[tuple[Path, str]]) -> dict:
    """
    Construit un dict structuré :
    {
        model_family: {
            quant: {
                "model_name": ...,
                "quant": ...,
                "scores": {metric: value, ...},
                "eval_time_seconds": ...,
                "total_samples": ...,
            }
        }
    }
    Si plusieurs .eval pour le même (family, quant, task), on garde le plus récent.
    """
    summaries: dict = defaultdict(lambda: defaultdict(lambda: {
        "scores": {},
        "eval_time_seconds": 0.0,
        "total_samples": 0,
        "_file_ts": {},  # task → timestamp du fichier retenu
    }))

    for eval_path, family in eval_files:
        try:
            header = read_header(eval_path)
        except Exception as e:
            print(f"  [ERR] {eval_path.name}: {e}", file=sys.stderr)
            continue

        if header.get("status") != "success":
            print(f"  [SKIP] {eval_path.name}: status={header.get('status')}", file=sys.stderr)
            continue

        model_str = header["eval"]["model"]
        model_name, quant = parse_model_string(model_str)
        task = header["eval"]["task_display_name"]
        file_ts = eval_path.name[:19]  # timestamp dans le nom de fichier

        entry = summaries[family][quant]
        entry["model_name"] = model_name
        entry["quant"] = quant

        # Garder le run le plus récent pour cette task
        if file_ts > entry["_file_ts"].get(task, ""):
            entry["_file_ts"][task] = file_ts

            scores = extract_scores(header)
            entry["scores"].update(scores)

            # Nombre d'échantillons propre à cette task — nécessaire pour
            # convertir un std par échantillon en erreur-type (cf. build.py).
            n_task = header.get("results", {}).get("total_samples", 0)
            entry["scores"][f"{task}_total_samples"] = n_task

            stats = header.get("stats", {})
            started = stats.get("started_at", "")
            completed = stats.get("completed_at", "")
            if started and completed:
                entry["eval_time_seconds"] = duration_seconds(started, completed)

            entry["total_samples"] = header.get("results", {}).get("total_samples", 0)

    return summaries


def write_outputs(summaries: dict, output_dir: Path):
    """Écrit les summary.json par modèle (les fichiers globaux sont gérés par aggregate.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_inspect_metrics: set = set()
    for family, quants in summaries.items():
        for quant, entry in quants.items():
            all_inspect_metrics.update(entry["scores"].keys())
    all_inspect_metrics_sorted = sorted(all_inspect_metrics)

    for family, quants in sorted(summaries.items()):
        family_dir = output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)

        family_records = []
        for quant, entry in sorted(quants.items()):
            # "model" = nom du dossier de la famille (unique dans results/),
            # pas le nom parsé du .gguf — évite les collisions entre familles
            # dont les fichiers GGUF partagent le même préfixe (ex. v1 vs v2 MTP).
            record: dict = {
                "model": family,
                "quant": quant,
                "eval_time_seconds": entry["eval_time_seconds"],
                "total_samples": entry["total_samples"],
            }
            for metric in all_inspect_metrics_sorted:
                record[metric] = entry["scores"].get(metric)

            family_records.append(record)

        summary_path = family_dir / "summary.json"
        summary_path.write_text(
            json.dumps(family_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  [OK] {summary_path}  ({len(family_records)} quants)")


def main():
    parser = argparse.ArgumentParser(
        description="Extrait les résultats inspect_ai (.eval) en summary JSON/CSV."
    )
    parser.add_argument(
        "--inspect-dir", "-i",
        type=Path,
        default=Path("./results/inspect_evals"),
        help="Dossier racine des .eval (défaut: ./results/inspect_evals)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./results"),
        help="Dossier de sortie (défaut: ./results)",
    )
    args = parser.parse_args()

    if not args.inspect_dir.is_dir():
        print(f"Erreur : {args.inspect_dir} n'est pas un dossier.", file=sys.stderr)
        sys.exit(1)

    print(f"Scan de {args.inspect_dir} ...")
    eval_files = scan_eval_files(args.inspect_dir)
    print(f"  Trouvé {len(eval_files)} fichiers .eval.")

    if not eval_files:
        print("Rien à traiter.", file=sys.stderr)
        sys.exit(0)

    print("Extraction des résultats ...")
    summaries = build_summaries(eval_files)

    print(f"Écriture dans {args.output} ...")
    write_outputs(summaries, args.output)

    print("Terminé.")


if __name__ == "__main__":
    main()
