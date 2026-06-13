#!/usr/bin/env python3
"""
aggregate.py — Agrège tous les summary.json en all_results.json + all_results.csv.

Lit tous les fichiers results/{model}/summary.json et les combine en deux fichiers
globaux. À lancer après extract_summary.py et/ou extract_inspect_summary.py.

Usage :
    python aggregate.py
    python aggregate.py --results-dir ./results
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SKIP_DIRS = {"inspect_evals"}


def load_all_summaries(results_dir: Path) -> list[dict]:
    records = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in SKIP_DIRS:
            continue
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [ERR] {summary_path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(f"  [SKIP] {summary_path}: format inattendu (pas une liste)", file=sys.stderr)
            continue
        records.extend(data)
        print(f"  {model_dir.name}: {len(data)} entrées")
    return records


def write_globals(records: list[dict], results_dir: Path):
    all_cols: set = set()
    for r in records:
        all_cols.update(r.keys())

    base_cols = ["model", "quant"]
    extra_cols = sorted(all_cols - set(base_cols))
    fieldnames = base_cols + extra_cols

    json_path = results_dir / "all_results.json"
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  [OK] {json_path}  ({len(records)} entrées)")

    csv_path = results_dir / "all_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"  [OK] {csv_path}  ({len(records)} lignes)")


def main():
    parser = argparse.ArgumentParser(
        description="Agrège tous les summary.json en all_results.json + all_results.csv."
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=Path,
        default=Path("./results"),
        help="Dossier contenant les sous-dossiers de modèles (défaut: ./results)",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Erreur : {args.results_dir} n'est pas un dossier.", file=sys.stderr)
        sys.exit(1)

    print(f"Lecture des summaries dans {args.results_dir} ...")
    records = load_all_summaries(args.results_dir)

    if not records:
        print("Aucune entrée trouvée.", file=sys.stderr)
        sys.exit(0)

    print(f"Écriture des fichiers globaux ({len(records)} entrées total) ...")
    write_globals(records, args.results_dir)
    print("Terminé.")


if __name__ == "__main__":
    main()
