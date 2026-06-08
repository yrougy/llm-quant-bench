#!/usr/bin/env python3
"""
extract_summary.py — Extrait et agrège les résultats lm-evaluation-harness.

Parcourt un dossier de benchmarks structuré comme :
    {base_dir}/{model}-{quant}.gguf-{bench}/{Publisher}__{Model}/results_*.json

Produit :
    - results/{model_family}/summary.json   (un JSON exploitable par modèle)
    - results/all_results.csv               (un CSV global, triable/filtrable)
    - results/all_results.json              (tout en un seul JSON)

Usage :
    python extract_summary.py /data/benches
    python extract_summary.py /data/benches --output ./results
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Mapping des métriques connues par benchmark
# On normalise les noms pour le summary
# ---------------------------------------------------------------------------

METRIC_EXTRACTION = {
    "gsm8k": [
        ("exact_match,flexible-extract", "gsm8k_flexible"),
        ("exact_match,strict-match",     "gsm8k_strict"),
    ],
    "arc_challenge_chat": [
        ("exact_match,remove_whitespace", "arc_chat"),
    ],
    "arc_challenge": [
        ("acc,none",      "arc_acc"),
        ("acc_norm,none", "arc_acc_norm"),
    ],
    "ifeval": [
        ("prompt_level_strict_acc,none",  "ifeval_prompt_strict"),
        ("prompt_level_loose_acc,none",   "ifeval_prompt_loose"),
        ("inst_level_strict_acc,none",    "ifeval_inst_strict"),
        ("inst_level_loose_acc,none",     "ifeval_inst_loose"),
    ],
    "humaneval": [
        ("pass@1,create_test", "humaneval_pass@1"),
    ],
    # Ajoute d'autres benchmarks ici au besoin
    "mmlu": [
        ("acc,none", "mmlu_acc"),
    ],
    "drop": [
        ("f1,none",           "drop_f1"),
        ("exact_match,none",  "drop_em"),
    ],
}


def parse_dirname(dirname: str) -> dict | None:
    """
    Parse un nom de dossier de type :
        Qwen3.5-4B-Q4_0.gguf-arc-chat
        gemma-4-26B-A4B-it-UD-Q5_K_M.gguf-gsm8k
        Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf-ifeval

    Retourne {"full_model": ..., "quant": ..., "bench_alias": ...} ou None.
    """
    # Pattern : tout avant .gguf = modèle+quant, tout après = bench
    m = re.match(r"^(.+)\.gguf-(.+)$", dirname)
    if not m:
        return None

    model_quant = m.group(1)  # ex: "Qwen3.5-4B-Q4_0"
    bench_alias = m.group(2)  # ex: "arc-chat"

    # Séparer la quantisation du nom du modèle.
    # La quant est le dernier segment qui ressemble à un type GGUF :
    #   Q4_0, Q4_K_M, IQ2_XXS, Q5_K_XL, MXFP4_MOE, UD-Q3_K_S, etc.
    # Stratégie : on cherche le dernier bloc qui commence par (IQ|Q|MXFP|BF|FP)
    # en tenant compte du préfixe "UD-" optionnel.
    quant_pattern = re.compile(
        r"^(.*?)[-_]((?:UD[-_])?(?:IQ|Q|MXFP|BF|FP)\S*)$"
    )
    qm = quant_pattern.match(model_quant)
    if qm:
        model_name = qm.group(1)
        quant = qm.group(2)
    else:
        # Fallback : tout est le modèle, pas de quant séparée
        model_name = model_quant
        quant = "unknown"

    return {
        "model_name": model_name,
        "quant": quant,
        "bench_alias": bench_alias,
        "full_dirname": dirname,
    }


def extract_scores(results_json: dict, bench_alias: str) -> dict:
    """
    Extrait les scores depuis le JSON de résultats.
    Retourne un dict {metric_name: {"value": float, "stderr": float|None}}.
    """
    scores = {}
    results = results_json.get("results", {})

    for task_name, task_results in results.items():
        # Trouver les métriques configurées pour ce task
        task_key = task_name.lower()
        metric_map = METRIC_EXTRACTION.get(task_key)

        if metric_map is None:
            # Benchmark inconnu : on extrait toutes les métriques numériques
            for key, value in task_results.items():
                if isinstance(value, (int, float)) and not key.startswith("name"):
                    stderr_key = key.replace(",", "_stderr,")
                    if "_stderr" not in key and key != "sample_len":
                        clean_name = f"{task_key}_{key.split(',')[0]}"
                        stderr_val = task_results.get(stderr_key)
                        scores[clean_name] = {
                            "value": round(value, 6),
                            "stderr": round(stderr_val, 6) if isinstance(stderr_val, (int, float)) else None,
                        }
        else:
            for metric_key, output_name in metric_map:
                val = task_results.get(metric_key)
                stderr_key = metric_key.replace(",", "_stderr,")
                stderr_val = task_results.get(stderr_key)
                if val is not None:
                    scores[output_name] = {
                        "value": round(val, 6),
                        "stderr": round(stderr_val, 6) if isinstance(stderr_val, (int, float)) else None,
                    }

    return scores


def extract_metadata(results_json: dict) -> dict:
    """Extrait les métadonnées utiles pour la reproductibilité."""
    meta = {}
    meta["lm_eval_version"] = results_json.get("lm_eval_version", "unknown")
    meta["date"] = results_json.get("date")
    meta["eval_time_seconds"] = results_json.get("total_evaluation_time_seconds")

    config = results_json.get("config", {})
    meta["batch_size"] = config.get("batch_size")

    # N-shot
    nshot = results_json.get("n-shot", {})
    for task, n in nshot.items():
        meta[f"nshot_{task}"] = n

    return meta


def find_results_files(base_dir: Path) -> list[tuple[Path, dict]]:
    """
    Trouve tous les fichiers results_*.json et retourne
    [(path, parsed_info), ...].
    """
    found = []
    for bench_dir in sorted(base_dir.iterdir()):
        if not bench_dir.is_dir():
            continue

        parsed = parse_dirname(bench_dir.name)
        if parsed is None:
            print(f"  [SKIP] Dossier non reconnu: {bench_dir.name}", file=sys.stderr)
            continue

        # Chercher le sous-dossier Publisher__Model puis results_*.json
        for sub in bench_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sorted(sub.glob("results_*.json")):
                found.append((f, parsed))

    return found


def build_summaries(entries: list[tuple[Path, dict]]) -> dict:
    """
    Construit un dict structuré :
    {
        model_name: {
            quant: {
                "model_name": ...,
                "quant": ...,
                "scores": {metric: {value, stderr}, ...},
                "metadata": {...}
            }
        }
    }
    """
    summaries = defaultdict(lambda: defaultdict(lambda: {
        "scores": {},
        "metadata": {},
    }))

    for filepath, parsed in entries:
        model = parsed["model_name"]
        quant = parsed["quant"]

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [ERR] {filepath}: {e}", file=sys.stderr)
            continue

        scores = extract_scores(data, parsed["bench_alias"])
        meta = extract_metadata(data)

        entry = summaries[model][quant]
        entry["model_name"] = model
        entry["quant"] = quant
        entry["scores"].update(scores)
        # Garder la metadata la plus récente
        entry["metadata"].update(meta)

    return summaries


def write_outputs(summaries: dict, output_dir: Path):
    """Écrit les summary.json par modèle (les fichiers globaux sont gérés par aggregate.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collecter toutes les métriques pour normaliser les colonnes par run
    all_metrics = set()
    for model, quants in summaries.items():
        for quant, entry in quants.items():
            all_metrics.update(entry["scores"].keys())
    all_metrics = sorted(all_metrics)

    for model, quants in sorted(summaries.items()):
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        model_summary = []
        for quant, entry in sorted(quants.items()):
            record = {
                "model": model,
                "quant": quant,
            }
            for metric in all_metrics:
                s = entry["scores"].get(metric)
                if s:
                    record[metric] = s["value"]
                    record[f"{metric}_stderr"] = s["stderr"]
                else:
                    record[metric] = None
                    record[f"{metric}_stderr"] = None

            model_summary.append(record)

        summary_path = model_dir / "summary.json"
        summary_path.write_text(
            json.dumps(model_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  [OK] {summary_path}  ({len(model_summary)} quants)")


def main():
    parser = argparse.ArgumentParser(
        description="Extrait les résultats lm-evaluation-harness en summary JSON/CSV."
    )
    parser.add_argument(
        "bench_dir",
        type=Path,
        help="Dossier racine des benchmarks (ex: /data/benches)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./results"),
        help="Dossier de sortie (défaut: ./results)",
    )
    args = parser.parse_args()

    if not args.bench_dir.is_dir():
        print(f"Erreur : {args.bench_dir} n'est pas un dossier.", file=sys.stderr)
        sys.exit(1)

    print(f"Scan de {args.bench_dir} ...")
    entries = find_results_files(args.bench_dir)
    print(f"  Trouvé {len(entries)} fichiers de résultats.")

    if not entries:
        print("Rien à traiter.", file=sys.stderr)
        sys.exit(0)

    print("Construction des summaries ...")
    summaries = build_summaries(entries)

    print(f"Écriture dans {args.output} ...")
    write_outputs(summaries, args.output)

    print("Terminé.")


if __name__ == "__main__":
    main()
