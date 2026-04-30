#!/usr/bin/env python3
"""
bench2md.py — Génère un tableau Markdown à partir des résultats lm-evaluation-harness.

Structure attendue :
  <NomModele>.gguf-<benchmark>/
    <sous-dossier>/
      results_*.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Métrique principale à extraire pour chaque type de benchmark connu.
# Format : nom_tâche -> clé dans results[nom_tâche]
KNOWN_METRICS = {
    "arc_challenge_chat": "exact_match,remove_whitespace",
    "humaneval":          "pass@1,create_test",
    "humaneval_plus":     "pass@1,create_test",
    "ifeval":             "prompt_level_strict_acc,none",
    "gsm8k":              "exact_match,strict-match",
}

# Noms d'affichage pour chaque tâche (clé = nom de tâche dans le JSON)
TASK_LABELS = {
    "arc_challenge_chat": "ARC-Chat",
    "humaneval":          "HumanEval",
    "humaneval_plus":     "HumanEval+",
    "ifeval":             "IFEval",
    "gsm8k":              "GSM8K",
}

# Ordre des colonnes dans le tableau (toujours affichées, même sans données)
COLUMN_ORDER = ["GSM8K", "IFEval", "ARC-Chat", "HumanEval", "HumanEval+"]


def extract_score(data: dict) -> tuple[str, float] | tuple[None, None]:
    """Retourne (nom_benchmark, score 0-100) depuis un dict de résultats JSON."""
    results = data.get("results", {})
    for task_name, task_results in results.items():
        metric_key = KNOWN_METRICS.get(task_name)
        if metric_key and metric_key in task_results:
            return task_name, task_results[metric_key] * 100

        # Benchmark inconnu : prendre le premier float qui ressemble à un score
        for k, v in task_results.items():
            if isinstance(v, float) and not k.endswith("_stderr") and ",none" not in k.replace(",none", ""):
                # Détecter si c'est déjà en % ou en fraction [0,1]
                score = v * 100 if v <= 1.0 else v
                return task_name, score

    return None, None


def main(root: Path):
    # {model_name: {bench_label: score}}
    table: dict[str, dict[str, float]] = defaultdict(dict)
    bench_labels_seen: list[str] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue

        name = entry.name
        # Doit contenir ".gguf-" pour être un répertoire de benchmark
        if ".gguf-" not in name:
            continue

        gguf_pos = name.index(".gguf-")
        model_name = name[:gguf_pos]          # ex. Qwen3.6-35B-A3B-UD-Q4_K_M
        bench_suffix = name[gguf_pos + 6:]    # ex. arc-chat

        json_files = sorted(entry.rglob("results_*.json"))
        if not json_files:
            print(f"  [WARN] Aucun fichier results_*.json dans {entry}", file=sys.stderr)
            continue

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            task_name, score = extract_score(data)
            if score is None:
                print(f"  [WARN] Score introuvable dans {json_file}", file=sys.stderr)
                continue

            label = TASK_LABELS.get(task_name, task_name)
            table[model_name][label] = score

            if label not in bench_labels_seen:
                bench_labels_seen.append(label)

    if not table:
        print("Aucun résultat trouvé.", file=sys.stderr)
        sys.exit(1)

    cols = list(COLUMN_ORDER)
    cols += [l for l in bench_labels_seen if l not in cols]

    # Pré-calculer toutes les valeurs pour connaître les largeurs
    sorted_models = sorted(table.keys())
    def cell(model, col):
        return f"{table[model][col]:.1f}%" if col in table[model] else "—"

    model_w = max(len("Modèle"), max(len(m) for m in sorted_models))
    col_w   = {col: max(len(col), max(len(cell(m, col)) for m in sorted_models)) for col in cols}

    def row(model_cell, cells, *, left_align_model=True):
        model_part = model_cell.ljust(model_w) if left_align_model else model_cell.ljust(model_w)
        return "| " + model_part + " | " + " | ".join(
            v.rjust(col_w[col]) for col, v in zip(cols, cells)
        ) + " |"

    header = row("Modèle", cols)
    sep    = "| " + "-" * model_w + " | " + " | ".join("-" * (col_w[c] - 1) + ":" for c in cols) + " |"
    lines  = [header, sep]

    for model in sorted_models:
        lines.append(row(model, [cell(model, col) for col in cols]))

    print("\n".join(lines))


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    main(root)
