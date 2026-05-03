#!/usr/bin/env python3
"""
build.py — Static site generator for llm-quant-bench

Reads:
  - models.yaml          (editorial metadata, takeaways, config)
  - results/*/summary.json  (benchmark data from lm-evaluation-harness)
  - template.html        (HTML template with placeholders)

Writes:
  - index.html           (ready to deploy)

Usage:
  python build.py
  python build.py --results-dir ../results --output docs/index.html
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

import yaml


# ═══════════════════════════════════════════════════════════════
#  Mapping from summary.json field names to models.yaml benchmark keys
# ═══════════════════════════════════════════════════════════════
# The summary.json uses names like "arc_chat", "humaneval_plus_pass@1",
# "ifeval_prompt_strict", "gsm8k_acc" etc.
# We map these to the canonical keys used in models.yaml benchmarks section.

FIELD_MAP = {
    "arc_chat":               "arc_chat",
    "arc_chat_stderr":        "arc_chat_stderr",
    "ifeval_prompt_strict":   "ifeval_prompt_strict",
    "ifeval_prompt_strict_stderr": "ifeval_prompt_strict_stderr",
    "ifeval_prompt_loose":    "ifeval_prompt_loose",
    "ifeval_inst_strict":     "ifeval_inst_strict",
    "ifeval_inst_loose":      "ifeval_inst_loose",
    "gsm8k":                  "gsm8k",
    "gsm8k_acc":              "gsm8k",
    "gsm8k_strict":           "gsm8k",
    "gsm8k_flexible":         "gsm8k",        # preferred — overwrites strict if both present
    "gsm8k_stderr":           "gsm8k_stderr",
    "gsm8k_acc_stderr":       "gsm8k_stderr",
    "gsm8k_strict_stderr":    "gsm8k_stderr",
    "gsm8k_flexible_stderr":  "gsm8k_stderr", # preferred stderr
    "humaneval_pass@1":       "humaneval_pass_at_1",
    "humaneval_pass@1_stderr":"humaneval_pass_at_1_stderr",
    "humaneval_plus_pass@1":  "humaneval_plus_pass_at_1",
    "humaneval_plus_pass@1_stderr": "humaneval_plus_pass_at_1_stderr",
}


def fetch_hf_files(repo: str) -> dict:
    """Fetch {FILENAME_UPPER: size_gb} for all .gguf files in a HuggingFace repo."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-quant-bench/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            files = json.loads(resp.read().decode())
        return {
            f["path"].upper(): f["size"] / (1024 ** 3)
            for f in files
            if f.get("path", "").endswith(".gguf") and "size" in f
        }
    except Exception as e:
        print(f"  ⚠ HF fetch failed for {repo}: {e}")
        return {}


def match_quant_size(quant: str, hf_files: dict) -> float | None:
    """Match a quant name to its cumulative file size (GB) from a HF file listing."""
    q_clean = re.sub(r"^UD-", "", quant).upper()
    q_pat = re.sub(r"[_-]", "[_-]?", q_clean)
    patterns = [
        re.compile(r"[._\-]" + q_pat + r"[.\-]"),
        re.compile(r"[._\-]UD[._\-]" + q_pat + r"[.\-]"),
    ]
    total = sum(
        size for fname, size in hf_files.items()
        if any(p.search(fname) for p in patterns)
    )
    return round(total, 3) if total > 0 else None


def fetch_model_sizes(model_meta: dict, quants: list) -> dict:
    """Return {quant: size_gb} for a model by querying its HuggingFace repo."""
    hf_repo = model_meta.get("hf_repo")
    if not hf_repo:
        return {}
    print(f"  → Fetching sizes: {hf_repo}")
    hf_files = fetch_hf_files(hf_repo)
    if not hf_files:
        return {}
    sizes = {q: s for q in quants if (s := match_quant_size(q, hf_files)) is not None}
    print(f"    {len(sizes)}/{len(quants)} quants matched")
    return sizes


def build_sizes_js(config: dict, results_dir: Path) -> str:
    """Fetch file sizes from HuggingFace for all models and return a JSON object."""
    all_sizes = {}
    for model_id, model_meta in config["models"].items():
        summary_path = results_dir / model_meta["results_dir"] / "summary.json"
        if not summary_path.exists():
            all_sizes[model_id] = {}
            continue
        summary = load_summary(summary_path)
        quants = [e.get("quant", "").strip() for e in summary if e.get("quant")]
        all_sizes[model_id] = fetch_model_sizes(model_meta, quants)
    return json.dumps(all_sizes, indent=2)


def normalize_quant_name(raw_quant: str) -> str:
    """
    Normalize quant names from summary.json to display format.
    E.g. "UD-IQ2_XXS" -> "IQ2_XXS", "Q3_K_M" -> "Q3_K_M"
    We keep the UD- prefix stripped for sorting but store the original
    for display purposes.
    """
    # Strip any model name prefix if present
    # The quant field in summary.json is just "IQ4_NL" or "UD-IQ2_M"
    return raw_quant.strip()


def quant_sort_key(quant_name: str, quant_order: list) -> int:
    """Return sort index for a quant name based on the defined order."""
    # Strip UD- prefix for matching
    clean = re.sub(r'^UD-', '', quant_name)
    try:
        return quant_order.index(clean)
    except ValueError:
        # Unknown quant goes to the end
        return len(quant_order) + hash(quant_name)


def load_summary(summary_path: Path) -> list:
    """Load a summary.json file."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_model_data(summary: list, bench_keys: list, quant_order: list) -> dict:
    """
    Extract benchmark data from summary.json entries.

    Returns:
        {
            "quants": ["IQ2_XXS", "IQ2_M", ...],  # sorted
            "arc_chat": [96.1, 96.5, ...],          # percentages or null
            "arc_chat_stderr": [0.5, 0.5, ...],
            ...
        }
    """
    # Collect all quant entries
    entries = []
    for entry in summary:
        quant = normalize_quant_name(entry.get("quant", ""))
        if not quant:
            continue

        row = {"quant": quant}
        for json_field, canonical in FIELD_MAP.items():
            if json_field in entry:
                val = entry[json_field]
                # Convert from decimal to percentage
                if val is not None and canonical in bench_keys:
                    val = round(val * 100, 1)
                elif val is not None and canonical.endswith("_stderr"):
                    val = round(val * 100, 1)
                row[canonical] = val

        entries.append(row)

    # Sort by quant order
    entries.sort(key=lambda e: quant_sort_key(e["quant"], quant_order))

    # Build output arrays
    result = {"quants": [e["quant"] for e in entries]}
    for key in bench_keys:
        result[key] = [e.get(key, None) for e in entries]
        stderr_key = key + "_stderr"
        result[stderr_key] = [e.get(stderr_key, None) for e in entries]

    return result


def build_models_js(config: dict, results_dir: Path) -> str:
    """
    Build the JavaScript MODELS object from config + summary.json files.
    """
    bench_keys = list(config["benchmarks"].keys())
    quant_order = config.get("quant_order", [])
    models = config["models"]

    js_models = {}

    for model_id, model_meta in models.items():
        results_subdir = model_meta["results_dir"]
        summary_path = results_dir / results_subdir / "summary.json"

        if not summary_path.exists():
            print(f"  ⚠ No summary.json found at {summary_path}, skipping {model_id}")
            continue

        print(f"  ✓ Loading {summary_path}")
        summary = load_summary(summary_path)
        data = extract_model_data(summary, bench_keys, quant_order)

        js_model = {
            "name": model_meta["name"],
            "provider": model_meta["provider"],
            "arch": model_meta["arch"],
            "hf_repos": [model_meta["hf_repo"]] if model_meta.get("hf_repo") else [],
            "quants": data["quants"],
            "takeaway": model_meta.get("takeaway", "").strip(),
            "compare_color": model_meta.get("compare_color", "#888"),
            "compare_style": model_meta.get("compare_style", "solid"),
        }

        # Add benchmark arrays
        for key in bench_keys:
            js_model[key] = data[key]

        js_models[model_id] = js_model

    return json.dumps(js_models, indent=2, ensure_ascii=False)


def build_bench_config_js(config: dict) -> str:
    """Build the JavaScript BENCH_CONFIG object."""
    return json.dumps(config["benchmarks"], indent=2, ensure_ascii=False)


def build_quant_order_js(config: dict) -> str:
    """Build the quant order array."""
    return json.dumps(config.get("quant_order", []), indent=2)


def build_compare_takeaways_js(config: dict) -> str:
    """Build compare takeaways object."""
    takeaways = config.get("compare_takeaways", {})
    # Clean up whitespace from YAML multiline strings
    cleaned = {k: v.strip() for k, v in takeaways.items()}
    return json.dumps(cleaned, indent=2, ensure_ascii=False)


def build_model_tabs_html(config: dict, results_dir: Path) -> str:
    """Build the HTML for model selector tabs."""
    models = config["models"]
    tabs = []
    first = True
    for model_id, model_meta in models.items():
        # Check if results exist
        summary_path = results_dir / model_meta["results_dir"] / "summary.json"
        if not summary_path.exists():
            continue

        active = " active" if first else ""
        arch_provider = f'{model_meta["arch"]} · {model_meta["provider"]}'
        tab = (
            f'    <button class="model-tab{active}" data-model="{model_id}">'
            f'{model_meta["name"]} <span class="size">{arch_provider}</span></button>'
        )
        tabs.append(tab)
        first = False

    return "\n".join(tabs)


def build_compare_tabs_html(config: dict) -> str:
    """Build the HTML for benchmark comparison tabs."""
    benchmarks = config["benchmarks"]
    tabs = []
    first = True
    for bench_key, bench_meta in benchmarks.items():
        active = " active" if first else ""
        tab = (
            f'    <button class="model-tab{active}" '
            f'data-bench="{bench_key}">{bench_meta["label"]}</button>'
        )
        tabs.append(tab)
        first = False
    return "\n".join(tabs)


def build(args):
    """Main build function."""
    site_dir = Path(args.site_dir)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    config_path = site_dir / "models.yaml"
    template_path = site_dir / "template.html"

    # Validate inputs
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)
    if not template_path.exists():
        print(f"✗ Template not found: {template_path}")
        sys.exit(1)
    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        sys.exit(1)

    # Load config
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load template
    print(f"Loading template from {template_path}")
    with open(template_path, 'r') as f:
        template = f.read()

    # Build data
    print(f"Reading results from {results_dir}")
    models_js = build_models_js(config, results_dir)
    bench_config_js = build_bench_config_js(config)
    compare_takeaways_js = build_compare_takeaways_js(config)
    model_tabs_html = build_model_tabs_html(config, results_dir)
    compare_tabs_html = build_compare_tabs_html(config)

    # Count models loaded
    models_loaded = models_js.count('"name"')
    print(f"  → {models_loaded} models loaded")

    # Fetch file sizes from HuggingFace
    if args.skip_sizes:
        print("Skipping HuggingFace size fetch (--skip-sizes)")
        sizes_js = json.dumps({m: {} for m in config["models"]})
    else:
        print("Fetching file sizes from HuggingFace...")
        sizes_js = build_sizes_js(config, results_dir)

    # Replace placeholders
    html = template
    html = html.replace("{{MODELS_DATA}}", models_js)
    html = html.replace("{{BENCH_CONFIG}}", bench_config_js)
    html = html.replace("{{COMPARE_TAKEAWAYS}}", compare_takeaways_js)
    html = html.replace("{{SIZES_DATA}}", sizes_js)
    html = html.replace("{{MODEL_TABS}}", model_tabs_html)
    html = html.replace("{{COMPARE_BENCH_TABS}}", compare_tabs_html)

    # Check for unreplaced placeholders
    remaining = re.findall(r'\{\{[A-Z_]+\}\}', html)
    if remaining:
        print(f"  ⚠ Unreplaced placeholders: {remaining}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n✓ Generated {output_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build llm-quant-bench site")
    parser.add_argument(
        "--site-dir", default="site",
        help="Directory containing models.yaml and template.html (default: site)"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing model result folders (default: results)"
    )
    parser.add_argument(
        "--output", default="docs/index.html",
        help="Output HTML file path (default: docs/index.html)"
    )
    parser.add_argument(
        "--skip-sizes", action="store_true",
        help="Skip HuggingFace file size fetch (faster local builds)"
    )
    args = parser.parse_args()
    build(args)
