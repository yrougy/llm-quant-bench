# GGUF Quantization Benchmark

Measuring the real-world accuracy cost of GGUF quantization on small open-weight LLMs, so you can pick the right quant for your use case.

## Why this exists

Quantization lets you run large models on consumer GPUs, but published benchmarks rarely test quantized GGUF variants head-to-head. This project fills that gap: same hardware, same evaluation harness, same prompts — only the quantization level changes.

The goal is **not** to rank models against each other, but to answer: *"For a given model, how much accuracy do I lose going from Q6_K down to IQ2_XXS?"*

## Benchmark suite

All evaluations run through [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_evals) with models served via [llama.cpp](https://github.com/ggml-org/llama.cpp). Scoring is fully deterministic — no LLM judge.

| Benchmark | What it measures | Scoring |
|-----------|-----------------|---------|
| **BigCodeBench** | Code generation across realistic programming tasks | Unit tests |
| **MUSR** | Multi-step soft reasoning on narratives | MCQ accuracy |
| **BFCL** | Function calling / tool use | AST matching |

BFCL runs on a fixed 1000-sample subset (seed 42), not the full benchmark — see [METHODOLOGY.md](METHODOLOGY.md).

GPQA was tried and dropped for now — too slow to run systematically across quants on this hardware (see [TODO.md](TODO.md)).

## Hardware

- **GPUs:** 2× RTX 3060 12 GiB (large models) · GTX 1070 8 GiB (small models)
- **Inference:** llama.cpp `llama-server`
- **KV cache:** q4_0 for both K and V
- **Context:** 16384–32768 tokens depending on model

## Installation

### 1. llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

### 2. inspect_evals

```bash
git clone https://github.com/UKGovernmentBEIS/inspect_evals
cd inspect_evals
uv sync
```

### 3. Dependencies for start_bench

```bash
apt install jq
pip install huggingface_hub
```

## Configuration

Copy `scripts/bench_config.json` and adapt it to your setup:

```json
{
  "model": {
    "hf_repo": "unsloth/Qwen3.6-27B-MTP-GGUF",
    "hf_tokenizer": "Qwen/Qwen3.6-27B",
    "quant_exclude_pattern": "F16|F32|Q8"
  },

  "llama_server": {
    "binary": "/path/to/llama-server",
    "port": 8050,
    "context_size": 32768,
    "cache_type_k": "q4_0",
    "cache_type_v": "q4_0",
    "n_parallel": 3,
    "sampling": {
      "temp": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "min_p": 0.0
    },
    "chat_template_kwargs": { "enable_thinking": false }
  },

  "inspect": {
    "working_dir": "/path/to/inspect_evals",
    "done_marker_dir": "/path/to/workdir",
    "base_url": "http://localhost:8050/v1/",
    "api_key": "sk1",
    "sample_shuffle": 42,
    "seed": 42,
    "tasks": {
      "bigcodebench": {
        "enabled": true,
        "limit": 0,
        "max_sandboxes": 4,
        "max_connections": 4,
        "time_limit": 300
      },
      "musr": {
        "enabled": true,
        "limit": 0,
        "max_sandboxes": 1,
        "max_connections": 1,
        "time_limit": 300
      },
      "bfcl": {
        "enabled": true,
        "limit": 1000,
        "max_sandboxes": 1,
        "max_connections": 1,
        "time_limit": 300
      }
    }
  }
}
```

`bench_config.json` also has entries for `gpqa_diamond`, `bbeh_mini`, `tau2_telecom` and `swe_bench` — disabled (`"enabled": false`) since they're not part of the current published suite. Set `enabled: true` to opt in.

Key fields:

| Field | Description |
|-------|-------------|
| `model.hf_repo` | HuggingFace repo to pull GGUFs from |
| `model.quant_exclude_pattern` | Regex to skip quants (e.g. `F16\|Q8`) |
| `llama_server.binary` | Absolute path to `llama-server` |
| `llama_server.n_parallel` | Parallel inference slots (set to match `max_connections`) |
| `inspect.working_dir` | Path to the cloned `inspect_evals` repo |
| `inspect.done_marker_dir` | Where `start_bench` writes `<model>_done` markers |
| `tasks.<name>.enabled` | Toggle a benchmark on/off without removing it |
| `tasks.<name>.limit` | Number of samples — `0` means full benchmark |

## Running a benchmark

```bash
# Benchmark all quants from the configured HuggingFace repo
scripts/start_bench

# Use a specific config file
scripts/start_bench -c /path/to/my_config.json

# Keep downloaded GGUF files after the run
scripts/start_bench -k

# Start llama-server only (useful for manual runs)
scripts/start_bench -s Qwen3.6-27B-UD-Q4_K_M.gguf
```

`start_bench` will, for each quant that hasn't been benchmarked yet:
1. Download the GGUF from HuggingFace
2. Start `llama-server`
3. Call `run2.sh`, which runs every enabled inspect_ai task
4. Stop the server, delete the GGUF (unless `-k`), and move to the next quant

Already-benchmarked quants are skipped via a `<model>_done` marker file.

## Extracting results

Drop raw `.eval` uploads (any folder structure — including re-uploads of runs already filed) into `results/inbox/`, then:

```bash
# results/inbox/** → results/inspect_evals/{model}/ (sorted by reading the model name from each .eval)
python scripts/import_inspect_uploads.py

# inspect_ai .eval files → results/{model}/summary.json
python scripts/extract_inspect_summary.py

# or both in one step:
python scripts/import_inspect_uploads.py --extract
```

## Repository structure

```
├── scripts/
│   ├── bench_config.json          # Main configuration file
│   ├── start_bench                # Entry point: iterates over quants
│   ├── run2.sh                    # Runs inspect_ai tasks for one model
│   ├── extract_inspect_summary.py # .eval → results/{model}/summary.json
│   ├── import_inspect_uploads.py  # results/inbox/** → results/inspect_evals/{model}/ (sorted, deduped)
│   ├── list_quants.py             # Lists available quants from HuggingFace
│   └── bench2md.py                # Unused — v1 (lm-evaluation-harness) leftover
├── results/
│   ├── {model}/summary.json
│   ├── inbox/                     # Dropzone for raw .eval uploads (gitignored)
│   └── inspect_evals/{model}/     # Raw .eval files, filed flat by import_inspect_uploads.py
├── site/
│   ├── models.yaml                # Editorial metadata: which models/benchmarks to show, and how
│   ├── build.py                   # Reads results/*/summary.json + models.yaml → docs/
│   ├── template.html              # Site page template
│   └── faq.html                   # Standalone FAQ page
├── docs/                          # Generated site (output of site/build.py)
├── METHODOLOGY.md
├── REPRODUCE.md
├── CAVEATS.md
└── LEGACY.md                      # Previous results (lm-evaluation-harness era)
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — share and adapt with attribution.

## Author

**Yves Rougy** — [rougy.net](https://rougy.net) · [GitHub](https://github.com/yrougy) · [LinkedIn](https://www.linkedin.com/in/yrougy/)
