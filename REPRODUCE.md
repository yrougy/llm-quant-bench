# How to reproduce

This guide walks you through reproducing the v2 benchmark results on your own hardware. (The legacy v1 lm-evaluation-harness workflow is documented in [LEGACY.md](LEGACY.md).)

## Prerequisites

- A GPU with enough VRAM for the model + KV cache (24 GB minimum for 26–35B models at low quants).
	- I use 2× RTX 3060 12 GiB for "big" models
	- I use a GTX 1070 8 GiB for small models
- Linux
	- Should work on other *nix
	- Ubuntu 22.04
	- NVidia drivers 550 (for stability)
	- CUDA 12.4 (for stability)
	- Software configuration is subject to change in the future
- Python 3.10+
- `jq` (used by the orchestration scripts)
- ~50 GB free disk space for model files

## Step 1: Install llama.cpp

Build llama.cpp with CUDA support:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

Verify that `llama-server` is built:

```bash
./build/bin/llama-server --help
```

## Step 2: Install inspect_evals

```bash
git clone https://github.com/UKGovernmentBEIS/inspect_evals
cd inspect_evals
uv sync
```

## Step 3: Install the remaining dependencies

```bash
apt install jq
pip install huggingface_hub          # GGUF downloads
pip install zipfile_zstd             # reading .eval files (extraction step)
pip install pyyaml                   # site generator (optional)
```

## Step 4: Configure the run

Copy `scripts/bench_config.json` and adapt it to your setup. The key fields:

| Field | Description |
|-------|-------------|
| `model.hf_repo` | HuggingFace repo to pull GGUFs from |
| `model.quant_exclude_pattern` | Regex of quants to skip (e.g. `F16\|F32\|Q8`) |
| `llama_server.binary` | Absolute path to your `llama-server` |
| `llama_server.context_size` | Context window (32768 for large models, less on small GPUs) |
| `llama_server.cache_type_k/v` | Keep `q4_0` to reproduce our numbers |
| `llama_server.n_parallel` | Parallel slots — set to match `max_connections` |
| `inspect.working_dir` | Path to your cloned `inspect_evals` repo |
| `inspect.done_marker_dir` | Where `<model>_done` markers are written |
| `inspect.tasks.<name>.enabled` | Toggle each benchmark on/off |
| `inspect.tasks.<name>.limit` | Sample count — `0` means full benchmark |

To match our runs exactly, keep `"seed": 42` and `"sample_shuffle": 42`, sampling at `temp 0.6 / top_p 0.95 / top_k 20 / min_p 0`, and `enable_thinking: false` for Qwen 3.x models.

## Step 5: Run the benchmarks

```bash
# Benchmark every quant of the configured HuggingFace repo
scripts/start_bench

# Use a specific config file
scripts/start_bench -c /path/to/my_config.json

# Keep the downloaded GGUF files after each run
scripts/start_bench -k

# Start llama-server only (useful for manual runs)
scripts/start_bench -s Qwen3.6-27B-UD-Q4_K_M.gguf
```

For each quant that hasn't been benchmarked yet, `start_bench` downloads the GGUF, starts `llama-server`, runs every enabled inspect_ai task via `run2.sh`, then stops the server and deletes the GGUF (unless `-k`). Already-benchmarked quants are skipped via a `<model>_done` marker file.

Expect long runtimes on consumer hardware: a full BigCodeBench pass takes roughly 1.5–5 hours per quant depending on model size and GPU.

## Step 6: Extract and aggregate the results

Place (or symlink) the produced `.eval` files under `results/inspect_evals/{model_family}/`, then:

```bash
# .eval files → results/{model_family}/summary.json
python scripts/extract_inspect_summary.py

# Merge all summary.json → results/all_results.{json,csv}
python scripts/aggregate.py
```

## Step 7 (optional): Rebuild the site

```bash
python site/build.py            # writes docs/index.html + assets
python site/build.py --skip-sizes   # skip the HuggingFace size lookups
```

## Common issues

- **Model doesn't load:** Check VRAM. With q4_0 KV cache, a 26B model needs ~12–20 GB depending on quant.
- **Scores abnormally low on a thinking model:** Check that `enable_thinking` is disabled in `chat_template_kwargs` — `<think>` blocks break deterministic scorers and blow up runtimes.
- **inspect_ai connection errors:** Verify `inspect.base_url` points at your llama-server (`http://localhost:8050/v1/` by default) and that `n_parallel` ≥ `max_connections`.
- **A quant keeps being re-run:** Delete or check the `<model>_done` marker files in `done_marker_dir`.
- **Extraction fails on .eval files:** Install `zipfile_zstd` — the .eval archives use Zstd compression.
