# How to reproduce

This guide walks you through reproducing the benchmark results on your own hardware.

## Prerequisites

- A GPU with enough VRAM for the model + KV cache (24 GB minimum for 26–35B models at low quants). 
	- I use 2 RTX3060-12GiB for "big" models
	- I use GTX1070-8GiB for small models 
- Linux 
	- Should work on other *Nix
	- Ubuntu 22.04
	- NVidia Drivers 550 (for staibility)
	- Cuda: 12.4 (for stability)
	- Software configuration is subject to change in the future
- Python 3.10+
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

## Step 2: Download GGUF models

Download the quantized models you want to test. We primarily used Unsloth's quantizations from HuggingFace:

- [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)

```bash
# Example with huggingface-cli
pip install huggingface_hub
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
    gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
    --local-dir /data/models/unsloth/
```

## Step 3: Install lm-evaluation-harness

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Step 4: Run a benchmark

Start the llama-server with your model:

```bash
./llama-server \
    -m /data/models/unsloth/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
    -c 65536 \
    -fit off \
    -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -b 1024 -ub 1024 \
    --port 8050 \
    --host 0.0.0.0 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --chat-template-kwargs '{"enable_thinking":true}'
```

Then run the evaluation (in a separate terminal, with the venv activated):

```bash
# GSM8K
lm_eval --model local-completions \
    --model_args "base_url=http://localhost:8050/v1/completions,api_key=EMPTY,pretrained=google/gemma-4-26B-A4B,tokenizer=google/gemma-4-26b-a4b-it" \
    --tasks gsm8k \
    --num_fewshot 8 \
    --batch_size 1 \
    --apply_chat_template \
    --output_path ./gemma-4-26B-A4B-it-UD-Q4_K_M.gguf-gsm8k

# ARC-Challenge (chat)
lm_eval --model local-completions \
    --model_args "model=google/gemma-4-26B-A4B-it,base_url=http://localhost:8050/v1/completions,api_key=EMPTY,tokenizer=google/gemma-4-26B-A4B-it" \
    --tasks arc_challenge_chat \
    --batch_size 1 \
    --apply_chat_template \
    --output_path ./gemma-4-26B-A4B-it-UD-Q4_K_M.gguf-arc-chat

# IFEval
lm_eval --model local-completions \
    --model_args "base_url=http://localhost:8050/v1/completions,api_key=EMPTY,pretrained=google/gemma-4-26B-A4B-it,tokenizer=google/gemma-4-26B-A4B-it" \
    --tasks ifeval \
    --num_fewshot 0 \
    --batch_size 1 \
    --apply_chat_template \
    --output_path ./gemma-4-26B-A4B-it-UD-Q4_K_M.gguf-ifeval
```

## Step 6: Automate multiple quants

Use the orchestration script to iterate over multiple GGUF files. Use the file in the repo, as I'm still writing it (it's very hardcoded at the moment). Adapt the paths to your setup:

```bash
#!/usr/bin/env bash
MODEL_DIR="/data/models/unsloth"
LLAMA_SERVER="/path/to/llama-server"

for gguf in "$MODEL_DIR"/gemma-4-26B-A4B-it-UD-*.gguf; do
    filename=$(basename "$gguf")
    echo "=== Testing $filename ==="

    $LLAMA_SERVER \
        -m "$gguf" \
        -c 65536 -fit off -fa on \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        -b 1024 -ub 1024 \
        --port 8050 --host 0.0.0.0 \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
        --chat-template-kwargs '{"enable_thinking":true}' &

    sleep 15  # Wait for server to load the model

    ./run.sh "$filename"

    pkill llama-server
    sleep 5
done
```

## Step 7: Extract summaries

Once all benchmarks are done, run the extraction script:

```bash
python scripts/extract_summary.py /path/to/bench/results --output results/
```

This produces:

- `results/{model}/summary.json` — one file per model family
- `results/all_results.csv` — global CSV
- `results/all_results.json` — global JSON

## Common issues

- **Model doesn't load:** Check VRAM. With q4_0 KV cache, a 26B model needs ~12–20 GB depending on quant.
- **GSM8K scores are 0%:** You probably forgot `--apply_chat_template` or the tokenizer path is wrong.
- **ARC-Challenge returns random scores:** Use `arc_challenge_chat`, not `arc_challenge`. See [CAVEATS.md](CAVEATS.md).
- **IFEval scores are abnormally low:** Check if `<think>` tags are being included in the response. Disable thinking or strip the tags.
