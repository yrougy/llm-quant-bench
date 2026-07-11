# Methodology

This document describes the benchmarks, the evaluation setup, and the reasoning behind each choice.

## Goal

The goal is to measure how much accuracy is lost when quantizing a model from high-precision (Q8_0, Q6_K) down to aggressive quants (IQ2_XXS, IQ1_M). We are **not** comparing models to each other — we are comparing quantization levels within the same model family.

This means:

- Scores should not be compared to official model benchmarks (different setup, different hardware, different prompts).
- What matters is the **delta** between quants, not the absolute numbers.
- Each model family has its own baseline (the highest quant tested).

## Evaluation harness

All v2 evaluations run through [inspect_ai](https://inspect.ai-safety-institute.org.uk/) using the [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) task library. Models are served locally via `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) through the OpenAI-compatible chat endpoint (`/v1/chat/completions`), and inspect_ai connects as an `openai-api` provider.

Scoring is **fully deterministic** — unit test execution, MCQ accuracy, or AST matching. No LLM judge is involved at any point.

Earlier results were produced with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on a different benchmark suite (v1). Those results and the reasons for moving away from that suite are preserved in [LEGACY.md](LEGACY.md).

## Inference setup

All tests share the following `llama-server` configuration (see `scripts/bench_config.json`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Context length | 16384–32768 | Up to 32k depending on model / GPU |
| Flash attention | on | |
| KV cache type (K) | q4_0 | Reduces VRAM usage, reflects consumer setups |
| KV cache type (V) | q4_0 | Same |
| Batch size | 1024 | |
| Parallel slots | 3 | Matched to inspect_ai `max_connections` |
| Temperature | 0.6 | |
| Top-p | 0.95 | |
| Top-k | 20 | |
| Min-p | 0.00 | |

For reproducibility, inspect_ai runs use `--seed 42` and `--sample-shuffle 42`.

For thinking models (Qwen 3.x), `enable_thinking` is set to `false` in `chat-template-kwargs`: `<think>` blocks pollute deterministic scorers and inflate evaluation time (see [CAVEATS.md](CAVEATS.md) for the v1 IFEval story that motivated this).

## Hardware

- **2× RTX 3060 12 GiB** — larger models (27B+)
- **GTX 1070 8 GiB** — small models / small quants

No cloud, no H100. This is deliberate: the results reflect what these quants actually do on the consumer hardware people run them on — including the q4_0 KV cache that wouldn't be needed with more VRAM.

## The v2 benchmark suite

The four benchmarks were chosen on three criteria: **non-saturated** on modern models, **deterministic scoring**, and **fast enough** to run systematically across 10–20 quants per model on consumer hardware.

### BigCodeBench — Code generation

- **Dataset:** 1,140 practical programming tasks across real-world libraries (`complete` split)
- **What it tests:** Practical coding ability — much harder than HumanEval
- **Evaluation:** Unit test execution, pass@1, sandboxed
- **Samples:** Full benchmark, no limit

### MUSR — Multi-step soft reasoning

- **Dataset:** ~750 narrative reasoning questions (murder mysteries, object placement, team allocation)
- **What it tests:** Multi-step reasoning over long narrative contexts
- **Evaluation:** MCQ accuracy
- **Samples:** Full dataset

### GPQA — Graduate-level science Q&A

- **Dataset:** ~450 questions in chemistry, physics, biology, written by domain experts
- **What it tests:** Expert scientific reasoning — even frontier models are far from the ceiling
- **Evaluation:** MCQ accuracy
- **Samples:** Full dataset

### BFCL — Function calling

- **Dataset:** Berkeley Function-Calling Leaderboard
- **What it tests:** Generating well-formed structured tool calls from a function spec — the practical agentic capability
- **Evaluation:** AST matching (deterministic)
- **Samples:** `--limit 1000` with `--seed 42` for reproducibility across models

### Current status

BigCodeBench runs are complete for the published models; MUSR, GPQA and BFCL runs are in progress and will be added as they complete.

## Why the v1 suite was dropped

The first version of this project used ARC-Challenge, IFEval, GSM8K, HumanEval and later BBEH-mini. They were dropped for concrete reasons:

- **Saturation** — modern 9B+ models score so close to the ceiling on ARC/IFEval/GSM8K that quant differences disappear into noise.
- **Format incompatibility** — HumanEval's completion format doesn't fit instruct models; IFEval scores were polluted by `<think>` tags.
- **Runtime** — BBEH-mini takes days per model on consumer hardware; MMLU takes 6–8 hours per quant. Impractical for systematic quant comparisons.
- **race_h** was also evaluated and abandoned (saturated at ~89% on a 9B model).

Full v1 results and per-benchmark details are in [LEGACY.md](LEGACY.md); the debugging war stories are in [CAVEATS.md](CAVEATS.md).

## What's NOT tested

- **Speed / throughput:** This project only measures accuracy. Quantization obviously affects speed, but that's hardware-dependent and well-documented elsewhere.
- **Perplexity:** While perplexity is the most sensitive measure of quantization quality, it doesn't directly predict task performance.
- **Creative / subjective tasks:** No benchmark here captures writing quality, conversational ability, translation, or summarization. These tests measure the mechanical capabilities that quantization is most likely to degrade.
