# Methodology

This document describes the benchmarks, the evaluation setup, and the reasoning behind each choice.

## Goal

The goal is to measure how much accuracy is lost when quantizing a model from high-precision (Q6_K, Q5_K) down to aggressive quants (IQ2_XXS, IQ1_M). We are **not** comparing models to each other — we are comparing quantization levels within the same model family.

This means:

- Scores should not be compared to official model benchmarks (different setup, different hardware, different prompts).
- What matters is the **delta** between quants, not the absolute numbers.
- Each model family has its own baseline (the highest quant tested).

## Evaluation tool

All evaluations use [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (version 0.4.12).

The models are served locally via `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) and queried through the OpenAI-compatible completions endpoint (`/v1/completions`). The harness connects as a `local-completions` backend.

## Inference setup

All tests share the following llama-server configuration:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Context length | 16384–65536 | Depends on model requirements |
| Flash attention | on | |
| KV cache type (K) | q4_0 | Reduces VRAM usage |
| KV cache type (V) | q4_0 | Same |
| Batch size | 1024 | |
| Temperature | 0.6 | |
| Top-p | 0.95 | |
| Top-k | 20 | |
| Min-p | 0.00 | |

For thinking models (Qwen 3.x, Gemma 4), the `enable_thinking` parameter in `chat-template-kwargs` is set depending on the test:

- **Thinking enabled** for Gemma 4 (GSM8K benefits from chain-of-thought)
- **Thinking disabled** for Qwen 3.6 (the `<think>` tags interfere with IFEval scoring — see [CAVEATS.md](CAVEATS.md))

## Benchmarks

### GSM8K — Grade School Math

- **Dataset:** ~8,500 grade-school math word problems
- **What it tests:** Multi-step arithmetic reasoning with intermediate dependencies
- **Evaluation:** Exact match on the final numerical answer
- **N-shot:** 8
- **Metrics reported:** `flexible-extract` (lenient parsing) and `strict-match` (exact format)

GSM8K is the most stable benchmark across quantization levels. The reasoning chain is relatively short and the answer space is numerical, so even aggressive quants tend to preserve the core capability.

The `flexible-extract` metric uses regex to find the final number in the model's response, while `strict-match` requires the answer to be in the expected format. The gap between the two reveals how much the model's formatting drifts at lower quants.

### ARC-Challenge (Chat variant)

- **Dataset:** ~1,172 science multiple-choice questions (filtered for difficulty)
- **What it tests:** Scientific reasoning, combining facts, resisting distractors
- **Evaluation:** Exact match on the selected answer letter (A/B/C/D)
- **N-shot:** 0
- **Metric:** `exact_match` with whitespace removal

**Important:** We use `arc_challenge_chat` instead of the standard `arc_challenge`. The standard version relies on token log-probabilities to select the answer, but llama.cpp's completions API does not return probabilities in a format compatible with lm-eval-harness. This produced inconsistent results.

The chat variant instead asks the model to generate a response ending with "The best answer is [letter]" and parses the output. This is less fine-grained (no probability ranking) but produces reliable, interpretable results with llama.cpp.

See [CAVEATS.md](CAVEATS.md) for the full story on this issue.

### IFEval — Instruction Following Evaluation

- **Dataset:** ~540 prompts with verifiable formatting constraints
- **What it tests:** Strict obedience to structural instructions (word count, formatting, keywords, etc.)
- **Evaluation:** Deterministic Python scripts that check constraint satisfaction
- **N-shot:** 0
- **Metrics reported:**
  - `prompt_level_strict_acc` — % of prompts where ALL constraints are met (strict parsing)
  - `prompt_level_loose_acc` — Same with lenient parsing
  - `inst_level_strict_acc` — % of individual instructions satisfied (strict)
  - `inst_level_loose_acc` — Same with lenient parsing

IFEval is the most practically relevant benchmark for agentic use cases. A model that can't follow formatting instructions reliably will break tool-calling pipelines, structured output generation, and multi-step agent workflows.

The strict/loose distinction matters: strict parsing checks exact compliance, while loose allows minor deviations (extra whitespace, case differences). A large gap between strict and loose suggests the model "understands" the instruction but is sloppy in execution.

### HumanEval — Code Generation

- **Dataset:** 164 Python programming problems
- **What it tests:** Ability to generate correct, runnable Python code
- **Evaluation:** pass@1 (single attempt, code is executed and tested)
- **N-shot:** 0

HumanEval tests practical coding ability. Each problem provides a function signature and docstring; the model must complete the function body. The generated code is then executed against test cases.

Note: HumanEval results can have high variance (stderr ~±3.8%). Small differences between quants should not be over-interpreted.

## What's NOT tested

- **Speed / throughput:** This project only measures accuracy. Quantization obviously affects speed, but that's hardware-dependent and well-documented elsewhere.
- **Perplexity:** While perplexity is the most sensitive measure of quantization quality, it doesn't directly predict task performance.
- **MMLU:** We ran MMLU on a couple of Gemma 4 quants but suspended it — each run takes 6+ hours on our setup, and the signal-to-noise ratio for quantization comparison was low. The results we do have are in the raw data.
- **Creative / subjective tasks:** No benchmark captures writing quality, conversational ability, or nuanced reasoning. These tests measure the mechanical capabilities that quantization is most likely to degrade.
