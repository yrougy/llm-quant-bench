# Caveats & lessons learned

This document records the pitfalls we encountered during benchmarking and how we addressed them. If you're setting up your own GGUF evaluation pipeline, reading this first may save you hours of debugging.

## ARC-Challenge and log-probabilities

**The problem:** The standard `arc_challenge` task in lm-eval-harness works by comparing the log-probabilities of each answer option (A, B, C, D). The model doesn't generate text — instead, the harness feeds each option as a continuation and picks the one with the highest probability.

This approach requires the inference backend to return accurate log-probabilities per token. llama.cpp's `/v1/completions` endpoint does return `logprobs`, but in a format that lm-eval-harness does not always interpret correctly. The result was wildly inconsistent scores — sometimes even below random chance (25%).

**The fix:** We switched to `arc_challenge_chat`, a generative variant where the model produces a complete answer ("The best answer is B") and the harness parses the letter. This is less fine-grained (no probability ranking, just pass/fail on the extracted letter) but produces consistent, interpretable results.

**Impact:** Scores from `arc_challenge_chat` should not be compared to standard `arc_challenge` numbers published elsewhere. They measure the same underlying capability but through a different evaluation method.

## Chat templates and tokenizers

**The problem:** Running lm-eval-harness against a chat model without specifying the correct tokenizer and applying the chat template produces garbage results. The model receives raw text instead of properly formatted `<|im_start|>user\n...<|im_end|>` sequences, which degrades all benchmarks.

**The fix:** Always pass both `--apply_chat_template` and the correct `tokenizer=` in `--model_args`. For Qwen models, this means `tokenizer=Qwen/Qwen3.5-4B` (or the appropriate model ID). For Gemma, `tokenizer=google/gemma-4-26b-a4b-it`.

Without this, GSM8K scores can drop to near 0% — not because the model can't do math, but because it doesn't understand the prompt format.

## Thinking tags and IFEval

**The problem:** Models with chain-of-thought (Qwen 3.x with `enable_thinking:true`, Gemma 4) wrap their reasoning in `<think>...</think>` tags before producing the final answer. IFEval's scoring scripts count words, check formatting constraints, and verify structural rules on the **entire** model output — including the thinking tags.

This means a model that thinks "let me count the words... the user wants exactly 100 words..." before answering will fail word-count constraints because the `<think>` block inflates the total. Similarly, a "start your response with X" constraint fails if the response starts with `<think>`.

**The fix:** For Qwen 3.6, we disabled thinking via `--chat-template-kwargs '{"enable_thinking":false}'`. For Gemma 4, thinking was left enabled because the impact on IFEval was smaller (Gemma's thinking tends to be shorter).

A more robust fix would be to strip `<think>...</think>` blocks before scoring, but this requires modifying the harness or post-processing the samples. We chose the simpler approach for now.

**Impact:** IFEval scores with thinking enabled are systematically lower than they "should" be. If you compare our numbers to benchmarks that strip thinking tags, expect a gap.

## MMLU evaluation time

**The problem:** MMLU consists of 57 subcategories and ~14,000 questions. At batch_size=1 through a local completions endpoint, a single full MMLU run takes 6–8 hours per quant on our hardware. Testing 10+ quants means 60–80 hours of GPU time for a single model.

**The decision:** We ran MMLU on two Gemma 4 quants (IQ2_XXS and IQ4_NL) to calibrate, and the results showed relatively small deltas (~6 points between the most and least aggressive quant). Given the time cost and the modest signal, we suspended MMLU testing and focused on benchmarks with better signal-to-time ratios.

The raw MMLU results we do have are included in the data for reference.

## HumanEval variance

**The problem:** HumanEval has only 164 problems, which means the standard error is relatively high (~±3.8%). A difference of 5 points between two quants is often within noise.

**What to watch for:** When we saw certain Qwen 3.6 A3B quants scoring 64–65% on HumanEval while neighboring quants scored 56–58%, we flagged these as potential anomalies (marked with ** in the results tables). These need to be re-run to confirm.

**General rule:** For HumanEval, only trust differences of 8+ points as meaningful. Anything smaller is likely noise.

## Batch size and reproducibility

All tests use `batch_size=1`. This is intentional: batch size can subtly affect results when using a local server (timing, context handling, padding). Using 1 ensures each question is evaluated independently, at the cost of speed.

## KV cache quantization

We use `q4_0` for both K and V cache. This introduces a small amount of noise compared to fp16 KV cache, but the effect is negligible relative to weight quantization and it allows testing larger models in limited VRAM.

If you want to reproduce our exact numbers, use the same KV cache settings. If you have more VRAM, using fp16 KV cache will give slightly cleaner results.

## Model sources

Most GGUF files come from [Unsloth](https://huggingface.co/unsloth), who provide a wide range of quantization levels including the "UD" (Unsloth Dynamic) variants with non-standard quant schemes. We also tested a few quants from [Bartowski](https://huggingface.co/bartowski) for cross-validation.

Different quantizers can produce slightly different results for the same quant type (e.g., Q4_K_M from Unsloth vs Bartowski). We note the source in the results when it matters.
