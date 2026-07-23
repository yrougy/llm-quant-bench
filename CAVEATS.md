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

## lm-eval-harness vs inspect_ai: mixed results

Starting with the Qwen 3.6 35B A3B MTP model, evaluations switched to inspect_ai. This means **results from this model are not directly comparable to lm-eval results for other models**, even for nominally identical benchmarks like ARC-Challenge.

The two harnesses differ in:
- **Prompt format:** lm-eval used the completions endpoint (`/v1/completions`) with manually applied chat templates. inspect_ai uses `/v1/chat/completions` via the OpenAI client — llama-server applies the chat template automatically.
- **Evaluation logic:** lm-eval uses token log-probabilities for some tasks; inspect_ai uses generative scoring throughout.
- **Task implementations:** The ARC-Challenge task in inspect_ai is not identical to `arc_challenge_chat` in lm-eval — the scoring method and prompt phrasing differ.

**What this means in practice:** Do not compare ARC or other overlapping scores between the MTP model (inspect_ai) and the other models (lm-eval). The comparison charts in section 3 of the site only include benchmarks where the same harness was used, or where the difference is clearly labeled.

## BBEH Mini: growing sample count (currently paused)

**Status:** `bbeh_mini` is disabled in `bench_config.json` and not part of the current published suite (see [METHODOLOGY.md](METHODOLOGY.md)) — the notes below describe an earlier experiment, not an ongoing benchmark.

The BBEH Mini results reflected 100 samples when this was last run. This is not a fixed limit — inspect_ai supports **incremental evaluation**, meaning a run can be paused and resumed later without discarding prior work. New samples could be appended to existing results if this benchmark is picked back up.

At the 100-sample count reached so far:

- Standard error is approximately ±5%
- Differences smaller than ~8 points should be treated as noise
- The gap between IQ1_M (~40%) and IQ4_NL (~54%) is large enough to be meaningful
- The clustering of IQ1_M / IQ2_XXS / Q3_K_M around 40–41% is likely real, but individual differences between those quants are not

The first goal was to produce **readable, discriminating graphs** rather than statistically definitive ones. Precision will improve as the sample count grows. Results in the data files include a `total_samples` field so you can always see how many samples backed a given score.

## Model sources

Most GGUF files come from [Unsloth](https://huggingface.co/unsloth), who provide a wide range of quantization levels including the "UD" (Unsloth Dynamic) variants with non-standard quant schemes. We also tested a few quants from [Bartowski](https://huggingface.co/bartowski) for cross-validation.

Different quantizers can produce slightly different results for the same quant type (e.g., Q4_K_M from Unsloth vs Bartowski). We note the source in the results when it matters.
