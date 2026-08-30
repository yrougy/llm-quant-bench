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

## `enable_thinking` silently ignored on Qwen3.5/3.6/3.8

**The problem:** `bench_config.json` disables thinking via `--chat-template-kwargs '{"enable_thinking":false}'`, which worked for Qwen 3.6 and earlier. Qwen3.8's chat template moved to a graded `reasoning_effort` parameter (`low` / `medium` / `xhigh`, default `xhigh`) instead of the old boolean switch, and does not honor `enable_thinking` at all — it silently stays at its default. This is a known issue in llama.cpp for this model family, not something specific to our setup: [#20182](https://github.com/ggml-org/llama.cpp/issues/20182), [#20409](https://github.com/ggml-org/llama.cpp/issues/20409), [#22255](https://github.com/ggml-org/llama.cpp/issues/22255).

The result was a full Qwen3.8-27B benchmark run (all quants, ~8 days of GPU time) that ran at `xhigh` reasoning effort instead of the intended off/low setting, without erroring or warning.

**How we caught it:** the BFCL scores looked off compared to Qwen3.6-27B-MTP — not just noisy, but *systematically* lower on every single quant (16/16, mean −6.1pp). Checking the raw `.eval` transcripts for matched samples confirmed a real behavioral difference, not just score noise — with one methodological wrinkle worth flagging: BFCL's multi-turn categories make several sequential model calls per sample, and a sample's own `output.usage` field only reflects the *last* of those calls. The true per-sample cost is `model_usage` (present both per-sample and, pre-summed for the whole run, under `stats.model_usage` in the header) — use that, not `output.usage`, for anything multi-turn. Measured correctly, on the full `IQ4_NL` runs: Qwen3.6-27B-MTP averages 281.0 output tokens/sample across all 1000 BFCL samples; Qwen3.8-xhigh averages 403.2 (+43%). On one matched multi-turn sample (`multi_turn_composite_176`, identical prompt and history in both runs), Qwen3.6 used 1,938 output tokens across its full turn sequence and executed the requested tool sequence (create ticket, close ticket); Qwen3.8-xhigh used 2,110 (+9%, not the dramatic multiple an earlier version of this entry claimed from reading only the last turn) and stalled, asking clarifying questions instead of calling the tools. MUSR moved the *opposite* direction (better on Qwen3.8, 14/16 quants, mean +4.0pp) — consistent with extra reasoning effort helping free-form narrative reasoning while hurting strict tool-call formatting, rather than a genuine model regression. BigCodeBench was roughly flat (mean +1.0pp, mixed direction) — code output seems more tolerant of a longer deliberation than a strict tool-call schema is.

**What does *not* show the effect, despite looking like it should:** `bfcl_multi_turn_composite_acc` sits near 0 in the xhigh run (mean 0.035 across quants) and it's tempting to read that as corroborating evidence. It isn't — this subcategory only has ~20–40 scored samples per quant (stderr swings ±3–5pp on a single flipped sample), and it sits in the same near-zero band for every model in this project regardless of thinking config: Qwen3.6-27B-MTP, benchmarked with thinking correctly disabled, averages 0.048 on the *same* metric (higher than the broken xhigh run), and Qwen3.5-9B averages 0.006 (lower). Don't use this specific submetric to judge whether a Qwen3.5/3.6/3.8 run was affected by this bug — it's too noisy to discriminate a working run from a broken one either way. The composite_176 walkthrough above and the aggregate token-count/BFCL-score deltas are the evidence that actually holds up; an earlier version of this entry cited `multi_turn_composite_acc` as supporting evidence, which was a mistake.

Two quants (`UD-Q4_K_XL`, `UD-Q6_K_XL`) OOM'd/were cancelled mid-run and produced no data — plausibly because xhigh's longer generations push harder on VRAM/KV-cache, and these two happened to be the largest quants in the batch.

**First look at a `reasoning_effort: medium` rerun (2026-08-25, `UD-IQ1_M`, Unsloth Dynamics 3.0):** measured correctly via `stats.model_usage` (see the note above — this was not caught by the first pass of this analysis, which read only last-turn `output.usage` and reported a misleadingly small 381 tokens/sample), the true average is **1,276 output tokens/sample** across the 1000-sample BFCL run — 3.2x the broken xhigh `IQ4_NL` run (403) and 4.5x the clean Qwen3.6 baseline (281). Total run cost (output + cache-replayed input across all turns) is **27.46M tokens** — confirmed against the actual `.eval` file, matching the "27M, a record for a 27B" figure from direct observation of the run. On the matched `multi_turn_composite_176` sample specifically: 6,982 output tokens — the *worst* of the three runs (vs. 1,938 for Qwen3.6, 2,110 for Qwen3.8-xhigh) — and the tool sequence was still not executed correctly (scored 0, same as xhigh). So on this sample, `medium` + `UD-IQ1_M` is not an improvement over `xhigh` + `IQ4_NL`; if anything it's worse, though the two runs differ in both settings at once (reasoning effort *and* quant), so this alone doesn't isolate which one is responsible.

**Quantization alone, holding reasoning effort constant, produces the same kind of blowup.** Qwen3.6-35B-A3B-MTP was benchmarked with a single, constant `bench_config.json` across all 11 quants (no reasoning-effort changes between them), so its BFCL runs isolate quantization aggressiveness as the only variable. True per-quant averages (`stats.model_usage`, same method as above):

| Quant | avg output tok/sample | total run tokens (output + cache) |
|---|---|---|
| `UD-Q4_K_M` | 302 | 19.1M |
| `UD-Q4_K_S` | 357 | 44.4M |
| `UD-IQ4_XS` | 340 | 26.6M |
| `UD-IQ4_NL` | 390 | 32.5M |
| `MXFP4_MOE` | 407 | 47.2M |
| `UD-Q2_K_XL` | 409 | 101.3M |
| `UD-IQ3_XXS` | 479 | 60.1M |
| `UD-Q3_K_M` | 481 | 70.4M |
| `UD-IQ2_M` | 494 | 92.7M |
| `UD-Q3_K_XL` | 503 | 80.3M |
| `UD-IQ1_M` | **607** | **127.7M** |

Not a clean monotonic curve (`UD-Q2_K_XL` sits below several nominally-better quants), but the two `Q4_K_*` quants are unambiguously the cheapest and `UD-IQ1_M` is unambiguously the most expensive — a 6.7x spread in total run cost with *zero* change to reasoning effort. This is real, independent-of-`reasoning_effort` evidence that aggressive quantization on its own drives this family toward longer, more repetitive multi-turn trajectories — supporting the read that the Qwen3.8-27B `UD-IQ1_M`/medium result above is at least partly a quantization effect, not purely a leftover reasoning-effort problem. It still doesn't fully separate the two for Qwen3.8 specifically — that needs a medium-effort run on a quant that also has a clean, same-config Qwen3.6 comparison point (e.g. `Q4_K_M`).

(Open question, not yet checked: whether Qwen3.6-35B-A3B-MTP's own `enable_thinking:false` was actually honored — the llama.cpp issues cited above mention Qwen3.6 alongside 3.5/3.8, so this run's baseline reasoning state hasn't been independently verified either.)

**The fix:** pass `reasoning_effort` explicitly instead of `enable_thinking` — either `--reasoning-effort medium` as a llama-server startup flag, or `--chat-template-kwargs '{"reasoning_effort":"medium"}'`. Only `low` / `medium` / `xhigh` are valid for this model's template (`high` is not).

**Impact:** the affected run is kept as `Qwen3.8-27B-xhigh` under `results/` and `results/inspect_evals/` for historical reference (see the README there), but is excluded from the site — the corresponding entry in `site/models.yaml` is commented out. A medium-effort rerun will replace it under the unsuffixed `Qwen3.8-27B` name once it's done. If you're benchmarking any Qwen3.5/3.6/3.8 model, verify `reasoning_effort` actually took effect (check output token counts or response latency) rather than trusting `enable_thinking`.

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
