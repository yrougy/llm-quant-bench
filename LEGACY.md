# GGUF Quantization Benchmark

Measuring the real-world accuracy cost of GGUF quantization on small open-weight LLMs, so you can pick the right quant for your use case.

## Why this exists

Quantization lets you run large models on consumer GPUs, but published benchmarks rarely test quantized GGUF variants head-to-head. This project fills that gap: same hardware, same evaluation harness, same prompts — only the quantization level changes.

The goal is **not** to rank models against each other, but to answer: *"For a given model, how much accuracy do I lose going from Q6_K down to IQ2_XXS?"*

## Note

I run all the benchmarks on my own hardware. It's not very powerful, and it's not dedicated to benchmarking, it has other uses, so it can take a while for a full benchmark. Qwen 3.6 27B bench without GSM8K run for about 100 hours in a row for example.

## What's tested

| Benchmark | What it measures | Why it matters |
|-----------|-----------------|----------------|
| **GSM8K** | Multi-step arithmetic reasoning | Core chain-of-thought ability |
| **ARC-Challenge (chat)** | Science QA with distractors | Semantic comprehension & robustness |
| **IFEval** | Strict instruction following | Critical for agentic / tool-use workflows |
| **HumanEval** | Python code generation (pass@1) | Practical coding ability |
| **HumanEval+** | Same problems, 80× more test cases | Catches false positives from HumanEval |


All evaluations run through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with models served via [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Models covered

| Model | Size | Type | Status |
|-------|------|------|--------|
| Qwen 3.6 27B (Unsloth UD) | 27B | Dense | Complete (no GSM8K) |
| Qwen 3.6 35B A3B (Unsloth UD) | 35B MoE | MoE | Complete (no GSM8K, some anomalies to rerun) |
| Qwen 3.5 4B | 4B | Dense | Complete (no GSM8K, no IFEval) |
| Qwen 3.5 9B | 9B | Dense | Complete (no GSM8K, no IFEval) |
| Gemma 4 27B A4B (Unsloth UD) | 27B MoE | MoE | Partial — to be redone with HumanEval |

## Quick results

### Qwen 3.6 35B A3B — Unsloth UD quants

> GSM8K not done yet


|Quantisation                     | GSM8K |   IfEval    |  Arc_Challenge_Chat | Human Eval  (err +- 3.8)  |
|---------------------------------|-------|-------------|---------------------|---------------------------|
| Qwen3.6-35B-A3B-UD-IQ1_M        |   -   |   81%/89%   |      95%            |    56%                    |
| Qwen3.6-35B-A3B-UD-IQ2_XXS      |   -   |   81%/89%   |      95%            |    48%                    |
| Qwen3.6-35B-A3B-UD-IQ2_M        |   -   |   82%/91%   |      96%            |    50%                    |
| Qwen3.6-35B-A3B-UD-Q2_K_XL      |   -   |   82%/91%   |      95%            |    61%                    |
| Qwen3.6-35B-A3B-UD-IQ3_XXS      |   -   |   83%/91%   |      95%            |    58%                    |
| Qwen3.6-35B-A3B-UD-IQ3_S        |   -   |   82%/91%   |      96%            |    58%                    |
| Qwen3.6-35B-A3B-UD-Q3_K_S       |   -   |   82%/90%   |      95%            |    60%                    |
| Qwen3.6-35B-A3B-UD-Q3_K_M       |   -   |   82%/91%   |      95%            |    61%                    |
| Qwen3.6-35B-A3B-UD-Q3_K_XL      |   -   |   82%/91%   |      95%            |    64%**                  |
| Qwen3.6-35B-A3B-UD-IQ4_XS       |   -   |   82%/91%   |      95%            |    56%                    |
| Qwen3.6-35B-A3B-UD-Q4_K_S       |   -   |   82%/91%   |      95%            |    65%**                  |
| Qwen3.6-35B-A3B-UD-IQ4_NL       |   -   |   81%/90%   |      95%            |    61%                    |
| Qwen3.6-35B-A3B-UD-Q4_K_M       |   -   |   83%/91%   |      95%            |    59%                    |
| Qwen3.6-35B-A3B-UD-Q4_K_XL      |   -   |   84%/91%   |      95%            |    58%                    |
| Qwen3.6-35B-A3B-UD-IQ4_NL_XL    |   -   |   81%/90%   |      95%            |    62%                    |
| Qwen3.6-35B-A3B-MXFP4_MOE       |   -   |   83%/91%   |      95%            |    62%                    |

** Looks like anomaly, will be redone

**Takeaway**: The MoE architecture makes this model exceptionally resilient. ARC-Chat barely moves across the entire range (95–96%), and IFEval strict stays within 3 points from `IQ1_M` to `Q4_K_XL`. Even `IQ1_M` — the most aggressive quant tested — produces usable results. HumanEval is noisier due to both the small dataset variance (±3.8%) and the MoE weight distribution; several results are flagged as anomalies and need re-running before drawing conclusions. Pending confirmation, `Q3_K_M` or `Q3_K_S` appear to be the floor for reliable coding output, with marginal gains above Q4.

> Full results for all models are in the [`results/`](results/) directory as JSON and CSV.



### Qwen 3.6 27B — Unsloth UD quants

> GSM8K not done yet

| Quantisation           | GSM8K | IFEval | ARC-Chat | HumanEval | HumanEval+ |
| ---------------------- | ----: | -----: | -------: | --------: | ---------: |
| Qwen3.6-27B-UD-IQ2_XXS |     — |  82.6% |    96.1% |     59.8% |      53.0% |
| Qwen3.6-27B-UD-IQ2_M   |     — |  84.7% |    96.5% |     71.3% |      67.1% |
| Qwen3.6-27B-UD-Q2_K_XL |     — |  85.0% |    96.3% |     76.2% |      68.3% |
| Qwen3.6-27B-UD-IQ3_XXS |     — |  84.3% |    97.1% |     79.3% |      72.0% |
| Qwen3.6-27B-Q3_K_S     |     — |  84.3% |    97.0% |     81.7% |      72.6% |
| Qwen3.6-27B-Q3_K_M     |     — |  85.6% |    97.1% |     79.3% |      73.2% |
| Qwen3.6-27B-UD-Q3_K_XL |     — |  85.0% |    97.2% |     77.4% |      72.0% |
| Qwen3.6-27B-IQ4_NL     |     — |  84.7% |    97.0% |     80.5% |      76.2% |
| Qwen3.6-27B-IQ4_XS     |     — |  86.1% |    96.9% |     79.9% |      75.6% |
| Qwen3.6-27B-Q4_K_S     |     — |  86.1% |    97.1% |     84.8% |      76.8% |
| Qwen3.6-27B-Q4_K_M     |     — |  85.0% |    97.2% |     82.3% |      76.8% |
| Qwen3.6-27B-UD-Q4_K_XL |     — |  86.0% |    97.3% |     84.1% |      75.6% |
| Qwen3.6-27B-Q4_0       |     — |  85.8% |    97.1% |     82.9% |      78.7% |
| Qwen3.6-27B-Q4_1       |     — |  84.8% |    97.2% |     81.7% |      76.8% |
| Qwen3.6-27B-Q5_K_S     |     — |  84.5% |    97.0% |     84.1% |      78.7% |
| Qwen3.6-27B-Q5_K_M     |     — |  84.3% |    97.1% |     84.1% |      77.4% |
| Qwen3.6-27B-UD-Q5_K_XL |     — |  85.4% |    97.2% |     84.1% |      78.0% |
| Qwen3.6-27B-Q6_K       |     — |  85.2% |    97.1% |     83.5% |      78.7% |


| Test                   | Stderr | 
| ---------------------- | -----: |
| GSM8K                  |    -   | 
| IFEval                 |   1.5% | 
| ARC-Chat               |  0.5%  | 
| HumanEval              |    3%  | 
| HumanEval+             |  3.4%  | 

**Takeaway**: ARC-Chat and IFEval remain remarkably stable across the entire quantization range — even `IQ2_XXS` loses less than 4 points on either benchmark. HumanEval is where quantization bites: `IQ2_XXS` scores 25 points below `Q4_K_S` (59.8% vs 84.8%), with a sharp 11-point drop between `IQ2_XXS` and `IQ2_M` alone. If coding ability matters, `IQ2_M` is the floor. The sweet spot is `Q4_K_S`: it matches or exceeds all higher quants on every benchmark, including `Q6_K`, at a significantly smaller file size.

> Full results for all models are in the [`results/`](results/) directory as JSON and CSV.


### Qwen 3.5 4B — Unsloth UD quants

> GSM8K & IfEval not done yet


| Quantisation          | GSM8K | IFEval | ARC-Chat | HumanEval | HumanEval+ |
| --------------------- | ----: | -----: | -------: | --------: | ---------: |
| Qwen3.5-4B-IQ4_NL     |     — |      — |    91.2% |     60.4% |      55.5% |
| Qwen3.5-4B-IQ4_XS     |     — |      — |    91.1% |     56.7% |      53.0% |
| Qwen3.5-4B-Q3_K_M     |     — |      — |    91.8% |     56.1% |      50.0% |
| Qwen3.5-4B-Q3_K_S     |     — |      — |    90.7% |     50.0% |      44.5% |
| Qwen3.5-4B-Q4_0       |     — |      — |    91.5% |     56.7% |      50.6% |
| Qwen3.5-4B-Q4_1       |     — |      — |    91.5% |     59.8% |      53.7% |
| Qwen3.5-4B-Q4_K_M     |     — |      — |    91.8% |     59.8% |      54.9% |
| Qwen3.5-4B-Q4_K_S     |     — |      — |    91.7% |     56.1% |      50.6% |
| Qwen3.5-4B-Q5_K_M     |     — |      — |    92.3% |     58.5% |      53.7% |
| Qwen3.5-4B-Q5_K_S     |     — |      — |    92.4% |     57.3% |      52.4% |
| Qwen3.5-4B-Q6_K       |     — |      — |    92.2% |     57.9% |      52.4% |
| Qwen3.5-4B-Q8_0       |     — |      — |    92.2% |     60.4% |      53.0% |
| Qwen3.5-4B-UD-IQ2_M   |     — |      — |    82.8% |     37.2% |      32.3% |
| Qwen3.5-4B-UD-IQ2_XXS |     — |      — |    34.0% |     17.7% |      15.9% |
| Qwen3.5-4B-UD-IQ3_XXS |     — |      — |    90.4% |     48.8% |      44.5% |
| Qwen3.5-4B-UD-Q2_K_XL |     — |      — |    86.1% |     39.6% |      32.9% |
| Qwen3.5-4B-UD-Q3_K_XL |     — |      — |    91.3% |     58.5% |      52.4% |
| Qwen3.5-4B-UD-Q4_K_XL |     — |      — |    92.2% |     61.6% |      55.5% |
| Qwen3.5-4B-UD-Q5_K_XL |     — |      — |    92.6% |     57.3% |      52.4% |
| Qwen3.5-4B-UD-Q6_K_XL |     — |      — |    92.5% |     57.3% |      51.8% |
| Qwen3.5-4B-UD-Q8_K_XL |     — |      — |    92.3% |     57.3% |      50.0% |

**Takeaway**: This 4B model has a hard cliff at 2-bit quantization. `IQ2_XXS` causes a near-total collapse (34% ARC, 17.7% HumanEval — worse than random on code). `IQ2_M` and `Q2_K_XL` are also significantly degraded. Above IQ3_XXS, ARC stabilizes around 91–92% and stays flat all the way to `Q8_0`. HumanEval plateaus in the 57–61% range from Q4 onward — going from `Q4_K_M` to `Q8_0` yields essentially nothing. The `UD-Q4_K_XL` is the best performer overall (61.6% HumanEval, 92.2% ARC). For this model, Q4 is both the floor and the ceiling worth caring about.

> Full results for all models are in the [`results/`](results/) directory as JSON and CSV.


### Qwen 3.5 9B — Unsloth UD quants


| Quantization          | GSM8K | IFEval | ARC-Chat | HumanEval | HumanEval+ |
| --------------------- | ----: | -----: | -------: | --------: | ---------: |
| Qwen3.5-9B-UD-IQ2_XXS | 62.3% |  64.5% |    86.3% |     25.0% |      22.6% |
| Qwen3.5-9B-UD-IQ2_M   | 82.8% |  75.6% |    90.8% |     53.7% |      46.3% |
| Qwen3.5-9B-UD-Q2_K_XL | 82.6% |  77.8% |    92.4% |     46.3% |      39.6% |
| Qwen3.5-9B-UD-IQ3_XXS | 86.7% |  80.4% |    93.1% |     60.4% |      54.3% |
| Qwen3.5-9B-Q3_K_S     | 81.6% |  83.0% |    92.2% |     63.4% |      57.3% |
| Qwen3.5-9B-Q3_K_M     | 87.7% |  83.0% |    93.3% |     70.1% |      63.4% |
| Qwen3.5-9B-UD-Q3_K_XL | 86.7% |  81.0% |    93.9% |     67.1% |      59.1% |
| Qwen3.5-9B-IQ4_XS     | 84.9% |  83.0% |    94.2% |     67.1% |      57.3% |
| Qwen3.5-9B-Q4_K_S     | 85.9% |  84.7% |    94.2% |     63.4% |      57.3% |
| Qwen3.5-9B-IQ4_NL     | 86.7% |  84.3% |    94.2% |     65.9% |      56.1% |
| Qwen3.5-9B-Q4_0       | 83.5% |  83.4% |    93.8% |     64.0% |      56.7% |
| Qwen3.5-9B-Q4_1       | 85.4% |  84.3% |    94.4% |     68.9% |      59.8% |
| Qwen3.5-9B-Q4_K_M     | 87.1% |  82.8% |    94.5% |     65.9% |      57.3% |
| Qwen3.5-9B-UD-Q4_K_XL | 84.5% |  81.9% |    93.9% |     67.7% |      61.6% |
| Qwen3.5-9B-Q5_K_S     | 84.7% |  83.4% |    93.9% |     67.1% |      59.8% |
| Qwen3.5-9B-Q5_K_M     | 85.1% |  83.2% |    93.6% |     66.5% |      59.8% |
| Qwen3.5-9B-UD-Q5_K_XL | 85.3% |  83.7% |    94.1% |     68.3% |      60.4% |
| Qwen3.5-9B-Q6_K       | 85.0% |  83.5% |    94.0% |     69.5% |      62.8% |
| Qwen3.5-9B-UD-Q6_K_XL | 85.7% |  83.4% |    94.0% |     70.1% |      64.0% |
| Qwen3.5-9B-Q8_0       | 84.9% |  82.4% |    94.3% |     72.6% |      65.2% |
| Qwen3.5-9B-UD-Q8_K_XL | 85.0% |  83.2% |    94.5% |     68.3% |      62.2% |


**Takeaway**: The 9B handles low-bit quantization significantly better than the 4B. `IQ2_XXS` still damages coding ability severely (HumanEval drops to 25%), but ARC holds at 86.3% — no total collapse like the 4B. `IQ2_M` is a more defensible floor (53.7% HumanEval, 90.8% ARC), though `Q2_K_XL` underperforms it despite similar size. From IQ3_XXS upward, ARC stabilizes around 93–94.5% and HumanEval settles in the 60–70% range, with `Q6_K` and `Q3_K_M` at the top (within noise of each other). Unlike the 4B, higher quants do show a slight but consistent upward trend on HumanEval+. The sweet spot is `Q4_K_M` or `Q4_1`: reliable ARC (~94.4–94.5%), solid HumanEval, and meaningfully smaller than `Q6_K`.

> Full results for all models are in the [`results/`](results/) directory as JSON and CSV.


### Gemma 4 27B A4B — Unsloth UD quants

Need to be redone with humaneval & humaneval+

> HumanEval leads to inconsistent data, I removed it waiting for another run.

|Quantisation               | GSM8K |   IfEval    |  Arc_Challenge_Chat | Human Eval |
|---------------------------|-------|-------------|---------------------|------------|
| Unsloth -it-UD-IQ2_XXS    | 88%   |   87%/93%   |       80%           |            |
| Unsloth -it-UD-Q3_K_S     | 88%   |   88%/92%   |       91%           |            |
| Unsloth -it-UD-Q3_K_M     | 88%   |   88%/92%   |       91%           |            |
| Unsloth -it-UD-Q3_K_XL    | 88%   |   88%/93%   |       92%           |            |
| Unsloth -it-UD-IQ4_NL     | 88%   |   87%/93%   |       93%           |            |
| Unsloth -it-UD-Q4_K_S     | 90%   |   88%/92%   |       93%           |            |
| Unsloth -it-UD-Q4_K_M     | 90%   |   88%/93%   |       92%           |            |
| Unsloth -it-UD-Q4_K_XL    | 90%   |   87%/92%   |       92%           |            |
| Unsloth -it-UD-Q5_K_S     | 90%   |   87%/92%   |       --            |            |
| Unsloth -it-UD-Q5_K_M     | 90%   |   89%/93%   |       92%           |            |
| Unsloth -it-UD-Q5_K_XL    | 90%   |   88%/93%   |       93%           |            |



**Takeaway**: `IQ2_XXS` takes a noticeable hit on ARC (-13 points), but GSM8K and IFEval remain surprisingly stable across quants. The sweet spot is around `Q4_K_S` for this model.
 
No full results for this one for now. I plan to redo it fully. I hope to find a solution for HumanEval.



## Repository structure

```
├── README.md              # You are here
├── METHODOLOGY.md         # Benchmark descriptions, setup, evaluation details
├── REPRODUCE.md           # Step-by-step guide to replicate the tests
├── CAVEATS.md             # Pitfalls encountered and lessons learned
├── results/
│   ├── all_results.csv        # Global CSV — rebuilt by aggregate.py
│   ├── all_results.json       # Same data as JSON — rebuilt by aggregate.py
│   ├── gemma-4-26B-A4B-it/
│   │   └── summary.json       # Per-model source of truth (lm-eval)
│   ├── Qwen3.6-35B-A3B/
│   │   └── summary.json
│   ├── Qwen3.6-36B-A3B-MTP/
│   │   └── summary.json       # Per-model source of truth (inspect_ai)
│   ├── inspect_evals/
│   │   └── {model}/           # Raw .eval files from inspect_ai runs
│   └── .../
├── scripts/
│   ├── extract_summary.py         # lm-eval → results/{model}/summary.json
│   ├── extract_inspect_summary.py # inspect_ai .eval → results/{model}/summary.json
│   ├── aggregate.py               # Merges all summary.json → all_results.{json,csv}
│   ├── bench2md.py                # Markdown table generator
│   ├── run.sh                     # Evaluation launcher called by autotest.sh
│   ├── autotest.sh                # Model iteration wrapper
│   └── list_quants.py             # Get list of quants from HF, called by autotest.sh
└── LICENSE
```

## Scripts workflow

Each extraction script only writes its own `results/{model}/summary.json`. The global files are always rebuilt from scratch by `aggregate.py`, so adding a new model never loses older results.

```bash
# After an lm-evaluation-harness run:
python scripts/extract_summary.py /path/to/bench/outputs

# After an inspect_ai run:
site/menv/bin/python scripts/extract_inspect_summary.py

# Always finish with:
python scripts/aggregate.py
```

## Hardware

All benchmarks were run on a single consumer GPU setup:

- **GPU:** Consumer Nvidia (RTX-3060 12 GiB and GTX 1070-8GiB for smallest models)
- **Inference engine:** llama.cpp (llama-server)
- **Evaluation:** lm-evaluation-harness 
- **KV cache:** q4_0 for both K and V
- **Context:** 16384–65536 tokens depending on model

## What's coming

All benchmarks will be rerun — notably using MTP (Multi-Token Prediction) variants where available, as they tend to perform better on coding and reasoning tasks. GSM8K is dropped entirely — too saturated to be useful.

**Benchmark changes:**

- **Logic / comprehension:** ARC-Challenge will be replaced by [BBEH Mini](https://github.com/google-deepmind/bbeh) or the full BBEH suite — designed to resist saturation, with harder distractors and broader reasoning coverage.
- **Code generation:** HumanEval and HumanEval+ are replaced by [BigCodeBench](https://bigcode-bench.github.io/), which covers a broader set of programming tasks and is harder to game.
- **Code understanding:** [SWE-Bench](https://www.swebench.com/) is added — tests the ability to understand an existing codebase and patch real GitHub issues, complementing BigCodeBench's generation focus.
- **Knowledge & analysis:** [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) is added — harder than standard MMLU (10-choice format, expert-level questions), useful as a proxy for complex content comprehension and analysis tasks.
- **Instruction following:** IFEval stays — no better alternative available for this capability.
- **Tool use:** Tau2-Telecom is currently under evaluation as a tool-use benchmark in a telecom context (multi-step agentic tasks with function calls), but not yet selected for the suite.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the data with attribution.

## Author

**Yves Rougy** — [rougy.net](https://rougy.net) · [GitHub](https://github.com/yrougy) · [LinkedIn](https://www.linkedin.com/in/yrougy/)

