# Qwen3.8-27B — xhigh reasoning effort (not on the site)

The raw `.eval` transcripts in this folder are kept for historical reference
only. They are **not** used to build the site — the corresponding entry in
`site/models.yaml` is commented out, and `results/Qwen3.8-27B-xhigh/summary.json`
is not linked from any published chart.

## Why

This run was intended to use the project's usual thinking-off / low-effort
setup, matching every other model in this suite. Instead it ran with the
model's default `reasoning_effort: xhigh` the whole time — the `enable_thinking`
flag set in `bench_config.json` is not honored by Qwen3.8's chat template
(it uses a graded `reasoning_effort` parameter instead of the older binary
switch, and is known to ignore `enable_thinking` in llama.cpp for this model
family). See [CAVEATS.md](../../../CAVEATS.md) for the full writeup and the
evidence found in these transcripts.

The practical effect: BFCL scores are depressed and noisy relative to a
normal run (the model hedges/asks clarifying questions instead of executing
multi-turn tool calls — `multi_turn_composite_acc` sits near 0 across every
quant), while MUSR looks *better* than it should (extra reasoning effort
genuinely helps free-form narrative reasoning). BigCodeBench is roughly
unaffected. None of this is a fair reflection of quantization behavior at
the settings this project normally tests, and none of it should be compared
to other models or to a future medium-effort rerun of Qwen3.8-27B.

Two quants (`UD-Q4_K_XL`, `UD-Q6_K_XL`) OOM'd during this run and produced no
usable data; their entries were removed from `summary.json` entirely.

## What this is still good for

- A worked example of how strongly `reasoning_effort` can move BFCL vs. MUSR
  in opposite directions — see the CAVEATS.md entry for the numbers.
- A baseline to diff against once the medium-effort rerun lands, if anyone
  wants to quantify the reasoning-effort effect directly rather than just
  noting that it exists.

The medium-effort rerun will use `results/Qwen3.8-27B/` (unsuffixed) once it
starts — see `scripts/import_inspect_uploads.py`, which routes new uploads
by re-parsing the GGUF filename from each `.eval` header, independently of
this folder's name.
