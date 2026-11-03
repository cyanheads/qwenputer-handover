<div align="center">
  <h1>qwenputer-handover</h1>
  <p><b>Fine-tunes Qwen3.5-4B to hand its arithmetic to Qwenputer's in-weights RISC-V calculator</b></p>
</div>

<div align="center">

[![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square)](./CHANGELOG.md) [![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square)](./pyproject.toml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](./LICENSE)

</div>

---

> [!NOTE]
> **A research experiment.** One LoRA run on one seed, evaluated on 100 problems per set. The numbers describe this run, not a benchmark.

[Qwenputer](https://github.com/divagr18/qwenputer) compiles an RV32 CPU running a calc firmware into Qwen3.5-4B's weights: when the model closes a `<tool_call>`, the circuit evaluates the expression and prints the exact answer as ordinary tokens. Upstream lists Qwen not handing the calculation over as its main risk, with a LoRA as the planned fix. This repo builds that LoRA. It never trains the merged model. Rollouts and training use the stock checkpoint, with upstream's reference machine standing in for the circuit; that machine's responses are byte-identical to the circuit's. The adapter is then folded back into a base-format checkpoint for upstream's merge.

## Results

### Stock model

Base Qwen3.5-4B with the reference machine as its tool, greedy with thinking off, with a budget of 4,096 tokens of Qwen's own writing per problem and n = 100 per row. Each set runs under two prompts, with the calc tool offered both times: upstream's "use calc for every step" system message (`nudge`), and no system message (`bare`).

| Set | Prompt | Exact | Hand-over | From machine | Budget hit | Calls per rollout | Calls repeated |
|:---|:---|---:|---:|---:|---:|---:|---:|
| `arith` | bare | 98% | 98% | 96% | 2% | 1.56 | 36.5% |
| `arith` | nudge | 93% | 100% | 93% | 7% | 4.37 | 76.9% |
| `dmx_div_big` | bare | 79% | 95% | 75% | 21% | 1.50 | 30.0% |
| `dmx_div_big` | nudge | 87% | 100% | 87% | 16% | 1.95 | 33.8% |
| `dmx_mul_div_multiple_longer` | bare | 96% | 64% | 47% | 1% | 1.62 | 0.0% |
| `dmx_mul_div_multiple_longer` | nudge | 95% | 100% | 96% | 5% | 5.14 | 1.9% |
| `gsm8k` | bare | 95% | 20% | 18% | 1% | 0.30 | 0.0% |
| `gsm8k` | nudge | 93% | 96% | 89% | 4% | 4.20 | 42.1% |
| `gsmhard` | bare | 49% | 47% | 39% | 23% | 0.73 | 1.4% |
| `gsmhard` | nudge | 48% | 98% | 80% | 22% | 3.22 | 36.6% |

Hand-over means at least one calc call. From machine means the final answer is a value the machine printed. Calls repeated is the share of calls that re-send an expression the machine already answered.

- **Unprompted hand-over is uneven.** With no system message, Qwen calls the machine on 95–98% of `arith` and `dmx_div_big` problems, 64% of `dmx_mul_div_multiple_longer` and 20% of `gsm8k`.
- **Runs out of budget on long problems.** 16–21% of `dmx_div_big` runs and 22–23% of `gsmhard` runs hit the 4,096-token budget under either prompt.
- **The nudge brings re-check loops.** Under it, 77% of `arith` calls repeat an earlier call (37% on `gsmhard`, 42% on `gsm8k`), and `arith` exactness is 5 points below the bare prompt.

### Fine-tuned

The fine-tune is training now with the recipe under [Running](#running). Its scores on the same 1,000 problems go here when its evals finish.

Training data: 12,000 sampled rollouts (temperature 0.7, top-p 0.8, top-k 20, budget 2,048) on 1,500 `synth` and 1,500 `gsm8k_train` problems, 2 samples per problem under each prompt, with re-check loops cut at their first repeat (1,198 cut). Selection kept 5,109 rollouts over 2,715 problems, up to 2 per problem, shortest first. Each is trained under both prompts: 10,218 examples, 3.04M labeled tokens, 2 epochs of 1,278 steps (LoRA rank 16, α 32, lr 1e-4).

### Merging a fine-tuned base

Before the full run, a small adapter (1 epoch on 72 rollouts) went through the whole path: export, upstream's merge, and its checks on the merged model.

| Check | Result |
|:---|:---|
| `textcheck`: KL(base ‖ merged) on upstream's 16 prompts | 5.40e-4, noise floor 5.35e-4 (upstream's stock merge: 6.1e-4 vs 5.1e-4) |
| `eval-rv1` on `arith`, n = 8: the merged model answering on its own, checked against the reference machine at every step | 8/8 exact, no lockstep errors |
| `replay` of those 8 records | 0 wrong predictions over 19,705 circuit-written positions |

The full adapter gets the same checks.

## Commands

| Command | What it does |
|:---|:---|
| `handover scan` | Samples base Qwen3.5 on a problem set; the reference machine answers each calc call, forced token by token. Writes `<set>.<variant>.jsonl` and resumes from what is already on disk. `--stop-at-repeat` ends a training rollout at its first repeated call |
| `handover metrics` | Prints exactness, hand-over rate, answers taken from the machine, budget hits, repeat cuts, error and repeated calls, per set and variant |
| `handover select` | Rejection-samples train rollouts into an SFT set |
| `handover train` | LoRA SFT on an SFT set in length-grouped, padded micro-batches; the loss covers only the tokens Qwen wrote |
| `handover export` | Folds an adapter into a copy of the base checkpoint (same keys, shards, index) |
| `make merge` · `textcheck` · `eval-rv1` · `eval-pytool` · `replay` | Upstream's merge, text-fidelity check, evals and IR replay on an exported checkpoint |

## Hardware

| Step | Needs | Measured on Colab's G4 (RTX PRO 6000, 96 GB) |
|:---|:---|:---|
| `make check` | CPU only; no model | |
| `scan` | A CUDA GPU | Training scan (batch 192, budget 2,048, `--stop-at-repeat`): 273 rollouts/min, 56 GB. Eval scan (batch 64, budget 4,096) while sharing the GPU: 52 rollouts/min |
| `train` | A CUDA GPU | 3.7–3.9K tokens/s at 32K-token micro-batches, peak 26–29 GB |
| `export`, `make merge` | ~31 GB RAM, ~30 GB disk; CUDA for the merge's anti-sink fit | Merged checkpoint: 18 GB |
| `make textcheck`, `make eval-rv1` | CUDA, ≥ 24 GB (the merged model is ~19 GB bf16) | `eval-rv1` with `BATCH=8`: 8 problems in 83 s |

`scan` and `train` refuse to start without CUDA. Under Windows display drivers (WSL included), a job that outgrows the card spills into shared system memory and slows to a crawl instead of failing with OOM, so check the first batch's memory there.

Linux installs take PyTorch's CUDA 13.0 wheel (see `pyproject.toml`). On Linux, `uv sync --extra fla` adds [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), whose Triton kernels replace transformers' torch fallback for the Gated DeltaNet layers; training needs it to run at a usable speed.

## Getting started

```bash
git clone https://github.com/cyanheads/qwenputer-handover.git && cd qwenputer-handover
uv sync                      # pulls chipformer from Qwenputer at the pinned commit
make check
```

On a Linux GPU machine, use `uv sync --extra fla` instead. `make check` builds the changelog with [Bun](https://bun.sh); without Bun, run `make lint typecheck test`.

Problem sets: `arith`, `gsm8k`, `gsmhard` (evaluation) and `synth`, `gsm8k_train` (training) need nothing local. Upstream's DeepMind sets (`dmx_*`, `dmi_*`, `dmxc_*`) read `~/data/dm/mathematics_dataset-v1.0/`, unpacked from [mathematics_dataset-v1.0.tar.gz](https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz). The NUPA sets read `~/data/nupa/test.json`.

## Running

The commands behind the results above.

```bash
# 0. baseline: greedy, both prompts, budget 4,096
uv run handover scan --data arith,gsmhard,dmx_div_big,dmx_mul_div_multiple_longer,gsm8k --n 100 --batch 64 --out runs/eval-base
uv run handover metrics runs/eval-base/*.jsonl

# 1. training rollouts (2 per problem under each prompt, loops cut at their first repeat), selection, LoRA SFT
uv run handover scan --data synth,gsm8k_train --n 1500 --samples 2 --temperature 0.7 --budget 2048 --stop-at-repeat --batch 192 --seed 1 --out runs/train
uv run handover select runs/train/*.jsonl --per-problem 2 --out runs/sft.jsonl
uv run handover train --sft runs/sft.jsonl --variants bare,nudge --batch-tokens 32768 --out runs/adapter

# 2. the fine-tuned base on the baseline's problems; then upstream's merge and its checks
uv run handover export --adapter runs/adapter --out runs/ft-base
uv run handover scan --model runs/ft-base --data arith,gsmhard,dmx_div_big,dmx_mul_div_multiple_longer,gsm8k --n 100 --batch 64 --out runs/eval-ft
make merge SRC=runs/ft-base MERGED=runs/ft-rv1
make textcheck SRC=runs/ft-base MERGED=runs/ft-rv1 OUT=runs/checks
make eval-rv1 MERGED=runs/ft-rv1 DATA=arith N=40 BATCH=8 OUT=runs/eval-rv1
make replay MERGED=runs/ft-rv1 DATA=arith OUT=runs/eval-rv1
```

## Design decisions

- **Train on the base model, merge after.** The circuit is exact and deterministic, and upstream's `BridgeHost` reproduces its output byte for byte. That makes base-model rollouts the merged model's conversations, at base-model speed. Whether a fine-tuned base still merges cleanly is checked with upstream's own tools: `textcheck` (KL against its own base), `eval-rv1` (lockstep against the reference machine) and `replay`.
- **Sample with the nudge, train without it.** Rollouts sampled under the nudge are trained under the `bare` prompt (context distillation). The goal is a model that hands arithmetic over unprompted. `train --variants bare,nudge` also trains each selected rollout under the nudge, which upstream's `eval-rv1` uses and where re-check loops dominate.
- **Loss on Qwen's tokens only.** The tool response the circuit prints is masked: the merged model never predicts those positions with Qwen's head.
- **Rejection criteria.** A rollout is kept when its answer is exact, is a value the machine printed, it ended on its own, and it never repeated a call. A refused call followed by a successful split is kept, since that recovery is worth learning. Re-verification loops are not.
- **Cut loops early in training scans.** A rollout that repeats a call can never be selected, and re-check loops are what runs most rollouts to the budget. `--stop-at-repeat` ends it at the repeated `</tool_call>`, before the machine runs, so training scans spend no decode on rollouts `select` would drop. Eval scans run without it: their metrics count the loops.
- **Batched, exact training.** Each optimizer step takes examples of similar length, run as right-padded micro-batches under a padded-token budget. Right padding is exact for a causal model (Gated DeltaNet also zeroes padded inputs), and a test checks the batched loss and gradients against the one-sequence path. The 248K-vocabulary head runs only at labeled positions, in checkpointed chunks.
- **Train and eval never mix.** Every problem carries its split, and `select` raises on an eval problem.
- **Adapters on the language model only.** Full attention, Gated DeltaNet and MLP projections get adapters; the vision tower, embeddings and tied head stay frozen. The export changes those weights and nothing else.
- **Upstream's decoder, sampled inside its CUDA graph.** Rollouts run on upstream's `BatchDecoder` (continuous batching, one CUDA graph per step), with a `Forcer` per rollout in its token hook. For sampled runs, `Sampling` replaces its argmax with a Gumbel-max draw over the top-k / top-p candidates, the noise hashed from each row's seed and position, so no host-side RNG runs inside the graph. It ran about 10× faster than HF `generate`, and greedy results matched it.

## Project structure

| Path | Purpose |
|:---|:---|
| `src/qwenputer_handover/problems/` | Problem record, `synth` generator, set loaders (upstream's for evaluation) |
| `src/qwenputer_handover/rollout/` | Prompt variants, the per-rollout `Forcer` and `RepeatCut`, in-graph `Sampling`, the `BatchDecoder` driver, rollout records |
| `src/qwenputer_handover/curation/` | Metrics and rejection sampling |
| `src/qwenputer_handover/training/` | SFT masking, the LoRA loop, the merge-ready export |
| `firmware/` | Upstream's `calc-bridge.elf` (Qwenputer `bec1a0c`, MIT; `LICENSE.qwenputer`) |
| `changelog/` | One file per release; `CHANGELOG.md` is generated |

## Development

- Gate: `make check` (ruff, mypy strict, pytest, changelog drift). Tests download nothing; the trainer's build a tiny random Qwen3.5 on the CPU.
- `chipformer` is pinned to one Qwenputer commit in `pyproject.toml`. When you bump it, re-vendor `firmware/calc-bridge.elf` from the same commit.
- Agent rules: [CLAUDE.md](./CLAUDE.md).

## License

Apache-2.0 ([LICENSE](./LICENSE)). `firmware/calc-bridge.elf` comes from [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal, under MIT ([firmware/LICENSE.qwenputer](./firmware/LICENSE.qwenputer)). The circuit, the reference machine, the merge and the evals are Qwenputer's; this repo adds the rollouts, selection, training and export around them.
