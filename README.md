<div align="center">
  <h1>qwenputer-handover</h1>
  <p><b>Fine-tunes Qwen3.5-4B to hand its arithmetic to Qwenputer's in-weights RISC-V calculator</b></p>
</div>

<div align="center">

[![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square)](./CHANGELOG.md) [![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square)](./pyproject.toml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](./LICENSE)

</div>

---

> [!NOTE]
> **A parked research experiment.** This repo is a weekend-scale exploration. It stopped after a partial headroom scan, before any real training run or merge check, so it makes no claim that the fine-tune works or helps. The numbers below come from one seed on samples of at most 100 problems.

[Qwenputer](https://github.com/divagr18/qwenputer) compiles an RV32 CPU running a calc firmware into Qwen3.5-4B's weights: when the model closes a `<tool_call>`, the circuit evaluates the expression and prints the exact answer as ordinary tokens. Upstream lists Qwen not handing the calculation over as its main risk, with a LoRA as the planned fix. This repo is a pipeline for that LoRA. It never trains the merged model. Rollouts and training use the stock checkpoint, with upstream's reference machine standing in for the circuit; that machine's responses are byte-identical to the circuit's. The adapter is then folded back into a base-format checkpoint for upstream's merge.

## Results so far

The headroom scan asks whether a fine-tune has anything to fix. It runs base Qwen3.5-4B with the reference machine as its tool, greedy with thinking off, with a budget of 4,096 tokens of Qwen's own writing per problem. Each set runs twice, with the calc tool offered both times: under upstream's "use calc for every step" system message (`nudge`) and with no system message (`bare`). The scan was stopped partway; the table covers the 596 of 800 planned rollouts written by then:

| Set | Prompt | n | Exact | Hand-over | From machine | Budget hit | Calls per rollout | Calls repeated |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| `arith` | bare | 100 | 98.0% | 97.0% | 95.0% | 2.0% | 1.33 | 25.6% |
| `arith` | nudge | 100 | 92.0% | 100.0% | 92.0% | 8.0% | 4.91 | 79.4% |
| `dmx_div_big` | bare | 85 | 96.5% | 91.8% | 88.2% | 0.0% | 0.99 | 1.2% |
| `dmx_div_big` | nudge | 87 | 97.7% | 100.0% | 97.7% | 0.0% | 1.13 | 2.0% |
| `gsmhard` | bare | 100 | 52.0% | 47.0% | 36.0% | 19.0% | 0.90 | 2.2% |
| `gsmhard` | nudge | 100 | 48.0% | 96.0% | 74.0% | 26.0% | 2.46 | 12.6% |
| `dmx_mul_div_multiple_longer` | bare | 11 | 90.9% | 81.8% | 54.5% | 0.0% | 1.64 | 0.0% |
| `dmx_mul_div_multiple_longer` | nudge | 13 | 100.0% | 100.0% | 100.0% | 0.0% | 3.23 | 0.0% |

Hand-over means at least one calc call. From machine means the final answer is a value the machine printed. Calls repeated is the share of calls that re-send an expression the machine already answered.

- **Hand-over mostly happens already.** With no prompt, Qwen calls the machine on 97% of `arith` and 92% of `dmx_div_big` problems.
- **The headroom is in two places.** On GSM-Hard word problems Qwen hands over only 47% of the time unprompted, and 19–26% of runs hit the budget under either prompt. Under the nudge, re-check loops take over: 79% of `arith` calls repeat an earlier one, and exactness falls 6 points below the bare prompt.
- **Sampling shows the same loops.** On 64 generated `synth` problems, 2 samples each at temperature 0.7, 54–63% of calls were repeats, and 21% of bare runs and 13% of nudged runs hit the budget (80.5% and 88.3% exact).

Not done: training on real data, the check that a fine-tuned base still merges cleanly, and any comparison with upstream's published evals. `scan → select → train → export` has run end to end only as a plumbing check on 4–8 problems.

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

| Step | Needs |
|:---|:---|
| `make check` | CPU only; no model |
| `scan`, `train` | A CUDA GPU; both refuse to start without one |
| `export`, `make merge` | ~31 GB RAM, ~30 GB disk; CUDA for the merge's anti-sink fit |
| `make textcheck`, `make eval-rv1` | CUDA, ≥ 24 GB (the merged model is ~19 GB bf16) |

The plumbing check, at a 512-token budget, peaked at 14.2 GB on a 16 GB RTX 4070 Ti SUPER. The headroom scan ran at batch 64 on a 96 GB RTX PRO 6000 (Colab's G4), where one decoder at the full 4,096-token budget takes about 35 GB. Under Windows display drivers (WSL included), a job that outgrows the card spills into shared system memory and slows to a crawl instead of failing with OOM, so check the first batch's memory there.

Linux installs take PyTorch's CUDA 13.0 wheel (see `pyproject.toml`). On Linux, `uv sync --extra fla` adds [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), whose Triton kernels replace transformers' torch fallback for the Gated DeltaNet layers; training needs it to run at a usable speed.

## Getting started

```bash
git clone https://github.com/cyanheads/qwenputer-handover.git && cd qwenputer-handover
uv sync                      # pulls chipformer from Qwenputer at the pinned commit
make check
```

`make check` builds the changelog with [Bun](https://bun.sh); without Bun, run `make lint typecheck test`.

Problem sets: `arith`, `gsm8k`, `gsmhard` (evaluation) and `synth`, `gsm8k_train` (training) need nothing local. Upstream's DeepMind sets (`dmx_*`, `dmi_*`, `dmxc_*`) read `~/data/dm/mathematics_dataset-v1.0/`, unpacked from [mathematics_dataset-v1.0.tar.gz](https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz). The NUPA sets read `~/data/nupa/test.json`.

## Running

The recipe the pipeline was built for. Step 0 produced the table above; steps 1 and 2 have only run as the plumbing check.

```bash
# 0. headroom: greedy, both prompts, upstream's eval sets
uv run handover scan --data arith,gsmhard,dmx_div_big,dmx_mul_div_multiple_longer --n 100 --batch 64 --out runs/headroom
uv run handover metrics runs/headroom/*.jsonl

# 1. training rollouts: sampled, both prompts, 4 per problem, loops cut at their first repeat
uv run handover scan --data synth,gsm8k_train --n 2000 --samples 4 --temperature 0.7 --stop-at-repeat --out runs/train
uv run handover select runs/train/*.jsonl --out runs/sft.jsonl
uv run handover train --sft runs/sft.jsonl --out runs/adapter

# 2. on a 24 GB+ GPU: export, upstream's merge, its checks
uv run handover export --adapter runs/adapter --out runs/ft-base
make merge SRC=runs/ft-base MERGED=runs/ft-rv1
make textcheck SRC=runs/ft-base MERGED=runs/ft-rv1 OUT=runs/checks
make eval-rv1 MERGED=runs/ft-rv1 DATA=dmx_div_big N=100 OUT=runs/eval
make replay MERGED=runs/ft-rv1 DATA=dmx_div_big OUT=runs/eval
```

## Design decisions

- **Train on the base model, merge after.** The circuit is exact and deterministic, and upstream's `BridgeHost` reproduces its output byte for byte. That makes base-model rollouts the merged model's conversations, at base-model speed. The open question is whether a fine-tuned base still merges cleanly. `textcheck` (KL against its own base) and `replay` are meant to answer it before any long training run.
- **Sample with the nudge, train without it.** Rollouts sampled under the nudge are trained under the `bare` prompt (context distillation). The goal is a model that hands arithmetic over unprompted.
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
| `src/qwenputer_handover/rollout/` | Prompt variants, the per-rollout `Forcer`, in-graph `Sampling`, the `BatchDecoder` driver, rollout records |
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
