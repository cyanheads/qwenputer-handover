# qwenputer-handover

Python (uv) package that teaches Qwen3.5-4B to hand arithmetic to the calculator Qwenputer compiles into its weights. It samples the stock model with upstream's reference machine as the tool (`handover scan`), rejection-samples an SFT set (`select`), trains a LoRA (`train`), and folds it into a base-format checkpoint (`export`) for upstream's `python -m chipformer.rv.merge`.

**Status:** a parked research experiment. A partial headroom scan ran, and no real training run or merge check has. README § Results so far has the numbers and what was never run.

**Orientation:** this file is the behavioral layer; `README.md` has the command table, the hardware each step needs, the run recipe and the design decisions.

## Stack & gate

- Python 3.12, uv, hatchling src layout. torch, transformers 5, peft, `chipformer[hf]` (Qwenputer's `experimental/riscv` package, a uv git source pinned to one commit in `pyproject.toml`).
- **Gate: `make check`** (ruff check + format, mypy strict, pytest, changelog drift). It needs no model weights and no GPU (the trainer tests build a tiny random Qwen3.5 on the CPU), and is the one command that must be green. The changelog step needs `bun`; on a machine without it, run `make lint typecheck test`.
- `PYTHONOPTIMIZE` strips `assert`, which the tests rely on. The Makefile unsets it; outside the Makefile use `env -u PYTHONOPTIMIZE uv run …` if your shell sets it.

## Running it

| Mode | Command | Notes |
|:---|:---|:---|
| Gate | `make check` | Anywhere |
| Rollouts | `uv run handover scan --data … --out runs/<name>` | CUDA only; resumable, appends each rollout with fsync |
| Summary | `uv run handover metrics runs/<name>/*.jsonl` | No torch import |
| SFT set | `uv run handover select … --out runs/sft.jsonl` | Train-split rollouts only |
| Training | `uv run handover train --sft … --out runs/adapter` | CUDA only |
| Export | `uv run handover export --adapter … --out runs/ft-base` | CPU; output dir must not exist |
| Upstream | `make merge` / `textcheck` / `eval-rv1` / `eval-pytool` / `replay` | Variables `SRC MERGED OUT DATA N`; see `Makefile` |

## Architecture

Problems (`problems/`) → prompt ids for a variant (`rollout/prompts.py`) → upstream's `BatchDecoder` (continuous batching, CUDA-graph steps), with `Sampling` (`rollout/sampling.py`) replacing its argmax and a `Forcer` per rollout (`rollout/forcing.py`) in its token hook, each rollout's `BridgeHost` forcing the machine's response tokens → `Rollout` JSONL (`rollout/records.py`) → `curation/metrics.py` / `curation/selection.py` → `training/sft_data.py` masks → `training/lora.py` → `training/export.py`.

| Path | Role |
|:---|:---|
| `src/qwenputer_handover/cli.py` | `handover` entry; subcommands import heavy modules lazily |
| `src/qwenputer_handover/problems/sources.py` | Every set by name; eval sets come from upstream's loaders |
| `src/qwenputer_handover/rollout/forcing.py` | `Forcer`: one rollout's machine forcing, Qwen-token budget and optional `RepeatCut`, driven by the decoder hook |
| `src/qwenputer_handover/rollout/sampling.py` | In-graph temperature / top-k / top-p sampling for `BatchDecoder` (Gumbel-max, hashed noise) |
| `src/qwenputer_handover/rollout/sampler.py` | Model load, the `BatchDecoder` run, `scan` driver |
| `src/qwenputer_handover/training/lora.py` | LoRA SFT: length-grouped steps, right-padded micro-batches, chunked head + cross-entropy at labeled positions |
| `src/qwenputer_handover/training/export.py` | Adapter → `W + scale·B·A` patched into the base's own shards |
| `firmware/calc-bridge.elf` | Upstream's calc firmware at the pinned commit |

## The rules that matter

- **CUDA only, no fallback.** `runtime.device()` raises without CUDA, by design; never add a CPU/MPS fallback. Sampling and training need a CUDA GPU; the merged model (~19 GB bf16) needs 24 GB+ for `make textcheck` and `make eval-rv1`. Under Windows display drivers (WSL included) a job that outgrows the card spills into shared memory and crawls instead of OOMing, so read the first batch's memory.
- **Cloud GPUs bill until stopped.** On Colab (`google-colab-cli`), `colab usage` shows the compute-unit balance and burn rate, and every `colab new` session burns units until `colab stop`. `runs/` is not synced anywhere: copy outputs off a VM before stopping it.
- **flash-linear-attention is the `fla` extra (Linux only).** `uv sync --extra fla` installs it; Gated DeltaNet then runs on its Triton kernels instead of transformers' torch fallback, which matters most for training.
- **The train/eval split is load-bearing.** Training sets are `synth` and `gsm8k_train` only. `select` raises on any `split == "eval"` rollout; never weaken that check.
- **Forced tokens are never labeled and never fed back to the host.** `Forcer` and `build_example` both depend on `Rollout.forced` spans being exact. Change one and you must change the other.
- **`--stop-at-repeat` is for training scans only.** It ends a rollout at its first repeated calc call, which `select` would reject anyway; an eval scan run with it under-reports budget hits and repeated calls. `RepeatCut` reads call bodies exactly as `records.parse_calls` does (`records.call_expression`); keep them in step.
- **Training batches must stay right-padded.** `training/lora.py` relies on right padding being exact for a causal model (a real token never sees a pad); `tests/test_lora.py` checks the padded batch's loss and gradients against the plain per-sequence path on a tiny random Qwen3.5.
- **`Sampling` reaches into `BatchDecoder` internals** (`_argmax`, `mark_prompt`, `levels`, `sizes`, `banned`) of the pinned chipformer commit. Re-check them when you bump the `rev`.
- **The export must stay base-format.** Keys, shard names, index and metadata are the base's; only adapted `model.language_model.*` weights change. Upstream's merge reads it as a stock Qwen3.5 checkpoint.
- **`chipformer` and the firmware move together.** When you bump the `rev` in `pyproject.toml`, re-copy `tests/data/rv/calc-bridge.elf` from the same upstream commit into `firmware/`.
- **No `type: ignore` anywhere.** chipformer, datasets and peft are untyped; type their call sites instead (see the mypy overrides in `pyproject.toml`).

## Where things live

- `runs/`: all run outputs (rollouts, SFT sets, adapters, exports, merged checkpoints). Gitignored.
- `changelog/<series>/<version>.md`: per-release notes; `CHANGELOG.md` is generated by `make changelog`, never hand-edited.
- `~/data/dm/…`, `~/data/nupa/…`: upstream's DeepMind and NUPA files (README § Getting started).
- `.env.example`: the library env vars worth setting (`HF_HOME`, `CUDA_VISIBLE_DEVICES`).

## Triggers

| When the ask is | Do this |
|:---|:---|
| "headroom scan", "how often does it hand over" | `handover scan` greedy on eval sets with both variants, then `handover metrics` |
| "build the SFT set", "rollouts for training" | `scan --data synth,gsm8k_train --samples K --temperature 0.7 --stop-at-repeat`, then `select` |
| "train it", "fine-tune" | `handover train` on a CUDA GPU |
| "does it still merge", "merge check" | On a 24 GB+ GPU with `runs/adapter`: `export` → `make merge` → `make textcheck` → `make eval-rv1` + `make replay` |
| "bump upstream" | New `rev` in `pyproject.toml` + firmware re-vendor + `uv lock` + `make check` |

## Related repos & links

- Upstream: https://github.com/divagr18/qwenputer (`experimental/riscv`); its `docs/rv32.md` and `rv32-p3-status.md` describe the bridge and the merge.
- Released merged model: `divagr1925/Qwen3.5-4B-Qwenputer` (rv1 layout).

## Commit stance

Conventional commits, `make check` green first. No releases or tags unless asked.
