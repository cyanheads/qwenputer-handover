"""src/qwenputer_handover/cli.py — the ``handover`` command.

    handover scan    sample base Qwen3.5 with the reference machine answering calc calls
    handover metrics summarize rollout files (hand-over, exactness, loops, budget hits)
    handover select  rejection-sample train rollouts into an SFT set
    handover train   LoRA SFT on an SFT set
    handover export  fold an adapter into a base-format checkpoint for upstream's merge

Each subcommand imports its heavy modules when it runs, so ``metrics`` and ``select`` start
without loading torch.
"""

import argparse
from pathlib import Path

from qwenputer_handover.rollout.variants import VARIANTS, parse_variants

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
FIRMWARE = Path(__file__).resolve().parents[2] / "firmware" / "calc-bridge.elf"


def _scan(a: argparse.Namespace) -> None:
    from qwenputer_handover.problems.sources import load
    from qwenputer_handover.rollout.sampler import Decoding, Sampler, scan
    from qwenputer_handover.runtime import resolve_model

    problems = [p for name in a.data.split(",") for p in load(name, a.n, a.seed)]
    sampler = Sampler(resolve_model(a.model), a.firmware)
    scan(
        sampler,
        problems,
        parse_variants(a.variants),
        a.samples,
        Decoding(a.temperature, a.top_p, a.top_k, a.budget, a.stop_at_repeat),
        a.out,
        a.batch,
        a.seed,
    )


def _metrics(a: argparse.Namespace) -> None:
    from qwenputer_handover.curation.metrics import summarize, table
    from qwenputer_handover.rollout.records import read

    print(table(summarize(r for f in a.rollouts for r in read(f))))


def _select(a: argparse.Namespace) -> None:
    from qwenputer_handover.curation.selection import select
    from qwenputer_handover.rollout.records import append, read

    if a.out.exists():
        raise SystemExit(f"{a.out} exists: select writes a fresh file")
    chosen = select((r for f in a.rollouts for r in read(f)), a.per_problem)
    append(a.out, chosen)
    print(f"{len(chosen)} rollouts over {len({r.problem_id for r in chosen})} problems -> {a.out}")


def _train(a: argparse.Namespace) -> None:
    from qwenputer_handover.rollout.records import read
    from qwenputer_handover.runtime import resolve_model
    from qwenputer_handover.training.lora import TrainConfig, train

    cfg = TrainConfig(
        rank=a.rank,
        alpha=a.alpha,
        lr=a.lr,
        epochs=a.epochs,
        grad_accum=a.grad_accum,
        max_len=a.max_len,
        batch_tokens=a.batch_tokens,
        seed=a.seed,
    )
    train(resolve_model(a.model), list(read(a.sft)), parse_variants(a.variants), a.out, cfg)


def _export(a: argparse.Namespace) -> None:
    from qwenputer_handover.runtime import resolve_model
    from qwenputer_handover.training.export import export

    keys = export(resolve_model(a.base), a.adapter, a.out)
    print(f"{len(keys)} weights merged -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="handover", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(required=True)

    s = sub.add_parser("scan", help="sample rollouts with the reference machine as the tool")
    s.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local checkpoint dir")
    s.add_argument(
        "--data",
        required=True,
        help="comma-separated sets: synth, gsm8k_train (train); arith, gsm8k, "
        "gsmhard, dmx_*, dmi_*, dmxc_*, nupa_*, bbh_mat2, eleu_* (eval)",
    )
    s.add_argument("--n", type=int, help="problems per set (default: the set's loader)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--variants", default=",".join(VARIANTS))
    s.add_argument("--samples", type=int, default=1, help="rollouts per problem and variant")
    s.add_argument("--temperature", type=float, default=0.0, help="0: greedy (upstream's evals)")
    s.add_argument("--top-p", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=20)
    s.add_argument("--budget", type=int, default=4096, help="Qwen-written tokens per rollout")
    s.add_argument(
        "--stop-at-repeat",
        action="store_true",
        help="end a rollout at its first calc call repeating an earlier expression (selection "
        "rejects it anyway): saves decode on training scans, skews an eval scan's metrics",
    )
    s.add_argument("--batch", type=int, default=16, help="rollouts decoded together")
    s.add_argument("--firmware", type=Path, default=FIRMWARE)
    s.add_argument("--out", type=Path, required=True, help="directory of <set>.<variant>.jsonl")
    s.set_defaults(run=_scan)

    m = sub.add_parser("metrics", help="summarize rollout files")
    m.add_argument("rollouts", type=Path, nargs="+")
    m.set_defaults(run=_metrics)

    c = sub.add_parser("select", help="rejection-sample train rollouts into an SFT set")
    c.add_argument("rollouts", type=Path, nargs="+")
    c.add_argument("--per-problem", type=int, default=1)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(run=_select)

    t = sub.add_parser("train", help="LoRA SFT on an SFT set")
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--sft", type=Path, required=True)
    t.add_argument(
        "--variants",
        default="bare",
        help="comma-separated prompts each generation is trained under (once per prompt)",
    )
    t.add_argument("--out", type=Path, required=True, help="adapter directory (must not exist)")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--grad-accum", type=int, default=16)
    t.add_argument("--max-len", type=int, default=4096)
    t.add_argument("--batch-tokens", type=int, default=16384, help="padded tokens per forward pass")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(run=_train)

    e = sub.add_parser("export", help="fold an adapter into a base-format checkpoint")
    e.add_argument("--base", default=DEFAULT_MODEL)
    e.add_argument("--adapter", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(run=_export)

    args = ap.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
