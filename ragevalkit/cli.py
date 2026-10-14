"""rag-eval command line.

  rag-eval retrieval --system pkg.module:factory --cases golden.yaml [--k 5] [--baseline b.json] [--update-baseline]
  rag-eval answers   --system pkg.module:factory --cases golden.yaml --out runs/baseline [--mode app|oracle|null|copy]
  rag-eval summary   --out runs/baseline
  rag-eval regrade   --out runs/baseline --cases golden.yaml

`factory` is any callable returning an object that implements ragevalkit.system.RAGSystem.
Exit code 1 on a retrieval regression or leak (use it as a CI gate).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from ragevalkit import answers, retrieval
from ragevalkit.dataset import load_cases


def load_system(spec: str):
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--system must look like package.module:factory")
    return getattr(importlib.import_module(module), attr)()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieval", help="deterministic retrieval metrics (no API key)")
    r.add_argument("--system", required=True)
    r.add_argument("--cases", required=True, type=Path)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--baseline", type=Path)
    r.add_argument("--update-baseline", action="store_true")
    r.add_argument("--json", type=Path)

    a = sub.add_parser("answers", help="answer evaluation through your real generate path (calls an LLM)")
    a.add_argument("--system", required=True)
    a.add_argument("--cases", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--mode", choices=["app", "oracle", "null", "copy"], default="app")
    a.add_argument("--reps", type=int, default=1)
    a.add_argument("--ids", help="comma-separated case ids (pilot)")
    a.add_argument("--frameworks", default="deepeval,ragas", help="comma list; empty = judge-only metrics")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--judge-reps", type=int, default=1, help="average each judge metric over N passes")

    s = sub.add_parser("summary", help="aggregate a run directory")
    s.add_argument("--out", required=True, type=Path)

    g = sub.add_parser("regrade", help="recompute programmatic metrics from stored traces (free)")
    g.add_argument("--out", required=True, type=Path)
    g.add_argument("--cases", required=True, type=Path)

    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path.cwd()))
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if args.cmd == "retrieval":
        cases = load_cases(args.cases)
        summary, scores = retrieval.evaluate(load_system(args.system), cases, k=args.k)
        print(retrieval.report(summary, scores, args.k))
        if args.json:
            retrieval.save(args.json, summary, scores)
        if args.baseline and args.update_baseline:
            args.baseline.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(f"\nbaseline written: {args.baseline}")
            return 0
        problems = retrieval.compare(summary, json.loads(args.baseline.read_text(encoding="utf-8"))) \
            if args.baseline and args.baseline.exists() else (["leaks > 0"] if summary["leaks"] else [])
        for p in problems:
            print(f"REGRESSION: {p}", file=sys.stderr)
        return 1 if problems else 0

    if args.cmd == "answers":
        cases = load_cases(args.cases)
        if args.ids:
            wanted = set(args.ids.split(","))
            cases = [c for c in cases if c.id in wanted]
        frameworks = {f for f in args.frameworks.split(",") if f}
        answers.Runner(load_system(args.system), args.out, args.mode, args.reps,
                       frameworks=frameworks, judge_reps=args.judge_reps).run(cases, args.workers)
        print(json.dumps({**answers.summarize(args.out), "agreement": answers.agreement(args.out)}, indent=2))
        return 0

    if args.cmd == "summary":
        print(json.dumps({**answers.summarize(args.out), "agreement": answers.agreement(args.out)}, indent=2))
        return 0

    if args.cmd == "regrade":
        print(f"{answers.regrade(args.out, load_cases(args.cases))} rows re-graded")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
