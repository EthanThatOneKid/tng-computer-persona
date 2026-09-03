"""Evaluation harness for the TNG computer agent scaffolding.

Holds out a seeded fraction of ``enterprise_computer_train.jsonl`` and scores
candidate agent configurations against the computer's *verified* responses:

- exact match (normalized)
- token F1
- ROUGE-L (LCS-based F1)
- gold-token coverage

Candidates (per backend):

- ``full (state+context)``, ``no-context``, ``no-state``, ``bare`` -- the four
  ablations of the assembled agent, so the value of each scaffold layer is
  visible.
- ``retrieval top-1`` (extractive backend only) -- answers with the single
  best-matching computer line of the episode, scoring the retrieval layer
  directly as an answerer.

Two state modes:

- default (leak-free): state is built from the train split only, so held-out
  rows can never teach the tracker their own answers.
- ``--full-state``: state is built from the whole corpus -- the realistic agent
  setup (a real ship computer would know the ship). Report the caveat when
  interpreting scores.

The retrieval probe is independent of any answerer: it reports whether the
verified answer line is among the top-k computer lines of the episode
retrieved for the query.

Usage (from the repo root):

    python skills/tng-computer/eval/evaluate.py              # offline, leak-free
    python skills/tng-computer/eval/evaluate.py --full-state  # offline, realistic
    python skills/tng-computer/eval/evaluate.py --backend openai  # real LLM
    python skills/tng-computer/eval/evaluate.py --limit 10    # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "skills" / "tng-computer" / "src"))

from agent import DEFAULT_FALLBACK, TNGComputerAgent  # noqa: E402
from retrieval import RetrievalIndex  # noqa: E402
from state import StateTracker  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "enterprise_computer_train.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------- scorers

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def exact_match(gold: str, pred: str) -> int:
    return int(_normalize(gold) == _normalize(pred))


def token_f1(gold: str, pred: str) -> float:
    g, p = set(_tokens(gold)), set(_tokens(pred))
    if not g or not p:
        return 0.0
    inter = len(g & p)
    return 2 * inter / (len(g) + len(p))


def rouge_l_f1(gold: str, pred: str) -> float:
    """LCS-based F1 over token sequences (ROUGE-L)."""
    g, p = _tokens(gold), _tokens(pred)
    if not g or not p:
        return 0.0
    prev = [0] * (len(p) + 1)
    for gt in g:
        cur = [0] * (len(p) + 1)
        for j, pt in enumerate(p, start=1):
            cur[j] = prev[j - 1] + 1 if gt == pt else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    return 2 * lcs / (len(g) + len(p))


def coverage(gold: str, pred: str) -> float:
    g, p = set(_tokens(gold)), set(_tokens(pred))
    if not g:
        return 0.0
    return len(g & p) / len(g)


# ---------------------------------------------------------------- harness

def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def holdout_split(rows: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = max(1, int(len(rows) * frac))
    holdout_idx = set(idx[:cut])
    holdout = [r for i, r in enumerate(rows) if i in holdout_idx]
    train = [r for i, r in enumerate(rows) if i not in holdout_idx]
    return train, holdout


def retrieval_probe(index: RetrievalIndex, row: dict, k: int = 5) -> bool:
    """Is the verified answer line among the top-k computer lines retrieved?"""
    gold = row["messages"][1]["content"]
    hits = index.search(
        row["messages"][0]["content"],
        k=k,
        episode=row["metadata"]["episode"],
        is_computer=True,
        exclude_ids={row["id"]},
    )
    return any(_normalize(h["text"]) == _normalize(gold) for h in hits)


def retrieval_answer(index: RetrievalIndex, row: dict) -> str:
    """Baseline answerer: the single best-matching computer line of the episode."""
    hits = index.search(
        row["messages"][0]["content"],
        k=1,
        episode=row["metadata"]["episode"],
        is_computer=True,
        exclude_ids={row["id"]},
    )
    return hits[0]["text"] if hits else DEFAULT_FALLBACK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="extractive", choices=["extractive", "openai"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-frac", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=0, help="cap holdout rows (smoke runs)")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--probe-k", type=int, default=5)
    parser.add_argument("--full-state", action="store_true",
                        help="build state from the whole corpus (realistic agent mode)")
    parser.add_argument("--memory-threshold", type=float, default=0.65,
                        help="extractive memory near-duplicate token-F1 threshold")
    args = parser.parse_args()

    rows = load_rows(TRAIN_PATH)
    train_rows, holdout = holdout_split(rows, args.holdout_frac, args.seed)
    if args.limit:
        holdout = holdout[: args.limit]
    holdout_ids = {r["id"] for r in holdout}
    print(f"backend={args.backend}  train={len(train_rows)}  holdout={len(holdout)}  "
          f"(seed {args.seed}, frac {args.holdout_frac})")

    # Shared scaffolding: build the index once; state excludes holdout rows
    # unless --full-state (realistic agent mode) is requested.
    index = RetrievalIndex()
    state = StateTracker().load(
        exclude_ids=None if args.full_state else holdout_ids
    )

    candidates: dict[str, dict] = {
        "full (state+context)": {"mode": "agent", "use_state": True, "use_context": True},
        "no-context": {"mode": "agent", "use_state": True, "use_context": False},
        "no-state": {"mode": "agent", "use_state": False, "use_context": True},
        "bare (system only)": {"mode": "agent", "use_state": False, "use_context": False},
    }
    if args.backend == "extractive":
        candidates["retrieval top-1 (computer lines)"] = {"mode": "retrieval"}

    agents = {
        name: TNGComputerAgent(
            backend=args.backend,
            holdout_ids=holdout_ids,
            state=state,
            retrieval=index,
            memory_threshold=args.memory_threshold,
        )
        for name, spec in candidates.items()
        if spec["mode"] == "agent"
    }

    # Probe retrieval independently of the answer pipeline.
    hits = sum(retrieval_probe(index, r, k=args.probe_k) for r in holdout)

    results: dict[str, list[dict]] = {name: [] for name in candidates}
    for row in holdout:
        query = row["messages"][0]["content"]
        gold = row["messages"][1]["content"]
        episode = row["metadata"]["episode"]
        scene = row["metadata"].get("scene")
        for name, spec in candidates.items():
            if spec["mode"] == "retrieval":
                pred = retrieval_answer(index, row)
                source = "retrieval"
            else:
                pred = agents[name].respond(
                    query, episode=episode, scene=scene,
                    use_state=spec["use_state"], use_context=spec["use_context"],
                )
                source = getattr(agents[name]._backend, "last_source", "fallback")
            results[name].append(
                {
                    "id": row["id"],
                    "episode": episode,
                    "query": query,
                    "gold": gold,
                    "pred": pred,
                    "exact": exact_match(gold, pred),
                    "f1": round(token_f1(gold, pred), 3),
                    "rouge_l": round(rouge_l_f1(gold, pred), 3),
                    "coverage": round(coverage(gold, pred), 3),
                    "source": source,
                }
            )

    def mean(key: str, name: str) -> float:
        vals = [r[key] for r in results[name]]
        return sum(vals) / len(vals) if vals else 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{args.backend}{'-full-state' if args.full_state else ''}"
    md_path = stem.with_suffix(".md")
    json_path = stem.with_suffix(".json")

    state_note = (
        "full-corpus state (realistic agent mode)"
        if args.full_state
        else "leak-free state (train split only)"
    )
    lines = [
        "# TNG Computer Agent Evaluation",
        "",
        f"- backend: `{args.backend}`",
        f"- corpus: {len(rows)} rows; holdout {len(holdout)} ({args.holdout_frac:.0%}, seed {args.seed})",
        f"- state: {state_note}",
        f"- retrieval probe hit@{args.probe_k} (verified answer in top-{args.probe_k} episode computer lines): "
        f"{hits}/{len(holdout)} ({hits / len(holdout):.1%})",
        "",
        "## Aggregate (mean over holdout)",
        "",
        "| candidate | exact | token F1 | ROUGE-L | coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in candidates:
        lines.append(
            f"| {name} | {mean('exact', name):.2f} | {mean('f1', name):.3f} | "
            f"{mean('rouge_l', name):.3f} | {mean('coverage', name):.3f} |"
        )

    lines += ["", "## Response sources", "", "| candidate | state | memory | retrieval | fallback |", "|---|---:|---:|---:|---:|"]
    for name in candidates:
        counts = {"state": 0, "memory": 0, "retrieval": 0, "fallback": 0}
        for r in results[name]:
            counts[r.get("source", "fallback")] += 1
        lines.append(
            f"| {name} | {counts['state']} | {counts['memory']} | {counts['retrieval']} | {counts['fallback']} |"
        )

    lines += ["", "## Per-row detail (full candidate)", "", "| id | episode | query | gold | pred | F1 |", "|---|---|---|---|---|---|"]
    for r in results["full (state+context)"]:
        lines.append(
            f"| {r['id'][:8]} | {r['episode']} | {r['query'][:60]!r} | "
            f"{r['gold'][:60]!r} | {r['pred'][:60]!r} | {r['f1']} |"
        )
    md_text = "\n".join(lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "seed": args.seed,
                "full_state": args.full_state,
                "memory_threshold": args.memory_threshold,
                "holdout": len(holdout),
                "retrieval_hit_at_k": {"k": args.probe_k, "hits": hits, "total": len(holdout)},
                "aggregate": {
                    name: {
                        "exact": mean("exact", name),
                        "token_f1": mean("f1", name),
                        "rouge_l": mean("rouge_l", name),
                        "coverage": mean("coverage", name),
                    }
                    for name in candidates
                },
                "rows": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    end = lines.index("## Per-row detail (full candidate)")
    print("\n" + "\n".join(lines[7:end]))
    print(f"\nwrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()