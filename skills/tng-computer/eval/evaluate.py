"""Evaluation harness for the TNG computer skill.

Holds out a seeded fraction of ``enterprise_computer_train.jsonl`` and scores
candidate configurations against the computer's *verified* responses:

- exact match (normalized)
- token F1
- ROUGE-L (LCS-based F1)
- gold-token coverage

Candidates:

- ``full (state+context)``, ``no-context``, ``no-state``, ``bare`` -- the four
  ablations of the layered offline answerer (state lookup -> verified golden
  memory -> best-matching episode computer line -> on-voice fallback), so the
  value of each layer is visible. With ``--backend openai`` the same ablations
  are evaluated through a real LLM: the system prompt (``system_prompt.md``)
  is filled with the ``SHIP STATE`` / ``EPISODE CONTEXT`` blocks according to
  the flags, and the LLM's reply is scored.
- ``retrieval top-1`` (extractive backend only) -- answers with the single
  best-matching computer line of the episode, scoring the retrieval layer
  directly as an answerer.

Two state modes:

- default (leak-free): state is built from the train split only, so held-out
  rows can never teach the tracker their own answers.
- ``--full-state``: state is built from the whole corpus -- the realistic agent
  setup (a real ship computer would know the ship). Read scores with that
  caveat in mind.

The retrieval probe is independent of any answerer: it reports whether the
verified answer line is among the top-k computer lines of the episode
retrieved for the query.

Usage (from the repo root):

    python skills/tng-computer/eval/evaluate.py                  # offline, leak-free
    python skills/tng-computer/eval/evaluate.py --full-state     # offline, realistic
    python skills/tng-computer/eval/evaluate.py --backend openai # real LLM (OPENAI_API_KEY)
    python skills/tng-computer/eval/evaluate.py --limit 10       # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]  # skills/tng-computer/
REPO_ROOT = SKILL_DIR.parents[1]
sys.path.insert(0, str(SKILL_DIR))

from retrieval import RetrievalIndex  # noqa: E402
from state import StateTracker  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "enterprise_computer_train.jsonl"
SYSTEM_PROMPT_PATH = SKILL_DIR / "system_prompt.md"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_FALLBACK = "That information is not available."


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


# ---------------------------------------------------------------- answerers

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


def build_memory(rows: list[dict], exclude_ids: set[str]) -> list[tuple[str, str, str]]:
    """Verified golden pairs as (id, normalized query, response)."""
    memory = []
    for row in rows:
        if row["id"] in exclude_ids:
            continue
        memory.append(
            (
                row["id"],
                _normalize(row["messages"][0]["content"]),
                row["messages"][1]["content"],
            )
        )
    return memory


def memory_hit(memory: list[tuple[str, str, str]], query: str, threshold: float) -> str | None:
    q = _normalize(query)
    best: tuple[float, str] | None = None
    for _, mem_q, response in memory:
        if q == mem_q:
            return response
        score = token_f1(q, mem_q)
        if score >= threshold and (best is None or score > best[0]):
            best = (score, response)
    return best[1] if best is not None else None


def layered_answer(
    query: str,
    episode: str | None,
    *,
    use_state: bool,
    use_context: bool,
    state: StateTracker,
    index: RetrievalIndex,
    memory: list[tuple[str, str, str]],
    exclude_ids: set[str],
    memory_threshold: float,
) -> tuple[str, str]:
    """Offline layered answerer: state -> memory -> retrieval -> fallback."""
    if use_state:
        person = state.extract_person(query)
        if person is not None:
            answer = state.answer_for(person, episode)
            if answer is not None:
                return answer, "state"
    if use_context:
        hit = memory_hit(memory, query, memory_threshold)
        if hit is not None:
            return hit, "memory"
        if episode is not None:
            top = index.search(
                query, k=1, episode=episode, is_computer=True, exclude_ids=exclude_ids
            )
            if top:
                return top[0]["text"], "retrieval"
    return DEFAULT_FALLBACK, "fallback"


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


# ---------------------------------------------------------------- LLM path

def build_messages(
    system_prompt_text: str,
    query: str,
    *,
    episode: str | None,
    scene: str | None,
    use_state: bool,
    use_context: bool,
    state: StateTracker,
    index: RetrievalIndex,
) -> list[dict]:
    state_block = state.snapshot(episode=episode) if use_state else "(none on file)"
    context_block = (
        index.context_for(query, episode=episode, scene=scene)
        if use_context
        else "(none on file)"
    )
    system = (
        system_prompt_text.replace("{{STATE}}", state_block).replace("{{CONTEXT}}", context_block)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def openai_chat(messages: list[dict], timeout: int = 60) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("--backend openai requires OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.0}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"LLM API error {err.code}: {err.read().decode()[:300]}") from err
    return body["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------- harness

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
                        help="memory near-duplicate token-F1 threshold")
    args = parser.parse_args()

    rows = load_rows(TRAIN_PATH)
    train_rows, holdout = holdout_split(rows, args.holdout_frac, args.seed)
    if args.limit:
        holdout = holdout[: args.limit]
    holdout_ids = {r["id"] for r in holdout}
    print(f"backend={args.backend}  train={len(train_rows)}  holdout={len(holdout)}  "
          f"(seed {args.seed}, frac {args.holdout_frac})")

    # Shared scaffolding: index once; state excludes holdout rows unless
    # --full-state (realistic agent mode) is requested.
    index = RetrievalIndex()
    state = StateTracker().load(exclude_ids=None if args.full_state else holdout_ids)
    memory = build_memory(train_rows, holdout_ids)
    system_prompt_text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    candidates: dict[str, dict] = {
        "full (state+context)": {"mode": "agent", "use_state": True, "use_context": True},
        "no-context": {"mode": "agent", "use_state": True, "use_context": False},
        "no-state": {"mode": "agent", "use_state": False, "use_context": True},
        "bare (system only)": {"mode": "agent", "use_state": False, "use_context": False},
    }
    if args.backend == "extractive":
        candidates["retrieval top-1 (computer lines)"] = {"mode": "retrieval"}

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
            elif args.backend == "extractive":
                pred, source = layered_answer(
                    query, episode,
                    use_state=spec["use_state"],
                    use_context=spec["use_context"],
                    state=state, index=index, memory=memory,
                    exclude_ids=holdout_ids, memory_threshold=args.memory_threshold,
                )
            else:
                messages = build_messages(
                    system_prompt_text, query, episode=episode, scene=scene,
                    use_state=spec["use_state"], use_context=spec["use_context"],
                    state=state, index=index,
                )
                pred = openai_chat(messages)
                source = "llm"
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
        "# TNG Computer Skill Evaluation",
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

    if args.backend == "extractive":
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