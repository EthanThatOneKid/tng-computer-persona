"""Retrieval script for the TNG computer skill.

Indexes ``data/dialogue.jsonl`` (all parsed TNG dialogue, including computer
lines) and can fall back to the raw transcripts under
``data/raw/star_trek_transcript_search/scripts/NextGen``. Search is a small
deterministic lexical scorer (token overlap weighted by IDF) -- deliberately
dependency-free and fast enough to rebuild in seconds.

Usage (from the repo root):

    python skills/tng-computer/scripts/retrieval.py "where is Commander Data?" --episode 100101.txt
    python skills/tng-computer/scripts/retrieval.py "shield status?" --episode 100161.txt --scene Bridge --k 8
    python skills/tng-computer/scripts/retrieval.py "Computer, locate the source of the signal" --computer

The default output is the ``EPISODE CONTEXT`` block for the agent prompt:
non-computer dialogue lines only (the computer's own lines are the thing being
predicted, so they are not shown as context). Pass ``--computer`` to retrieve
computer lines instead (useful for checking what the ship said).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]  # repo root
DIALOGUE_PATH = BASE_DIR / "data" / "dialogue.jsonl"
TRANSCRIPTS_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "star_trek_transcript_search"
    / "scripts"
    / "NextGen"
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric token stream (keeps apostrophes)."""
    return _TOKEN_RE.findall(text.lower())


class RetrievalIndex:
    """In-memory index over the parsed dialogue corpus."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows: list[dict] = rows if rows is not None else self._load()
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._doc_tokens: list[set[str]] = []
        for i, row in enumerate(self.rows):
            toks = set(tokenize(row["text"]))
            self._doc_tokens.append(toks)
            for tok in toks:
                self._postings[tok].append(i)
        n = len(self.rows)
        self._idf = {
            tok: math.log((n + 1) / (len(post) + 1)) + 1.0
            for tok, post in self._postings.items()
        }

    @staticmethod
    def _load() -> list[dict]:
        with open(DIALOGUE_PATH, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        episode: str | None = None,
        scene: str | None = None,
        is_computer: bool | None = None,
        exclude_ids: set[str] | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Top-k dialogue rows for ``query``, ranked by IDF-weighted overlap."""
        exclude_ids = exclude_ids or set()
        q_toks = set(tokenize(query))
        if not q_toks:
            return []
        scores: dict[int, float] = {}
        for tok in q_toks:
            for i in self._postings.get(tok, ()):
                scores[i] = scores.get(i, 0.0) + self._idf[tok]
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[dict] = []
        for i, score in ranked:
            if score < min_score:
                break
            row = self.rows[i]
            if episode is not None and row["episode"] != episode:
                continue
            if scene is not None and row["scene"] != scene:
                continue
            if is_computer is not None and bool(row.get("is_computer")) != is_computer:
                continue
            if row["id"] in exclude_ids:
                continue
            out.append(row)
            if len(out) >= k:
                break
        return out

    def context_for(
        self,
        query: str,
        *,
        episode: str | None = None,
        scene: str | None = None,
        k: int = 6,
        exclude_ids: set[str] | None = None,
        is_computer: bool | None = False,
    ) -> str:
        """Formatted context block for the system prompt.

        Defaults to non-computer dialogue (the computer's own lines are the
        thing we are trying to predict, so they are not shown as context).
        """
        hits = self.search(
            query,
            k=k,
            episode=episode,
            scene=scene,
            is_computer=is_computer,
            exclude_ids=exclude_ids,
        )
        if not hits:
            return "(none on file)"
        return "\n".join(
            f"- [{row['scene']}] {row['speaker']}: {row['text']}" for row in hits
        )

    def transcript_slice(self, episode: str, scene: str | None = None, window: int = 4) -> str:
        """Raw-transcript fallback: plain text lines around ``scene``."""
        path = TRANSCRIPTS_DIR / episode
        if not path.exists():
            return "(raw transcript not on file)"
        lines = path.read_text(encoding="utf-8").splitlines()
        if scene is None:
            return "\n".join(lines[: window * 8])
        for i, line in enumerate(lines):
            if f"[{scene}]" in line:
                start = max(0, i - 2)
                return "\n".join(lines[start : i + window * 8])
        return "(scene not found in raw transcript)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="the crew's question/command")
    parser.add_argument("--episode", default=None, help="episode file, e.g. 100101.txt")
    parser.add_argument("--scene", default=None, help="scene, e.g. Bridge")
    parser.add_argument("--k", type=int, default=6, help="number of lines to retrieve")
    parser.add_argument("--computer", action="store_true",
                        help="retrieve computer lines instead of dialogue context")
    args = parser.parse_args()

    index = RetrievalIndex()
    kw = {"is_computer": True} if args.computer else {}
    block = index.context_for(
        " ".join(args.query),
        episode=args.episode,
        scene=args.scene,
        k=args.k,
        **kw,
    )
    print(block)


if __name__ == "__main__":
    main()