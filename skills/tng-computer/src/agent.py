"""TNG computer agent: system prompt + retrieval + state, pluggable backends.

Two backends:

- ``extractive`` (offline, no API key needed): answers from (1) the state
  tracker when the query is a known crew-location lookup, (2) verified golden
  pairs whose query nearly matches (the ship's "memory"), (3) the best-matching
  computer line of the episode (the deterministic stand-in for the LLM reading
  retrieved context), else the on-voice fallback "That information is not
  available." Use it to smoke-test the scaffolding and to run the eval harness
  without a key.
- ``openai`` (OpenAI-compatible chat completions over stdlib urllib): the real
  backend. Reads ``OPENAI_API_KEY`` (required), ``OPENAI_BASE_URL`` (default
  ``https://api.openai.com/v1``), ``OPENAI_MODEL`` (default ``gpt-4o-mini``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from retrieval import RetrievalIndex
from state import StateTracker

BASE_DIR = Path(__file__).resolve().parents[1]  # skills/tng-computer/
SYSTEM_PROMPT_PATH = BASE_DIR / "src" / "system_prompt.md"
DATA_DIR = Path(__file__).resolve().parents[3] / "data"  # repo root
TRAIN_PATH = DATA_DIR / "enterprise_computer_train.jsonl"

DEFAULT_FALLBACK = "That information is not available."


@dataclass(frozen=True)
class Request:
    query: str
    episode: str | None = None
    scene: str | None = None
    use_state: bool = True
    use_context: bool = True
    exclude_ids: frozenset[str] = frozenset()


class Backend(Protocol):
    def respond(self, messages: list[dict], req: Request) -> str: ...


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


def _token_f1(a: str, b: str) -> float:
    import re

    ta = set(re.findall(r"[a-z0-9']+", a.lower()))
    tb = set(re.findall(r"[a-z0-9']+", b.lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 2 * inter / (len(ta) + len(tb))


class ExtractiveBackend:
    """Deterministic offline backend: state -> golden memory -> retrieval -> fallback."""

    def __init__(
        self,
        state: StateTracker,
        retrieval: RetrievalIndex,
        exclude_ids: set[str] | None = None,
        memory_path: Path = TRAIN_PATH,
        memory_threshold: float = 0.65,
    ) -> None:
        self.state = state
        self.retrieval = retrieval
        self.exclude_ids = exclude_ids or set()
        self.last_source = "fallback"
        self.memory_threshold = memory_threshold
        self.memory: list[tuple[str, str, str]] = []  # (id, normalized query, response)
        with open(memory_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.memory.append(
                    (
                        row["id"],
                        _normalize(row["messages"][0]["content"]),
                        row["messages"][1]["content"],
                    )
                )

    def respond(self, messages: list[dict], req: Request) -> str:
        if req.use_state:
            person = self.state.extract_person(req.query)
            if person is not None:
                answer = self.state.answer_for(person, req.episode)
                if answer is not None:
                    self.last_source = "state"
                    return answer
        if req.use_context:
            hit = self._memory_hit(req.query)
            if hit is not None:
                self.last_source = "memory"
                return hit
            # Third layer: the best-matching computer line of the episode -- the
            # deterministic stand-in for the LLM reading retrieved context.
            if req.episode is not None:
                top = self.retrieval.search(
                    req.query,
                    k=1,
                    episode=req.episode,
                    is_computer=True,
                    exclude_ids=set(self.exclude_ids),
                )
                if top:
                    self.last_source = "retrieval"
                    return top[0]["text"]
        self.last_source = "fallback"
        return DEFAULT_FALLBACK

    def _memory_hit(self, query: str, threshold: float | None = None) -> str | None:
        q = _normalize(query)
        best: tuple[float, str] | None = None
        for row_id, mem_q, response in self.memory:
            if row_id in self.exclude_ids:
                continue
            if q == mem_q:
                return response
            score = _token_f1(q, mem_q)
            if score >= (threshold or self.memory_threshold) and (best is None or score > best[0]):
                best = (score, response)
        return best[1] if best is not None else None


class OpenAIBackend:
    """OpenAI-compatible chat-completions backend (stdlib urllib only)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "OpenAIBackend requires OPENAI_API_KEY (or pass api_key=)."
            )

    def respond(self, messages: list[dict], req: Request) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0.0}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"LLM API error {err.code}: {err.read().decode()[:300]}") from err
        return body["choices"][0]["message"]["content"].strip()


class TNGComputerAgent:
    """The assembled agent. Answers a user query as the Enterprise computer."""

    def __init__(
        self,
        backend: str = "extractive",
        *,
        system_prompt: str | None = None,
        holdout_ids: set[str] | None = None,
        state: StateTracker | None = None,
        retrieval: RetrievalIndex | None = None,
        memory_threshold: float = 0.65,
    ) -> None:
        self.retrieval = retrieval or RetrievalIndex()
        self.state = state or StateTracker().load(exclude_ids=holdout_ids)
        self.system_prompt = system_prompt or SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        if backend == "extractive":
            self._backend: Backend = ExtractiveBackend(
                self.state,
                self.retrieval,
                exclude_ids=holdout_ids,
                memory_threshold=memory_threshold,
            )
        elif backend == "openai":
            self._backend = OpenAIBackend()
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def build_messages(
        self, query: str, *, episode: str | None = None, scene: str | None = None,
        use_state: bool = True, use_context: bool = True,
        exclude_ids: set[str] | None = None,
    ) -> list[dict]:
        state_block = (
            self.state.snapshot(episode=episode) if use_state else "(none on file)"
        )
        context_block = (
            self.retrieval.context_for(
                query, episode=episode, scene=scene, exclude_ids=exclude_ids
            )
            if use_context
            else "(none on file)"
        )
        system = (
            self.system_prompt.replace("{{STATE}}", state_block)
            .replace("{{CONTEXT}}", context_block)
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

    def respond(
        self,
        query: str,
        *,
        episode: str | None = None,
        scene: str | None = None,
        use_state: bool = True,
        use_context: bool = True,
    ) -> str:
        req = Request(
            query=query,
            episode=episode,
            scene=scene,
            use_state=use_state,
            use_context=use_context,
        )
        messages = self.build_messages(
            query, episode=episode, scene=scene,
            use_state=use_state, use_context=use_context,
        )
        return self._backend.respond(messages, req)


def demo(queries: list[str], episode: str | None = None) -> None:
    agent = TNGComputerAgent(backend="extractive")
    for q in queries:
        print(f"Q: {q}")
        print(f"A: {agent.respond(q, episode=episode)}\n")


if __name__ == "__main__":
    demo(
        [
            "Computer, give me a location on Captain Picard.",
            "Computer, where is Commander Data?",
            "Computer, what is the structural integrity of the warp nacelles?",
        ]
    )