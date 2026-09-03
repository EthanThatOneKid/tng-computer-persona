# TNG Computer Agent Skill

The TNG computer agent skill, grounded in this repo's verified computer
corpus. Lives at `skills/tng-computer/` in this repo -- the single canonical
copy.

Two documents, two audiences:

- **`SKILL.md`** is the *persona definition* -- consumed by an LLM to speak as
  the USS Enterprise (NCC-1701-D) main computer. Voice, stock phrases,
  behaviour rules, grounding sources. No engineering content.
- **This README** is the *engineering doc* -- for humans and agents building
  or evaluating that persona: architecture, CLI, backends, evaluation.

## Architecture

```
skills/tng-computer/
  SKILL.md            skill definition (frontmatter + protocol)
  README.md           this file
  agent_cli.py        one-shot CLI
  src/
    system_prompt.md  system prompt distilled from docs/persona-notes.md
    retrieval.py      IDF token index over data/dialogue.jsonl (+ raw fallback)
    state.py          crew-location tracker from verified computer responses
    agent.py          prompt assembly + pluggable backends
  eval/
    evaluate.py       seeded holdout eval + ablations + retrieval probe
    results/          regenerated reports (gitignored)
```

Three layers, one prompt:

1. **System prompt** (`src/system_prompt.md`) — the operational voice from
   `docs/persona-notes.md`: terse, impersonal, exact, low-emotion,
   status-oriented, with the stock phrases and behaviour rules (no
   fabrication, no breaking character, safety confirmation protocol).
2. **Retrieval** (`src/retrieval.py`) — a dependency-free IDF token index over
   `data/dialogue.jsonl` (60K rows, builds in ~2s). `context_for(query,
   episode, scene)` returns the top non-computer dialogue lines for the
   prompt's `EPISODE CONTEXT` block; raw-transcript fallback included.
3. **State** (`src/state.py`) — crew locations parsed only from the computer's
   *verified* responses (never character dialogue). The prompt's `SHIP STATE`
   block, refreshed per query.

## Backends

- `extractive` (default, offline): state lookup → verified golden-pair memory
  (near-duplicate query) → best-matching computer line of the episode →
  `That information is not available.` No API key.
- `openai`: OpenAI-compatible chat completions via stdlib urllib. Env:
  `OPENAI_API_KEY` (required), `OPENAI_BASE_URL` (default
  `https://api.openai.com/v1`), `OPENAI_MODEL` (default `gpt-4o-mini`).

## Quick start (from the repo root)

```bash
python skills/tng-computer/agent_cli.py "Computer, give me a location on Captain Picard."
python skills/tng-computer/agent_cli.py "where is Commander Data?" --episode 100101.txt
python skills/tng-computer/agent_cli.py "shield status?" --backend openai

python skills/tng-computer/eval/evaluate.py          # offline scaffolding eval
python skills/tng-computer/eval/evaluate.py --backend openai  # real-LLM eval
python skills/tng-computer/eval/evaluate.py --limit 10        # smoke run
```

## Evaluation

`skills/tng-computer/eval/evaluate.py` holds out a seeded 10% of
`enterprise_computer_train.jsonl` (the verified golden set) and scores five
candidates — the four ablations (full / no-context / no-state / bare) plus a
`retrieval top-1` baseline that answers with the best-matching computer line —
by exact match, token F1, ROUGE-L, and gold-token coverage. A retrieval probe
independently reports whether the verified answer line appears in the top-k
computer lines retrieved for the query, and a response-source table shows how
often each layer (state / memory / retrieval / fallback) produced the answer.
Reports land in `skills/tng-computer/eval/results/`.

Holding-out IDs are excluded from the extractive backend's memory, so offline
numbers measure real generalization, not leakage. `--full-state` switches to
realistic agent mode (state built from the whole corpus) — use it when you
want to see the ship-knowledge layer contribute, and read its scores with the
leak caveat in mind.

## Provenance & hygiene

- Only `data/enterprise_computer_train.jsonl` is golden (311 verified rows).
  Mis-paired queries were repaired/blanked and narrative-degraded rows excluded
  in PR #3 (`6ee4fb0`); queryless rows carry `queryless_reason`.
- Never edit generated data by hand — rerun `python -m scripts.run_pipeline`
  (see `AGENTS.md`).
- Results under `skills/tng-computer/eval/results/` are regenerable artifacts.