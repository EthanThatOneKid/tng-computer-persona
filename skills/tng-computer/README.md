# TNG Computer Agent Skill

The TNG computer agent skill, grounded in this repo's verified computer
corpus. Lives at `skills/tng-computer/` in this repo — the single canonical
copy.

Two documents, two audiences:

- **`SKILL.md`** is the *persona definition* — consumed by an LLM to speak as
  the USS Enterprise (NCC-1701-D) main computer. Voice, stock phrases,
  behaviour rules, grounding sources. No engineering content.
- **This README** is the *engineering doc* — for humans and agents building
  or evaluating that persona: the grounding scripts and the evaluation.

## Structure

```
skills/tng-computer/
  SKILL.md            persona definition (consumed by an LLM)
  README.md           this file (engineering doc)
  system_prompt.md    the prompt template with {{STATE}} / {{CONTEXT}} slots
  retrieval.py        runnable script: episode-context retrieval
  state.py            runnable script: verified ship-state / crew locations
  eval/
    evaluate.py       seeded holdout eval + ablations + retrieval probe
    results/          regenerated reports (gitignored)
```

No application code — the consuming agent already *is* the LLM harness. This
package only exposes the grounding logic as dependency-free scripts (repo
convention: deterministic, stdlib-only) plus the persona definition.

## Grounding scripts

- **`state.py`** — builds crew-location facts from the ship's *verified*
  computer responses in `data/enterprise_computer_train.jsonl` (never
  character dialogue), keyed per episode (locations are episode-relative).
  `--episode` prints the `SHIP STATE` block; `--query` answers a
  crew-location query from verified state.
- **`retrieval.py`** — an IDF token index over `data/dialogue.jsonl` (60K
  rows, builds in ~2s) with raw-transcript fallback. Prints the `EPISODE
  CONTEXT` block for a query/episode/scene (non-computer lines by default;
  `--computer` retrieves the ship's own lines).

## Quick start (from the repo root)

```bash
python skills/tng-computer/state.py --episode 100110.txt
python skills/tng-computer/state.py --query "where is Commander Data?" --episode 100101.txt
python skills/tng-computer/retrieval.py "where is Commander Data?" --episode 100101.txt

python skills/tng-computer/eval/evaluate.py                   # offline eval
python skills/tng-computer/eval/evaluate.py --backend openai  # real-LLM eval
python skills/tng-computer/eval/evaluate.py --limit 10        # smoke run
```

## Evaluation

`eval/evaluate.py` holds out a seeded 10% of
`enterprise_computer_train.jsonl` (the verified golden set) and scores five
candidates — the four ablations of the layered offline answerer (full /
no-context / no-state / bare; state → verified golden memory → best-matching
episode computer line → `That information is not available.`) plus a
`retrieval top-1` baseline — by exact match, token F1, ROUGE-L, and
gold-token coverage. A retrieval probe independently reports whether the
verified answer line appears in the top-k computer lines retrieved for the
query, and a response-source table shows how often each layer produced the
answer. Reports land in `eval/results/`.

Holding-out IDs are excluded from the offline answerer's memory, so the
numbers measure real generalization, not leakage. `--full-state` switches to
realistic agent mode (state from the whole corpus) — read its scores with the
leak caveat in mind. `--backend openai` fills `system_prompt.md`'s
`{{STATE}}`/`{{CONTEXT}}` slots per the ablation flags and scores the LLM's
replies (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`).

## Provenance & hygiene

- Only `data/enterprise_computer_train.jsonl` is golden (311 verified rows).
  Mis-paired queries were repaired/blanked and narrative-degraded rows excluded
  in PR #3 (`6ee4fb0`); queryless rows carry `queryless_reason`.
- Never edit generated data by hand — rerun `python -m scripts.run_pipeline`
  (see `AGENTS.md`).
- Results under `eval/results/` are regenerable artifacts.