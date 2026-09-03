from __future__ import annotations

from collections import defaultdict
import re

from scripts.shared import DATA_DIR, read_json, read_jsonl, stable_id, write_jsonl

DIRECT_QUERY_RE = re.compile(
    r"\b(computer|identify|locate|location|display|run|scan|track|access|status|where|what|who|when|how|open|close|seal|release|engage|deactivate|activate|program|override|diagnostic|find|transfer|set|request|inquiry|lock)\b",
    re.IGNORECASE,
)


def looks_like_direct_query(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    return bool(DIRECT_QUERY_RE.search(stripped))


def build_enterprise_computer_records(interactions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for interaction in interactions:
        if interaction.get("narrative_degraded"):
            # Narrative-driven degradation (hijack, virus, alien interference) is
            # not representative of the capable computer persona; keep it out of
            # the golden training set.
            continue
        if not interaction["query_text"] or not interaction["response_text"]:
            continue
        if not looks_like_direct_query(interaction["query_text"]):
            continue
        rows.append(
            {
                "id": interaction["id"],
                "messages": [
                    {"role": "user", "content": interaction["query_text"]},
                    {"role": "assistant", "content": interaction["response_text"]},
                ],
                "metadata": {
                    "speaker": "COMPUTER",
                    "query_speaker": interaction["query_speaker"],
                    "season": interaction["season"],
                    "episode": interaction["episode"],
                    "scene": interaction["scene"],
                },
            }
        )
    return rows


def build_character_records(dialogue: list[dict]) -> list[dict]:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in dialogue:
        by_episode[row["episode"]].append(row)

    rows: list[dict] = []
    for episode, lines in by_episode.items():
        lines.sort(key=lambda row: row["line_num"])
        for index, current in enumerate(lines):
            if current["is_computer"]:
                continue

            context_rows = [line for line in lines[max(0, index - 2):index] if not line["is_computer"]]
            if not context_rows:
                continue

            context_text = "\n".join(f"{line['speaker']}: {line['text']}" for line in context_rows)
            system_text = f"Respond in the voice of {current['speaker']} from Star Trek: The Next Generation."
            rows.append(
                {
                    "id": stable_id(episode, current["line_num"], current["speaker"], current["text"]),
                    "messages": [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": context_text},
                        {"role": "assistant", "content": current["text"]},
                    ],
                    "metadata": {
                        "speaker": current["speaker"],
                        "season": current["season"],
                        "episode": current["episode"],
                        "scene": current["scene"],
                    },
                }
            )
    return rows


def main() -> None:
    dialogue = read_jsonl(DATA_DIR / "dialogue.jsonl")
    interactions = read_json(DATA_DIR / "computer_interactions.json")

    enterprise_rows = build_enterprise_computer_records(interactions)
    character_rows = build_character_records(dialogue)

    write_jsonl(DATA_DIR / "enterprise_computer_train.jsonl", enterprise_rows)
    write_jsonl(DATA_DIR / "tng_character_train.jsonl", character_rows)

    print(f"Wrote {len(enterprise_rows)} enterprise computer training rows")
    print(f"Wrote {len(character_rows)} character-conditioned training rows")


if __name__ == "__main__":
    main()
