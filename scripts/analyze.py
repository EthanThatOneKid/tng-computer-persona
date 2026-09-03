from __future__ import annotations

from collections import Counter

from scripts.shared import DATA_DIR, read_json, read_jsonl


def main() -> None:
    dialogue = read_jsonl(DATA_DIR / "dialogue.jsonl")
    interactions = read_json(DATA_DIR / "computer_interactions.json")
    profiles = read_json(DATA_DIR / "speaker_profiles.json")
    enterprise_rows = read_jsonl(DATA_DIR / "enterprise_computer_train.jsonl")
    character_rows = read_jsonl(DATA_DIR / "tng_character_train.jsonl")

    by_season = Counter(row["season"] for row in dialogue)
    top_speakers = sorted(profiles, key=lambda row: row["line_count"], reverse=True)[:12]
    query_speakers = Counter(row["query_speaker"] for row in interactions if row["query_speaker"])

    narrative_degraded = [row for row in interactions if row.get("narrative_degraded")]
    degraded_by_episode = Counter(row["episode"] for row in narrative_degraded)

    report_lines = [
        "# TNG Persona Dataset Report",
        "",
        f"- Dialogue rows: {len(dialogue)}",
        f"- Computer interactions: {len(interactions)}",
        f"- Narrative-degraded interactions (flagged): {len(narrative_degraded)}",
        f"- Speaker profiles: {len(profiles)}",
        f"- Enterprise computer training rows: {len(enterprise_rows)}",
        f"- Character-conditioned training rows: {len(character_rows)}",
        "",
        "## Dialogue by season",
        "",
        "| Season | Dialogue lines |",
        "|---|---:|",
    ]

    for season, count in sorted(by_season.items()):
        report_lines.append(f"| {season} | {count} |")

    report_lines.extend(
        [
            "",
            "## Most frequent speakers",
            "",
            "| Speaker | Lines | Episodes | Avg words | Question rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in top_speakers:
        report_lines.append(
            f"| {row['speaker']} | {row['line_count']} | {row['episode_count']} | {row['avg_word_count']} | {row['question_rate']} |"
        )

    report_lines.extend(
        [
            "",
            "## Main speakers addressing the computer",
            "",
            "| Speaker | Interactions |",
            "|---|---:|",
        ]
    )

    for speaker, count in query_speakers.most_common(12):
        report_lines.append(f"| {speaker} | {count} |")

    report_lines.extend(
        [
            "",
            "## Narrative-degraded interactions",
            "",
            "Interactions where the plot intentionally degrades the computer's capability",
            "(hijack, virus, alien interference, possession, simulated failure). They are",
            "flagged in `computer_interactions.json` and excluded from the golden enterprise",
            "computer training set.",
            "",
            "| Episode | Flagged |",
            "|---|---:|",
        ]
    )

    episode_order = {
        row["episode"]: (row["season"], row["episode_number"]) for row in interactions
    }
    for episode, count in sorted(
        degraded_by_episode.items(),
        key=lambda item: episode_order.get(item[0], (99, 99)),
    ):
        report_lines.append(f"| {episode} | {count} |")

    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The enterprise computer subset is best suited for terse operational personas.",
            "- The character-conditioned JSONL keeps a speaker label in metadata so you can filter for Picard, Data, Guinan, or any other voice later.",
        ]
    )

    report_path = DATA_DIR.parent / "docs" / "DATASET_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
