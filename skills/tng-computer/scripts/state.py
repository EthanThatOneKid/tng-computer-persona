"""Ship-state script for the TNG computer skill.

Builds verified crew-location facts from the ship's own computer responses in
``data/enterprise_computer_train.jsonl``. Only the computer's *verified*
replies are trusted (never character dialogue, which can be speculation,
gossip, or a lie). Extraction is deterministic regex -- no NLP dependencies --
and skips negations ("no longer aboard") and refusals.

Facts are keyed by *episode*: locations in the show are episode-relative
(Picard is in Ten Forward in one episode, Transporter room three in another),
so an episode-scoped SHIP STATE block is required to avoid answering with the
wrong episode's facts. ``load(..., exclude_ids=...)`` supports hold-out
exclusion for evaluation so the tracker never learns from rows being scored.

Usage (from the repo root):

    python skills/tng-computer/scripts/state.py --episode 100110.txt
    python skills/tng-computer/scripts/state.py --query "where is Commander Data?" --episode 100101.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]  # repo root
TRAIN_PATH = BASE_DIR / "data" / "enterprise_computer_train.jsonl"

# "Captain Picard is in Transporter room three." /
# "Lieutenant Commander Data now located in Holodeck area 4J."
# Responses are title-cased in the corpus, so the whole pattern is
# case-insensitive; the matched rank/verb are kept so answers can reproduce
# the gold style exactly.
_RANK = r"Lieutenant Commander|Commander|Captain|Counselor|Doctor|Ensign|Ambassador|Mr\.|Ms\.|Mrs\.|Lieutenant"
_LOC_RE = re.compile(
    rf"^(?:(?P<rank>{_RANK})\s+)?"
    rf"(?P<name>[A-Za-z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)"
    rf"\s+(?P<verb>is now located in|is located in|now located in|is in|is on|is at|is aboard)\s+"
    rf"(?P<place>[^.!?]+?)\.?$",
    re.IGNORECASE,
)

# Location queries from the corpus: "Computer, give me a location on Captain
# Picard." / "Computer, what is the location of Alexander Rozhenko?" /
# "Ensign, can you help me find Commander Data?" / "Where is Commander Riker?"
_PERSON_QUERY_RE = re.compile(
    r"(?:locate|location on|location of|find|where(?:'s| is))\s+"
    r"(?:(?:Lieutenant Commander|Commander|Captain|Counselor|Doctor|Ensign|"
    r"Ambassador|Mr\.|Ms\.|Mrs\.|Lieutenant)\s+)?"
    r"(?P<name>[A-Za-z][A-Za-z]+)",
    re.IGNORECASE,
)

# "That information is not available." / "Unknown." / "Picard is no longer
# aboard the Enterprise." must never enter state.
_SKIP_MARKERS = ("not available", "unknown", "no longer", "not on board")


class StateTracker:
    """Episode-scoped crew-location facts.

    ``facts[(episode, name)] = (display_name, verb, place)``
    """

    def __init__(self) -> None:
        self.facts: dict[tuple[str, str], tuple[str, str, str]] = {}

    def load(
        self, path: Path = TRAIN_PATH, exclude_ids: set[str] | None = None
    ) -> "StateTracker":
        exclude_ids = exclude_ids or set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in exclude_ids:
                    continue
                self.update_from_response(
                    row["metadata"]["episode"],
                    row["messages"][1]["content"],
                )
        return self

    def update_from_response(self, episode: str, response_text: str) -> None:
        lowered = response_text.lower()
        if any(marker in lowered for marker in _SKIP_MARKERS):
            return
        match = _LOC_RE.match(response_text.strip())
        if match is None:
            return
        raw_name = match.group("name")
        place = match.group("place").strip()
        # "his quarters" / "her quarters" are pronouns, not facts worth tracking
        # under the person's own name; skip them.
        if re.fullmatch(r"(his|her|their) .+", place, re.IGNORECASE):
            return
        rank = match.group("rank") or ""
        display = f"{rank} {raw_name}" if rank else raw_name
        verb = match.group("verb").strip()
        self.facts[(episode, raw_name.upper())] = (display, verb, place)

    def extract_person(self, query: str) -> str | None:
        """Uppercase crew name a location query asks about, if any.

        Only a direct regex capture counts -- no fuzzy matching, so queries
        that merely *mention* a name ("(imitating Picard) ...") never fire.
        """
        match = _PERSON_QUERY_RE.search(query)
        if match is None:
            return None
        return match.group("name").upper()

    def answer_for(self, name: str, episode: str | None) -> str | None:
        """Canonical location answer in the gold style, scoped to ``episode``."""
        key = name.upper()
        fact = self.facts.get((episode, key)) if episode is not None else None
        if fact is None:
            return None
        display, verb, place = fact
        return f"{display} {verb} {place}."

    def snapshot(self, episode: str | None = None) -> str:
        facts = (
            [(k, v) for (ep, _), v in self.facts.items()] if episode is None
            else [(k, v) for (ep, k), v in self.facts.items() if ep == episode]
        )
        if not facts:
            return "(none on file)"
        scope = f" ({episode})" if episode is not None else " (all episodes)"
        lines = [f"- CREW LOCATIONS{scope} (verified computer reports):"]
        for name, (display, verb, place) in sorted(facts):
            lines.append(f"  {name}: {place}")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default=None, help="episode file, e.g. 100101.txt")
    parser.add_argument("--query", default=None,
                        help="answer a crew-location query from verified state")
    args = parser.parse_args()

    tracker = StateTracker().load()
    if args.query:
        person = tracker.extract_person(args.query)
        if person is None:
            print("(no person identified in query)")
            return
        answer = tracker.answer_for(person, args.episode)
        print(answer if answer is not None else "That information is not available.")
    else:
        print(tracker.snapshot(episode=args.episode))


if __name__ == "__main__":
    main()