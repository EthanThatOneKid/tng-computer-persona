from __future__ import annotations

import re
from collections import defaultdict

from scripts.shared import DATA_DIR, read_jsonl, stable_id, top_tokens, write_json

TRIVIAL_ACK_RE = re.compile(
    r"^(thank you|thanks|you're welcome|you are welcome|yes sir|no sir|right sir|very good|bye|goodbye|hello)\b",
    re.IGNORECASE,
)

# Curated narrative-degradation flags: interaction IDs whose response is driven by the
# plot intentionally degrading the computer's capability (hijack, virus, alien
# interference, possession, simulated failure). Such responses are not representative
# of the capable Enterprise computer persona and must not be treated as golden training
# data. Keyed by interaction ID with the episode and reason. This classification
# requires plot knowledge, so it is curated explicitly rather than inferred from text.
NARRATIVE_DEGRADED: dict[str, str] = {
    # "11001001" -- the Bynars hijack the Enterprise computer to steal the ship.
    "8cb569a3614e": "11001001: computer hijacked by Bynars (altered male voice during containment crisis)",
    "244f5e2ccd03": "11001001: Bynar-coded fake evacuation announcement",
    "55d2532195ec": "11001001: Bynar-coded fake evacuation announcement",
    "0b04a6aeba2a": "11001001: false report under Bynar control ('All decks empty.')",
    "fc5eeb8b2910": "11001001: explains the Bynar-programmed red alert",
    "39386d78b6d8": "11001001: false evacuation claim under Bynar control",
    "d26d78684cca": "11001001: withholds the Bynar theft ('That information is not available.')",
    "7b04f1a397df": "11001001: altered male voice during Bynar-modified auto-destruct",
    "e9b9c3fbc3e5": "11001001: altered male voice during Bynar-modified auto-destruct",
    # "Home Soil" -- the crystalline life form's energy field degrades ship systems.
    "c2a4621d822f": "Home Soil: translator fried by alien interference ('(gibberish)')",
    "f488bf798efc": "Home Soil: sensors overloaded by alien energy field",
    "7c71bb2e38d7": "Home Soil: instruments deactivated by alien energy field",
    "a3eff6e23613": "Home Soil: translator failing under alien interference",
    # "The Child" -- the alien entity (Ian) answers through the computer in a male voice.
    "c39834d10614": "The Child: alien entity speaks through the computer (male voice during module diagnostic)",
    "45caaf322da6": "The Child: alien entity confirms its own growth through the computer (same possession exchange)",
    # "Contagion" -- the Iconian computer virus corrupts the computer.
    "be24541481ba": "Contagion: Iconian virus corrupts computer (unintelligible 'Kandar' speech)",
    "14254e08f69c": "Contagion: Iconian virus corrupts computer (unintelligible 'Kandar' speech)",
    "4f35f947137e": "Contagion: Iconian virus corrupts computer (unintelligible 'Kandar' speech)",
    # "Evolution" -- the nanites take over the computer.
    "9a8aed995a62": "Evolution: denies real control malfunction while nanites interfere",
    "3bcaf0cf560f": "Evolution: nanite takeover -- computer stuck replying with chess moves",
    "ec4133040f4f": "Evolution: nanite takeover -- computer stuck replying with chess moves",
    # "Rascals" -- the classroom-7 computer is a child's educational toy (a talking
    # fish) that refuses real commands and offers games and spelling lessons instead;
    # not representative of the capable Enterprise computer persona.
    "39990ff06682": "Rascals: classroom-7 child's computer greets the children (toy persona)",
    "35337ff6e620": "Rascals: classroom-7 child's computer refuses the command and offers a game",
    "dd0c268d4f90": "Rascals: classroom-7 child's computer refuses the command and offers plants/animals",
    "3eae989d57f3": "Rascals: classroom-7 child's computer gives a canned spelling lesson ('E N T E R P R I S E')",
    # "Shades of Gray" -- Riker's coma-dream clip show replays the Bynar-modified
    # auto-destruct from "11001001", including the computer's altered male voice.
    "68a34e760985": "Shades of Gray: clip replay of the Bynar-modified auto-destruct ('Recognise Picard...' male voice), auto-paired with Troi's Sickbay line across the scene cut",
    # "Ship in a Bottle" -- Moriarty seizes the Enterprise computer.
    "1830fe54bdd3": "Ship in a Bottle: Moriarty takeover -- 'Command functions are offline.'",
    "c07bc06589b8": "Ship in a Bottle: Moriarty takeover -- 'Authorisation denied.'",
    "fb285c43322d": "Ship in a Bottle: Moriarty takeover -- 'Picard command codes are no longer valid.'",
    "88880700b27f": "Ship in a Bottle: Moriarty takeover -- 'Command functions are offline.'",
    "270e4509cb7b": "Ship in a Bottle: computer serves Moriarty ('Interface complete.')",
}

# Curated query-repair overrides: interaction IDs whose auto-paired query is not the
# utterance that actually prompted the computer's response. The extractor pairs each
# computer line with the best-scoring nearby non-computer line, which goes wrong when
# the computer opens a new scene (greeting/announcement) shortly after unrelated
# dialogue from the previous scene. Keyed by interaction ID -> (query_speaker, query_text).
# Requires plot knowledge, so it is curated explicitly rather than inferred from text.
QUERY_REPAIRS: dict[str, tuple[str, str]] = {
    # "Rascals" -- the classroom computer greets the children when they reach the school
    # room; the extractor mis-paired the greeting with Lurin's Ready-room line
    # ('Very hazardous, Commander.'). The real command is the one Picard Jr. gives the
    # classroom computer right after the greeting.
    "39990ff06682": ("PICARD JR", "Computer, display interior security grid."),
    # "Suspicions" -- Crusher, inside the shuttle testing the metaphasic shield, asks the
    # computer for shield status; the extractor paired the report with Picard's question
    # to Crusher on the viewscreen (a line to a person, one row above the real query).
    "f2d2b4a0e8c4": ("CRUSHER", "I'm all right. Computer, shield status?"),
}

# Curated query-blank overrides: interaction IDs whose auto-paired query must be
# removed entirely because no utterance actually elicited the computer's line.
# These are unsolicited computer announcements/prompts -- ship-wide warnings,
# broadcasts, countdowns, holodeck/test prompts, or automated status reports --
# that the extractor latched onto unrelated nearby dialogue for (often across a
# scene cut the heuristic cannot see). Such interactions carry null query fields
# plus a materialised queryless_reason on the row itself (the corpus already
# carries queryless rows of this kind). Keyed by interaction ID with the episode
# and reason. Requires plot knowledge, so it is curated explicitly rather than
# inferred from text.
QUERY_BLANKS: dict[str, str] = {
    # "The Big Goodbye" -- holodeck setup prompt right after the Ready-room scene.
    "3c9490fc92e7": "The Big Goodbye: unsolicited holodeck prompt ('Programme desired location.') auto-paired with Troi's Ready-room line",
    # "11001001" -- announcements around the Bynar hijack.
    "324d8b3fff0d": "11001001: 'Bridge access denied' warning auto-paired with Picard's observation in Engineering",
    "4decf0154e0c": "11001001: 'cleared the starbase perimeter' announcement auto-paired with Bynar dialogue",
    # "Coming Of Age" -- classroom computer announcing the next test on a schedule.
    "bd44437f6159": "Coming Of Age: unsolicited classroom test announcement auto-paired with Riker's corridor line",
    # "Where Silence Has Lease" -- the bluffed auto-destruct sequence: voice-print
    # recognition and ship-wide countdown broadcasts no utterance elicited.
    "1a187ac9db4a": "Where Silence Has Lease: destruct voice-print recognition auto-paired with Pulaski's lounge line",
    "ef542ec997b8": "Where Silence Has Lease: destruct countdown broadcast auto-paired with Riker's crew debate",
    "fe466d6d59e9": "Where Silence Has Lease: destruct countdown broadcast auto-paired with Picard's question to Troi",
    # "The Defector" -- incoming-priority-message alert at a scene cut.
    "6e5c70a49fc6": "The Defector: incoming-message alert auto-paired with Setal's line in the debriefing room",
    # "Final Mission" -- radiation warning while the away team is off-ship.
    "65c28fa73790": "Final Mission: unsolicited radiation warning auto-paired with Dirgo's line on the planet",
    # "Data's Day" -- holodeck program-start announcement at a scene cut.
    "8a1312d4cce6": "Data's Day: unsolicited holodeck start announcement auto-paired with T'Pel's line",
    # "The Chase" -- autonomous search-result report at a scene cut.
    "2a5e0c7d59ce": "The Chase: unsolicited 'Pattern match found' report auto-paired with Data's Bridge suggestion",
    # "Where Silence Has Lease" -- ship-wide alert when the entity drains the ship's
    # power; auto-paired with Picard hailing the transporter room (a comms call, not a command).
    "6098b0d8fc42": "Where Silence Has Lease: unsolicited 'Emergency power engaged' alert auto-paired with Picard hailing the transporter room",
    # "Remember Me" -- life-support countdown broadcast while Crusher is trapped in the
    # collapsing warp bubble; auto-paired with her self-talk about a stable threshold.
    "18a17041338e": "Remember Me: unsolicited life-support countdown auto-paired with Crusher's self-talk",
    # "Eye Of The Beholder" -- plasma-venting warning during Troi's vision; auto-paired
    # with Worf's line to Troi ('What are you doing?').
    "235d1b6d3ec5": "Eye Of The Beholder: unsolicited plasma-venting warning auto-paired with Worf's question to Troi",
    # "Rascals" -- shuttle structural-failure warning after the Ferengi attack;
    # auto-paired with Picard's order to Ro at the shuttle controls.
    "21c3e8890fda": "Rascals: unsolicited shuttle structural-failure warning auto-paired with Picard's order to Ro",
    # "In Theory" -- decompression warning when the false planet appears; auto-paired
    # with Picard's order to Data (a person, not the computer).
    "f688b1caf691": "In Theory: unsolicited decompression warning auto-paired with Picard's order to Data",
    # "Elementary, Dear Data" -- program-completion announcement while Data, La Forge and
    # Pulaski converse outside the holodeck (no computer address in the window).
    "a9ceb20f0306": "Elementary, Dear Data: unsolicited 'Programme complete. You may enter.' announcement auto-paired with La Forge's question to Data",
    # "Booby Trap" -- real-ship warnings during the trap crisis, interleaved with the
    # crew's planning dialogue (Geordi's holodeck sims with Brahms and the bridge debate).
    "6d437d3d7946": "Booby Trap: unsolicited energy-critical/terminate-simulations announcement auto-paired with La Forge's line to the Brahms simulation",
    "3fad3c2090db": "Booby Trap: unsolicited shield/radiation warning auto-paired with La Forge's remark to the Brahms simulation",
    "6d715e358f88": "Booby Trap: unsolicited shield/radiation warning auto-paired with La Forge's escape plan to Picard",
    "ebe7f1ec4b6b": "Booby Trap: unsolicited shield/radiation warning auto-paired with Riker's all-hands announcement",
    # "The Defector" -- incoming-priority-message alert interrupting the Bridge debate.
    "b4d48b06cd93": "The Defector: unsolicited priority-one Starfleet message alert auto-paired with Picard's musing on Romulan cloaking",
    # "Hollow Pursuits" -- structural-failure warning while the runaway-acceleration crisis
    # unfolds; Riker's 'Injector status?' was a question to Data, who answered it first.
    "a3bab9683340": "Hollow Pursuits: unsolicited structural-failure warning auto-paired with Riker's question to Data",
    # "Brothers" -- deck-one life-support evacuation broadcast while the crew clears the
    # Bridge; auto-paired with Riker's order to La Forge.
    "3ee3aa802f29": "Brothers: unsolicited life-support evacuation broadcast auto-paired with Riker's order to La Forge",
    # "Remember Me" -- the second life-support countdown, right after Crusher tells the
    # computer she is not talking to it (the first countdown of the same sequence at
    # line 410 is blanked above).
    "931b5b59bde5": "Remember Me: unsolicited life-support countdown auto-paired with Crusher's aside ('I'm not talking to you.')",
}

# Reason attached when the pairing heuristic itself found no qualifying utterance
# in the window (as opposed to a curated QUERY_BLANKS override).
NO_QUERY_CANDIDATE_REASON = "no qualifying non-computer utterance within the pairing window"


def score_query(candidate: dict, response: dict) -> int:
    score = 0
    gap = response["line_num"] - candidate["line_num"]
    if gap <= 2:
        score += 2
    elif gap <= 4:
        score += 1
    if candidate["scene"] == response["scene"]:
        score += 1
    lower_text = candidate["text"].lower()
    if "computer" in lower_text:
        score += 3
    if candidate["question"]:
        score += 1
    if TRIVIAL_ACK_RE.match(candidate["text"]):
        score -= 4
    return score


def main() -> None:
    dialogue = read_jsonl(DATA_DIR / "dialogue.jsonl")
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in dialogue:
        episodes[row["episode"]].append(row)

    interactions: list[dict] = []

    for episode, rows in episodes.items():
        rows.sort(key=lambda row: row["line_num"])
        index = 0
        while index < len(rows):
            row = rows[index]
            if not row["is_computer"]:
                index += 1
                continue

            cluster = [row]
            next_index = index + 1
            while next_index < len(rows) and rows[next_index]["is_computer"] and rows[next_index]["line_num"] - cluster[-1]["line_num"] <= 2:
                cluster.append(rows[next_index])
                next_index += 1

            query = None
            best_score = -999
            for candidate in reversed(rows[:index]):
                if candidate["is_computer"]:
                    break
                if row["line_num"] - candidate["line_num"] > 4:
                    break
                candidate_score = score_query(candidate, row)
                if candidate_score > best_score:
                    best_score = candidate_score
                    query = candidate
            if best_score < 0:
                query = None

            context_start = max(0, index - 2)
            context_end = min(len(rows), next_index + 1)
            context = rows[context_start:context_end]

            interaction_id = stable_id(episode, row["line_num"], cluster[0]["text"])
            query_speaker = query["speaker"] if query else None
            query_text = query["text"] if query else None
            queryless_reason = None

            repair = QUERY_REPAIRS.get(interaction_id)
            if repair is not None:
                # Replace an auto-paired query that does not drive the response (e.g.
                # the computer opens a scene right after unrelated dialogue from another
                # scene). See QUERY_REPAIRS above.
                query_speaker, query_text = repair
            if interaction_id in QUERY_BLANKS:
                # No utterance elicited this computer line (unsolicited announcement,
                # prompt, or broadcast); represent that explicitly with null query
                # fields and a materialised reason. See QUERY_BLANKS above.
                query_speaker, query_text = None, None
                queryless_reason = QUERY_BLANKS[interaction_id]
            elif query_speaker is None:
                # The pairing heuristic found no qualifying utterance in the window.
                queryless_reason = NO_QUERY_CANDIDATE_REASON

            interaction = {
                "id": interaction_id,
                "episode": episode,
                "episode_number": row["episode_number"],
                "season": row["season"],
                "stardate": row["stardate"],
                "scene": row["scene"],
                "query_speaker": query_speaker,
                "query_text": query_text,
                **(  # only queryless interactions carry a reason
                    {"queryless_reason": queryless_reason}
                    if queryless_reason is not None
                    else {}
                ),
                "response_text": " ".join(item["text"] for item in cluster),
                "response_lines": [item["line_num"] for item in cluster],
                "response_count": len(cluster),
                "context": [
                    {
                        "speaker": item["speaker"],
                        "text": item["text"],
                        "line_num": item["line_num"],
                        "is_computer": item["is_computer"],
                    }
                    for item in context
                ],
            }
            interaction["response_keywords"] = top_tokens([interaction["response_text"]], limit=6)
            if interaction["id"] in NARRATIVE_DEGRADED:
                interaction["narrative_degraded"] = True
                interaction["narrative_degraded_reason"] = NARRATIVE_DEGRADED[interaction["id"]]
            interactions.append(interaction)
            index = next_index

    interactions.sort(key=lambda row: (row["season"], row["episode_number"], row["response_lines"][0]))
    write_json(DATA_DIR / "computer_interactions.json", interactions)

    with_query = sum(1 for row in interactions if row["query_text"])
    narrative_degraded = sum(1 for row in interactions if row.get("narrative_degraded"))
    print(f"Wrote {len(interactions)} computer interactions")
    print(f"Interactions with paired query: {with_query}")
    print(f"Narrative-degraded interactions flagged: {narrative_degraded}")

    expected = set(NARRATIVE_DEGRADED)
    flagged = {row["id"] for row in interactions if row.get("narrative_degraded")}
    if expected != flagged:
        missing = sorted(expected - flagged)
        extra = sorted(flagged - expected)
        raise SystemExit(
            f"Narrative-degraded flag mismatch. Missing: {missing}. Unexpected: {extra}"
        )

    for interaction_id in sorted(QUERY_REPAIRS):
        row = next((row for row in interactions if row["id"] == interaction_id), None)
        if row is None:
            raise SystemExit(f"Query-repair override target not found: {interaction_id}")
        if (row["query_speaker"], row["query_text"]) != QUERY_REPAIRS[interaction_id]:
            raise SystemExit(
                f"Query-repair override not applied for {interaction_id}: got "
                f"({row['query_speaker']!r}, {row['query_text']!r})"
            )

    for interaction_id in sorted(QUERY_BLANKS):
        row = next((row for row in interactions if row["id"] == interaction_id), None)
        if row is None:
            raise SystemExit(f"Query-blank override target not found: {interaction_id}")
        if row["query_speaker"] is not None or row["query_text"] is not None:
            raise SystemExit(
                f"Query-blank override not applied for {interaction_id}: got "
                f"({row['query_speaker']!r}, {row['query_text']!r})"
            )
        if row.get("queryless_reason") != QUERY_BLANKS[interaction_id]:
            raise SystemExit(
                f"Query-blank reason not applied for {interaction_id}: got "
                f"{row.get('queryless_reason')!r}"
            )


if __name__ == "__main__":
    main()
