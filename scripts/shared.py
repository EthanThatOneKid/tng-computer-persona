from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "raw" / "star_trek_transcript_search" / "scripts" / "NextGen"

SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 '.-]*):\s*(.+)$")
SCENE_RE = re.compile(r"^\[(.+)\]$")
STARDATE_RE = re.compile(r"Stardate\s+(\d+\.?\d*)", re.IGNORECASE)
COMPUTER_RE = re.compile(r"^(COMPUTER|COMPUTER VOICE|SHIP'S COMPUTER)$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9']+")

SEASON_LENGTHS = {
    1: 26,
    2: 22,
    3: 26,
    4: 26,
    5: 26,
    6: 26,
    7: 24,
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have",
    "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "so",
    "that", "the", "their", "them", "there", "this", "to", "we", "what",
    "with", "you", "your",
}


def iter_episode_paths() -> list[Path]:
    return sorted(TRANSCRIPTS_DIR.glob("*.txt"))


def get_episode_number(path: Path) -> int:
    digits = re.sub(r"\D", "", path.stem)
    if not digits:
        return 0
    if len(digits) >= 3:
        return int(digits[-3:])
    return int(digits)


@lru_cache(maxsize=1)
def season_by_episode_name() -> dict[str, int]:
    mapping: dict[str, int] = {}
    ordered_paths = iter_episode_paths()
    offset = 0
    for season, count in SEASON_LENGTHS.items():
        for path in ordered_paths[offset:offset + count]:
            mapping[path.name] = season
        offset += count
    return mapping


def get_season(path: Path) -> int:
    return season_by_episode_name().get(path.name, 0)


def extract_stardate(text: str) -> str:
    match = STARDATE_RE.search(text[:2000])
    return match.group(1) if match else ""


def is_computer_speaker(speaker: str) -> bool:
    return bool(COMPUTER_RE.match(speaker.strip()))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path) -> object:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def parse_episode(path: Path) -> tuple[list[dict], dict]:
    # Pin UTF-8 explicitly: the transcripts contain UTF-8 bytes (e.g. U+FFFD for
    # source-corrupted characters) that platform-default decoding (cp1252 on
    # Windows) would mangle, breaking reproducibility across machines.
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    episode_number = get_episode_number(path)
    season = get_season(path)
    stardate = extract_stardate(text)
    scene = ""
    records: list[dict] = []

    for line_num, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        scene_match = SCENE_RE.match(stripped)
        if scene_match:
            scene = scene_match.group(1)
            continue
        speaker_match = SPEAKER_RE.match(stripped)
        if not speaker_match:
            continue

        speaker = normalize_space(speaker_match.group(1))
        text_value = normalize_space(speaker_match.group(2))
        record = {
            "id": stable_id(path.name, line_num, speaker, text_value),
            "series": "NextGen",
            "series_title": "Star Trek: The Next Generation",
            "season": season,
            "episode": path.name,
            "episode_number": episode_number,
            "stardate": stardate,
            "scene": scene,
            "line_num": line_num,
            "speaker": speaker,
            "text": text_value,
            "is_computer": is_computer_speaker(speaker),
            "word_count": len(tokenize(text_value)),
            "question": "?" in text_value,
        }
        records.append(record)

    episode_summary = {
        "episode": path.name,
        "episode_number": episode_number,
        "season": season,
        "stardate": stardate,
        "dialogue_lines": len(records),
        "computer_lines": sum(1 for record in records if record["is_computer"]),
        "speakers": sorted({record["speaker"] for record in records}),
    }
    return records, episode_summary


def top_tokens(texts: list[str], limit: int = 10) -> list[str]:
    counter = Counter()
    for text in texts:
        for token in tokenize(text):
            if token not in STOPWORDS and len(token) > 2:
                counter[token] += 1
    return [token for token, _ in counter.most_common(limit)]
