# tng-computer-persona

Deterministic preprocessing pipeline for `Star Trek: The Next Generation` dialogue and TNG computer interactions.

This repository turns raw episode transcripts into:

- a normalized episode dialogue corpus
- a structured TNG computer interaction dataset
- speaker profiles for major TNG voices
- training JSONL for both general TNG persona work and computer-specific tuning

This is now the canonical home for the current TNG computer persona work.

## Source data

Raw transcripts are vendored as a git submodule:

- `data/raw/star_trek_transcript_search`

The pipeline reads only the `scripts/NextGen/*.txt` transcripts from that source.

## Outputs

- `data/dialogue.jsonl` — every parsed dialogue line in TNG
- `data/episode_index.json` — per-episode dialogue counts
- `data/computer_interactions.json` — query/reply pairs involving the TNG computer
- `data/speaker_profiles.json` — aggregate voice stats by speaker
- `data/enterprise_computer_train.jsonl` — user/computer training pairs
- `data/tng_character_train.jsonl` — character-conditioned training records
- `docs/DATASET_REPORT.md` — generated summary report
- `docs/persona-notes.md` — practical prompt guidance for Zo personas

## Quick start

```bash
git clone --recurse-submodules https://github.com/EthanThatOneKid/tng-computer-persona.git
cd tng-computer-persona
python -m scripts.run_pipeline
```

If the repo is already cloned:

```bash
git submodule update --init
python -m scripts.run_pipeline
```

## Pipeline

```text
raw TNG transcripts
  -> scripts.export_dialogue
  -> scripts.extract_computer_interactions
  -> scripts.profile_speakers
  -> scripts.build_training_jsonl
  -> scripts.analyze
```

## Why this repo exists

This repo keeps the TNG computer extraction workflow, but expands it into a reusable TNG persona corpus so you can model:

- the TNG computer
- Jean-Luc Picard's command voice
- Data's literal precision
- other TNG speaking styles through aggregated speaker profiles

The pipeline is deterministic and reproducible from raw transcripts. No API call is required to rebuild the committed dataset.
