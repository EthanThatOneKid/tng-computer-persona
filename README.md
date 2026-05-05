# tng-computer-persona

Deterministic preprocessing pipeline for `Star Trek: The Next Generation` dialogue and Enterprise computer interactions.

This repository is the TNG-specific companion to the broader Star Trek computer work. It turns raw episode transcripts into:

- a normalized episode dialogue corpus
- a structured Enterprise computer interaction dataset
- speaker profiles for major TNG voices
- training JSONL for both general TNG persona work and computer-specific tuning

This is now the canonical home for the TNG Enterprise computer persona work that previously lived in `enterprise-computer` and `startrek-computer`.
The dedicated Data persona lives in the sibling repo `tng-data-persona`.
The dedicated Riva persona lives in the sibling repo `tng-riva-persona`.

## Source data

Raw transcripts are vendored as a git submodule:

- `data/raw/star_trek_transcript_search`

The pipeline reads only the `scripts/NextGen/*.txt` transcripts from that source.

## Outputs

- `data/dialogue.jsonl` — every parsed dialogue line in TNG
- `data/episode_index.json` — per-episode dialogue counts
- `data/computer_interactions.json` — query/reply pairs involving the Enterprise computer
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

The earlier `enterprise-computer` repo focused on extracting computer exchanges from TNG. This repo keeps that use case, but expands it into a reusable TNG persona corpus so you can model:

- the Enterprise computer
- Jean-Luc Picard's command voice
- Data's literal precision
- other TNG speaking styles through aggregated speaker profiles

If you are looking for the archived predecessor repo, use `enterprise-computer` or `startrek-computer` only for historical context. This repo is the active one for the shared TNG computer corpus.

The pipeline is deterministic and reproducible from raw transcripts. No API call is required to rebuild the committed dataset.
