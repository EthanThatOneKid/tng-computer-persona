# TNG Persona Dataset Report

- Dialogue rows: 60290
- Computer interactions: 517
- Narrative-degraded interactions (flagged): 30
- Speaker profiles: 723
- Enterprise computer training rows: 321
- Character-conditioned training rows: 59577

## Dialogue by season

| Season | Dialogue lines |
|---|---:|
| 1 | 9471 |
| 2 | 7975 |
| 3 | 8699 |
| 4 | 8655 |
| 5 | 9098 |
| 6 | 8112 |
| 7 | 8280 |

## Most frequent speakers

| Speaker | Lines | Episodes | Avg words | Question rate |
|---|---:|---:|---:|---:|
| PICARD | 11969 | 176 | 11.91 | 0.293 |
| RIKER | 7012 | 176 | 9.84 | 0.3 |
| DATA | 6078 | 173 | 13.23 | 0.13 |
| LAFORGE | 4325 | 168 | 12.9 | 0.199 |
| WORF | 3693 | 173 | 8.91 | 0.126 |
| TROI | 3155 | 163 | 11.38 | 0.26 |
| CRUSHER | 3107 | 152 | 13.18 | 0.232 |
| WESLEY | 1369 | 68 | 9.55 | 0.246 |
| Q | 540 | 8 | 17.42 | 0.293 |
| COMPUTER | 527 | 107 | 7.54 | 0.038 |
| TASHA | 504 | 25 | 10.37 | 0.183 |
| PULASKI | 495 | 20 | 12.1 | 0.218 |

## Main speakers addressing the computer

| Speaker | Interactions |
|---|---:|
| PICARD | 111 |
| LAFORGE | 101 |
| DATA | 89 |
| RIKER | 48 |
| CRUSHER | 45 |
| TROI | 18 |
| WORF | 13 |
| BARCLAY | 10 |
| K'EHLEYR | 8 |
| WESLEY | 7 |
| SCOTT | 4 |
| LYNCH | 3 |

## Narrative-degraded interactions

Interactions where the plot intentionally degrades the computer's capability
(hijack, virus, alien interference, possession, simulated failure). They are
flagged in `computer_interactions.json` and excluded from the golden enterprise
computer training set.

| Episode | Flagged |
|---|---:|
| 100116.txt | 9 |
| 100117.txt | 4 |
| 100127.txt | 2 |
| 100137.txt | 3 |
| 100150.txt | 3 |
| 100233.txt | 4 |
| 100238.txt | 5 |

## Notes

- The enterprise computer subset is best suited for terse operational personas.
- The character-conditioned JSONL keeps a speaker label in metadata so you can filter for Picard, Data, Guinan, or any other voice later.
