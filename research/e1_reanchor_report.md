# Twilight Struggle AI: Massive Tournament Evaluation Report

- **Total Models**: 4
- **Total Games Played**: 6,000
- **Games Per Matchup Pair**: 1,000 (500 per side)
- **Total Evaluation Time**: 20.1 seconds (298.9 games/sec)

## 1. Bradley-Terry MLE Elo Leaderboard

| Rank | Model | Elo Rating | Total Matches | Total Record (W-L-D) | Overall Win Rate |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **dec_turns40_final** | **1880.4** | 3,000 | 2,459W - 539L - 2D | **82.0%** |
| **2** | **sp2_pool_off_final** | **1836.5** | 3,000 | 2,307W - 691L - 2D | **76.9%** |
| **3** | **HeuristicBot** | **1500.0** | 3,000 | 1,197W - 1,803L - 0D | **39.9%** |
| **4** | **RandomBot** | **898.9** | 3,000 | 35W - 2,965L - 0D | **1.2%** |

---

## 2. Head-to-Head Total Win Rate Matrix (% Win for Row vs Column)

| Model | **dec_turns40_final** | **sp2_pool_off_final** | **HeuristicBot** | **RandomBot** |
|:---|---:|---:|---:|---:|
| **dec_turns40_final** | — | 56.0% | 90.2% | 99.7% |
| **sp2_pool_off_final** | 43.8% | — | 87.2% | 99.7% |
| **HeuristicBot** | 9.8% | 12.8% | — | 97.1% |
| **RandomBot** | 0.3% | 0.3% | 2.9% | — |

---

