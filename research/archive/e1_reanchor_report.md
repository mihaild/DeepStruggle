# Twilight Struggle AI: Massive Tournament Evaluation Report

- **Total Models**: 4
- **Total Games Played**: 6,000
- **Games Per Matchup Pair**: 1,000 (500 per side)
- **Total Evaluation Time**: 21.0 seconds (286.4 games/sec)

## 1. Bradley-Terry MLE Elo Leaderboard

| Rank | Model | Elo Rating | Total Matches | Total Record (W-L-D) | Overall Win Rate |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **dec_turns40_final** | **1866.2** | 3,000 | 2,434W - 562L - 4D | **81.1%** |
| **2** | **sp2_pool_off_final** | **1833.3** | 3,000 | 2,319W - 677L - 4D | **77.3%** |
| **3** | **HeuristicBot** | **1500.0** | 3,000 | 1,205W - 1,795L - 0D | **40.2%** |
| **4** | **RandomBot** | **908.4** | 3,000 | 38W - 2,962L - 0D | **1.3%** |

---

## 2. Head-to-Head Total Win Rate Matrix (% Win for Row vs Column)

| Model | **dec_turns40_final** | **sp2_pool_off_final** | **HeuristicBot** | **RandomBot** |
|:---|---:|---:|---:|---:|
| **dec_turns40_final** | — | 53.6% | 90.3% | 99.5% |
| **sp2_pool_off_final** | 46.0% | — | 86.2% | 99.7% |
| **HeuristicBot** | 9.7% | 13.8% | — | 97.0% |
| **RandomBot** | 0.5% | 0.3% | 3.0% | — |

---

