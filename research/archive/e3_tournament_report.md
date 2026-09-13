# Twilight Struggle AI: Massive Tournament Evaluation Report

- **Total Models**: 5
- **Total Games Played**: 10,000
- **Games Per Matchup Pair**: 1,000 (500 per side)
- **Total Evaluation Time**: 36.2 seconds (276.2 games/sec)

## 1. Bradley-Terry MLE Elo Leaderboard

| Rank | Model | Elo Rating | Total Matches | Total Record (W-L-D) | Overall Win Rate |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **e3_synth_final** | **1852.9** | 4,000 | 2,980W - 1,015L - 5D | **74.5%** |
| **2** | **dec_turns40_final** | **1824.7** | 4,000 | 2,841W - 1,155L - 4D | **71.0%** |
| **3** | **e3_human_final** | **1810.3** | 4,000 | 2,768W - 1,227L - 5D | **69.2%** |
| **4** | **HeuristicBot** | **1500.0** | 4,000 | 1,365W - 2,635L - 0D | **34.1%** |
| **5** | **RandomBot** | **888.6** | 4,000 | 39W - 3,961L - 0D | **1.0%** |

---

## 2. Head-to-Head Total Win Rate Matrix (% Win for Row vs Column)

| Model | **e3_synth_final** | **dec_turns40_final** | **e3_human_final** | **HeuristicBot** | **RandomBot** |
|:---|---:|---:|---:|---:|---:|
| **e3_synth_final** | — | 56.8% | 57.4% | 84.1% | 99.7% |
| **dec_turns40_final** | 43.0% | — | 50.6% | 90.5% | 100.0% |
| **e3_human_final** | 42.3% | 49.2% | — | 85.8% | 99.5% |
| **HeuristicBot** | 15.9% | 9.5% | 14.2% | — | 96.9% |
| **RandomBot** | 0.3% | 0.0% | 0.5% | 3.1% | — |

---

