# Paper Review: Student of Games (SoG)

**Title**: *Student of Games: A unified learning algorithm for both perfect and imperfect information games*  
**Authors**: Martin Schmid, Matej Moravcik, Neil Burch, Rudolf Kadlec, Michael Bowling et al. (Google DeepMind)  
**Publication**: *Science Advances* 9, eadg3256 (November 2023)

---

## 1. Core Problem Addressed
Historically, game AI algorithms were split into two disjoint paradigms:
- **Perfect Information (AlphaZero / Minimax)**: Tree search + heuristic value estimation.
- **Imperfect Information (CFR / Libratus)**: Global matrix solving or depth-limited subgame solving without unified search representations.

---

## 2. Algorithmic Innovation: Unified Sound Search & Learning
Student of Games introduces a single algorithm capable of mastering both perfect-information games (Chess, Go) and imperfect-information games (Poker, Scotland Yard) by integrating:
1. **Growing-Tree Search**: Expands search trees incrementally in both public and private information sets.
2. **Continual Resolving**: Solves subgames during real-time play using value and policy networks learned through self-play.
3. **Soundness Guarantee**: Proves that the online search procedure is *game-theoretically sound*—guaranteed to converge to an approximate Nash equilibrium as computation increases.

---

## 3. Key Takeaways & Relevance for Twilight Struggle
- Provides the theoretical framework for **Turn-Level Subgame Resolving** in Twilight Struggle: solving the localized 7-AR subgame of the current turn using public board states and leaf value evaluation from neural networks.
