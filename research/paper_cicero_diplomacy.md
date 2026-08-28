# Paper Review: CICERO — Human-Level Strategic Planning in Diplomacy

**Title**: *Human-level play in the game of Diplomacy combining language models with strategic reasoning*  
**Authors**: Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty et al. (Meta AI Research / FAIR)  
**Publication**: *Science* 378, 1067–1074 (December 2022)

---

## 1. Core Problem Addressed
*Diplomacy* is a 7-player zero-sum board game with simultaneous moves, imperfect information, and natural-language negotiation. Standard game-theoretic algorithms fail because:
1. Pure self-play produces idiosyncratic policies that do not coordinate with or understand human opponents.
2. The simultaneous-move, long-horizon nature of the game makes standard Minimax / MCTS inapplicable.

---

## 2. Key Algorithmic Innovations
1. **$\pi\text{KL}$ (Policy-Regularized Planning)**:
   - Rather than computing an unconstrained Nash equilibrium (which often results in paranoid, non-cooperative strategies), CICERO regularizes the planning objective toward a human behavioral anchor policy $\pi_{\text{human}}$:
     $$\max_{\pi_i} \mathbb{E} [ u_i(\pi_i, \pi_{-i}) ] - \tau D_{\text{KL}}(\pi_i \parallel \pi_{\text{human}})$$
2. **Iterative Subgame Planning**:
   - At each turn, CICERO runs an iterative search: (1) predicts opponent intentions conditioned on dialogue, (2) computes regularized best-response order profiles, and (3) re-evaluates board control across candidate outcomes.

---

## 3. Key Takeaways & Relevance for Twilight Struggle
- Demonstrates the power of **Anchor-Regularized Planning ($\pi\text{KL}$)** in long-horizon geopolitical board games.
- Provides proof that iterative subgame planning regularized against a reference policy produces unexploitable, robust strategic decision-making.
