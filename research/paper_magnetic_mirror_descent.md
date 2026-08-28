# Paper Review: Magnetic Mirror Descent (MMD)

**Title**: *Magnetic Mirror Descent for Imperfect-Information Games*  
**Authors**: Samuel Sokota, Ryan D'Orazio, J. Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, Christian Kroer  
**Publication**: *ICML* 2023 / *arXiv* 2306.00287

---

## 1. Core Problem Addressed
Standard first-order optimization (gradient descent / mirror descent) in zero-sum imperfect-information games exhibits oscillatory, cyclical dynamics. While CFR converges in tabular settings, it is difficult to scale efficiently to deep neural networks without large memory overhead.

---

## 2. Algorithmic Innovation: Magnetic Mirror Descent
1. **Proximal Regularization to a Magnet**:
   - MMD introduces a stationary "magnetic anchor" into the mirror descent update rule, creating a strong restoring force that dampens cyclical best-response oscillations.
   - Provides **last-iterate convergence** in extensive-form zero-sum games.
2. **MMD Search (Decision-Time Planning)**:
   - Formulates an online search algorithm that plans directly over information sets without requiring full public-belief-state conversion, making decision-time planning effective in non-public information settings.

---

## 3. Key Takeaways & Relevance for Twilight Struggle
- Provides theoretical and empirical validation for why anchor-regularized methods (like NashPG and MMD) achieve stable, monotonic convergence in deep multi-agent reinforcement learning.
- Informs fast decision-time planning over candidate hand allocations in Twilight Struggle.
