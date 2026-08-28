# Paper Review: DouZero & PerfectDou — Card Play & Hand Scheduling

**Titles**: 
- *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning* (Zha et al., ICML 2021)
- *PerfectDou: Dominating DouDizhu with Perfect-Information Monte Carlo and Deep RL* (Guan et al., NeurIPS 2022)

---

## 1. Core Problem Addressed
Trick-taking and multi-round card games require complex multi-step combinatorial hand planning, card retention, and cooperative/adversarial card-counting across rounds.

---

## 2. Key Techniques & Innovations
1. **Matrix / Set Hand Encodings**: Treats player hands as structured card sets rather than flat categorical features, processing combinations using self-attention.
2. **Deep Value Functions for Card Sequences**: Accurately estimates intermediate game states during multi-action card play rounds.
3. **Card Counting Features**: Explicit historical card tracking tensors encoding played, held, and unrevealed cards.

---

## 3. Key Takeaways & Relevance for Twilight Struggle
- Motivates the **Card Transformer Backbone** in `ColdWarNetV4`: encoding all 110 cards through self-attention to capture full hand synergy, Space Race dumping candidates, and dangerous opponent event combinations.
