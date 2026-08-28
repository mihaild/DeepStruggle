# Paper Review: ReBeL — Recursive Belief-based Learning

**Title**: *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games*  
**Authors**: Noam Brown, Anton Bakhtin, Adam Lerer, Qucheng Gong (Meta AI Research / FAIR)  
**Publication**: *NeurIPS* 33 (2020)

---

## 1. Core Problem Addressed
In imperfect-information games, standard reinforcement learning algorithms (like AlphaZero) cannot be directly applied because the value of an action depends on the probability distribution over hidden opponent states.

---

## 2. Algorithmic Innovation: Public Belief States (PBS)
ReBeL transforms imperfect-information games into continuous Markov Decision Processes over **Public Belief States (PBS)**:

$$S_{\text{PBS}} = (S_{\text{public}}, \beta_1, \beta_2)$$

where $S_{\text{public}}$ is the common-knowledge public history and $\beta_i$ is the probability distribution over player $i$'s private hands.

### Key Mechanisms:
1. **Depth-Limited Subgame Solving**: At test time, ReBeL runs search on the PBS representation, evaluating leaf nodes with a learned value network $V(S_{\text{PBS}})$.
2. **Bayesian Continual Resolving**: As actions occur, beliefs are updated via Bayes' rule: $\beta(h \mid a) \propto \beta(h) \cdot \pi(a \mid h)$.
3. **Superhuman Efficiency**: Achieved superhuman performance in Heads-Up No-Limit Texas Hold'em with orders of magnitude less domain knowledge than previous poker bots.

---

## 3. Key Takeaways & Relevance for Twilight Struggle
- Formalizes the representation for Twilight Struggle: the board and public discard/removed piles form $S_{\text{public}}$, while an auxiliary head models the opponent hand probability distribution $\beta_{\text{opp}} \in [0, 1]^{110}$.
