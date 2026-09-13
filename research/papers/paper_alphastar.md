# Paper Review: AlphaStar — Grandmaster Imperfect-Information Multi-Agent RL

**Title**: *Grandmaster level in StarCraft II using multi-agent reinforcement learning*  
**Authors**: Oriol Vinyals, Igor Babuschkin, Wojciech M. Czarnecki et al. (Google DeepMind)  
**Publication**: *Nature* 575, 350–354 (2019)

---

## 1. Core Problem Addressed
Real-time strategy and complex imperfect-information games feature vast action spaces, hidden fog of war, and complex multi-player rock-paper-scissors strategic cycles that cause standard self-play to get stuck in local optima.

---

## 2. Algorithmic Innovation: League Training & Attention Architectures
1. **The League Ecosystem**:
   - **Main Agents**: Prioritize unexploitability and general competence against all past and current strategies.
   - **Main Exploiters**: Explicitly trained to find and exploit weaknesses in the Main Agents.
   - **League Exploiters**: Trained to exploit entire populations and find blind spots across the entire league.
   - **Historical Frozen Snapshots**: Prevent catastrophic forgetting of legacy strategies.
2. **Transformer & Spatial Encoders**: Sequence models and spatial transformers processing long history and imperfect-information macro-strategy.

---

## 3. Key Takeaways & Relevance for Twilight Struggle
- Direct inspiration for our proposed **Twilight Struggle Multi-Agent League**: training a Main Agent alongside dedicated **DEFCON Trap Exploiters** (incentivized to find and trigger opponent suicide traps) and **Territory Rush Exploiters**.
