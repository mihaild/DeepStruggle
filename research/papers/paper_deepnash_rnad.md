# Paper Review: DeepNash — Regularized Nash Dynamics (R-NaD)

**Title**: *Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning*  
**Authors**: Julien Pérolat, Bart De Vylder, Dharshan Kumaran, Julien Perolat, Karl Tuyls et al. (Google DeepMind)  
**Publication**: *Science* 378, 972–980 (December 2022)

---

## 1. Core Problem Addressed
In large imperfect-information games (Stratego with $10^{535}$ states, Poker, Twilight Struggle):
- **Strategy Cycling**: Standard policy gradient methods (PPO, REINFORCE) cycle endlessly in competitive zero-sum settings instead of converging to a stable equilibrium.
- **Search Limitations**: Standard tree search without belief-state consistency is theoretically unsound because optimal decisions depend on hidden piece/card distributions and unreached counterfactual branches.

---

## 2. Algorithmic Innovation: R-NaD
DeepNash introduced **Regularized Nash Dynamics (R-NaD)**, a model-free reinforcement learning algorithm that guarantees last-iterate convergence to an approximate $\epsilon$-Nash equilibrium without game-tree search.

### Mathematical Formulation
At outer loop $k$, given a frozen reference policy $\pi_{\text{ref}}^{(k)}$, transition rewards are transformed along trajectories:
$$\tilde{r}_t = r_t - \eta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}^{(k)}(a_t \mid s_t)}$$

Under continuous-time Lyapunov analysis, this regularized payoff creates a strictly concave-convex game with a unique equilibrium. By updating the reference anchor $\pi_{\text{ref}}^{(k+1)} \leftarrow \pi_\theta^{(k)}$ periodically, the trajectory converges monotonically to the unregularized Nash equilibrium.

---

## 3. Key Takeaways & Relevance for Twilight Struggle
1. **Model-Free Equilibrium Convergence**: Proved that deep neural networks can learn grandmaster-level imperfect-information play purely through regularized self-play without search.
2. **Inspiration for NashPG**: Provided the theoretical foundation for outer-loop reference policy anchoring, which NashPG later adapted into an objective-space formulation for standard policy gradient training.
