# Paper Review: Nash Policy Gradient (NashPG)

**Title**: *Nash Policy Gradient: Iteratively Refined Regularization for Two-Player Zero-Sum Games*  
**Authors**: Recent Multi-Agent RL Research  
**Publication**: *arXiv / OpenReview* (2024 / 2025)

---

## 1. Core Problem & Motivation: Improving upon R-NaD
While DeepMind's R-NaD established convergence guarantees, it suffered from practical integration hurdles:
1. **Trajectory Reward Modification**: R-NaD applies regularization directly to the transition reward ($\tilde{r}_t = r_t - \eta \log \frac{\pi}{\pi_{\text{ref}}}$), requiring specialized trajectory evaluation (NeuRD, V-Trace) and tight coupling to environment rewards.
2. **Regularization Annealing Issues**: Classical regularized game theory required annealing $\eta \to 0$ to reach unregularized Nash equilibrium, causing training instability as regularization vanishes.

---

## 2. Algorithmic Innovation: NashPG
**Nash Policy Gradient (NashPG)** solves these issues by:
1. **Fixed Large Regularization ($\eta > 0$)**: Keeps $\eta$ fixed at a stable constant.
2. **Objective-Space KL Penalty**: Moves the Bregman / KL penalty directly into the policy optimization objective:
   $$\mathcal{L}_{\text{NashPG}}(\theta) = \mathcal{L}_{\text{PPO}}(\theta) + \eta \cdot \mathbb{E} \left[ D_{\text{KL}}\left(\pi_\theta(\cdot \mid s) \parallel \pi_{\text{ref}}^{(k)}(\cdot \mid s)\right) \right] - c_{\text{ent}} \mathcal{H}(\pi_\theta)$$
3. **Monotonic Bregman Convergence**: Proves that periodically updating $\pi_{\text{ref}}^{(k+1)} \leftarrow \pi_\theta^{(k)}$ guarantees strictly monotonic reduction in Bregman divergence to the true unregularized Nash equilibrium:
   $$D_{\text{Bregman}}(\pi_\theta^{(k+1)}, \Pi^*) < D_{\text{Bregman}}(\pi_\theta^{(k)}, \Pi^*)$$

---

## 3. Implementation in Our Codebase
NashPG serves as the core training engine in [`ai/training/nash_pg.py`](file:///home/mihaild/prog/ts_ai/ai/training/nash_pg.py):
- Maintains the frozen reference anchor network $\pi_{\text{ref}}^{(k)}$.
- Evaluates legal-action KL divergence over batch rollouts.
- Integrates seamlessly with GAE, PPO clipping, and multi-temperature exploration.
