# Paper Review: Suphx — Mastering Mahjong with Deep Reinforcement Learning

**Title**: *Suphx: Mastering Mahjong with Deep Reinforcement Learning*  
**Authors**: Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang et al. (Microsoft Research Asia)  
**Publication**: *arXiv* 2003.12777 (2020)

---

## 1. Core Problem Addressed
Mahjong is a 4-player imperfect-information tile game with complex scoring rules, hidden opponent hands (unseen tiles), and long-horizon round-based tournament structures. Traditional RL struggles with:
1. **Severe Information Asymmetry**: Each player only sees their own hand and public discards.
2. **Sparse, Delayed Multi-Round Rewards**: The game is won over multiple sequential rounds.

---

## 2. Key Algorithmic Innovations
1. **Oracle Guiding (Asymmetric Actor-Critic)**:
   - During self-play training, the **Oracle Critic** is granted *privileged full-information access* (seeing all players' hidden tiles and the unseen wall).
   - The **Public Actor** (which only sees legal observations) is trained using value targets from the Oracle Critic and distillation loss. This accelerates feature learning by giving the network perfect supervisory value gradients during training without cheating at inference time.
2. **Global Reward Prediction (GRP)**:
   - Instead of evaluating rounds in isolation, a dedicated GRP neural network predicts final tournament standing points from intermediate round board states, bridging long-horizon credit assignment.
3. **Run-time Policy Adaptation**:
   - Uses Monte Carlo simulations during inference to adapt defensive posturing when opponents enter Riichi (aggressive hand declaration).

---

## 3. Key Takeaways & Relevance for Twilight Struggle
- **Asymmetric Critic (Oracle Training)**: In Twilight Struggle self-play, our C++ engine knows both players' hands and the draw deck. We can train an **Oracle Value Network** (conditioned on true hidden cards) to guide the training of our **Public Policy Network**, completely eliminating noisy credit assignment!
- **Global Reward Prediction**: Bridges multi-turn VP fluctuations to final 10-turn win/loss expectation.
