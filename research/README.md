# Twilight Struggle AI: Research, Literature & Strategic Roadmap

This directory documents foundational literature, game-theoretic algorithms, and architectural designs aimed at building a **super-human artificial intelligence agent** for the Deluxe Edition of **Twilight Struggle**.

---

## 1. Central Evolving Document

- [**ideas_and_plans.md**](ideas_and_plans.md): **Ideas, Strategic Insights & Technical Roadmap**  
  *The single, evolving design document covering domain insights (hand scheduling, DEFCON trapping, asymmetric war phase access), tabula rasa learning without demonstrations, ColdWarNetV4 architecture with Oracle Guiding, Multi-Agent League training, and the two-level NashPG + $\pi\text{KL}$ Subgame Resolving duality.*

---

## 2. Literature Review: Reviewed Papers Index

| Paper Review Document | Authors / Lab | Publication | Core Concept & Innovation | Relevance to Twilight Struggle |
|:---|:---|:---:|:---|:---|
| [**paper_nash_policy_gradient.md**](paper_nash_policy_gradient.md) | Recent MARL Research | *OpenReview / arXiv* (2024/2025) | **Nash Policy Gradient (NashPG)**: Iteratively refined reference regularization in objective space. | Direct successor to DeepNash; guarantees monotonic Bregman convergence to Nash equilibrium using fixed $\eta$. Core training engine in [`ai/training/nash_pg.py`](../ai/training/nash_pg.py). |
| [**paper_kl_regularized_search.md**](paper_kl_regularized_search.md) | Austin, Bakhtin, Brown et al. (Meta AI) | *ICML* (2022) | **KL-Regularized Search ($\pi\text{KL}$-Hedge)**: Decision-time planning regularized toward an anchor policy prior. | Mathematical foundation for real-time subgame search anchored to the NashPG policy prior. |
| [**paper_deepnash_rnad.md**](paper_deepnash_rnad.md) | Pérolat et al. (DeepMind) | *Science* (2022) | **Regularized Nash Dynamics (R-NaD)**: Model-free convergence to Nash equilibrium in imperfect-information board games without search. | Foundational ancestor of iterative reference regularization. |
| [**paper_cicero_diplomacy.md**](paper_cicero_diplomacy.md) | Bakhtin et al. (Meta AI) | *Science* (2022) | **CICERO ($\pi\text{KL}$ Planning)**: Policy-regularized planning and iterative subgame solving in geopolitical strategy. | Proves that regularized iterative subgame planning produces unexploitable, robust decision-making. |
| [**paper_suphx_mahjong.md**](paper_suphx_mahjong.md) | Li et al. (Microsoft Research Asia) | *arXiv* (2020) | **Suphx (Oracle Guiding & GRP)**: Asymmetric actor-critic training where privileged Oracle critic guides public actor. | Blueprint for using privileged full-information critic during C++ self-play training. |
| [**paper_student_of_games.md**](paper_student_of_games.md) | Schmid et al. (DeepMind) | *Science Advances* (2023) | **Student of Games**: Unified sound search and learning for imperfect-information games. | Framework for turn-level (AR1–AR7) subgame resolving in Twilight Struggle. |
| [**paper_rebel.md**](paper_rebel.md) | Brown et al. (Meta AI) | *NeurIPS* (2020) | **Public Belief States (PBS)**: Converts imperfect-information games into continuous MDPs over belief states. | Foundation for our auxiliary opponent hand card-counting head $\beta_{\text{opp}} \in [0, 1]^{110}$. |
| [**paper_magnetic_mirror_descent.md**](paper_magnetic_mirror_descent.md) | Sokota et al. (ICML) | *ICML* (2023) | **Magnetic Mirror Descent (MMD)**: Proximal regularization to a magnetic anchor with MMD search. | Provides theoretical stability and online search algorithms for non-public information settings. |
| [**paper_off_belief_learning.md**](paper_off_belief_learning.md) | Hu, Foerster et al. | *ICML / NeurIPS* (2021) | **Off-Belief Learning (OBL)**: Prevents brittle self-play conventions by grounding beliefs in base dynamics. | Ensures our bot does not develop fragile self-play artifacts against humans. |
| [**paper_alphastar.md**](paper_alphastar.md) | Vinyals et al. (DeepMind) | *Nature* (2019) | **League Training**: Main Agent + Exploiters + Historical Pool. | Ecosystem for training dedicated DEFCON Trap Exploiters and preventing strategy cycling. |
| [**paper_douzero_perfectdou.md**](paper_douzero_perfectdou.md) | Zha et al. / Guan et al. | *ICML / NeurIPS* (2021/2022) | **Card Set Encodings & Hand Planning**: Deep RL for complex multi-round trick-taking and card retention. | Paradigms for multi-action hand scheduling and card combination self-attention. |
