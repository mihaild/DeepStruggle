# Twilight Struggle AI: Ideas, Strategic Insights & Technical Roadmap

This is the central, evolving document tracking domain insights, algorithmic ideas, architectural designs, and implementation plans for building a super-human Twilight Struggle agent.

---

## 1. Domain & Game-Theoretic Insights

### 1.1 The Strategic Hand Scheduling Problem (Why 1-Ply Masks Fail)
Twilight Struggle is not won by 1-step reactive play. At the start of every turn (drawing 8–9 cards), an expert player evaluates an implicit **Constraint Scheduling Graph**:
- **Headline**: High-initiative tempo play or safe dump.
- **AR1**: Battleground coup or urgent defensive access.
- **AR2–AR5**: Board expansion, stability reinforcement, and scoring play.
- **Space Race Slot**: Discarding the single most dangerous unplayable opponent event.
- **Carryover Slot**: Retaining 1 high-ops or safe event card into the next turn.

**The AR7 Danger**: If a player holds 2 deadly opponent suicide events (*CIA Created*, *Duck & Cover* at DEFCON 2) and has only 1 Space Race outlet, playing 6 safe cards leaves only suicide cards on AR7. A 1-ply mask has zero legal moves on AR7 and collapses. The decision to space or neutralize the suicide card must be scheduled at Turn Start.

### 1.2 DEFCON Trapping & Opponent Constriction
Super-human play is not just about avoiding self-suicide—it is about **actively weaponizing DEFCON against the opponent**:
- When holding safe cards while the opponent is suspected of holding unplayable cards, keep DEFCON strictly at 2.
- Deny safe non-battleground coup outlets and space race avenues across AR4–AR6.
- Constrict the opponent's options until they are forced to play their suicide event on AR7.

### 1.3 Asymmetric Regional Timing & Why Full-Game Training is Mandatory
Artificially truncating training to Early War (Turns 1–3) produces catastrophic strategic myopia:
- **USSR Mid War Access Imperative**: In Early War, USSR *must* fight aggressively for access into Africa and Latin America (via *Decolonization*, *Destalinization*, or early access coups) because Mid War adds scoring cards for Central America, South America, and Africa. Once US establishes presence, USSR has almost no natural geographic entry points.
- **US Asymmetric Geography**: The US has direct geographic adjacency to Central and South America (via Mexico and Panama) and powerful Late War events, meaning US does not need to rush early presence with the same urgency as USSR.
- **Design Invariant**: Training must always operate over **full 10-turn games**, allowing the learned value network $V(s)$ to naturally price the future strategic value of Early War access for Mid War scoring.

---

## 2. Tabula Rasa Learning Without Demonstrations

Because we do not have human Grandmaster game transcripts, and demonstrations from weak bots only clone bad habits, the agent must master board development **completely tabula rasa from scratch**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TABULA RASA LEARNING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Mathematical Potential Shaping Φ(s) (Ng et al. 1999):                    │
│    • Objective scoring formula evaluated on state s:                        │
│      - Regional Presence (+1), Domination (+3 to +7), Control (+5 to +10).  │
│      - Battleground control status and stability defense buffer.            │
│      - Inter-regional access connectivity.                                  │
│    • Ng et al. Theorem: Preserves the exact true Nash equilibrium π*.       │
│                                                                             │
│ 2. Asymmetric Actor-Critic (Oracle Guiding - Suphx 2020):                   │
│    • During C++ self-play training, the Critic sees true hidden cards       │
│      (Oracle Critic), providing low-variance value targets to guide the     │
│      Public Policy Network without cheating at inference time.              │
│                                                                             │
│ 3. Smooth Continuous Annealing Schedule:                                    │
│    • R_t = (1 - α_t) · [γ Φ(s') - Φ(s)] + α_t · R_GameOutcome              │
│    • Directs initial random exploration toward coherent spatial expansion,  │
│      then smoothly fades out, letting NashPG optimize pure game victory.    │
│                                                                             │
│ 4. Forward-Wave Value Bootstrapping:                                        │
│    • Value network first masters Turn 1 opening structures.                 │
│    • Early War competence naturally stabilizes the starting state of Mid    │
│      War, propagating sound board development forward through time.         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture Specification: `ColdWarNetV4`

```
Inputs:
  1. Spatial Graph Tensor (84 countries × 32 features)
  2. Card Set Tensor (110 cards × 16 status/location features)
  3. Action History Sequence (Last 32 MicroActions, Embed Dim 128)
  4. Global Scalars (DEFCON, VP, MilOps, Space, Turn, Action Round, Phase)

Backbone:
  ├── Spatial GNN: 3-layer Graph Convolution over map topology -> (84 × 128)
  ├── Card Transformer: 4-layer Self-Attention over 110 card tokens -> (110 × 128)
  ├── Causal History Transformer: 2-layer Sequence Encoder over actions -> (32 × 128)
  └── Cross-Attention Fusion: Cross-attends Board <-> Cards <-> History -> Latent (512-dim)

Output Heads:
  ├── Policy Head π(a | s): 212-dim Flat Action Logits (Masked Softmax)
  ├── Categorical Value Head V_cat(s): 51-Atom Distribution over [-20, +20] VP & End Types
  ├── Opponent Belief Head β_opp(s): 110-dim Multi-Label Sigmoid (P(Card c in Opp Hand))
  ├── Oracle Critic V_oracle(s_full): Privileged value head for training distillation
  └── Regularized Value Head V_reg(s): R-NaD / NashPG regularized value scalar
```

---

## 4. Multi-Agent League Training System (AlphaStar Style)

To prevent self-play cycling and build unexploitable play against all strategies:

```
┌────────────────────────────────────────────────────────────────────────┐
│                       LEAGUE TRAINING ECOSYSTEM                        │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Main Agent (60% training):                                          │
│    - Trains via NashPG against all League agents and past snapshots.  │
│    - Objective: Unexploitability and global Nash equilibrium.          │
│                                                                        │
│ 2. DEFCON Trap Exploiter (20% training):                               │
│    - Specialized in DEFCON 2 traps, hand constriction, and AR7 forcing.│
│    - Rewards earned by provoking DEFCON suicides on Main Agent.        │
│                                                                        │
│ 3. Territory / Space Rush Exploiter (10% training):                    │
│    - Focuses on board domination rushes and Space Race victory.        │
│                                                                        │
│ 4. Historical Snapshot Pool (10% training):                            │
│    - 50+ past checkpoints to guarantee zero catastrophic forgetting.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Test-Time Search: $\pi\text{KL}$ Subgame Resolving & Interaction with NashPG

### 5.1 The Mathematical Duality: NashPG (Offline Training) $\leftrightarrow$ $\pi\text{KL}$ (Online Search)

There is a profound mathematical symmetry between **NashPG** and **$\pi\text{KL}$ Subgame Resolving**: both are Mirror Descent algorithms with KL regularization against a reference anchor:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           THE TWO-LEVEL MIRROR DESCENT ARCHITECTURE                         │
├───────────────────────────────────────┬─────────────────────────────────────────────────────┤
│ LEVEL 1: GLOBAL TRAINING (NashPG)     │ LEVEL 2: LOCAL INFERENCE (πKL Subgame Resolving)    │
├───────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ • Objective:                          │ • Objective:                                        │
│   max_θ E[A] - η D_KL(π_θ || π_ref)   │   max_σ E[Q] - τ D_KL(σ || π_θ)                     │
│                                       │                                                     │
│ • Horizon: Global (Full 10-turn game) │ • Horizon: Local (Current Turn AR1–AR7)             │
│ • Reference Anchor: π_ref^(k)         │ • Reference Anchor: Learned Neural Policy π_θ       │
│ • Update Frequency: Every 200k steps  │ • Execution: Real-time in C++ (<5 ms, 50 iterations)│
│ • Role: Macro-strategy, card valuation│ • Role: Tactical search, eliminates AR7 traps       │
└───────────────────────────────────────┴─────────────────────────────────────────────────────┘
```

### 5.2 The "Search-in-the-Loop" NashPG Flywheel
During self-play training:
1. **Search Generates Trajectories**: Instead of acting on raw $\pi_\theta$, environments act on the improved search policy $\sigma = \pi\text{KL}\text{-Search}(\pi_\theta, V_\theta)$.
2. **NashPG Distills Search**: The policy network $\pi_\theta$ is updated toward $\sigma$.
3. **Continuous Elevation**: The search policy $\sigma$ is strictly stronger than $\pi_\theta$, meaning the neural network is constantly learning from a superior version of itself.

---

## 6. Implementation Roadmap

| Phase | Milestone / Action Item | Target Metric / Deliverable | Status |
|:---:|:---|:---|:---:|
| **Phase 1** | **NashPG Codebase Unification** | Single source of truth in `ai/training/nash_pg.py` | **COMPLETE** |
| **Phase 2** | **Research Literature Reviews** | Standalone paper reviews + evolving ideas document in `research/` | **COMPLETE** |
| **Phase 3** | **Continuous Potential Annealing** | Smooth transition $R_t = (1-\alpha_t)\Phi(s) + \alpha_t R_{\text{game}}$ in `generic_trainer.py` | Next |
| **Phase 4** | **`ColdWarNetV4` Architecture** | Card Transformer + Opponent Belief Head + Oracle Critic | Next |
| **Phase 5** | **$\pi\text{KL}$ Subgame Resolving Engine** | C++ Turn-Level Subgame Solver for real-time inference and search-in-the-loop | Next |
| **Phase 6** | **Multi-Agent League Self-Play** | Main Agent + DEFCON Trap Exploiters + Historical Pool | Next |
