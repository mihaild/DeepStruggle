# Canonical order for permutation-invariant actions: what the literature does (2026-11-07)

Literature review for [P43](../plans/P43_canonical_placement_order.md) (canonical placement order) and the problem
note [placement_order_redundancy.md](placement_order_redundancy.md): our policy places influence one point at a time,
a run of one side's placements reaches the same board in many orders (Marshall Plan: 5,040 orders for one board), the
per-step PPO gradient never selects an order and the per-step entropy bonus rewards spreading over orders. Three
questions: how known systems order permutation-invariant multi-part actions; whether anyone trains policies by
marginalising over orders; and whether Jaxcalibur's Pokémon ([P41](../plans/P41_jaxcalibur_ingredients.md)) has the
problem at all. Every claim carries its source; "inferred" marks my own reasoning from a source, and "not found"
means a search came back empty, not that the thing does not exist.

## Summary

* **A fixed decoding order is the standard treatment** wherever a turn is a set generated autoregressively: every
  Diplomacy agent from DipNet to Cicero decodes one order per unit in one fixed, hand-chosen location order
  ("top-left to bottom-right ordering based on topological sorting", chosen so the sequence "does not jump across
  the map"); DeepNash deploys Stratego's 40 pieces one at a time in a fixed piece-type order; GraphRNN/GRAN
  generate graphs in a canonical (BFS / DFS) node order; Vinyals et al. report "improvements of 5% absolute accuracy
  or more" from restricting a set output to one canonical sequence. Nobody in this set marginalises over orders at
  training time, and nobody compares a canonical order against a free one under RL -- the order is fixed by hand
  before the first experiment, and the comparisons that exist are between fixed orders.
* **The order chosen matters, and locality / "most informative first" are the two rules the evidence supports.**
  DipNet's authors tried several orders and found a non-jumping one "prevents performance dropping when decoding a
  long sequence"; Vinyals found a star graphical model learned faster with the head variable decoded first and a bad
  5-gram order cost 25% perplexity; GRAN found single orderings differ by two orders of magnitude on one metric and
  that "the optimal set of canonical orderings is application/dataset dependent". A naive "let the model pick the
  order" gets stuck on a random order (Vinyals); learned-order schemes exist (InDIGO, Chen et al. 2021's variational
  posterior over orders) but are heavy machinery.
* **Where a canonical order is trained but not enforced at sampling, the model leaks off it** (GraphGEN: "only 9.1%
  of the generated graphs use the canonical order"). P43 enforces the order in the legal mask, so this failure mode
  cannot occur, and invalid-action masking is a valid policy gradient (Huang & Ontañón).
* **Order-marginal training exists outside RL and in one RL-adjacent line:** maximum marginal likelihood for program
  induction weights each equivalent latent sequence by its posterior given the outcome -- exactly the weight the
  order-marginal loss derives -- and Guu et al. 2017 analyse how it differs from REINFORCE; MAPO enumerates the
  high-reward sequences exactly and samples the rest. Combinatorial-optimisation RL goes the other way: POMO and
  Sym-NCO keep all symmetric sequences and use them as a shared baseline. No paper was found that computes entropy
  or a KL regulariser on the compound (order-free) action; POMO's remark is the closest: it "directly maximizes the
  entropy on the first action" by construction instead of an entropy term.
* **Pokémon random singles is immune by structure, and Jaxcalibur's documentation confirms the structure:** one
  action per side per turn, a single categorical over "a logit for switching to that slot" per team position and
  "two logits" per move (plain and Terastallized); the format has no team preview (Showdown's `gen9randombattle`
  ruleset lacks `Team Preview`, unlike OU's). Nothing in a singles battle is a multi-part action by one side, so
  there is no set of micro-actions to order. The redundancy is specific to games with multi-unit or multi-placement
  turns -- Diplomacy, Stratego's setup, Risk's reinforcement phase, StarCraft unit selection, and ours.

## 1. Canonical ordering in known systems

### 1a. Diplomacy: DipNet, Anthony et al., Gray et al., DORA, Cicero

A Diplomacy turn is one order per unit. The policy decodes the units' orders **autoregressively in a fixed location
order** in every agent of the line:

* **DipNet (Paquette et al. 2019, [arXiv 1909.02128](https://arxiv.org/abs/1909.02128)).** "In order to achieve
  coordination between units, sequential decoding is required. However there is no natural sequential ordering."
  "We hypothesize that orders are usually given to a cluster of nearby units." "We used a top-left to bottom-right
  ordering based on topological sorting" so as not to be "jumping across the map during decoding". An LSTM decodes
  one order per step from the province's encoder embedding and the previous order, through a masked softmax over
  that location's valid orders. In the code the order is the constant `STANDARD_TOPO_LOCS` ("# Predefined location
  order": YOR, EDI, LON, LVP, NTH, WAL, CLY, NWG, ENG, IRI, NAO, BEL, DEN, HEL, HOL, NWY, ...) and
  `get_orderable_locs_for_powers` sorts a power's units by it, with an optional shuffle
  ([state_space.py](https://raw.githubusercontent.com/diplomacy/research/master/diplomacy_research/models/state_space.py)).
  **They did compare orders**, in the NeurIPS rebuttal
  ([AuthorFeedback](https://papers.nips.cc/paper_files/paper/2019/file/84b20b1f5a0d103f5710bb67a043cd78-AuthorFeedback.pdf)):
  "We also tried different sequential orders during decoding, and found that using a sequential order that does not
  jump across the map prevents performance dropping when decoding a long sequence with multiple units. We
  hypothesize that this is because a unit's orders are more influenced by nearby units than distant ones." The
  paper reports no numbers for the alternatives. They also compared decoding granularity: token-based decoding
  (`A`, `PAR`, `H`) had higher token accuracy (90.6 vs 82.1% under teacher forcing) but lower unit-order accuracy
  (60.3 vs 61.3%) than decoding each unit's order as one atom (Table 7).
* **Anthony et al. 2020 ([arXiv 2006.04635](https://arxiv.org/abs/2006.04635))** replace the LSTM with a
  "Relational Order Decoder" (four residual GNN layers) that builds orders one province at a time "according to a
  fixed order", each step conditioned on the embeddings of the orders already placed; its stated purpose is that it
  "ensures inter-province consistency". The order itself is not named in the text I could read.
* **Gray et al. 2020 ([arXiv 2010.02923](https://arxiv.org/abs/2010.02923)), the DORA / Cicero architecture's
  parent:** "Unit orders are predicted in the global order used by" Paquette et al. Two further details matter
  here. (i) **Builds are decoded as a set token, not autoregressively:** because the source locations are not known
  in advance, "it is tricky to predict sets auto-regressively with teacher forcing", so each possible *set* of
  builds is one token (170 sets across all powers). Where the output is a true set with no natural key, FAIR
  enumerated whole sets rather than place them one at a time. (ii) They kept the LSTM decoder over Anthony et al.'s
  relational decoder (marginal gain after dropout tuning) and note Paquette et al. found token-based and
  transformer decoders worse.
* **DORA ([arXiv 2110.02924](https://arxiv.org/abs/2110.02924))** describes "the LSTM policy decoder, which decodes
  orders through sequential attention over each successive location in the encoder output to produce an action";
  RL and search then treat a power's whole order set as one joint action (candidate sets from the policy, an
  equilibrium over them).
* **Cicero's released code** ([base_strategy_model.py](https://raw.githubusercontent.com/facebookresearch/diplomacy_cicero/main/fairdiplomacy/models/base_strategy_model/base_strategy_model.py))
  is the same decoder: a loop `for step in range(...)` over orderable locations, step *k* attending to the location
  numbered *k* in `loc_idxs`, the previous order's embedding fed to the next step, teacher forcing by a sentinel;
  and the docstring: "Models are generally only trained on location idxs decoding in sorted order." (That Cicero's
  Science paper inherits this is inferred from the code and from the DORA / Diplodocus lineage; the Science text
  itself was not read -- see the existing note [paper_cicero_diplomacy.md](../papers/paper_cicero_diplomacy.md).)

**What Diplomacy's order is and is not (inferred).** A Diplomacy turn is a *function* unit → order: each unit gets
exactly one order, so the turn is a set keyed by unit and there is no many-to-one from sequences to outcomes once the
decoding order is fixed by the key. The fixed location order is therefore a choice of *factorisation* (which
conditional comes first), not the removal of a redundancy. Our placements are a *multiset over countries*
(Poland 3, East Germany 3) with no key, which is why the free order gives 5,040 sequences per Marshall board. The
Diplomacy evidence transfers on the factorisation question (the order affects learnability; local/coherent orders
decode long sequences better) and on the mechanics (sorted order, teacher forcing on human data in that order, a mask
per step), not on the redundancy itself. The nearest Diplomacy analogue of our problem is the build phase, and there
FAIR chose set tokens.

### 1b. AlphaStar: unit selection as an autoregressive set

* AlphaStar's action is sampled autoregressively over arguments "in the following order: function, delay, queued,
  repeat, unit_tags, target_unit_tag and world", and "unit_tags is actually obtained by sampling the pointer network
  64 times autoregressively, so conceptually, unit_tags represent 64 arguments" (AlphaStar Unplugged,
  [arXiv 2308.03526](https://arxiv.org/abs/2308.03526)). The Nature supplement's description, as reproduced by a
  secondary source, is a per-entity key, an LSTM query "repeated for selecting up to 64 units", each pick "sampled
  using the mask and temperature 0.8" and "added to subsequent autoregressive_embedding"
  ([cyk1337 notes](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II); the
  Nature paper itself is paywalled and was not read). **The selected-unit set is generated as a sequence with no
  canonical order in any source I found**: whether the supervised targets sort the units, whether already-selected
  units are masked, and how the end of selection is signalled are not stated in what I could read (not found). So
  AlphaStar is a system that lives with a k!-fold sequence redundancy on a set action -- and trains it by imitation
  first, then RL with a KL to the human policy, which pins the order distribution to the human one (inferred).
* **The independent-heads alternative was tried and rejected** by TStarBot-X
  ([arXiv 2011.13729](https://arxiv.org/abs/2011.13729)): a "multi-binary output that each unit is independently
  determined to be selected or not" "generally works well and is much more efficient compared to sequential
  predictions" but "unavoidably suffers from the unit independency that a few units intended to be selected are
  often missed"; they kept the sequential head with "one additional label to indicate the selection termination".
  This is the consistency problem P43 names (a second Austria point is worth less than the first) in another game:
  conditionally independent per-element heads cannot express it; an autoregressive factorisation can, and a
  canonical order keeps the factorisation while removing the redundancy.

### 1c. Combinatorial optimisation RL: symmetric solutions exploited, not removed

* **Kool et al. 2019, the Attention Model ([arXiv 1803.08475](https://arxiv.org/abs/1803.08475))** says nothing
  about the 2n sequences of one TSP tour; the first step uses learned placeholders and a greedy-rollout baseline.
* **POMO (Kwon et al. 2020, [arXiv 2010.16011](https://arxiv.org/abs/2010.16011))** is the paper about this
  symmetry. With a START token the model "favor[s] particular starting points, which may lead to a biased strategy"
  and each trajectory "stays close to a single optimal path"; POMO instead rolls out from every node, so it
  "directly maximizes the entropy on the first action" without an entropy term, and uses the mean of the N
  trajectories as a shared baseline, which makes training "highly resistant to local minima"; instance augmentation
  at inference seeks "different ways to arrive at the same optimal solution" (TSP100 gap 0.14% against 3.51% for
  greedy AM). POMO's trajectories from different starts are *different* tours, so it is exploration over symmetric
  representations and variance reduction by comparing them, not a marginal over equivalent sequences.
* **Sym-NCO (Kim et al. 2022, [arXiv 2205.13209](https://arxiv.org/abs/2205.13209))** names "solution
  symmetricity" (same objective value) and "problem symmetricity" (rotations) and uses both as REINFORCE baselines;
  "symmetricity is a strong inductive bias that can support the training process of DRL by making compact training
  space", while "solution symmetricity cannot be identified easily as the properties of the solutions are distinct
  for every COP".
* **Multi-Action Self-Improvement for NCO (Luttmann & Xie 2025, [arXiv 2510.12273](https://arxiv.org/abs/2510.12273))**
  states our problem in the imitation setting: a Plackett-Luce loss over an expert's agent sequence "remains
  sensitive to the permutation of the expert sequence" although "every permutation of the assignments ... yields the
  same solution"; averaging over the M! orderings is "computationally intractable"; their remedy is a
  permutation-invariant surrogate that treats each agent-task pair as an independent supervised instance, with small
  gains over the fixed-order loss (FJSP 10x5: 97.64 vs 97.99).

### 1d. Set and multiset outputs in sequence models

* **Vinyals, Bengio & Kudlur 2015, "Order Matters" ([arXiv 1511.06391](https://arxiv.org/abs/1511.06391)).**
  Output order changes learnability even when the output is a set: parsing with a depth-first linearisation 89.5 F1
  against 81.5 breadth-first; a star-shaped graphical model's joint learned more easily with the head variable
  decoded first, except with 20,000 samples or very peaky marginals; a 5-gram model at perplexity 225 in the natural
  order against 280 for the order (5,1,3,4,2). For combinatorial outputs (sorting, convex hull, triangulation),
  restricting the output to one canonical sequence of its equivalence class gave "improvements of 5% absolute
  accuracy or more". Their learned-order loss, max over orderings of log p(Y_π|X), fails naively -- it "would pick a
  random ordering (as a function of the initial parameters), and would get stuck on it permanently" -- and works
  with 1,000 steps of pretraining under a uniform prior over orderings (a sum instead of the max) or by sampling an
  ordering with probability ∝ p(Y_π|X); in the 5-gram experiment the model then settled on the natural order or its
  reverse. **This is the paper P43 cites, and it supports both of P43's choices: a canonical order (the 5% result)
  and putting the most informative elements first (the head-variable result).**
* **Graph generation.** GraphRNN (You et al. 2018, [arXiv 1802.08773](https://arxiv.org/abs/1802.08773)) writes
  p(G) as a marginal over node orderings, notes that flattening the adjacency matrix "requires training on all
  possible node permutations or specifying a canonical permutation", and trains on BFS orderings of random
  permutations, a many-to-one family that bounds the look-back width (O(Mn)); no random-vs-BFS ablation is
  reported. GRAN (Liao et al. 2019, [arXiv 1910.00760](https://arxiv.org/abs/1910.00760)): "Ordering is important
  for an auto-regressive model"; the marginal is "intractable due to the number of orderings π being factorial in
  the graph size", so they maximise a lower bound over "a family of canonical orderings" (degree-descending,
  BFS/DFS from the largest-degree node, k-core); on grid graphs single orderings differed by orders of magnitude
  (degree MMD: DFS 1.5e-5, k-core 1.4e-2, default 7.1e-2, BFS 0.16), DFS alone matched the five-ordering set, and
  "the optimal set of canonical orderings is application/dataset dependent". Chen et al. 2021, "Order Matters:
  Probabilistic Modeling of Node Sequence for Graph Generation" ([arXiv 2106.06189](https://arxiv.org/abs/2106.06189))
  learn a variational posterior q(π|G) over orderings and report much tighter likelihood bounds than uniform or
  canonical orders (Community-small: GraphRNN uniform −154.6 against −53.7); their diagnosis of a single canonical
  order trained without enforcement: for GraphGEN "only 9.1% of the generated graphs use the canonical order" used
  to compute p(G) in training, so the sampling distribution is inconsistent with the trained likelihood.
  **Inferred for P43:** that failure needs the order to be unenforced at sampling time; a mask that admits only the
  canonical sequence makes training and sampling distributions coincide by construction.
* **Molecules.** Randomized SMILES (Arús-Pous et al. 2019,
  [J. Cheminform. 11:71](https://link.springer.com/content/pdf/10.1186/s13321-019-0393-0.pdf)) is the standing
  counterexample to canonical order: training an RNN on many random string orderings of each molecule instead of
  the canonical one roughly doubled the unique molecules generated on ChEMBL, and from 1,000 training molecules
  recovered 34% of GDB-13 against 14% for canonical SMILES, because "the idiosyncrasies of the canonical SMILES
  syntax can lead to training biased models". **Inferred:** this is data amplification for a fixed supervised set --
  more distinct strings per molecule -- and the gain is in coverage of the sequence model's input space, not in the
  gradient on a redundancy; in self-play RL the data is generated, so the argument does not carry over, and the
  randomized model keeps the many-to-one sequence→molecule map that is our problem.
* **Multi-label classification with RNNs** (Yazici et al. 2020,
  [arXiv 1911.09996](https://arxiv.org/abs/1911.09996)): prior work imposed frequent-first, rare-first or
  dictionary label orders; the fixed order means "the LSTM loss will penalize otherwise correct predictions" and the
  fixed-order models generate duplicates in 24-33% of cases against 0.04-0.1% with a per-sample alignment of the
  targets to the predictions (Hungarian), O-F1 68.1 → 69.7. Related supervised losses that score the best alignment
  rather than a fixed order: OaXE for non-autoregressive translation ([arXiv 2106.05093](https://arxiv.org/abs/2106.05093),
  "removes the penalty for word order errors, and computes the cross entropy loss based on the best possible
  alignment") and DETR's bipartite-matching set loss ([arXiv 2005.12872](https://arxiv.org/abs/2005.12872)).
  InDIGO ([arXiv 1902.01370](https://arxiv.org/abs/1902.01370)) learns adaptive generation orders by insertion and
  is "competitive or even better" than left-to-right. All of these are supervised: they need a target set to align
  to, which an RL return does not give.

### 1e. Factored / autoregressive action spaces in RL

* **Metz et al. 2017, SDQN ([arXiv 1705.05035](https://arxiv.org/abs/1705.05035))** turn an N-dimensional action
  into N one-dimensional choices in index order and say so as a limitation: "this work imposes a prespecified
  ordering of actions which may negatively impact training". Tested: Half Cheetah "with random action ordering, 10
  times", "the overall changes in performance are small"; on Hopper the last dimension's Q-surface was noisier, so
  "the order of the actions dimensions matters" (qualitative). The one controlled test of order in RL I found, and
  it is small-scale.
* **Action branching (Tavakoli et al. 2018, [arXiv 1711.08946](https://arxiv.org/abs/1711.08946))**: one branch per
  action dimension after "a shared decision module", with "a level of independence for each individual action
  dimension" and no autoregressive link. **Growing Action Spaces (Farquhar et al. 2020,
  [arXiv 1906.12266](https://arxiv.org/abs/1906.12266))** is a curriculum over action-space resolution, not an
  ordering scheme. **GridNet** (microRTS and the Lux AI season-1 winner, per Goodfriend 2024,
  [arXiv 2402.08112](https://arxiv.org/abs/2402.08112)) emits every unit's action in one policy call from per-cell
  heads, the alternative being a per-unit loop that simulates the state after each unit's action. Google Research
  Football's default set is 19 discrete actions per controlled player
  ([observation.md](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md)). None
  of these has a multiset action; they are keyed sets (one slot per unit / dimension), like Diplomacy.
* **Invalid action masking** (Huang & Ontañón 2022, [arXiv 2006.14171](https://arxiv.org/abs/2006.14171)): the
  gradient under a state-dependent mask "is a valid policy gradient", masking is "a state-dependent differentiable
  function" applied to the logits, and it "scales well" as the invalid set grows where a penalty does not. P43's
  mechanism is exactly a state-dependent mask, so the trained object is the renormalised per-step policy, with no
  correction needed.

### 1f. Board and card games with multi-placement turns

* **DeepNash / Stratego (Perolat et al. 2022, [arXiv 2206.15378](https://arxiv.org/abs/2206.15378)).** The closest
  published analogue of our setup phase. Deployment is 40 placements, one action per piece: "the piece deployment
  order is fixed and sorted by piece-type" (Flag, Bombs, Marshal, General, ..., Scout, Spy); a deployment head
  outputs a 10x10 distribution over legal squares, and because the order is fixed "it needs no extra piece-identity
  input"; the play-phase action space is "discrete and of size 100 throughout the game", a move being two actions
  (select piece, then destination). **Inferred, not discussed in the paper:** the fixed type order removes the
  across-type order redundancy but not the within-type one -- the six bombs are identical, so a bomb layout still has
  6! placement sequences, scouts 8!, and so on -- and DeepNash's per-step regulariser toward a reference policy
  (R-NaD) acts on those redundant conditionals exactly as ours does. The paper reports no difficulty from it; it
  also does not look for one.
* **Risk (Blomqvist 2020, KTH thesis, [PDF](https://forums.triplea-game.org/assets/uploads/files/1637032761173-fulltext01.pdf)).**
  The reinforcement phase is our problem in another game, and the thesis sets out the two designs: one-shot
  distribution of N troops over T territories, with "T^N / N!" unique distributions (their count), or "N decisions
  of placing one", so that "each of these decisions can be taken with a policy spanning over only T < 42 possible
  options", which "does however add N−1 decisions which is an apparent drawback for MCTS-based agents". They took
  the per-troop design and added macro options -- "placing either all or half (rounded up) of the available troops
  per decision" (place policy 1x126 = 42 x 3) -- the P42 pattern. No canonical order, and the order redundancy is
  not mentioned.
* **Hex-and-counter wargames (Cánovas & Bou 2025, [arXiv 2502.13918](https://arxiv.org/abs/2502.13918))**: each
  action names a unit and a one-hex step, the agent chooses which unit acts next, reinforcements are placed "one at
  a time"; no fixed unit order and no discussion of order.
* **Hearthstone, Gwent, Settlers of Catan, Lux AI season 2:** no source on their action ordering was read in this
  pass. Inferred from the rules only: in those games the order of a turn's plays changes the state (mana and
  resources spent, board effects triggered, units moved before others), so a turn's sequence is mostly not
  redundant and the question does not arise in the same form.

### 1g. Known downsides of a canonical order, and remedies

* **A bad fixed order costs learnability** (Vinyals: 225 vs 280 perplexity; parsing 89.5 vs 81.5; GRAN: BFS vs DFS
  four orders of magnitude on grid MMD; DipNet: a jumping order degrades long sequences). Remedies used: choose by
  locality / coherence (DipNet), put the most informative or most constraining element first (Vinyals' head
  variable), try a small family and pick empirically (GRAN), or learn the order (Vinyals' sampled orderings, Chen
  2021, InDIGO) -- the last at real cost.
* **A canonical order trained without being enforced at sampling leaks** (GraphGEN's 9.1%). Remedy: enforce it,
  which P43's mask does.
* **Per-step regulariser bias along the order.** Not found in the literature as a named effect; POMO's complaint
  that a START token makes the policy "favor particular starting points" and its answer -- make the first action
  uniform by construction and leave the entropy term out -- is the nearest statement.
* **Last-position noise** (Metz: the last dimension's Q-surface noisier). Inferred reason: the last conditional in
  a canonical order sees the smallest legal set and the fewest samples per context.

## 2. Order-marginal and permutation-invariant training of policies

* **Maximum marginal likelihood against REINFORCE (Guu et al. 2017,
  [arXiv 1704.07926](https://arxiv.org/abs/1704.07926))** is the cleanest statement of remedy 4's weight. Both
  objectives give a gradient Σ_z q(z) ∇ log p_θ(z|x); REINFORCE weights by q_RL(z) = p_θ(z|x), MML by
  q_MML(z) = p_θ(z|x, R(z) ≠ 0), "the policy distribution restricted to and renormalized over only reward-earning
  programs" -- the posterior over latent sequences given the outcome, which is w_σ = P(σ)/P(M) in
  [placement_order_redundancy.md](placement_order_redundancy.md). Their failure mode is *spurious* programs (wrong
  programs with the right answer, "at least 1600 on average for each example"): "the rich get richer, and the poor
  get poorer", a spurious program at 100x the probability of the right one takes "99%" of the gradient. Their
  β-meritocratic update flattens the posterior (β = 0: "completely uniform across all reward-earning programs").
  **Inferred for us:** orders of one board are *not* spurious -- every order is the same action -- so the
  posterior weighting has no wrong-program risk, and the flattening (β < 1) would only spend gradient equalising
  orders, which is the thing to avoid.
* **MAPO (Liang et al. 2018, [arXiv 1807.02322](https://arxiv.org/abs/1807.02322))** splits the expected return
  into an exact enumeration over a buffer of high-reward sequences and a sampled remainder: "the first expectation
  can be computed exactly by enumerating all the trajectories in the buffer", and variance falls because the
  sampled part lives in "a smaller stochastic space of measure (1−π_B)". The lattice forward pass of remedy 4 is the
  same move over the orders of one board.
* **Combinatorial optimisation RL** (POMO, Sym-NCO, §1c) keeps every symmetric sequence and uses the group as a
  shared baseline; the 2025 multi-agent NCO paper calls the exact average over M! orderings intractable and uses a
  permutation-invariant surrogate.
* **Graph generation** (GRAN, Chen 2021, §1d) maximises a lower bound of the marginal over a canonical family or a
  learned posterior over orders; these are likelihoods, not returns, but the estimators transfer.
* **Compound-action entropy and KL.** No RL paper was found that computes the entropy or a KL regulariser of a
  macro-action over its equivalence class of micro-sequences, or that discusses how a per-step entropy bonus behaves
  on redundant orders (searches on "macro-action redundancy", "equivalent action sequences", "permutation-invariant
  policy gradient" returned nothing on point). What exists: POMO's "maximizes the entropy on the first action" by
  construction; Vinyals' uniform prior over orderings as a pretraining phase only. The chain rule of entropy is the
  relevant fact and needs no citation: with one sequence per board, H(board) = Σ_k E[H(π_k(·|prefix))], so under a
  canonical order the summed per-step entropy *is* the board entropy in expectation; what differs is the gradient
  (the per-step bonus has no term for the entropy a choice leads to), which is P43's remedy 1 in its own words.

## 3. Jaxcalibur and Pokémon

**Documented** ([jaxcalibur.github.io](https://jaxcalibur.github.io/)): "It plays random singles battles" on
Pokémon Showdown's Gen 9 random battle ladder. The policy's action outputs: "For action selection, each team
position emits a logit for switching to that slot." "Each move position emits two logits: one for using the move and
one for Terastallizing and using the move." Struggle is an extra candidate-move token. Moves are simultaneous ("In
Pokémon, both players often choose actions at the same time", so the equilibrium is mixed and the search samples by
regret matching). Training: PPO with GAE, win/loss reward, entropy plus "a zero-avoiding term ... which is essentially
KL divergence toward a uniform policy", ~100M self-play games in a week on one H100, search only at play time. The
words "team preview", "doubles" and "VGC" do not appear on the page; no paper or code is released.

**Documented** (Showdown's [config/formats.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/config/formats.ts)
and [data/rulesets.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/rulesets.ts)):
`[Gen 9] Random Battle` has `team: random` and the ruleset `PotD, Obtainable, Species Clause, HP Percentage Mod,
Cancel Mod, Sleep Clause Mod, Illusion Level Mod` -- no `Team Preview`; `[Gen 9] Random Doubles Battle` likewise; `[Gen 9] OU`
uses `Standard`, which reaches `Team Preview` through `Standard AG`. The `teampreview` rule "allows players to see
the Pokemon on their opponent's team before selecting their lead Pokemon".

**Inferred from the structure:**

* In singles, every decision one side makes is a **single categorical choice**: at most 4 moves x 2 (plain /
  Terastallized) + up to 5 switches (+ Struggle) ≈ 14 options, one per turn, and the forced switch after a faint or
  a pivot move is again one choice. There is no sequence of micro-actions by one side between opponent or chance
  events, so no set of sequences maps to one outcome. Jaxcalibur's domain is immune by construction, not by any
  treatment in its recipe. P41 can therefore take nothing from it on this question; none of its four ingredients
  touches order.
* The one place Pokémon offers a two-step factorisation -- "Terastallize or not, then which move" -- Jaxcalibur
  flattened into a joint categorical (two logits per move). That is the joint-product pattern of P42's form (c), not
  a sequential one.
* Where Pokémon *would* have our redundancy: formats with team preview. In singles OU a player submits a full team
  order of which only the lead matters, so one effective choice has 5! = 120 equivalent orderings if a policy had
  to emit the whole order (most bots emit only the lead); in VGC doubles the back pair's order is irrelevant
  (2 orderings) and the leads' slots are keyed (left/right), so the redundancy is small. Random battles have no
  team preview, so Jaxcalibur never faces even this.
* Simultaneous moves are a different issue (mixed equilibria, handled in search), not a redundancy.

**Contrast with Twilight Struggle** (from [placement_order_redundancy.md](placement_order_redundancy.md)): 49% of
our decisions are single influence points, in runs averaging 2.19 points over 1.80 countries, with the setup runs of
6 / 7 / 2, Marshall Plan's seven distinct countries (5,040 orders per board), Warsaw Pact 5, The China Card 5, The
Reformer 6. Pokémon has nothing of this shape; Diplomacy, Stratego's deployment, Risk's reinforcement and StarCraft's
unit selection do, and of those only Diplomacy and Stratego fixed the order -- in both cases from the start, without a
comparison.

## What this means for P43

1. **A canonical order enforced in the mask is the established treatment, and the precedents are the strongest
   agents in the nearest games.** Diplomacy's whole line (DipNet → Cicero) and DeepNash's deployment fix the order
   by hand before training; the sequence-model literature quantifies the gain of restricting a set to one sequence
   (Vinyals: ≥ 5% absolute on combinatorial outputs) and the cost of leaving a trained canonical order unenforced
   (GraphGEN's 9.1%). Masking keeps the policy gradient valid (Huang & Ontañón). Nothing found argues for leaving
   the redundancy in place; the randomized-SMILES result is about supervised data coverage and does not carry over
   to self-play.
2. **Choose the order by the two rules the evidence supports, and test one alternative in Phase A.** (a) *Most
   informative first:* battlegrounds before non-battlegrounds (Vinyals' head variable; P43's own proposal). (b)
   *Coherent, non-jumping:* within the run keep neighbouring countries adjacent in rank (DipNet's one measured
   finding, on long sequences -- ours are the setups' 6-7 and Marshall's 7). A region-major order (the setup and
   most event runs are within one region anyway), battlegrounds first within the region, then adjacency rather than
   card id, satisfies both; plain country id is the control. GRAN's lesson is that the right order is dataset
   dependent and cannot be predicted, Metz's that in low dimension the difference may be small -- so Phase A should
   run the id-only order beside the chosen one, cheap at the setup. Do not try to learn the order: Vinyals' naive
   max gets stuck, and the working schemes (sampled orderings, a variational posterior) are machinery for a
   likelihood, not a return.
3. **Do not marginalise over orders as the first move.** Every system reviewed either removes the symmetry
   (canonical order) or exploits it as a baseline (POMO, Sym-NCO); the exact marginal is called intractable in
   general (GRAN, Chen, Luttmann & Xie) and no RL system was found doing it for action orders. Remedy 4's
   posterior-weighted credit is the MML weight of Guu et al. and is sound here (orders are not spurious programs);
   its lattice makes the sum tractable for our run lengths. It stays the fallback if the hand-fixed order proves
   hard to learn in, as the problem note already says, and the MAPO precedent says the exact part is where the
   variance goes.
4. **Keep the autoregressive factorisation; do not switch to independent per-country counts.** TStarBot-X's
   multi-binary selection head "often missed" units that belonged together, and Luttmann & Xie's
   permutation-invariant surrogate bought little. Consistency across a run ("a second point is worth less") needs
   conditioning on what was placed, which canonical order preserves. P42's "until control" macro composes with it
   as P43 states (the bound moves to the macro's country); Risk's agent used one-troop steps plus all/half macros
   together, the same pairing.
5. **Regularisers.** The literature has no per-step-regulariser-under-redundancy result to lean on; the chain rule
   says that under one sequence per board the summed per-step entropy already measures board entropy, and the
   remaining issue is the gradient's missing lookahead term and the suffix-shrinking legal set, as P43 derives.
   POMO's practice -- make the first choice the one that carries the diversity -- matches P43's remedy 2 (per-step
   terms on the run's first decision only, where the legal set is the full set) as the cheap first version, with
   the board-level terms (remedy 1) as the principled one; measure the stray-point-by-rank effect in Phase A as
   planned. Note that the uniform-over-boards target puts *more* first-step mass on low ranks (a low rank has the
   most completions), the opposite direction of the per-step suffix bias, so the two will not cancel by luck.
6. **Human data:** the Cicero line trains on human orders re-sorted into its fixed location order with teacher
   forcing ("only trained on location idxs decoding in sorted order"); re-sorting a human placement run into the
   canonical order for behaviour cloning has that precedent and is lossless.
7. **What remains unmeasured anywhere:** canonical versus free order under policy-gradient self-play with a
   per-step entropy bonus. Phase A would be, as far as this search found, the first such measurement; the setup
   metrics it lists (outcome entropy against per-step entropy, stray single points by rank, second-point
   suppression) are the right ones, and worth recording for that reason.

## Sources

Diplomacy: [DipNet arXiv 1909.02128](https://arxiv.org/abs/1909.02128) ·
[DipNet NeurIPS rebuttal](https://papers.nips.cc/paper_files/paper/2019/file/84b20b1f5a0d103f5710bb67a043cd78-AuthorFeedback.pdf) ·
[diplomacy/research state_space.py](https://raw.githubusercontent.com/diplomacy/research/master/diplomacy_research/models/state_space.py) ·
[Anthony et al. arXiv 2006.04635](https://arxiv.org/abs/2006.04635) · [Gray et al. arXiv 2010.02923](https://arxiv.org/abs/2010.02923) ·
[DORA arXiv 2110.02924](https://arxiv.org/abs/2110.02924) · [Diplodocus arXiv 2210.05492](https://arxiv.org/abs/2210.05492) ·
[diplomacy_cicero base_strategy_model.py](https://raw.githubusercontent.com/facebookresearch/diplomacy_cicero/main/fairdiplomacy/models/base_strategy_model/base_strategy_model.py).
StarCraft: [AlphaStar Unplugged arXiv 2308.03526](https://arxiv.org/abs/2308.03526) ·
[AlphaStar architecture notes (secondary)](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II) ·
[TStarBot-X arXiv 2011.13729](https://arxiv.org/abs/2011.13729); see also [paper_alphastar.md](../papers/paper_alphastar.md).
Combinatorial optimisation: [Kool et al. arXiv 1803.08475](https://arxiv.org/abs/1803.08475) ·
[POMO arXiv 2010.16011](https://arxiv.org/abs/2010.16011) · [Sym-NCO arXiv 2205.13209](https://arxiv.org/abs/2205.13209) ·
[Luttmann & Xie arXiv 2510.12273](https://arxiv.org/abs/2510.12273).
Sets and graphs: [Order Matters arXiv 1511.06391](https://arxiv.org/abs/1511.06391) ·
[GraphRNN arXiv 1802.08773](https://arxiv.org/abs/1802.08773) · [GRAN arXiv 1910.00760](https://arxiv.org/abs/1910.00760) ·
[Chen et al. 2021 arXiv 2106.06189](https://arxiv.org/abs/2106.06189) ·
[Randomized SMILES, J. Cheminform. 2019](https://link.springer.com/content/pdf/10.1186/s13321-019-0393-0.pdf) ·
[Yazici et al. arXiv 1911.09996](https://arxiv.org/abs/1911.09996) · [OaXE arXiv 2106.05093](https://arxiv.org/abs/2106.05093) ·
[DETR arXiv 2005.12872](https://arxiv.org/abs/2005.12872) · [InDIGO arXiv 1902.01370](https://arxiv.org/abs/1902.01370).
Factored actions in RL: [SDQN arXiv 1705.05035](https://arxiv.org/abs/1705.05035) ·
[Action branching arXiv 1711.08946](https://arxiv.org/abs/1711.08946) · [Growing Action Spaces arXiv 1906.12266](https://arxiv.org/abs/1906.12266) ·
[Invalid action masking arXiv 2006.14171](https://arxiv.org/abs/2006.14171) · [microRTS / GridNet arXiv 2402.08112](https://arxiv.org/abs/2402.08112) ·
[Google Research Football actions](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md).
Games: [DeepNash arXiv 2206.15378](https://arxiv.org/abs/2206.15378) (see also [paper_deepnash_rnad.md](../papers/paper_deepnash_rnad.md)) ·
[Risk thesis, Blomqvist 2020](https://forums.triplea-game.org/assets/uploads/files/1637032761173-fulltext01.pdf) ·
[Hex wargames arXiv 2502.13918](https://arxiv.org/abs/2502.13918).
Marginal training: [Guu et al. arXiv 1704.07926](https://arxiv.org/abs/1704.07926) · [MAPO arXiv 1807.02322](https://arxiv.org/abs/1807.02322).
Pokémon: [Jaxcalibur](https://jaxcalibur.github.io/) · [Showdown formats.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/config/formats.ts) ·
[Showdown rulesets.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/rulesets.ts).
