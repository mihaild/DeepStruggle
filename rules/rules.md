# Formal Game Rules Specification: Twilight Struggle (Deluxe Edition)

This document presents a deterministic, agent-friendly, and mathematically formal specification of the complete rulebook from `Rules_Final.pdf` for the Deluxe Edition of **Twilight Struggle** (110 Cards). It maps directly to the C++20 zero-allocation engine architecture specified in [`specification.txt`](specification.txt), [`game_state.txt`](game_state.txt), [`action_interface.txt`](action_interface.txt), and [`flags.json`](flags.json).

---

## 1. System Invariants & Primitives

### 1.1 Superpowers & Players
The simulation models a two-player zero-sum contest between two superpowers:
- **US (`Player::US = 1`)**: Maximizes Victory Points towards $+20$.
- **USSR (`Player::USSR = -1`)**: Minimizes Victory Points towards $-20$.
- **Neutral / None (`Player::NONE = 0`)**: Denotes uncontrolled regions or non-affiliated cards.

The opponent mapping operator is defined as:
$$\text{get\_opponent}(p) = -p$$

### 1.2 Global Tracks & State Bounds
| Track | State Variable | Type | Bounds | Initial Value | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Victory Points** | `victory_points` | `int8_t` | $[-20, +20]$ | $0$ | $+20$ is instant US win, $-20$ is instant USSR win |
| **DEFCON Status** | `defcon` | `uint8_t` | $[1, 5]$ | $5$ | $1$ triggers immediate nuclear war loss for phasing player |
| **US Military Ops** | `us_mil_ops` | `uint8_t` | $[0, 5]$ | $0$ | Required per turn equals DEFCON level |
| **USSR Military Ops**| `ussr_mil_ops` | `uint8_t` | $[0, 5]$ | $0$ | Required per turn equals DEFCON level |
| **US Space Track** | `us_space_track` | `uint8_t` | $[0, 8]$ | $0$ | Max box 8; grants VPs & milestone abilities |
| **USSR Space Track**| `ussr_space_track`| `uint8_t` | $[0, 8]$ | $0$ | Max box 8; grants VPs & milestone abilities |
| **Turn** | `turn` | `uint8_t` | $[1, 10]$ | $1$ | Turns 1–3: Early, 4–7: Mid, 8–10: Late War |
| **Action Round** | `action_round` | `uint8_t` | $[1, 8]$ | $1$ | 6 ARs (T1–3), 7 ARs (T4–10), 8 ARs (North Sea Oil / Space 8) |

---

## 2. Board Topology & Country Control

### 2.1 Graph Definition
The world map is represented as a planar graph $G = (V, E)$ consisting of $|V| = 84$ countries and two off-board superpower nodes (US and USSR).
- **Regions (6)**: Europe ($0$), Asia ($1$), Middle East ($2$), Africa ($3$), Central America ($4$), South America ($5$).
- **Sub-regions**:
  - **Western Europe** and **Eastern Europe** partition Europe, except **Austria** and **Finland**, which belong to *both* Western and Eastern Europe.
  - **Southeast Asia** is a sub-region of Asia.
- **Geopolitical Anomalies**:
  - **Canada** and **Turkey** belong to Europe.
  - **Libya** and **Egypt** belong to the Middle East.
- **Node Classification**: Each country $v \in V$ possesses a constant Stability Number $S(v) \in \{1, 2, 3, 4, 5\}$ and a boolean classification $\text{IsBattleground}(v) \in \{0, 1\}$.

### 2.2 Country Control Invariant
For country $v \in V$, let $I_{\text{US}}(v) \in \mathbb{N}_0$ and $I_{\text{USSR}}(v) \in \mathbb{N}_0$ denote current influence. Control predicate $\text{Control}_p(v)$ for player $p \in \{\text{US}, \text{USSR}\}$ is:

$$\text{Control}_p(v) \iff \Big(I_p(v) \ge S(v)\Big) \land \Big(I_p(v) - I_{-p}(v) \ge S(v)\Big)$$

If neither $\text{Control}_{\text{US}}(v)$ nor $\text{Control}_{\text{USSR}}(v)$ holds, country $v$ is **Uncontrolled**. Both players can never simultaneously control the same country.

---

## 3. Game Setup

1. **Deck Initialization**: Shuffle the 34 Early War cards (Cards 1–35, excluding `#6 The China Card`).
2. **Deal Hands**: Deal 8 cards to US and 8 cards to USSR.
3. **The China Card (#6)**: Placed face-up with USSR (`china_card_holder = Player::USSR`, `china_card_playable = 1`).
4. **Initial Fixed Influence Placement**:
   - **USSR** (5 countries, 9 influence):
     - Syria: $1$
     - Iraq: $1$
     - North Korea: $3$
     - East Germany: $3$
     - Finland: $1$
   - **US** (9 countries, 13 influence):
     - Iran: $1$
     - Israel: $1$
     - Japan: $1$
     - Australia: $1$
     - Philippines: $1$
     - South Korea: $1$
     - Panama: $1$
     - South Africa: $1$
     - United Kingdom: $5$
5. **Initial Discretionary Influence Placement**:
   - **USSR** places $6$ additional influence anywhere in Eastern Europe.
   - **US** places $7$ additional influence anywhere in Western Europe.
6. **Marker Alignment**: Set VP to $0$, DEFCON to $5$, MilOps to $0/0$, Space Race to $0/0$, Turn to $1$.

---

## 4. Complete Turn Execution State Machine

```
              ┌──────────────────────────────────────────────┐
              │           Start of Turn (1..10)              │
              └──────────────────────┬───────────────────────┘
                                     │
                                     ▼
                    [ Phase A: Improve DEFCON (+1) ]
                                     │
                                     ▼
                    [ Phase B: Deal Cards to Hand ]
                                     │
                                     ▼
                    [ Phase C: Headline Phase ]
                                     │
                                     ▼
             ┌──────[ Phase D: Action Rounds 1..6/7/8 ]──────┐
             │                                               │
             │   USSR AR ──▶ US AR ──▶ Next AR               │
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
             [ Phase E: Check Required Military Operations ]
                                     │
                                     ▼
             [ Phase F: Reveal & Verify Held Cards ]
                                     │
                                     ▼
             [ Phase G: Flip China Card Face-Up ]
                                     │
                                     ▼
             [ Phase H: Advance Turn / Reshuffle Decks ]
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
             [ Turn < 10 ]                     [ Turn == 10 ]
             Loop to Start                           │
                                                     ▼
                                      [ Phase I: Final Scoring ]
```

### 4.1 Phase A: Improve DEFCON Status
$$\text{defcon} \leftarrow \min(5, \text{defcon} + 1)$$

### 4.2 Phase B: Deal Cards
- **Early War (Turns 1–3)**: Target hand size is $8$.
- **Mid War (Turns 4–7)**: Target hand size is $9$. At start of Turn 4, Mid War cards (Cards 36–79) are shuffled into the draw deck.
- **Late War (Turns 8–10)**: Target hand size is $9$. At start of Turn 8, Late War cards (Cards 80–110) are shuffled into the draw deck.
- **Reshuffle Invariant**: When draw deck is exhausted, shuffle the discard pile (excluding permanently removed cards marked with asterisk `*`) to form a new draw deck. Mid/Late War cards are added to the existing draw deck without reshuffling discards until the draw deck is empty. The China Card is never counted toward hand limits.

### 4.3 Phase C: Headline Phase
1. **Selection**: Each player secretly selects 1 card from hand (The China Card is prohibited).
   - *Space Race Box 4 Modifier*: If one player has reached Space Race box 4 ("Man in Space") and the opponent has not, the opponent must select and reveal their headline card *before* the space leader selects.
2. **Simultaneous Reveal & Resolution Priority**:
   - Headline Value $\text{HV}(C) = \text{base\_ops}(C)$.
   - Scoring cards have $\text{HV} = 0$.
   - The card with higher $\text{HV}$ resolves first.
   - **Tie-Breaking Rule**: US headline resolves first on ties.
   - **Double Scoring Tie**: If both players headline Scoring cards, US scoring card resolves first.
3. **Opponent Event Execution**: If a player plays an opponent-affiliated headline card, the opponent implements the event as the temporary phasing player.
4. **Headline Ops Restriction**: Headline cards generate no Operations points unless explicitly stated on the card.

### 4.4 Phase D: Action Rounds
- Alternating Action Rounds: USSR plays AR1, US plays AR1, USSR AR2, US AR2, ..., up to AR6 (Turns 1–3) or AR7 (Turns 4–10).
- **Extra Action Round**: US receives an 8th Action Round if `#86 North Sea Oil` was played by US or if US reached Space Race box 8.
- **Held Card Invariant**: Players may hold at most 1 card across turns. **Scoring cards may never be held**.
- **Passing / Card Shortage**: Players cannot voluntarily pass. If a player runs out of cards before completing all Action Rounds, that player sits out while the opponent completes remaining ARs.

### 4.5 Phase E: Military Operations Status Check
At turn end, calculate required military operations penalty:
$$\text{Deficit}_p = \max(0, \text{defcon} - \text{mil\_ops}_p)$$
$$\text{victory\_points} \leftarrow \text{victory\_points} + \text{Deficit}_{\text{USSR}} - \text{Deficit}_{\text{US}}$$
$$\text{us\_mil\_ops} \leftarrow 0, \quad \text{ussr\_mil\_ops} \leftarrow 0$$

### 4.6 Phase F: Reveal Held Cards (Tournament Rules)
Both players reveal their held card. If any player holds a Scoring Card:
$$\text{Holding Player Loses Immediately (Accidental Nuclear War)}$$

### 4.7 Phase G: Flip China Card
If The China Card was passed face-down during the turn, flip it face-up (`china_card_playable = 1`).

### 4.8 Phase H: Advance Turn Marker
- Increment `turn += 1`. Reset turn-persistent flags and operational aggregates.
- If entering Turn 4, shuffle Mid War deck into draw deck.
- If entering Turn 8, shuffle Late War deck into draw deck.

### 4.9 Phase I: Final Scoring (End of Turn 10)
If no automatic victory occurred prior to Turn 10 completion:
1. Score Europe (Europe Control triggers immediate victory).
2. Score Asia (including Southeast Asia).
3. Score Middle East.
4. Score Africa.
5. Score Central America.
6. Score South America.
7. Award $+1\text{ VP}$ to the player holding The China Card.
8. Evaluate winner: $\text{VP} > 0 \implies \text{US Win}$, $\text{VP} < 0 \implies \text{USSR Win}$, $\text{VP} == 0 \implies \text{Draw}$.

---

## 5. Operations Mechanics

When a card is played for Operations, the phasing player must allocate **all** Operations points to exactly one of four modes:

```
                               [ Play Card for Ops ]
                                         │
       ┌────────────────────┬────────────┴────────────┬────────────────────┐
       ▼                    ▼                         ▼                    ▼
[ Mode 0: Influence ] [ Mode 1: Realign ]     [ Mode 2: Coup ]    [ Mode 3: Space Race ]
• 1 Op friendly/uncon • 1 Op per roll         • Costs full card   • Costs full card
• 2 Ops enemy control • Both roll 1d6 + mods  • 1d6 + Ops vs 2xS  • Advance + VP/Ability
• Dynamic cost drop   • Remove diff inf       • MilOps credit     • Event canceled
                                              • Battleground      • Card to discard
                                                degrades DEFCON
```

### 5.1 Mode 0: Influence Placement
- **Cost**:
  - $1\text{ Op}$ per influence marker in an Uncontrolled or Friendly-Controlled country.
  - $2\text{ Ops}$ per influence marker in an Enemy-Controlled country.
  - **Dynamic Cost Transition**: If enemy control is broken during multi-point placement, subsequent influence placements in that country during the same Action Round cost $1\text{ Op}$.
- **Adjacency Requirement**: Influence markers may only be placed in:
  1. A country where the player already has influence;
  2. A country adjacent to a friendly influence marker that was present at the **start** of the Action Round;
  3. A country adjacent to the player's superpower home space.

### 5.2 Mode 1: Realignment Rolls
- **Cost**: $1\text{ Op}$ per realignment attempt. Target may be targeted multiple times in the same AR.
- **Target Condition**: Target country must contain opponent influence. Friendly influence or adjacency is not required.
- **Resolution Algorithm**:
  - Both players roll $1d6$.
  - Modifiers added to each player's roll (+1 each):
    - $+1$ for each adjacent country controlled by that player.
    - $+1$ if the player has strictly more influence in the target country than the opponent.
    - $+1$ if the player's superpower home space is adjacent to the target country.
  - **Outcome**:
    $$\Delta = \text{ModifiedRoll}_{\text{High}} - \text{ModifiedRoll}_{\text{Low}}$$
    High roller removes $\Delta$ influence of the opponent from the target country.
    If $\text{ModifiedRoll}_{\text{US}} = \text{ModifiedRoll}_{\text{USSR}}$, result is a tie (no change).
  - **Invariant**: No influence is ever added via realignment.

### 5.3 Mode 2: Coup Attempts
- **Cost**: Entire card (all base Ops of the played card).
- **Target Condition**: Target country must contain opponent influence.
- **DEFCON Invariant**: Coups cannot be attempted in regions restricted by the current DEFCON level:
  - DEFCON 4: Europe prohibited.
  - DEFCON 3: Europe and Asia prohibited.
  - DEFCON 2: Europe, Asia, and Middle East prohibited.
- **Execution & Resolution**:
  1. **Military Operations Credit**: $\text{mil\_ops}_p \leftarrow \min(5, \text{mil\_ops}_p + \text{base\_ops}(C))$.
  2. **DEFCON Degradation**: If target is a Battleground, $\text{defcon} \leftarrow \text{defcon} - 1$.
     *(DEFCON suicide check occurs immediately)*.
  3. **Resolution Formula**:
     $$\text{Total} = 1d6 + \text{effective\_ops}(C)$$
     $$\text{Margin} = \text{Total} - 2 \times S(\text{target})$$
  4. **Influence Modification**:
     - If $\text{Margin} \le 0$: Coup fails, no influence change.
     - If $\text{Margin} > 0$:
       $$\text{Removed} = \min\big(\text{Margin}, I_{-p}(\text{target})\big)$$
       $$I_{-p}(\text{target}) \leftarrow I_{-p}(\text{target}) - \text{Removed}$$
       $$\text{Added} = \text{Margin} - \text{Removed}$$
       $$I_p(\text{target}) \leftarrow I_p(\text{target}) + \text{Added}$$

### 5.4 Mode 3: Space Race
- **Requirement**: Played card must have $\text{base\_ops}(C) \ge \text{required\_ops}(\text{target\_box})$.
- **Turn Frequency Limit**: Max 1 attempt per turn (increased to 2 attempts per turn if Space Race box 2 "Animal in Space" is active).
- **Event Cancellation**: The event on a card used for Space Race **never** occurs, regardless of affiliation (serves as opponent event dump).
- **Discard Policy**: Card moves to discard pile (even if marked with asterisk `*`).
- **Milestone Table**:

| Box | Name | Min Ops | Success Roll | VP (1st / 2nd) | Special Ability (1st Player Only) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Earth Satellite | 2 | 1–3 | 2 / 1 | None |
| **2** | Animal in Space | 2 | 1–4 | 0 / 0 | May play 2 Space Race cards per turn |
| **3** | Man in Orbit | 2 | 1–4 | 2 / 0 | None |
| **4** | Man in Space | 2 | 1–3 | 0 / 0 | Opponent must reveal Headline card first |
| **5** | Lunar Probe | 3 | 1–4 | 3 / 1 | None |
| **6** | Space Walk | 3 | 1–3 | 0 / 0 | May discard held card at end of turn |
| **7** | Space Station | 3 | 1–4 | 4 / 2 | None |
| **8** | Eagle/Bear Landed | 4 | 1–2 | 2 / 0 | May play 8 Action Rounds per turn |

*Special abilities are canceled immediately when the second player reaches that box.*

---

## 6. Card Play Semantics & Opponent Event Priority

### 6.1 Opponent Card Resolution & Timing Choice
When a player plays a card associated solely with the opponent for Operations:
1. The opponent's event **must** trigger.
2. The phasing player chooses the execution timing:
   - **`OPS_FIRST` (0)**: Phasing player conducts Ops, then opponent event resolves.
   - **`EVENT_FIRST` (1)**: Opponent event resolves, then phasing player declares Op mode and conducts Ops.
3. The opponent makes all decisions required by the event as the active decision player.
4. **Prerequisite & Cancellation Edge Cases**:
   - If event cannot occur because prerequisite is unmet (e.g. `#21 NATO` before `#23 Marshall Plan` or `#16 Warsaw Pact`), event does not occur and asterisk card is sent to **discard pile** (not removed).
   - If event is prohibited by a superseding event (e.g. `#13 Arab-Israeli War` after `#65 Camp David Accords`), event does not occur and card is played for Ops only; asterisk card sent to **discard pile**.
   - If event occurs but has zero legal targets/effect (e.g. `#33 Alliance for Progress` with no US battlegrounds), event is considered played and asterisk card **is permanently removed**.

### 6.2 DEFCON Suicide Priority Rule
The phasing player is strictly responsible for DEFCON dropping to 1 during their Action Round:
- If conducting Ops first causes DEFCON to reach 1 (e.g. couping a battleground at DEFCON 2), the phasing player **loses immediately**, before the opponent's event can resolve.
- If executing the opponent's event first causes DEFCON to reach 1 (e.g. `#4 Duck and Cover` at DEFCON 2), the phasing player **loses immediately**, because DEFCON reached 1 during their active turn.

---

## 7. Scoring & Victory Conditions

### 7.1 Regional Scoring Predicates
For a region $R$:
- Let $C(R)$ be the set of countries in $R$.
- Let $B(R) \subset C(R)$ be the Battleground countries in $R$.
- Let $N(R) = C(R) \setminus B(R)$ be the Non-battleground countries in $R$.
- Let $K_p(R) = \{v \in C(R) \mid \text{Control}_p(v)\}$.
- Let $K_p^B(R) = \{v \in B(R) \mid \text{Control}_p(v)\}$.
- Let $K_p^N(R) = \{v \in N(R) \mid \text{Control}_p(v)\}$.

| Status | Mathematical Predicate | Base VP Award |
| :--- | :--- | :--- |
| **Presence** | $\|K_p(R)\| \ge 1$ | Specific to Region Card |
| **Domination** | $(\|K_p(R)\| > \|K_{-p}(R)\|) \land (\|K_p^B(R)\| > \|K_{-p}^B(R)\|) \land (\|K_p^B(R)\| \ge 1) \land (\|K_p^N(R)\| \ge 1)$ | Specific to Region Card |
| **Control** | $(\|K_p(R)\| > \|K_{-p}(R)\|) \land (K_p^B(R) = B(R))$ | Specific to Region Card |

#### Regional Scoring Math
For player $p \in \{\text{US}, \text{USSR}\}$:
$$\text{Score}_p(R) = \text{VP}_{\text{Status}}(p, R) + \|K_p^B(R)\| + \|\{v \in K_p(R) \mid \text{Adjacent}(v, \text{Superpower}_{-p})\}\|$$
$$\Delta\text{VP} = \text{Score}_{\text{US}}(R) - \text{Score}_{\text{USSR}}(R)$$
$$\text{victory\_points} \leftarrow \text{clamp}(\text{victory\_points} + \Delta\text{VP}, -20, +20)$$

### 7.2 Automatic Victory Conditions
1. **VP Threshold**: Instant victory if $\text{victory\_points} = +20$ (US) or $\text{victory\_points} = -20$ (USSR).
2. **Europe Control**: Instant victory if either player achieves Control of Europe when `#2 Europe Scoring` is played or during Final Scoring.
3. **Nuclear War**: Instant victory for opponent when phasing player causes $\text{defcon} = 1$.
4. **Held Scoring Card**: Instant loss at Phase F if holding a scoring card.
5. **Wargames (#107)**: Phasing player may give opponent $6\text{ VP}$ at DEFCON 2 to immediately end the game and evaluate current VP.

---

## 8. DEFCON Regional Restrictions Table

| DEFCON Level | Realignment / Coup Prohibitions | Military Ops Required | Status Name |
| :---: | :--- | :---: | :--- |
| **5** | None | 5 | Peace |
| **4** | Europe prohibited | 4 | DEFCON 4 |
| **3** | Europe, Asia prohibited | 3 | DEFCON 3 |
| **2** | Europe, Asia, Middle East prohibited | 2 | DEFCON 2 |
| **1** | Immediate Game Over (Phasing Player Loses) | 1 | Nuclear War |

---

## 9. Special Card Decision State Machine Rules

### 9.1 The China Card (#6)
- Starts face-up with USSR.
- Provides $4\text{ Ops}$, or $5\text{ Ops}$ if all points are spent in Asia / Southeast Asia.
- Passed face-down to opponent upon play; flipped face-up at turn end.
- Cannot be played in Headline phase.
- Cannot be played if doing so prevents playing a Scoring card in hand.
- Cannot be discarded by events.
- Awards $+1\text{ VP}$ to holder at end of Turn 10.

### 9.2 UN Intervention (#32)
- Played as `PLAY_EVENT`.
- Prompts `SELECT_CARD` for an opponent-affiliated card in hand.
- Both cards are discarded; phasing player uses the opponent card's Ops value with **zero** opponent event triggering.

### 9.3 Quagmire (#42) & Bear Trap (#44)
- Trapped player must discard a card with $\text{Ops} \ge 2$ on each Action Round and roll $1d6$:
  - On $1\text{--}4$: Flag cleared, player escapes, Action Round ends.
  - On $5\text{--}6$: Remains trapped, Action Round ends.
- If player holds no $2+\text{ Ops}$ cards, player must play all held Scoring cards, then sits out remaining ARs.

### 9.4 Missile Envy (#49)
- Opponent gives highest Ops card to player.
- If opponent card is returned, opponent is forced to play `#49 Missile Envy` for $2\text{ Ops}$ on their next Action Round.

### 9.5 Re-entrant Sub-Events (Decision Stack Depth $\le 3$)
Events triggering nested card plays (e.g. `#85 Star Wars`, `#5 Five Year Plan`) push a new `DecisionContext` onto `GameState::ctx_stack`, execute child micro-decisions, and pop back cleanly upon completion with zero heap allocations.
