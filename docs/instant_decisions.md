# Instant-Win and Instant-Loss Decisions

Decisions that end the game immediately. Unlike strategic claims, these need no answer key:
either the engine reaches `Phase::GAME_OVER` with the acting player losing, or it does not.
Every entry below was verified by constructing the position with `ai/eval/positions.py` and
stepping the engine to a terminal state; the observed `victory_points` and
`get_terminal_utility` are recorded.

**The DEFCON-1 loser is always the phasing player** (`ts/defcon.hpp`), regardless of whose
event fired. That single rule generates most of this catalogue.

**An opponent-associated card can never be played as an event.** `action_mask.cpp` offers
EVENT mode only for your own or neutral cards, so playing an opponent's card for Ops is
what fires their event — unavoidably. For those cards the real choice is Ops vs Space vs
not selecting the card at all.

---

## Category 1 — UNCONDITIONAL instant loss (DEFCON-degrading event at DEFCON 2)

The event degrades DEFCON with no intervening choice by anyone. Firing it at DEFCON 2 ends
the game on the spot. These are the strongest possible test claims: no opponent modelling,
no board conditions beyond DEFCON.

| card | side | acting player | how it fires | verified terminal |
|---|---|---|---|---|
| #4 Duck and Cover | US | USSR plays for **Ops** | US event fires under USSR's AR | defcon=1, VP=+20, util=+1.0 → **USSR loses** |
| #4 Duck and Cover | US | US plays for **event** | own event | defcon=1, VP=−20, util=−1.0 → **US loses** |
| #50 We Will Bury You | USSR | US plays for **Ops** | USSR event fires under US's AR | defcon=1, VP=−20, util=−1.0 → **US loses** |
| #50 We Will Bury You | USSR | USSR plays for **event** | own event | defcon=1, VP=+20, util=+1.0 → **USSR loses** |
| #89 Soviets Shoot Down KAL-007 | US | USSR plays for **Ops** | US event fires under USSR's AR | defcon=1, VP=+20, util=+1.0 → **USSR loses** |
| #89 Soviets Shoot Down KAL-007 | US | US plays for **event** | own event | defcon=1, VP=−20, util=−1.0 → **US loses** |
| #46 How I Learned to Stop Worrying | NEUTRAL | either, for **event** | sets DEFCON directly via `clamp(primary_id,1,5)` | defcon=1, VP=−20 → **player loses** |

Engine sites: `early_war.cpp:18` (Duck and Cover), `mid_war.cpp:210` (We Will Bury You),
`late_war.cpp:94` (KAL-007), `card_dispatcher.cpp:764` (How I Learned).

**Precondition for all:** `defcon == 2`, card in hand, and for the Ops variants the acting
player is the phasing player. Space is the safe disposal for an opponent card.

---

## Category 2 — CONDITIONAL instant loss (opponent gains a winning option)

The event does not itself lose the game; it hands the OPPONENT a free coup. If they coup a
battleground at DEFCON 2, DEFCON falls to 1 and the phasing player — the one who played the
card — loses. Confirmed by exhaustive branch search: a losing line exists, but it requires
the opponent to take it.

**This is a materially weaker claim than Category 1** and should be tested separately: it
assumes a competent opponent. Against HeuristicBot, which does not take these lines, the
play is not actually punished — which is part of why agents never learned to avoid it.

| card | side | acting player | precondition | result |
|---|---|---|---|---|
| #91 Ortega Elected in Nicaragua | USSR | US plays for Ops | US influence in Cuba | defcon=1, VP=−20 → **US loses** (reproduced greedily, not only in search) |
| #62 Lone Gunman | USSR | US plays for Ops | US influence in a coup-eligible battleground | losing line exists: defcon=1, VP=−20 |
| #67 Grain Sales to Soviets | US | USSR plays for Ops | USSR influence in a coup-eligible battleground | losing line exists: defcon=1, VP=+20 |
| #96 Tear Down This Wall | US | USSR plays for Ops | USSR influence in a European battleground (E. or W. Germany both reproduce) | losing line exists: defcon=1, VP=+20 |
| #20 Olympic Games | NEUTRAL | either, for event | opponent chooses to **boycott** | losing line exists both directions: defcon=1, VP=∓20 |

Olympic Games sits here rather than in Category 1 because the DEFCON drop requires the
opponent to pick the boycott branch (`card_dispatcher.cpp:452`). A greedy rollout that takes
the participate branch never reaches the loss, which is why it first appeared safe.

---

## Category 3 — Couping a battleground at DEFCON 2

`ops.cpp:215` degrades DEFCON on any battleground coup; `ops.cpp:117` implements the Rule
8.1.5 regional restrictions that forbid coups in Europe / Asia / Middle East at DEFCON 2.
So the losing move is couping a battleground in a region still permitted at DEFCON 2 —
Africa and Central/South America. `observation.cpp:88` already exposes a per-country
"couping here causes DEFCON 1" flag, so the network is given the answer as an input feature.

**Precondition:** `defcon == 2`, acting player is phasing, target is a battleground in an
unrestricted region.

---

## Category 4 — Wargames (#100)

`late_war.cpp:259` offers the branch only at DEFCON 2; `card_dispatcher.cpp:878` awards the
**opponent** 6 VP and ends the game.

| your VP lead | after −6 | outcome |
|---|---|---|
| ≥ 7 | still ahead | **win** — should trigger |
| exactly 6 | 0 | draw |
| ≤ 5 | behind | **loss** — must not trigger |

**Correction to the original claim:** the losing threshold is a lead of *less than 6*, not
less than 5; a lead of exactly 6 draws. The winning threshold of 7+ was correct.

---

## Category 5 — Held scoring card

Holding a scoring card at end of turn loses the game (`Engine::is_held_scoring_loss`). The
qualification is real: if playing that scoring card would itself have lost the game on VP,
the choice is between two losses and the claim is vacuous. A test must therefore assert the
play only when scoring it does NOT cross the player's own −20 threshold.

---

## Category 6 — VP threshold crossings

Any VP award that takes the running total to +20 (US) or −20 (USSR) ends the game
immediately — the engine clamps to ±20 and sets `GAME_OVER` at each award site. This makes
every VP-awarding event a potential instant win or instant loss depending on the current
score, and it generalises the scoring-card case in Category 5.

**Not yet enumerated.** The full set of VP-awarding events with their amounts, and whether
each is fixed or board-dependent, is the main gap in this catalogue. The enumeration needs
the award amount and its bounds per card, since the precondition is only checkable if the
award can be bounded.

---

## MISSED GUARANTEED WINS (symmetric claims)

Failing to take a forced win is a distinct failure from committing a blunder, and no
measurement in this repo currently covers it.

1. **Playable scoring card that reaches your 20 VP threshold** — play it.
2. **Wargames at DEFCON 2 with a 7+ VP lead** — trigger it.
3. **Opponent offers Olympic Games at DEFCON 2, or fires an event granting you a coup, and
   they hold influence in an eligible battleground** — take the coup and win.
4. **Missile Envy (#49) at DEFCON 2** — if your highest-Ops card is an opponent event that
   unconditionally degrades DEFCON (Category 1 only, not the coup-granting cards), handing
   it over fires the event under *their* action round and they lose. Needs verification of
   who is phasing at resolution time; **not yet verified**.

---

## DISPROVED OR QUALIFIED

- **CIA Created (#26) is a US card, not a USSR one.** The original claim listed it among
  opponent events granting a coup. For the US it is their own card, and no losing line was
  found for the US playing it for Ops with influence in Cuba. The reverse framing (USSR
  plays it for Ops, firing the US event) is the checkable one and is not yet verified.
- **Wargames threshold** — off by one, see Category 4.
- **Olympic Games** — conditional on the opponent's boycott choice, so Category 2 not 1.

## NOT YET VERIFIED

Missile Envy hand-over (claim 4 above); the CIA Created reverse framing; the full VP-award
enumeration in Category 6; and whether every coup-granting event reproduces for both sides.
