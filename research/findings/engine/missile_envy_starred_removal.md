# Missile Envy removes a starred card whose Event never fired

**Status: diagnosed, not fixed.** Engine changes need the owner's approval (invariant 11). Found
while analysing the flattened-stack design, because the fix is exactly what that design proposes.

## The bug

Missile Envy hands the opponent's highest-Ops card to the player. If that card is the *opponent's*
own, the player uses it for Operations and **its Event does not occur**. A starred card is removed
from the game only when its Event occurs (`rules.md` 280-282), so it should be **discarded**.

It is removed from the game instead.

## Demonstration

Constructed, not found in a log — see "reachability" below.

```
USSR plays Missile Envy. The US hand holds NATO (id 21, 4 Ops, US-sided, starred).
Missile Envy takes it; the USSR spends its 4 Ops.

after trigger: decision = SELECT_OP_MODE  pending_op_card = NATO  timing_branch = 255
FINAL location of NATO: REMOVED_FROM_GAME      <-- expected DISCARD_PILE
```

NATO is gone from the game permanently, and its Event never happened.

## Cause

Two different questions are answered by two different fields, and only one of them was set.

`mid_war.cpp:180-188`, the use-for-Ops branch, sets `timing_branch = 255`. That is enough to stop
the Event **firing**: the guard at `state_machine.cpp:437` requires
`timing_branch == TimingBranch::OPS_FIRST`.

But the *relocation* decision asks a different function, `event_occurred_on_ops_play`
(`state_machine.cpp:318-322`):

```cpp
if (card == 0) return false;
if (!CardData::is_opponent_card(card, p)) return false;
return state.ctx().suppress_op_card_event == 0;
```

It never looks at `timing_branch`. The taken card *is* an opponent card, and Missile Envy never
sets `suppress_op_card_event`, so it reports the Event as having occurred and
`relocate_played_card` removes the card. The in-hand gate at `state_machine.cpp:461` is satisfied
because Missile Envy puts the taken card in the player's hand.

UN Intervention — the other card that plays an opponent's card without its Event — does set
`suppress_op_card_event = 1` (`card_dispatcher.cpp:770`), and is therefore correct. Missile Envy is
the same situation reached by a different route, and only one route sets the flag.

## Fix

One line: set `suppress_op_card_event = 1` in Missile Envy's use-for-Ops branch, alongside the
existing `timing_branch = 255`. That makes the two cards agree, and it is exactly the
"present it as an opponent card with the Event prevented" treatment the flattened-stack design
calls for — see [P17_stack_layout.md](../../plans/P17_stack_layout.md).

Worth considering separately: `event_occurred_on_ops_play` answering "did the Event occur?" without
consulting `timing_branch` is the underlying hazard. Any future path that blocks an Event by timing
rather than by flag inherits this bug.

## Reachability

The case above is constructed. It needs the opponent's highest-Ops card to be their own side and
starred — common, not exotic: NATO, Marshall Plan, De-Stalinization and Socialist Governments are
all starred and 3-4 Ops.

**Not yet confirmed in the human corpus.** A corpus sweep for a Missile Envy play whose taken card
is opponent-sided and starred would settle whether any recorded game is affected; if one is, the
converter would currently reproduce the wrong location and the board check would have caught it, so
the more likely reading is that it has not come up. That sweep is not done.
