"""Encoded behavioural claims from docs/behavioral_test_claims.md.

Only claims whose precondition is mechanically checkable and whose answer key survived
review are encoded here. Each carries its claim id so a failure points straight back at
the documented rationale.

Tiers separate two very different kinds of mistake:

* ``L`` -- **instant loss**. The action ends the game immediately with the acting player
  losing: firing a DEFCON-degrading event at DEFCON 2, or handing the opponent a free
  battleground coup there. Verified by stepping the engine to a terminal state. These are
  objective and catastrophic, so they carry a much tighter probability ceiling.
* ``S`` -- **strategic waste**. The action is poor value but not fatal: burning a 4-Ops
  card on a weak event, or triggering an event whose effect is already achieved. Real
  mistakes, but a policy can be competent overall while occasionally making them.
* ``B`` -- **comparative preference**. An ordering between two options in one position.

Mixing L and S in a single pass rate hides the thing that matters most, since a policy
that never loses outright but wastes Ops is in far better shape than the reverse.

Note on phrasing: an opponent-associated card can never be played *as* an event
(`action_mask.cpp` offers EVENT mode only when `!is_opponent_card`). Playing it for Ops is
what fires the owner's event. So claims about "USSR triggering a US event" are encoded as
choices between Ops and Space, never as an event choice the player does not have.
"""

from typing import List

import ts_engine as ts

from ai.eval.behavioral_suite import BehavioralTest, NeverArgmax, Prefer
from ai.eval.positions import PLAY_MODE_ACTION, PositionBuilder, card_action

# Card ids
NATO = 21
MARSHALL_PLAN = 23
US_JAPAN = 27
WARSAW_PACT = 16
DE_GAULLE = 17
DUCK_AND_COVER = 4
OLYMPIC_GAMES = 20
RED_SCARE = 31
COMECON = 14
ROMANIAN_ABDICATION = 12
FIVE_YEAR_PLAN = 5
NUCLEAR_TEST_BAN = 34
ORTEGA = 91

# Country ids
FRANCE = 8
JAPAN = 45
POLAND = 15
EAST_GERMANY = 14
CUBA = 71

# An instant loss must be all but absent from the distribution; strategic waste is a
# softer claim, so it gets the looser ceiling.
LOSS_CEILING = 0.05
STRATEGY_CEILING = 0.20

EVENT = PLAY_MODE_ACTION["event"]
OPS = PLAY_MODE_ACTION["ops"]
SPACE = PLAY_MODE_ACTION["space"]


def build_claims() -> List[BehavioralTest]:
    tests: List[BehavioralTest] = []

    # --- a-03: US should not waste NATO's 4 Ops on its weak event. -------------------
    # Reviewer scoped this to a hard invariant on turns 1-2 while US VP < 10.
    tests.append(BehavioralTest(
        claim_id="a-03-nato-us-never-event",
        description="US must not play NATO for the event (turn <= 2, US VP < 10)",
        builder=PositionBuilder(
            hand=[NATO, MARSHALL_PLAN, US_JAPAN], side=ts.Player.US,
            flags=[ts.EffectBits.WARSAW_PACT_PLAYED], turn=2, defcon=3, victory_points=2),
        play_card_first=NATO,
        assertion=NeverArgmax(EVENT, "NATO as event", max_prob=STRATEGY_CEILING),
        tier="S",
    ))

    # --- a-04: US/Japan is only worth the event if the USSR actually holds Japan. ----
    tests.append(BehavioralTest(
        claim_id="a-04-us-japan-never-event-unless-ussr-controls-japan",
        description="US must not play US/Japan for the event while USSR does not control Japan",
        builder=PositionBuilder(
            hand=[US_JAPAN, MARSHALL_PLAN], side=ts.Player.US, turn=3,
            influence=[(JAPAN, ts.Player.US, 4)], clear_influence=[(JAPAN, ts.Player.USSR)]),
        play_card_first=US_JAPAN,
        assertion=NeverArgmax(EVENT, "US/Japan as event", max_prob=STRATEGY_CEILING),
        tier="S",
    ))

    # --- a-05: De Gaulle is pointless when the USSR already holds France. -----------
    tests.append(BehavioralTest(
        claim_id="a-05-de-gaulle-ussr-never-event-when-controlling-france",
        description="USSR must not play De Gaulle for the event while already controlling France",
        builder=PositionBuilder(
            hand=[DE_GAULLE, COMECON], side=ts.Player.USSR, turn=2,
            influence=[(FRANCE, ts.Player.USSR, 6)], clear_influence=[(FRANCE, ts.Player.US)]),
        play_card_first=DE_GAULLE,
        assertion=NeverArgmax(EVENT, "De Gaulle as event", max_prob=STRATEGY_CEILING),
        tier="S",
    ))

    # --- n-01: burn NATO while its event cannot fire. -------------------------------
    # With neither Warsaw Pact nor Marshall Plan played, NATO's event is inert
    # (card_dispatcher.cpp:139), so the USSR gets a free 4 Ops and denies it later.
    tests.append(BehavioralTest(
        claim_id="n-01-ussr-burn-nato-while-inert",
        description="USSR should spend NATO before another 4-Ops card while NATO is inert",
        builder=PositionBuilder(
            hand=[NATO, NUCLEAR_TEST_BAN], side=ts.Player.USSR, turn=2, defcon=4),
        assertion=Prefer(card_action(NATO), card_action(NUCLEAR_TEST_BAN),
                         "play NATO", "play Nuclear Test Ban"),
        tier="B",
    ))

    # --- n-02: even once live, NATO goes before the other big US cards. -------------
    tests.append(BehavioralTest(
        claim_id="n-02-ussr-nato-before-marshall-plan",
        description="USSR should play NATO before Marshall Plan once NATO is live",
        builder=PositionBuilder(
            hand=[NATO, MARSHALL_PLAN], side=ts.Player.USSR, turn=3,
            flags=[ts.EffectBits.WARSAW_PACT_PLAYED]),
        assertion=Prefer(card_action(NATO), card_action(MARSHALL_PLAN),
                         "play NATO", "play Marshall Plan"),
        tier="B",
    ))

    tests.append(BehavioralTest(
        claim_id="n-02b-ussr-nato-before-us-japan",
        description="USSR should play NATO before US/Japan once NATO is live",
        builder=PositionBuilder(
            hand=[NATO, US_JAPAN], side=ts.Player.USSR, turn=3,
            flags=[ts.EffectBits.WARSAW_PACT_PLAYED]),
        assertion=Prefer(card_action(NATO), card_action(US_JAPAN),
                         "play NATO", "play US/Japan"),
        tier="B",
    ))

    # --- c-03 (reviewer-conditioned): Warsaw Pact is a deterrent, not an event. ------
    # Reviewer: only when the US is demonstrably not contesting Eastern Europe --
    # no US influence in Poland or East Germany and at least 3 USSR in each.
    tests.append(BehavioralTest(
        claim_id="c-03-warsaw-pact-ussr-not-event-when-ee-uncontested",
        description="USSR must not trigger Warsaw Pact while Eastern Europe is uncontested",
        builder=PositionBuilder(
            hand=[WARSAW_PACT, COMECON], side=ts.Player.USSR, turn=2,
            influence=[(POLAND, ts.Player.USSR, 4), (EAST_GERMANY, ts.Player.USSR, 4)],
            clear_influence=[(POLAND, ts.Player.US), (EAST_GERMANY, ts.Player.US)]),
        play_card_first=WARSAW_PACT,
        assertion=NeverArgmax(EVENT, "Warsaw Pact as event", max_prob=STRATEGY_CEILING),
        tier="S",
    ))

    # --- a-01: Duck and Cover at DEFCON 2 -- the blunder is Ops, not holding the card.
    # Duck and Cover is US-owned, so the USSR's legal modes here are exactly Ops and
    # Space. Choosing Ops fires the US event whichever timing branch is taken, dropping
    # DEFCON to 1 and losing the game for the phasing USSR (verified by stepping the
    # engine to a terminal state: defcon=1, VP=+20, utility +1.0 to the US).
    #
    # Selecting the card is NOT itself a mistake: sending it to space discards it without
    # firing the event, which is the standard way to dispose of a dangerous opponent card.
    # So the assertion lives at the play-mode node, and the position must offer space --
    # otherwise "prefer space" would be vacuous.
    tests.append(BehavioralTest(
        claim_id="a-01-duck-and-cover-ussr-space-not-ops-at-defcon2",
        description="USSR must space Duck and Cover rather than play it for Ops at DEFCON 2",
        builder=PositionBuilder(
            hand=[DUCK_AND_COVER, COMECON], side=ts.Player.USSR, turn=4, defcon=2),
        play_card_first=DUCK_AND_COVER,
        assertion=NeverArgmax(OPS, "Duck and Cover for Ops", max_prob=LOSS_CEILING),
        requires_legal=(SPACE,),
        tier="L",
    ))

    # --- a-02: Olympic Games at DEFCON 2, own card, so a genuine event choice. ------
    tests.append(BehavioralTest(
        claim_id="a-02-olympic-games-never-event-at-defcon2",
        description="US must not play Olympic Games for the event at DEFCON 2",
        builder=PositionBuilder(
            hand=[OLYMPIC_GAMES, MARSHALL_PLAN], side=ts.Player.US, turn=4, defcon=2),
        play_card_first=OLYMPIC_GAMES,
        assertion=NeverArgmax(EVENT, "Olympic Games as event", max_prob=LOSS_CEILING),
        tier="L",
    ))

    # --- a-15: Ortega at DEFCON 2 while the US holds influence in Cuba. -------------
    # Ortega is USSR-owned, so the US choice is Ops or Space. Playing it for Ops fires the
    # USSR event, which grants a free coup adjacent to Nicaragua -- Cuba is a battleground,
    # so the coup drops DEFCON to 1 and the phasing US loses. Verified by stepping to a
    # terminal state: defcon=1, VP=-20, utility -1.0. Spacing it is the safe disposal.
    tests.append(BehavioralTest(
        claim_id="a-15-ortega-us-space-not-ops-at-defcon2",
        description="US must space Ortega rather than play it for Ops at DEFCON 2 with influence in Cuba",
        builder=PositionBuilder(
            hand=[ORTEGA, MARSHALL_PLAN], side=ts.Player.US, turn=7, defcon=2,
            influence=[(CUBA, ts.Player.US, 3)]),
        play_card_first=ORTEGA,
        assertion=NeverArgmax(OPS, "Ortega for Ops", max_prob=LOSS_CEILING),
        requires_legal=(SPACE,),
        tier="L",
    ))

    # --- b-01: Five Year Plan is a US card the US should simply spend. --------------
    tests.append(BehavioralTest(
        claim_id="b-01-five-year-plan-us-prefer-ops",
        description="US should play Five Year Plan for Ops rather than the event",
        builder=PositionBuilder(
            hand=[FIVE_YEAR_PLAN, MARSHALL_PLAN], side=ts.Player.US, turn=2),
        play_card_first=FIVE_YEAR_PLAN,
        assertion=Prefer(OPS, EVENT, "Five Year Plan for Ops", "for event"),
        tier="B",
    ))

    return tests
