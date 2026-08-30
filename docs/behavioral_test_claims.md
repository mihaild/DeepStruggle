# Twilight Struggle Behavioural Test-Suite Claims

Extracted from `/workspace/docs/card_strategies.md` (consolidated encyclopedia) and its era-split
sources `docs/strategy/early_war.md`, `mid_war.md`, `late_war.md`, `README.md`. Target competence
level: a human who has played a handful of games — not an expert. Deck-counting, reshuffle-timing,
and elaborate multi-turn setups that require reading the opponent's exact hand are deliberately
excluded (see "Excluded / out of scope" at the end).

## Summary

| Type | Count |
|---|---|
| A — hard invariant | 14 |
| B — comparative preference | 19 |
| C — rate tendency | 19 |
| N — reviewer-supplied | 6 |
| **Total** | **58** |

All claims carry a `review:` field recording the reviewer's verdict. Four extracted
claims were removed on review: `a-13` (Iron Lady, US never event), `a-16` (John Paul II,
US always event), `b-03` (US prefers losing West Germany), `b-07` (Defectors headline).

None of the claims below require observing the opponent's hidden hand — every precondition is
checkable from public game state (board control, DEFCON, turn number, Space Race track, which
starred events have already fired, and the *acting* player's own hand). Candidate claims that
genuinely required opponent hand-knowledge were excluded; see "Excluded / out of scope."

---

# TYPE A — Hard Invariants

### a-01-duck-cover-ussr-defcon2-suicide
- **cards:** #4 Duck and Cover
- **side:** USSR
- **era:** early (instance of a general DEFCON-suicide pattern)
- **precondition:** USSR to act, DEFCON == 2, Duck and Cover in hand, considering playing it for the event.
- **claim:** USSR must never play Duck and Cover for the event at DEFCON 2 — it degrades DEFCON to 1, an immediate USSR loss.
- **assertion:** never-argmax — P(USSR selects "play Duck and Cover as event") ≈ 0 when DEFCON==2.
- **confidence:** high
- **source:** #4 Duck and Cover, "As USSR" ("It is unplayable and one of the DEFCON suicide cards")
- **review:** include.

### a-02-olympic-games-defcon2-never-event
- **cards:** #20 Olympic Games
- **side:** either
- **era:** early (general)
- **precondition:** acting player holds Olympic Games, DEFCON == 2, considering playing it for the event.
- **claim:** A player must never play Olympic Games for the event at DEFCON 2 — a boycotting opponent degrades DEFCON to 1, an immediate loss for the player who played it.
- **assertion:** never-argmax — P(select "play Olympic Games as event") ≈ 0 when DEFCON==2.
- **confidence:** high
- **source:** #20 Olympic Games ("You can't play Olympic Games for the event when DEFCON is at 2 ... you will lose on DEFCON")
- **review:** include.

### a-03-nato-us-never-event
- **cards:** #21 NATO
- **side:** US
- **era:** early/mid
- **precondition:** US to act, NATO in hand and eligible to be played as an event (Warsaw Pact Formed or Marshall Plan already triggered).
- **claim:** US should essentially never play NATO for the event; it should always be played for its 4 Ops.
- **assertion:** never-argmax — P(US selects "play NATO as event") ≈ 0.
- **confidence:** high
- **source:** #21 NATO, "As US" ("I have never played this card for the event as the US ... You should never, however, play NATO to boost Special Relationship")
- **review:** hard invariant for turns 1-2 when US VP < 10. After - there can be some very rare situations where NATO is useful as event for US.

### a-04-us-japan-us-never-event-unless-ussr-controls-japan
- **cards:** #27 US/Japan Mutual Defense Pact
- **side:** US
- **era:** early/mid
- **precondition:** US to act, US/Japan Mutual Defense Pact in hand, USSR does NOT currently control Japan.
- **claim:** US should never play US/Japan Mutual Defense Pact for the event unless USSR currently controls Japan; otherwise it should be played for its 4 Ops.
- **assertion:** never-argmax — P(US selects "play as event") ≈ 0 when USSR does not control Japan.
- **confidence:** high
- **source:** #27, "As US" ("Unless the USSR has actually taken over Japan ... this is never worth playing for the event")
- **review:** include.

### a-05-de-gaulle-ussr-never-event-if-already-controls-france
- **cards:** #17 De Gaulle Leads France
- **side:** USSR
- **era:** early
- **precondition:** USSR to act, De Gaulle Leads France in hand, USSR already controls France.
- **claim:** USSR should never play De Gaulle Leads France for the event when it already controls France.
- **assertion:** never-argmax — P(USSR selects "play as event") ≈ 0 when USSR already controls France.
- **confidence:** high
- **source:** #17, "As USSR" ("If you already control France, then there is obviously no need to play De Gaulle")
- **review:** include.

### a-06-romanian-abdication-ussr-always-ops
- **cards:** #12 Romanian Abdication
- **side:** USSR
- **era:** early
- **precondition:** USSR to act, Romanian Abdication in hand; not a desperation headline against a suspected Defectors headline.
- **claim:** USSR should virtually always play Romanian Abdication for its 1 Op — no decent US opponent contests Romania.
- **assertion:** never-argmax (outside the narrow desperation-headline exception) — P(USSR plays for event) ≈ 0.
- **confidence:** high
- **source:** #12, "As USSR" ("this card is always played for Ops")
- **review:** include with addition "when taking Romania will not affect scoring".

### a-07-comecon-ussr-always-ops-except-t1-trap
- **cards:** #14 Comecon
- **side:** USSR
- **era:** early
- **precondition:** USSR to act, Comecon in hand, not executing the specific Turn-1 "Comecon Trap" opening (headline Comecon + AR1 realignment of West Germany/Italy).
- **claim:** Outside the narrow Turn-1 Comecon Trap, USSR should always play Comecon for its 3 Ops rather than the scattered 4-influence event.
- **assertion:** never-argmax outside the Turn-1 trap window — P(USSR plays for event) ≈ 0.
- **confidence:** high
- **source:** #14, "As USSR" ("I always play Comecon for Ops ... I have never seen any competent USSR player play this for the event" outside the Turn 1 trap)
- **review:** include for early and midwar. Can be useful for late war.

### a-08-voa-ussr-never-trigger
- **cards:** #74 The Voice of America
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds The Voice of America.
- **claim:** USSR should never voluntarily trigger Voice of America's event; prefer spacing it, or playing it for Ops only if forced.
- **assertion:** never-argmax — P(USSR selects "play VOA for event") ≈ 0.
- **confidence:** high
- **source:** #74, "As USSR" ("wholly unplayable ... never safe ... never mitigatable")
- **review:** include except in scenarios where USSR will have to play either that or defcon suicide.

### a-09-cultural-revolution-us-sequencing
- **cards:** #58 Cultural Revolution, #6 The China Card
- **side:** US
- **era:** mid
- **precondition:** US holds both The China Card and Cultural Revolution in the same hand/turn.
- **claim:** US should never trigger Cultural Revolution before first playing the China Card — playing China Card first, then Cultural Revolution, avoids ever handing it over.
- **assertion:** never-argmax — P(US triggers Cultural Revolution while still holding an unplayed China Card) ≈ 0.
- **confidence:** high
- **source:** #58, "As US" (play China Card first, no reason not to)
- **review:** include with "Holding Face-up China Card and VP < 15 in USSR favor".

### a-10-summit-defcon2-avoid
- **cards:** #45 Summit
- **side:** either
- **era:** mid
- **precondition:** DEFCON == 2, acting player holds Summit, player does not have a commanding lead in region control.
- **claim:** Playing Summit's event at DEFCON 2 should essentially never be chosen absent a massive positional lead.
- **assertion:** never-argmax under this precondition.
- **confidence:** high
- **source:** #45 ("foolhardy ... unless a massive lead in regions")
- **review:** include.

### a-11-star-wars-ussr-never-trigger
- **cards:** #85 Star Wars
- **side:** USSR
- **era:** late
- **precondition:** USSR holds Star Wars and it is eligible to be played as an event.
- **claim:** USSR should never play Star Wars for the event — the risk of the US freely picking any card from the discard pile is too great regardless of discard-pile contents.
- **assertion:** never-argmax — P(USSR triggers Star Wars event) ≈ 0.
- **confidence:** high (mechanical caveat: the card's stated eligibility condition references being "ahead on the Space Race," which needs verifying against the actual engine implementation for who can legally trigger it before hard-coding this test)
- **source:** #85, "As USSR" ("I have never seen a discard pile safe enough to play Star Wars as the USSR")
- **review:** include, after engine test, and when discard pile contains powerful US or neutral events. Hard-invariant on DEFCON 2 and discard pile contains card that degrade defcon, or when DEFCON is anything and discard pile contains "How I learned".
- **engine verified (2026-08-30):** `card_dispatcher.cpp:157` gates the event on
  `us_space_track > ussr_space_track`, and `late_war.cpp:41` hands the discard choice to
  `Player::US` regardless of who played the card. So the guide's "USSR triggering Star
  Wars" does mean the USSR playing it for Ops and the US picking. The event is a no-op
  when the US is not ahead, or when the discard pile holds no non-scoring card, so the
  precondition must include both. The reviewer's hard-invariant sub-case checks out: the
  US pick resolves as an event under the USSR's action round, so a DEFCON-degrading pick
  at DEFCON 2 -- or "How I Learned to Stop Worrying" (#87) at any DEFCON, which sets
  DEFCON directly via `clamp(primary_id, 1, 5)` -- ends the game with the phasing USSR
  losing.

### a-12-tdtw-ussr-always-space
- **cards:** #96 Tear Down this Wall
- **side:** USSR
- **era:** late
- **precondition:** USSR holds Tear Down this Wall.
- **claim:** USSR should always send Tear Down this Wall to the Space Race rather than playing it for Operations.
- **assertion:** never-argmax — P(USSR plays #96 for Ops instead of spacing) ≈ 0.
- **confidence:** high
- **source:** #96, "As USSR" ("The USSR always sends Tear Down This Wall to space")
- **review:** on DEFCON 2 when US has influence in European Batlleground.

### a-14-awacs-us-no-value-when-mr-gone-and-us-controls-sa
- **cards:** #110 AWACS Sale to Saudis, #56 Muslim Revolution
- **side:** US
- **era:** late
- **precondition:** US holds AWACS; Muslim Revolution has already been played/removed from the deck; US already controls Saudi Arabia.
- **claim:** Under these conditions, triggering AWACS's event accomplishes nothing and should never be chosen over playing it for Ops.
- **assertion:** never-argmax under this precondition.
- **confidence:** high
- **source:** #110, "As US" ("there is definitely no point to wasting the 3 Ops")
- **review:** include.

### a-15-ortega-us-defcon-suicide
- **cards:** #91 Ortega Elected in Nicaragua
- **side:** US
- **era:** late
- **precondition:** US would trigger Ortega's event, DEFCON == 2, US currently has influence in Cuba.
- **claim:** Triggering Ortega Elected in Nicaragua's event under this precondition is a DEFCON-suicide-equivalent blunder and must never be chosen.
- **assertion:** never-argmax under this precondition.
- **confidence:** high
- **source:** #91, "As US" ("Ortega is the hipster's DEFCON suicide card. If you have any influence in Cuba, then Ortega is unplayable at DEFCON 2")
- **review:** include.

### b-01-five-year-plan-us-prefer-ops
- **cards:** #5 Five Year Plan
- **side:** US
- **era:** early
- **precondition:** US to act, Five Year Plan in hand, no specific plan to force a USSR DEFCON-suicide-card discard.
- **claim:** As US in the Early War, prefer playing Five Year Plan for its 3 Ops over triggering its event — the event risks randomly discarding (and auto-triggering) a US card like Duck and Cover into a bad DEFCON spot.
- **assertion:** P(US plays for Ops) > P(US plays for event) absent a specific discard-forcing plan.
- **confidence:** high
- **source:** #5, "As US" ("you should always play this for operations ... risks drawing Duck and Cover and losing the game")
- **review:** include.

### b-02-blockade-ussr-prefer-final-ar-over-headline
- **cards:** #10 Blockade
- **side:** USSR
- **era:** early
- **precondition:** USSR to act, Blockade in hand.
- **claim:** As USSR, prefer playing Blockade's event on the final Action Round of a turn over headlining it or playing it mid-turn.
- **assertion:** P(USSR plays Blockade on the final AR) > P(USSR headlines Blockade or plays it mid-turn).
- **confidence:** high
- **source:** #10, "As USSR" ("Beginner USSR players will do things like headline Blockade or play it in the middle of a turn. Better USSR players play Blockade ... on their final Action Round")
- **review:** include.

### b-04-containment-us-prefer-headline
- **cards:** #25 Containment
- **side:** US
- **era:** early
- **precondition:** US to act in the Headline Phase, Containment in hand.
- **claim:** As US, strongly prefer headlining Containment over playing it for Ops during an Action Round.
- **assertion:** P(US headlines Containment | in hand at headline selection) > P(US plays it for Ops that turn instead).
- **confidence:** high
- **source:** #25, "As US" ("One of the four great US headlines in the Early War ... I almost always try to headline Containment")
- **review:** include.

### b-05-marshall-plan-us-prefer-event-early
- **cards:** #23 Marshall Plan
- **side:** US
- **era:** early
- **precondition:** US to act, Marshall Plan in hand, still in the Early War (ideally Turn 1-2).
- **claim:** Unlike most 4-Ops US cards, US should prefer triggering Marshall Plan's event over plain Ops in the Early War — letting it survive to a later reshuffle wastes most of its value.
- **assertion:** P(US plays Marshall Plan for the event in the Early War) > P(US plays it for Ops in the Early War).
- **confidence:** high
- **source:** #23, "As US" ("the only starred US event in the Early War that I will almost always play for the event")
- **review:** include.

### b-06-decolonization-prefer-africa-over-sea
- **cards:** #30 Decolonization
- **side:** USSR
- **era:** early/mid
- **precondition:** USSR to act, Decolonization in hand, both Africa and Southeast Asia have open placement options.
- **claim:** USSR should prefer allocating most Decolonization influence to Africa over Southeast Asia — Africa has more open battlegrounds and is otherwise much harder for USSR to break into.
- **assertion:** P(majority of Decolonization influence placed in Africa) > P(majority placed in Southeast Asia) when both are open.
- **confidence:** high
- **source:** #30, "Where should I Decolonize into?" ("As much as possible, you want to Decolonize into Africa rather than Southeast Asia")
- **review:** include for early war.

### b-08-grain-sales-ussr-top-space-priority
- **cards:** #67 Grain Sales to Soviets
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds Grain Sales to Soviets plus at least one other spaceable card; a Space Race slot is available.
- **claim:** USSR should prefer spacing Grain Sales to Soviets over any other card in hand.
- **assertion:** P(USSR spaces Grain Sales) > P(USSR spaces a different available card).
- **confidence:** high
- **source:** #67, "As USSR"
- **review:** include.

### b-09-cultural-revolution-ussr-claim-card
- **cards:** #58 Cultural Revolution, #6 The China Card
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds Cultural Revolution, US holds The China Card, USSR can spare 3 Ops.
- **claim:** USSR should prefer triggering Cultural Revolution's event (claiming the China Card face-up) over playing it for Ops.
- **assertion:** P(USSR triggers event) > P(USSR plays for Ops).
- **confidence:** medium-high
- **source:** #58, "As USSR"
- **review:** include.

### b-10-sadat-camp-david-order
- **cards:** #65 Camp David Accords, #72 Sadat Expels Soviets
- **side:** US
- **era:** mid
- **precondition:** US holds both cards in the same turn, USSR has influence in Egypt.
- **claim:** US should play Camp David Accords before Sadat Expels Soviets.
- **assertion:** P(US plays Camp David before Sadat) > P(US plays the reverse order).
- **confidence:** medium-high
- **source:** #72, "As US"
- **review:** include.

### b-11-brush-war-target-2-stability
- **cards:** #36 Brush War
- **side:** either
- **era:** mid
- **precondition:** Brush War available; both a hard-to-coup 2-stability battleground and an easy 1-stability African battleground are legal, comparably isolated targets.
- **claim:** Prefer targeting the harder-to-coup 2-stability battleground with Brush War over the easy 1-stability one — the latter can usually be taken by ordinary means anyway.
- **assertion:** P(target the 2-stability battleground) > P(target the 1-stability battleground).
- **confidence:** medium
- **source:** #36
- **review:** include.

### b-12-alliance-progress-ussr-threshold
- **cards:** #78 Alliance for Progress
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds Alliance for Progress; compare US-controlled Central/South American battleground count to a threshold of 2.
- **claim:** USSR should trigger the event when US controls ≤2 Central/South American battlegrounds, and should space it instead when US controls more than 2.
- **assertion:** P(USSR triggers event) > P(USSR spaces it) when US battlegrounds ≤2; the ordering reverses when US battlegrounds > 2.
- **confidence:** medium-high
- **source:** #78, "As USSR"
- **review:** include.

### b-13-south-african-unrest-ussr-split
- **cards:** #53 South African Unrest
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds South African Unrest; USSR does not already control both Angola and Botswana.
- **claim:** USSR should prefer a "1 into South Africa + split across neighbors" allocation over dumping both influence into South Africa alone.
- **assertion:** P(USSR splits the 2 influence between South Africa and a neighbor) > P(USSR places both into South Africa).
- **confidence:** high
- **source:** #53, "As USSR"
- **review:** include unless 2 influence breaks US control of South Africa and 1 doesn't.

### b-14-arms-race-vp-late-mid-war
- **cards:** #39 Arms Race
- **side:** either
- **era:** mid
- **precondition:** turn ≥ 6; acting player's Military Ops track ≥ opponent's; using the 3 Ops instead would not flip an additional region Domination.
- **claim:** Under these conditions, prefer taking the 3 VP for Arms Race superiority over playing the card for 3 Ops.
- **assertion:** P(player takes the VP) > P(player plays it for Ops).
- **confidence:** high
- **source:** #39
- **review:** include.

### b-15-reformer-ussr-prefer-event
- **cards:** #87 The Reformer
- **side:** USSR
- **era:** late
- **precondition:** USSR holds The Reformer.
- **claim:** As USSR, prefer playing The Reformer for the event over playing it for Operations.
- **assertion:** P(USSR plays for event) > P(USSR plays for Ops).
- **confidence:** high
- **source:** #87, "As USSR" ("An outstanding event ... can be used either on offense, defense, or both")
- **review:** include.

### b-16-reformer-us-always-space
- **cards:** #87 The Reformer
- **side:** US
- **era:** late
- **precondition:** US holds The Reformer.
- **claim:** As US, prefer sending The Reformer to the Space Race over playing it for Operations.
- **assertion:** P(US spaces it) > P(US plays it for Ops).
- **confidence:** high
- **source:** #87, "As US" ("Send Mr. Gorbachev to space, every time")
- **review:** include.

### b-17-tdtw-us-realign-not-coup
- **cards:** #96 Tear Down this Wall
- **side:** US
- **era:** late
- **precondition:** US plays #96 for the event; free-action target is a European battleground other than Italy (i.e. East Germany, France, or West Germany).
- **claim:** When using Tear Down this Wall's free action against a European battleground other than Italy, US should realign rather than coup.
- **assertion:** P(US realigns) > P(US coups) for targets in {East Germany, France, West Germany}.
- **confidence:** high
- **source:** #96, "As US" ("You almost always want to realign, since only Italy among the battlegrounds has a low enough stability to consider couping")
- **review:** include with condition "US has positive realignment modifier or no influence in target country".

### b-18-glasnost-conditional-on-reformer
- **cards:** #90 Glasnost, #87 The Reformer
- **side:** USSR
- **era:** late
- **precondition:** USSR holds Glasnost; check whether The Reformer has already been played this game.
- **claim:** USSR should prefer triggering Glasnost's event over spacing it once The Reformer has already been played; should prefer NOT triggering it (space or hold) while The Reformer has not yet been played, absent a pressing VP need.
- **assertion:** P(USSR plays Glasnost for event | Reformer already played) > P(USSR plays for event | Reformer not yet played).
- **confidence:** high
- **source:** #90, "As USSR"
- **review:** include.

### b-19-kal007-ussr-avoid-triggering-at-defcon3
- **cards:** #89 Soviets Shoot Down KAL-007
- **side:** USSR
- **era:** late
- **precondition:** USSR holds #89, DEFCON == 3.
- **claim:** As USSR, prefer playing #89 for Ops or spacing it over triggering the event even at DEFCON 3 — triggering hands the US 2 VP.
- **assertion:** P(USSR plays for Ops or spaces) > P(USSR triggers the event).
- **confidence:** medium
- **source:** #89, "As USSR"
- **review:** include as "P(USSR plays for Ops) < P(USSR spaces)". USSR can't trigger US event directly.

### b-20-iron-lady-ussr-prefer-event
- **cards:** #83 The Iron Lady
- **side:** USSR
- **era:** late
- **precondition:** USSR holds The Iron Lady.
- **claim:** As USSR, prefer playing The Iron Lady for the event over playing it for Operations — the cost (1 VP, trivial UK influence, loses access to Socialist Governments) is minimal.
- **assertion:** P(USSR plays for event) > P(USSR plays for Ops).
- **confidence:** high
- **source:** #83, "As USSR".
- **review:** include as P(USSR plays for ops) > P(USSR spaces).

### b-21-awacs-us-trigger-when-mr-live-and-ussr-holds-sa
- **cards:** #110 AWACS Sale to Saudis, #56 Muslim Revolution
- **side:** US
- **era:** late
- **precondition:** US holds AWACS; Muslim Revolution has not yet been played/discarded; USSR currently controls Saudi Arabia.
- **claim:** As US, prefer triggering AWACS's event over playing it for Ops — it both repairs Saudi Arabia and permanently blocks Muslim Revolution.
- **assertion:** P(US plays for event) > P(US plays for Ops).
- **confidence:** high
- **source:** #110, "As US" scenario 1
- **review:** include.

### c-01-china-card-ussr-avoid-turn1-unless-necessary
- **cards:** #6 The China Card
- **side:** USSR
- **era:** early
- **precondition:** Turn 1, USSR to act, The China Card in hand, an alternative card can accomplish the needed play (e.g. the Iran coup).
- **claim:** USSR should rarely play the China Card on Turn 1 unless no other card can accomplish the needed play — doing so forecloses holding it to protect the USSR's hand into Turn 3.
- **assertion:** rate below — frequency of USSR playing The China Card on Turn 1 (when an alternative exists) should be low.
- **confidence:** medium
- **source:** #6, "2. Protecting your hand" ("I try not to use the China Card on Turn 1 ... unless absolutely necessary")
- **review:** include.

### c-02-korean-war-us-trigger-asap
- **cards:** #11 Korean War
- **side:** US
- **era:** early
- **precondition:** US to act, Korean War in hand.
- **claim:** US should generally trigger Korean War as soon as reasonably possible rather than delaying — the odds and stakes only worsen for the US over time.
- **assertion:** rate above — frequency of US triggering Korean War promptly after drawing it (rather than holding/delaying) should be high.
- **confidence:** medium
- **source:** #11, "As US" ("You usually want to play Korean War as fast as you can")
- **review:** include.

### c-03-warsaw-pact-ussr-rarely-triggers-early
- **cards:** #16 Warsaw Pact Formed
- **side:** USSR
- **era:** early/mid
- **precondition:** USSR to act, Warsaw Pact Formed in hand, still Early or Mid War (before Turn 7).
- **claim:** USSR should rarely trigger Warsaw Pact Formed's event before the Late War, preferring to spend it for Ops and preserve it as a standing deterrent against the eventual US Late-War Europe push.
- **assertion:** rate below — frequency of USSR triggering the event before Turn 7 should be low.
- **confidence:** high
- **source:** #16, "As USSR" ("I always play Warsaw Pact Formed for Ops in the Early War / Mid War so that I can keep it in the deck")
- **review:** include with condition "no US influence in Poland and East Germany, at least 3 USSR influence in each".

### c-04-red-scare-purge-almost-always-event
- **cards:** #31 Red Scare/Purge
- **side:** either
- **era:** early/mid
- **precondition:** either side to act, Red Scare/Purge in hand.
- **claim:** Red Scare/Purge should almost always be played for its event rather than for plain Ops, regardless of side.
- **assertion:** rate above — frequency of playing for the event (vs. Ops) should be very high (e.g. >85%).
- **confidence:** high
- **source:** #31 ("I will almost always play Red Scare/Purge for the event")
- **review:** check only headlining + AR1 frequency. Add invariant test that model doesn't play Red Scare/Purge for event on last AR in round.

### c-05-nuclear-test-ban-almost-always-ops
- **cards:** #34 Nuclear Test Ban
- **side:** either
- **era:** early/mid
- **precondition:** either side to act, Nuclear Test Ban in hand, DEFCON not unusually high and no DEFCON-suicide card being sheltered.
- **claim:** Nuclear Test Ban should almost always be played for its 4 Ops rather than the event — the VP swing from the event is rarely favorable.
- **assertion:** rate above — frequency of playing for Ops (vs. event) should be very high (the guide's own estimate is ~98%).
- **confidence:** high
- **source:** #34 ("98% of the time, this is just a powerful 4Ops card whose event text you skip over")
- **review:** include.

### c-06-de-stalinization-ussr-only-at-defcon2-rarely-headline
- **cards:** #33 De-Stalinization
- **side:** USSR
- **era:** early
- **precondition:** USSR to act, De-Stalinization in hand.
- **claim:** USSR should generally only trigger De-Stalinization's event when DEFCON ≤ 2 (so the US cannot immediately coup the moved influence), and should rarely headline it given the risk of losing it to Defectors.
- **assertion:** rate above — frequency of USSR playing the event at DEFCON≤2 (vs. higher DEFCON) should be high; rate below — frequency of headlining it should be low.
- **confidence:** high
- **source:** #33, "As USSR" ("you have to play this at DEFCON 2 ... I also rarely headline this card")
- **review:** include.

### c-07-quagmire-self-play-avoid
- **cards:** #42 Quagmire
- **side:** US
- **era:** mid
- **precondition:** US holds Quagmire; US is not forced into it as the only way to avoid a worse discard situation.
- **claim:** US should send Quagmire to space rather than trigger it on itself, except when forced.
- **assertion:** rate below — US self-plays Quagmire's event in well under 10% of eligible turns.
- **confidence:** high
- **source:** #42, "As US"
- **review:** include.

### c-08-grain-sales-us-headline-preference
- **cards:** #67 Grain Sales to Soviets
- **side:** US
- **era:** mid
- **precondition:** US holds Grain Sales at the Headline Phase.
- **claim:** US should headline Grain Sales almost whenever it is available.
- **assertion:** rate above — headlined in >80% of eligible turns.
- **confidence:** high
- **source:** #67, "As US" ("the best US event in the game")
- **review:** include.

### c-09-abm-treaty-play-for-event
- **cards:** #57 ABM Treaty
- **side:** either
- **era:** mid
- **precondition:** player holds ABM Treaty and has any legal use for the resulting 4 Ops.
- **claim:** ABM Treaty should almost always be played for its event (DEFCON+1, then 4 Ops) rather than declined — the event strictly subsumes the plain-Ops use.
- **assertion:** rate above — played for-event in >90% of instances where it is played at all.
- **confidence:** high
- **source:** #57 ("one of the best events in the game")
- **review:** include.

### c-10-opec-us-space-priority
- **cards:** #61 OPEC
- **side:** US
- **era:** mid
- **precondition:** US holds OPEC; triggering it would net USSR ≥3 VP.
- **claim:** US should usually send OPEC to space rather than let its event trigger.
- **assertion:** rate above — spaced/mitigated in >70% of eligible turns.
- **confidence:** high
- **source:** #61, "As US"
- **review:** include.

### c-11-oas-founded-ussr-space-priority
- **cards:** #70 OAS Founded
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds OAS Founded; the Americas are not already thoroughly locked down.
- **claim:** USSR should treat OAS Founded as a top disposal priority (space or mitigate) rather than accept its unmitigated event.
- **assertion:** rate above — avoids unmitigated play in >70% of eligible turns.
- **confidence:** high
- **source:** #70, "As USSR"
- **review:** include.

### c-12-liberation-theology-ussr-trigger
- **cards:** #75 Liberation Theology
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds Liberation Theology; Central America Scoring is not about to trigger unusually early.
- **claim:** USSR should almost always trigger Liberation Theology's event.
- **assertion:** rate above — event-trigger rate should be high (>85%).
- **confidence:** high
- **source:** #75, "As USSR"
- **review:** include.

### c-13-ussuri-ussr-space
- **cards:** #76 Ussuri River Skirmish
- **side:** USSR
- **era:** mid
- **precondition:** USSR holds Ussuri River Skirmish.
- **claim:** USSR should almost always send Ussuri River Skirmish to space rather than play it.
- **assertion:** rate above — spaced in >80% of eligible turns.
- **confidence:** high
- **source:** #76, "As USSR" ("As USSR, I almost always space it")
- **review:** include.

### c-14-star-wars-us-hold-for-strong-discard
- **cards:** #85 Star Wars
- **side:** US
- **era:** late
- **precondition:** US holds Star Wars, US ahead on the Space Race, no standout event currently in the discard pile.
- **claim:** US should generally hold Star Wars rather than trigger it immediately, waiting for a strong event to appear in the discard pile.
- **assertion:** rate below — immediate-play rate should be low when the discard pile has no high-value event.
- **confidence:** low (the "no standout event" trigger condition is fairly vague/judgement-based)
- **source:** #85, "As US"
- **review:** include when discard pile includes Voice of America or Grain Sales.

### c-15-terrorism-ihc-combo-ussr
- **cards:** #92 Terrorism, #82 Iranian Hostage Crisis
- **side:** USSR
- **era:** late
- **precondition:** USSR holds Terrorism, and Iranian Hostage Crisis's event has already been triggered this game.
- **claim:** As USSR, near-always trigger Terrorism's event once Iranian Hostage Crisis has been played (forced double-discard).
- **assertion:** rate above — event-trigger rate should be high (>80%) under this precondition.
- **confidence:** high
- **source:** #92 ("an outstanding event if Iranian Hostage Crisis has been played, one I almost always trigger")
- **review:** include.

### c-16-aldrich-ames-remix-ussr-headline
- **cards:** #98 Aldrich Ames Remix
- **side:** USSR
- **era:** late
- **precondition:** USSR holds Aldrich Ames Remix at the Headline Phase.
- **claim:** Aldrich Ames Remix is one of the best USSR Late War headlines and should be headlined at a high rate when drawn.
- **assertion:** rate above — headline-play rate should be high.
- **confidence:** high
- **source:** #98, "As USSR" ("almost impossible to backfire")
- **review:** include. Also include strategic use of Aldrich Ames by US: if US has Aldrich Ames, scoring card which scores heavily against US, and bunch of neutral events, US should not play Aldrich Ames mid turn, and should play it as action round, when only 2 cards it has left are Aldrich Ames and scoring.

### c-17-ihc-us-space-when-controls-iran
- **cards:** #82 Iranian Hostage Crisis
- **side:** US
- **era:** late
- **precondition:** US holds Iranian Hostage Crisis, US currently controls Iran.
- **claim:** As US, prefer sending Iranian Hostage Crisis to space (or playing it for Ops if Iran is overcontrolled/uninteresting) over triggering the event, which only benefits USSR.
- **assertion:** rate above — combined space-or-Ops rate should exceed event-trigger rate under this precondition.
- **confidence:** medium
- **source:** #82, "As US"
- **review:** include.

### c-18-ladc-ussr-only-when-influence-stacked
- **cards:** #95 Latin American Debt Crisis
- **side:** USSR
- **era:** late
- **precondition:** USSR holds Latin American Debt Crisis; no South American country currently has heavily stacked bilateral influence.
- **claim:** As USSR, don't bother triggering LADC's event when South American influence stacks are low — it's ineffective in that case.
- **assertion:** rate below — event-trigger rate should be low when no South American country has large bilateral stacks.
- **confidence:** medium
- **source:** #95, "As USSR" ("If they aren't ... then don't bother with the event")
- **review:** include.

### c-19-chernobyl-ussr-overprotect-europe
- **cards:** #94 Chernobyl
- **side:** USSR
- **era:** late
- **precondition:** turn ≥ 7, USSR controls East Germany and Poland.
- **claim:** USSR should overprotect East Germany and Poland (stack influence above minimum control) as insurance against a Late War Chernobyl-driven Europe assault.
- **assertion:** rate above — rate of USSR maintaining EG/Poland influence above the minimal-control threshold should be higher in the Late War than in earlier eras.
- **confidence:** medium
- **source:** #94, "As USSR"
- **review:** include.


---

# MECHANICAL CONSTRAINTS THE TESTS MUST RESPECT

**An opponent-associated card can never be played *as* an event.** `action_mask.cpp`
offers EVENT play mode only when `!is_opponent_card`, matching the rules. Playing an
opponent's card for Operations is what fires their event, as a side effect the player
cannot decline. So a claim of the form "side X should/shouldn't play <card belonging to
Y> for the event" is not expressible: the testable form is always about playing it for
**Ops**, **spacing** it, or **holding** it.

A screen of every claim against `CardData` side ownership flagged the following as
involving a card the acting side does not own; each was re-read and phrased in terms of
Ops/space/hold rather than event choice:

`a-01` (Duck and Cover), `a-08` (VOA), `a-09` (Cultural Revolution), `a-11` (Star Wars),
`a-15` (Ortega), `b-12` (Alliance for Progress), `b-19` (KAL-007), `b-20` (Iron Lady),
`c-02` (Korean War), `c-07` (Quagmire), `c-19` (Chernobyl).

Claims where the opponent's card appears only as *context* rather than as the card being
played -- `a-14` and `b-21` (AWACS, US-owned, with Muslim Revolution as a precondition),
`c-10`, `c-15`, `c-17` -- are unaffected.

**NATO's event is inert until it is live.** `card_dispatcher.cpp:139` gates NATO on
`MARSHALL_PLAN_PLAYED || WARSAW_PACT_PLAYED`. Before either fires, the USSR can play NATO
for its 4 Ops with no event consequence at all, which is what makes `n-01` a free burn.

---

# TYPE N — REVIEWER-SUPPLIED CLAIMS

Claims contributed during review rather than extracted from the guide.

### n-01-ussr-burn-nato-while-inert
- **cards:** #21 NATO, plus any other 4-Ops card in hand
- **side:** USSR
- **era:** early
- **type:** B (comparative preference)
- **precondition:** USSR to act, NATO in hand alongside at least one other 4-Ops card, and
  NEITHER `WARSAW_PACT_PLAYED` nor `MARSHALL_PLAN_PLAYED` set, so NATO's event cannot fire.
- **claim:** USSR should prefer spending NATO over any other 4-Ops card while it is still
  inert, reducing the risk that the US later makes it live and the USSR is left holding it.
- **assertion:** P(USSR plays NATO for Ops) > P(USSR plays the other 4-Ops card).
- **confidence:** high (reviewer)
- **source:** reviewer

### n-02-ussr-nato-before-other-us-four-ops
- **cards:** #21 NATO, #23 Marshall Plan, #27 US/Japan Mutual Defense Pact
- **side:** USSR
- **era:** early/mid
- **type:** B (comparative preference)
- **precondition:** USSR to act, holding NATO together with Marshall Plan or US/Japan
  Mutual Defense Pact, with `WARSAW_PACT_PLAYED` or `MARSHALL_PLAN_PLAYED` already set.
- **claim:** Even once NATO is live, USSR should play NATO before Marshall Plan or
  US/Japan Mutual Defense Pact.
- **assertion:** P(USSR plays NATO for Ops) > P(USSR plays Marshall Plan or US/Japan for Ops).
- **confidence:** high (reviewer)
- **source:** reviewer

### n-03-wargames-trigger-when-winning-at-defcon2
- **cards:** #100 Wargames
- **side:** either
- **era:** late
- **type:** A (hard invariant)
- **precondition:** DEFCON 2, acting player holds Wargames, and the acting player leads by
  more than 6 VP.
- **claim:** The acting player should trigger Wargames; it ends the game immediately and,
  at that margin, wins it.
- **assertion:** never-argmax on anything else -- P(trigger Wargames) should dominate.
- **confidence:** high (reviewer)
- **source:** reviewer, resolving contradiction 3

### n-04-ask-not-discard-losing-scoring-card
- **cards:** #77 "Ask Not What Your Country Can Do For You...", scoring cards
- **side:** US
- **era:** mid/late
- **type:** A (hard invariant)
- **precondition:** US to act holding only "Ask Not" and a scoring card whose resolution
  would take the USSR to 20 VP and end the game.
- **claim:** US should play "Ask Not" and discard that scoring card rather than be forced
  to play it.
- **assertion:** never-argmax on playing the scoring card.
- **confidence:** high (reviewer)
- **source:** reviewer, promoted from Excluded

### n-05-red-scare-purge-not-event-on-last-ar
- **cards:** #31 Red Scare/Purge
- **side:** either
- **era:** any
- **type:** A (hard invariant)
- **precondition:** acting player holds Red Scare/Purge and it is the final Action Round of
  the turn, so the opponent has no remaining Action Round for the -1 Ops penalty to bite.
- **claim:** The event is worthless when the opponent will not act again; the card should
  not be played for the event on the last Action Round.
- **assertion:** never-argmax on playing Red Scare/Purge as the event.
- **confidence:** high (reviewer)
- **source:** reviewer, refining `c-04`

### n-06-us-delays-aldrich-ames-to-final-ar
- **cards:** #98 Aldrich Ames Remix, scoring cards
- **side:** US
- **era:** late
- **type:** B (comparative preference)
- **precondition:** US holds Aldrich Ames Remix plus a scoring card that scores heavily
  against the US, and other playable cards remain in hand.
- **claim:** US should hold Aldrich Ames rather than play it mid-turn, playing it on the
  Action Round where its only remaining cards are Aldrich Ames and the scoring card.
- **assertion:** P(US plays Aldrich Ames mid-turn with other cards available) < P(US plays
  another card).
- **confidence:** medium (reviewer)
- **source:** reviewer, extending `c-16`

---

# CONTRADICTIONS AND AMBIGUITIES

1. **Star Wars (#85) mechanical precondition.** The card text implies event-eligibility depends on being *ahead* on the Space Race track and is normally framed as a US-side action, yet the guide's "As USSR" section discusses the USSR holding and potentially triggering it. Claim `a-11` is kept (the "USSR should never trigger it" advice is clearly and strongly stated), but the exact mechanical precondition should be checked against the actual engine card implementation before being hard-coded into a test.

   **Resolved on review:** Star Wars is a US event, live only while the US leads the Space
   Race. "USSR triggering it" means the USSR plays it for Operations and thereby fires the
   US event -- an opponent's card can never be played *as* an event.

2. **Chernobyl (#94) headline timing is contested within the guide itself.** One passage calls it a strong headline; another explicitly criticizes players who "headline it immediately and go gung-ho for Europe" as being "in error," preferring to hold it for a scoring-card combo or until Europe is weakened. This is the author disagreeing with common practice rather than a strict rule, so no absolute claim was extracted for headline timing — only the softer, uncontested `c-19` overprotection claim survived into the final set.

   **Resolved on review:** too many defensible lines. No Chernobyl-specific check.

3. **Wargames (#100) is an explicitly branching decision tree** (depends on whether the opponent can contest AR1, exact VP margin, and DEFCON-suicide risk tolerance). Any flattened claim discards real nuance the guide itself insists on, and one variant additionally depends on believing the opponent holds Wargames — a hidden-hand condition. Both candidate Wargames claims were dropped from the final 56 for this reason.

   **Resolved on review:** one branch is a clean invariant and is added as `n-03`.

4. **Nixon Plays the China Card (#71), USSR side:** the guide's own advice on whether to give up 2 VP vs. the China Card is explicitly conditioned on "the state of the scoring track," which is more than a simple threshold — dropped from the final set as too holistic for a clean claim, though the underlying game-state check (has scoring track state) is in principle available to the agent.

   **Resolved on review:** dropped.

5. **Marshall Plan (#23) is a deliberate exception, not a contradiction.** Most 4-Ops starred US cards in the Early War (NATO, US/Japan Mutual Defense Pact, Nuclear Test Ban) are strongly preferred for Ops over event. Marshall Plan is explicitly called out by the guide as the one Early War starred US event that should be preferred for the *event* instead — this is intentional card-specific advice, not an inconsistency, and both patterns are captured as separate claims (`a-03`, `a-04`, `c-05` vs. `b-05`).

   **Confirmed on review.**

6. **Comecon/Warsaw Pact Formed vs. Decolonization/De-Stalinization (all USSR Early War starred events).** Comecon and Warsaw Pact Formed are held for Ops and their events avoided/delayed; Decolonization and De-Stalinization are the opposite — their whole value is in the event, and USSR actively seeks to trigger them (at the right DEFCON/timing). This is a genuine card-by-card split in the source material, not a contradiction — captured via separate claims per card.

   **Confirmed on review.**

7. **John Paul II Elected Pope (#68) is a Mid War card** (per the README's own card table) but its clearest strategic statement appears in the Late War guide's Solidarity (#101) article. Claim `a-16` uses the correct (Mid War) era tag despite being sourced from the Late War document; flagged here to avoid confusion if the source file is re-checked.

   **Resolved on review:** claim `a-16` removed.

---

# EXCLUDED / OUT OF SCOPE

Claims considered but deliberately excluded because they require information the agent cannot
observe (the opponent's exact hand contents) or reasoning the target competence level explicitly
excludes (deck-counting, reshuffle timing, multi-turn opponent-hand-reading setups):

- **Missile Envy (#49) headline timing** — partly depends on tracking which DEFCON-suicide cards remain in the *opponent's* hand.
- **Cuban Missile Crisis / Cuba-foothold (#40, #8 Fidel)** — a multi-turn positional heuristic rather than a single checkable decision; kept only as a note, not a formal claim.
- **Wargames (#100), opponent-holds-the-card variant** — requires believing the opponent holds a specific card.
- **The Cambridge Five (#104) headline timing** — the guide's own advice ("positively identify a particular scoring card... via reshuffle counting") is explicit deck-counting, out of scope for this competence level.
- **Muslim Revolution (#56)** — its effect hinges on combining three separate conditions (US battleground stability, US regional access, current Middle East score margin) that require holistic judgement rather than a single mechanical precondition.
- **"Ask Not What Your Country Can Do For You..." (#77)** — advice is a multi-turn hand-quality judgement ("hold and build the worst possible hand to discard"), not a single checkable decision.
  (One branch is mechanically checkable and is added as `n-04`.)
- **Truman Doctrine (#19) Turn-1 setup trick** — depends on reading the opponent's specific opening influence bid/setup; closer to an opening-book trick than a general behavioural claim.
