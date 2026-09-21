# How completely an event must be executed — three categories, not a boolean

An event that opens a `POINT_NODE` needs two independent facts recorded about it, and the engine
currently conflates them into one flag plus a hardcoded list of four cards.

1. **May the player stop early?** — `ctx.allow_early_stop`.
2. **If the board cannot supply enough targets, is that news?** — `may_fizzle::allowed`
   (`engine/include/ts/constants.hpp:300`).

## The categories

| category | early stop | short/empty pool | card text |
|:---|:---|:---|:---|
| **PARTIAL_OK** | legal — declining is a move | legal, quiet | "may", "up to N" |
| **BEST_EFFORT** | **not** legal | legal, quiet | exact count against a pool that can run dry |
| **MUST_COMPLETE** | **not** legal | a defect — report it | exact count against a pool that cannot |

`may_fizzle::allowed` is exactly BEST_EFFORT, and it lists four cards. It should be ten.

## The two fixes are coupled, and landing one alone makes things worse

`engine/src/action_mask.cpp:596`:

```cpp
if (ctx.allow_early_stop) {
    mask_212[211] = 1;
} else if (!any_target) {
    if (!may_fizzle::allowed(ctx.resolving_card)) {
        report_anomaly("POINT_NODE with no legal target and no early stop", state);
    }
    mask_212[211] = 1;
}
```

The anomaly branch is reachable **only when `allow_early_stop == 0`**. All six BEST_EFFORT cards
missing from the whitelist currently set `allow_early_stop = 1`, so today they never reach it —
they are shielded by the very bug that needs fixing.

**Fixing `allow_early_stop` without extending the whitelist in the same change turns six silent
cards into six false-positive anomaly reports**, in the log whose entire value is that it is
clean. This is why the §6a batch has to move both together.

## The BEST_EFFORT set — ten cards, six of them missing

Each verified against both the trigger (which sets `remaining_steps` / `allow_early_stop`) and
the mask predicate in `CardHandlers::get_event_action_mask`. A card qualifies only if
`remaining_steps` is the card's literal count — *not* recomputed against the live pool — **and**
the mask predicate is board-state dependent, so the pool can fall short.

| id | name | clause | why it can run dry | in whitelist |
|---:|:---|:---|:---|:---:|
| 19 | Truman Doctrine | remove all USSR inf. from a single uncontrolled European country | none may be uncontrolled *and* hold USSR influence | yes |
| 28 | Suez Crisis | a total of 4 US inf. from France/UK/Israel | those three may hold < 4 | yes |
| 29 | East European Unrest | 1–2 USSR inf. from 3 Eastern European countries | fewer than 3 may hold USSR influence | yes |
| 56 | Muslim Revolution | all US inf. from 2 of 8 named countries | fewer than 2 may hold US influence | yes |
| 7 | Socialist Governments | a total of 3 US inf. from Western Europe | WE may hold < 3 | **no** |
| 14 | Comecon | 1 USSR inf. to each of 4 non-US-controlled Eastern European countries | fewer than 4 may be non-US-controlled | **no** |
| 23 | Marshall Plan | 1 US inf. to each of any 7 non-USSR-controlled Western European countries | fewer than 7 may qualify | **no** |
| 74 | The Voice of America | 4 USSR inf. from countries outside Europe | non-Europe may hold < 4 | **no** |
| 88 | Marine Barracks Bombing | a total of 2 US inf. from the Middle East | ME (ex-Lebanon) may hold < 2 | **no** |
| 99 | Pershing II Deployed | 1 US inf. from any 3 Western European countries | fewer than 3 may hold US influence | **no** |

## Warsaw Pact already shows the better shape

`card_dispatcher.cpp:497` counts the eligible pool first, returns immediately when it is empty,
and clamps `remaining_steps = std::min<uint8_t>(4, count)`. A card that clamps can never be short
mid-loop, so it needs no whitelist entry at all — the fizzle stops being a special case and
becomes arithmetic.

That is the structurally better fix, and it is worth asking whether the BEST_EFFORT category
should exist as data at all or be dissolved into clamping at every site. The cost of clamping is
that it must be written per card, against each card's own pool predicate; the cost of the
category is a list that goes stale — which is exactly how six cards came to be missing. (Warsaw
Pact still sets `allow_early_stop = 1`, which its text does not support, so it needs the early-stop
half of the fix regardless.)

## Wider than the three cards originally named

The early-stop misclassification is systemic, not confined to Marshall Plan / Comecon / Socialist
Governments. At least twelve cards set `allow_early_stop = 1` with no "may" or "up to" in their
text. Six are the BEST_EFFORT cards above. The other six —
Colonial Rear Guards (63), Decolonization (30), OAS Founded (70), Liberation Theology (75),
Ussuri River Skirmish (76), The Reformer (87) — have unrestricted region masks whose pools cannot
run short of the required count, so they need the early-stop flip but **not** a whitelist entry.

Ruled out after checking the engine, not just the text: Warsaw Pact (16, clamps),
Independent Reds (22) and The Cambridge Five (104) (both guard with an upfront emptiness check),
Special Relationship (105) and Junta (47) (masks not board-state restricted),
Puppet Governments (66) and Latin American Debt Crisis (95) (text says "may" — correctly
PARTIAL_OK, correctly flagged).

## Provenance

Classified by a subagent over all 110 cards in `rules/cards.json` plus the handlers in
`engine/src/events/{early,mid,late}_war.cpp` and `engine/src/card_dispatcher.cpp`. Spot-checked
independently: the `action_mask.cpp:596` branch ordering, `allow_early_stop = 1` on Socialist
Governments / Comecon / Marshall Plan, and the Warsaw Pact clamp. Feeds §6a of
[`../../plans/P17_action_representation.md`](../../archive/E3_ladder/plans/P17_action_representation.md).
