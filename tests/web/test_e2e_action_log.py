"""The page's action log says what each click did, the turn's last one included.

An action round that ends the turn stops on the turn's cleanup, a chance node the page resolves in
the same click: final VP, the military-ops reset, DEFCON, the next deal. Every engine step clears the
roll record, so read from the end state the turn's last realignment or coup lost its dice, and the
turn's end read as part of it. The page now describes the click from the engine's settled state (the
WebAssembly wrapper's `ts_settled_display_json`), logs the turn's end as its own entry, and prints
every die the click rolled (`ts_click_rolls_json`) -- a realignment with the opponent's war card rolls
twice. Games are seeded random play with realignments steered in late rounds, through the page.

Needs web/ui/dist (tools/scripts/build_web.sh) and a Playwright Chromium.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import ts_engine as ts

from tests.web.test_e2e_card_choices import _give_us, _token, _us_action_round
from tests.web.test_e2e_workbench import _open, browser, static_site  # noqa: F401 -- fixtures
from tools.lib.game_step import drain_chance

ALDRICH_AMES, FIVE_YEAR_PLAN, CHERNOBYL, TERRORISM, MISSILE_ENVY = 98, 5, 94, 92, 49
USSR_HAND = (ts.CardLocation.HAND_USSR_UNKNOWN, ts.CardLocation.HAND_USSR_KNOWN)
_CARDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rules", "cards.json")
SIDE = {int(c["id"]): c["side"] for c in json.load(open(_CARDS))}

#: Seeded random legal play (realignment, flat 114, whenever legal from round 6), until the first
#: realignment target choice for which `want(entries, before, after)` holds; returns that click's
#: log entries and the turns around it.
_FIND = """want => {
  const wb = window.__wb, ok = new Function('added', 'before', 'after', 'return ' + want);
  let x = 4242; const rnd = n => { x = (x * 1103515245 + 12345) % 2147483648; return x % n; };
  for (let g = 0; g < 40; g++) {
    wb.session.newGame(1000 + g); wb.refresh();
    for (let n = 0; n < 6000 && !wb.state.is_terminal; n++) {
      const before = wb.state, ctx = before.decision_context;
      const m = wb.engine.mask(false), legal = [];
      m.forEach((on, i) => { if (on) legal.push(i); });
      const a = (before.action_round >= 6 && legal.includes(114)) ? 114 : legal[rnd(legal.length)];
      const from = wb.session.logs.length;
      wb.sendFlatAction(a, 0, true);
      const after = wb.state;
      const realign = ctx.decision_type_name === 'POINT_NODE' && (ctx.resolving_card || 0) === 0 && ctx.op_mode === 2;
      const added = wb.englishLog().slice(from);
      if (realign && ok(added, before, after)) return { added, turn: before.turn, after_turn: after.turn };
    }
  }
  return null;
}"""


def _find(page: Any, want: str) -> Dict[str, Any]:
    found = page.evaluate(_FIND, want)
    assert found, f"no realignment matching {want!r} in 40 seeded games"
    return found


def test_the_turns_last_realignment_keeps_its_dice_and_the_turn_end_is_its_own_entry(browser: Any,
                                                                                     static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off")
    found = _find(page, "after.turn !== before.turn")
    added: List[Dict[str, Any]] = found["added"]
    action, rest = added[0], added[1:]
    assert "Realignment" in action["text"]
    assert any("🎲 Realignment" in d for d in action.get("details") or []), "the realignment's dice"
    assert not any("DRAW_DECK -> HAND" in d for d in action.get("details") or []), "the deal is not the realignment's"
    turn_end = [e for e in rest if e["phase"] == "TURN_END"]
    assert turn_end and turn_end[0]["text"] == f"End of turn {found['turn']}"
    assert any("DRAW_DECK -> HAND" in d for d in turn_end[0].get("details") or []), "the next deal is the turn's end"
    assert page.errors == []
    page.close()


def test_a_click_that_rolls_twice_logs_both_dice(browser: Any, static_site: str) -> None:
    """A realignment with the opponent's war card: the realignment rolls, then the war's event does."""
    page = _open(browser, f"{static_site}/?model=off")
    found = _find(page, "(added[0].details || []).some(d => d.includes('War Roll'))")
    details = found["added"][0].get("details") or []
    dice = [d for d in details if "🎲" in d]
    assert len(dice) == 2 and "Realignment" in dice[0] and "War Roll" in dice[1], dice
    assert page.errors == []
    page.close()


def _ussr_late_headline() -> ts.GameState:
    """A headline where the USSR is choosing its card, Aldrich Ames Remix swapped into its hand (any era:
    the card is put there, not drawn)."""
    for s0 in range(30, 80):
        s = ts.GameState()
        ts.Engine.init_game(s, s0)
        drain_chance(s)
        rng = random.Random(s0)
        for _ in range(4000):
            c = s.ctx()
            if (s.current_phase == ts.Phase.HEADLINE and s.turn >= 2 and c.decision_type == ts.DecisionType.SELECT_CARD
                    and c.decision_player == ts.Player.USSR and c.resolving_card == 0):
                held = [i for i in range(1, 111) if s.get_card_location(i) in USSR_HAND and i not in (6, ALDRICH_AMES)]
                old = s.get_card_location(ALDRICH_AMES)
                s.set_card_location(held[0], old if old not in USSR_HAND else ts.CardLocation.DISCARD_PILE)
                s.set_card_location(ALDRICH_AMES, ts.CardLocation.HAND_USSR_KNOWN)
                return s
            if ts.Engine.is_terminal(s):
                break
            legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(s, False))[0]]
            ts.Engine.step_flat(s, rng.choice(legal))
            drain_chance(s)
    raise AssertionError("no USSR headline choice found")


def test_a_card_aldrich_ames_discards_in_the_headline_is_a_discard_not_a_headline(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&pos={_token(_ussr_late_headline())}")
    page.evaluate(f"window.__wb.sendAction({{decision_type: 1, primary_id: {ALDRICH_AMES}, secondary_id: 0, flags: 0}})")
    # The US's headline, then whatever resolves before Aldrich Ames asks the USSR for a US card.
    for _ in range(40):
        ctx = page.evaluate("window.__wb.state.decision_context")
        if ctx.get("resolving_card") == ALDRICH_AMES and ctx.get("decision_type_name") == "SELECT_CARD":
            break
        page.evaluate("window.__wb.sendFlatAction(window.__wb.engine.mask(false).findIndex(x => x), 0, true)")
    else:
        raise AssertionError("Aldrich Ames never asked the USSR for a card")
    pick = page.evaluate("window.__wb.state.legal_actions.valid_ids.find(i => i >= 1 && i <= 110)")
    name = page.evaluate(f"window.__wb.engine.cardName({pick})")
    page.evaluate(f"window.__wb.sendAction({{decision_type: 1, primary_id: {pick}, secondary_id: 0, flags: 0}})")
    last = page.evaluate("window.__wb.englishLog().at(-1)")
    assert last["text"] == f"USSR discards {name} (#{pick}) from the US hand (Aldrich Ames Remix)", last["text"]
    log = page.inner_text("#action-log-stream")
    assert f"commits Headline Card: {name}" not in log, "the discarded card is not a headline"
    assert page.errors == []
    page.close()


def test_five_year_plans_random_discard_is_noted_with_whether_its_event_fires(browser: Any, static_site: str) -> None:
    s = _us_action_round(7)
    _give_us(s, FIVE_YEAR_PLAN)
    page = _open(browser, f"{static_site}/?model=off&pos={_token(s)}")
    page.evaluate(f"window.__wb.sendAction({{decision_type: 1, primary_id: {FIVE_YEAR_PLAN}, secondary_id: 0, flags: 0}})")
    page.evaluate("window.__wb.sendAction({decision_type: 2, primary_id: 0, secondary_id: 0, flags: 0})")   # the event
    lines = [d for e in page.evaluate("window.__wb.englishLog()") for d in (e.get("details") or [])]
    noted = [d for d in lines if d.startswith("  ✂ USSR discards") and "at random (Five Year Plan)" in d]
    assert len(noted) == 1, lines
    cid = int(noted[0].split("(#")[1].split(")")[0])
    assert ("its event fires" in noted[0]) == (SIDE[cid] == "us"), noted[0]
    page.close()


def test_terrorisms_random_discard_is_its_own_line(browser: Any, static_site: str) -> None:
    s = _us_action_round(7)
    _give_us(s, TERRORISM)
    page = _open(browser, f"{static_site}/?model=off&pos={_token(s)}")
    page.evaluate(f"window.__wb.sendAction({{decision_type: 1, primary_id: {TERRORISM}, secondary_id: 0, flags: 0}})")
    page.evaluate("window.__wb.sendAction({decision_type: 2, primary_id: 0, secondary_id: 0, flags: 0})")   # the event
    lines = [d for e in page.evaluate("window.__wb.englishLog()") for d in (e.get("details") or [])]
    noted = [d for d in lines if d.startswith("  ✂ USSR discards") and d.endswith("at random (Terrorism)")]
    assert len(noted) == 1, lines
    cid = int(noted[0].split("(#")[1].split(")")[0])
    assert page.evaluate(f"window.__wb.state.card_locations['{cid}']") == "DISCARD_PILE"
    page.close()


def test_chernobyls_options_are_regions_in_the_hud_and_the_log(browser: Any, static_site: str) -> None:
    """An event's branch is shown by what it means (web/ui/src/game/branches.ts), not its number:
    Chernobyl's six are the regions it can close, and the log names the one chosen."""
    s = _us_action_round(7)
    _give_us(s, CHERNOBYL)
    page = _open(browser, f"{static_site}/?model=off&pos={_token(s)}")
    page.evaluate(f"window.__wb.sendAction({{decision_type: 1, primary_id: {CHERNOBYL}, secondary_id: 0, flags: 0}})")
    page.evaluate("window.__wb.sendAction({decision_type: 2, primary_id: 0, secondary_id: 0, flags: 0})")   # the event
    page.wait_for_function(f"window.__wb.state.decision_context.resolving_card === {CHERNOBYL}")
    titles = page.eval_on_selector_all("#decision-body .btn-choice-card",
                                       "els => els.map(e => e.querySelector('div').innerText.trim())")
    assert titles == ["Europe", "Asia", "Middle East", "Africa", "Central America", "South America"], titles
    assert "Option" not in page.inner_text("#decision-body")
    page.click("#decision-body .btn-choice-card[data-primary='3']")   # Africa
    page.wait_for_function(f"window.__wb.state.decision_context.resolving_card !== {CHERNOBYL}")
    texts = [e["text"] for e in page.evaluate("window.__wb.englishLog()")]
    assert "US designates Africa (Chernobyl)" in texts, texts[-3:]
    assert page.errors == []
    page.close()


def test_missile_envys_exchange_says_what_is_taken_and_how_it_resolves(browser: Any, static_site: str) -> None:
    s = _us_action_round(7)
    _give_us(s, MISSILE_ENVY)
    page = _open(browser, f"{static_site}/?model=off&pos={_token(s)}")
    page.evaluate(f"window.__wb.sendAction({{decision_type: 1, primary_id: {MISSILE_ENVY}, secondary_id: 0, flags: 0}})")
    page.evaluate("window.__wb.sendAction({decision_type: 2, primary_id: 0, secondary_id: 0, flags: 0})")   # the event
    ctx = page.evaluate("window.__wb.state.decision_context")
    if ctx.get("resolving_card") == MISSILE_ENVY and ctx.get("decision_type_name") == "SELECT_CARD":
        pick = page.evaluate("window.__wb.state.legal_actions.valid_ids.find(i => i >= 1 && i <= 110)")
        page.evaluate(f"window.__wb.sendAction({{decision_type: 1, primary_id: {pick}, secondary_id: 0, flags: 0}})")
    lines = [d for e in page.evaluate("window.__wb.englishLog()") for d in (e.get("details") or [])]
    taken = [d for d in lines if d.startswith("  ⇄ The US takes")]
    assert len(taken) == 1, lines
    cid = int(taken[0].split("(#")[1].split(")")[0])
    fires = SIDE[cid] in ("us", "neutral")
    assert ("so its event fires for the US" in taken[0]) == fires and ("uses its" in taken[0]) == (not fires), taken[0]
    assert "  ⇄ Missile Envy goes to the USSR hand: the USSR must play it for Ops in its next action round" in lines
    assert page.errors == []
    page.close()
