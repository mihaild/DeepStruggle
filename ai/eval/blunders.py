"""Named mistakes with a position-independent answer, counted as they happen.

`dominance.py` measures whether a policy *ranks* two options correctly, offline. This counts what
a policy actually *did*, cheaply enough to run inside a self-play game or a snapshot evaluation.
Every rule has the same shape: the play was available, a safe alternative was also available, and
the play is wrong in every position -- so a count is a count of errors, not of disagreements.

Scope, per the owner: **an action round at DEFCON 2**. Not the headline, where strategic reasons
to take the risk exist, and not DEFCON 3 or better, where the mechanism does not reach 1.

The rules:

* **Spacing your own or a neutral card** while holding an opponent card of the same Ops. The space
  track is the one outlet that spends a card without firing its event, so spending your own there
  and keeping theirs is backwards. Delegated to `dominance.space_dominance_outcome`, which already
  encodes this and is what sections 4.8 and 9.3 measured; there is no second implementation.

* **Olympic Games for the Event at DEFCON 2.** The opponent boycotts, DEFCON drops to 1, and the
  phasing player loses. Only a direct play counts -- a card forced through Missile Envy was not
  chosen by the player holding it.

* **A DEFCON-suicide card while a card that is not one was playable.** Either its own event drops
  DEFCON, or it hands the opponent Operations they can coup with. The conditional entries need a
  target: at DEFCON 2 only Africa and the Americas are still coupable, so with no influence there
  the gift is harmless. Nuclear Subs removes the danger from the Operations-granting ones outright,
  because US battleground coups then do not degrade DEFCON.

None of this is fed to the agent. It measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import ts_engine as ts
from bindings.settle import SettleMode, settle

from ai.eval import dominance
from ai.stats import wilson_interval

# --- card ids, from engine/include/ts/constants.hpp -------------------------------------------
DUCK_AND_COVER = 4
FIVE_YEAR_PLAN = 5
OLYMPIC_GAMES = 20
CIA_CREATED = 26
HOW_I_LEARNED = 46
JUNTA = 47
WE_WILL_BURY_YOU = 50
LONE_GUNMAN = 62
GRAIN_SALES = 67
STAR_WARS = 85
KAL_007 = 89
ORTEGA_ELECTED = 91
TEAR_DOWN_THIS_WALL = 96

#: Country ids, likewise. Ortega's free coup reaches Cuba, the one battleground adjacent to
#: Nicaragua, so US influence there is what makes the card dangerous to hold.
CUBA = 71

#: effect_bits::NUCLEAR_SUBS_ACTIVE. The bit values are not bound to Python, so this mirrors
#: engine/include/ts/constants.hpp:304 and is asserted against the engine in the tests.
NUCLEAR_SUBS_ACTIVE = 1 << 13

EUROPE, ASIA, MIDDLE_EAST, AFRICA, CENTRAL_AMERICA, SOUTH_AMERICA = 0, 1, 2, 3, 4, 5

#: At DEFCON 2 these are the only regions still open to a coup, so they are the only places an
#: opponent handed Operations can take DEFCON to 1.
COUPABLE_AT_DEFCON_2 = (AFRICA, CENTRAL_AMERICA, SOUTH_AMERICA)
AMERICAS = (CENTRAL_AMERICA, SOUTH_AMERICA)

#: What Star Wars can pull out of the discard pile and play immediately. The pick is mandatory
#: and fizzles only on an empty pile (`late_war.cpp`), so a discard holding one of these is a
#: risk the US takes by playing the card at all.
#:
#: Not every DEFCON-degrading card belongs here. Lone Gunman and Ortega hand the *USSR* the
#: Operations, and Five Year Plan makes the USSR discard, so the action that would reach DEFCON 1
#: is the opponent's, not the player's. How I Learned is here and nowhere else: it sets DEFCON
#: outright rather than by way of a coup.
#:
#: Junta is deliberately absent. Taken by the *US*, its free coup is the US's to aim, so the US
#: can steer it off a battleground -- avoidable, and not a suicide. Taken by the US after the
#: *USSR* played Star Wars for Operations, the USSR has no such say, which is why Junta appears
#: in the USSR's list below instead.
STAR_WARS_DISCARD_DANGERS = (OLYMPIC_GAMES, DUCK_AND_COVER, KAL_007, CIA_CREATED, GRAIN_SALES,
                             TEAR_DOWN_THIS_WALL, WE_WILL_BURY_YOU, HOW_I_LEARNED)

#: Star Wars against the *USSR* is position-dependent -- each degrader carries its own condition
#: -- so it is computed by `us_played_degraders` rather than listed as a constant.

RULES = ("spaced_own_or_neutral", "olympic_games_at_defcon2", "defcon_suicide_with_alternative")


@dataclass
class Blunder:
    rule: str
    player: str
    card_id: int
    card_name: str
    turn: int
    action_round: int
    detail: str

    def __str__(self) -> str:
        return (f"T{self.turn} AR{self.action_round} {self.player}: {self.rule} -- "
                f"{self.card_name} (#{self.card_id}); {self.detail}")


@dataclass
class BlunderCounts:
    """Committed and opportunities, per rule.

    Both halves matter. A policy that never held Olympic Games at DEFCON 2 has not demonstrated
    anything by not misplaying it, and a rate with no denominator cannot be compared across runs.
    """

    committed: Dict[str, int] = field(default_factory=dict)
    opportunities: Dict[str, int] = field(default_factory=dict)
    examples: List[Blunder] = field(default_factory=list)
    max_examples: int = 50
    # Which rules `metrics` and `summary` enumerate. It is the module's own RULES here, and the
    # sequencing probe passes its own -- the counting, the Wilson band and the examples are the
    # same machinery, only the rule names differ. Enumerating a fixed list rather than whatever
    # keys happen to be present is deliberate: a rule with no chances this run must still emit
    # its metric keys, or the TensorBoard tags come and go between runs.
    rules: Sequence[str] = RULES

    def note_opportunity(self, rule: str) -> None:
        self.opportunities[rule] = self.opportunities.get(rule, 0) + 1

    def note_blunder(self, b: Blunder) -> None:
        self.committed[b.rule] = self.committed.get(b.rule, 0) + 1
        if len(self.examples) < self.max_examples:
            self.examples.append(b)

    def rate(self, rule: str) -> float:
        n = self.opportunities.get(rule, 0)
        return (self.committed.get(rule, 0) / n) if n else 0.0

    def metrics(self, prefix: str = "blunder") -> Dict[str, float]:
        """Rate with a 95% Wilson interval, plus the raw counts.

        The interval is what makes the rate readable on its own: it carries the denominator,
        so a rule with two chances shows as a band across most of [0, 1] and a rule with four
        hundred shows as a tight one. `_count` and `_chances` stay in the record for anything
        that wants the raw numbers, but they no longer need a chart each.
        """
        out: Dict[str, float] = {}
        for rule in self.rules:
            committed = int(self.committed.get(rule, 0))
            chances = int(self.opportunities.get(rule, 0))
            lo, hi = wilson_interval(committed, chances)
            out[f"{prefix}_{rule}_rate"] = self.rate(rule)
            out[f"{prefix}_{rule}_ci_low"] = lo
            out[f"{prefix}_{rule}_ci_high"] = hi
            out[f"{prefix}_{rule}_count"] = float(committed)
            out[f"{prefix}_{rule}_chances"] = float(chances)
        return out

    def merge(self, other: "BlunderCounts") -> None:
        for k, v in other.committed.items():
            self.committed[k] = self.committed.get(k, 0) + v
        for k, v in other.opportunities.items():
            self.opportunities[k] = self.opportunities.get(k, 0) + v
        room = max(0, self.max_examples - len(self.examples))
        self.examples.extend(other.examples[:room])

    def summary(self) -> str:
        lines = []
        for rule in self.rules:
            n = self.opportunities.get(rule, 0)
            c = self.committed.get(rule, 0)
            if n:
                lines.append(f"  {rule:34} {c:5}/{n:<6} ({100.0 * c / n:5.1f}%)")
            else:
                lines.append(f"  {rule:34} {'no chances':>12}")
        return "\n".join(lines)


# --- helpers ----------------------------------------------------------------------------------

def _info(card_id: int) -> Dict[str, object]:
    return dict(ts.CardData.get_card_info(int(card_id)))


def _name(card_id: int) -> str:
    try:
        return str(_info(card_id)["name"])
    except Exception:
        return f"card {card_id}"


def in_action_round_at_defcon_2(state: ts.GameState) -> bool:
    """The only window these rules apply to.

    The headline is excluded because a player may have a reason to accept the risk there, and
    DEFCON 3 or better is excluded because one degradation cannot reach 1.
    """
    return int(state.defcon) == 2 and state.current_phase == ts.Phase.ACTION_ROUND


def has_influence_in(state: ts.GameState, player: ts.Player, regions: Sequence[int],
                    battleground_only: bool = False) -> bool:
    """Does `player` hold influence anywhere in these regions?

    This is what makes a gift of Operations dangerous: a coup needs a country the opponent has
    influence in, so with none in the regions still open at DEFCON 2 there is nothing to coup.

    `battleground_only` is what the DEFCON rules actually need. Only a coup in a *battleground*
    degrades DEFCON, so influence in a non-battleground is a country the opponent can take
    without the game ending. Counting those made every one of these cards look dangerous in
    positions where it was not.
    """
    want = set(int(r) for r in regions)
    for cid in range(84):
        info = ts.MapData.get_country_info(cid)
        if int(info["region"]) not in want:
            continue
        if battleground_only and not bool(info["battleground"]):
            continue
        c = state.get_country(cid)
        inf = c.ussr_influence if player == ts.Player.USSR else c.us_influence
        if int(inf) > 0:
            return True
    return False


def hand_of(state: ts.GameState, player: ts.Player) -> List[int]:
    return [c for c in range(1, 111) if ts.in_hand_of(state.get_card_location(c), player)]


def discard_pile(state: ts.GameState) -> List[int]:
    return [c for c in range(1, 111)
            if state.get_card_location(c) == ts.CardLocation.DISCARD_PILE]


def is_us_event(card_id: int) -> bool:
    """Is this card's event the US's? Read from the engine, not from a list kept here."""
    return str(ts.CardData.get_card_info(int(card_id)).get("side")) == "US"


def us_played_degraders(state: ts.GameState, subs: bool) -> Set[int]:
    """Cards that take DEFCON to 1 if the *US* plays them now, with no USSR decision in between.

    This is what makes Star Wars fatal for the USSR: the USSR plays it for Operations, the US
    event fires, and the US takes a card out of the discard and plays it at once. The USSR gets
    no say in the pick, so the question is only whether the pile holds something that degrades
    without asking it anything.

    Which is not the same as "degrades DEFCON". Three shapes:

    * **Unconditional** -- Duck and Cover, KAL-007 degrade from their own text, and How I
      Learned sets the level outright with the US choosing it. These reach the USSR wherever it
      stands, which is why there is no blanket influence condition on this set.
    * **Conditional on a coup having somewhere to land** -- CIA Created and Grain Sales hand the
      US Operations (Africa or the Americas); Junta grants a free coup in the Americas only;
      Tear Down This Wall grants one in Europe, overriding the DEFCON 2 closure. Each needs USSR
      influence in a battleground of *its own* region, and each is a US coup, so Nuclear Subs
      exempts them.
    * **Conditional on the hand** -- Five Year Plan fires a random USSR card only if that card is
      a *US event*, so it is dangerous exactly when one of the above is in hand **and US-side**.
      Junta and How I Learned are neutral, so Five Year Plan cannot fire them and a hand holding
      only those is safe from it.

    Cards that hand the *USSR* the Operations or the choice are absent: it simply declines. That
    is Lone Gunman, Ortega and Olympic Games. We Will Bury You is absent for a different reason --
    the US playing it degrades DEFCON by the US's own action, so the US would be picking its own
    loss.
    """
    out: Set[int] = {DUCK_AND_COVER, KAL_007, HOW_I_LEARNED}
    if not subs:
        if has_influence_in(state, ts.Player.USSR, COUPABLE_AT_DEFCON_2,
                            battleground_only=True):
            out.update({CIA_CREATED, GRAIN_SALES})
        if has_influence_in(state, ts.Player.USSR, AMERICAS, battleground_only=True):
            out.add(JUNTA)
        if has_influence_in(state, ts.Player.USSR, (EUROPE,), battleground_only=True):
            out.add(TEAR_DOWN_THIS_WALL)
    if any(c in out and is_us_event(c) for c in hand_of(state, ts.Player.USSR)):
        out.add(FIVE_YEAR_PLAN)
    return out


def defcon_suicide_cards(state: ts.GameState, player: ts.Player) -> Set[int]:
    """Which cards this player must not play here, in this position.

    "Must not" means the *opponent's* event fires whether the player likes it or not, and takes
    DEFCON to 1 by the player's own action. A card whose own event the player chooses -- Olympic
    Games, or We Will Bury You in the USSR's own hand -- is not here: declining the event is
    free, so playing it for Operations is safe and only the Event is the mistake. Olympic Games
    has its own rule for exactly that reason.

    Only a coup in a **battleground** degrades DEFCON, so every "handed the opponent Operations"
    card below is conditioned on battleground influence, not on influence anywhere.

    Nuclear Subs exempts *US* battleground coups from degrading DEFCON, so it disarms the cards
    that hand the US Operations -- and only those. The USSR's coups still degrade DEFCON with it
    in play, so the cards in the US's own hand stay dangerous.
    """
    out: Set[int] = set()
    subs = bool(state.has_flag(NUCLEAR_SUBS_ACTIVE))

    if player == ts.Player.USSR:
        # Both degrade DEFCON from their own text, with nothing for the USSR to avoid.
        out.update({DUCK_AND_COVER, KAL_007})
        if not subs:
            # Each hands the US the Operations to coup with.
            if has_influence_in(state, ts.Player.USSR, COUPABLE_AT_DEFCON_2,
                                battleground_only=True):
                out.update({CIA_CREATED, GRAIN_SALES})
            # Tear Down This Wall grants its coup *in Europe*, overriding the DEFCON 2 rule that
            # closes the region -- so European battlegrounds are exposed by this card alone.
            if has_influence_in(state, ts.Player.USSR, (EUROPE,), battleground_only=True):
                out.add(TEAR_DOWN_THIS_WALL)
        # Star Wars is a US event, so the USSR playing it for Operations fires it: with the US
        # ahead on the track the US takes a card out of the discard and plays it at once. Outside
        # the Nuclear Subs guard, because the unconditional degraders in that set are not coups
        # and Nuclear Subs does not touch them.
        if (int(state.us_space_track) > int(state.ussr_space_track)
                and any(c in us_played_degraders(state, subs) for c in discard_pile(state))):
            out.add(STAR_WARS)
        # Five Year Plan discards a random USSR card and fires it only if it is a *US* event,
        # so it is dangerous exactly when one of the above is in hand and US-side. Everything in
        # `out` here is US-side already; the check is explicit so a neutral card added to this
        # set later cannot arm it by accident.
        if any(c in out and is_us_event(c) for c in hand_of(state, ts.Player.USSR)):
            out.add(FIVE_YEAR_PLAN)
    else:
        # A USSR event that degrades DEFCON outright; the US cannot decline it.
        out.add(WE_WILL_BURY_YOU)
        # Hands the USSR the Operations to coup with.
        if has_influence_in(state, ts.Player.US, COUPABLE_AT_DEFCON_2, battleground_only=True):
            out.add(LONE_GUNMAN)
        # Ortega's free coup reaches a country adjacent to Nicaragua; Cuba is the battleground
        # among them, so US influence there is what makes DEFCON reachable.
        if int(state.get_country(CUBA).us_influence) > 0:
            out.add(ORTEGA_ELECTED)
        # Star Wars' pick is mandatory and fizzles only on an empty pile, so a discard holding
        # anything that degrades DEFCON by the US's own hand is a risk taken by playing it.
        if int(state.us_space_track) > int(state.ussr_space_track) and \
                any(c in STAR_WARS_DISCARD_DANGERS for c in discard_pile(state)):
            out.add(STAR_WARS)
    return out


# --- the check ---------------------------------------------------------------------------------

def check_play(state: ts.GameState, player: ts.Player, card_id: int, mode: str,
               forced: bool = False, counts: Optional[BlunderCounts] = None) -> List[Blunder]:
    """Blunders committed by playing `card_id` as `mode` from this state.

    `mode` is "EVENT", "OPS" or "SPACE". `forced` marks a play the player did not choose --
    Missile Envy hands a card over and the recipient must play it, which is not their error.

    Pass `counts` to record the opportunity as well as the blunder; a rate needs both.
    """
    found: List[Blunder] = []
    if forced or not (1 <= int(card_id) <= 110):
        return found

    side = "USSR" if player == ts.Player.USSR else "US"
    turn, ar = int(state.turn), int(state.action_round)

    # The space rule applies whenever a card is spaced, at any DEFCON.
    if mode == "SPACE":
        outcome = dominance.space_dominance_outcome(state, player, int(card_id))
        if outcome is not None:
            if counts is not None:
                counts.note_opportunity("spaced_own_or_neutral")
            if outcome is False:
                found.append(Blunder(
                    "spaced_own_or_neutral", side, card_id, _name(card_id), turn, ar,
                    "spent an own or neutral card on the track while holding an "
                    "equal-Ops opponent card"))

    # The DEFCON rules apply only in an action round at DEFCON 2.
    if in_action_round_at_defcon_2(state):
        hand = hand_of(state, player)

        if OLYMPIC_GAMES in hand or int(card_id) == OLYMPIC_GAMES:
            if counts is not None:
                counts.note_opportunity("olympic_games_at_defcon2")
            if int(card_id) == OLYMPIC_GAMES and mode == "EVENT":
                found.append(Blunder(
                    "olympic_games_at_defcon2", side, card_id, _name(card_id), turn, ar,
                    "played for the Event at DEFCON 2; a boycott ends the game"))

        banned = defcon_suicide_cards(state, player)
        safe = [c for c in hand if c not in banned and c != int(card_id)]
        if safe and (int(card_id) in banned or any(c in banned for c in hand)):
            if counts is not None:
                counts.note_opportunity("defcon_suicide_with_alternative")
            if int(card_id) in banned and mode != "SPACE":
                found.append(Blunder(
                    "defcon_suicide_with_alternative", side, card_id, _name(card_id), turn, ar,
                    f"at DEFCON 2 with {len(safe)} non-suicide card(s) available"))

    if counts is not None:
        for b in found:
            counts.note_blunder(b)
    return found


# --- measuring a policy's blunder rate ----------------------------------------------------

def _drain(state: ts.GameState) -> None:
    """Resolve pending die rolls, so the policy is only ever asked for real decisions.

    CHANCE, not FORCED, and NOT an oversight: `measure_blunders_batched` walks the same games
    through TsVectorizedEnv, whose settling lives in the C++ runner, and the two probes must
    count the same opportunities -- that is what test_the_two_probes_agree_on_the_same_games
    pins. Moving this one to FORCED alone made them disagree (0 batched vs 1 sequential on
    spaced_own_or_neutral). The pair moves together, through the env, or not at all.
    """
    settle(state, SettleMode.CHANCE)


def measure_blunders(
    select_action: Callable[[ts.GameState, Any], Optional[int]],
    num_games: int = 32,
    base_seed: int = 830_000,
    max_steps: int = 3000,
) -> BlunderCounts:
    """Play games and count the named mistakes, with their denominators.

    `select_action(state, player) -> flat action index`, matching PlayerAgent.select_action, so
    the caller owns the observation layout rather than this module guessing at it -- a probe
    that guessed is exactly how position_diagnostics came to report noise for a whole run.

    Single-state rather than batched on purpose: a blunder is defined on the decision, and the
    rule needs the card, the mode and the hand at the moment of play. That costs seconds per
    snapshot, and only runs at snapshots.

    The detection is the same path as `tools/lib/self_play.py`: SELECT_PLAY_MODE is where both
    halves are known -- which card, and what it is being spent on -- and the card is still in
    its owner's hand there, which the rules read.
    """
    counts = BlunderCounts()
    for i in range(num_games):
        state = ts.GameState()
        # The same games VectorizedBatchRunner deals for this base_seed, so this probe and
        # measure_blunders_batched sample one population and can be checked against each other.
        ts.Engine.init_game(state, base_seed + i * 10007 + 1)
        last_card: Dict[str, int] = {}
        for _ in range(max_steps):
            _drain(state)
            if ts.Engine.is_terminal(state):
                break
            player = state.ctx().decision_player
            if player == ts.Player.NONE:
                player = state.phasing_player
            action = select_action(state, player)
            if action is None:
                break
            ma = ts.ActionMask.decode_flat_action(state, int(action))
            side = "US" if player == ts.Player.US else "USSR"
            dt = int(ma.decision_type)
            if dt == 1:
                last_card[side] = int(ma.primary_id)
            elif dt == 2:
                mode = {0: "EVENT", 1: "OPS", 2: "SPACE"}.get(int(ma.primary_id))
                card = last_card.get(side, 0)
                if mode and 1 <= card <= 110:
                    # Missile Envy forces a card on its recipient, so that play is not theirs.
                    forced = (int(getattr(state, "forced_card_id", 0)) == card
                              and getattr(state, "forced_card_player", None) == player)
                    check_play(state, player, card, mode, forced=forced, counts=counts)
            ts.Engine.step_flat(state, int(action))
    return counts


def measure_blunders_batched(
    model: Any,
    num_games: int = 32,
    base_seed: int = 830_000,
    temperature: float = 0.1,
    max_iters: int = 20_000,
) -> BlunderCounts:
    """`measure_blunders` over parallel environments, with one forward pass per batch.

    Same counting rule, same detection point. The difference is only where the policy is asked:
    the single-state version hands the network one state at a time, which measured **90 of the
    122 seconds** of a snapshot evaluation -- 74% of it, for a probe that is a rounding error in
    the batched version. Everything else in a snapshot was already batched (the decisive and
    position probes run 128 environments at ~106,000 decisions/sec against ~890 one at a time).

    The rule still needs per-decision state -- which card, what it is being spent on, and the
    hand at that moment -- so the loop still walks the environments that are at a SELECT_PLAY_MODE
    node. That inspection is cheap; it was the forward pass that was not.
    """
    import numpy as np
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=base_seed)
    obs, masks, _ = env.reset_all()

    counts = BlunderCounts()
    last_card: List[Dict[str, int]] = [{} for _ in range(num_games)]
    done = [False] * num_games

    try:
        for _ in range(max_iters):
            if all(done):
                break
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
            mask_t = torch.from_numpy(np.asarray(masks)).to(device)
            with torch.no_grad():
                actions, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)
            acts = actions.cpu().numpy().astype(np.int64)

            for i in range(num_games):
                if done[i]:
                    continue
                state = env.runner.get_state(i)
                if ts.Engine.is_terminal(state):
                    done[i] = True
                    continue
                ma = ts.ActionMask.decode_flat_action(state, int(acts[i]))
                player = state.ctx().decision_player
                if player == ts.Player.NONE:
                    player = state.phasing_player
                side = "US" if player == ts.Player.US else "USSR"
                dt = int(ma.decision_type)
                if dt == 1:
                    last_card[i][side] = int(ma.primary_id)
                elif dt == 2:
                    mode = {0: "EVENT", 1: "OPS", 2: "SPACE"}.get(int(ma.primary_id))
                    card = last_card[i].get(side, 0)
                    if mode and 1 <= card <= 110:
                        forced = (int(getattr(state, "forced_card_id", 0)) == card
                                  and getattr(state, "forced_card_player", None) == player)
                        check_play(state, player, card, mode, forced=forced, counts=counts)

            obs, masks, _, dones, _ = env.step(acts)
            for i, d in enumerate(dones):
                if d:
                    done[i] = True
    finally:
        if was_training:
            model.train()
    return counts
