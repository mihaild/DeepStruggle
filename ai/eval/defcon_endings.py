"""Why a game ended at DEFCON 1, in four categories rather than two.

The engine's own split is `self` versus `provoked` -- did the losing player's action trigger it,
or the opponent's. That is too coarse in both directions. On the `self` side it puts a coup that
is an unforced loss with no upside in the same bucket as Summit, which is a deliberate gamble. On
the `provoked` side it does not say whether the loser had any choice.

    1 own goal      A single action that loses to any semi-competent opponent for no benefit
                    at all: couping a battleground at DEFCON 2, playing one's *own* mandatory
                    DEFCON degrader for the Event, or Olympic Games for the Event at DEFCON 2.
    2 bad bet       An action that need not have been taken and does not guarantee a loss, and
                    that sometimes pays: Summit, Missile Envy, Five Year Plan in US hands.
                    Usually wrong, but a strategic question rather than an error of sight.
    3 forced trap   An opponent-associated card was played whose event reaches DEFCON 1 -- and
                    nothing safer was available. The mistake, if any, was earlier hand
                    management, not this play.
    4 unforced trap The same play with a safe alternative in hand: another card, or spacing the
                    one that killed them.

**Headline endings are counted separately, before any of this.** The two headline cards are
chosen simultaneously and neither player sees the other's, so the same play can be sound or fatal
depending on a choice that was not visible: Olympic Games is a perfectly reasonable headline that
only backfires if the opponent headlined a degrader of their own. Reasoning about those needs the
opponent's card too, so they are reported as their own line rather than forced into a category
built for sequential decisions.

(4) is the category worth optimising: a short-horizon decision with an immediate, total
consequence and a visible alternative. (3) is where essentially all human DEFCON-1 endings sit.

A provoked ending is attributed to the *card play*, not to the opponent's coup that finished it:
the culprit decision is several micro-actions earlier and belongs to the loser, so the loser's
most recent card play is what gets classified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ts_engine as ts

from ai.eval.blunders import (DUCK_AND_COVER, FIVE_YEAR_PLAN, KAL_007, OLYMPIC_GAMES,
                              WE_WILL_BURY_YOU, defcon_suicide_cards, hand_of)
from ai.stats import wilson_interval

SUMMIT = 45
MISSILE_ENVY = 49

#: Degraders a player can fire *at themselves* by choosing the Event on their own card.
OWN_MANDATORY_DEGRADERS = {
    ts.Player.US: {DUCK_AND_COVER, KAL_007},
    ts.Player.USSR: {WE_WILL_BURY_YOU},
}

#: Plays that risk DEFCON without guaranteeing it, and that sometimes pay.
BETS = {SUMMIT, MISSILE_ENVY, FIVE_YEAR_PLAN}

#: Action-round categories. The headline phase is counted separately rather than as one of
#: these -- see the note in the module docstring.
CATEGORIES = ("own_goal", "bad_bet", "forced_trap", "unforced_trap", "unclassified")


@dataclass
class Play:
    """The last card a player committed, and what was available instead."""
    card_id: int
    mode: str
    had_safe_alternative: bool
    safe_count: int


@dataclass
class DefconEndings:
    counts: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in CATEGORIES})
    games: int = 0
    defcon1: int = 0
    in_headline: int = 0
    in_action_round: int = 0
    headline_cards: Dict[int, int] = field(default_factory=dict)
    headline_pairs: Dict[Any, int] = field(default_factory=dict)
    examples: Dict[str, List[str]] = field(default_factory=lambda: {k: [] for k in CATEGORIES})

    def note(self, category: str, detail: str) -> None:
        self.counts[category] = self.counts.get(category, 0) + 1
        if len(self.examples.setdefault(category, [])) < 6:
            self.examples[category].append(detail)

    def metrics(self, prefix: str = "defcon1") -> Dict[str, float]:
        """Shares of *all games*, with a Wilson band -- the denominator that can be compared."""
        out: Dict[str, float] = {}
        for c in CATEGORIES:
            k = self.counts.get(c, 0)
            lo, hi = wilson_interval(k, self.games)
            out[f"{prefix}_{c}_share"] = (k / self.games) if self.games else 0.0
            out[f"{prefix}_{c}_ci_low"] = lo
            out[f"{prefix}_{c}_ci_high"] = hi
            out[f"{prefix}_{c}_count"] = float(k)
        out[f"{prefix}_total_share"] = (self.defcon1 / self.games) if self.games else 0.0
        return out

    def note_headline(self, card_id: int, opponent_card: int = 0) -> None:
        """Both cards, because a headline death is a property of the pair.

        Headlines resolve by Ops, highest first, ties to the US. We Will Bury You is 4 Ops, so
        the only way its own player loses to it is an opponent headline that also degrades DEFCON
        *and* resolves first -- which needs 4+ Ops. Duck and Cover is 3 and resolves second, so
        it kills the US instead. That leaves KAL-007, at 4, as the one card that traps it.
        """
        self.in_headline += 1
        key = (card_id, opponent_card)
        self.headline_pairs[key] = self.headline_pairs.get(key, 0) + 1
        self.headline_cards[card_id] = self.headline_cards.get(card_id, 0) + 1

    def summary(self) -> str:
        g = max(self.games, 1)
        lines = [f"  {self.defcon1:,} of {self.games:,} games ended at DEFCON 1 "
                 f"({self.defcon1 / g * 100:.1f}%)",
                 f"    in the HEADLINE   {self.in_headline:5,}  "
                 f"{self.in_headline / g * 100:5.1f}% of games   "
                 f"{self.in_headline / max(self.defcon1, 1) * 100:5.1f}% of DEFCON-1"]
        if self.headline_pairs:
            top = sorted(self.headline_pairs.items(), key=lambda kv: -kv[1])[:5]
            for (mine, theirs), k in top:
                lines.append(f"        x{k:<3d} loser headlined {_name(mine):26s} "
                             f"against {_name(theirs) if theirs else 'unknown'}")
        lines.append(f"    in an ACTION ROUND {self.in_action_round:4,}  "
                     f"{self.in_action_round / g * 100:5.1f}% of games   "
                     f"{self.in_action_round / max(self.defcon1, 1) * 100:5.1f}% of DEFCON-1")
        for c in CATEGORIES:
            k = self.counts.get(c, 0)
            if not k and c == "unclassified":
                continue
            lo, hi = wilson_interval(k, self.games)
            lines.append(f"        {c:16s} {k:5,}  {k / g * 100:5.1f}% of games "
                         f"[{lo*100:.1f}, {hi*100:.1f}]   "
                         f"{k / max(self.in_action_round, 1) * 100:5.1f}% of action-round ones")
        return "\n".join(lines)


def _mode_name(primary_id: int) -> str:
    return {0: "EVENT", 1: "OPS", 2: "SPACE", 3: "PASS"}.get(int(primary_id), "?")


def classify_ending(loser: ts.Player, provoked: bool, last_play: Optional[Play],
                    coup_at_defcon2_on_battleground: bool,
                    out: DefconEndings) -> None:
    """Assign one DEFCON-1 ending to a category."""
    if not provoked:
        if coup_at_defcon2_on_battleground:
            out.note("own_goal", "couped a battleground at DEFCON 2")
            return
        if last_play is not None and last_play.mode == "EVENT":
            own = OWN_MANDATORY_DEGRADERS.get(loser, set())
            if last_play.card_id in own:
                out.note("own_goal", f"played its own {_name(last_play.card_id)} for the Event")
                return
            if last_play.card_id == OLYMPIC_GAMES:
                out.note("own_goal", "played Olympic Games for the Event at DEFCON 2")
                return
        if last_play is not None and last_play.card_id in BETS:
            out.note("bad_bet", f"{_name(last_play.card_id)} came in badly")
            return
        out.note("unclassified", f"self-inflicted, last play "
                                 f"{_name(last_play.card_id) if last_play else 'unknown'}")
        return

    if last_play is None:
        out.note("unclassified", "provoked, but no card play was recorded for the loser")
        return
    if last_play.had_safe_alternative:
        out.note("unforced_trap",
                 f"played {_name(last_play.card_id)} with {last_play.safe_count} safe card(s) left")
    else:
        out.note("forced_trap", f"played {_name(last_play.card_id)}; nothing safer in hand")


def _name(card_id: int) -> str:
    try:
        return ts.CardData.get_card_name(int(card_id))
    except Exception:
        return f"#{card_id}"


def measure_defcon_endings(model: Any, num_games: int = 500, base_seed: int = 611_000,
                           temperature: float = 0.1,
                           max_iters: int = 20_000) -> DefconEndings:
    """Drive self-play and classify every DEFCON-1 ending.

    Temperature 0.1, matching `measure_blunders_batched` and, more to the point, matching what
    training actually rolls out. At 1.0 the same checkpoint ends 60.3% of games at DEFCON 1
    against the 31.4% its training logged, and self-inflicted endings inflate fourfold -- a
    policy sampled far hotter than it plays coups battlegrounds it would never coup. Checked
    against the logs before trusting anything built on this: at 0.1 the probe reproduces them
    (33.9% total, 9.1% self) and at 1.0 it does not.
    """
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=base_seed)
    obs, masks, _ = env.reset_all()
    out = DefconEndings()

    # Per env: the last card each side committed, and whether the action just taken was a coup
    # on a battleground at DEFCON 2.
    last_play: List[Dict[Any, Play]] = [dict() for _ in range(num_games)]
    pending_card: List[Dict[Any, int]] = [dict() for _ in range(num_games)]
    coup_flag = [False] * num_games
    finished = [False] * num_games
    # The env auto-resets on termination, so the phasing player has to be read *before* the
    # step that ends the game -- afterwards it belongs to a fresh deal.
    phasing = [ts.Player.NONE] * num_games
    # Which phase the game was in when the drop happened. Headline endings are their own line:
    # the two cards are simultaneous and neither player saw the other's.
    phase_at: List[Any] = [None] * num_games

    try:
        for _ in range(max_iters):
            if all(finished):
                break
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
            mask_t = torch.from_numpy(np.asarray(masks)).to(device)
            with torch.no_grad():
                actions, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)
            acts = actions.cpu().numpy().astype(np.int64)

            for i in range(num_games):
                if finished[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                ctx = st.ctx()
                p = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                    else st.phasing_player
                phasing[i] = st.phasing_player
                phase_at[i] = st.current_phase
                ma = ts.ActionMask.decode_flat_action(st, int(acts[i]))
                dt = int(ma.decision_type)
                coup_flag[i] = False

                if dt == 1:                                   # SELECT_CARD
                    if st.current_phase == ts.Phase.HEADLINE:
                        # A headline resolves as an event with no play-mode decision, so it has
                        # to be recorded here or `last_play` keeps a stale card from an earlier
                        # action round -- which is what made these endings unclassifiable.
                        last_play[i][p] = Play(int(ma.primary_id), "HEADLINE", False, 0)
                    elif int(ctx.pending_op_card) == 0:
                        pending_card[i][p] = int(ma.primary_id)
                elif dt == 2:                                 # SELECT_PLAY_MODE
                    card = pending_card[i].get(p, 0)
                    if 1 <= card <= 110:
                        banned = defcon_suicide_cards(st, p)
                        safe = [c for c in hand_of(st, p) if c not in banned and c != card]
                        last_play[i][p] = Play(card, _mode_name(ma.primary_id),
                                               bool(safe), len(safe))
                elif dt == 5 and int(st.defcon) == 2:         # POINT_NODE -- an operation target
                    tgt = int(ma.primary_id)
                    # Only a *coup* in a battleground degrades DEFCON. Influence and
                    # realignment at the same node do not, and OpMode.COUP is 1.
                    if (0 <= tgt < 84 and int(ctx.op_mode) == int(ts.OpMode.COUP)
                            and bool(ts.MapData.get_country_info(tgt)["battleground"])):
                        coup_flag[i] = True

            obs, masks, _, dones, info = env.step(acts)
            for i, reason in enumerate(info["ending_reasons"]):
                if not reason or finished[i]:
                    continue
                out.games += 1
                finished[i] = True
                if not reason.startswith("defcon1"):
                    continue
                out.defcon1 += 1
                # Who lost, from the victory-point track rather than from phasing_player.
                # During the headline phase phasing_player does not identify the side whose
                # card caused the drop -- headlines resolve by Ops, not by turn order -- and
                # using it attributed We Will Bury You deaths to the USSR when Ops ordering
                # says the US is the one that dies to that pair.
                vp = int(np.asarray(info["victory_points"])[i])
                loser = ts.Player.USSR if vp > 0 else (
                    ts.Player.US if vp < 0 else phasing[i])
                lp = last_play[i].get(loser)
                if phase_at[i] == ts.Phase.HEADLINE:
                    other = ts.Player.US if loser == ts.Player.USSR else ts.Player.USSR
                    op = last_play[i].get(other)
                    out.note_headline(lp.card_id if lp is not None else 0,
                                      op.card_id if op is not None else 0)
                    continue
                out.in_action_round += 1
                classify_ending(loser, reason == "defcon1_provoked", lp, coup_flag[i], out)
    finally:
        if was_training:
            model.train()
    return out
