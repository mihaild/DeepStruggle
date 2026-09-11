"""What the policy does with its opening placement, against what humans do with theirs.

The opening is the cheapest thing in the game to measure and one of the most legible. It is a
fixed block of fifteen decisions -- USSR places 6 in Eastern Europe, then the US places 7 in
Western Europe and 2 more wherever it already sits -- taken before a single card is played, with
no rollout and no chance. Fifteen batched forward passes describe a checkpoint's entire opening
repertoire.

It is also where the anecdotes are. A USSR that does not hold Poland, or a US that does not hold
West Germany, has given up the two most contested countries on the board before the game starts,
and `experiments.md` §12 records that the agent "fights where it was placed and never opens a new
front" -- which makes the placement the whole front.

**Control is not the bar; control with a point to spare is.** A country held at exactly the
control threshold is taken back by a single influence, and in Europe one card routinely supplies
it. The standard USSR opening is Poland 4 and East Germany 4 -- both overcontrolled by one -- and
the US buys the same buffer in Italy at 3. The human corpus agrees: 97.3% of games finish setup
with Poland at 4 or more and East Germany at 4 or more, and 1.95 European countries held with a
buffer against 0.08 held at exactly control. So every threshold here is reported twice, at
control and at control-plus-one, and the finished board is what they are read from -- not the
placement, because East Germany starts at 3 and a single point there buys the buffer.

Two things this deliberately does *not* do:

* It does not score placements. There is no single correct opening, and a probe that graded
  against one would measure agreement with its author. It reports the distribution and the two
  named statistics, next to the same numbers from 266 human games.
* It does not read the observation. The policy is driven through the engine's own action mask,
  so what is measured is what the network would actually play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.stats import wilson_interval

#: Flat action index of country `cid` at a POINT_NODE (`bindings/action_encoder.py`).
NODE_OFFSET = 119

#: The setup block is a fixed length: 6 USSR, then 7 US, then 2 US bonus. Every env is in
#: lockstep through it, so after this many batched steps every game has just left SETUP.
SETUP_DECISIONS = 15

POLAND = 15
WEST_GERMANY = 7
EUROPE_SCORING = 2

#: Control thresholds, which are what the queue's goal statement names. Control is
#: `mine >= theirs + stability`, and with the opponent at zero that is just the stability.
POLAND_TARGET = 3
WEST_GERMANY_TARGET = 4

#: The overcontrol bar: one point past control. A country held at exactly control is taken back
#: by a single influence, which in Europe one card routinely supplies -- so the standard opening
#: buys the buffer. Humans put a mean 4.21 into Poland and bring East Germany from 3 to 4.
POLAND_BUFFERED = 4
WEST_GERMANY_BUFFERED = 5

_STABILITY: np.ndarray = np.array(
    [int(ts.MapData.get_country_info(c)["stability"]) for c in range(84)], dtype=np.int16)
_REGION: np.ndarray = np.array(
    [int(ts.MapData.get_country_info(c)["region"]) for c in range(84)], dtype=np.int16)
EUROPE = int(ts.Region.EUROPE)


@dataclass
class SideSetup:
    """Every placement one side made, one row per game."""

    influence: np.ndarray          # (games, 84) int8 -- influence placed during setup
    hand_has_europe_scoring: np.ndarray   # (games,) bool
    #: (games, 84) -- this side's influence on the finished board, and the opponent's. The
    #: placements alone cannot answer the question that matters, because the board is not empty
    #: when the block starts: East Germany already holds 3 USSR.
    final: np.ndarray = field(default_factory=lambda: np.zeros((0, 84), np.int16))
    opponent_final: np.ndarray = field(default_factory=lambda: np.zeros((0, 84), np.int16))

    def rate_at_least(self, cid: int, threshold: int) -> Tuple[float, float, float]:
        """P(influence *placed* in `cid` >= threshold), with a Wilson 95% band."""
        return _rate(self.influence[:, cid] >= threshold)

    def final_at_least(self, cid: int, threshold: int) -> Tuple[float, float, float]:
        """P(influence in `cid` *on the finished board* >= threshold).

        This, not the placement, is what the thresholds mean. East Germany starts at 3, so a
        USSR that places one there holds it at 4 -- and a probe reading the placement alone
        would call that "1" and miss that the country is now buffered.
        """
        return _rate(self.final[:, cid] >= threshold)

    def mean_per_country(self) -> np.ndarray:
        return self.influence.mean(axis=0)

    def entropy(self) -> float:
        """Entropy of the placement distribution over countries, in bits.

        Zero means the same countries every game. It is reported because a policy that always
        opens identically is a different failure from one that opens badly, and the two are not
        distinguishable from the mean alone.
        """
        total = self.influence.sum()
        if total <= 0:
            return float("nan")
        p = self.influence.sum(axis=0).astype(np.float64) / float(total)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def openings(self, top: int = 10) -> List[Tuple[str, int]]:
        """The most common distinct placements, as `"country:n, country:n"` strings."""
        counts: Dict[str, int] = {}
        for row in self.influence:
            nz = np.flatnonzero(row)
            key = ", ".join(f"{_country_name(int(c))}:{int(row[c])}" for c in nz)
            counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return ranked[:top]

    def distinct_openings(self) -> int:
        return len({tuple(row.tolist()) for row in self.influence})

    def margin(self) -> np.ndarray:
        """(games, 84): influence above the amount control requires.

        `mine - (theirs + stability)`. Zero is control with nothing to spare -- a single
        opponent influence, from one card, takes the country back. One is the buffer a standard
        opening buys, and it is why placing into a country you already control is not waste:
        East Germany starts at exactly 3 against stability 3, so it is held at margin zero until
        a fourth goes in.

        Negative means not controlled, and is reported only through `controls()`.
        """
        return (self.final.astype(np.int16)
                - self.opponent_final.astype(np.int16)
                - _STABILITY[None, :])

    def controls(self) -> np.ndarray:
        return self.margin() >= 0

    def fragile(self, region: Optional[int] = None) -> np.ndarray:
        """Per game: controlled countries held at margin exactly zero, i.e. one draw from losing."""
        sel = self.controls() & (self.margin() == 0)
        if region is not None:
            sel = sel & (_REGION[None, :] == region)
        return sel.sum(axis=1).astype(np.int16)

    def buffered(self, region: Optional[int] = None) -> np.ndarray:
        """Per game: controlled countries held with at least one point to spare."""
        sel = self.controls() & (self.margin() >= 1)
        if region is not None:
            sel = sel & (_REGION[None, :] == region)
        return sel.sum(axis=1).astype(np.int16)

    def surplus(self) -> np.ndarray:
        """Per game: influence beyond a one-point buffer, where the side placed any.

        Reported, not judged. A second point of buffer is a real choice -- the owner's note on
        Italy is "3, and even one more eventually" -- so this is set beside the human figure
        rather than called waste. What it does catch is the opening that puts three more into a
        country already held.
        """
        extra = np.clip(self.margin() - 1, 0, None)
        touched = self.influence > 0
        return (extra * touched).sum(axis=1).astype(np.int16)


@dataclass
class SetupMeasurement:
    us: SideSetup
    ussr: SideSetup
    games: int
    #: Games where a placement decision had no legal action, which would mean the block was
    #: shorter than SETUP_DECISIONS and every row after it is misaligned. Must be 0.
    malformed: int = 0


def _rate(hits: np.ndarray) -> Tuple[float, float, float]:
    n = int(hits.shape[0])
    k = int(hits.sum())
    lo, hi = wilson_interval(k, n)
    return (k / n if n else float("nan")), lo, hi


def _country_name(cid: int) -> str:
    return str(ts.MapData.get_country_info(cid)["name"])


def measure(select_action: Any, num_games: int = 2000, base_seed: int = 20260911,
            batch_size: int = 512) -> SetupMeasurement:
    """Run `select_action` through the setup block of `num_games` fresh games.

    `select_action(obs, masks) -> array of flat action indices`, batched, matching what a model's
    `sample_action` gives. Kept as a callable so a bot or a fixed policy can be measured with the
    same code as a network.
    """
    us_rows: List[np.ndarray] = []
    ussr_rows: List[np.ndarray] = []
    us_has_es: List[bool] = []
    ussr_has_es: List[bool] = []
    us_final: List[np.ndarray] = []
    ussr_final: List[np.ndarray] = []
    malformed = 0

    done = 0
    while done < num_games:
        n = min(batch_size, num_games - done)
        runner = ts.VectorizedBatchRunner(n, base_seed + done * 7919)

        # The deal precedes setup, so who holds Europe Scoring is readable before a placement is
        # made -- which is what makes conditioning on it free.
        for i in range(n):
            loc = runner.get_state(i).get_card_location(EUROPE_SCORING)
            us_has_es.append(bool(ts.in_hand_of(loc, ts.Player.US)))
            ussr_has_es.append(bool(ts.in_hand_of(loc, ts.Player.USSR)))

        before = np.stack([_influence_row(runner.get_state(i)) for i in range(n)])

        for _ in range(SETUP_DECISIONS):
            obs = np.asarray(runner.get_observations(), dtype=np.float32)
            masks = np.asarray(runner.get_action_masks())
            if not masks.any(axis=1).all():
                malformed += int((~masks.any(axis=1)).sum())
            actions = select_action(obs, masks)
            runner.step_flat_all([int(a) for a in actions], auto_advance=False)

        for i in range(n):
            state = runner.get_state(i)
            if state.current_phase == ts.Phase.SETUP:
                malformed += 1
            after = _influence_row(state)
            us_rows.append((after[0] - before[i][0]).astype(np.int8))
            ussr_rows.append((after[1] - before[i][1]).astype(np.int8))
            us_final.append(after[0].copy())
            ussr_final.append(after[1].copy())
        done += n

    us_fin, ussr_fin = np.stack(us_final), np.stack(ussr_final)
    return SetupMeasurement(
        us=SideSetup(np.stack(us_rows), np.asarray(us_has_es, dtype=bool), us_fin, ussr_fin),
        ussr=SideSetup(np.stack(ussr_rows), np.asarray(ussr_has_es, dtype=bool), ussr_fin, us_fin),
        games=num_games,
        malformed=malformed,
    )


def _influence_row(state: ts.GameState) -> np.ndarray:
    """(2, 84) of US and USSR influence."""
    out = np.zeros((2, 84), dtype=np.int16)
    for cid in range(84):
        c = state.get_country(cid)
        out[0, cid] = int(c.us_influence)
        out[1, cid] = int(c.ussr_influence)
    return out


def torch_policy(model: Any, device: Any = "cpu", temperature: float = 0.1):
    """`select_action` for a ColdWar network. Checks the width before it can be misread."""
    import torch

    from bindings.ts_env import check_obs_width

    check_obs_width(model)
    model.eval()

    def select(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            actions, *_ = model.sample_action(
                torch.from_numpy(obs).to(device),
                torch.from_numpy(masks).to(device),
                temperature=temperature,
                deterministic=(temperature <= 0.05))
        return actions.reshape(-1).cpu().numpy()

    return select


def uniform_policy(seed: int = 0):
    """A control. If the probe cannot tell this from a real policy it is measuring nothing."""
    rng = np.random.default_rng(seed)

    def select(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        del obs
        return np.array([rng.choice(np.flatnonzero(row)) for row in masks])

    return select


# --- the human yardstick ----------------------------------------------------------------------

def measure_corpus(paths: Optional[Sequence[str]] = None) -> SetupMeasurement:
    """The same statistics from the human corpus.

    Read off the **board of the first log entry**, not out of a conversion. The opening placement
    is simultaneous and the log states it as a per-country total, so the finished board is the
    whole of it -- which means every game in the corpus counts, including the roughly half whose
    recording stops later and which a conversion therefore cannot use.

    The fixed starting influence (5 US in the UK, 3 USSR in East Germany, and the rest) is
    subtracted, exactly as the engine-side probe subtracts the board before the block. What is
    left is what the players chose.
    """
    import glob
    import gzip
    import json
    import os

    from tools.lib.corpus_paths import corpus_dir
    from tools.lib.ts_replayer_parse import country_id, initial_board

    if paths is None:
        paths = sorted(glob.glob(os.path.join(str(corpus_dir()), "*.json.gz")))

    base = initial_board()
    us_final: List[np.ndarray] = []
    ussr_final: List[np.ndarray] = []
    us_rows: List[np.ndarray] = []
    ussr_rows: List[np.ndarray] = []
    malformed = 0

    for path in paths:
        try:
            with gzip.open(path, "rt") as fh:
                game = json.load(fh)
            countries = game["all_turns"][0]["countries"]
        except Exception:
            malformed += 1
            continue

        us = np.zeros(84, dtype=np.int16)
        ussr = np.zeros(84, dtype=np.int16)
        for name, rec in countries.items():
            cid = country_id(name)
            if cid is None:
                continue
            us[cid] = int(rec.get("inflUS", 0) or 0)
            ussr[cid] = int(rec.get("inflUSSR", 0) or 0)
        board_us, board_ussr = us.copy(), ussr.copy()
        for cid, (bu, bs) in base.items():
            us[cid] -= bu
            ussr[cid] -= bs

        # A negative cell means the first recorded board is already past the opening -- an event
        # resolved into it, or the log starts late. Those games cannot answer this question.
        if (us < 0).any() or (ussr < 0).any():
            malformed += 1
            continue
        us_rows.append(us.astype(np.int8))
        ussr_rows.append(ussr.astype(np.int8))
        us_final.append(board_us)
        ussr_final.append(board_ussr)

    n = len(us_rows)
    empty = np.zeros(n, dtype=bool)     # the log does not state hands at setup
    zeros = np.zeros((0, 84), np.int8)
    us_fin = np.stack(us_final) if n else np.zeros((0, 84), np.int16)
    ussr_fin = np.stack(ussr_final) if n else np.zeros((0, 84), np.int16)
    return SetupMeasurement(
        us=SideSetup(np.stack(us_rows) if n else zeros, empty, us_fin, ussr_fin),
        ussr=SideSetup(np.stack(ussr_rows) if n else zeros, empty, ussr_fin, us_fin),
        games=n,
        malformed=malformed,
    )


def bid_split(m: SetupMeasurement) -> Dict[str, np.ndarray]:
    """Which games were played with a handicap, by how much each side placed.

    The engine's setup is 6 for the USSR and 9 for the US (7 plus the standard 2). A corpus game
    that placed more than that was played with a bid, and its Poland and West Germany figures are
    inflated relative to a game without one -- so the yardstick is reported both ways rather than
    as one number.
    """
    return {
        "ussr_placed": m.ussr.influence.sum(axis=1),
        "us_placed": m.us.influence.sum(axis=1),
        "standard": (m.ussr.influence.sum(axis=1) == 6) & (m.us.influence.sum(axis=1) == 9),
    }


def format_corpus_report(m: SetupMeasurement) -> str:
    split = bid_split(m)
    std = split["standard"]
    lines = [f"=== setup probe: human corpus ({m.games} games"
             + (f", {m.malformed} unusable" if m.malformed else "") + ") ==="]
    for name, side, cid, target, buffered in (
            ("USSR", m.ussr, POLAND, POLAND_TARGET, POLAND_BUFFERED),
            ("US", m.us, WEST_GERMANY, WEST_GERMANY_TARGET, WEST_GERMANY_BUFFERED)):
        held = side.final_at_least(cid, target)
        buf = side.final_at_least(cid, buffered)
        lines.append(f"  {name}: {_country_name(cid)} controlled {held[0] * 100:5.1f}%, "
                     f"with a buffer (>= {buffered}) {buf[0] * 100:5.1f}%   "
                     f"mean {side.final[:, cid].mean():.2f}   "
                     f"entropy {side.entropy():.2f} bits")
        lines.append(_margin_line(side))
        sub = SideSetup(side.influence[std], side.hand_has_europe_scoring[std],
                        side.final[std], side.opponent_final[std])
        if sub.influence.shape[0]:
            r, l, h = sub.final_at_least(cid, buffered)
            lines.append(f"      no handicap           n={int(std.sum()):5d}  "
                         f"buffered {r * 100:5.1f}% [{l * 100:.1f}, {h * 100:.1f}]")
        top = side.mean_per_country()
        order = np.argsort(-top)[:6]
        lines.append("      mean placement: " + ", ".join(
            f"{_country_name(int(c))} {top[c]:.2f}" for c in order if top[c] > 0.01))
    lines.append(f"  placed per game: USSR {split['ussr_placed'].mean():.2f} (engine 6), "
                 f"US {split['us_placed'].mean():.2f} (engine 9); "
                 f"{int(std.sum())} of {m.games} played without a handicap")
    return "\n".join(lines)


def _margin_line(side: "SideSetup") -> str:
    """Europe, held how firmly. Control at margin zero is one opponent influence from gone."""
    frag = side.fragile(EUROPE).mean()
    buff = side.buffered(EUROPE).mean()
    return (f"      Europe after setup: {buff:.2f} controlled with a buffer, "
            f"{frag:.2f} at exactly control (one influence from losing it); "
            f"surplus beyond one buffer {side.surplus().mean():.2f}")


def format_report(m: SetupMeasurement, label: str = "") -> str:
    lines: List[str] = [f"=== setup probe: {label or 'policy'} ({m.games} games) ==="]
    if m.malformed:
        lines.append(f"  !! {m.malformed} games did not have a well-formed setup block; "
                     f"every number below is suspect")

    for name, side, cid, target, buffered in (
            ("USSR", m.ussr, POLAND, POLAND_TARGET, POLAND_BUFFERED),
            ("US", m.us, WEST_GERMANY, WEST_GERMANY_TARGET, WEST_GERMANY_BUFFERED)):
        held = side.final_at_least(cid, target)
        buf = side.final_at_least(cid, buffered)
        lines.append(
            f"  {name}: {_country_name(cid)} controlled {held[0] * 100:5.1f}% "
            f"[{held[1] * 100:.1f}, {held[2] * 100:.1f}], "
            f"with a buffer (>= {buffered}) {buf[0] * 100:5.1f}% "
            f"[{buf[1] * 100:.1f}, {buf[2] * 100:.1f}]   "
            f"mean {side.final[:, cid].mean():.2f}   "
            f"entropy {side.entropy():.2f} bits   "
            f"{side.distinct_openings()} distinct openings")
        lines.append(_margin_line(side))

        # Conditioning on Europe Scoring: the one hand feature that plausibly moves the answer.
        for holder, sel in (("holds Europe Scoring", side.hand_has_europe_scoring),
                            ("does not", ~side.hand_has_europe_scoring)):
            n = int(sel.sum())
            if n == 0:
                continue
            sub = SideSetup(side.influence[sel], side.hand_has_europe_scoring[sel],
                            side.final[sel], side.opponent_final[sel])
            r, l, h = sub.final_at_least(cid, target)
            lines.append(f"      {holder:22s} n={n:5d}  controls {r * 100:5.1f}% "
                         f"[{l * 100:.1f}, {h * 100:.1f}]")

        top = side.mean_per_country()
        order = np.argsort(-top)[:6]
        lines.append("      mean placement: " + ", ".join(
            f"{_country_name(int(c))} {top[c]:.2f}" for c in order if top[c] > 0.01))
        lines.append("      most common openings:")
        for opening, count in side.openings(top=3):
            lines.append(f"        {count:4d}x  {opening}")
    return "\n".join(lines)
