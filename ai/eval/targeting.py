"""Does the network pick *where* well, and does where depend on *what it is doing*?

Two questions the Elo number cannot separate, both about the country-choice node.

**Ranking.** When placing influence, the engine can say exactly what each legal country is worth
in regional VP right now. So the model's choice can be scored against the truth: does it take the
best one, and where does its pick sit in the true ordering? Reported as lift over choosing
uniformly among the legal targets, which is the only baseline that is fair when the legal set is
itself informative.

**Conditioning.** The same 84 country actions serve very different operations. Resolving The
Voice of America removes Soviet influence, so the US should want countries the USSR holds
heavily; placing influence from an ordinary card wants countries where the US can gain. A network
with one static country preference would target the same places for both. Measured as the
opponent-influence lift of the chosen country, compared across operations in the same rollout.

The legal masks differ between operations, so the raw choice distributions are not comparable and
are not compared. Everything here is a lift *within* an operation's own legal set, which the mask
cannot manufacture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

import ts_engine as ts
from ai.eval.marginal_scoring import marginal_placement_value, region_investment
from bindings.action_encoder import ActionEncoder


@dataclass
class OpStats:
    """Accumulated choices for one kind of operation."""
    n: int = 0
    legal_sizes: List[int] = field(default_factory=list)
    #: opponent influence in the chosen country, and the mean over that node's legal set
    opp_chosen: List[float] = field(default_factory=list)
    opp_uniform: List[float] = field(default_factory=list)
    #: battleground share, chosen vs uniform over legal
    bg_chosen: List[float] = field(default_factory=list)
    bg_uniform: List[float] = field(default_factory=list)
    #: influence-placement only: true regional-VP value of the pick, and the legal-set stats
    vp_chosen: List[float] = field(default_factory=list)
    vp_uniform: List[float] = field(default_factory=list)
    vp_best: List[float] = field(default_factory=list)
    took_best: List[float] = field(default_factory=list)
    percentile: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        def m(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else float("nan")
        out = {
            "decisions": float(self.n),
            "mean_legal_targets": m([float(x) for x in self.legal_sizes]),
            "opp_influence_chosen": m(self.opp_chosen),
            "opp_influence_uniform": m(self.opp_uniform),
            "opp_influence_lift": m(self.opp_chosen) - m(self.opp_uniform),
            "battleground_chosen": m(self.bg_chosen),
            "battleground_uniform": m(self.bg_uniform),
        }
        if self.vp_chosen:
            out.update({
                "vp_chosen": m(self.vp_chosen),
                "vp_uniform": m(self.vp_uniform),
                "vp_best_available": m(self.vp_best),
                "vp_lift": m(self.vp_chosen) - m(self.vp_uniform),
                "took_the_best": m(self.took_best),
                "rank_percentile": m(self.percentile),
            })
        return out


def region_appetite(model: Any, num_games: int = 192, temperature: float = 0.1,
                    max_iters: int = 20_000,
                    min_turn: int = 0,
                    investment_every: int = 24) -> Dict[str, Dict[str, float]]:
    """Per region at influence-placement nodes: can it place there, does it, and is it worth it?

    Low placement into a region has three very different explanations, and they are separated
    here rather than guessed at:

    * **It cannot.** Placement needs adjacency to something the player controls, so a region with
      no foothold offers no legal targets at all. `legal_share` is that.
    * **It will not.** `chosen_share / legal_share` is the preference ratio: 1.0 is indifference
      to the region, below 1.0 is avoidance *given the chance*.
    * **There is nothing there.** `vp_available` is the engine's true marginal regional VP over
      that region's legal targets. A region can be legal, ignored, and correctly ignored.

    `min_turn` exists because `vp_available` is the VP a region would pay *now*, while three of
    the six regions are scored by mid-war cards that are not in the deck early. Under-investing
    in them on turn 3 may be correct rather than blind, so the comparison has to be repeatable on
    later turns where those cards are live.

    `vp_available` is the *one-step* marginal and understates a region badly, because Presence
    needs a controlled country: a single point into a region you are absent from scores zero, so
    every fresh investment looks worthless and only finishing counts. `vp_at_4` and
    `points_to_next_status` come from `region_investment` and say what sustained play would buy.
    They cost about 500 engine evaluations per position, so they are sampled every
    `investment_every` nodes rather than computed at each.

    The gap worth acting on is a region with legal targets, a preference ratio well under 1, and
    real VP reachable within a few points.
    """
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=616_000)
    obs, masks, _ = env.reset_all()
    finished = [False] * num_games
    legal_n: Dict[Any, float] = {}
    chosen_n: Dict[Any, float] = {}
    vp_avail: Dict[Any, List[float]] = {}
    vp_taken: Dict[Any, List[float]] = {}
    turns: List[float] = []
    inv_vp4: Dict[Any, List[float]] = {}
    inv_next: Dict[Any, List[float]] = {}
    nodes = 0

    for _ in range(max_iters):
        if all(finished):
            break
        with torch.no_grad():
            acts, *_ = model.sample_action(
                torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(masks)).to(device), temperature=temperature)
        a = acts.cpu().numpy().astype(np.int64)

        for i in range(num_games):
            if finished[i]:
                continue
            st = env.runner.get_state(i)
            if ts.Engine.is_terminal(st):
                continue
            ctx = st.ctx()
            if ctx.decision_type != ts.DecisionType.POINT_NODE:
                continue
            if int(ctx.resolving_card) != 0 or int(ctx.op_mode) != int(ts.OpMode.INFLUENCE):
                continue
            p = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                else st.phasing_player
            if p == ts.Player.NONE:
                continue
            if int(st.turn) < min_turn:
                continue
            legal = [c for c in range(84) if masks[i][ActionEncoder.NODE_OFFSET + c]]
            if len(legal) < 4:
                continue
            ma = ts.ActionMask.decode_flat_action(st, int(a[i]))
            chosen = int(ma.primary_id)
            if chosen not in legal:
                continue

            nodes += 1
            turns.append(float(st.turn))
            v = marginal_placement_value(st, p)
            for c in legal:
                r = ts.MapData.get_country_info(c)["region"]
                legal_n[r] = legal_n.get(r, 0.0) + 1.0
                vp_avail.setdefault(r, []).append(float(v[c]))
            rc = ts.MapData.get_country_info(chosen)["region"]
            chosen_n[rc] = chosen_n.get(rc, 0.0) + 1.0
            vp_taken.setdefault(rc, []).append(float(v[chosen]))

            if nodes % investment_every == 0:
                for r, prof in region_investment(st, p).items():
                    inv_vp4.setdefault(r, []).append(float(prof["vp_at_4"]))
                    inv_next.setdefault(r, []).append(float(prof["points_to_next_status"]))

        obs, masks, _, dones, info = env.step(a)
        for i, r in enumerate(info["ending_reasons"]):
            if r and not finished[i]:
                finished[i] = True

    total_legal = sum(legal_n.values()) or 1.0
    out: Dict[str, Dict[str, float]] = {}
    for r in legal_n:
        ls = legal_n[r] / total_legal
        cs = chosen_n.get(r, 0.0) / max(nodes, 1)
        out[str(r)] = {
            "legal_share": ls,
            "chosen_share": cs,
            "preference_ratio": cs / ls if ls > 1e-9 else float("nan"),
            "vp_available": float(np.mean(vp_avail[r])) if vp_avail.get(r) else float("nan"),
            "vp_taken": float(np.mean(vp_taken[r])) if vp_taken.get(r) else float("nan"),
            "legal_targets_per_node": legal_n[r] / max(nodes, 1),
            "vp_at_4": float(np.mean(inv_vp4[r])) if inv_vp4.get(r) else float("nan"),
            "points_to_next_status": (float(np.mean(inv_next[r]))
                                      if inv_next.get(r) else float("nan")),
        }
    out["_meta"] = {"nodes": float(nodes), "mean_turn": float(np.mean(turns)) if turns else 0.0}
    return out


def _op_label(state: ts.GameState) -> str:
    """What this country choice is *for*."""
    ctx = state.ctx()
    card = int(ctx.resolving_card)
    if card != 0:
        return f"event:{ts.CardData.get_card_name(card)}"
    mode = int(ctx.op_mode)
    return {int(ts.OpMode.INFLUENCE): "ops:influence",
            int(ts.OpMode.COUP): "ops:coup",
            int(ts.OpMode.REALIGN): "ops:realign"}.get(mode, f"ops:{mode}")


def measure_targeting(model: Any, num_games: int = 256, temperature: float = 0.1,
                      max_iters: int = 20_000,
                      score_vp_for: str = "ops:influence") -> Dict[str, Dict[str, float]]:
    """Country choices by operation, each scored against its own legal set.

    `score_vp_for` names the one operation whose choices are also scored against the engine's
    true marginal regional VP. That answer key costs 84 engine evaluations per decision, so it is
    computed for influence placement only -- the operation it actually means something for.
    """
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=828_000)
    obs, masks, _ = env.reset_all()
    stats: Dict[str, OpStats] = {}
    finished = [False] * num_games

    bg = np.array([bool(ts.MapData.get_country_info(c)["battleground"]) for c in range(84)],
                  dtype=np.float64)

    for _ in range(max_iters):
        if all(finished):
            break
        with torch.no_grad():
            acts, *_ = model.sample_action(
                torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(masks)).to(device), temperature=temperature)
        a = acts.cpu().numpy().astype(np.int64)

        for i in range(num_games):
            if finished[i]:
                continue
            st = env.runner.get_state(i)
            if ts.Engine.is_terminal(st):
                continue
            ctx = st.ctx()
            if ctx.decision_type != ts.DecisionType.POINT_NODE:
                continue
            p = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                else st.phasing_player
            if p == ts.Player.NONE:
                continue

            ma = ts.ActionMask.decode_flat_action(st, int(a[i]))
            chosen = int(ma.primary_id)
            if not 0 <= chosen < 84:
                continue
            # Country actions occupy 119..202 of the flat space.
            legal = [c for c in range(84) if masks[i][ActionEncoder.NODE_OFFSET + c]]
            if len(legal) < 2 or chosen not in legal:
                continue

            label = _op_label(st)
            s = stats.setdefault(label, OpStats())
            s.n += 1
            s.legal_sizes.append(len(legal))

            opp = np.array([float(st.get_country(c).ussr_influence if p == ts.Player.US
                                 else st.get_country(c).us_influence) for c in legal])
            s.opp_chosen.append(float(opp[legal.index(chosen)]))
            s.opp_uniform.append(float(opp.mean()))
            s.bg_chosen.append(float(bg[chosen]))
            s.bg_uniform.append(float(bg[legal].mean()))

            if label == score_vp_for:
                v = marginal_placement_value(st, p)[legal]
                pick = float(v[legal.index(chosen)])
                s.vp_chosen.append(pick)
                s.vp_uniform.append(float(v.mean()))
                s.vp_best.append(float(v.max()))
                s.took_best.append(1.0 if pick >= v.max() - 1e-9 else 0.0)
                s.percentile.append(float((v <= pick + 1e-9).mean()))

        obs, masks, _, dones, info = env.step(a)
        for i, r in enumerate(info["ending_reasons"]):
            if r and not finished[i]:
                finished[i] = True

    return {k: v.summary() for k, v in sorted(stats.items(), key=lambda kv: -kv[1].n)}
