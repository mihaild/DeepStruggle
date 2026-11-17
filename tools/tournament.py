#!/usr/bin/env python3
"""Unified Tournament & Matchup Evaluator for Twilight Struggle AI.

Supports both 2-model head-to-head benchmarking with granular loss cause diagnostics
and massive round-robin tournaments with Bradley-Terry MLE Elo rating matrices.
"""

import os

# Idle OpenMP workers sleep instead of spinning. Both PyTorch and the engine's batch runner use
# OpenMP, and with the default policy their idle workers spun between parallel regions: ~8.5 cores
# per training process for the same throughput that passive waiting reaches on ~1.6
# (research/log/training_throughput_cpu.md). Must be set before torch or ts_engine loads libgomp;
# an explicit OMP_WAIT_POLICY in the environment still wins.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import sys
import time
import json
import argparse
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine  # noqa: F401 -- the build require_fresh() checks before a game is played
from tools.lib.engine_fingerprint import require_fresh
from tools.lib.player_agent import PlayerAgent, load_agent, resolve_device
from tools.lib.batch_tournament import BatchMatchRunner, MatchupResult, compute_mle_elo
from tools.lib.parallel_tournament import (MatchOptions, Pair, Part, PartConfig,
                                           check_workers_device, merge_shards, plan_shards,
                                           pool_parts, run_matchups_parallel, run_shards,
                                           select_part, write_games_log, write_part)


def _write_text(path: str, text: str, what: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {what} to: {path}")


def format_loss_causes(causes: Dict[str, int], total_losses: int) -> str:
    if not causes or total_losses == 0:
        return "None (0 losses)"
    tot = max(sum(causes.values()), total_losses, 1)
    sorted_items = sorted(causes.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k}: {v} ({v/tot*100:.1f}%)" for k, v in sorted_items])


def run_head_to_head_report(
    name_a: str,
    name_b: str,
    matchup: Dict[str, Any],
    elo_ratings: Dict[str, float],
    elapsed_time: float,
) -> str:
    """Formats a detailed head-to-head match report for exactly two agents."""
    tot = matchup["total_games"]
    w_a = matchup["a_wins"]
    w_b = matchup["b_wins"]
    d = matchup["draws"]
    wr_a = (w_a / tot) * 100.0
    wr_b = (w_b / tot) * 100.0

    lines: List[str] = []
    lines.append("\n" + "=" * 85 + "\n")
    lines.append(f"TWILIGHT STRUGGLE HEAD-TO-HEAD BENCHMARK: {name_a} vs {name_b}\n")
    lines.append("=" * 85 + "\n\n")

    lines.append(f"• Total Games Played: {tot:,} ({tot // 2:,} per side) in {elapsed_time:.1f}s ({tot / max(0.1, elapsed_time):.1f} games/s)\n")
    lines.append(f"• Overall Result: {name_a} {w_a}W - {w_b}L - {d}D ({wr_a:.1f}% win rate)\n")
    lines.append(f"• Elo Ratings: {name_a} = {elo_ratings[name_a]:.1f} | {name_b} = {elo_ratings[name_b]:.1f} (Δ {elo_ratings[name_a] - elo_ratings[name_b]:+.1f})\n\n")

    lines.append("### Side-Specific Breakdown:\n")
    a_us_w = matchup["a_wins_as_us"]
    a_us_l = matchup["a_losses_as_us"]
    a_us_d = matchup.get("a_draws_as_us", 0)
    a_us_tot = a_us_w + a_us_l + a_us_d
    lines.append(f"  1. {name_a} (US) vs {name_b} (USSR):\n")
    lines.append(f"     - Record: {a_us_w}W - {a_us_l}L - {a_us_d}D ({a_us_w / max(1, a_us_tot) * 100:.1f}% US win rate)\n")
    lines.append(f"     - Average Steps: {matchup.get('avg_steps', 0.0):.1f}\n")
    lines.append(f"     - Losses by {name_a} (US): {format_loss_causes(matchup['causes_loss_us'], a_us_l)}\n\n")

    a_ussr_w = matchup["a_wins_as_ussr"]
    a_ussr_l = matchup["a_losses_as_ussr"]
    a_ussr_d = matchup.get("a_draws_as_ussr", 0)
    a_ussr_tot = a_ussr_w + a_ussr_l + a_ussr_d
    lines.append(f"  2. {name_a} (USSR) vs {name_b} (US):\n")
    lines.append(f"     - Record: {a_ussr_w}W - {a_ussr_l}L - {a_ussr_d}D ({a_ussr_w / max(1, a_ussr_tot) * 100:.1f}% USSR win rate)\n")
    lines.append(f"     - Average Steps: {matchup.get('avg_steps', 0.0):.1f}\n")
    lines.append(f"     - Losses by {name_a} (USSR): {format_loss_causes(matchup['causes_loss_ussr'], a_ussr_l)}\n\n")

    if "choice_stats" in matchup:
        cs = matchup["choice_stats"]
        lines.append("=" * 85 + "\n")
        lines.append("MICRO-ACTIONS & DETERMINISTIC SINGLE-CHOICE ANALYSIS (Grouped by Side)\n")
        lines.append("=" * 85 + "\n\n")
        lines.append(f"• USSR Decisions: {cs['ussr_single_choice_micro_actions']:,} single-choice / {cs['ussr_total_micro_actions']:,} total micro-actions ({cs['ussr_single_choice_pct']:.2f}% deterministic)\n")
        lines.append(f"• US Decisions:   {cs['us_single_choice_micro_actions']:,} single-choice / {cs['us_total_micro_actions']:,} total micro-actions ({cs['us_single_choice_pct']:.2f}% deterministic)\n")
        lines.append(f"• Combined Total: {cs['overall_single_choice_micro_actions']:,} single-choice / {cs['overall_total_micro_actions']:,} total micro-actions ({cs['overall_single_choice_pct']:.2f}% deterministic)\n\n")

        avg = cs["avg_per_game"]
        lines.append("• Per-Game Averages:\n")
        lines.append(f"  - USSR: {avg['ussr_total']:.1f} total micro-actions, {avg['ussr_single']:.1f} single-choice ({avg['ussr_single']/max(0.1, avg['ussr_total'])*100:.2f}%)\n")
        lines.append(f"  - US:   {avg['us_total']:.1f} total micro-actions, {avg['us_single']:.1f} single-choice ({avg['us_single']/max(0.1, avg['us_total'])*100:.2f}%)\n")
        lines.append(f"  - Both: {avg['combined_total']:.1f} total micro-actions, {avg['combined_single']:.1f} single-choice ({avg['combined_single']/max(0.1, avg['combined_total'])*100:.2f}%)\n\n")

        lines.append("### Breakdown of Deterministic (1 Valid Choice) Actions by Category:\n\n")
        lines.append("| Action Category | USSR Count (% of USSR 1-choice) | US Count (% of US 1-choice) | Total (% of All 1-choice) |\n")
        lines.append("|:---|:---:|:---:|:---:|\n")
        all_cats = sorted(set(list(cs["category_counts_ussr"].keys()) + list(cs["category_counts_us"].keys())))
        for cat in all_cats:
            cnt_ussr = cs["category_counts_ussr"].get(cat, 0)
            cnt_us = cs["category_counts_us"].get(cat, 0)
            cnt_tot = cnt_ussr + cnt_us
            pct_ussr = (cnt_ussr / max(1, cs["ussr_single_choice_micro_actions"])) * 100.0
            pct_us = (cnt_us / max(1, cs["us_single_choice_micro_actions"])) * 100.0
            pct_tot = (cnt_tot / max(1, cs["overall_single_choice_micro_actions"])) * 100.0
            lines.append(f"| **{cat}** | {cnt_ussr:,} ({pct_ussr:.1f}%) | {cnt_us:,} ({pct_us:.1f}%) | {cnt_tot:,} ({pct_tot:.1f}%) |\n")
        lines.append("\n")

    lines.append("=" * 85 + "\n")
    return "".join(lines)


def _pack_schedule(M: int, pack_pairs: int,
                   only: Optional[Sequence[Pair]] = None) -> List[List[Pair]]:
    """Every pairing (i < j) of M agents, grouped into packs that involve few distinct agents.

    A pack's cost per step is one forward per distinct agent in it, so the pairings of a pack
    should share agents: the agents are split into blocks of g = floor(sqrt(pack_pairs)), and a
    pack is either all g x g pairings between two blocks (2g agents) or the pairings inside
    several blocks (g agents each), filled up to pack_pairs. Every pairing appears exactly once.
    With `only`, the same packs keep just those pairings (a gauntlet rather than a round robin).
    """
    g = max(2, int(pack_pairs ** 0.5))
    blocks = [list(range(b, min(b + g, M))) for b in range(0, M, g)]
    packs: List[List[Pair]] = []
    for x in range(len(blocks)):
        for y in range(x + 1, len(blocks)):
            packs.append([(i, j) for i in blocks[x] for j in blocks[y]])
    inside: List[Pair] = []
    for blk in blocks:
        within = [(i, j) for a, i in enumerate(blk) for j in blk[a + 1:]]
        if inside and len(inside) + len(within) > pack_pairs:
            packs.append(inside)
            inside = []
        inside += within
    if inside:
        packs.append(inside)
    if only is not None:
        wanted = set(only)
        packs = [[pr for pr in p if pr in wanted] for p in packs]
    return [p for p in packs if p]


def _load_entrants(model_specs: List[str], device: torch.device,
                   opening: Optional[str],
                   names: Optional[Sequence[str]] = None) -> Tuple[List[PlayerAgent], List[str]]:
    """Every entrant, loaded, and the spec a worker process loads to get the same agent.

    With `opening`, every agent's setup is the named opening (tools/lib/openings.py; a per-agent
    `opening:<name>:` prefix on a spec takes precedence) and its name says so. Workers load their
    agents afresh from specs, so the opening travels as each spec's own prefix. `names`, one per
    spec, replace the agents' own names (tools/leaderboard.py names each entrant by its player id).
    """
    if names is not None and len(names) != len(model_specs):
        raise ValueError(f"{len(names)} names for {len(model_specs)} entrants")
    agents: List[PlayerAgent] = []
    worker_specs: List[str] = []
    for spec in model_specs:
        agent = load_agent(spec, device=device)
        worker_spec = spec
        if opening and not getattr(agent, "forced_opening", None):
            setattr(agent, "forced_opening", opening)
            setattr(agent, "name", f"{agent.name}+{opening}")
            worker_spec = f"opening:{opening}:{spec}"
        if names is not None:
            setattr(agent, "name", names[len(agents)])
        agents.append(agent)
        worker_specs.append(worker_spec)
        print(f" Loaded Agent: {agent.name:<35s} (from {spec})")

    # Names must be distinct AND attributable. They used to be disambiguated by appending #1 and
    # #2, which kept the tournament running but made its report unciteable: two runs snapshot at
    # identical step counts, so a field could report "snapshot_150011904steps#1 beat
    # snapshot_150011904steps#2" with no way to tell which run either was. Agents are now named
    # <run>@<steps> by checkpoint_id, so a genuine clash is a mistake and is refused.
    # --self-play enters one checkpoint twice on purpose; its second copy is named for what it is,
    # which stays attributable (both are the same checkpoint) where #1/#2 on two runs was not.
    if len(model_specs) == 2 and model_specs[0] == model_specs[1]:
        setattr(agents[1], "name", f"{agents[1].name}#self")
    name_counts: Dict[str, int] = {}
    for a in agents:
        name_counts[a.name] = name_counts.get(a.name, 0) + 1
    clashes = sorted(n for n, c in name_counts.items() if c > 1)
    if clashes:
        raise ValueError(
            "tournament entrants must have distinct names; these are repeated: "
            + ", ".join(clashes)
            + ". Each is <run>@<steps>, so a repeat means the same checkpoint was entered twice "
              "or two runs share a short name.")
    return agents, worker_specs


def _banner(model_specs: List[str], games_per_side: int, dev: torch.device,
            batch_chunk_size: int, workers: Optional[int], shard_pairs: int) -> None:
    print("\n" + "=" * 85)
    print(f" INITIALIZING TOURNAMENT EVALUATOR: {len(model_specs)} Models, {games_per_side * 2} Games/Pair")
    print(f" Device: {dev} | Chunk Size: {batch_chunk_size}"
          + (f" | Workers: {workers} x {shard_pairs} pairs/shard" if workers is not None else ""))
    print("=" * 85)


def _round_robin(n: int) -> List[Pair]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def run_massive_tournament(
    model_specs: List[str],
    games_per_side: int = 500,
    temperature: float = 0.1,
    batch_chunk_size: int = 1000,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
    output_report: Optional[str] = None,
    output_json: Optional[str] = None,
    device: str = "cuda",
    track_choices: bool = False,
    log_games: Optional[str] = None,
    auto_advance: bool = True,
    pack_pairs: int = 25,
    opening: Optional[str] = None,
    workers: Optional[int] = None,
    shard_pairs: int = 10,
    matchups: Optional[Sequence[Pair]] = None,
    entrant_names: Optional[Sequence[str]] = None,
    base_seed: int = 10000,
    on_matchup: Optional[Callable[[str, str, MatchupResult], None]] = None,
    progress_every: float = 0.0,
) -> Dict[str, Any]:
    """Runs high-throughput round-robin tournament across all specified models.

    `matchups` (pairs of indices into `model_specs`, i < j) plays only those pairings instead of
    the round robin; `entrant_names` names the agents; `base_seed` is the first game pair's deal
    seed (the same pairing at the same seed replays the same deals); `on_matchup(a, b, result)`
    is called as each pairing finishes, so a caller can keep what is done if the rest fails.

    `pack_pairs` pairings are played in one engine batch (BatchMatchRunner.play_packed_matchups),
    each agent's positions across them in one forward per step; the same deal seeds per game as
    one pairing at a time. 1 plays pairings one at a time, as before; choice tracking and game
    logs always do.

    `workers`, when given, splits every matchup into shards of `shard_pairs` game pairs and plays
    them in that many processes (tools/lib/parallel_tournament.py) instead of packing -- one worker
    included, so that at a fixed `shard_pairs` every worker count plays the same games. Packing and
    sharding play the same deals, but a sampling agent's draws depend on what shares its batch, so
    only deterministic agents play the same games both ways. One machine's share of a sharded
    tournament is `play_tournament_part`.
    """
    dev = resolve_device(device)
    if workers is not None:
        check_workers_device(workers, device)
    _banner(model_specs, games_per_side, dev, batch_chunk_size, workers, shard_pairs)
    agents, worker_specs = _load_entrants(model_specs, dev, opening, entrant_names)
    model_names = [a.name for a in agents]
    if matchups is None:
        matchups = _round_robin(len(agents))
    else:
        matchups = [(int(i), int(j)) for i, j in matchups]
        bad = [m for m in matchups if not 0 <= m[0] < m[1] < len(agents)]
        if bad or len(set(matchups)) != len(matchups):
            raise ValueError(f"matchups must be distinct index pairs i < j < {len(agents)}; got {bad or matchups}")
    results: Dict[Pair, MatchupResult] = {}
    t_start = time.time()

    def record(i: int, j: int, m_res: MatchupResult, pair_time: float) -> None:
        results[(i, j)] = m_res
        if on_matchup is not None:
            on_matchup(model_names[i], model_names[j], m_res)
        w_a, w_b, d, tot = m_res["a_wins"], m_res["b_wins"], m_res["draws"], m_res["total_games"]
        print(
            f"[{len(results):2d}/{len(matchups):2d}] {model_names[i]:<25s} vs {model_names[j]:<25s} -> "
            f"{w_a:4d}W - {w_b:4d}L - {d:2d}D ({(w_a / tot) * 100.0:5.1f}% win) in {pair_time:5.1f}s"
        )

    if workers is not None:
        # Wall time since the tournament started: shards of every matchup run at once, so a
        # matchup has no start time of its own.
        run_matchups_parallel(
            worker_specs, matchups, games_per_side, workers, shard_pairs,
            MatchOptions(device=str(dev), base_seed=base_seed, temperature=temperature,
                         batch_chunk_size=batch_chunk_size, track_choices=track_choices,
                         auto_advance=auto_advance),
            log_games=log_games,
            on_matchup_done=lambda p, r: record(p[0], p[1], r, time.time() - t_start))
    else:
        pack = 1 if (track_choices or log_games) else max(1, int(pack_pairs))
        groups_of_pairs = (_pack_schedule(len(agents), pack, matchups) if pack > 1
                           else [[pr] for pr in matchups])
        for group in groups_of_pairs:
            t_g = time.time()
            if len(group) == 1:
                (gi, gj), = group
                res = [BatchMatchRunner.play_parallel_matchup(
                    agents[gi], agents[gj], games_per_side=games_per_side, device=dev,
                    base_seed=base_seed, temperature=temperature, batch_chunk_size=batch_chunk_size,
                    track_choices=track_choices, log_games_file=log_games,
                    auto_advance=auto_advance, progress_every=progress_every)]
            else:
                res = BatchMatchRunner.play_packed_matchups(
                    [(agents[gi], agents[gj]) for gi, gj in group], games_per_side=games_per_side,
                    base_seed=base_seed, device=dev, temperature=temperature,
                    batch_chunk_size=batch_chunk_size,
                    auto_advance=auto_advance, progress_every=progress_every)
            share = (time.time() - t_g) / len(group)
            for (i, j), m_res in zip(group, res):
                record(i, j, m_res, share)

    return report_tournament(
        model_names, results, games_per_side, temperature, time.time() - t_start,
        anchor_model=anchor_model, anchor_elo=anchor_elo,
        output_report=output_report, output_json=output_json)


def play_tournament_part(
    model_specs: List[str],
    part: Part,
    output_part: str,
    games_per_side: int = 500,
    temperature: float = 0.1,
    batch_chunk_size: int = 1000,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
    device: str = "cuda",
    track_choices: bool = False,
    auto_advance: bool = True,
    opening: Optional[str] = None,
    workers: int = 1,
    shard_pairs: int = 10,
    base_seed: int = 10000,
) -> PartConfig:
    """One machine's share of a tournament spread over several: part `index` of `count` of the
    shards `run_massive_tournament(workers=...)` would play, written raw to `output_part` with
    their game logs. `pool_tournament_parts` turns a complete set of parts into the report."""
    dev = resolve_device(device)
    check_workers_device(workers, device)
    _banner(model_specs, games_per_side, dev, batch_chunk_size, workers, shard_pairs)
    agents, worker_specs = _load_entrants(model_specs, dev, opening)
    matchups = _round_robin(len(agents))
    t_start = time.time()
    index, count = part
    mine = select_part(plan_shards(matchups, games_per_side, shard_pairs), index, count)
    print(f" Part {index + 1} of {count}: {len(mine)} shard(s) of {shard_pairs} pair(s)")
    played = run_shards(
        worker_specs, mine, games_per_side, workers,
        MatchOptions(device=str(dev), base_seed=base_seed, temperature=temperature,
                     batch_chunk_size=batch_chunk_size, track_choices=track_choices,
                     auto_advance=auto_advance),
        capture_games=True,
        on_shard_done=lambda r: print(
            f"  shard {r.shard.label}: {r.result['a_wins']}W - {r.result['b_wins']}L - "
            f"{r.result['draws']}D", flush=True))
    config: PartConfig = {
        "models": [a.name for a in agents], "matchups": [list(m) for m in matchups],
        "games_per_side": games_per_side, "shard_pairs": shard_pairs,
        "temperature": temperature, "batch_chunk_size": batch_chunk_size,
        "track_choices": track_choices, "auto_advance": auto_advance,
        "anchor_model": anchor_model, "anchor_elo": anchor_elo,
    }
    if base_seed != 10000:
        config["base_seed"] = base_seed     # parts of the default deals stay poolable with older ones
    write_part(output_part, config, part, played, time.time() - t_start)
    print(f"Saved part {index + 1}/{count} to: {output_part}")
    return config


def pool_tournament_parts(
    part_paths: List[str],
    output_report: Optional[str] = None,
    output_json: Optional[str] = None,
    log_games: Optional[str] = None,
) -> Dict[str, Any]:
    """The report of a tournament played in parts on several machines (`--part`), from their
    part files. Refuses a set that is not exactly one tournament's shards."""
    pooled = pool_parts(part_paths)
    config = pooled.config
    print(f" Pooled {len(part_paths)} part(s): {len(pooled.shards)} shard(s), "
          f"slowest part {pooled.wall_seconds:.1f}s")
    if log_games:
        write_games_log(log_games, pooled.shards)
    merged = merge_shards(pooled.shards)
    results = {(m[0], m[1]): merged[(m[0], m[1])] for m in config["matchups"]}
    return report_tournament(
        config["models"], results, config["games_per_side"], config["temperature"],
        pooled.wall_seconds, anchor_model=config["anchor_model"], anchor_elo=config["anchor_elo"],
        output_report=output_report, output_json=output_json)


def report_tournament(
    model_names: List[str],
    results: Dict[Pair, MatchupResult],
    games_per_side: int,
    temperature: float,
    total_tournament_time: float,
    *,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
    output_report: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Elo, matrices and reports from each matchup's result, keyed by (row, column) index."""
    M = len(model_names)
    win_matrix = np.zeros((M, M), dtype=np.int32)
    loss_matrix = np.zeros((M, M), dtype=np.int32)
    draw_matrix = np.zeros((M, M), dtype=np.int32)
    ussr_win_matrix = np.zeros((M, M), dtype=np.int32)
    us_win_matrix = np.zeros((M, M), dtype=np.int32)
    total_matrix = np.zeros((M, M), dtype=np.int32)
    matchup_details: Dict[str, Any] = {}

    for (i, j), m_res in results.items():
        matchup_details[f"{model_names[i]}_vs_{model_names[j]}"] = m_res
        w_a, w_b, d, tot = m_res["a_wins"], m_res["b_wins"], m_res["draws"], m_res["total_games"]
        win_matrix[i, j] = w_a
        win_matrix[j, i] = w_b
        loss_matrix[i, j] = w_b
        loss_matrix[j, i] = w_a
        draw_matrix[i, j] = d
        draw_matrix[j, i] = d
        total_matrix[i, j] = tot
        total_matrix[j, i] = tot
        ussr_win_matrix[i, j] = m_res["a_wins_as_ussr"]
        ussr_win_matrix[j, i] = m_res["a_losses_as_us"]
        us_win_matrix[i, j] = m_res["a_wins_as_us"]
        us_win_matrix[j, i] = m_res["a_losses_as_ussr"]

    total_games_played = np.sum(total_matrix) // 2

    # Compute Bradley-Terry MLE Elo ratings
    elo_ratings = compute_mle_elo(model_names, win_matrix, total_matrix, anchor_model=anchor_model, anchor_elo=anchor_elo)

    # If exactly 2 models, print detailed head-to-head report
    if M == 2:
        m_key = f"{model_names[0]}_vs_{model_names[1]}"
        report_text = run_head_to_head_report(model_names[0], model_names[1], matchup_details[m_key], elo_ratings, total_tournament_time)
        print(report_text)
        if output_report:
            _write_text(output_report, report_text, "Markdown Head-to-Head Report")
        head_to_head = {
            "models": model_names,
            "elo_ratings": elo_ratings,
            "matchup": matchup_details[m_key],
            "games_per_side": games_per_side,
            "temperature": temperature,
            "total_time_seconds": total_tournament_time,
        }
        if output_json:
            _write_text(output_json, json.dumps(head_to_head, indent=2), "JSON Tournament Data")
        return head_to_head

    # Full Round-Robin Tournament Report
    sorted_indices = sorted(range(M), key=lambda idx: elo_ratings[model_names[idx]], reverse=True)
    sorted_names = [model_names[idx] for idx in sorted_indices]

    total_wr_matrix = np.zeros((M, M))
    ussr_wr_matrix = np.zeros((M, M))
    us_wr_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                total_wr_matrix[i, j] = (win_matrix[i, j] / max(1, total_matrix[i, j])) * 100.0
                ussr_wr_matrix[i, j] = (ussr_win_matrix[i, j] / max(1, games_per_side)) * 100.0
                us_wr_matrix[i, j] = (us_win_matrix[i, j] / max(1, games_per_side)) * 100.0

    report_lines: List[str] = []
    report_lines.append("# Twilight Struggle AI: Massive Tournament Evaluation Report\n\n")
    report_lines.append(f"- **Total Models**: {M}\n")
    report_lines.append(f"- **Total Games Played**: {total_games_played:,}\n")
    report_lines.append(f"- **Games Per Matchup Pair**: {games_per_side * 2:,} ({games_per_side} per side)\n")
    report_lines.append(f"- **Sampling Temperature**: {temperature} ({'deterministic' if temperature <= 0.05 else 'sampled'})\n")
    report_lines.append(f"- **Total Evaluation Time**: {total_tournament_time:.1f} seconds ({total_games_played / max(0.1, total_tournament_time):.1f} games/sec)\n\n")

    report_lines.append("## 1. Bradley-Terry MLE Elo Leaderboard\n\n")
    report_lines.append("| Rank | Model | Elo Rating | Total Matches | Total Record (W-L-D) | Overall Win Rate |\n")
    report_lines.append("|:---:|:---|:---:|:---:|:---:|:---:|\n")

    for rank, idx in enumerate(sorted_indices, start=1):
        name = model_names[idx]
        elo = elo_ratings[name]
        tot_w = int(np.sum(win_matrix[idx, :]))
        tot_l = int(np.sum(loss_matrix[idx, :]))
        tot_d = int(np.sum(draw_matrix[idx, :]))
        tot_g = tot_w + tot_l + tot_d
        wr = (tot_w / max(1, tot_g)) * 100.0
        report_lines.append(f"| **{rank}** | **{name}** | **{elo:.1f}** | {tot_g:,} | {tot_w:,}W - {tot_l:,}L - {tot_d:,}D | **{wr:.1f}%** |\n")
    report_lines.append("\n---\n\n")

    # Matrices
    report_lines.append("## 2. Head-to-Head Total Win Rate Matrix (% Win for Row vs Column)\n\n")
    header_cols = " | ".join([f"**{name}**" for name in sorted_names])
    report_lines.append(f"| Model | {header_cols} |\n")
    report_lines.append(f"|:---|{'---:|' * M}\n")
    for i in sorted_indices:
        row_vals = []
        for j in sorted_indices:
            row_vals.append("—" if i == j else f"{total_wr_matrix[i, j]:.1f}%")
        report_lines.append(f"| **{model_names[i]}** | {' | '.join(row_vals)} |\n")
    report_lines.append("\n---\n\n")

    # Per-side matrix. The pooled matrix above averages the two sides, which hides exactly the
    # thing worth seeing in an asymmetric game: an arm can be at 50% overall while winning
    # almost every game on one side and almost none on the other.
    report_lines.append("## 3. Per-Side Win Rate Matrix (Row vs Column: as USSR / as US)\n\n")
    report_lines.append(
        "Each cell is the **row** model's win rate against the **column** model, "
        f"playing USSR and playing US, over {games_per_side} games per side.\n\n")
    report_lines.append(f"| Model | {header_cols} |\n")
    report_lines.append(f"|:---|{'---:|' * M}\n")
    for i in sorted_indices:
        row_vals = []
        for j in sorted_indices:
            if i == j:
                row_vals.append("—")
            else:
                row_vals.append(f"{ussr_wr_matrix[i, j]:.1f}% / {us_wr_matrix[i, j]:.1f}%")
        report_lines.append(f"| **{model_names[i]}** | {' | '.join(row_vals)} |\n")
    report_lines.append("\n")

    # The side split each arm shows across the whole field, and the gap between the two.
    report_lines.append("### Side balance per arm (across all opponents)\n\n")
    report_lines.append("| Model | as USSR | as US | USSR - US |\n")
    report_lines.append("|:---|---:|---:|---:|\n")
    for i in sorted_indices:
        opps = [j for j in range(M) if j != i]
        if not opps:
            continue
        as_ussr = float(np.mean([ussr_wr_matrix[i, j] for j in opps]))
        as_us = float(np.mean([us_wr_matrix[i, j] for j in opps]))
        report_lines.append(
            f"| **{model_names[i]}** | {as_ussr:.1f}% | {as_us:.1f}% | "
            f"{as_ussr - as_us:+.1f} pp |\n")
    report_lines.append("\n---\n\n")

    report_text = "".join(report_lines)

    if output_report:
        _write_text(output_report, report_text, "Markdown Tournament Report")

    # Per-pair, per-seat records. `win_matrix` aggregates the two seats into one number, which is
    # the wrong shape for the question a round robin over one lineage's snapshots is usually
    # asked: a seat collapse shows as one side falling while the other holds, and an aggregate
    # averages exactly that signal away. BatchMatchRunner already computes the split for every
    # pair -- it was simply never serialised, so every caller that wanted it re-ran the games.
    per_side = {
        key: {
            "a_wins_as_us": m["a_wins_as_us"],
            "a_wins_as_ussr": m["a_wins_as_ussr"],
            "win_rate_a_as_us": m["win_rate_a_as_us"],
            "win_rate_a_as_ussr": m["win_rate_a_as_ussr"],
            "games_per_side": m["games_per_side"],
            "draws": m["draws"],
            "avg_turn": m["avg_turn"],
        }
        for key, m in matchup_details.items()
    }

    summary_data = {
        "models": model_names,
        "elo_ratings": elo_ratings,
        "win_matrix": win_matrix.tolist(),
        "total_matrix": total_matrix.tolist(),
        "per_side": per_side,
        "games_per_side": games_per_side,
        "temperature": temperature,
        "total_time_seconds": total_tournament_time,
    }

    if output_json:
        _write_text(output_json, json.dumps(summary_data, indent=2), "JSON Tournament Data")

    print("\n" + report_text)
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Massive Parallel Tournament & Elo Rating Benchmark")
    parser.add_argument("--models", nargs="+", default=None, help="List of model checkpoints or bot names ('random', 'heuristic')")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to auto-discover all snapshot checkpoints")
    parser.add_argument("--include-baselines", action="store_true", default=False, help="Explicitly include RandomBot and HeuristicBot")
    parser.add_argument("--games-per-side", type=int, default=500, help="Games per side per matchup (total 2x games per pair)")
    parser.add_argument("--base-seed", type=int, default=10000,
                        help="The first game pair's deal seed. Searchers and samplers are seeded from the "
                             "deal, so the same entrants with the same seed replay the same games; a "
                             "replicate of a match needs another seed (e.g. 20000).")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature for every neural agent. <= 0.05 plays the "
                             "argmax. The default 0.1 samples, which is fine for comparing "
                             "models of similar sharpness but confounds a comparison between "
                             "models whose policy entropy differs -- the sharper one plays "
                             "closer to its own argmax and wins on temperature rather than on "
                             "strength. Re-rate at 0.0 when the arms differ in entropy.")
    parser.add_argument("--batch-chunk-size", type=int, default=1000, help="Max parallel games executed in a single vectorized batch")
    parser.add_argument("--progress-every", type=float, default=600.0,
                        help="Seconds between progress lines while a matchup plays (games finished, the first "
                             "agent's score so far, the live games' turn, a rough time left); 0 = none. "
                             "Not reported with --workers.")
    parser.add_argument("--pack-pairs", type=int, default=25,
                        help="Pairings played together in one engine batch, each agent's positions "
                             "in one forward per step (same deal seeds per game as one at a time). "
                             "1 = one pairing at a time. Choice tracking and game logs force 1.")
    parser.add_argument("--anchor-model", type=str, default="HeuristicBot", help="Model name to anchor Elo ratings")
    parser.add_argument("--anchor-elo", type=float, default=1500.0, help="Anchor Elo rating value")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save Markdown report")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON results")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device; falls back to cpu when there is no GPU (resolve_device). "
             "The default was cpu, which contradicted this module's own function "
             "signature and made every tournament far slower than it needed to be -- "
             "200 games took 118s on cpu against 6.4s on cuda, an 18x difference.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Processes to play games in, one core each; 0 = every core. Given at "
                             "all (1 included), each matchup is split into shards of --shard-pairs "
                             "game pairs; omitted, the tournament runs in this process with "
                             "--pack-pairs. Worth it for bots that decide in Python "
                             "(heuristic_mcts, search:); a network-only field is already batched "
                             "and gains little. Above 1 needs --device cpu.")
    parser.add_argument("--shard-pairs", type=int, default=10,
                        help="Game pairs (one deal played from both sides) per shard with "
                             "--workers. Results depend on this, not on --workers: at a fixed "
                             "value any number of workers, 1 included, plays the same games.")
    parser.add_argument("--part", type=str, default=None, metavar="I/N",
                        help="Play only part I of N (1-based) of the tournament's shards and write "
                             "them to --output-part, for a tournament spread over N machines. "
                             "Every part must use the same entrants, --games-per-side, "
                             "--shard-pairs, temperature and chunk size; --pool-parts checks it.")
    parser.add_argument("--output-part", type=str, default=None,
                        help="Where --part writes its shards (JSON).")
    parser.add_argument("--pool-parts", nargs="+", default=None, metavar="PART_JSON",
                        help="Report a tournament from the part files of all N --part runs, "
                             "without playing a game. Refuses a set that misses or repeats a "
                             "shard. Honours --output-report, --output-json and --log-games.")
    parser.add_argument("--track-choices", action="store_true", default=False, help="Track and report micro-actions with exactly 1 valid choice")
    parser.add_argument("--log-games", type=str, default=None, help="Path to save per-game JSONL execution logs")
    parser.add_argument("--opening", type=str, default=None,
                        help="Every agent's setup is this named opening (tools/lib/openings.py) instead "
                             "of its own placements. One agent alone: prefix its spec with "
                             "opening:<name>:. Needed to rate a checkpoint trained with "
                             "--forced-opening, whose setup was never learned.")
    parser.add_argument("--self-play", action="store_true", default=False, help="Evaluate model against itself")
    # Default ON. The function signature defaulted to True while this flag defaulted to
    # False and line 413 passes it through unconditionally, so every CLI run settled
    # nothing and the signature change was invisible -- "the training setup you think you
    # configured is not the one that ran". The opt-out is explicit instead.
    parser.add_argument("--no-auto-advance", dest="auto_advance", action="store_false",
                        default=True,
                        help="Present decisions with a single legal action instead of "
                             "settling them in the engine. Changes the decision stream, so "
                             "a run with it is not comparable with a run without.")

    args = parser.parse_args()

    if args.pool_parts:
        # Pools results already played (each part records its engine); plays nothing.
        pool_tournament_parts(args.pool_parts, output_report=args.output_report,
                              output_json=args.output_json, log_games=args.log_games)
        return
    require_fresh()

    part: Optional[Part] = None
    if args.part:
        try:
            i_str, n_str = args.part.split("/")
            part = (int(i_str) - 1, int(n_str))
        except ValueError:
            parser.error(f"--part must look like I/N, got {args.part!r}")
        if not 0 <= part[0] < part[1]:
            parser.error(f"--part {args.part}: I must be in 1..N")
        if not args.output_part:
            parser.error("--part needs --output-part")
        # A part writes its raw shards and nothing else; the report, its JSON and the game log
        # come from --pool-parts over every part.
        for flag, value in (("--output-report", args.output_report),
                            ("--output-json", args.output_json),
                            ("--log-games", args.log_games)):
            if value:
                parser.error(f"{flag} does not apply to --part; pass it to --pool-parts")

    models = list(args.models) if args.models else []
    if args.checkpoint_dir:
        if not os.path.isdir(args.checkpoint_dir):
            parser.error(f"--checkpoint-dir {args.checkpoint_dir}: no such directory")
        # resume_*.pt is training state (optimizer, RNG), not a policy, and cannot be loaded as one.
        discovered = [os.path.join(args.checkpoint_dir, f) for f in sorted(os.listdir(args.checkpoint_dir))
                      if f.endswith(".pt") and not f.startswith("resume_")]
        if not discovered:
            parser.error(f"--checkpoint-dir {args.checkpoint_dir}: no checkpoint (.pt) in it")
        models.extend(discovered)
        # When evaluating a checkpoint directory, default to including baselines unless explicitly specified
        if not args.models and not args.include_baselines:
            args.include_baselines = True

    if args.include_baselines:
        if "random" not in models and "RandomBot" not in models:
            models.insert(0, "random")
        if "heuristic" not in models and "HeuristicBot" not in models:
            models.insert(1, "heuristic")

    if (args.self_play or len(models) == 1) and len(models) == 1:
        models = [models[0], models[0]]

    if len(models) < 2:
        print("Error: Need at least 2 models for an evaluation or tournament (or pass --self-play).")
        sys.exit(1)

    workers: Optional[int] = None
    if args.workers is not None:
        workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
        try:
            check_workers_device(workers, args.device)
        except ValueError as e:
            parser.error(str(e))
    if part is not None:
        play_tournament_part(
            models, part, args.output_part,
            games_per_side=args.games_per_side,
            temperature=args.temperature,
            batch_chunk_size=args.batch_chunk_size,
            anchor_model=args.anchor_model,
            anchor_elo=args.anchor_elo,
            device=args.device,
            track_choices=args.track_choices,
            auto_advance=args.auto_advance,
            opening=args.opening,
            workers=1 if workers is None else workers,
            shard_pairs=args.shard_pairs,
            base_seed=args.base_seed,
        )
        return

    run_massive_tournament(
        model_specs=models,
        games_per_side=args.games_per_side,
        temperature=args.temperature,
        batch_chunk_size=args.batch_chunk_size,
        pack_pairs=args.pack_pairs,
        progress_every=args.progress_every,
        opening=args.opening,
        anchor_model=args.anchor_model,
        anchor_elo=args.anchor_elo,
        output_report=args.output_report,
        output_json=args.output_json,
        device=args.device,
        track_choices=args.track_choices,
        log_games=args.log_games,
        auto_advance=args.auto_advance,
        workers=workers,
        shard_pairs=args.shard_pairs,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
