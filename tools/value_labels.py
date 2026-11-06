#!/usr/bin/env python3
"""P34's labelling tool: decision positions labelled by greedy playouts to the game's end.

Three stages, each shardable with --part i/n (research/plans/P34_search_centric_evaluator.md):

  positions  Self-play of the generator at --temperature, one decision in --keep kept: the state's
             bytes, its game and ply, the mover, the generator's own game result from the mover's
             side, and the generator's top --candidates legal moves (the sibling slice's
             candidates). Game g is played from --seed and g; games with g % 20 == 0 ... are held out
             by --holdout (by game, never by position).
  label      Every position's --worlds worlds played to the end by the generator on both sides
             under the safe-playout rule (bindings/playout_labels.hpp), and, for the first
             --siblings positions of a fixed shuffle, each candidate in --sibling-worlds shared
             worlds with its leaf (the mover's next decision) kept.
  merge      The parts joined into train / holdout files with the labels as means, a manifest with
             the D code, the generator, the engine, device, seed and sha256 of every file.
  exact      The exact slice (CPU, after label): --exact positions of turn 10's last action rounds,
             each solved exactly in the labeller's first --deals worlds (tools/scripts/blunder_bank.
             exact_value: dice averaged, every later choice the mover's best and the opponent's best
             reply). Against the playout label on the same positions it is the bias the labels
             inherit from the generator's greedy follow-up. `merge` reports it when present.

    tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/value_labels.py \\
        positions --model <network or .pt> --games 20000 --out data/datasets/p34/<name>
    ... label --dir data/datasets/p34/<name> [--part 0/1]
    ... merge --dir data/datasets/p34/<name>     # the D code is derived (tools/scripts/run_codes.py)

The labeller's network forward runs as a CUDA graph over a fixed batch, from page-locked buffers,
with two labellers alternating so the engine steps one while the GPU evaluates the other.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from ai.search.rollout_root import RolloutConfig, RolloutRoot
from bindings.ts_env import model_obs_features
from tools.lib import leaderboard
from tools.lib.action_view import checkpoint_merged_influence
from tools.lib.checkpoint_id import checkpoint_label
from tools.lib.data_root import data_root
from tools.lib.engine_behaviour import measure
from tools.lib.engine_fingerprint import fingerprint, require_fresh
from tools.lib.jsonl import sha256_file
from tools.lib.player_agent import NeuralAgent
from tools.scripts.blunder_bank import Unprovable, exact_value
from tools.scripts.run_codes import labelling_code

A = ActionEncoder.BASE_ACTION_SIZE
#: A position's id is game * PLY_SPAN + ply.
PLY_SPAN = 100_000
REGIONS = ("europe", "asia", "middle_east", "africa", "central_america", "south_america", "southeast_asia")


# ---- shared ------------------------------------------------------------------------------------

def _part(text: str) -> Tuple[int, int]:
    i, n = (int(x) for x in text.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise argparse.ArgumentTypeError(f"--part {text}: want i/n with 0 <= i < n")
    return i, n


def resolve_model(spec: str) -> Tuple[str, str]:
    """(path, id) for a registered network name or a checkpoint path."""
    board = leaderboard.load()
    if spec in board.networks:
        return os.path.join(data_root(), str(board.networks[spec]["file"])), spec
    if not os.path.isfile(spec):
        raise SystemExit(f"{spec}: neither a registered network nor a file")
    return spec, checkpoint_label(spec)


def load_model(path: str, device: torch.device) -> Any:
    if checkpoint_merged_influence(path):
        # The labeller steps the E4 view, as the rollout root does.
        raise SystemExit(f"{path} decides in the merged-influence view, which the labeller does not play")
    return NeuralAgent.from_checkpoint(path, device=str(device)).model.eval()


_BEHAVIOUR: List[str] = []


def behaviour() -> str:
    """The imported engine's behaviour digest id (tools/lib/engine_behaviour.py), measured once."""
    if not _BEHAVIOUR:
        _BEHAVIOUR.append(leaderboard.behaviour_id(measure()))
    return _BEHAVIOUR[0]


def same_games(meta: Dict[str, Any], engine: str) -> bool:
    """The build that wrote `meta` is this one (the same engine source fingerprint), so the raw state bytes
    the dataset stores are read as the positions they were. Only the same build: raw bytes are the struct's
    memory (`ts.state_from_raw_bytes`), and a build that plays the same games can still lay the struct out
    differently -- main's c9e5689 added fields to GameState with the behaviour digest unchanged, and a
    dataset written before it decoded as other positions (P36c, 2026-10-19). An earlier version accepted
    the same behaviour digest here, which is not enough."""
    return meta["engine"] == engine


def engine_checked() -> str:
    require_fresh()
    return fingerprint()


class GraphedPolicy:
    """logits = model(obs) over a fixed batch for one labeller: fill(labeller) writes the live rows'
    observations into page-locked memory, launch(n) runs the forward asynchronously, and wait()
    returns the logits.

    With `codes` (CUDA only) each observation float travels as its one-byte code in the labeller's
    codebook and is decoded on the GPU inside the graph: exact, and a quarter of the upload, which
    is what bounds the loop otherwise. Policies built with the same `streams` share one upload
    stream and one compute stream: one policy's upload overlaps another's forward, and no two graphs
    ever replay at once (two graphs replaying concurrently on separate streams hung the GPU,
    2026-10-17)."""

    def __init__(self, model: Any, capacity: int, width: int, device: torch.device, graphs: bool = True,
                 streams: Optional[Tuple[Any, Any]] = None, codes: bool = True) -> None:
        self.model = model
        self.device = device
        cuda = device.type == "cuda"
        self.codes = codes and cuda
        self.buf_t = torch.zeros((capacity, width), dtype=torch.uint8 if self.codes else torch.float32,
                                 pin_memory=cuda)
        self.buf = self.buf_t.numpy()
        self.logits_t = torch.zeros((capacity, A), dtype=torch.float32, pin_memory=cuda)
        self.logits = self.logits_t.numpy()
        self.graph: Optional[Any] = None
        self.event: Optional[Any] = None
        self.book_n = 0
        if not cuda:
            return
        self.buf_dev = torch.zeros((capacity, width), dtype=self.buf_t.dtype, device=device)
        self.book_t = torch.zeros(256, dtype=torch.float32, pin_memory=True)
        self.book_dev = torch.zeros((256, 1), dtype=torch.float32, device=device)
        self.event = torch.cuda.Event()
        self.uploaded = torch.cuda.Event()
        self.copy_stream, self.compute = streams if streams is not None else policy_streams(device)
        if not graphs:
            return
        self.compute.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.compute), torch.no_grad():
            for _ in range(2):
                self._forward()
        torch.cuda.current_stream(device).wait_stream(self.compute)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out_dev = self._forward()
        torch.cuda.synchronize(device)

    def _forward(self, n: Optional[int] = None) -> torch.Tensor:
        x = self.buf_dev if n is None else self.buf_dev[:n]
        if self.codes:
            x = torch.nn.functional.embedding(x.int(), self.book_dev).squeeze(-1)
        return self.model(x, None)[0].float()

    def fill(self, labeller: Any) -> int:
        if not self.codes:
            return int(labeller.fill_live(self.buf))
        n = int(labeller.fill_live_codes(self.buf))
        book = labeller.codebook()
        if book.size != self.book_n:
            self.book_t[:book.size] = torch.from_numpy(book)
            self.book_n = int(book.size)
        return n

    def launch(self, n: int) -> None:
        if n == 0:
            return
        if self.device.type != "cuda":
            with torch.no_grad():
                self.logits_t[:n] = self.model(self.buf_t[:n], None)[0].float()
            return
        assert self.event is not None
        with torch.cuda.stream(self.copy_stream):
            self.buf_dev[:n].copy_(self.buf_t[:n], non_blocking=True)
            if self.codes:
                self.book_dev.view(-1).copy_(self.book_t, non_blocking=True)
            self.uploaded.record(self.copy_stream)
        with torch.cuda.stream(self.compute):
            self.compute.wait_event(self.uploaded)
            if self.graph is not None:
                self.graph.replay()
                out = self.out_dev
            else:
                with torch.no_grad():
                    out = self._forward(n)
            self.logits_t[:n].copy_(out[:n], non_blocking=True)
            self.event.record(self.compute)

    def wait(self) -> np.ndarray:
        if self.event is not None:
            self.event.synchronize()
        return self.logits


def policy_streams(device: torch.device) -> Tuple[Any, Any]:
    """(upload stream, compute stream) for GraphedPolicy objects that alternate."""
    return torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)


# ---- positions ---------------------------------------------------------------------------------

def run_positions(a: argparse.Namespace) -> int:
    engine = engine_checked()
    device = torch.device(a.device)
    path, model_id = resolve_model(a.model)
    model = load_model(path, device)
    features = model_obs_features(model)
    part, parts = a.part
    games = [g for g in range(a.games) if g % parts == part]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    envs = min(a.envs, len(games))
    runner = ts.VectorizedBatchRunner(envs, a.seed)
    runner.set_obs_features([features] * envs, [features] * envs)
    rng = np.random.default_rng([a.seed, part, 1])
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed * 1_000_003 + part)
    nxt = 0
    game_of = np.full(envs, -1, dtype=np.int64)
    ply_of = np.zeros(envs, dtype=np.int64)
    for i in range(envs):
        game_of[i] = games[nxt]
        runner.reset_game(i, a.seed * PLY_SPAN + games[nxt])
        nxt += 1
    kept: Dict[str, List[Any]] = {k: [] for k in ("state", "id", "mover", "turn", "cand")}
    result_of: Dict[int, float] = {}   # game -> US score (1 / 0.5 / 0)
    done_games = 0
    t0 = time.time()
    while done_games < len(games):
        live = game_of >= 0
        obs = torch.from_numpy(np.asarray(runner.get_observations())).to(device, non_blocking=True)
        masks = torch.from_numpy(np.asarray(runner.get_action_masks())).to(device)
        with torch.no_grad():
            logits = model(obs, None)[0].float()
        legal = masks > 0
        legal[:, 0] |= ~legal.any(dim=1)   # a finished env has no legal move; it is not stepped
        z = logits.masked_fill(~legal, float("-inf"))
        probs = torch.softmax(z / a.temperature, dim=-1)
        acts = torch.multinomial(probs, 1, generator=gen).squeeze(1).cpu().numpy()
        top = z.topk(a.candidates, dim=-1)
        top_idx = top.indices.cpu().numpy()
        top_ok = torch.isfinite(top.values).cpu().numpy()
        keep = live & (rng.random(envs) < 1.0 / a.keep)
        movers = runner.get_decision_players()
        for i in np.flatnonzero(keep):
            st = runner.get_state(int(i))
            kept["state"].append(np.frombuffer(st.raw_bytes(), dtype=np.uint8))
            kept["id"].append(int(game_of[i]) * PLY_SPAN + int(ply_of[i]))
            kept["mover"].append(int(movers[i]))
            kept["turn"].append(int(st.turn))
            kept["cand"].append(np.where(top_ok[i], top_idx[i], -1).astype(np.int16))
        res = runner.step_flat_all([int(x) for x in acts], True)
        if any(res[i] == 0 for i in np.flatnonzero(live)):
            raise RuntimeError("the engine refused a sampled self-play action")
        ply_of[live] += 1
        terms = runner.get_terminals()
        utils = runner.get_terminal_utilities()
        for i in np.flatnonzero(live):
            if not terms[i]:
                continue
            result_of[int(game_of[i])] = 0.5 * (1.0 + float(utils[i]))
            done_games += 1
            if nxt < len(games):
                game_of[i] = games[nxt]
                ply_of[i] = 0
                runner.reset_game(int(i), a.seed * PLY_SPAN + games[nxt])
                nxt += 1
            else:
                game_of[i] = -1
        if done_games and done_games % 1000 == 0 and terms.count(True):
            print(f"  {done_games}/{len(games)} games, {len(kept['id'])} positions, {time.time() - t0:.0f}s",
                  flush=True)
    ids = np.asarray(kept["id"], dtype=np.int64)
    game = ids // PLY_SPAN
    us_score = np.asarray([result_of[int(g)] for g in game], dtype=np.float32)
    mover = np.asarray(kept["mover"], dtype=np.int8)
    us = int(ts.Player.US)
    order = np.argsort(ids)
    file = out / f"positions_{part:03d}_of_{parts:03d}.npz"
    np.savez(file,
             state=np.stack(kept["state"])[order] if kept["state"] else np.zeros((0, ts.state_size()), np.uint8),
             id=ids[order],
             mover=mover[order],
             turn=np.asarray(kept["turn"], dtype=np.int8)[order],
             cand=np.stack(kept["cand"])[order] if kept["cand"] else np.zeros((0, a.candidates), np.int16),
             game_score=np.where(mover == us, us_score, 1.0 - us_score)[order].astype(np.float32),
             holdout=(_holdout(game, a.holdout, a.seed))[order])
    meta = {"stage": "positions", "generator": model_id, "generator_file": path,
            "generator_sha256": sha256_file(path), "engine": engine, "games": a.games,
            "part": f"{part}/{parts}", "seed": a.seed, "temperature": a.temperature, "keep": a.keep,
            "candidates": a.candidates, "holdout": a.holdout, "obs_features": features, "behaviour": behaviour(),
            "device": str(device), "positions": int(ids.size), "seconds": round(time.time() - t0, 1)}
    file.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{file}: {ids.size} positions from {len(games)} games in {time.time() - t0:.0f}s")
    return 0


def _holdout(game: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """A game's split from its index alone, so every part and every stage agrees."""
    h = (game.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed)) >> np.uint64(40)
    return (h.astype(np.float64) / float(1 << 24)) < frac


# ---- label -------------------------------------------------------------------------------------

def _load_positions(d: pathlib.Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    files = sorted(d.glob("positions_*.npz"))
    if not files:
        raise SystemExit(f"{d}: no positions_*.npz")
    metas = [json.loads(f.with_suffix(".json").read_text()) for f in files]
    for m in metas[1:]:
        for k in ("generator_sha256", "engine", "seed", "temperature", "keep"):
            if m[k] != metas[0][k]:
                raise SystemExit(f"positions parts disagree on {k}")
    parts = {int(m["part"].split("/")[1]) for m in metas}
    if len(parts) != 1 or len(files) != parts.pop():
        raise SystemExit(f"{d}: the positions parts are incomplete")
    arrs = [dict(np.load(f)) for f in files]
    cat = {k: np.concatenate([x[k] for x in arrs]) for k in arrs[0]}
    order = np.argsort(cat["id"])
    return {k: v[order] for k, v in cat.items()}, metas[0]


def sibling_ids(ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    """The sibling slice: the first n of a fixed shuffle of the position ids."""
    perm = np.random.default_rng([seed, 2]).permutation(ids.size)
    return np.sort(ids[perm[:n]])


class SearchFollow:
    """D3's stronger follow-up: at each playout's entry decisions -- action-round card choices and
    headline cards -- from the position's turn through `window` turns after it, the move is the keyed
    rollout root's (k=4, `worlds`, 4 boundaries, stop=earlier, leaf=own) instead of the greedy argmax.
    Elsewhere the playout stays greedy. `searched` counts the decisions it chose."""

    SELECT_CARD = int(ts.DecisionType.SELECT_CARD)

    def __init__(self, model: Any, device: torch.device, worlds: int, window: int, seed: int,
                 rule: str = "z2") -> None:
        self.root = RolloutRoot(model, RolloutConfig(k=4, worlds=worlds, horizon=4, rule=rule, seed=seed,
                                                     stop="earlier", leaf_side="own", keyed=True), device=device)
        self.window = window
        self.searched = 0
        self.deviated = 0       # searched decisions where the root's move is not the network's own

    def __call__(self, pl: Any) -> Optional[np.ndarray]:
        info = pl.listed_info()
        if info.shape[0] == 0:
            return None
        want = np.flatnonzero((info[:, 0] == self.SELECT_CARD) & (info[:, 2] - info[:, 3] <= self.window))
        if want.size == 0:
            return None
        picks = self.root.choose([pl.listed_state(int(j)) for j in want])
        self.deviated += sum(1 for p, st in zip(picks, self.root.last_stats)
                             if st["candidates"] and p != st["candidates"][0])
        forced = np.full(info.shape[0], -1, dtype=np.int32)
        forced[want] = np.asarray(picks, dtype=np.int32)
        self.searched += int(want.size)
        return forced


def _drive(labellers: Sequence[Any], policies: Sequence[GraphedPolicy], feed: Any, sink: Any,
           follow: Optional[SearchFollow] = None) -> int:
    """Run the labellers to idle, alternating so one steps while the other's batch is on the GPU; with
    `follow`, its choices override the greedy move of the rows it picks."""
    n = [0] * len(labellers)
    for j, (pl, pol) in enumerate(zip(labellers, policies)):
        feed(pl)
        n[j] = pol.fill(pl)
        pol.launch(n[j])
    rows = 0
    while any(not pl.idle() for pl in labellers):
        for j, (pl, pol) in enumerate(zip(labellers, policies)):
            if n[j] == 0 and pl.idle():
                feed(pl)
                if pl.idle():
                    continue
            logits = pol.wait()
            if n[j]:
                forced = follow(pl) if follow is not None else None
                pl.step(logits, forced)
                rows += n[j]
            sink(pl)
            feed(pl)
            n[j] = pol.fill(pl)
            pol.launch(n[j])
    return rows


def run_label(a: argparse.Namespace) -> int:
    engine = engine_checked()
    d = pathlib.Path(a.dir)
    pos, pmeta = _load_positions(d)
    if not same_games(pmeta, engine):
        raise SystemExit(f"positions were written by engine {pmeta['engine'][:8]}, this is {engine[:8]}, and no "
                         "leaderboard epoch lists both")
    device = torch.device(a.device)
    path, model_id = resolve_model(a.model) if a.model else (pmeta["generator_file"], pmeta["generator"])
    if sha256_file(path) != pmeta["generator_sha256"] and not a.model:
        raise SystemExit("the generator file changed since the positions were written")
    model = load_model(path, device)
    features = model_obs_features(model)
    part, parts = a.part
    mine = np.arange(pos["id"].size) % parts == part
    ids, states = pos["id"][mine], pos["state"][mine]
    sib = np.isin(ids, sibling_ids(pos["id"], a.siblings, a.seed))
    cand = pos["cand"][mine]
    t0 = time.time()

    labellers = [ts.PlayoutLabels(features=features, worlds=a.worlds, capacity=a.capacity, seed=a.seed,
                                  keyed=a.keyed, keyed2=a.keyed2) for _ in range(2)]
    sib_labellers = [ts.PlayoutLabels(features=features, worlds=a.sibling_worlds, capacity=a.capacity,
                                      seed=a.seed + 1, leaf_horizon=a.leaf_horizon, keyed=a.keyed,
                                      keyed2=a.keyed2) for _ in range(2)]
    streams = policy_streams(device) if device.type == "cuda" else None
    policies = [GraphedPolicy(model, a.capacity, labellers[0].obs_width(), device, graphs=not a.no_graphs,
                              streams=streams, codes=not a.float_upload) for _ in range(2)]

    main_out: List[Dict[str, np.ndarray]] = []
    sib_out: List[Dict[str, np.ndarray]] = []
    cursor = {"main": 0, "sib": 0}
    chunk = max(1, 2 * a.capacity // a.worlds)
    sib_rows = np.flatnonzero(sib)

    def feed_main(pl: Any) -> None:
        while pl.queued() < chunk and cursor["main"] < ids.size:
            lo = cursor["main"]
            hi = min(ids.size, lo + chunk)
            pl.enqueue_raw(np.ascontiguousarray(states[lo:hi]), np.ascontiguousarray(ids[lo:hi]))
            cursor["main"] = hi

    def feed_sib(pl: Any) -> None:
        k = a.sibling_k
        per = max(1, 2 * a.capacity // (a.sibling_worlds * k))
        while pl.queued() < per and cursor["sib"] < sib_rows.size:
            lo = cursor["sib"]
            take = sib_rows[lo:lo + per]
            cursor["sib"] = lo + take.size
            c = cand[take, :k].astype(np.int32)
            ok = (c >= 0).all(axis=1)   # positions with fewer than k legal moves stay out of the slice
            if ok.any():
                pl.enqueue_raw(np.ascontiguousarray(states[take[ok]]), np.ascontiguousarray(ids[take[ok]]),
                               np.ascontiguousarray(c[ok]))

    last = [time.time()]

    def sink_main(pl: Any) -> None:
        got = pl.take_positions()
        if got["id"].size:
            main_out.append(got)
        if time.time() - last[0] > 60:
            last[0] = time.time()
            done = sum(x["id"].size for x in main_out)
            print(f"  {done}/{ids.size} positions labelled, {time.time() - t0:.0f}s", flush=True)

    def sink_sib(pl: Any) -> None:
        got = pl.take_siblings()
        if got["id"].size:
            sib_out.append(got)

    follow = (SearchFollow(model, device, a.search_worlds, a.search_window, a.seed, a.search_rule)
              if a.follow == "search" else None)
    rows = 0 if a.no_main else _drive(labellers, policies, feed_main, sink_main, follow)
    t_main = time.time() - t0
    print(f"main labels: {ids.size} positions x {a.worlds} worlds, {rows} network rows in {t_main:.0f}s "
          f"({rows / max(t_main, 1e-9):,.0f} rows/s)", flush=True)
    srows = _drive(sib_labellers, policies, feed_sib, sink_sib, follow) if sib_rows.size else 0
    t_all = time.time() - t0

    def cat(chunks: List[Dict[str, np.ndarray]], key: str, width: Tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.concatenate([c[key] for c in chunks]) if chunks else np.zeros((0,) + width, dtype)

    file = d / f"labels_{part:03d}_of_{parts:03d}.npz"
    if not a.no_main:
        np.savez(file, id=cat(main_out, "id", (), np.int64), count=cat(main_out, "count", (2,), np.int32),
                 sums=cat(main_out, "sums", (5,), np.float32), swing_sum=cat(main_out, "swing_sum", (7,), np.float32),
                 swing_n=cat(main_out, "swing_n", (7,), np.int32))
    sfile = d / f"siblings_{part:03d}_of_{parts:03d}.npz"
    np.savez(sfile, id=cat(sib_out, "id", (), np.int64), meta=cat(sib_out, "meta", (5,), np.int32),
             cols=cat(sib_out, "cols", (4,), np.float32), swing=cat(sib_out, "swing", (7,), np.int32),
             leaf=cat(sib_out, "leaf", (ts.state_size(),), np.uint8))
    meta = {"stage": "label", "labeller": model_id, "labeller_sha256": sha256_file(path),
            "engine": engine, "behaviour": behaviour(), "positions_engine": pmeta["engine"], "part": f"{part}/{parts}", "seed": a.seed, "worlds": 0 if a.no_main else a.worlds,
            "main": not a.no_main, "safe": True,
            "siblings": a.siblings, "sibling_worlds": a.sibling_worlds, "sibling_k": a.sibling_k,
            "leaf_horizon": a.leaf_horizon, "keyed": a.keyed, **({"keyed2": True} if a.keyed2 else {}),
            "capacity": a.capacity, "device": str(device),
            "follow": a.follow, "search_worlds": a.search_worlds, "search_window": a.search_window,
            "searched_decisions": follow.searched if follow is not None else 0,
            "search_rule": a.search_rule, "deviated_decisions": follow.deviated if follow is not None else 0,
            "torch": torch.__version__, "cuda_graphs": not a.no_graphs, "code_upload": not a.float_upload,
            "network_rows": int(rows + srows), "seconds": round(t_all, 1)}
    # A siblings-only part (--no-main) has no labels file; its record goes beside the siblings.
    (sfile if a.no_main else file).with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{file}: {sum(x['id'].size for x in main_out)} positions; {sfile}: "
          f"{sum(x['id'].size for x in sib_out)} sibling rows; {rows + srows} network rows in {t_all:.0f}s")
    return 0


# ---- select --------------------------------------------------------------------------------------

def run_select(a: argparse.Namespace) -> int:
    """A subset of a labelled set's positions as a positions file of its own, to be labelled again:
    `contested` -- entry decisions (card choice, headline) whose label lies in [lo, hi], positions of
    the sibling slice whose candidate means differ by --spread or more first, then a fixed shuffle --
    or `exact`, the exact slice's positions."""
    src = pathlib.Path(a.dir)
    pos, pmeta = _load_positions(src)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "exact":
        files = sorted(src.glob("exact_*.npz"))
        if not files:
            raise SystemExit(f"{src}: no exact slice")
        want = np.unique(np.concatenate([np.load(f)["id"] for f in files]))
        for f in files:      # the exact values travel with their positions
            (out / f.name).write_bytes(f.read_bytes())
            if f.with_suffix(".json").exists():
                (out / f.with_suffix(".json").name).write_bytes(f.with_suffix(".json").read_bytes())
        params: Dict[str, Any] = {"what": "exact", "of": pmeta.get("dataset", src.name)}
    else:
        lab = [dict(np.load(f)) for f in sorted(src.glob("labels_*.npz"))]
        if not lab:
            raise SystemExit(f"{src}: no main labels")
        L = {k: np.concatenate([x[k] for x in lab]) for k in lab[0]}
        o = np.argsort(L["id"])
        score = (L["sums"][o, 0] / np.maximum(L["count"][o, 0], 1)).astype(np.float64)
        entry = np.asarray([ts.state_from_raw_bytes(b.tobytes()).ctx().decision_type == ts.DecisionType.SELECT_CARD
                            for b in pos["state"]])
        ok = entry & (score >= a.lo) & (score <= a.hi)
        first = np.zeros(pos["id"].size, bool)
        if a.siblings:
            z = np.load(pathlib.Path(a.siblings) / "siblings.npz")
            meta, ids, y = z["meta"], z["id"], z["cols"][:, 0]
            u, inv = np.unique(ids, return_inverse=True)
            sums = np.zeros((u.size, 4)); cnt = np.zeros((u.size, 4))
            np.add.at(sums, (inv, meta[:, 0]), y)
            np.add.at(cnt, (inv, meta[:, 0]), 1)
            means = sums / np.maximum(cnt, 1)
            spread = means.max(1) - means.min(1)
            first = np.isin(pos["id"], u[spread >= a.spread])
        rng = np.random.default_rng([a.seed, 4])
        order = rng.permutation(pos["id"].size)
        order = np.concatenate([order[(ok & first)[order]], order[(ok & ~first)[order]]])
        want = np.sort(pos["id"][order[:a.n]])
        params = {"what": "contested", "of": pmeta.get("dataset", src.name), "n": int(want.size), "lo": a.lo,
                  "hi": a.hi, "spread": a.spread, "seed": a.seed, "eligible": int(ok.sum()), "first": int((ok & first).sum())}
    sel = np.isin(pos["id"], want)
    file = out / "positions_000_of_001.npz"
    np.savez(file, state=pos["state"][sel], id=pos["id"][sel], mover=pos["mover"][sel], turn=pos["turn"][sel],
             cand=pos["cand"][sel], game_score=pos["game_score"][sel], holdout=pos["holdout"][sel])
    meta = dict(pmeta)
    meta.update(part="0/1", positions=int(sel.sum()), subset=params)
    file.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{file}: {int(sel.sum())} positions ({params})")
    return 0


# ---- exact -------------------------------------------------------------------------------------

def exact_ids(pos: Dict[str, np.ndarray], n: int, seed: int, min_round: int) -> np.ndarray:
    """The exact slice: n positions of turn 10 at action round >= min_round, from a fixed shuffle."""
    rounds = np.asarray([ts.state_from_raw_bytes(b.tobytes()).action_round if t == 10 else 0
                         for b, t in zip(pos["state"], pos["turn"])])
    pool = pos["id"][(pos["turn"] == 10) & (rounds >= min_round)]
    perm = np.random.default_rng([seed, 3]).permutation(pool.size)
    return np.sort(pool[perm[:n]])


def run_exact(a: argparse.Namespace) -> int:
    engine_checked()
    d = pathlib.Path(a.dir)
    pos, pmeta = _load_positions(d)
    part, parts = a.part
    want = exact_ids(pos, a.exact, a.seed, a.min_round)
    want = want[np.arange(want.size) % parts == part]
    row_of = {int(i): r for r, i in enumerate(pos["id"].tolist())}
    # The labeller's own worlds: world w of a position is a function of (seed, id, w) alone.
    pl = ts.PlayoutLabels(worlds=a.deals, capacity=a.deals, seed=a.seed)
    values = np.full((want.size, a.deals), np.nan, dtype=np.float32)
    t0 = time.time()
    for j, pid in enumerate(want.tolist()):
        base = ts.state_from_raw_bytes(pos["state"][row_of[pid]].tobytes())
        mover = ts.Player(int(pos["mover"][row_of[pid]]))
        for w in range(a.deals):
            world = pl.world(base, pid, w)
            try:
                values[j, w] = exact_value(world, mover, int(world.turn), [a.budget], "min")
            except Unprovable:
                pass
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{want.size} positions, {time.time() - t0:.0f}s", flush=True)
    file = d / f"exact_{part:03d}_of_{parts:03d}.npz"
    np.savez(file, id=want, value=values)
    meta = {"stage": "exact", "engine": pmeta["engine"], "part": f"{part}/{parts}", "seed": a.seed,
            "exact": a.exact, "deals": a.deals, "budget": a.budget, "min_round": a.min_round,
            "solved_worlds": int(np.isfinite(values).sum()), "seconds": round(time.time() - t0, 1)}
    file.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{file}: {want.size} positions, {meta['solved_worlds']} of {values.size} worlds solved")
    return 0


def exact_reads(d: pathlib.Path, pos: Dict[str, np.ndarray], score: np.ndarray) -> Optional[Dict[str, Any]]:
    """The exact slice against the labels: the mean of (label - exact) over positions solved in every
    world, overall and by side -- the bias of greedy follow-up play."""
    files = sorted(d.glob("exact_*.npz"))
    if not files:
        return None
    parts = [dict(np.load(f)) for f in files]
    ids = np.concatenate([p["id"] for p in parts])
    vals = np.concatenate([p["value"] for p in parts])
    full = np.isfinite(vals).all(axis=1)
    rows = np.searchsorted(pos["id"], ids)
    gap = score[rows] - vals.mean(axis=1)
    out: Dict[str, Any] = {"positions": int(ids.size), "solved": int(full.sum())}
    if full.any():
        out["bias"] = float(gap[full].mean())
        out["bias_se"] = float(gap[full].std() / max(1.0, np.sqrt(full.sum() - 1)))
        out["mean_abs_gap"] = float(np.abs(gap[full]).mean())
        for side, name in ((int(ts.Player.US), "us"), (int(ts.Player.USSR), "ussr")):
            m = full & (pos["mover"][rows] == side)
            if m.any():
                out[f"bias_{name}"] = float(gap[m].mean())
    return out


# ---- merge -------------------------------------------------------------------------------------

def labelling_signature(pmeta: Dict[str, Any], lmeta: Dict[str, Any]) -> Dict[str, Any]:
    """The labelling recipe: every setting that changes what a dataset holds, none of where or how
    fast it was made. Its D code comes from tools/scripts/run_codes.py's `labellings`."""
    return {"temperature": pmeta["temperature"], "keep": pmeta["keep"], "candidates": pmeta["candidates"],
            "holdout": pmeta["holdout"], "worlds": lmeta["worlds"] if lmeta.get("main", True) else 0, "stop": "game", "safe": lmeta["safe"],
            "siblings": lmeta["siblings"], "sibling_worlds": lmeta["sibling_worlds"],
            "sibling_k": lmeta["sibling_k"], "leaf_horizon": lmeta["leaf_horizon"],
            "labels": ["score", "final_vp", "turn_vp", "turn_defcon", "region_swing"],
            # keyed chance (bindings/keyed_chance.hpp); recorded only when on, so D1 keeps its code
            **({"chance": "keyed2"} if lmeta.get("keyed2") else {"chance": "keyed"} if lmeta.get("keyed") else {}),
            **({"main": False} if lmeta.get("main") is False else {}),
            # the follow-up: recorded only when not greedy, so D1 and D2 keep their codes
            **({"follow": {"player": "rollout(k=4,stop=earlier,leaf=own,chance=keyed) at entry decisions",
                           "worlds": lmeta["search_worlds"], "window_turns": lmeta["search_window"],
                           "rule": lmeta.get("search_rule", "z2")}}
               if lmeta.get("follow", "greedy") != "greedy" else {}),
            **({"subset": pmeta["subset"]} if pmeta.get("subset") else {})}


def merge_siblings_only(a: argparse.Namespace, d: pathlib.Path, pos: Dict[str, np.ndarray],
                        pmeta: Dict[str, Any]) -> int:
    """A dataset of the sibling slice alone (label --no-main): siblings.npz with each row's split, the
    manifest and the ledger row."""
    sfiles = sorted(d.glob("siblings_*.npz"))
    smetas = [json.loads(f.with_suffix(".json").read_text()) for f in sfiles]
    if not sfiles or len(sfiles) != int(smetas[0]["part"].split("/")[1]):
        raise SystemExit(f"{d}: the sibling parts are incomplete")
    for m in smetas:
        if not same_games(m, pmeta["engine"]):
            raise SystemExit("a sibling part was written by another engine than the positions")
    sib = [dict(np.load(f)) for f in sfiles]
    S = {k: np.concatenate([x[k] for x in sib]) for k in sib[0]}
    hold_games = np.unique(pos["id"][pos["holdout"]] // PLY_SPAN)
    S["holdout"] = np.isin(S["id"] // PLY_SPAN, hold_games)
    f = d / "siblings.npz"
    np.savez(f, id=S["id"], meta=S["meta"], cols=S["cols"], swing=S["swing"], leaf=S["leaf"], holdout=S["holdout"])
    files = {f.name: sha256_file(str(f))}
    sig = labelling_signature(pmeta, smetas[0])
    code = labelling_code(sig, update=True)
    assert code is not None
    if a.code and a.code != code:
        raise SystemExit(f"the labelling settings are {code}'s, not {a.code}'s (tools/scripts/run_codes.py labellings)")
    name = f"{pmeta['generator']}~{code}(seed={pmeta['seed']},games={pmeta['games']})"
    manifest = {"dataset": name, "code": code, "labelling": sig, "generator": pmeta["generator"],
                "generator_sha256": pmeta["generator_sha256"], "engine": pmeta["engine"],
                "positions": {k: v for k, v in pmeta.items() if k not in ("part", "positions", "seconds")},
                "label": {k: v for k, v in smetas[0].items() if k not in ("part", "network_rows", "seconds")},
                "rows": {"sibling_rows": int(S["id"].size), "sibling_positions": int(np.unique(S["id"]).size),
                         "network_rows": int(sum(m["network_rows"] for m in smetas))},
                "label_seconds": round(sum(m["seconds"] for m in smetas), 1), "files": files}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["rows"]))
    print(f"| `{name}` | {manifest['rows']['sibling_positions']:,} (sibling slice only) | "
          f"{manifest['rows']['network_rows']:,} | {smetas[0]['device']} | siblings.npz {files['siblings.npz'][:12]} | |")
    return 0


def run_merge(a: argparse.Namespace) -> int:
    d = pathlib.Path(a.dir)
    pos, pmeta = _load_positions(d)
    if not list(d.glob("labels_*.npz")):
        return merge_siblings_only(a, d, pos, pmeta)
    lfiles = sorted(d.glob("labels_*.npz"))
    lmetas = [json.loads(f.with_suffix(".json").read_text()) for f in lfiles]
    if not lfiles or len(lfiles) != int(lmetas[0]["part"].split("/")[1]):
        raise SystemExit(f"{d}: the label parts are incomplete")
    for m in lmetas:
        if not same_games(m, pmeta["engine"]):
            raise SystemExit("a label part was written by another engine than the positions")
    lab = [dict(np.load(f)) for f in lfiles]
    L = {k: np.concatenate([x[k] for x in lab]) for k in lab[0]}
    order = np.argsort(L["id"])
    L = {k: v[order] for k, v in L.items()}
    if not np.array_equal(L["id"], pos["id"]):
        raise SystemExit("labels and positions do not cover the same ids")
    n = L["count"][:, 0].astype(np.float64)
    if (n == 0).any():
        raise SystemExit(f"{int((n == 0).sum())} positions have no finished world")
    s = L["sums"].astype(np.float64)
    score = s[:, 0] / n
    var = np.maximum(s[:, 1] / n - score ** 2, 0.0)
    labels = {
        "score": score.astype(np.float32),
        "score_se": np.sqrt(var / np.maximum(n - 1, 1)).astype(np.float32),
        "final_vp": (s[:, 2] / n / 20.0).astype(np.float32),
        "turn_vp": (s[:, 3] / n / 20.0).astype(np.float32),
        "turn_defcon": (s[:, 4] / n).astype(np.float32),
        "swing": np.where(L["swing_n"] > 0, L["swing_sum"] / np.maximum(L["swing_n"], 1), np.nan).astype(np.float32),
        "swing_frac": (L["swing_n"] / n[:, None]).astype(np.float32),
        "worlds": L["count"][:, 0].astype(np.int16),
        "failed": L["count"][:, 1].astype(np.int16),
    }
    files: Dict[str, str] = {}
    for split, sel in (("train", ~pos["holdout"]), ("holdout", pos["holdout"])):
        f = d / f"{split}.npz"
        np.savez(f, **{k: v[sel] for k, v in pos.items()}, **{k: v[sel] for k, v in labels.items()})
        files[f.name] = sha256_file(str(f))
    sfiles = sorted(d.glob("siblings_*.npz"))
    if sfiles:
        sib = [dict(np.load(f)) for f in sfiles]
        S = {k: np.concatenate([x[k] for x in sib]) for k in sib[0]}
        hold_games = set((pos["id"][pos["holdout"]] // PLY_SPAN).tolist())
        S["holdout"] = np.isin(S["id"] // PLY_SPAN, list(hold_games))
        f = d / "siblings.npz"
        np.savez(f, id=S["id"], meta=S["meta"], cols=S["cols"], swing=S["swing"], leaf=S["leaf"],
                 holdout=S["holdout"])
        files[f.name] = sha256_file(str(f))
    code = labelling_code(labelling_signature(pmeta, lmetas[0]), update=True)
    assert code is not None
    if a.code and a.code != code:
        raise SystemExit(f"the labelling settings are {code}'s, not {a.code}'s (tools/scripts/run_codes.py labellings)")
    name = f"{pmeta['generator']}~{code}(seed={pmeta['seed']},games={pmeta['games']})"
    manifest = {"dataset": name, "code": code, "labelling": labelling_signature(pmeta, lmetas[0]),
                "generator": pmeta["generator"],
                "generator_sha256": pmeta["generator_sha256"], "engine": pmeta["engine"],
                "positions": {k: v for k, v in pmeta.items() if k not in ("part", "positions", "seconds")},
                "label": {k: v for k, v in lmetas[0].items() if k not in ("part", "network_rows", "seconds")},
                "rows": {"positions": int(pos["id"].size), "train": int((~pos["holdout"]).sum()),
                         "holdout": int(pos["holdout"].sum()),
                         "network_rows": int(sum(m["network_rows"] for m in lmetas))},
                "label_seconds": round(sum(m["seconds"] for m in lmetas), 1),
                "regions": list(REGIONS), "files": files,
                "exact": exact_reads(d, pos, labels["score"]),
                "reads": {"mean_score_se": float(labels["score_se"].mean()),
                          "failed_worlds": int(labels["failed"].sum()),
                          "holdout_games_in_train": int(np.isin(pos["id"][~pos["holdout"]] // PLY_SPAN,
                                                               list(set((pos["id"][pos["holdout"]] // PLY_SPAN).tolist()))).sum())}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["rows"]), json.dumps(manifest["reads"]))
    print(f"| `{name}` | {manifest['rows']['positions']:,} | {manifest['rows']['network_rows']:,} | "
          f"{lmetas[0]['device']} | {', '.join(f'{k} {v[:12]}' for k, v in files.items())} | |")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    po = sub.add_parser("positions", help="self-play decision positions of the generator")
    po.add_argument("--model", required=True, help="a registered network name or a .pt checkpoint")
    po.add_argument("--games", type=int, default=20000)
    po.add_argument("--temperature", type=float, default=0.25)
    po.add_argument("--keep", type=int, default=8, help="keep one decision in this many")
    po.add_argument("--candidates", type=int, default=4, help="the generator's top legal moves recorded")
    po.add_argument("--holdout", type=float, default=0.05, help="fraction of games held out")
    po.add_argument("--seed", type=int, default=44)
    po.add_argument("--envs", type=int, default=2048)
    po.add_argument("--part", type=_part, default=(0, 1))
    po.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    po.add_argument("--out", required=True)
    la = sub.add_parser("label", help="playout labels for the positions in --dir")
    la.add_argument("--dir", required=True)
    la.add_argument("--model", default=None, help="the playout network (default: the generator)")
    la.add_argument("--worlds", type=int, default=32)
    la.add_argument("--siblings", type=int, default=50000, help="positions in the sibling slice")
    la.add_argument("--sibling-worlds", type=int, default=16)
    la.add_argument("--sibling-k", type=int, default=4)
    la.add_argument("--leaf-horizon", type=int, default=0,
                    help="sibling leaves: boundaries before the mover's own decision is the leaf (0: the next one)")
    la.add_argument("--follow", choices=["greedy", "search"], default="greedy",
                    help="the playouts' player: greedy, or the keyed rollout root at entry decisions (card choices, "
                         "headlines) in the position's turn and --search-window turns after it (D3)")
    la.add_argument("--search-worlds", type=int, default=8, help="--follow search: the root's worlds")
    la.add_argument("--search-rule", default="z2", help="--follow search: the root's rule (z2, argmax, kl<t>)")
    la.add_argument("--search-window", type=int, default=0,
                    help="--follow search: turns after the position's in which entry decisions are searched")
    la.add_argument("--no-main", action="store_true",
                    help="the sibling slice only: no main labels (a slice relabelled, e.g. keyed, beside an existing set)")
    la.add_argument("--keyed", action="store_true",
                    help="keyed chance: each world's dice and deals tied to their situation, so siblings draw alike")
    la.add_argument("--keyed2", action="store_true",
                    help="keyed chance with the situation's player fields narrowed (bindings/keyed_chance.hpp), "
                         "so a USSR side does not hide the turn; implies --keyed")
    la.add_argument("--capacity", type=int, default=16384, help="rows per labeller batch")
    la.add_argument("--seed", type=int, default=44)
    la.add_argument("--part", type=_part, default=(0, 1))
    la.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    la.add_argument("--no-graphs", action="store_true", help="plain forwards instead of CUDA graphs")
    la.add_argument("--float-upload", action="store_true",
                    help="upload observations as float32 instead of their exact one-byte codes")
    se = sub.add_parser("select", help="a subset of a labelled set's positions, to label again")
    se.add_argument("--dir", required=True)
    se.add_argument("--out", required=True)
    se.add_argument("--what", choices=["contested", "exact"], default="contested")
    se.add_argument("--n", type=int, default=50000)
    se.add_argument("--lo", type=float, default=0.25, help="contested: the label's lower bound")
    se.add_argument("--hi", type=float, default=0.75, help="contested: the label's upper bound")
    se.add_argument("--siblings", default=None, help="contested: a sibling slice whose decisions with spread go first")
    se.add_argument("--spread", type=float, default=0.25, help="contested: candidate means differing by this much")
    se.add_argument("--seed", type=int, default=44)
    ex = sub.add_parser("exact", help="the exact slice: turn-10 last-round positions solved exactly (CPU)")
    ex.add_argument("--dir", required=True)
    ex.add_argument("--exact", type=int, default=2000, help="positions in the slice")
    ex.add_argument("--deals", type=int, default=4, help="the labeller's first worlds, solved each")
    ex.add_argument("--budget", type=int, default=300_000, help="search nodes per world")
    ex.add_argument("--min-round", type=int, default=7,
                    help="turn 10's action rounds from this one (from 6, most worlds exceed the budget)")
    ex.add_argument("--seed", type=int, default=44, help="the label stage's seed (its worlds)")
    ex.add_argument("--part", type=_part, default=(0, 1))
    me = sub.add_parser("merge", help="join the parts into train / holdout files and the manifest")
    me.add_argument("--dir", required=True)
    me.add_argument("--code", default=None,
                    help="the D code expected; refused if the one derived from the settings differs")
    a = ap.parse_args(argv)
    if a.stage == "positions":
        return run_positions(a)
    if a.stage == "label":
        return run_label(a)
    if a.stage == "exact":
        return run_exact(a)
    if a.stage == "select":
        return run_select(a)
    return run_merge(a)


if __name__ == "__main__":
    sys.exit(main())
