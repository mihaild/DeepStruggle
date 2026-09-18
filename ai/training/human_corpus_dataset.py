"""Behaviour-cloning data built from the ts-replayer corpus of human games.

The self-play format stores `(seed, [flat_action, ...])` and re-derives observations by driving
the engine, which keeps it forward-compatible: change the observation and the dataset follows.
Human games cannot use it. Their dice come from the log rather than from the engine's PRNG, and
their hands are solved as a constraint problem, so there is no seed that replays them -- the
conversion in `tools/lib/ts_replayer_convert.py` is what reproduces a game, and it costs about a
second per game.

So observations are stored materialised, and the cost of that is the reason for the layout here:
one memory-mapped `.npy` per column rather than a single archive. `np.load(mmap_mode="r")` then
gives bounded streaming with no monolithic read, which invariant 8 requires and which a
`.npz` cannot do. Observations are float16: the features are influence counts over ten, stability
over five, and one-hots, none of which needs more, and it halves 1.2 GB of otherwise-idle disk.

**Value targets are masked on unfinished games.** Half the corpus stops mid-game -- the recording
ends, not the game -- so those positions have no outcome. `has_outcome` is 0 there and a trainer
must exclude them from the value loss while still using them for the policy loss. Labelling them
with a guess would teach the critic results that never happened.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from bindings.action_encoder import ActionEncoder

OBS_DTYPE = np.float16
MASK_BITS = ActionEncoder.FLAT_ACTION_SIZE
_COLUMNS = ("obs", "mask", "action", "win", "vp", "has_outcome", "side", "game",
            "play")
_META = "meta.json"


def _mask_bytes() -> int:
    return (MASK_BITS + 7) // 8


class HumanCorpusWriter:
    """Accumulates converted samples and writes the column files."""

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.obs: List[np.ndarray] = []
        self.mask: List[np.ndarray] = []
        self.action: List[int] = []
        self.win: List[float] = []
        self.vp: List[float] = []
        self.has_outcome: List[int] = []
        self.side: List[int] = []
        # Which game each sample came from. Needed for a held-out split: samples inside one game
        # are heavily correlated, so splitting on samples measures memorisation of positions the
        # model has effectively already seen. A split has to be by game.
        self.game: List[int] = []
        # Which play. A card spends its Operations one point at a time and the engine asks once
        # per point, but the order those points were written in carries no decision -- so
        # agreement over a play should be scored on the multiset of countries, not the sequence
        # (ai/eval/agreement). Recorded here so that can be done without re-running conversion,
        # which is the expensive part.
        self.play: List[int] = []
        self._plays = 0
        self.games = 0
        self.games_with_outcome = 0

    def add_game(self, samples: List[Tuple[Any, Any, int, int]],
                 us_utility: Optional[float], final_vp: Optional[int],
                 plays: Optional[List[List[int]]] = None) -> None:
        """Add one converted game.

        `samples` is `Conversion.samples`: (observation, mask, action, side) with side +1 for the
        US and -1 for the USSR. `us_utility` is None when the game never reached a terminal
        state, which is what marks its positions as having no value target.
        """
        self.games += 1
        settled = us_utility is not None
        if settled:
            self.games_with_outcome += 1
        utility = 0.0 if us_utility is None else float(us_utility)
        score = 0.0 if final_vp is None else float(final_vp)
        # One id per play, or one per sample where the caller did not group them.
        play_of: Dict[int, int] = {}
        if plays:
            for group in plays:
                self._plays += 1
                for idx in group:
                    play_of[idx] = self._plays
        for position, (obs, mask, action, side) in enumerate(samples):
            arr = np.asarray(obs, dtype=OBS_DTYPE)
            packed = np.packbits(np.asarray(mask, dtype=np.uint8)[:MASK_BITS])
            self.obs.append(arr)
            self.mask.append(packed)
            self.action.append(int(action))
            self.side.append(int(side))
            # Perspective-aligned: a sample is labelled from the mover's point of view, so the
            # US utility flips for a USSR decision.
            self.win.append(utility * int(side) if settled else 0.0)
            self.vp.append((score / 20.0) * int(side) if settled else 0.0)
            self.has_outcome.append(1 if settled else 0)
            self.game.append(self.games - 1)
            if position in play_of:
                self.play.append(play_of[position])
            else:
                self._plays += 1
                self.play.append(self._plays)

    def write(self) -> Dict[str, Any]:
        os.makedirs(self.out_dir, exist_ok=True)
        if not self.obs:
            raise ValueError("nothing to write: no samples were added")

        arrays: Dict[str, np.ndarray] = {
            "obs": np.stack(self.obs).astype(OBS_DTYPE),
            "mask": np.stack(self.mask).astype(np.uint8),
            "action": np.asarray(self.action, dtype=np.int16),
            "win": np.asarray(self.win, dtype=np.float16),
            "vp": np.asarray(self.vp, dtype=np.float16),
            "has_outcome": np.asarray(self.has_outcome, dtype=np.uint8),
            "side": np.asarray(self.side, dtype=np.int8),
            "game": np.asarray(self.game, dtype=np.int32),
            "play": np.asarray(self.play, dtype=np.int32),
        }
        for name, arr in arrays.items():
            np.save(os.path.join(self.out_dir, f"{name}.npy"), arr)

        meta: Dict[str, Any] = {
            "samples": int(arrays["obs"].shape[0]),
            "obs_dim": int(arrays["obs"].shape[1]),
            "mask_bits": MASK_BITS,
            "games": self.games,
            "games_with_outcome": self.games_with_outcome,
            "samples_with_outcome": int(arrays["has_outcome"].sum()),
        }
        with open(os.path.join(self.out_dir, _META), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return meta


class HumanCorpusDataset:
    """Streams the written columns, mirroring `WarmupDataset`'s interface.

    Memory-mapped, so peak memory is a batch rather than the dataset.
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory
        meta_path = os.path.join(directory, _META)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{directory} is not a human-corpus dataset ({_META} missing). "
                f"Build one with tools/build_human_dataset.py")
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta: Dict[str, Any] = json.load(fh)

    def _column(self, name: str) -> np.ndarray:
        return np.load(os.path.join(self.directory, f"{name}.npy"), mmap_mode="r")

    def __len__(self) -> int:
        return int(self.meta["samples"])

    def stream_transitions(
        self, max_samples: Optional[int] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, int, float, float, int]]:
        """(observation, mask, action, win_target, vp_target, has_outcome) one at a time."""
        obs, mask = self._column("obs"), self._column("mask")
        action, win, vp = self._column("action"), self._column("win"), self._column("vp")
        has_outcome = self._column("has_outcome")
        n = len(self) if max_samples is None else min(len(self), max_samples)
        for i in range(n):
            unpacked = np.unpackbits(np.asarray(mask[i]))[:MASK_BITS]
            yield (np.asarray(obs[i], dtype=np.float32), unpacked,
                   int(action[i]), float(win[i]), float(vp[i]), int(has_outcome[i]))

    def stream_batches(
        self,
        batch_size: int = 512,
        device: torch.device = torch.device("cpu"),
        shuffle: bool = True,
        seed: int = 2026,
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        torch.Tensor, torch.Tensor]]:
        """Mini-batches of (obs, mask, action, win, vp, has_outcome).

        `has_outcome` rides along so the caller can zero the value loss on positions from games
        whose recording stopped; dropping them from the policy loss too would throw away half
        the corpus for no reason.
        """
        obs, mask = self._column("obs"), self._column("mask")
        action, win, vp = self._column("action"), self._column("win"), self._column("vp")
        has_outcome = self._column("has_outcome")

        n = len(self) if max_samples is None else min(len(self), max_samples)
        order = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)

        for start in range(0, n - batch_size + 1, batch_size):
            idx = np.sort(order[start:start + batch_size])
            b_mask = np.unpackbits(np.asarray(mask[idx]), axis=1)[:, :MASK_BITS]
            yield (
                torch.from_numpy(np.asarray(obs[idx], dtype=np.float32)).to(device),
                torch.from_numpy(b_mask.astype(np.uint8)).to(device),
                torch.from_numpy(np.asarray(action[idx], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(win[idx], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(vp[idx], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(has_outcome[idx], dtype=np.float32)).to(device),
            )
