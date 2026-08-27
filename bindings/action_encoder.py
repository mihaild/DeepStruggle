"""Action Encoder for Twilight Struggle Flat 212-dimensional MicroAction Space."""

from typing import List, Optional
import numpy as np
import ts_engine as ts


class ActionEncoder:
    """Encodes and decodes between discrete flat action index [0..211] and ts.MicroAction.

    Action Vocabulary Breakdown (N = 212):
      [0..109]   (110): SELECT_CARD (Card IDs 1..110, index = card_id - 1)
      [110..113] (4)  : SELECT_PLAY_MODE (0: EVENT, 1: OPS, 2: SPACE, 3: PASS)
      [114..115] (2)  : CHOOSE_TIMING_BRANCH (0: OPS_FIRST, 1: EVENT_FIRST)
      [116..118] (3)  : SELECT_OP_MODE (0: INFLUENCE, 1: COUP, 2: REALIGN)
      [119..202] (84) : POINT_NODE (Country IDs 0..83, index = 119 + country_id)
      [203..210] (8)  : CHOOSE_BRANCH (Branches 0..7, index = 203 + branch_id)
      [211]      (1)  : CONFIRM_DONE / PASS (0x80 / early stop)
    """

    FLAT_ACTION_SIZE = 212

    CARD_OFFSET = 0
    PLAY_MODE_OFFSET = 110
    TIMING_OFFSET = 114
    OP_MODE_OFFSET = 116
    NODE_OFFSET = 119
    BRANCH_OFFSET = 203
    CONFIRM_DONE_INDEX = 211

    @staticmethod
    def encode(state: ts.GameState, action: ts.MicroAction) -> int:
        """Encode ts.MicroAction to flat index [0..211]."""
        return int(ts.encode_micro_action(state, action))

    @staticmethod
    def decode(state: ts.GameState, action_idx: int) -> ts.MicroAction:
        """Decode flat index [0..211] to ts.MicroAction."""
        if action_idx < 0 or action_idx >= ActionEncoder.FLAT_ACTION_SIZE:
            raise ValueError(f"Action index {action_idx} out of range [0..{ActionEncoder.FLAT_ACTION_SIZE-1}]")
        return ts.decode_flat_action(state, action_idx)

    @staticmethod
    def get_legal_mask(state: ts.GameState) -> np.ndarray:
        """Return binary legal action mask shape (212,) with dtype uint8."""
        return ts.get_flat_action_mask(state)

    @staticmethod
    def get_legal_indices(state: ts.GameState) -> List[int]:
        """Return list of legal action indices."""
        mask = ts.get_flat_action_mask(state)
        return [int(i) for i in np.where(mask > 0)[0]]

    @staticmethod
    def get_action_name(state: ts.GameState, action_idx: int) -> str:
        """Return human-readable descriptive string for the action index."""
        if action_idx == ActionEncoder.CONFIRM_DONE_INDEX:
            return "CONFIRM_DONE / PASS"

        if action_idx < ActionEncoder.PLAY_MODE_OFFSET:
            card_id = action_idx + 1
            try:
                c_info = ts.CardData.get_card_info(card_id)
                return f"SelectCard #{card_id} ({c_info['name']})"
            except Exception:
                return f"SelectCard #{card_id}"

        if action_idx < ActionEncoder.TIMING_OFFSET:
            mode_id = action_idx - ActionEncoder.PLAY_MODE_OFFSET
            mode_names = ["EVENT", "OPS", "SPACE", "PASS"]
            return f"PlayMode: {mode_names[mode_id] if mode_id < 4 else mode_id}"

        if action_idx < ActionEncoder.OP_MODE_OFFSET:
            timing_id = action_idx - ActionEncoder.TIMING_OFFSET
            timing_names = ["OPS_FIRST", "EVENT_FIRST"]
            return f"Timing: {timing_names[timing_id] if timing_id < 2 else timing_id}"

        if action_idx < ActionEncoder.NODE_OFFSET:
            op_id = action_idx - ActionEncoder.OP_MODE_OFFSET
            op_names = ["INFLUENCE", "COUP", "REALIGN"]
            return f"OpMode: {op_names[op_id] if op_id < 3 else op_id}"

        if action_idx < ActionEncoder.BRANCH_OFFSET:
            country_id = action_idx - ActionEncoder.NODE_OFFSET
            try:
                name = ts.MapData.get_country_name(country_id)
                return f"PointNode #{country_id} ({name})"
            except Exception:
                return f"PointNode #{country_id}"

        if action_idx < ActionEncoder.CONFIRM_DONE_INDEX:
            branch_id = action_idx - ActionEncoder.BRANCH_OFFSET
            return f"Branch #{branch_id}"

        return f"UnknownAction #{action_idx}"
