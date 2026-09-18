"""Action Encoder for Twilight Struggle Flat 212-dimensional MicroAction Space."""

from typing import List, Optional
import numpy as np
import ts_engine as ts


class ActionEncoder:
    """Encodes and decodes between discrete flat action index [0..211] and ts.MicroAction.

    Action Vocabulary Breakdown (N = 212):
      [0..109]   (110): SELECT_CARD (Card IDs 1..110, index = card_id - 1)
      [110..114] (5)  : SELECT_PLAY_MODE -- the merged resolution node
                        (0: EVENT, 1: SPACE, 2: OPS_INFLUENCE, 3: OPS_COUP, 4: OPS_REALIGN)
      [112..114]      : SELECT_OP_MODE (0: INFLUENCE, 1: COUP, 2: REALIGN) -- the *same* three
                        slots, reused for the deferred Ops choice that follows an event-first
                        event. "Spend Ops on X" means the same thing at either node, so they
                        share an index and the decision type says which one is being asked.
      [115]      (1)  : ROLL_DIE
      [116..118] (3)  : unassigned after the P17 merge -- decoding one is refused
      [119..202] (84) : POINT_NODE (Country IDs 0..83, index = 119 + country_id)
      [203..210] (8)  : CHOOSE_BRANCH (Branches 0..7, index = 203 + branch_id)
      [211]      (1)  : CONFIRM_DONE / PASS (0x80 / early stop)

    CHOOSE_TIMING_BRANCH is retired: on an opponent card EVENT is the event-first branch and
    any OPS_* is the ops-first branch, so the timing is carried by the resolution itself.
    """

    FLAT_ACTION_SIZE = 220

    CARD_OFFSET = 0
    PLAY_MODE_OFFSET = 110
    # The deferred Ops choice lands on the three OPS_* resolution slots, not on a block of its
    # own: OP_MODE_OFFSET + OpMode.INFLUENCE == PLAY_MODE_OFFSET + Resolution.OPS_INFLUENCE.
    OP_MODE_OFFSET = 112
    ROLL_DIE_INDEX = 115
    NODE_OFFSET = 116
    BRANCH_OFFSET = 200
    CONFIRM_DONE_INDEX = 208
    #: P17 section 4. Two heads that name concepts the game already has, rather than generic
    #: branch slots: "set DEFCON to V" (Summit, How I Learned) and a region (Chernobyl).
    DEFCON_VALUE_OFFSET = 209
    REGION_OFFSET = 214

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

        if action_idx < ActionEncoder.ROLL_DIE_INDEX:
            mode_id = action_idx - ActionEncoder.PLAY_MODE_OFFSET
            mode_names = ["EVENT", "SPACE", "OPS_INFLUENCE", "OPS_COUP", "OPS_REALIGN"]
            # [112..114] are also the deferred Ops choice. Without the state there is no way to
            # tell which node is being named, so say both rather than pick one and be wrong.
            name = mode_names[mode_id] if mode_id < 5 else str(mode_id)
            if action_idx >= ActionEncoder.OP_MODE_OFFSET:
                op_names = ["INFLUENCE", "COUP", "REALIGN"]
                op = op_names[action_idx - ActionEncoder.OP_MODE_OFFSET]
                return f"Resolution: {name} / OpMode: {op}"
            return f"Resolution: {name}"

        if action_idx == ActionEncoder.ROLL_DIE_INDEX:
            return "RollDie"

        if action_idx < ActionEncoder.NODE_OFFSET:
            return f"Unassigned #{action_idx}"

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
