"""Action encoder for the flat MicroAction space (220 wide since P17; "212" is a historical name)."""

from typing import List
import numpy as np
import ts_engine as ts


#: ts::Region order (engine/include/ts/types.hpp), the order of the REGION_OFFSET slots.
REGION_NAMES = ("Europe", "Asia", "Middle East", "Africa", "Central America", "South America")


class ActionEncoder:
    """Encodes and decodes between a flat action index and ts.MicroAction.

    The slots are the engine's (ts::flat_slots, read here from the ts.FLAT_* exports):
      CARD_OFFSET          110 wide: SELECT_CARD, index = card_id - 1
      PLAY_MODE_OFFSET       5 wide: SELECT_PLAY_MODE, the merged resolution node
                                     (0 EVENT, 1 SPACE, 2 OPS_INFLUENCE, 3 OPS_COUP, 4 OPS_REALIGN)
      OP_MODE_OFFSET         3 wide: SELECT_OP_MODE (0 INFLUENCE, 1 COUP, 2 REALIGN) -- the *same*
                                     slots as the three OPS_* resolutions: "spend Ops on X" means the
                                     same at either node, and the decision type says which is asked
      ROLL_DIE_INDEX         1:      ROLL_DIE
      NODE_OFFSET           84 wide: POINT_NODE, index = NODE_OFFSET + country_id
      BRANCH_OFFSET          8 wide: CHOOSE_BRANCH, index = BRANCH_OFFSET + branch_id
      CONFIRM_DONE_INDEX     1:      CONFIRM_DONE / PASS
      DEFCON_VALUE_OFFSET    5 wide: "set DEFCON to V" (Summit, How I Learned)
      REGION_OFFSET          6 wide: a region (Chernobyl)

    CHOOSE_TIMING_BRANCH is retired: on an opponent card EVENT is the event-first branch and
    any OPS_* is the ops-first branch, so the timing is carried by the resolution itself.
    """

    FLAT_ACTION_SIZE: int = ts.FLAT_ACTION_SPACE_SIZE
    #: The base slots, the same in every view (0..219): a base model's output width and a stored mask's bits.
    BASE_ACTION_SIZE: int = ts.FLAT_BASE_SIZE
    #: P42's until-control slots, the macro view only: MACRO_NODE_OFFSET + country_id.
    MACRO_NODE_OFFSET: int = ts.FLAT_MACRO_NODE

    CARD_OFFSET: int = ts.FLAT_CARD
    PLAY_MODE_OFFSET: int = ts.FLAT_RESOLUTION
    # The deferred Ops choice lands on the three OPS_* resolution slots, not on a block of its
    # own: OP_MODE_OFFSET + OpMode.INFLUENCE == PLAY_MODE_OFFSET + Resolution.OPS_INFLUENCE.
    OP_MODE_OFFSET: int = ts.FLAT_OP_MODE
    ROLL_DIE_INDEX: int = ts.FLAT_ROLL_DIE
    NODE_OFFSET: int = ts.FLAT_NODE
    BRANCH_OFFSET: int = ts.FLAT_BRANCH
    CONFIRM_DONE_INDEX: int = ts.FLAT_CONFIRM_DONE
    #: P17 section 4. Two heads that name concepts the game already has, rather than generic
    #: branch slots: "set DEFCON to V" (Summit, How I Learned) and a region (Chernobyl).
    DEFCON_VALUE_OFFSET: int = ts.FLAT_DEFCON_VALUE
    REGION_OFFSET: int = ts.FLAT_REGION

    @staticmethod
    def encode(state: ts.GameState, action: ts.MicroAction) -> int:
        """The flat index of a MicroAction; ValueError for one no slot encodes."""
        return int(ts.encode_micro_action(state, action))

    @staticmethod
    def decode(state: ts.GameState, action_idx: int) -> ts.MicroAction:
        """The MicroAction a flat index stands for; ValueError outside the space."""
        if action_idx < 0 or action_idx >= ActionEncoder.FLAT_ACTION_SIZE:
            raise ValueError(f"Action index {action_idx} out of range [0..{ActionEncoder.FLAT_ACTION_SIZE-1}]")
        return ts.decode_flat_action(state, action_idx)

    @staticmethod
    def get_legal_mask(state: ts.GameState, merged_influence: bool = False) -> np.ndarray:
        """Return binary legal action mask shape (220,) with dtype uint8.

        `merged_influence` selects the E4.1 view (P23), in which a NODE slot at an op-choice node
        means "ops for influence, first point here"; the default is the E4 mask.
        """
        return ts.get_flat_action_mask(state, merged_influence)

    @staticmethod
    def get_legal_indices(state: ts.GameState, merged_influence: bool = False) -> List[int]:
        """Return list of legal action indices."""
        mask = ts.get_flat_action_mask(state, merged_influence)
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

        if ActionEncoder.DEFCON_VALUE_OFFSET <= action_idx < ActionEncoder.REGION_OFFSET:
            # Summit / How I Learned to Stop Worrying: set DEFCON to 1..5.
            return f"SetDEFCON {action_idx - ActionEncoder.DEFCON_VALUE_OFFSET + 1}"

        if action_idx < ActionEncoder.BASE_ACTION_SIZE:
            # Chernobyl's region, in ts::Region order.
            region_id = action_idx - ActionEncoder.REGION_OFFSET
            name = REGION_NAMES[region_id] if region_id < len(REGION_NAMES) else "?"
            return f"Region #{region_id} ({name})"

        if action_idx < ActionEncoder.FLAT_ACTION_SIZE:
            # P42's until-control slots (the macro view only).
            country_id = action_idx - ActionEncoder.MACRO_NODE_OFFSET
            try:
                return f"UntilControl #{country_id} ({ts.MapData.get_country_name(country_id)})"
            except Exception:
                return f"UntilControl #{country_id}"

        return f"UnknownAction #{action_idx}"
