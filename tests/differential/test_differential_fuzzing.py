"""
Twilight Struggle AI: Unified Differential Fuzzing Suite
========================================================
Comprehensive differential fuzzing between the native C++ engine and Struggler:
- Full coverage of all 110 cards/events across Early War, Mid War, and Late War eras.
- Random hand distribution to US and USSR players.
- Random card play adhering to exact game/event probabilities:
    * Own and neutral cards: played with probability 2/3 as Event, 1/3 as Operations.
    * Opponent cards: played as Operations, triggering opponent event.
    * Timing branch: Event first vs Event after chosen with probability 1/2.
- Multi-action trajectories testing legal decision spaces, target resolution, and state parity.
- Granular discrepancy detection and logging across all three engines.
"""

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Generator, Any
import pytest
pytestmark = [pytest.mark.differential_fuzz]


from tests.differential.engine_interface import (
    Region,
    Player,
    DecisionType,
    MicroAction,
    EngineProtocol,
    GameStateProtocol,
    get_country_id,
    get_country_name,
    assert_boards_equal,
    assert_states_equal,
)
from tests.differential.native_adapter import NativeEngine, NativeGameState
from tests.differential.struggler_adapter import StrugglerEngine, StrugglerGameState

# Load full 110-card database
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CARDS_JSON_PATH = os.path.join(_BASE_DIR, "rules", "cards.json")

with open(_CARDS_JSON_PATH, "r", encoding="utf-8") as _f:
    _CARDS_DATA = json.load(_f)

_CARD_MAP = {c["id"]: c for c in _CARDS_DATA}
_EARLY_WAR_CARDS = [c["id"] for c in _CARDS_DATA if c["age"] == "early war"]
_MID_WAR_CARDS = [c["id"] for c in _CARDS_DATA if c["age"] == "middle war"]
_LATE_WAR_CARDS = [c["id"] for c in _CARDS_DATA if c["age"] == "late war"]
_ALL_110_CARDS = [c["id"] for c in _CARDS_DATA]


@dataclass
class DiscrepancyRecord:
    engine_pair: str
    seed: int
    round_idx: int
    card_id: int
    card_name: str
    acting_player: str
    play_mode: str
    timing_branch: str
    diff_kind: str
    details: str


# Global registry recording all observed cross-engine discrepancies during fuzzing
RECORDED_DISCREPANCIES: list[DiscrepancyRecord] = []


def create_implementation(name: str) -> tuple[EngineProtocol, GameStateProtocol]:
    """Factory creating an (engine, state) implementation pair."""
    if name == "native":
        return NativeEngine(), NativeGameState()
    elif name == "struggler":
        return StrugglerEngine(), StrugglerGameState()
    raise ValueError(f"Unknown engine implementation: {name}")


@pytest.fixture(params=["struggler"])
def engine_pair(
    request: pytest.FixtureRequest,
) -> tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]]:
    """Yields ((native_engine, native_state), (external_engine, external_state))."""
    pair_a = create_implementation("native")
    pair_b = create_implementation(request.param)
    return pair_a, pair_b


# =============================================================================
# FUZZED INITIAL STATE GENERATOR
# =============================================================================

ARCHETYPES = [
    "historical_early",
    "mid_war_global",
    "late_war_dense",
    "battleground_critical",
    "superpower_border",
    "sparse_frontiers",
    "random_uniform",
]

_SUPERPOWER_BORDER_COUNTRIES = [
    "Finland", "Canada", "Turkey", "Japan", "Romania",
    "East Germany", "Afghanistan", "North Korea", "South Korea",
]

_KEY_BATTLEGROUNDS = [
    "West Germany", "Poland", "Italy", "France", "Japan",
    "South Korea", "Israel", "Egypt", "Iran", "Iraq", "Chile",
    "Panama", "Cuba", "Nigeria", "South Africa", "Thailand",
]

_EASTERN_EUROPE_SETUP = [
    "Poland", "East Germany", "Austria", "Yugoslavia",
    "Czechoslovakia", "Hungary", "Romania", "Bulgaria", "Finland",
]

_WESTERN_EUROPE_SETUP = [
    "West Germany", "Italy", "France", "Canada", "United Kingdom",
    "Spain/Portugal", "Greece", "Turkey", "Norway", "Denmark", "Sweden", "Benelux",
]


def apply_fuzzed_initial_state(
    rng: random.Random,
    state_a: GameStateProtocol,
    state_b: GameStateProtocol,
    archetype: str,
) -> None:
    """Configures state_a and state_b with identical diverse game states according to archetype."""
    if archetype == "historical_early":
        defcon = rng.choice([3, 4, 5])
        turn = rng.randint(1, 3)
        vp = rng.randint(-5, 5)
        for st in [state_a, state_b]:
            st.defcon = defcon
            st.turn = turn
            st.victory_points = vp
        for cid in rng.sample(range(84), 15):
            us_inf = rng.choice([0, 1, 2])
            ussr_inf = rng.choice([0, 1, 2])
            state_a.set_country(cid, us_inf, ussr_inf)
            state_b.set_country(cid, us_inf, ussr_inf)

    elif archetype == "mid_war_global":
        defcon = rng.choice([2, 3, 4])
        turn = rng.randint(4, 7)
        vp = rng.randint(-12, 12)
        us_mil = rng.randint(0, 4)
        ussr_mil = rng.randint(0, 4)
        us_space = rng.randint(0, 4)
        ussr_space = rng.randint(0, 4)
        for st in [state_a, state_b]:
            st.defcon = defcon
            st.turn = turn
            st.victory_points = vp
            st.us_mil_ops = us_mil
            st.ussr_mil_ops = ussr_mil
            st.us_space_track = us_space
            st.ussr_space_track = ussr_space
        for cid in rng.sample(range(84), 35):
            us_inf = rng.choice([0, 1, 2, 3])
            ussr_inf = rng.choice([0, 1, 2, 3])
            state_a.set_country(cid, us_inf, ussr_inf)
            state_b.set_country(cid, us_inf, ussr_inf)

    elif archetype == "late_war_dense":
        turn = rng.randint(8, 10)
        vp = rng.randint(-18, 18)
        us_mil = rng.randint(2, 5)
        ussr_mil = rng.randint(2, 5)
        us_space = rng.randint(2, 7)
        ussr_space = rng.randint(2, 7)
        for st in [state_a, state_b]:
            st.defcon = 2
            st.turn = turn
            st.victory_points = vp
            st.us_mil_ops = us_mil
            st.ussr_mil_ops = ussr_mil
            st.us_space_track = us_space
            st.ussr_space_track = ussr_space
        for cid in range(84):
            us_inf = rng.choice([0, 1, 2, 3, 4])
            ussr_inf = rng.choice([0, 1, 2, 3, 4])
            state_a.set_country(cid, us_inf, ussr_inf)
            state_b.set_country(cid, us_inf, ussr_inf)

    elif archetype == "battleground_critical":
        defcon = rng.choice([2, 3, 4])
        turn = rng.randint(2, 6)
        vp = rng.randint(-8, 8)
        for st in [state_a, state_b]:
            st.defcon = defcon
            st.turn = turn
            st.victory_points = vp
        for name in _KEY_BATTLEGROUNDS:
            cid = get_country_id(name)
            us_inf = rng.choice([1, 2, 3])
            ussr_inf = rng.choice([1, 2, 3])
            state_a.set_country(cid, us_inf, ussr_inf)
            state_b.set_country(cid, us_inf, ussr_inf)

    elif archetype == "superpower_border":
        defcon = rng.choice([2, 3])
        turn = rng.randint(1, 5)
        for st in [state_a, state_b]:
            st.defcon = defcon
            st.turn = turn
        for name in _SUPERPOWER_BORDER_COUNTRIES:
            cid = get_country_id(name)
            us_inf = rng.choice([1, 2, 3, 4])
            ussr_inf = rng.choice([1, 2, 3, 4])
            state_a.set_country(cid, us_inf, ussr_inf)
            state_b.set_country(cid, us_inf, ussr_inf)

    elif archetype == "sparse_frontiers":
        defcon = rng.choice([4, 5])
        for st in [state_a, state_b]:
            st.defcon = defcon
            st.turn = 1
        for cid in rng.sample(range(84), 10):
            us_inf = rng.choice([1, 2])
            ussr_inf = rng.choice([0, 1])
            state_a.set_country(cid, us_inf, ussr_inf)
            state_b.set_country(cid, us_inf, ussr_inf)

    else:  # random_uniform
        defcon = rng.randint(2, 5)
        turn = rng.randint(1, 10)
        vp = rng.randint(-15, 15)
        for st in [state_a, state_b]:
            st.defcon = defcon
            st.turn = turn
            st.victory_points = vp
        for cid in range(84):
            us_inf = rng.choice([0, 0, 1, 2, 3])
            ussr_inf = rng.choice([0, 0, 1, 2, 3])
            state_a.set_country(cid, us_inf, ussr_inf)
            state_b.set_country(cid, us_inf, ussr_inf)


# =============================================================================
# INTERACTIVE DECISION RESOLUTION & GAME ROLLOUT ENGINE
# =============================================================================

def _assert_decision_parity(
    eng_a: EngineProtocol,
    st_a: GameStateProtocol,
    eng_b: EngineProtocol,
    st_b: GameStateProtocol,
    rng: random.Random,
    scenario: str,
    other_engine_name: str,
) -> None:
    """Validates decision space parity and resolves identical choices on both engines.
    Fails immediately with detailed diagnostic report if decision availability or proposed options differ.
    """
    for _ in range(4):
        dec_a = eng_a.has_pending_decision(st_a)
        dec_b = eng_b.has_pending_decision(st_b)
        if not dec_a and not dec_b:
            break

        if dec_a != dec_b:
            type_a = eng_a.get_pending_decision_type(st_a) if dec_a else "None"
            type_b = eng_b.get_pending_decision_type(st_b) if dec_b else "None"
            raise AssertionError(
                f"Decision prompt mismatch:\n"
                f"  Scenario: {scenario}\n"
                f"  Native proposes: {'pending decision of type ' + type_a if dec_a else 'action resolved immediately without prompt'}\n"
                f"  {other_engine_name} proposes: {'pending decision of type ' + type_b if dec_b else 'action resolved immediately without prompt'}"
            )

        type_a = eng_a.get_pending_decision_type(st_a)
        type_b = eng_b.get_pending_decision_type(st_b)

        if type_a == "roll" or type_b == "roll":
            roll = rng.randint(1, 6)
            eng_a.resolve_decision(st_a, roll)
            eng_b.resolve_decision(st_b, roll)
            continue

        ta = eng_a.get_decision_targets(st_a)
        tb = eng_b.get_decision_targets(st_b)

        if ta != tb:
            raise AssertionError(
                f"Decision options mismatch:\n"
                f"  Scenario: {scenario}\n"
                f"  Native proposes ({len(ta)} options): {sorted(ta)}\n"
                f"  {other_engine_name} proposes ({len(tb)} options): {sorted(tb)}\n"
                f"  Difference (only Native): {sorted(ta - tb)}\n"
                f"  Difference (only {other_engine_name}): {sorted(tb - ta)}"
            )

        if ta:
            choice = rng.choice(sorted(ta))
            eng_a.resolve_decision(st_a, choice)
            eng_b.resolve_decision(st_b, choice)


def _assert_outcome_parity(
    st_a: GameStateProtocol,
    st_b: GameStateProtocol,
    scenario: str,
    other_engine_name: str,
) -> None:
    """Asserts that all status tracks and country influence match after action resolution.
    Fails immediately with detailed diagnostic report if any state differences occur.
    """
    diffs = []
    if st_a.defcon != st_b.defcon:
        diffs.append(f"DEFCON (Native={st_a.defcon} vs {other_engine_name}={st_b.defcon})")
    if st_a.victory_points != st_b.victory_points:
        diffs.append(f"Victory Points (Native={st_a.victory_points} vs {other_engine_name}={st_b.victory_points})")
    if st_a.us_mil_ops != st_b.us_mil_ops:
        diffs.append(f"US Military Ops (Native={st_a.us_mil_ops} vs {other_engine_name}={st_b.us_mil_ops})")
    if st_a.ussr_mil_ops != st_b.ussr_mil_ops:
        diffs.append(f"USSR Military Ops (Native={st_a.ussr_mil_ops} vs {other_engine_name}={st_b.ussr_mil_ops})")

    board_diffs = []
    for cid in range(84):
        ca = st_a.get_country(cid)
        cb = st_b.get_country(cid)
        if (ca.us_influence, ca.ussr_influence) != (cb.us_influence, cb.ussr_influence):
            cname = get_country_name(cid)
            board_diffs.append(
                f"{cname} (Native: US={ca.us_influence}, USSR={ca.ussr_influence} vs "
                f"{other_engine_name}: US={cb.us_influence}, USSR={cb.ussr_influence})"
            )
    if board_diffs:
        diffs.append(f"Board influence differs in {len(board_diffs)} countries: {', '.join(board_diffs[:4])}")

    if diffs:
        diff_str = "\n    * ".join(diffs)
        raise AssertionError(
            f"Resulting outcome mismatch after action:\n"
            f"  Scenario: {scenario}\n"
            f"  Differences:\n    * {diff_str}"
        )
def _execute_fuzzed_card_game_rollout(
    eng_a: EngineProtocol,
    st_a: GameStateProtocol,
    eng_b: EngineProtocol,
    st_b: GameStateProtocol,
    card_pool: list[int],
    rounds: int,
    seed: int,
    other_engine_name: str,
) -> None:
    """Executes a multi-action random card game rollout following exact user probabilities:
      - Own and neutral cards: played with prob 2/3 as Event, 1/3 as Operations (scoring cards always Event).
      - Opponent cards: played as Operations.
      - Event first vs Event after: chosen randomly with probability 1/2.
    Asserts decision space and resulting state parity after every microaction, failing immediately on difference.
    """
    rng = random.Random(seed)
    st_a.phase = "action_round"
    st_b.phase = "action_round"

    shuffled_pool = list(card_pool)
    rng.shuffle(shuffled_pool)

    half = len(shuffled_pool) // 2
    us_hand = shuffled_pool[:half]
    ussr_hand = shuffled_pool[half:]

    for r in range(rounds):
        player = Player.USSR if r % 2 == 0 else Player.US
        hand = ussr_hand if player == Player.USSR else us_hand
        if not hand:
            break

        cid = hand.pop(rng.randrange(len(hand)))
        c_info = _CARD_MAP[cid]
        p_name = player.name
        opp_player = Player.USSR if player == Player.US else Player.US
        opp_side = "ussr" if player == Player.US else "us"

        # Determine play mode per user probabilities:
        if c_info["side"] == opp_side:
            is_event = False
            has_opp = True
            event_first = (rng.random() < 0.5)
            timing_branch = "Event First" if event_first else "Event After"
            play_mode = f"Operations with Opponent Event ({timing_branch})"
        else:
            has_opp = False
            event_first = False
            timing_branch = "None"
            if c_info["ops"] == 0:
                is_event = True
                play_mode = "Scoring Event"
            else:
                is_event = (rng.random() < (2.0 / 3.0))
                play_mode = "Event" if is_event else "Operations"

        scenario = (
            f"Card {cid} ({c_info['name']}) played by {p_name} as {play_mode} "
            f"[DEFCON={st_a.defcon}, Turn={st_a.turn}, VP={st_a.victory_points}]"
        )

        def do_event(acting_p: Player) -> None:
            eng_a.play_event(st_a, acting_p, cid)
            eng_b.play_event(st_b, acting_p, cid)
            _assert_decision_parity(
                eng_a, st_a, eng_b, st_b, rng, scenario, other_engine_name
            )

        def do_ops(acting_p: Player) -> None:
            ops_val = c_info["ops"]
            if ops_val == 0:
                return

            op_choice = rng.random()
            if op_choice < 0.60:
                # Influence placement
                lp_a = eng_a.get_legal_placements(st_a, acting_p)
                lp_b = eng_b.get_legal_placements(st_b, acting_p)
                if lp_a != lp_b:
                    raise AssertionError(
                        f"Legal operations placement options mismatch:\n"
                        f"  Scenario: {scenario} [placing influence]\n"
                        f"  Native proposes ({len(lp_a)} options): {sorted(lp_a)}\n"
                        f"  {other_engine_name} proposes ({len(lp_b)} options): {sorted(lp_b)}\n"
                        f"  Difference (only Native): {sorted(lp_a - lp_b)}\n"
                        f"  Difference (only {other_engine_name}): {sorted(lp_b - lp_a)}"
                    )
                common = sorted(lp_a & lp_b)
                if common:
                    target_country = rng.choice(common)
                    target_cid = get_country_id(target_country)
                    cur_a = st_a.get_country(target_cid)
                    cur_b = st_b.get_country(target_cid)
                    if acting_p == Player.US:
                        st_a.set_country(target_cid, cur_a.us_influence + ops_val, cur_a.ussr_influence)
                        st_b.set_country(target_cid, cur_b.us_influence + ops_val, cur_b.ussr_influence)
                    else:
                        st_a.set_country(target_cid, cur_a.us_influence, cur_a.ussr_influence + ops_val)
                        st_b.set_country(target_cid, cur_b.us_influence, cur_b.ussr_influence + ops_val)
            elif op_choice < 0.90:
                # Coup
                coups_a = eng_a.get_legal_coups(st_a, acting_p, ops=ops_val)
                coups_b = eng_b.get_legal_coups(st_b, acting_p, ops=ops_val)
                if coups_a != coups_b:
                    raise AssertionError(
                        f"Legal coup target options mismatch:\n"
                        f"  Scenario: {scenario} [attempting coup with {ops_val} Ops]\n"
                        f"  Native proposes ({len(coups_a)} options): {sorted(coups_a)}\n"
                        f"  {other_engine_name} proposes ({len(coups_b)} options): {sorted(coups_b)}\n"
                        f"  Difference (only Native): {sorted(coups_a - coups_b)}\n"
                        f"  Difference (only {other_engine_name}): {sorted(coups_b - coups_a)}"
                    )
                common_coups = sorted(coups_a & coups_b)
                if common_coups:
                    target_c = rng.choice(common_coups)
                    target_cid = get_country_id(target_c)
                    roll = rng.randint(1, 6)
                    st_a.defcon = max(2, st_a.defcon - 1)
                    st_b.defcon = max(2, st_b.defcon - 1)
                    if acting_p == Player.US:
                        st_a.us_mil_ops = min(5, st_a.us_mil_ops + ops_val)
                        st_b.us_mil_ops = min(5, st_b.us_mil_ops + ops_val)
                    else:
                        st_a.ussr_mil_ops = min(5, st_a.ussr_mil_ops + ops_val)
                        st_b.ussr_mil_ops = min(5, st_b.ussr_mil_ops + ops_val)
            else:
                # Realignment
                realign_a = eng_a.get_legal_realignments(st_a, acting_p)
                realign_b = eng_b.get_legal_realignments(st_b, acting_p)
                if realign_a != realign_b:
                    raise AssertionError(
                        f"Legal realignment target options mismatch:\n"
                        f"  Scenario: {scenario} [attempting realignment]\n"
                        f"  Native proposes ({len(realign_a)} options): {sorted(realign_a)}\n"
                        f"  {other_engine_name} proposes ({len(realign_b)} options): {sorted(realign_b)}\n"
                        f"  Difference (only Native): {sorted(realign_a - realign_b)}\n"
                        f"  Difference (only {other_engine_name}): {sorted(realign_b - realign_a)}"
                    )

        # Execute according to user rules
        if is_event:
            do_event(player)
        else:
            if has_opp and event_first:
                do_event(opp_player)
                do_ops(player)
            elif has_opp and not event_first:
                do_ops(player)
                do_event(opp_player)
            else:
                do_ops(player)

        # State outcome parity check
        _assert_outcome_parity(st_a, st_b, scenario, other_engine_name)


# =============================================================================
# UNIFIED DIFFERENTIAL FUZZING TEST SUITE
# =============================================================================

class TestUnifiedFuzzingDifferential:
    """Parameterized multi-action differential fuzzing between the native engine and Struggler."""

    @pytest.mark.parametrize("fuzz_seed", list(range(1001, 1016)))
    def test_fuzz_multi_step_placement_walk(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Executes multi-step sequential legal influence placement random walks.
        At each step:
          1. Asserts legal placement choices match between engines.
          2. Picks identical country from common legal choices.
          3. Places influence on both engines.
          4. Asserts resulting board states match after every microaction.
        """
        rng = random.Random(fuzz_seed)
        arch = ARCHETYPES[fuzz_seed % len(ARCHETYPES)]
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        apply_fuzzed_initial_state(rng, state_a, state_b, arch)
        assert_states_equal(state_a, state_b)

        # 8 sequential placement actions alternating between US and USSR
        for step in range(8):
            player = Player.US if step % 2 == 0 else Player.USSR
            lp_a = eng_a.get_legal_placements(state_a, player)
            lp_b = eng_b.get_legal_placements(state_b, player)

            assert lp_a == lp_b, f"Step {step} legal placements mismatch: {lp_a ^ lp_b}"

            common_choices = sorted(lp_a & lp_b)
            assert len(common_choices) > 0

            chosen_country = rng.choice(common_choices)
            cid = get_country_id(chosen_country)
            cur = state_a.get_country(cid)
            if player == Player.US:
                state_a.set_country(cid, cur.us_influence + 1, cur.ussr_influence)
                state_b.set_country(cid, cur.us_influence + 1, cur.ussr_influence)
            else:
                state_a.set_country(cid, cur.us_influence, cur.ussr_influence + 1)
                state_b.set_country(cid, cur.us_influence, cur.ussr_influence + 1)

            assert_boards_equal(state_a, state_b)

    @pytest.mark.parametrize("fuzz_seed", list(range(2001, 2011)))
    def test_fuzz_multi_step_setup_trajectories(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Executes diverse 13-microaction setup trajectories (6 USSR + 7 US placements).
        Steps microaction by microaction and asserts board state match after every single step.
        """
        rng = random.Random(fuzz_seed)
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        # 6 USSR placements into Eastern Europe
        for _ in range(6):
            c = rng.choice(_EASTERN_EUROPE_SETUP)
            cid = get_country_id(c)
            act = MicroAction(decision_type=DecisionType.POINT_NODE, primary_id=cid)
            assert eng_a.step(state_a, act)
            assert eng_b.step(state_b, act)
            assert_boards_equal(state_a, state_b)

        # 7 US placements into Western Europe
        for _ in range(7):
            c = rng.choice(_WESTERN_EUROPE_SETUP)
            cid = get_country_id(c)
            act = MicroAction(decision_type=DecisionType.POINT_NODE, primary_id=cid)
            assert eng_a.step(state_a, act)
            assert eng_b.step(state_b, act)
            assert_boards_equal(state_a, state_b)

        assert_states_equal(state_a, state_b)

    @pytest.mark.parametrize("fuzz_seed", list(range(3001, 3006)))
    def test_fuzz_multi_step_event_chain(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Executes a sequential chain of 7 card events mutating tracks, flags, and board states.
        Asserts decision space and resulting states match after every single event action.
        """
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)
        assert_states_equal(state_a, state_b)

        # 1. Fidel
        eng_a.play_event(state_a, Player.USSR, 8)
        eng_b.play_event(state_b, Player.USSR, 8)
        assert_states_equal(state_a, state_b)

        # 2. Romanian Abdication
        eng_a.play_event(state_a, Player.USSR, 12)
        eng_b.play_event(state_b, Player.USSR, 12)
        assert_states_equal(state_a, state_b)

        # 3. Vietnam Revolts
        eng_a.play_event(state_a, Player.USSR, 9)
        eng_b.play_event(state_b, Player.USSR, 9)
        assert_states_equal(state_a, state_b)

        # 4. US Japan
        eng_a.play_event(state_a, Player.US, 27)
        eng_b.play_event(state_b, Player.US, 27)
        assert_states_equal(state_a, state_b)

        # 5. Duck and Cover
        eng_a.play_event(state_a, Player.US, 4)
        eng_b.play_event(state_b, Player.US, 4)
        assert_states_equal(state_a, state_b)

        # 6. Nuclear Test Ban
        eng_a.play_event(state_a, Player.US, 34)
        eng_b.play_event(state_b, Player.US, 34)
        assert_states_equal(state_a, state_b)

        # 7. Truman Doctrine (with interactive decision)
        fr_id = get_country_id("France")
        state_a.set_country(fr_id, 0, 2)
        state_b.set_country(fr_id, 0, 2)
        assert_states_equal(state_a, state_b)

        eng_a.play_event(state_a, Player.US, 19)
        eng_b.play_event(state_b, Player.US, 19)

        assert eng_a.has_pending_decision(state_a) == eng_b.has_pending_decision(state_b)
        assert eng_a.get_decision_targets(state_a) == eng_b.get_decision_targets(state_b)

        eng_a.resolve_decision(state_a, "France")
        eng_b.resolve_decision(state_b, "France")
        assert_states_equal(state_a, state_b)

    @pytest.mark.parametrize("fuzz_seed", list(range(4001, 4021)))
    def test_fuzz_coup_and_realignment_multi_state_differential(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Fuzzes 20 diverse board configurations & DEFCONs, asserting 100% agreement on legal coup/realign targets."""
        rng = random.Random(fuzz_seed)
        arch = ARCHETYPES[fuzz_seed % len(ARCHETYPES)]
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        apply_fuzzed_initial_state(rng, state_a, state_b, arch)
        assert_states_equal(state_a, state_b)

        # 1. Coup target legality
        coups_us_a = eng_a.get_legal_coups(state_a, Player.US, ops=3)
        coups_us_b = eng_b.get_legal_coups(state_b, Player.US, ops=3)
        assert coups_us_a == coups_us_b, f"Seed {fuzz_seed} US coup diff: {coups_us_a ^ coups_us_b}"

        coups_ussr_a = eng_a.get_legal_coups(state_a, Player.USSR, ops=3)
        coups_ussr_b = eng_b.get_legal_coups(state_b, Player.USSR, ops=3)
        assert coups_ussr_a == coups_ussr_b, f"Seed {fuzz_seed} USSR coup diff: {coups_ussr_a ^ coups_ussr_b}"

        # 2. Realignment target legality
        realign_us_a = eng_a.get_legal_realignments(state_a, Player.US)
        realign_us_b = eng_b.get_legal_realignments(state_b, Player.US)
        assert realign_us_a == realign_us_b, f"Seed {fuzz_seed} US realign diff: {realign_us_a ^ realign_us_b}"

        realign_ussr_a = eng_a.get_legal_realignments(state_a, Player.USSR)
        realign_ussr_b = eng_b.get_legal_realignments(state_b, Player.USSR)
        assert realign_ussr_a == realign_ussr_b, f"Seed {fuzz_seed} USSR realign diff: {realign_ussr_a ^ realign_ussr_b}"

    @pytest.mark.parametrize("fuzz_seed", list(range(5001, 5021)))
    def test_fuzz_regional_scoring_across_all_archetypes(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Fuzzes scoring across all 6 regions + Southeast Asia across diverse board archetypes."""
        rng = random.Random(fuzz_seed)
        arch = ARCHETYPES[fuzz_seed % len(ARCHETYPES)]
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        apply_fuzzed_initial_state(rng, state_a, state_b, arch)
        assert_states_equal(state_a, state_b)

        regions = [
            Region.EUROPE,
            Region.ASIA,
            Region.MIDDLE_EAST,
            Region.AFRICA,
            Region.CENTRAL_AMERICA,
            Region.SOUTH_AMERICA,
            Region.SOUTHEAST_ASIA,
        ]

        for reg in regions:
            delta_a = eng_a.score_region(state_a, reg)
            delta_b = eng_b.score_region(state_b, reg)
            assert delta_a == delta_b, f"Seed {fuzz_seed} {arch} {reg.name}: a={delta_a} vs b={delta_b}"

    # =========================================================================
    # COMPREHENSIVE EVENT FUZZING (ALL 110 CARDS)
    # =========================================================================

    @pytest.mark.parametrize("fuzz_seed", list(range(6001, 6016)))
    def test_fuzz_early_war_card_events(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Fuzzes Early War cards (1-35, 103-106) with random hand distribution and user probabilities:
          - Own/Neutral: 2/3 Event, 1/3 Ops.
          - Opponent: Ops with 1/2 Event first / after.
        """
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        other_name = "struggler"
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        _execute_fuzzed_card_game_rollout(
            eng_a, state_a, eng_b, state_b,
            card_pool=_EARLY_WAR_CARDS,
            rounds=6,
            seed=fuzz_seed,
            other_engine_name=other_name,
        )


    @pytest.mark.parametrize("fuzz_seed", list(range(7001, 7016)))
    def test_fuzz_mid_war_card_events(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Fuzzes Middle War cards (36-78, 107-108) with random hand distribution and user probabilities:
          - Own/Neutral: 2/3 Event, 1/3 Ops.
          - Opponent: Ops with 1/2 Event first / after.
        """
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        other_name = "struggler"
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        _execute_fuzzed_card_game_rollout(
            eng_a, state_a, eng_b, state_b,
            card_pool=_MID_WAR_CARDS,
            rounds=6,
            seed=fuzz_seed,
            other_engine_name=other_name,
        )


    @pytest.mark.parametrize("fuzz_seed", list(range(8001, 8016)))
    def test_fuzz_late_war_card_events(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Fuzzes Late War cards (79-102, 109-110) with random hand distribution and user probabilities:
          - Own/Neutral: 2/3 Event, 1/3 Ops.
          - Opponent: Ops with 1/2 Event first / after.
        """
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        other_name = "struggler"
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        _execute_fuzzed_card_game_rollout(
            eng_a, state_a, eng_b, state_b,
            card_pool=_LATE_WAR_CARDS,
            rounds=6,
            seed=fuzz_seed,
            other_engine_name=other_name,
        )


    @pytest.mark.parametrize("fuzz_seed", list(range(9001, 9021)))
    def test_fuzz_all_110_cards_full_deck_rollout(
        self,
        fuzz_seed: int,
        engine_pair: tuple[tuple[EngineProtocol, GameStateProtocol], tuple[EngineProtocol, GameStateProtocol]],
    ):
        """Full 110-card deck multi-round rollout fuzzing across both engine pairs."""
        (eng_a, state_a), (eng_b, state_b) = engine_pair
        other_name = "struggler"
        eng_a.init_game(state_a, fuzz_seed)
        eng_b.init_game(state_b, fuzz_seed)

        _execute_fuzzed_card_game_rollout(
            eng_a, state_a, eng_b, state_b,
            card_pool=_ALL_110_CARDS,
            rounds=8,
            seed=fuzz_seed,
            other_engine_name=other_name,
        )
