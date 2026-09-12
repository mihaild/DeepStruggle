import pytest
import numpy as np
import ts_engine as ts
from bindings.action_encoder import ActionEncoder

def test_engine_auto_advance_flag_exposed():
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    # Test step signatures accept auto_advance
    mask = ts.Engine.get_flat_action_mask(state)
    legal = [i for i, m in enumerate(mask) if m]
    assert len(legal) > 0
    act_idx = legal[0]
    ma = ts.ActionMask.decode_flat_action(state, act_idx)
    
    # step with auto_advance=False
    s1 = ts.GameState()
    ts.Engine.init_game(s1, 12345)
    ok1 = ts.Engine.step(s1, ma, False)
    assert ok1 is True

    # step with auto_advance=True
    s2 = ts.GameState()
    ts.Engine.init_game(s2, 12345)
    ok2 = ts.Engine.step(s2, ma, True)
    assert ok2 is True

    # step_flat with auto_advance
    s3 = ts.GameState()
    ts.Engine.init_game(s3, 12345)
    ok3 = ts.Engine.step_flat(s3, act_idx, True)
    assert ok3 is True

def test_suez_crisis_auto_advance():
    state = ts.GameState()
    ts.Engine.init_game(state, 42)
    state.current_phase = ts.Phase.ACTION_ROUND

    # Set UK(1)=2, France(8)=1, Israel(23)=0 -> sum=3 <= 4 using state.set_country
    state.set_country(1, 2, 0)
    state.set_country(8, 1, 0)
    state.set_country(23, 0, 0)

    ts.CardHandlers.trigger_event(state, 28, ts.Player.USSR)
    assert state.ctx().resolving_card == 28

    adv = ts.Engine.auto_advance_step(state)
    assert adv >= 3
    assert state.ctx().resolving_card == 0
    assert state.get_country(1).us_influence == 0
    assert state.get_country(8).us_influence == 0
    assert state.get_country(23).us_influence == 0

def test_muslim_revolution_auto_advance():
    state = ts.GameState()
    ts.Engine.init_game(state, 42)
    state.current_phase = ts.Phase.ACTION_ROUND

    # Targets: Sudan(51), Iran(25), Iraq(24), Egypt(29), Libya(30), Saudi(28), Syria(22), Jordan(26)
    # Set only Sudan=2 and Egypt=1 with US influence (total 2 countries <= 2)
    for cid in [51, 25, 24, 29, 30, 28, 22, 26]:
        state.set_country(cid, 0, 0)
    state.set_country(51, 2, 0) # Sudan
    state.set_country(29, 1, 0) # Egypt

    ts.CardHandlers.trigger_event(state, 56, ts.Player.USSR)
    assert state.ctx().resolving_card == 56

    adv = ts.Engine.auto_advance_step(state)
    assert adv >= 2
    assert state.ctx().resolving_card == 0
    assert state.get_country(51).us_influence == 0
    assert state.get_country(29).us_influence == 0

def test_east_european_unrest_auto_advance():
    state = ts.GameState()
    ts.Engine.init_game(state, 42)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.turn = 3 # Early War

    # EEU targets in Early War: Finland(5), Austria(13), East Germany(14), Poland(15), Czechoslovakia(16),
    # Hungary(17), Yugoslavia(18), Romania(19), Bulgaria(20)
    for cid in [5, 13, 14, 15, 16, 17, 18, 19, 20]:
        state.set_country(cid, 0, 0)
    state.set_country(15, 0, 3) # Poland
    state.set_country(16, 0, 1) # Czechoslovakia

    ts.CardHandlers.trigger_event(state, 29, ts.Player.US)
    assert state.ctx().resolving_card == 29

    adv = ts.Engine.auto_advance_step(state)
    assert adv >= 2
    assert state.ctx().resolving_card == 0
    assert state.get_country(15).ussr_influence == 2
    assert state.get_country(16).ussr_influence == 0

def test_truman_and_independent_reds_auto_advance():
    # Truman Doctrine (19): 1 eligible uncontrolled country with USSR influence
    s_truman = ts.GameState()
    ts.Engine.init_game(s_truman, 42)
    s_truman.current_phase = ts.Phase.ACTION_ROUND
    # Clear European countries and set only France with USSR influence, uncontrolled
    for cid in range(84):
        s_truman.set_country(cid, 0, 0)
    s_truman.set_country(8, 0, 2) # France: stability 3, USSR 2 -> uncontrolled
    ts.CardHandlers.trigger_event(s_truman, 19, ts.Player.US)
    assert s_truman.ctx().resolving_card == 19
    adv = ts.Engine.auto_advance_step(s_truman)
    assert adv >= 1
    assert s_truman.ctx().resolving_card == 0
    assert s_truman.get_country(8).ussr_influence == 0

    # Independent Reds (22): 1 eligible country
    s_ir = ts.GameState()
    ts.Engine.init_game(s_ir, 42)
    s_ir.current_phase = ts.Phase.ACTION_ROUND
    for cid in [18, 19, 20, 17, 16]: # Yugoslavia, Romania, Bulgaria, Hungary, Czechoslovakia
        s_ir.set_country(cid, 0, 0)
    s_ir.set_country(18, 0, 2) # Yugoslavia has 2 USSR influence
    ts.CardHandlers.trigger_event(s_ir, 22, ts.Player.US)
    assert s_ir.ctx().resolving_card == 22
    adv = ts.Engine.auto_advance_step(s_ir)
    assert adv >= 1
    assert s_ir.ctx().resolving_card == 0
    assert s_ir.get_country(18).us_influence == 2

def test_vectorized_batch_runner_step_flat_all_auto_advance():
    runner = ts.VectorizedBatchRunner(4, 10000)
    for i in range(4):
        runner.reset_game(i, 10000 + i)
    runner.refresh_all()

    # Get legal action
    masks = runner.get_action_masks()
    actions = [int(np.argmax(masks[i])) for i in range(4)]
    
    # step_flat_all with auto_advance=True
    res = runner.step_flat_all(actions, auto_advance=True)
    assert len(res) == 4
    assert all(r == 1 for r in res)

def test_backward_compatibility_bit_for_bit():
    """Verify that when auto_advance is False, behavior is 100% bit-for-bit identical to default."""
    s_default = ts.GameState()
    ts.Engine.init_game(s_default, 999)

    s_explicit_false = ts.GameState()
    ts.Engine.init_game(s_explicit_false, 999)

    for _ in range(50):
        if ts.Engine.is_terminal(s_default):
            break
        m1 = ts.Engine.get_flat_action_mask(s_default)
        m2 = ts.Engine.get_flat_action_mask(s_explicit_false)
        assert np.array_equal(m1, m2)

        leg = [i for i, v in enumerate(m1) if v]
        act = leg[0]

        ts.Engine.step_flat(s_default, act) # default false
        ts.Engine.step_flat(s_explicit_false, act, False) # explicit false

        assert s_default.victory_points == s_explicit_false.victory_points
        assert s_default.turn == s_explicit_false.turn
        assert s_default.action_round == s_explicit_false.action_round
        assert s_default.defcon == s_explicit_false.defcon
        for c in range(84):
            assert s_default.get_country(c).us_influence == s_explicit_false.get_country(c).us_influence
            assert s_default.get_country(c).ussr_influence == s_explicit_false.get_country(c).ussr_influence

def _deterministic_policy(mask):
    # CONFIRM_DONE (211) is the highest flat action index, so a plain "lowest legal
    # index" policy would never pick it unless it's the only legal action -- which
    # would never exercise a genuine early-stop decision (allow_early_stop=1) at all,
    # and could hide any place where auto_advance wrongly forces through such a choice.
    # Preferring it whenever it's legal makes the two runs actually diverge on those
    # branches, so equivalence between them is a real check.
    leg = np.where(mask > 0)[0]
    if 211 in leg:
        return 211
    return int(leg[0])

def test_policy_equivalence_games():
    """Play games with deterministic policies across 20 seeds: Run A (auto_advance=False) vs Run B (auto_advance=True).
    Both runs must reach the EXACT same final board state, VPs, defcon, card locations, and winner."""
    seeds = [1, 7, 13, 42, 69, 100, 200, 333, 500, 777, 999, 1234, 2026, 4000, 5555, 7123, 8888, 9999, 10042, 12345]

    total_steps_A = 0
    total_steps_B = 0

    for seed in seeds:
        sA = ts.GameState()
        ts.Engine.init_game(sA, seed)
        stepsA = 0
        while not ts.Engine.is_terminal(sA) and stepsA < 1500:
            mA = ts.Engine.get_flat_action_mask(sA)
            actA = _deterministic_policy(mA)
            ts.Engine.step_flat(sA, actA, False)
            stepsA += 1

        sB = ts.GameState()
        ts.Engine.init_game(sB, seed)
        ts.Engine.auto_advance_step(sB)
        stepsB = 0
        while not ts.Engine.is_terminal(sB) and stepsB < 1500:
            mB = ts.Engine.get_flat_action_mask(sB)
            actB = _deterministic_policy(mB)
            ts.Engine.step_flat(sB, actB, True)
            stepsB += 1

        # Equivalence assertions
        assert sA.victory_points == sB.victory_points, f"VP mismatch on seed {seed}: {sA.victory_points} vs {sB.victory_points}"
        assert sA.defcon == sB.defcon
        assert sA.turn == sB.turn
        assert sA.current_phase == sB.current_phase
        assert ts.Engine.get_terminal_utility(sA) == ts.Engine.get_terminal_utility(sB)
        for c in range(84):
            assert sA.get_country(c).us_influence == sB.get_country(c).us_influence
            assert sA.get_country(c).ussr_influence == sB.get_country(c).ussr_influence
        for card in range(1, 111):
            assert sA.get_card_location(card) == sB.get_card_location(card)

        # Auto-advance must have strictly reduced the number of micro-actions!
        assert stepsB < stepsA, f"Auto-advance did not reduce steps on seed {seed}: {stepsA} vs {stepsB}"
        total_steps_A += stepsA
        total_steps_B += stepsB

    overall_reduction = (total_steps_A - total_steps_B) / total_steps_A * 100.0
    print(f"\n20-Game Equivalence: Total Steps {total_steps_A} -> {total_steps_B} (Saved {total_steps_A - total_steps_B} steps, {overall_reduction:.2f}%)")

def test_batch_runner_equivalence():
    """Verify that VectorizedBatchRunner with auto_advance=False vs auto_advance=True reaches identical results across 16 parallel environments."""
    num_envs = 16
    base_seed = 50000

    # Run A: auto_advance = False
    runnerA = ts.VectorizedBatchRunner(num_envs, base_seed)
    for i in range(num_envs):
        runnerA.reset_game(i, base_seed + i)
    runnerA.refresh_all()

    activeA = np.ones(num_envs, dtype=bool)
    stepsA = np.zeros(num_envs, dtype=int)
    max_steps = 1500
    while np.any(activeA) and np.max(stepsA) < max_steps:
        masksA = runnerA.get_action_masks()
        actionsA = []
        for i in range(num_envs):
            if activeA[i]:
                act = _deterministic_policy(masksA[i])
                actionsA.append(act)
                stepsA[i] += 1
            else:
                actionsA.append(0)
        runnerA.step_flat_all(actionsA, auto_advance=False)
        termsA = runnerA.get_terminals()
        for i in range(num_envs):
            if activeA[i] and termsA[i]:
                activeA[i] = False

    # Run B: auto_advance = True
    runnerB = ts.VectorizedBatchRunner(num_envs, base_seed)
    for i in range(num_envs):
        runnerB.reset_game(i, base_seed + i)
    runnerB.refresh_all()

    activeB = np.ones(num_envs, dtype=bool)
    stepsB = np.zeros(num_envs, dtype=int)
    while np.any(activeB) and np.max(stepsB) < max_steps:
        masksB = runnerB.get_action_masks()
        actionsB = []
        for i in range(num_envs):
            if activeB[i]:
                act = _deterministic_policy(masksB[i])
                actionsB.append(act)
                stepsB[i] += 1
            else:
                actionsB.append(0)
        runnerB.step_flat_all(actionsB, auto_advance=True)
        termsB = runnerB.get_terminals()
        for i in range(num_envs):
            if activeB[i] and termsB[i]:
                activeB[i] = False

    # Verify identical terminal states for every env
    for i in range(num_envs):
        sA = runnerA.get_state(i)
        sB = runnerB.get_state(i)
        assert sA.victory_points == sB.victory_points, f"Env {i} VP mismatch: {sA.victory_points} vs {sB.victory_points}"
        assert sA.defcon == sB.defcon
        assert sA.turn == sB.turn
        assert ts.Engine.get_terminal_utility(sA) == ts.Engine.get_terminal_utility(sB)
        for c in range(84):
            assert sA.get_country(c).us_influence == sB.get_country(c).us_influence
            assert sA.get_country(c).ussr_influence == sB.get_country(c).ussr_influence
        for card in range(1, 111):
            assert sA.get_card_location(card) == sB.get_card_location(card)
        assert stepsB[i] < stepsA[i], f"Env {i} steps not reduced: {stepsA[i]} vs {stepsB[i]}"

    print(f"\nBatch Runner 16-Env Equivalence: Total Steps {np.sum(stepsA)} -> {np.sum(stepsB)} (Saved {np.sum(stepsA) - np.sum(stepsB)} steps)")
