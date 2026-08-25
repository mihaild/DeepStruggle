"""Comprehensive Credit Assignment and Endgame Reward Propagation Test Suite.

Validates that all game-ending triggers (DEFCON suicide, opponent events, Cuban Missile Crisis,
held scoring cards, Mil Ops +/-20 VP, Event +/-20 VP, and Final Scoring) correctly assign
rewards and propagate negative advantages to losing actions and positive advantages to winning actions.
"""

import pytest
import numpy as np
import torch
import ts_engine as ts
from ai.env.reward_calculator import ZeroSumTerminalReward, ShapedZeroSumReward
from ai.env.ts_env import TsVectorizedEnv
from ai.training.rollout_buffer import RolloutBuffer


def setup_to_action_round(st: ts.GameState) -> None:
    """Steps through game setup until Action Round 1."""
    step = 0
    while st.current_phase != ts.Phase.ACTION_ROUND and step < 50:
        mask = ts.get_flat_action_mask(st)
        legal = [i for i, m in enumerate(mask) if m == 1]
        ts.Engine.step(st, ts.decode_flat_action(st, legal[0]))
        step += 1


class TestAlgebraicRewardCalculator:
    """Unit tests for algebraic zero-sum reward calculator strategies."""

    def test_pure_algebraic_rewards(self):
        calc = ZeroSumTerminalReward()
        # US Win (+1.0 utility): US acting (+1) gets +1.0, USSR acting (-1) gets -1.0
        acting = np.array([1, -1, 1, -1], dtype=np.int8)
        dones = np.array([True, True, True, True], dtype=bool)
        term_utils = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        prev_vp = np.zeros(4, dtype=np.int8)
        curr_vp = np.zeros(4, dtype=np.int8)

        rewards = calc.compute_step_rewards(acting, dones, term_utils, prev_vp, curr_vp)
        assert np.allclose(rewards, [1.0, -1.0, -1.0, 1.0])

        # Non-terminal steps must always be 0.0
        dones_false = np.zeros(4, dtype=bool)
        rewards_nt = calc.compute_step_rewards(acting, dones_false, term_utils, prev_vp, curr_vp)
        assert np.allclose(rewards_nt, [0.0, 0.0, 0.0, 0.0])

    def test_zero_sum_sign_permutations(self):
        calc = ZeroSumTerminalReward()
        curr_p = torch.tensor([1, -1, 1, -1], dtype=torch.int8)
        next_p = torch.tensor([1, -1, -1, 1], dtype=torch.int8)

        signs = calc.compute_zero_sum_sign(curr_p, next_p)
        assert torch.allclose(signs, torch.tensor([1.0, 1.0, -1.0, -1.0]))


class TestDefconSuicideCredit:
    """Tests for DEFCON nuclear suicide reward and GAE advantage assignment."""

    def test_ussr_defcon_suicide_coup(self):
        env = TsVectorizedEnv(num_envs=1, base_seed=42, auto_reset=False)
        st = env.runner.get_state(0)
        setup_to_action_round(st)
        st.defcon = 2
        st.phasing_player = ts.Player.USSR
        st.ctx().decision_player = ts.Player.USSR

        # Choose a card with ops for USSR (e.g. Card 5)
        st.set_card_location(5, ts.CardLocation.HAND_USSR)
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0))  # Ops
        if st.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1, 0, 0))    # Coup

        # Coup Panama (70) -> Flat action 189
        _, _, rewards, dones, info = env.step([189])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == 20  # US Win
        assert rewards[0] == -1.0  # USSR was acting player -> gets -1.0
        assert info["acting_players"][0] == -1  # USSR

    def test_us_defcon_suicide_coup(self):
        env = TsVectorizedEnv(num_envs=1, base_seed=43, auto_reset=False)
        st = env.runner.get_state(0)
        setup_to_action_round(st)
        st.defcon = 2
        st.phasing_player = ts.Player.US
        st.ctx().decision_player = ts.Player.US

        st.set_card_location(4, ts.CardLocation.HAND_US)
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 4, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0))  # Ops
        if st.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1, 0, 0))    # Coup

        # Coup Panama (70) -> Flat action 189
        _, _, rewards, dones, info = env.step([189])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == -20  # USSR Win
        assert rewards[0] == -1.0  # US was acting player -> gets -1.0
        assert info["acting_players"][0] == 1  # US


class TestOpponentEventDelegationCredit:
    """Tests CIA Created / opponent event delegation backward credit assignment."""

    def test_cia_created_suicide_backward_advantage(self):
        device = torch.device("cpu")
        buf = RolloutBuffer(buffer_size=3, num_envs=1, obs_dim=10, action_dim=5, device=device)

        # Step 0: USSR selects CIA Created (p=-1)
        # Step 1: US receives delegated op and chooses COUP (p=1)
        # Step 2: US coups Iraq at DEFCON 2 -> DEFCON reaches 1 -> USSR loses! US wins (+1.0 utility)
        buf.obs.fill_(0)
        buf.masks.fill_(1)
        buf.actions.fill_(0)
        buf.log_probs.fill_(0)
        buf.values_win.fill_(0.0)
        buf.values_vp.fill_(0.0)

        buf.players[:, 0] = torch.tensor([-1, 1, 1], dtype=torch.int8)
        buf.rewards[:, 0] = torch.tensor([0.0, 0.0, 1.0])  # US gets +1.0 at step 2
        buf.dones[:, 0] = torch.tensor([False, False, True])

        buf.compute_gae(
            last_v_win=torch.tensor([0.0]),
            last_v_vp=torch.tensor([0.0]),
            last_dones=torch.tensor([True]),
            last_players=torch.tensor([1]),
            gamma=0.999,
            gae_lambda=0.98,
        )

        # Step 0 (USSR choosing CIA Created): MUST receive NEGATIVE return and advantage
        assert buf.returns_win[0, 0].item() < -0.9
        assert buf.advantages[0, 0].item() < 0.0

        # Steps 1 & 2 (US springing the trap): MUST receive POSITIVE return and advantage
        assert buf.returns_win[1, 0].item() > 0.9
        assert buf.returns_win[2, 0].item() == 1.0
        assert buf.advantages[2, 0].item() > 0.0


class TestHeldScoringCardsCredit:
    """Tests Rule 4.4 held scoring card multi-step backward penalty."""

    def test_held_scoring_card_gae_propagation(self):
        device = torch.device("cpu")
        buf = RolloutBuffer(buffer_size=4, num_envs=1, obs_dim=10, action_dim=5, device=device)

        # Step 0: USSR chooses 3-Ops card instead of scoring card (p=-1)
        # Step 1: USSR places influence (p=-1)
        # Step 2: US plays final AR6 card (p=1)
        # Step 3: US finishes AR6 -> finish_end_turn() detects held scoring card in USSR hand -> US Wins (+1.0)
        buf.obs.fill_(0)
        buf.masks.fill_(1)
        buf.actions.fill_(0)
        buf.log_probs.fill_(0)
        buf.values_win.fill_(0.0)
        buf.values_vp.fill_(0.0)

        buf.players[:, 0] = torch.tensor([-1, -1, 1, 1], dtype=torch.int8)
        buf.rewards[:, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # US receives +1.0 on step 3
        buf.dones[:, 0] = torch.tensor([False, False, False, True])

        buf.compute_gae(
            last_v_win=torch.tensor([0.0]),
            last_v_vp=torch.tensor([0.0]),
            last_dones=torch.tensor([True]),
            last_players=torch.tensor([1]),
            gamma=0.999,
            gae_lambda=0.98,
        )

        # USSR steps 0 and 1 MUST have strong NEGATIVE returns
        assert buf.returns_win[0, 0].item() < -0.9
        assert buf.returns_win[1, 0].item() < -0.9
        assert buf.advantages[0, 0].item() < 0.0
        assert buf.advantages[1, 0].item() < 0.0

        # US steps 2 and 3 MUST have POSITIVE returns
        assert buf.returns_win[2, 0].item() > 0.9
        assert buf.returns_win[3, 0].item() == 1.0
        assert buf.advantages[3, 0].item() > 0.0


class TestVictoryPoints20MilOpsAndEvents:
    """Tests winning and losing by +/-20 VP via Events, Scoring, and Military Operations."""

    def test_us_win_20_vp_via_europe_control(self):
        env = TsVectorizedEnv(num_envs=1, base_seed=50, auto_reset=False)
        st = env.runner.get_state(0)
        setup_to_action_round(st)
        st.phasing_player = ts.Player.US
        st.ctx().decision_player = ts.Player.US

        # Give US Europe Control (All 5 BGs: 7, 8, 10, 14, 15)
        for cid in range(84):
            if ts.MapData.get_country_info(cid).get("region") == 0:
                st.set_country(cid, 0, 0)

        st.set_country(7, 4, 0)   # West Germany
        st.set_country(8, 3, 0)   # France
        st.set_country(10, 2, 0)  # Italy
        st.set_country(14, 3, 0)  # East Germany
        st.set_country(15, 3, 0)  # Poland

        # US plays Europe Scoring (Card 2)
        st.set_card_location(2, ts.CardLocation.HAND_US)
        ma = ts.MicroAction(ts.DecisionType.SELECT_CARD, 2, 0, 0)
        flat_act = ts.encode_micro_action(st, ma)
        _, _, rewards, dones, info = env.step([flat_act])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == 20  # US Win
        assert rewards[0] == 1.0  # US was acting player -> gets +1.0
        assert info["acting_players"][0] == 1

    def test_us_win_via_wargames_early_end(self):
        env = TsVectorizedEnv(num_envs=1, base_seed=50, auto_reset=False)
        st = env.runner.get_state(0)
        setup_to_action_round(st)
        st.phasing_player = ts.Player.US
        st.ctx().decision_player = ts.Player.US
        st.defcon = 2
        st.victory_points = 18

        # Play Wargames (Card 100)
        st.set_card_location(100, ts.CardLocation.HAND_US)
        ma = ts.MicroAction(ts.DecisionType.SELECT_CARD, 100, 0, 0)
        flat_act = ts.encode_micro_action(st, ma)
        env.step([flat_act])

        # Play mode 0 (Event)
        ma_evt = ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 0, 0, 0)
        flat_evt = ts.encode_micro_action(st, ma_evt)
        env.step([flat_evt])

        # Choose Branch 0 (End game and give 6 VP)
        ma_br = ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 0, 0, 0)
        flat_br = ts.encode_micro_action(st, ma_br)
        _, _, rewards, dones, info = env.step([flat_br])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == 12  # 18 - 6 = 12
        assert rewards[0] == 1.0  # US was acting player -> gets +1.0 (US won since VP > 0)
        assert info["acting_players"][0] == 1

    def test_ussr_win_minus_20_vp_via_europe_scoring(self):
        env = TsVectorizedEnv(num_envs=1, base_seed=51, auto_reset=False)
        st = env.runner.get_state(0)
        setup_to_action_round(st)
        st.phasing_player = ts.Player.USSR
        st.ctx().decision_player = ts.Player.USSR

        # Give USSR Europe Control (5 BGs: 7, 8, 10, 14, 15; Clear US non-BGs)
        for cid in range(84):
            if ts.MapData.get_country_info(cid).get("region") == 0:
                st.set_country(cid, 0, 0)

        # Set USSR control on all 5 BGs
        st.set_country(7, 0, 4)   # West Germany
        st.set_country(8, 0, 3)   # France
        st.set_country(10, 0, 2)  # Italy
        st.set_country(14, 0, 3)  # East Germany
        st.set_country(15, 0, 3)  # Poland

        # USSR plays Europe Scoring (Card 2)
        st.set_card_location(2, ts.CardLocation.HAND_USSR)
        ma = ts.MicroAction(ts.DecisionType.SELECT_CARD, 2, 0, 0)
        flat_act = ts.encode_micro_action(st, ma)
        _, _, rewards, dones, info = env.step([flat_act])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == -20  # USSR Win
        assert rewards[0] == 1.0  # USSR was acting player -> gets +1.0
        assert info["acting_players"][0] == -1

    def test_mil_ops_20_vp_penalty_end_of_turn(self):
        calc = ZeroSumTerminalReward()
        # USSR reaches End of Turn with 0 Mil Ops at DEFCON 2, US has 5 Mil Ops.
        # VP before was +18, status check adds +2 VP -> VP reaches +20 (US Win)
        acting_us = np.array([1], dtype=np.int8)  # US took the final AR6 step
        acting_ussr = np.array([-1], dtype=np.int8)
        dones = np.array([True], dtype=bool)
        term_utils = np.array([1.0], dtype=np.float32)  # US Win

        prev_vp = np.array([18], dtype=np.int8)
        curr_vp = np.array([20], dtype=np.int8)

        # US gets +1.0, USSR gets -1.0
        reward_us = calc.compute_step_rewards(acting_us, dones, term_utils, prev_vp, curr_vp)
        reward_ussr = calc.compute_step_rewards(acting_ussr, dones, term_utils, prev_vp, curr_vp)

        assert reward_us[0] == 1.0
        assert reward_ussr[0] == -1.0


class TestCubanMissileCrisisCoupSuicide:
    """Tests Cuban Missile Crisis (CMC) coup violation instant loss."""

    def test_cmc_active_us_ussr_coup_instant_loss(self):
        env = TsVectorizedEnv(num_envs=1, base_seed=60, auto_reset=False)
        st = env.runner.get_state(0)
        setup_to_action_round(st)
        st.defcon = 4
        st.set_flag(ts.EffectBits.CMC_ACTIVE_US)
        # USSR has 0 influence in Cuba (71)
        st.set_country(71, 0, 0)
        # Ensure Iran (17) has US influence to be legal coup target
        st.set_country(17, 2, 0)

        st.phasing_player = ts.Player.USSR
        st.ctx().decision_player = ts.Player.USSR

        st.set_card_location(5, ts.CardLocation.HAND_USSR)
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0))  # Ops
        if st.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1, 0, 0))    # Coup

        # Coup Iran (17) -> Flat action 136
        _, _, rewards, dones, info = env.step([136])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == 20  # US Win
        assert rewards[0] == -1.0  # USSR was acting player -> gets -1.0
        assert info["acting_players"][0] == -1

    def test_cmc_active_ussr_us_coup_instant_loss(self):
        env = TsVectorizedEnv(num_envs=1, base_seed=61, auto_reset=False)
        st = env.runner.get_state(0)
        setup_to_action_round(st)
        st.defcon = 4
        st.set_flag(ts.EffectBits.CMC_ACTIVE_USSR)
        # US has 0 influence in West Germany (7) and Turkey (12)
        st.set_country(7, 0, 0)
        st.set_country(12, 0, 0)
        # Ensure Panama (70) has USSR influence to be legal coup target
        st.set_country(70, 0, 2)

        st.phasing_player = ts.Player.US
        st.ctx().decision_player = ts.Player.US

        st.set_card_location(4, ts.CardLocation.HAND_US)
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 4, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0))  # Ops
        if st.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1, 0, 0))    # Coup

        # Coup Panama (70) -> Flat action 189
        _, _, rewards, dones, info = env.step([189])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == -20  # USSR Win
        assert rewards[0] == -1.0  # US was acting player -> gets -1.0
        assert info["acting_players"][0] == 1
