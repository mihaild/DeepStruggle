"""Comprehensive Credit Assignment and Endgame Reward Propagation Test Suite.

Validates that all game-ending triggers (DEFCON suicide, opponent events, Cuban Missile Crisis,
held scoring cards, Mil Ops +/-20 VP, Event +/-20 VP, and Final Scoring) correctly assign
rewards and propagate negative advantages to losing actions and positive advantages to winning actions.
"""

import pytest
import numpy as np
import torch
import ts_engine as ts
from ai.rewards.reward_calculator import ZeroSumTerminalReward, ShapedZeroSumReward, BlunderAwareRewardCalculator, UsefulActionsReward
from bindings.ts_env import TsVectorizedEnv
from ai.training.rollout_buffer import RolloutBuffer
from bindings.action_encoder import ActionEncoder


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
        st.set_card_location(5, ts.hand_of(ts.Player.USSR))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))
        # P17: one resolution node. OPS_COUP on an opponent card is the
        # ops-first branch the retired CHOOSE_TIMING_BRANCH used to select.
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE,
                                          int(ts.Resolution.OPS_COUP), 0, 0))

        # Coup Panama (70). Written as the offset plus the country, not as the sum:
        # the sum was 189 before the repack and is a different country now.
        _, _, rewards, dones, info = env.step([ActionEncoder.NODE_OFFSET + 70])

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

        # A US coup needs USSR influence in the target. Panama has none after setup, so COUP was
        # not a legal op mode at all -- the sibling test above works only because it is the USSR
        # couping, and Panama does carry US influence. Engine::step did not check op-mode legality
        # before it validated against the mask, so this passed while couping an untouchable
        # country.
        st.set_country(70, 1, 2)

        st.set_card_location(4, ts.hand_of(ts.Player.US))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 4, 0, 0))
        # P17: one resolution node. OPS_COUP on an opponent card is the
        # ops-first branch the retired CHOOSE_TIMING_BRANCH used to select.
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE,
                                          int(ts.Resolution.OPS_COUP), 0, 0))

        # Coup Panama (70). Written as the offset plus the country, not as the sum:
        # the sum was 189 before the repack and is a different country now.
        _, _, rewards, dones, info = env.step([ActionEncoder.NODE_OFFSET + 70])

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
        st.set_card_location(2, ts.hand_of(ts.Player.US))
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
        st.set_card_location(100, ts.hand_of(ts.Player.US))
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
        st.set_card_location(2, ts.hand_of(ts.Player.USSR))
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
        # Iran is country 25, not 17 -- 17 is Hungary, which is in Europe, where a coup is
        # forbidden at DEFCON 4. This test couped Hungary believing it was couping Iran, and
        # passed only because Engine::step did not validate coup targets while the mask did.
        st.set_country(25, 2, 0)

        st.phasing_player = ts.Player.USSR
        st.ctx().decision_player = ts.Player.USSR

        st.set_card_location(5, ts.hand_of(ts.Player.USSR))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))
        # P17: one resolution node. OPS_COUP on an opponent card is the
        # ops-first branch the retired CHOOSE_TIMING_BRANCH used to select.
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE,
                                          int(ts.Resolution.OPS_COUP), 0, 0))

        # Coup Iran (25).
        _, _, rewards, dones, info = env.step([ActionEncoder.NODE_OFFSET + 25])

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

        st.set_card_location(4, ts.hand_of(ts.Player.US))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 4, 0, 0))
        # P17: one resolution node. OPS_COUP on an opponent card is the
        # ops-first branch the retired CHOOSE_TIMING_BRANCH used to select.
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE,
                                          int(ts.Resolution.OPS_COUP), 0, 0))

        # Coup Panama (70). Written as the offset plus the country, not as the sum:
        # the sum was 189 before the repack and is a different country now.
        _, _, rewards, dones, info = env.step([ActionEncoder.NODE_OFFSET + 70])

        assert dones[0] is True or dones[0] == 1
        assert st.victory_points == -20  # USSR Win
        assert rewards[0] == -1.0  # US was acting player -> gets -1.0
        assert info["acting_players"][0] == 1


class TestBlunderAwareRewardCalculator:
    """Verifies BlunderAwareRewardCalculator and Spurious Reward Shielding."""

    def test_held_scoring_card_blunder_reward_shielding(self):
        calc = BlunderAwareRewardCalculator()
        st = ts.GameState()
        ts.Engine.init_game(st, 100)
        # Clear all scoring cards so only the intended player holds one
        for card_idx in range(1, 111):
            if ts.CardData.get_card_info(card_idx).get("is_scoring"):
                st.set_card_location(card_idx, ts.CardLocation.DISCARD_PILE)
        st.set_card_location(1, ts.hand_of(ts.Player.USSR))  # Only USSR holds Asia Scoring
        st.turn = 3
        st.action_round = 7
        st.current_phase = ts.Phase.GAME_OVER

        # If USSR is acting player when game ends due to held scoring:
        r_ussr = calc.compute_step_rewards(
            acting_players=np.array([-1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),  # US won
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r_ussr[0] == -1.0, f"Blundering loser (USSR) must receive -1.0 penalty, got {r_ussr[0]}"

        # If US is acting player when only USSR held scoring card:
        r_us = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),  # US won
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r_us[0] == 0.0, f"Winner (US) not holding scoring must be shielded with 0.0 reward, got {r_us[0]}"

        # If BOTH players hold scoring cards at turn end: each side holding scoring receives -1.0 penalty
        st.set_card_location(2, ts.hand_of(ts.Player.US))  # US also holds Europe Scoring
        r_us_both = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        r_ussr_both = calc.compute_step_rewards(
            acting_players=np.array([-1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r_us_both[0] == -1.0, f"US holding scoring card when both hold must receive -1.0, got {r_us_both[0]}"
        assert r_ussr_both[0] == -1.0, f"USSR holding scoring card when both hold must receive -1.0, got {r_ussr_both[0]}"

    def test_voluntary_defcon_coup_suicide_reward_shielding(self):
        calc = BlunderAwareRewardCalculator()
        st = ts.GameState()
        ts.Engine.init_game(st, 200)
        st.defcon = 1  # DEFCON 1 Nuclear suicide
        st.phasing_player = ts.Player.USSR

        # USSR commits unprovoked suicide on USSR turn (acting_player == phasing_player) -> -1.0
        r_ussr = calc.compute_step_rewards(
            acting_players=np.array([-1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),  # US won
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r_ussr[0] == -1.0, "Suiciding player must receive -1.0 penalty"

        # US commits unprovoked suicide on US turn (acting_player == phasing_player) -> -1.0
        st.phasing_player = ts.Player.US
        r_us = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([-1.0], dtype=np.float32),  # USSR won
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r_us[0] == -1.0, "US suiciding on US turn must receive -1.0 penalty"

    def test_strategic_win_preserves_full_zero_sum(self):
        calc = BlunderAwareRewardCalculator()
        st = ts.GameState()
        ts.Engine.init_game(st, 300)
        st.defcon = 3
        st.victory_points = 20  # +20 VP Sudden Death

        # US won by strategic 20 VP -> US gets +1.0, USSR gets -1.0
        r_us = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),
            prev_victory_points=np.array([18], dtype=np.int8),
            curr_victory_points=np.array([20], dtype=np.int8),
            states=[st],
        )
        assert r_us[0] == 1.0, "Strategic winner must receive +1.0"

        r_ussr = calc.compute_step_rewards(
            acting_players=np.array([-1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),
            prev_victory_points=np.array([18], dtype=np.int8),
            curr_victory_points=np.array([20], dtype=np.int8),
            states=[st],
        )
        assert r_ussr[0] == -1.0, "Strategic loser must receive -1.0"


class TestTurnBoundaryCreditSlicing:
    """Verifies that GAE credit slicing prevents turn T blunders from corrupting turns 1..T-1."""

    def test_turn_boundary_gae_slicing(self):
        dev = "cpu"
        buf = RolloutBuffer(buffer_size=4, num_envs=1, obs_dim=10, action_dim=5, device=dev)

        # Step 0: Turn 1 (Good move)
        # Step 1: Turn 2 (Good move)
        # Step 2: Turn 3 AR1 (Turn 3 decision)
        # Step 3: Turn 3 AR2 (Held scoring blunder -> penalty -1.0, done=True)
        obs = np.zeros((1, 10), dtype=np.float32)
        mask = np.ones((1, 5), dtype=np.uint8)
        act = np.zeros(1, dtype=np.int64)
        lp = torch.zeros(1)
        v_win = torch.zeros(1)
        v_vp = torch.zeros(1)
        player = np.array([-1], dtype=np.int8)  # USSR

        # Add steps with turn information
        buf.add(obs, mask, act, lp, np.array([0.0]), np.array([False]), v_win, v_vp, player, turns=np.array([1]))
        buf.add(obs, mask, act, lp, np.array([0.0]), np.array([False]), v_win, v_vp, player, turns=np.array([2]))
        buf.add(obs, mask, act, lp, np.array([0.0]), np.array([False]), v_win, v_vp, player, turns=np.array([3]))
        buf.add(obs, mask, act, lp, np.array([-1.0]), np.array([True]), v_win, v_vp, player, turns=np.array([3]))

        # Compute GAE with slice_turn_boundaries=True
        buf.compute_gae(
            last_v_win=torch.zeros(1),
            last_v_vp=torch.zeros(1),
            last_dones=torch.tensor([True]),
            last_players=torch.tensor([-1]),
            gamma=1.0,
            gae_lambda=1.0,
            slice_turn_boundaries=True,
        )

        # Raw advantages before batch normalization
        # In Turn 3 (steps 2 and 3), penalty is present:
        # Step 3: delta = -1.0, adv = -1.0
        # Step 2: delta = 0.0, adv = -1.0 (propagated within Turn 3)
        # Across Turn 2/3 boundary (Step 1): adv sliced! delta = 0, next adv masked = 0.
        # Step 0 (Turn 1): adv = 0.
        # Check raw returns_win:
        assert buf.returns_win[3, 0].item() == -1.0
        assert buf.returns_win[2, 0].item() == -1.0
        assert buf.returns_win[1, 0].item() == 0.0, f"Turn 2 return must be 0.0 (sliced), got {buf.returns_win[1, 0].item()}"
        assert buf.returns_win[0, 0].item() == 0.0, f"Turn 1 return must be 0.0 (sliced), got {buf.returns_win[0, 0].item()}"

    def test_provoked_defcon_suicide_reward_propagation(self):
        """Tests that in an Event Trap (provoked DEFCON suicide), the winner gets +1.0

        and zero-sum alternating GAE propagates -1.0 penalty to the trapped loser.
        """
        calc = BlunderAwareRewardCalculator()
        st = ts.GameState()
        ts.Engine.init_game(st, 300)
        st.defcon = 1
        st.phasing_player = ts.Player.USSR  # USSR turn, but US acts -> provoked suicide!

        # US is acting player who executed the coup, US won
        r_us = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([20], dtype=np.int8),
            states=[st],
        )
        assert r_us[0] == 1.0, f"Acting winner (US) executing event trap coup must receive +1.0, got {r_us[0]}"

        # In GAE buffer: Step t-1 (USSR played dangerous card), Step t (US executed coup)
        buf = RolloutBuffer(buffer_size=2, num_envs=1, obs_dim=10, action_dim=5, device="cpu")
        obs = np.zeros((1, 10), dtype=np.float32)
        mask = np.ones((1, 5), dtype=np.uint8)
        act = np.zeros(1, dtype=np.int64)
        lp = torch.zeros(1)
        v_win = torch.zeros(1)
        v_vp = torch.zeros(1)

        # Step 0: USSR (-1) played CIA Created (non-terminal, r=0.0)
        buf.add(obs, mask, act, lp, np.array([0.0]), np.array([False]), v_win, v_vp, np.array([-1], dtype=np.int8))
        # Step 1: US (+1) executed coup (terminal, r=+1.0)
        buf.add(obs, mask, act, lp, np.array([1.0]), np.array([True]), v_win, v_vp, np.array([1], dtype=np.int8))

        buf.compute_gae(
            last_v_win=torch.zeros(1),
            last_v_vp=torch.zeros(1),
            last_dones=torch.tensor([True]),
            last_players=torch.tensor([1]),
            gamma=1.0,
            gae_lambda=1.0,
        )

        assert buf.returns_win[1, 0].item() == 1.0, "US return must be +1.0"
        assert buf.returns_win[0, 0].item() == -1.0, f"USSR return must be -1.0 (penalized for playing trap card), got {buf.returns_win[0, 0].item()}"


class TestHeadlineResolutionOrder:
    """Verifies official Twilight Struggle Rule 4.3 headline resolution order:

    1. Higher printed Ops card resolves first.
    2. Ties resolve US card first.
    3. If first headline terminates the game, second headline is not resolved.
    """

    def test_higher_ops_resolves_first(self):
        st = ts.GameState()
        ts.Engine.init_game(st, 42)
        st.turn = 7
        st.defcon = 3
        st.current_phase = ts.Phase.HEADLINE

        # Card 50 (We Will Bury You) = 4 Ops; Card 4 (Duck and Cover) = 3 Ops
        st.set_card_location(4, ts.hand_of(ts.Player.US))
        st.set_card_location(50, ts.hand_of(ts.Player.USSR))

        # US selects Duck and Cover (3 Ops)
        st.ctx().decision_player = ts.Player.US
        st.ctx().decision_type = ts.DecisionType.SELECT_CARD
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 4, 0, 0))

        # USSR selects We Will Bury You (4 Ops)
        st.ctx().decision_player = ts.Player.USSR
        st.ctx().decision_type = ts.DecisionType.SELECT_CARD
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 50, 0, 0))

        # WWBY (4 Ops) resolves first -> DEFCON 3 -> 2
        # Duck & Cover (3 Ops) resolves second on US phasing -> DEFCON 2 -> 1 -> US loses!
        assert st.defcon == 1, f"DEFCON must reach 1, got {st.defcon}"
        assert st.current_phase == ts.Phase.GAME_OVER, "Game must be over"
        assert st.victory_points == -20, f"USSR must win (-20 VP), got {st.victory_points}"
        assert ts.Engine.get_terminal_utility(st) == -1.0, "Terminal utility must be -1.0 (USSR Win)"

    def test_tie_in_ops_resolves_us_first(self):
        st = ts.GameState()
        ts.Engine.init_game(st, 42)
        st.turn = 3
        st.defcon = 3
        st.current_phase = ts.Phase.HEADLINE

        # Card 4 (Duck and Cover) = 3 Ops; Card 8 (Fidel) = 3 Ops
        st.set_card_location(4, ts.hand_of(ts.Player.US))
        st.set_card_location(8, ts.hand_of(ts.Player.USSR))

        # US selects Duck and Cover (3 Ops)
        st.ctx().decision_player = ts.Player.US
        st.ctx().decision_type = ts.DecisionType.SELECT_CARD
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 4, 0, 0))

        # USSR selects Fidel (3 Ops)
        st.ctx().decision_player = ts.Player.USSR
        st.ctx().decision_type = ts.DecisionType.SELECT_CARD
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 8, 0, 0))

        # On tie, US card must resolve first
        assert st.headline_first_card == 4, f"US card (4) must resolve first on tie, got {st.headline_first_card}"


class TestUsefulActionsRewardCalculator:
    """Verifies UsefulActionsReward potential shaping and BlunderAware inheritance."""

    def test_vp_potential_component(self):
        calc = UsefulActionsReward(potential_scale=1.0)
        st = ts.GameState()
        ts.Engine.init_game(st, 100)

        base_us = calc.compute_potential(st)
        assert -1.0 <= base_us <= 1.0

        # Increase VP by +2 for US
        st.victory_points += 2
        pot_us = calc.compute_potential(st)
        delta = pot_us - base_us
        expected_delta = (2.0 / 20.0) * 0.5  # weight 0.5
        assert abs(delta - expected_delta) < 1e-4, f"VP delta must be {expected_delta}, got {delta}"

    def test_battleground_control_component(self):
        calc = UsefulActionsReward(potential_scale=1.0)
        st = ts.GameState()
        ts.Engine.init_game(st, 100)

        base_us = calc.compute_potential(st, 1)
        # Give US control of South Korea (country 36, stability 3)
        st.set_country(36, 5, 0)
        pot_us = calc.compute_potential(st, 1)
        assert pot_us > base_us, "Gaining battleground control must increase potential"

    def test_country_access_component(self):
        calc = UsefulActionsReward(potential_scale=1.0)
        st = ts.GameState()
        ts.Engine.init_game(st, 100)

        base_us = calc.compute_potential(st, 1)
        # Give US influence in remote country Angola (57)
        st.set_country(57, 1, 0)
        pot_us = calc.compute_potential(st, 1)
        assert pot_us > base_us, "Expanding country access must increase potential"

    def test_unscored_regions_component_and_discard_behavior(self):
        calc = UsefulActionsReward(potential_scale=1.0)
        st = ts.GameState()
        ts.Engine.init_game(st, 100)

        # Discarding Asia Scoring card (1) means Asia is no longer unscored
        p_unscored = calc.compute_potential(st, 1)
        st.set_card_location(1, ts.CardLocation.DISCARD_PILE)
        p_scored = calc.compute_potential(st, 1)
        assert p_unscored != p_scored, "Discarding scoring card must remove region from unscored potential"

    def test_useful_actions_reward_inherits_blunder_aware_terminal(self):
        calc = UsefulActionsReward(potential_scale=1.0)
        st = ts.GameState()
        ts.Engine.init_game(st, 200)
        st.defcon = 1  # DEFCON suicide
        st.phasing_player = ts.Player.USSR

        # USSR commits unprovoked suicide on USSR turn -> receives -1.0
        r_ussr = calc.compute_step_rewards(
            acting_players=np.array([-1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r_ussr[0] == -1.0, "Blundering player on terminal step must receive -1.0"

        # US is non-phasing player provoking USSR suicide -> receives +1.0
        r_us = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([True]),
            terminal_utilities=np.array([1.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r_us[0] == 1.0, "Acting player provoking opponent DEFCON suicide must receive +1.0"

    def test_useful_actions_reward_step_potential_delta(self):
        calc = UsefulActionsReward(potential_scale=1.0)
        st = ts.GameState()
        ts.Engine.init_game(st, 300)

        # Step 0: init
        r0 = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([False]),
            terminal_utilities=np.array([0.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r0[0] == 0.0, "Initial step without delta has 0.0 reward"

        # Step 1: US captures battleground
        st.set_country(36, 5, 0)
        r1 = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([False]),
            terminal_utilities=np.array([0.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert r1[0] > 0.0, f"Capturing battleground must grant positive shaped reward, got {r1[0]}"

    def test_vectorized_runner_potentials(self):
        runner = ts.VectorizedBatchRunner(8, 42)
        acting_players = [1, -1, 1, -1, 1, -1, 1, -1]
        pots = runner.compute_useful_actions_potentials(acting_players)
        assert len(pots) == 8
        calc = UsefulActionsReward()
        for i in range(8):
            st = runner.get_state(i)
            expected = calc.compute_potential(st)
            assert abs(pots[i] - expected) < 1e-5, f"Vectorized potential at env {i} must match scalar calculation"


class TestHeldScoringAndPotentialFixes:
    """Verifies Bug 1 and Bug 2 specific fixes, engine detection, and GAE credit slicing."""

    def test_engine_held_scoring_detection(self):
        st = ts.GameState()
        ts.Engine.init_game(st, 42)
        assert not ts.Engine.is_held_scoring_game_over(st)

        # Clear scoring cards
        for c in range(1, 111):
            if ts.CardData.get_card_info(c).get("is_scoring"):
                st.set_card_location(c, ts.CardLocation.DISCARD_PILE)

        # Put Asia Scoring in USSR hand
        st.set_card_location(1, ts.hand_of(ts.Player.USSR))
        assert ts.Engine.has_held_scoring_card(st, ts.Player.USSR)
        assert not ts.Engine.has_held_scoring_card(st, ts.Player.US)

        # Mid-turn game over (e.g. Europe Control during AR 2) -> NOT held scoring game over!
        st.current_phase = ts.Phase.GAME_OVER
        st.turn = 1
        st.action_round = 2
        st.victory_points = 20
        assert not ts.Engine.is_held_scoring_game_over(st)
        assert not ts.Engine.is_held_scoring_loss(st, ts.Player.USSR)

        # Turn-end game over (action_round > max_ar, turn 1 max_ar = 6) -> IS held scoring game over!
        st.action_round = 7
        assert ts.Engine.is_held_scoring_game_over(st)
        assert ts.Engine.is_held_scoring_loss(st, ts.Player.USSR)
        assert not ts.Engine.is_held_scoring_loss(st, ts.Player.US)

    def test_both_hold_scoring_cards_detection(self):
        st = ts.GameState()
        ts.Engine.init_game(st, 42)
        st.set_card_location(1, ts.hand_of(ts.Player.USSR))  # Asia
        st.set_card_location(2, ts.hand_of(ts.Player.US))    # Europe
        st.current_phase = ts.Phase.GAME_OVER
        st.turn = 4
        st.action_round = 8  # Mid/late war ends after AR 7
        assert ts.Engine.is_held_scoring_game_over(st)
        assert ts.Engine.is_held_scoring_loss(st, ts.Player.US)
        assert ts.Engine.is_held_scoring_loss(st, ts.Player.USSR)

    def test_held_scoring_gae_turn_slicing_and_opponent_shielding(self):
        from ai.training.rollout_buffer import RolloutBuffer
        device = torch.device("cpu")
        buffer = RolloutBuffer(buffer_size=6, num_envs=1, device=device)

        obs = np.zeros((1, int(ts.OBS_SIZE)), dtype=np.float32)
        mask = np.zeros((1, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.uint8)
        mask[0, 0] = 1

        steps = [
            (2, 1, 0.0, False, False, False),
            (2, -1, 0.0, False, False, False),
            (3, 1, 0.0, False, False, False),
            (3, -1, 0.0, False, False, False),
            (3, 1, 0.0, False, False, False),
            (3, 1, 0.0, True, False, True),  # USSR held scoring card blunder!
        ]

        for s_idx, (turn, player, rew, done, hs_us, hs_ussr) in enumerate(steps):
            buffer.add(
                obs=obs,
                masks=mask,
                actions=np.array([0]),
                log_probs=torch.tensor([0.0]),
                rewards=np.array([rew], dtype=np.float32),
                dones=np.array([done], dtype=bool),
                values_win=torch.tensor([0.0]),
                values_vp=torch.tensor([0.0]),
                players=np.array([player], dtype=np.int8),
                turns=np.array([turn], dtype=np.int8),
                held_scoring_us=np.array([hs_us], dtype=bool),
                held_scoring_ussr=np.array([hs_ussr], dtype=bool),
            )

        buffer.compute_gae(
            last_v_win=torch.tensor([0.0]),
            last_v_vp=torch.tensor([0.0]),
            last_dones=torch.tensor([True]),
            last_players=torch.tensor([1]),
            slice_turn_boundaries=True,
        )

        # USSR actions in Turn 3 (step 3) must receive -1.0 target return
        assert buffer.returns_win[3, 0].item() == -1.0

        # US actions in Turn 3 (steps 2, 4, 5) must receive baseline 0.0 return (shielded from opponent's blunder!)
        assert buffer.returns_win[2, 0].item() == 0.0
        assert buffer.returns_win[4, 0].item() == 0.0
        assert buffer.returns_win[5, 0].item() == 0.0

        # Steps in Turn 2 (steps 0, 1) must be sliced before Turn 3: baseline 0.0 return
        assert buffer.returns_win[0, 0].item() == 0.0
        assert buffer.returns_win[1, 0].item() == 0.0

        # Normalized advantages: USSR advantage is strongly penalized relative to shielded US
        assert buffer.advantages[3, 0].item() < -1.0
        assert buffer.advantages[2, 0].item() > buffer.advantages[3, 0].item()

    def test_potential_shaping_zero_on_turn_switches(self):
        calc = UsefulActionsReward(potential_scale=1.0)
        st = ts.GameState()
        ts.Engine.init_game(st, 100)

        # Initial step with US
        r_us = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([False]),
            terminal_utilities=np.array([0.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )

        # Next step with USSR on unchanged board
        r_ussr = calc.compute_step_rewards(
            acting_players=np.array([-1], dtype=np.int8),
            dones=np.array([False]),
            terminal_utilities=np.array([0.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert abs(r_ussr[0]) < 1e-6, f"Reward on unchanged board during player switch must be 0.0, got {r_ussr[0]}"

        # Next step with US again on unchanged board
        r_us_again = calc.compute_step_rewards(
            acting_players=np.array([1], dtype=np.int8),
            dones=np.array([False]),
            terminal_utilities=np.array([0.0], dtype=np.float32),
            prev_victory_points=np.array([0], dtype=np.int8),
            curr_victory_points=np.array([0], dtype=np.int8),
            states=[st],
        )
        assert abs(r_us_again[0]) < 1e-6, f"Reward on unchanged board must be 0.0, got {r_us_again[0]}"
