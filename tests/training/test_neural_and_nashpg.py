"""Comprehensive Unit and Integration Tests for Twilight Struggle Neural Network & NashPG RL."""

import pytest
import numpy as np
import torch

import ts_engine as ts
from bindings import ActionEncoder, TsSingleEnv, TsVectorizedEnv
from ai.models import ColdWarNet, create_coldwar_net, ColdWarNetV2, create_coldwar_net_v2
from ai.training import RolloutBuffer, BehavioralCloningTrainer, NashPGTrainer
from tools.lib import TournamentEvaluator, NeuralAgent, RandomAgent
from bot.neural_bot import NeuralBot


class TestActionEncoderAndMasks:
    def test_flat_action_space_bounds(self):
        assert ActionEncoder.FLAT_ACTION_SIZE == 212
        assert ActionEncoder.CONFIRM_DONE_INDEX == 211

    def test_initial_state_mask_and_codec_bijection(self):
        state = ts.GameState()
        ts.Engine.init_game(state, 42)

        mask = ActionEncoder.get_legal_mask(state)
        assert mask.shape == (212,)
        assert mask.dtype == np.uint8
        assert np.sum(mask) > 0

        legal_indices = ActionEncoder.get_legal_indices(state)
        assert len(legal_indices) > 0

        for idx in legal_indices:
            action = ActionEncoder.decode(state, idx)
            encoded = ActionEncoder.encode(state, action)
            assert encoded == idx, f"Action bijection failed for index {idx} -> decoded {action} -> encoded {encoded}"
            name = ActionEncoder.get_action_name(state, idx)
            assert len(name) > 0

    def test_all_decision_types_decoding(self):
        state = ts.GameState()
        ts.Engine.init_game(state, 100)

        # Card action
        a_card = ActionEncoder.decode(state, 10) # Card #11
        assert a_card.decision_type == ts.DecisionType.SELECT_CARD
        assert a_card.primary_id == 11

        # Play mode action
        a_mode = ActionEncoder.decode(state, 110 + int(ts.PlayMode.OPS))
        assert a_mode.decision_type == ts.DecisionType.SELECT_PLAY_MODE
        assert a_mode.primary_id == int(ts.PlayMode.OPS)

        # Timing branch
        a_timing = ActionEncoder.decode(state, 114 + int(ts.TimingBranch.OPS_FIRST))
        assert a_timing.decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH
        assert a_timing.primary_id == int(ts.TimingBranch.OPS_FIRST)

        # Op mode
        a_op = ActionEncoder.decode(state, 116 + int(ts.OpMode.COUP))
        assert a_op.decision_type == ts.DecisionType.SELECT_OP_MODE
        assert a_op.primary_id == int(ts.OpMode.COUP)

        # Country node
        a_node = ActionEncoder.decode(state, 119 + 25) # Country #25
        assert a_node.decision_type == ts.DecisionType.POINT_NODE
        assert a_node.primary_id == 25

        # Branch
        a_branch = ActionEncoder.decode(state, 203 + 3) # Branch #3
        assert a_branch.decision_type == ts.DecisionType.CHOOSE_BRANCH
        assert a_branch.primary_id == 3

        # Confirm Done
        a_done = ActionEncoder.decode(state, 211)
        assert a_done.is_confirm_done() or a_done.primary_id == 255 or a_done.primary_id == 0


class TestObservationExtraction:
    def test_observation_dimensions_and_ranges(self):
        state = ts.GameState()
        ts.Engine.init_game(state, 777)

        for p in [ts.Player.US, ts.Player.USSR]:
            obs = ts.extract_observation(state, p)
            assert obs.shape == (int(ts.OBS_SIZE),)
            assert obs.dtype == np.float32
            assert not np.isnan(obs).any(), "Observation must not contain NaNs"
            assert not np.isinf(obs).any(), "Observation must not contain Infs"
            assert np.all(obs >= -2.0) and np.all(obs <= 2.0), f"Observation values out of expected range: min={obs.min()}, max={obs.max()}"

    def test_observation_hides_opponent_hand(self):
        state = ts.GameState()
        ts.Engine.init_game(state, 42)

        # USSR holds card 5, US holds card 10
        state.set_card_location(5, ts.hand_of(ts.Player.USSR))
        state.set_card_location(10, ts.hand_of(ts.Player.US))

        # US perspective
        obs_us = ts.extract_observation(state, ts.Player.US)
        board, features = 84 * 26, 14
        off_10 = board + (10 - 1) * features
        off_5 = board + (5 - 1) * features
        assert obs_us[off_10 + 1] == 1.0  # MY_HAND
        assert obs_us[off_5 + 0] == 1.0   # Hidden: folded into DRAW_DECK/UNKNOWN (slot 0)
        assert obs_us[off_5 + 2] == 0.0   # Slot 2 must be 0
        assert obs_us[board + 110 * features + 70] > 0.0  # Public opponent hand count

        # USSR perspective
        obs_ussr = ts.extract_observation(state, ts.Player.USSR)
        assert obs_ussr[off_5 + 1] == 1.0   # MY_HAND
        assert obs_ussr[off_10 + 0] == 1.0  # Hidden: folded into DRAW_DECK/UNKNOWN (slot 0)
        assert obs_ussr[off_10 + 2] == 0.0  # Slot 2 must be 0


class TestColdWarNet:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_pass_and_masking(self, device):
        model = create_coldwar_net(device)
        B = 4
        dummy_obs = torch.randn(B, int(ts.OBS_SIZE), device=device)
        dummy_mask = torch.zeros(B, 212, dtype=torch.uint8, device=device)
        dummy_mask[:, [0, 10, 110, 119, 211]] = 1

        logits, v_win, v_vp = model(dummy_obs, dummy_mask)
        assert logits.shape == (B, 212)
        assert v_win.shape == (B, 1)
        assert v_vp.shape == (B, 1)
        assert torch.all(v_win >= -1.0) and torch.all(v_win <= 1.0)

        # Illegal actions must have exact 0 probability after softmax
        probs = torch.softmax(logits, dim=-1)
        illegal = (dummy_mask == 0)
        assert torch.all(probs[illegal] == 0.0)

    def test_sample_and_evaluate_actions(self, device):
        model = create_coldwar_net(device)
        model.eval()
        B = 8
        dummy_obs = torch.randn(B, int(ts.OBS_SIZE), device=device)
        dummy_mask = torch.zeros(B, 212, dtype=torch.uint8, device=device)
        dummy_mask[:, [5, 12, 110, 211]] = 1

        actions, log_probs, v_win, v_vp, entropy = model.sample_action(dummy_obs, dummy_mask)
        assert actions.shape == (B,)
        assert log_probs.shape == (B,)
        assert v_win.shape == (B,)
        assert v_vp.shape == (B,)
        assert entropy.shape == (B,)

        for i in range(B):
            assert dummy_mask[i, actions[i]] == 1, "Sampled action must be legal"

        eval_lp, eval_ent, eval_vw, eval_vv = model.evaluate_actions(dummy_obs, dummy_mask, actions)
        assert torch.allclose(eval_lp, log_probs, atol=1e-5)
        assert torch.allclose(eval_ent, entropy, atol=1e-5)


class TestVectorizedEnvironment:
    def test_vectorized_env_rollout(self):
        num_envs = 16
        env = TsVectorizedEnv(num_envs=num_envs, base_seed=42)
        obs, masks, _ = env.reset_all()

        assert obs.shape == (num_envs, int(ts.OBS_SIZE))
        assert masks.shape == (num_envs, 212)

        for _ in range(20):
            actions = [int(np.random.choice(np.where(masks[i] > 0)[0])) for i in range(num_envs)]
            obs, masks, rewards, dones, info = env.step(actions)
            assert obs.shape == (num_envs, int(ts.OBS_SIZE))
            assert masks.shape == (num_envs, 212)
            assert rewards.shape == (num_envs,)
            assert dones.shape == (num_envs,)

    def test_vectorized_env_auto_reset_preserves_terminal_vp(self):
        num_envs = 4
        env = TsVectorizedEnv(num_envs=num_envs, base_seed=42, auto_reset=True)
        env.reset_all()

        st = env.runner.get_state(0)
        # Step through setup to AR1
        for _ in range(50):
            if st.current_phase == ts.Phase.ACTION_ROUND:
                break
            mask = ts.get_flat_action_mask(st)
            legal = [i for i, m in enumerate(mask) if m == 1]
            if not legal:
                break
            ts.Engine.step_flat(st, legal[0])

        st.defcon = 2
        st.phasing_player = ts.Player.USSR
        st.ctx().decision_player = ts.Player.USSR
        st.set_country(17, 2, 0) # Iran has US influence -> BG coup target

        st.set_card_location(5, ts.hand_of(ts.Player.USSR)) # 5op card
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0))  # Ops
        if st.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0))
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1, 0, 0))    # Coup

        # Step coup on Iran (primary_id=17 -> flat action 182) in env 0, dummy actions for others
        actions = [182]
        for e in range(1, num_envs):
            m = env.runner.get_action_masks()[e]
            legal = [i for i, val in enumerate(m) if val == 1]
            actions.append(legal[0] if legal else 211)

        _, _, rewards, dones, info = env.step(actions)
        assert bool(dones[0]) is True
        # info['victory_points'][0] must preserve the terminal VP (+20 for US win on USSR DEFCON suicide)
        # instead of 0 from the post-reset new game
        assert int(info["victory_points"][0]) == 20
        # Post-reset game state in runner is reset to 0
        assert int(env.runner.get_victory_points()[0]) == 0


class TestTrainingPipelines:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_behavioral_cloning_training_step(self, device):
        model = create_coldwar_net(device)
        trainer = BehavioralCloningTrainer(model, device=device, lr=1e-3)
        dataset = trainer.generate_demonstration_dataset(num_games=5, max_steps_per_game=50)
        history = trainer.train(dataset, epochs=2, batch_size=32)
        assert len(history["loss"]) == 2
        assert history["loss"][-1] < history["loss"][0]

    def test_nashpg_training_iterations(self, device):
        model = create_coldwar_net(device)
        trainer = NashPGTrainer(
            active_net=model,
            num_envs=16,
            buffer_size=32,
            lr=3e-4,
            eta=0.1,
            ref_update_freq=500,
            device=device,
        )
        history = [trainer.train_iteration() for _ in range(2)]
        assert len(history) == 2
        assert "loss" in history[0]
        assert "policy_loss" in history[0]
        assert "kl_div" in history[0]
        assert "val_loss" in history[0]


class TestNeuralBotAndArena:
    def test_neural_bot_selection(self):
        # The bot plays from the engine's own observation and nothing else: the Python
        # reconstruction it used to fall back on reproduced a retired layout, and never
        # reproduced even that exactly.
        import base64
        st = ts.GameState()
        ts.Engine.init_game(st, 4242)
        engine_obs = np.asarray(ts.extract_observation(st, ts.Player.USSR), dtype=np.float32)
        state_dict = {
            "turn": 1,
            "action_round": 0,
            "current_phase": 0,
            "defcon": 5,
            "victory_points": 0,
            "countries": [{"id": 14, "us_influence": 0, "ussr_influence": 3}],
            "observation_b64": base64.b64encode(engine_obs.tobytes()).decode("ascii"),
        }
        legal_actions = {
            "decision_type": int(ts.DecisionType.POINT_NODE),
            "valid_ids": [14, 15, 12],
            "allow_early_stop": False,
        }

        bot = NeuralBot(role="USSR", device="cpu")
        action = bot.select_action(state_dict, legal_actions)
        assert action is not None
        assert action["decision_type"] == int(ts.DecisionType.POINT_NODE)
        assert action["primary_id"] in [14, 15, 12]

    def test_tournament_evaluator_matchup(self):
        model = create_coldwar_net("cpu")
        res = TournamentEvaluator.play_matchup(NeuralAgent(model, name="TestNeural"), RandomAgent(), games_per_side=2)
        assert res["total_games"] == 4
        assert res["a_wins"] + res["b_wins"] + res["draws"] == 4


