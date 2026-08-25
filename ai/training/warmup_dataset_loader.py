import gzip
import json
import torch
import numpy as np
from typing import Iterator, Tuple, List, Dict, Any, Optional
import ts_engine as ts

class WarmupDataset:
    """
    Forward-compatible Warmup Dataset Loader.
    Reads stored action traces from warmup games and dynamically re-extracts
    observation tensors using the current C++ engine state machine and observation code.
    """
    def __init__(self, filepath: str = 'data/warmup_5k_games.jsonl.gz'):
        self.filepath = filepath

    def stream_transitions(self, max_games: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray, int, float, float]]:
        """
        Streams (observation, action_mask, action_id, win_target, vp_target) tuples.
        Replays the actual state machine on-the-fly, ensuring full compatibility
        with future changes to Observation::extract or feature representations.
        """
        game_count = 0
        with gzip.open(self.filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if max_games is not None and game_count >= max_games:
                    break
                game = json.loads(line)
                st = ts.GameState()
                ts.Engine.init_game(st, game['seed'])
                winner = game.get('winner', 'DRAW')
                final_vp = game.get('final_vp', 0)
                
                for a in game['actions']:
                    p = st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE else st.phasing_player
                    obs = np.array(ts.extract_observation(st, p), copy=True)
                    mask = np.array(ts.get_flat_action_mask(st), copy=True)
                    
                    # Compute canonical perspective returns
                    sign = 1.0 if (p == ts.Player.US or p == 1) else -1.0
                    win_ret = (1.0 if winner == 'US' else (-1.0 if winner == 'USSR' else 0.0)) * sign
                    vp_ret = (final_vp / 20.0) * sign
                    
                    flat_act = a['flat_action']
                    yield obs, mask, flat_act, win_ret, vp_ret
                    
                    ma = ts.decode_flat_action(st, flat_act)
                    ts.Engine.step(st, ma)
                    
                game_count += 1

    def build_in_memory_tensors(self, max_games: Optional[int] = None, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """Extracts and loads dataset directly into PyTorch tensors."""
        all_obs = []
        all_masks = []
        all_acts = []
        all_win_rets = []
        all_vp_rets = []
        
        for obs, mask, act, w_ret, vp_ret in self.stream_transitions(max_games=max_games):
            all_obs.append(obs)
            all_masks.append(mask)
            all_acts.append(act)
            all_win_rets.append(w_ret)
            all_vp_rets.append(vp_ret)
            
        return {
            'observations': torch.from_numpy(np.array(all_obs, dtype=np.float32)).to(device),
            'action_masks': torch.from_numpy(np.array(all_masks, dtype=np.uint8)).to(device),
            'actions': torch.tensor(all_acts, dtype=torch.long).to(device),
            'win_targets': torch.tensor(all_win_rets, dtype=torch.float32).to(device),
            'vp_targets': torch.tensor(all_vp_rets, dtype=torch.float32).to(device),
        }
