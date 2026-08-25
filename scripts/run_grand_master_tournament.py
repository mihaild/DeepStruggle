import os
import time
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import ts_engine as ts
from ai.models import ColdWarNet, create_coldwar_net
from ai.env import ActionEncoder

class GrandMasterTournament:
    def __init__(self, checkpoints: Dict[str, str], games_per_pair: int = 1000, batch_size: int = 500, temperature: float = 0.1, device_str: str = 'cuda'):
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.checkpoints = checkpoints
        self.games_per_pair = games_per_pair
        self.batch_size = batch_size
        self.temperature = temperature
        
        self.models: Dict[str, ColdWarNet] = {}
        for name, path in self.checkpoints.items():
            print(f'Loading {name} from {path}...')
            m = create_coldwar_net(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.eval()
            self.models[name] = m
            
    def play_matchup(self, ussr_name: str, us_name: str, total_games: int = 1000) -> Dict[str, Any]:
        m_ussr = self.models[ussr_name]
        m_us = self.models[us_name]
        
        ussr_wins = 0
        us_wins = 0
        draws = 0
        turn_counts = []
        step_counts = []
        
        games_played = 0
        batch_size = min(self.batch_size, total_games)
        
        t0 = time.time()
        while games_played < total_games:
            current_batch = min(batch_size, total_games - games_played)
            runner = ts.VectorizedBatchRunner(current_batch, int(time.time() * 1000) % 1_000_000_000 + games_played)
            
            completed = [False] * current_batch
            
            steps = 0
            while not all(completed) and steps < 4000:
                steps += 1
                obs_all = np.array(runner.get_observations(), copy=False)
                masks_all = np.array(runner.get_action_masks(), copy=False)
                players = runner.get_decision_players()
                terminals = runner.get_terminals()
                term_utils = runner.get_terminal_utilities()
                
                for i in range(current_batch):
                    if not completed[i] and terminals[i]:
                        completed[i] = True
                        st = runner.get_state(i)
                        turn_counts.append(st.turn)
                        step_counts.append(steps)
                        if term_utils[i] < 0:
                            ussr_wins += 1
                        elif term_utils[i] > 0:
                            us_wins += 1
                        else:
                            draws += 1
                            
                if all(completed):
                    break
                    
                active = [i for i in range(current_batch) if not completed[i]]
                actions = [0] * current_batch
                
                # USSR player is m_ussr, US player is m_us
                ussr_indices = [i for i in active if players[i] == -1]
                us_indices = [i for i in active if players[i] == 1]
                
                if ussr_indices:
                    obs_sub = torch.from_numpy(obs_all[ussr_indices]).float().to(self.device)
                    mask_sub = torch.from_numpy(masks_all[ussr_indices]).to(self.device)
                    with torch.no_grad():
                        act_t, _, _, _, _ = m_ussr.sample_action(obs_sub, mask_sub, temperature=self.temperature, deterministic=(self.temperature <= 0.05))
                    act_list = act_t.cpu().numpy().tolist()
                    for idx, env_i in enumerate(ussr_indices):
                        actions[env_i] = int(act_list[idx])
                        
                if us_indices:
                    obs_sub = torch.from_numpy(obs_all[us_indices]).float().to(self.device)
                    mask_sub = torch.from_numpy(masks_all[us_indices]).to(self.device)
                    with torch.no_grad():
                        act_t, _, _, _, _ = m_us.sample_action(obs_sub, mask_sub, temperature=self.temperature, deterministic=(self.temperature <= 0.05))
                    act_list = act_t.cpu().numpy().tolist()
                    for idx, env_i in enumerate(us_indices):
                        actions[env_i] = int(act_list[idx])
                        
                runner.step_flat_all(actions)
                
            games_played += current_batch
            
        elapsed = time.time() - t0
        ussr_rate = ussr_wins / total_games
        us_rate = us_wins / total_games
        draw_rate = draws / total_games
        
        return {
            'ussr_name': ussr_name,
            'us_name': us_name,
            'total_games': total_games,
            'ussr_wins': ussr_wins,
            'us_wins': us_wins,
            'draws': draws,
            'ussr_win_rate': ussr_rate,
            'us_win_rate': us_rate,
            'draw_rate': draw_rate,
            'avg_turn': float(np.mean(turn_counts)) if turn_counts else 0.0,
            'avg_steps': float(np.mean(step_counts)) if step_counts else 0.0,
            'elapsed_sec': elapsed,
            'speed_gps': total_games / max(elapsed, 0.001)
        }

    def run_all_pairs(self) -> Dict[str, Any]:
        names = list(self.checkpoints.keys())
        results = []
        cross_table = {u: {v: 0.0 for v in names} for u in names}
        
        total_pairs = len(names) * len(names)
        pair_idx = 0
        t_start = time.time()
        
        print(chr(10) + '='*80)
        print(f' Grand Master Tournament: {len(names)} Top Models x {len(names)} Top Models = {total_pairs} Ordered Pairs')
        print(f' {self.games_per_pair:,} Games per Ordered Pair | Total Games: {total_pairs * self.games_per_pair:,}')
        print(f' Temperature: {self.temperature} | Parallel Envs: {self.batch_size} | Device: {self.device}')
        print('='*80 + chr(10))
        
        for ussr_name in names:
            for us_name in names:
                pair_idx += 1
                print(f'[{pair_idx:2d}/{total_pairs:2d}] Simulating {self.games_per_pair:,} games: USSR [{ussr_name:13s}] vs US [{us_name:13s}]...', end='', flush=True)
                res = self.play_matchup(ussr_name, us_name, self.games_per_pair)
                results.append(res)
                cross_table[ussr_name][us_name] = res['ussr_win_rate']
                print(f" Done in {res['elapsed_sec']:4.1f}s ({res['speed_gps']:4.0f} games/s) | USSR Win: {res['ussr_win_rate']*100:5.1f}% | US Win: {res['us_win_rate']*100:5.1f}% | Avg Turn: {res['avg_turn']:.1f}")
                
        total_time = time.time() - t_start
        return {
            'names': names,
            'results': results,
            'cross_table': cross_table,
            'total_time': total_time,
            'total_games': total_pairs * self.games_per_pair
        }

def compute_bt_elo(names: List[str], results: List[Dict[str, Any]], anchor_name: str = 'Snapshot_180m', anchor_elo: float = 1954.9) -> Dict[str, float]:
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}
    
    wins = np.zeros(n, dtype=np.float64)
    opp_matrix = np.zeros((n, n), dtype=np.float64)
    
    for r in results:
        u_idx = name_to_idx[r['ussr_name']]
        v_idx = name_to_idx[r['us_name']]
        
        wins[u_idx] += r['ussr_wins'] + 0.5 * r['draws']
        wins[v_idx] += r['us_wins'] + 0.5 * r['draws']
        opp_matrix[u_idx, v_idx] += r['total_games']
        opp_matrix[v_idx, u_idx] += r['total_games']
        
    gamma = np.ones(n, dtype=np.float64)
    for _ in range(200):
        new_gamma = np.zeros(n, dtype=np.float64)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i != j and opp_matrix[i, j] > 0:
                    denom += opp_matrix[i, j] / (gamma[i] + gamma[j])
            new_gamma[i] = wins[i] / max(denom, 1e-8)
        new_gamma /= np.mean(new_gamma)
        gamma = new_gamma
        
    log_gamma = np.log(np.maximum(gamma, 1e-8))
    scale = 400.0 / np.log(10.0)
    raw_elos = log_gamma * scale
    
    anchor_idx = name_to_idx.get(anchor_name, 0)
    shift = anchor_elo - raw_elos[anchor_idx]
    
    return {name: float(raw_elos[i] + shift) for i, name in enumerate(names)}

if __name__ == '__main__':
    top_checkpoints = {
        'Snapshot_180m': 'checkpoints/run_20260825_093352/snapshot_180m.pt',
        'Snapshot_210m': 'checkpoints/run_20260825_093352/snapshot_210m.pt',
        'Snapshot_360m': 'checkpoints/run_20260825_093352/snapshot_360m.pt',
        'Snapshot_480m': 'checkpoints/run_20260825_093352/snapshot_480m.pt',
    }
    
    tourney = GrandMasterTournament(top_checkpoints, games_per_pair=1000, batch_size=500, temperature=0.1, device_str='cuda')
    data = tourney.run_all_pairs()
    
    elos = compute_bt_elo(data['names'], data['results'], anchor_name='Snapshot_180m', anchor_elo=1954.9)
    
    print(chr(10) + '='*80)
    print(' FINAL GRAND MASTER TOURNAMENT RESULTS (16,000 GAMES)')
    print('='*80)
    print(chr(10) + '=== Bradley-Terry Elo Ratings ===')
    for rank, (name, r) in enumerate(sorted(elos.items(), key=lambda x: -x[1]), 1):
        print(f'  {rank:2d} | {name:16s} | {r:7.1f} Elo')
        
    print(chr(10) + '=== Cross-Table: USSR Win Rate (Row as USSR vs Column as US) ===')
    header = 'USSR \ US      | ' + ' | '.join(f'{n:13s}' for n in data['names']) + ' | Row Avg (USSR)'
    print(header)
    print('-' * len(header))
    for u in data['names']:
        row_rates = [data['cross_table'][u][v] * 100 for v in data['names']]
        row_str = f'{u:13s} | ' + ' | '.join(f'{r:12.1f}%' for r in row_rates) + f' | {np.mean(row_rates):13.1f}%'
        print(row_str)
        
    report_path = 'checkpoints/grand_master_tournament_report.md'
    with open(report_path, 'w') as f:
        f.write('# Twilight Struggle Grand Master Tournament Report (16,000 Games)' + chr(10) + chr(10))
        f.write(f'- **Total Games**: {data["total_games"]:,} ({len(data["names"])*len(data["names"])} ordered pairs x 1,000 games)' + chr(10))
        f.write(f'- **Total Duration**: {data["total_time"]:.1f} seconds ({data["total_games"]/data["total_time"]:,.0f} games/sec)' + chr(10))
        f.write('- **Evaluation Temperature**: T = 0.1 (Crisp Tactical Play)' + chr(10) + chr(10))
        f.write('## Bradley-Terry Elo Leaderboard' + chr(10) + chr(10))
        f.write('| Rank | Model Snapshot | Elo Rating |' + chr(10) + '|:---:|:---|:---:|' + chr(10))
        for rank, (name, r) in enumerate(sorted(elos.items(), key=lambda x: -x[1]), 1):
            f.write(f'| **{rank}** | **{name}** | **{r:.1f}** |' + chr(10))
            
        f.write(chr(10) + '## Cross-Table: Win Rate Matrix (1,000 Games per Cell)' + chr(10) + chr(10))
        f.write('| USSR \ US | ' + ' | '.join(data['names']) + ' | **Row Avg (USSR Win %)** |' + chr(10))
        f.write('|:---|' + ':---:|' * len(data['names']) + ':---:|' + chr(10))
        for u in data['names']:
            row_rates = [data['cross_table'][u][v] * 100 for v in data['names']]
            f.write(f'| **{u}** | ' + ' | '.join(f'{r:.1f}%' for r in row_rates) + f' | **{np.mean(row_rates):.1f}%** |' + chr(10))
            
        f.write(chr(10) + '## Ordered Pair Matchup Details (1,000 Games Each)' + chr(10) + chr(10))
        f.write('| USSR Model | US Model | USSR Wins | US Wins | Draws | USSR Win Rate | Avg Turn | Avg Steps |' + chr(10))
        f.write('|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|' + chr(10))
        for r in data['results']:
            f.write(f"| **{r['ussr_name']}** | **{r['us_name']}** | {r['ussr_wins']} | {r['us_wins']} | {r['draws']} | **{r['ussr_win_rate']*100:.1f}%** | {r['avg_turn']:.1f} | {r['avg_steps']:.0f} |" + chr(10))
            
    print(f'Tournament report successfully written to {report_path}')
