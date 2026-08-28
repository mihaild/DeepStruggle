"""ColdWarNetV4: Deep Card Transformer, Causal History, Opponent Belief Head & Oracle Critic."""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ts_engine as ts


def build_normalized_adjacency_matrix() -> torch.Tensor:
    """Builds the 84x84 normalized adjacency matrix with self-loops from MapData."""
    adj = np.eye(84, dtype=np.float32)
    for i in range(84):
        info = ts.MapData.get_country_info(i)
        for n in info["neighbors"]:
            adj[i, n] = 1.0
            adj[n, i] = 1.0

    degrees = np.sum(adj, axis=1)
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    np.power(degrees, -0.5, where=degrees > 0, out=d_inv_sqrt)
    d_mat = np.diag(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat
    return torch.from_numpy(norm_adj.astype(np.float32))


def get_country_region_indices() -> torch.Tensor:
    """Returns the 84-element tensor containing region indices [0..5] for each country."""
    regions = np.array([ts.MapData.get_country_info(i)["region"] for i in range(84)], dtype=np.int64)
    return torch.from_numpy(regions)


class GraphConvLayer(nn.Module):
    """Batched Graph Convolution layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        out = torch.matmul(norm_adj, support) + self.bias
        return F.gelu(out)


class ResBlock(nn.Module):
    """Pre-LN Residual MLP Block."""

    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(self.ln1(x)))
        h = self.dropout(h)
        h = self.fc2(self.ln2(h))
        return x + h


class ColdWarNetV4(nn.Module):
    """ColdWarNetV4: Deep Card Transformer, Causal History, Opponent Belief Head & Oracle Critic."""

    BOARD_OFFSET = 0
    BOARD_SIZE = 84 * 28  # 2352

    CARD_OFFSET = 2352
    CARD_SIZE = 110 * 12  # 1320

    GLOBAL_OFFSET = 2352 + 1320  # 3672
    GLOBAL_SIZE = 76

    HIST_OFFSET = 3672 + 76  # 3748
    HIST_SIZE = 16 * 32  # 512

    TOTAL_OBS_SIZE = 4293
    ACTION_SPACE_SIZE = 212

    CARD_ACTION_START = 0
    CARD_ACTION_END = 110
    NODE_ACTION_START = 119
    NODE_ACTION_END = 203

    def __init__(
        self,
        hidden_dim: int = 512,
        card_dim: int = 128,
        node_dim: int = 64,
        hist_dim: int = 64,
        num_card_layers: int = 4,
        num_res_blocks: int = 4,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        self.register_buffer("norm_adj", build_normalized_adjacency_matrix())
        self.register_buffer("country_regions", get_country_region_indices())

        self.hidden_dim = hidden_dim
        self.card_dim = card_dim
        self.node_dim = node_dim
        self.hist_dim = hist_dim

        # 1. Country Spatial Graph Encoder (GCN + Region Embeddings + Self-Attention)
        self.gconv1 = GraphConvLayer(28, node_dim)
        self.gconv2 = GraphConvLayer(node_dim, node_dim)
        self.region_emb = nn.Embedding(6, node_dim)

        self.board_self_attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_attn_heads,
            dropout=0.05,
            batch_first=True,
        )
        self.board_self_ln = nn.LayerNorm(node_dim)
        self.board_proj = nn.Sequential(
            nn.Linear(node_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 2. Deep Card Transformer (Self-Attention over 110 Cards)
        self.card_feat_in = nn.Sequential(
            nn.Linear(12, card_dim),
            nn.LayerNorm(card_dim),
            nn.GELU(),
        )
        self.card_loc_emb = nn.Embedding(7, card_dim)

        card_layer = nn.TransformerEncoderLayer(
            d_model=card_dim,
            nhead=num_attn_heads,
            dim_feedforward=card_dim * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.card_transformer = nn.TransformerEncoder(card_layer, num_layers=num_card_layers, enable_nested_tensor=False)
        self.card_proj = nn.Sequential(
            nn.Linear(card_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 3. Action History Sequence Encoder (Causal Transformer over 16 MicroActions)
        self.hist_in = nn.Sequential(
            nn.Linear(32, hist_dim),
            nn.LayerNorm(hist_dim),
            nn.GELU(),
        )
        hist_layer = nn.TransformerEncoderLayer(
            d_model=hist_dim,
            nhead=2,
            dim_feedforward=hist_dim * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hist_transformer = nn.TransformerEncoder(hist_layer, num_layers=2, enable_nested_tensor=False)
        self.hist_proj = nn.Sequential(
            nn.Linear(hist_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 4. Global Scalars & Cross-Attention Fusion
        self.global_fc = nn.Sequential(
            nn.Linear(self.GLOBAL_SIZE, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        fusion_in_dim = 256 + 256 + 128 + 128  # board(256) + cards(256) + hist(128) + global(128) = 768
        self.fusion_in = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=0.05) for _ in range(num_res_blocks)])

        # 5. Dual Pointer-Generator Policy Network
        self.policy_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.modal_head = nn.Linear(hidden_dim, self.ACTION_SPACE_SIZE)
        self.card_pointer_proj = nn.Linear(hidden_dim, card_dim)
        self.node_pointer_proj = nn.Linear(hidden_dim, node_dim)

        # 6. Value Heads (Win/Loss + VP)
        self.value_head_win = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.value_head_vp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # 7. Opponent Hand Belief Head: P(Card c in Opponent Hand) for all 110 cards
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 110),
            nn.Sigmoid(),
        )

        # 8. Privileged Oracle Critic Head (Suphx Oracle Guiding)
        self.oracle_critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 110, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def _extract_inputs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b_raw = obs[:, self.BOARD_OFFSET : self.BOARD_OFFSET + self.BOARD_SIZE]
        c_raw = obs[:, self.CARD_OFFSET : self.CARD_OFFSET + self.CARD_SIZE]
        g_raw = obs[:, self.GLOBAL_OFFSET : self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
        h_raw = obs[:, self.HIST_OFFSET : self.HIST_OFFSET + self.HIST_SIZE]
        return b_raw, c_raw, g_raw, h_raw

    def forward(
        self, obs: torch.Tensor, legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing masked action logits and dual value predictions."""
        b_raw, c_raw, g_raw, h_raw = self._extract_inputs(obs)
        B = obs.size(0)

        # 1. Country Graph Encoding
        b_nodes = b_raw.view(B, 84, 28)
        norm_adj = self.norm_adj.to(obs.device)
        h_gcn1 = self.gconv1(b_nodes, norm_adj)
        h_gcn2 = self.gconv2(h_gcn1, norm_adj)
        r_emb = self.region_emb(self.country_regions.to(obs.device)).unsqueeze(0).expand(B, -1, -1)
        h_nodes = h_gcn2 + r_emb

        h_nodes_attn, _ = self.board_self_attn(h_nodes, h_nodes, h_nodes)
        h_nodes_res = self.board_self_ln(h_nodes + h_nodes_attn)

        b_mean = h_nodes_res.mean(dim=1)
        b_max = h_nodes_res.max(dim=1).values
        b_rep = self.board_proj(torch.cat([b_mean, b_max], dim=-1))

        # 2. Deep Card Transformer Encoding
        c_nodes = c_raw.view(B, 110, 12)
        loc_idx = c_nodes[:, :, 0].long().clamp(0, 6)
        c_feat = self.card_feat_in(c_nodes) + self.card_loc_emb(loc_idx)
        h_cards = self.card_transformer(c_feat)

        c_mean = h_cards.mean(dim=1)
        c_rep = self.card_proj(c_mean)

        # 3. Action History Encoding
        h_steps = h_raw.view(B, 16, 32)
        h_seq = self.hist_in(h_steps)
        h_trans = self.hist_transformer(h_seq)
        h_rep = self.hist_proj(h_trans[:, -1, :])

        # 4. Global Scalars
        g_rep = self.global_fc(g_raw)

        # 5. Fusion & ResNet Trunk
        fused = torch.cat([b_rep, c_rep, h_rep, g_rep], dim=-1)
        latent = self.fusion_in(fused)
        for block in self.res_blocks:
            latent = block(latent)

        # 6. Dual Pointer-Generator Action Logits
        pol_latent = self.policy_trunk(latent)
        base_logits = self.modal_head(pol_latent)

        # Card Pointers (0..109)
        q_card = self.card_pointer_proj(pol_latent).unsqueeze(1)
        card_scores = (q_card * h_cards).sum(dim=-1) / math.sqrt(self.card_dim)

        # Country Pointers (119..202)
        q_node = self.node_pointer_proj(pol_latent).unsqueeze(1)
        node_scores = (q_node * h_nodes_res).sum(dim=-1) / math.sqrt(self.node_dim)

        logits = base_logits.clone()
        logits[:, self.CARD_ACTION_START : self.CARD_ACTION_END] += card_scores
        logits[:, self.NODE_ACTION_START : self.NODE_ACTION_END] += node_scores

        # Apply Action Masking
        masked_logits = torch.where(
            legal_mask.bool(),
            logits,
            torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
        )

        # 7. Dual Value Predictions
        v_win = self.value_head_win(latent)
        v_vp = self.value_head_vp(latent)

        return masked_logits, v_win, v_vp

    def forward_all(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
        true_opp_cards: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Unified multi-task forward pass computing policy, values, belief, and oracle in 1 single pass."""
        b_raw, c_raw, g_raw, h_raw = self._extract_inputs(obs)
        B = obs.size(0)

        # 1. Country Graph Encoding
        b_nodes = b_raw.view(B, 84, 28)
        norm_adj = self.norm_adj.to(obs.device)
        h_gcn1 = self.gconv1(b_nodes, norm_adj)
        h_gcn2 = self.gconv2(h_gcn1, norm_adj)
        r_emb = self.region_emb(self.country_regions.to(obs.device)).unsqueeze(0).expand(B, -1, -1)
        h_nodes = h_gcn2 + r_emb

        h_nodes_attn, _ = self.board_self_attn(h_nodes, h_nodes, h_nodes)
        h_nodes_res = self.board_self_ln(h_nodes + h_nodes_attn)

        b_mean = h_nodes_res.mean(dim=1)
        b_max = h_nodes_res.max(dim=1).values
        b_rep = self.board_proj(torch.cat([b_mean, b_max], dim=-1))

        # 2. Deep Card Transformer Encoding
        c_nodes = c_raw.view(B, 110, 12)
        loc_idx = c_nodes[:, :, 0].long().clamp(0, 6)
        c_feat = self.card_feat_in(c_nodes) + self.card_loc_emb(loc_idx)
        h_cards = self.card_transformer(c_feat)

        c_mean = h_cards.mean(dim=1)
        c_rep = self.card_proj(c_mean)

        # 3. Action History Encoding
        h_steps = h_raw.view(B, 16, 32)
        h_seq = self.hist_in(h_steps)
        h_trans = self.hist_transformer(h_seq)
        h_rep = self.hist_proj(h_trans[:, -1, :])

        # 4. Global Scalars
        g_rep = self.global_fc(g_raw)

        # 5. Fusion & ResNet Trunk
        fused = torch.cat([b_rep, c_rep, h_rep, g_rep], dim=-1)
        latent = self.fusion_in(fused)
        for block in self.res_blocks:
            latent = block(latent)

        # 6. Dual Pointer-Generator Action Logits
        pol_latent = self.policy_trunk(latent)
        base_logits = self.modal_head(pol_latent)

        # Card Pointers (0..109)
        q_card = self.card_pointer_proj(pol_latent).unsqueeze(1)
        card_scores = (q_card * h_cards).sum(dim=-1) / math.sqrt(self.card_dim)

        # Country Pointers (119..202)
        q_node = self.node_pointer_proj(pol_latent).unsqueeze(1)
        node_scores = (q_node * h_nodes_res).sum(dim=-1) / math.sqrt(self.node_dim)

        logits = base_logits.clone()
        logits[:, self.CARD_ACTION_START : self.CARD_ACTION_END] += card_scores
        logits[:, self.NODE_ACTION_START : self.NODE_ACTION_END] += node_scores

        masked_logits = torch.where(
            legal_mask.bool(),
            logits,
            torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
        )

        v_win = self.value_head_win(latent)
        v_vp = self.value_head_vp(latent)
        pred_belief = self.belief_head(latent)

        v_oracle = None
        if true_opp_cards is not None:
            oracle_in = torch.cat([latent, true_opp_cards.float()], dim=-1)
            v_oracle = self.oracle_critic_head(oracle_in)

        return masked_logits, v_win, v_vp, pred_belief, v_oracle

    def predict_belief(self, obs: torch.Tensor) -> torch.Tensor:
        """Predicts opponent hand card probabilities beta_opp in [0, 1]^110."""
        b_raw, c_raw, g_raw, h_raw = self._extract_inputs(obs)
        B = obs.size(0)

        b_nodes = b_raw.view(B, 84, 28)
        norm_adj = self.norm_adj.to(obs.device)
        h_gcn = self.gconv2(self.gconv1(b_nodes, norm_adj), norm_adj)
        r_emb = self.region_emb(self.country_regions.to(obs.device)).unsqueeze(0).expand(B, -1, -1)
        h_nodes, _ = self.board_self_attn(h_gcn + r_emb, h_gcn + r_emb, h_gcn + r_emb)
        b_rep = self.board_proj(torch.cat([h_nodes.mean(dim=1), h_nodes.max(dim=1).values], dim=-1))

        c_nodes = c_raw.view(B, 110, 12)
        loc_idx = c_nodes[:, :, 0].long().clamp(0, 6)
        c_feat = self.card_feat_in(c_nodes) + self.card_loc_emb(loc_idx)
        h_cards = self.card_transformer(c_feat)
        c_rep = self.card_proj(h_cards.mean(dim=1))

        h_seq = self.hist_transformer(self.hist_in(h_raw.view(B, 16, 32)))
        h_rep = self.hist_proj(h_seq[:, -1, :])
        g_rep = self.global_fc(g_raw)

        latent = self.fusion_in(torch.cat([b_rep, c_rep, h_rep, g_rep], dim=-1))
        for block in self.res_blocks:
            latent = block(latent)

        return self.belief_head(latent)

    def evaluate_oracle(self, obs: torch.Tensor, true_opponent_cards: torch.Tensor) -> torch.Tensor:
        """Evaluates privileged oracle value head conditioned on true hidden opponent cards."""
        b_raw, c_raw, g_raw, h_raw = self._extract_inputs(obs)
        B = obs.size(0)

        b_nodes = b_raw.view(B, 84, 28)
        norm_adj = self.norm_adj.to(obs.device)
        h_gcn = self.gconv2(self.gconv1(b_nodes, norm_adj), norm_adj)
        r_emb = self.region_emb(self.country_regions.to(obs.device)).unsqueeze(0).expand(B, -1, -1)
        h_nodes, _ = self.board_self_attn(h_gcn + r_emb, h_gcn + r_emb, h_gcn + r_emb)
        b_rep = self.board_proj(torch.cat([h_nodes.mean(dim=1), h_nodes.max(dim=1).values], dim=-1))

        c_nodes = c_raw.view(B, 110, 12)
        loc_idx = c_nodes[:, :, 0].long().clamp(0, 6)
        c_feat = self.card_feat_in(c_nodes) + self.card_loc_emb(loc_idx)
        h_cards = self.card_transformer(c_feat)
        c_rep = self.card_proj(h_cards.mean(dim=1))

        h_seq = self.hist_transformer(self.hist_in(h_raw.view(B, 16, 32)))
        h_rep = self.hist_proj(h_seq[:, -1, :])
        g_rep = self.global_fc(g_raw)

        latent = self.fusion_in(torch.cat([b_rep, c_rep, h_rep, g_rep], dim=-1))
        for block in self.res_blocks:
            latent = block(latent)

        oracle_in = torch.cat([latent, true_opponent_cards.float()], dim=-1)
        return self.oracle_critic_head(oracle_in)

    def sample_action(
        self, obs: torch.Tensor, legal_mask: torch.Tensor, temperature: float = 1.0, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_logits, v_win, v_vp = self.forward(obs, legal_mask)
        if deterministic or temperature == 0.0:
            actions = torch.argmax(masked_logits, dim=-1)
        else:
            scaled_logits = masked_logits / max(temperature, 1e-4)
            probs = F.softmax(scaled_logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        unscaled_log_probs = F.log_softmax(masked_logits, dim=-1)
        log_probs = unscaled_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        probs_all = F.softmax(masked_logits, dim=-1)
        entropy = -(probs_all * torch.nan_to_num(unscaled_log_probs, 0.0)).sum(dim=-1)

        return actions, log_probs, v_win.squeeze(-1), v_vp.squeeze(-1), entropy

    def evaluate_actions(
        self, obs: torch.Tensor, legal_mask: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_logits, v_win, v_vp = self.forward(obs, legal_mask)
        unscaled_log_probs = F.log_softmax(masked_logits, dim=-1)
        log_probs = unscaled_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        probs_all = F.softmax(masked_logits, dim=-1)
        entropy = -(probs_all * torch.nan_to_num(unscaled_log_probs, 0.0)).sum(dim=-1)
        return log_probs, entropy, v_win.squeeze(-1), v_vp.squeeze(-1)


def create_coldwar_net_v4(device: torch.device | str = "cuda") -> ColdWarNetV4:
    """Factory function for initializing ColdWarNetV4 on the target device."""
    dev = torch.device(device if (torch.cuda.is_available() and device == "cuda") else ("cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"))
    net = ColdWarNetV4(hidden_dim=512, card_dim=128, node_dim=64, hist_dim=64, num_card_layers=4, num_res_blocks=4)
    return net.to(dev)
