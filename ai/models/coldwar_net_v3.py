"""ColdWarNetV3: Dual Pointer-Generator Architecture with Bidirectional Co-Attention and Graph Transformers."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ts_engine as ts


def build_normalized_adjacency_matrix() -> torch.Tensor:
    """Builds the 84x84 normalized adjacency matrix with self-loops from MapData."""
    adj = np.eye(84, dtype=np.float32)
    for i in range(84):
        info = ts.MapData.get_country_info(i)
        for n in info["neighbors"]:
            adj[i, n] = 1.0
            adj[n, i] = 1.0

    # Degree normalization: D^(-1/2) * A * D^(-1/2)
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
        # x: (B, 84, in_features)
        # norm_adj: (84, 84)
        support = self.linear(x)  # (B, 84, out_features)
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


class ColdWarNetV3(nn.Module):
    """ColdWarNetV3: Dual Pointer-Generator Architecture for Twilight Struggle.

    Key Architectural Innovations:
    1. Country Graph Transformer:
       - 2-layer local GCN (28 -> 64 -> 64) augmented with learned Region ID embeddings (6 regions)
       - Multi-Head Country Self-Attention over all 84 countries for full-theatre regional coordination
    2. Card Registry & Location Masking:
       - Card feature encoder (12 -> 64) + learned card location embeddings (7 locations)
       - Attention masking for cards permanently removed from the game
    3. Bidirectional Co-Attention:
       - Cards attend to Countries (Q=Cards, K=V=Countries)
       - Countries attend to Cards (Q=Countries, K=V=Cards)
    4. Dual Pointer-Generator Policy Heads:
       - Actions 119..202 (84 countries): Modulated via Country Pointer Head (Q_action * H_country)
       - Actions 0..109 (110 cards): Modulated via Card Pointer Head (Q_action * H_card)
       - Modal Actions: 110..118 and 203..211 handled by global policy trunk
    5. Dual Value Heads:
       - Win/Loss utility in [-1, 1] (Tanh)
       - Auxiliary VP differential in [-20, 20]
    """

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

    # Flat Action Offsets
    CARD_ACTION_START = 0
    CARD_ACTION_END = 110
    NODE_ACTION_START = 119
    NODE_ACTION_END = 203

    def __init__(self, hidden_dim: int = 512, num_res_blocks: int = 4, num_attn_heads: int = 4):
        super().__init__()
        self.register_buffer("norm_adj", build_normalized_adjacency_matrix())
        self.register_buffer("country_regions", get_country_region_indices())

        node_dim = 64
        self.node_dim = node_dim

        # 1. Country Representation: GCN + Region Positional Embedding + Graph Self-Attention
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

        # 2. Card Representation: Feature Projection + Location Embedding
        self.card_fc = nn.Sequential(
            nn.Linear(12, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
        )
        self.card_loc_emb = nn.Embedding(7, node_dim)
        self.card_proj = nn.Sequential(
            nn.Linear(node_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 3. Bidirectional Co-Attention Layers
        # Path A: Cards attend to Countries
        self.cross_c2b = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_attn_heads,
            dropout=0.05,
            batch_first=True,
        )
        self.cross_c2b_ln = nn.LayerNorm(node_dim)

        # Path B: Countries attend to Cards (with location-aware masking)
        self.cross_b2c = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_attn_heads,
            dropout=0.05,
            batch_first=True,
        )
        self.cross_b2c_ln = nn.LayerNorm(node_dim)

        # 4. Global Scalars & Flags Encoder (76 features -> 128)
        self.global_proj = nn.Sequential(
            nn.Linear(76, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 5. History Sequence Encoder (16 tokens x 32 features -> 128)
        self.hist_conv = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 6. Fusion Trunk: 256 (board) + 256 (card) + 128 (global) + 128 (hist) = 768 -> hidden_dim
        self.fusion_in = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_res_blocks)])

        # 7. Policy Heads
        # A. Base Global Policy Head (212 Actions)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.ACTION_SPACE_SIZE),
        )

        # B. Country Pointer Head (modulates Actions 119..202)
        self.node_pointer_proj = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim),
        )
        self.node_bias = nn.Parameter(torch.zeros(84))

        # C. Card Pointer Head (modulates Actions 0..109)
        self.card_pointer_proj = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim),
        )
        self.card_bias = nn.Parameter(torch.zeros(110))

        # 8. Dual Value Heads (Win/Loss Utility in [-1, 1] and Auxiliary VP in [-20, 20])
        self.val_win_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.val_vp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def extract_features(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts refined country embeddings, card embeddings, and fused trunk representation.

        Returns:
            h_trunk: (B, hidden_dim)
            h_board: (B, 84, node_dim)
            h_cards: (B, 110, node_dim)
        """
        batch_size = obs.shape[0]

        # 1. Board Graph Features: (B, 84, 28)
        board_raw = obs[:, self.BOARD_OFFSET : self.BOARD_OFFSET + self.BOARD_SIZE]
        board_nodes = board_raw.view(batch_size, 84, 28)

        # Local graph convolutions
        h_board = self.gconv1(board_nodes, self.norm_adj)
        h_board = self.gconv2(h_board, self.norm_adj)  # (B, 84, 64)

        # Add region positional embeddings
        reg_embs = self.region_emb(self.country_regions)  # (84, 64)
        h_board = h_board + reg_embs.unsqueeze(0)

        # Regional Graph Transformer Self-Attention
        attn_board_self, _ = self.board_self_attn(h_board, h_board, h_board)
        h_board = self.board_self_ln(h_board + attn_board_self)

        # 2. Card Features: (B, 110, 12)
        card_raw = obs[:, self.CARD_OFFSET : self.CARD_OFFSET + self.CARD_SIZE]
        card_nodes = card_raw.view(batch_size, 110, 12)
        h_cards = self.card_fc(card_nodes)  # (B, 110, 64)

        # Add card location embeddings (features 0..6 represent 1-hot location)
        card_loc_indices = torch.argmax(card_nodes[:, :, 0:7], dim=-1)  # (B, 110)
        h_cards = h_cards + self.card_loc_emb(card_loc_indices)

        # Location-based key padding mask: mask cards removed from the game (index 4)
        # Ensure we do not mask out all cards if all were somehow removed (guard against NaN)
        removed_mask = (card_loc_indices == 4)
        all_removed = removed_mask.all(dim=-1, keepdim=True)
        key_padding_mask = removed_mask & (~all_removed)

        # 3. Bidirectional Co-Attention
        # Path A: Cards attend to Countries
        attn_c2b, _ = self.cross_c2b(h_cards, h_board, h_board)
        h_cards = self.cross_c2b_ln(h_cards + attn_c2b)

        # Path B: Countries attend to Cards
        attn_b2c, _ = self.cross_b2c(h_board, h_cards, h_cards, key_padding_mask=key_padding_mask)
        h_board = self.cross_b2c_ln(h_board + attn_b2c)

        # 4. Pooled Board and Card Summaries
        board_mean = torch.mean(h_board, dim=1)
        board_max, _ = torch.max(h_board, dim=1)
        e_board = self.board_proj(torch.cat([board_mean, board_max], dim=-1))  # (B, 256)

        card_mean = torch.mean(h_cards, dim=1)
        card_max, _ = torch.max(h_cards, dim=1)
        e_card = self.card_proj(torch.cat([card_mean, card_max], dim=-1))  # (B, 256)

        # 5. Global Scalar Features: (B, 76) -> (B, 128)
        global_raw = obs[:, self.GLOBAL_OFFSET : self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
        e_global = self.global_proj(global_raw)

        # 6. History ConvNet Features: (B, 16, 32) -> (B, 128)
        hist_raw = obs[:, self.HIST_OFFSET : self.HIST_OFFSET + self.HIST_SIZE]
        hist_tokens = hist_raw.view(batch_size, 16, 32).transpose(1, 2)
        e_hist = self.hist_conv(hist_tokens)

        # 7. ResNet Fusion Trunk
        fused = torch.cat([e_board, e_card, e_global, e_hist], dim=-1)  # (B, 768)
        h_trunk = self.fusion_in(fused)
        for block in self.res_blocks:
            h_trunk = block(h_trunk)

        return h_trunk, h_board, h_cards

    def forward(
        self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (B, 4293) float tensor.
            mask: (B, 212) uint8 / bool tensor of legal actions.

        Returns:
            masked_logits: (B, 212) float tensor.
            v_win: (B, 1) float tensor in [-1, 1].
            v_vp: (B, 1) float tensor in [-20, 20].
        """
        h_trunk, h_board, h_cards = self.extract_features(obs)

        # 1. Base Logits from global trunk
        raw_logits = self.policy_head(h_trunk)

        # 2. Country Pointer Head: directly modulate actions [119..202]
        # Q_node: (B, 1, node_dim), H_board: (B, 84, node_dim)
        q_node = self.node_pointer_proj(h_trunk).unsqueeze(1)
        node_ptr = torch.bmm(h_board, q_node.transpose(1, 2)).squeeze(-1) / (self.node_dim ** 0.5)
        node_ptr = node_ptr + self.node_bias

        # 3. Card Pointer Head: directly modulate actions [0..109]
        # Q_card: (B, 1, node_dim), H_cards: (B, 110, node_dim)
        q_card = self.card_pointer_proj(h_trunk).unsqueeze(1)
        card_ptr = torch.bmm(h_cards, q_card.transpose(1, 2)).squeeze(-1) / (self.node_dim ** 0.5)
        card_ptr = card_ptr + self.card_bias

        # Combine global prior with pointer modulations
        logits = raw_logits.clone()
        logits[:, self.CARD_ACTION_START : self.CARD_ACTION_END] = (
            logits[:, self.CARD_ACTION_START : self.CARD_ACTION_END] + card_ptr
        )
        logits[:, self.NODE_ACTION_START : self.NODE_ACTION_END] = (
            logits[:, self.NODE_ACTION_START : self.NODE_ACTION_END] + node_ptr
        )

        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            masked_logits = torch.where(
                mask_bool, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
            )
        else:
            masked_logits = logits

        v_win = self.val_win_head(h_trunk)
        v_vp = self.val_vp_head(h_trunk)
        return masked_logits, v_win, v_vp

    @torch.no_grad()
    def sample_action(
        self, obs: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples action indices given observations and legal masks."""
        masked_logits, v_win, v_vp = self.forward(obs, mask)

        if deterministic:
            actions = torch.argmax(masked_logits, dim=-1)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        else:
            scaled_logits = masked_logits / max(temperature, 1e-4)
            dist = torch.distributions.Categorical(logits=scaled_logits)
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return actions, action_log_probs, v_win.squeeze(-1), v_vp.squeeze(-1), entropy

    def evaluate_actions(
        self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates chosen actions for PPO / NashPG training updates."""
        masked_logits, v_win, v_vp = self.forward(obs, mask)
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, v_win.squeeze(-1), v_vp.squeeze(-1)


def create_coldwar_net_v3(device: torch.device | str = "cpu") -> ColdWarNetV3:
    """Factory helper to instantiate ColdWarNetV3 on specified device."""
    model = ColdWarNetV3(hidden_dim=512, num_res_blocks=4, num_attn_heads=4)
    return model.to(device)
