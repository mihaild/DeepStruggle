"""ColdWarNet: Multi-Modal Graph-Card-Global Neural Network for Twilight Struggle."""

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


class GraphConvLayer(nn.Module):
    """Simple and fast batched Graph Convolution layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # x: (B, 84, in_features)
        # norm_adj: (84, 84)
        support = self.linear(x) # (B, 84, out_features)
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


class ColdWarNet(nn.Module):
    """ColdWarNet: Multi-modal Policy-Value network for Twilight Struggle."""

    BOARD_OFFSET = 0
    BOARD_SIZE = 84 * 28 # 2352

    CARD_OFFSET = 2352
    CARD_SIZE = 110 * 12 # 1320

    GLOBAL_OFFSET = 2352 + 1320 # 3672
    GLOBAL_SIZE = 76

    HIST_OFFSET = 3672 + 76 # 3748
    HIST_SIZE = 16 * 32 # 512

    TOTAL_OBS_SIZE = 4293
    ACTION_SPACE_SIZE = 212

    def __init__(self, hidden_dim: int = 512, num_res_blocks: int = 4):
        super().__init__()
        self.register_buffer("norm_adj", build_normalized_adjacency_matrix())

        # 1. Board Graph Encoder (84 nodes x 28 features)
        self.gconv1 = GraphConvLayer(28, 64)
        self.gconv2 = GraphConvLayer(64, 64)
        self.board_proj = nn.Sequential(
            nn.Linear(64 * 2, 256), # Mean + Max pooling over 84 nodes
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 2. Card Registry Encoder (110 cards x 12 features)
        self.card_fc = nn.Sequential(
            nn.Linear(12, 32),
            nn.GELU(),
        )
        self.card_proj = nn.Sequential(
            nn.Linear(32 * 2, 256), # Mean + Max pooling over 110 cards
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 3. Global Scalars & Flags Encoder (76 features)
        self.global_proj = nn.Sequential(
            nn.Linear(76, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 4. History Sequence Encoder (16 tokens x 32 features)
        self.hist_conv = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 5. Fusion Trunk (256 + 256 + 128 + 128 = 768 -> hidden_dim)
        self.fusion_in = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_res_blocks)])

        # 6. Masked Policy Head (212 Actions)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.ACTION_SPACE_SIZE),
        )

        # 7. Dual Value Heads (Win/Loss Utility in [-1, 1] and Auxiliary VP in [-20, 20])
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

        # 8. Auxiliary DEFCON-risk head: logit of "the player to move loses to DEFCON 1
        # (or a Cuban Missile Crisis coup) within the next few of its own plies".
        #
        # val_win_head is close to a restatement of the VP margin -- measured over the
        # review replays, corr(v_win, v_vp) = 0.86 -- which is why it reads self-inflicted
        # DEFCON-1 deaths as roughly even positions (-0.27) while pricing ordinary losing
        # positions correctly (-0.72). This head puts terminal risk into the shared latent
        # as an explicit target rather than hoping v_win infers it from a scoreboard that
        # does not contain it. Emits a logit; apply sigmoid for a probability.
        self.defcon_risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extracts fused latent state representation."""
        batch_size = obs.shape[0]

        # 1. Board Graph Features: (B, 84, 28)
        board_raw = obs[:, self.BOARD_OFFSET : self.BOARD_OFFSET + self.BOARD_SIZE]
        board_nodes = board_raw.view(batch_size, 84, 28)
        h_board = self.gconv1(board_nodes, self.norm_adj)
        h_board = self.gconv2(h_board, self.norm_adj) # (B, 84, 64)
        board_mean = torch.mean(h_board, dim=1) # (B, 64)
        board_max, _ = torch.max(h_board, dim=1) # (B, 64)
        e_board = self.board_proj(torch.cat([board_mean, board_max], dim=-1)) # (B, 256)

        # 2. Card Features: (B, 110, 12)
        card_raw = obs[:, self.CARD_OFFSET : self.CARD_OFFSET + self.CARD_SIZE]
        card_nodes = card_raw.view(batch_size, 110, 12)
        h_cards = self.card_fc(card_nodes) # (B, 110, 32)
        card_mean = torch.mean(h_cards, dim=1) # (B, 32)
        card_max, _ = torch.max(h_cards, dim=1) # (B, 32)
        e_card = self.card_proj(torch.cat([card_mean, card_max], dim=-1)) # (B, 256)

        # 3. Global Features: (B, 76)
        global_raw = obs[:, self.GLOBAL_OFFSET : self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
        e_global = self.global_proj(global_raw) # (B, 128)

        # 4. History Features: (B, 16, 32) -> transpose to (B, 32, 16) for Conv1D
        hist_raw = obs[:, self.HIST_OFFSET : self.HIST_OFFSET + self.HIST_SIZE]
        hist_tokens = hist_raw.view(batch_size, 16, 32).transpose(1, 2)
        e_hist = self.hist_conv(hist_tokens) # (B, 128)

        # 5. Fusion Trunk
        fused = torch.cat([e_board, e_card, e_global, e_hist], dim=-1) # (B, 768)
        h = self.fusion_in(fused)
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (B, 4293) float tensor.
            mask: (B, 212) uint8 / bool tensor of legal actions.

        Returns:
            masked_logits: (B, 212) float tensor.
            v_win: (B, 1) float tensor in [-1, 1].
            v_vp: (B, 1) float tensor in [-20, 20].
        """
        h = self.extract_features(obs)
        raw_logits = self.policy_head(h)

        if mask is not None:
            # Mask illegal actions with -1e9
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            masked_logits = torch.where(mask_bool, raw_logits, torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype))
        else:
            masked_logits = raw_logits

        v_win = self.val_win_head(h)
        v_vp = self.val_vp_head(h)
        return masked_logits, v_win, v_vp

    def forward_with_risk(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward() plus the auxiliary DEFCON-risk logit, sharing one backbone pass.

        Kept separate from forward() so the existing three-value contract, which every
        other caller depends on, is untouched. Returns (logits, v_win, v_vp, risk_logit).
        """
        h = self.extract_features(obs)
        raw_logits = self.policy_head(h)
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            masked_logits = torch.where(
                mask_bool, raw_logits,
                torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype),
            )
        else:
            masked_logits = raw_logits
        return (masked_logits, self.val_win_head(h), self.val_vp_head(h),
                self.defcon_risk_head(h))

    @torch.no_grad()
    def defcon_risk(self, obs: torch.Tensor) -> torch.Tensor:
        """Probability that the player to move is about to lose to DEFCON 1."""
        return torch.sigmoid(self.defcon_risk_head(self.extract_features(obs)))

    @torch.no_grad()
    def sample_action(
        self, obs: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates chosen actions for PPO / NashPG training updates."""
        masked_logits, v_win, v_vp = self.forward(obs, mask)
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, v_win.squeeze(-1), v_vp.squeeze(-1)


def create_coldwar_net(device: torch.device | str = "cpu") -> ColdWarNet:
    """Factory helper to instantiate ColdWarNet on specified device."""
    model = ColdWarNet(hidden_dim=512, num_res_blocks=4)
    return model.to(device)
