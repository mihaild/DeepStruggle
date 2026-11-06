"""ColdWarNet: Multi-Modal Graph-Card-Global Neural Network for Twilight Struggle."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.models.coldwar_net_v2 import GraphConvLayer, ResBlock, build_normalized_adjacency_matrix, mask_logits
from bindings.action_encoder import ActionEncoder


class ColdWarNet(nn.Module):
    """ColdWarNet: Multi-modal Policy-Value network for Twilight Struggle."""

    # Observation layout v2.3, the only layout the engine emits. This architecture was written
    # against the legacy layout; retargeting it cost four constants and the history branch,
    # which had nothing to read even then -- ActionHistoryBuffer::record() is called nowhere.
    BOARD_OFFSET = 0
    BOARD_FEATURES = 26
    BOARD_SIZE = 84 * 26 # 2184

    CARD_OFFSET = 2184
    CARD_FEATURES = 14
    CARD_SIZE = 110 * 14 # 1540

    GLOBAL_OFFSET = 2184 + 1540 # 3724
    GLOBAL_SIZE = 100

    TOTAL_OBS_SIZE = 3824
    ACTION_SPACE_SIZE = ActionEncoder.BASE_ACTION_SIZE

    def __init__(self, hidden_dim: int = 512, num_res_blocks: int = 4):
        super().__init__()
        self.register_buffer("norm_adj", build_normalized_adjacency_matrix())

        # 1. Board Graph Encoder (84 nodes x 26 features)
        self.gconv1 = GraphConvLayer(self.BOARD_FEATURES, 64)
        self.gconv2 = GraphConvLayer(64, 64)
        self.board_proj = nn.Sequential(
            nn.Linear(64 * 2, 256), # Mean + Max pooling over 84 nodes
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 2. Card Registry Encoder (110 cards x 14 features)
        self.card_fc = nn.Sequential(
            nn.Linear(self.CARD_FEATURES, 32),
            nn.GELU(),
        )
        self.card_proj = nn.Sequential(
            nn.Linear(32 * 2, 256), # Mean + Max pooling over 110 cards
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 3. Global Scalars & Flags Encoder (100 features)
        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_SIZE, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 4. Fusion Trunk (256 board + 256 cards + 128 global = 640 -> hidden_dim)
        self.fusion_in = nn.Sequential(
            nn.Linear(640, hidden_dim),
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
        if obs.shape[-1] != self.TOTAL_OBS_SIZE:
            raise ValueError(
                f"observation is {obs.shape[-1]} floats wide; this model reads "
                f"{self.TOTAL_OBS_SIZE}. Every slice below is taken at a fixed offset, so a "
                f"mismatched vector is misread rather than rejected.")
        batch_size = obs.shape[0]

        # 1. Board Graph Features: (B, 84, 26)
        board_raw = obs[:, self.BOARD_OFFSET : self.BOARD_OFFSET + self.BOARD_SIZE]
        board_nodes = board_raw.view(batch_size, 84, self.BOARD_FEATURES)
        h_board = self.gconv1(board_nodes, self.norm_adj)
        h_board = self.gconv2(h_board, self.norm_adj) # (B, 84, 64)
        board_mean = torch.mean(h_board, dim=1) # (B, 64)
        board_max, _ = torch.max(h_board, dim=1) # (B, 64)
        e_board = self.board_proj(torch.cat([board_mean, board_max], dim=-1)) # (B, 256)

        # 2. Card Features: (B, 110, 14)
        card_raw = obs[:, self.CARD_OFFSET : self.CARD_OFFSET + self.CARD_SIZE]
        card_nodes = card_raw.view(batch_size, 110, self.CARD_FEATURES)
        h_cards = self.card_fc(card_nodes) # (B, 110, 32)
        card_mean = torch.mean(h_cards, dim=1) # (B, 32)
        card_max, _ = torch.max(h_cards, dim=1) # (B, 32)
        e_card = self.card_proj(torch.cat([card_mean, card_max], dim=-1)) # (B, 256)

        # 3. Global Features: (B, 100)
        global_raw = obs[:, self.GLOBAL_OFFSET : self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
        e_global = self.global_proj(global_raw) # (B, 128)

        # 4. Fusion Trunk
        fused = torch.cat([e_board, e_card, e_global], dim=-1) # (B, 640)
        h = self.fusion_in(fused)
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (B, TOTAL_OBS_SIZE) float tensor (the v2.3 observation).
            mask: (B, FLAT_ACTION_SIZE) uint8 / bool tensor of legal actions.

        Returns:
            masked_logits: (B, FLAT_ACTION_SIZE) float tensor.
            v_win: (B, 1) float tensor in [-1, 1].
            v_vp: (B, 1) float tensor in [-20, 20].
        """
        h = self.extract_features(obs)
        raw_logits = self.policy_head(h)

        masked_logits = mask_logits(raw_logits, mask)

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
        masked_logits = mask_logits(raw_logits, mask)
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
