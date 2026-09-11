"""ColdWarNetV2: Graph-Card Cross-Attention Neural Network for Twilight Struggle."""


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


class ColdWarNetV2(nn.Module):
    """
    ColdWarNetV2: Enhanced Policy-Value Architecture with Card <-> Country Cross-Attention.
    
    One observation layout, v2.3, and the class constants describe it. The constructor still
    takes the four dimensions and recomputes every offset from them, so a model can be built to
    a checkpoint's own shape -- but there is no second shape to build.

    | | v2.3 |
    |:---|---:|
    | board, per country | 26 |
    | card, per card | 14 |
    | global scalars | 100 |
    | **observation** | **3824** |
    | parameters | 3,091,735 |

    Branches:

    - 2-layer GraphConv over the 84 country nodes on the map adjacency (26 -> 64 -> 64), then
      mean- and max-pooled over all countries and projected to 256.
    - 110-card MLP embedder (14 -> 64), likewise mean/max-pooled and projected to 256.
    - Multi-head cross-attention, cards querying country nodes, pooled to 256.
    - Global scalar projection (100 -> 128).
    - Fusion trunk: concat -> 896 -> 512, four Pre-LN residual blocks.
    - Masked policy head (212) and dual value heads (win/loss tanh, auxiliary VP).

    **There is no history branch.** The 512-float block was a constant zero vector --
    `ActionHistoryBuffer::record()` is called nowhere in the engine -- so v2.1 dropped it and no
    layout has carried one since. `use_history` survives as a constructor argument and defaults
    to False; with it set, the temporal ConvNet and its 128 floats of trunk input come back and
    the trunk takes 1024 rather than 896. Nothing in the engine would fill them.

    Note the board branch pools across all 84 countries *before* the global block is seen, and
    the policy head is dense off the fused vector -- there is no per-country output path. See
    `research/metrics.md` 1.4.2 for what that costs.
    """

    # Observation layout v2.3, which is the only layout the engine emits. The offsets are still
    # recomputed in __init__ from the four dimensions rather than assumed, because the class
    # constants are defaults and an instance may be built to a checkpoint's own shape.
    BOARD_OFFSET = 0
    BOARD_FEATURES = 26
    BOARD_SIZE = 84 * 26  # 2184

    CARD_OFFSET = 2184
    CARD_FEATURES = 14
    CARD_SIZE = 110 * 14  # 1540

    GLOBAL_OFFSET = 2184 + 1540  # 3724
    GLOBAL_SIZE = 100

    # Retained only to size the branch when a model is built with use_history=True. Nothing
    # writes an action history into the observation and no current layout carries one.
    HIST_OFFSET = 3724 + 100  # 3824
    HIST_SIZE = 16 * 32  # 512

    TOTAL_OBS_SIZE = 3824
    ACTION_SPACE_SIZE = 212

    def __init__(self, hidden_dim: int = 512, num_res_blocks: int = 4, num_attn_heads: int = 4,
                 card_features: int = CARD_FEATURES, use_history: bool = False,
                 global_features: int = GLOBAL_SIZE, has_tail: bool = False,
                 board_features: int = BOARD_FEATURES):
        super().__init__()
        self.register_buffer("norm_adj", build_normalized_adjacency_matrix())

        # 14 in v2.3: eight slots saying where the card is, five card properties, and one
        # saying whether this decision is about that card.
        self.card_features = int(card_features)
        # The history block is a constant zero vector -- ActionHistoryBuffer::record() is called
        # nowhere -- so with use_history=False both the branch that encodes it and its share of
        # the fusion trunk go away, and the observation is that much narrower.
        self.use_history = bool(use_history)
        # 100: 72 board-and-track scalars, then the 28-float decision context -- decision type,
        # op mode, points remaining, the per-country cap, the timing branch.
        self.GLOBAL_SIZE = int(global_features)
        # Retired layouts ended with turn_aggregates (32) and active_player (1). v2.3 has
        # neither: the forward pass never sliced them, so of the 33 floats not one ever reached
        # a network.
        self.has_tail = bool(has_tail)
        # 26 in v2.3, which has no per-country realignment legality pair -- can_realign differs
        # from can_coup only under The Reformer.
        self.board_features = int(board_features)
        self.BOARD_SIZE = 84 * self.board_features
        self.CARD_OFFSET = self.BOARD_SIZE
        self.CARD_SIZE = 110 * self.card_features
        self.GLOBAL_OFFSET = self.CARD_OFFSET + self.CARD_SIZE
        self.HIST_OFFSET = self.GLOBAL_OFFSET + self.GLOBAL_SIZE
        hist_width = self.HIST_SIZE if self.use_history else 0
        self.TOTAL_OBS_SIZE = self.HIST_OFFSET + hist_width + (33 if self.has_tail else 0)

        # 1. Board Graph Encoder (84 nodes x 28 features -> 64)
        self.gconv1 = GraphConvLayer(self.board_features, 64)
        self.gconv2 = GraphConvLayer(64, 64)
        self.board_proj = nn.Sequential(
            nn.Linear(64 * 2, 256),  # Mean + Max pooling over 84 nodes
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 2. Card Registry Encoder (110 cards x 12 features -> 64)
        self.card_fc = nn.Sequential(
            nn.Linear(self.card_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.card_proj = nn.Sequential(
            nn.Linear(64 * 2, 256),  # Mean + Max pooling over 110 cards
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 3. Card <-> Country Multi-Head Cross-Attention Layer
        # Queries: Cards (B, 110, 64), Keys/Values: Countries (B, 84, 64)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=num_attn_heads,
            dropout=0.05,
            batch_first=True,
        )
        self.cross_attn_ln = nn.LayerNorm(64)
        self.cross_card_proj = nn.Sequential(
            nn.Linear(64 * 2, 256),  # Mean + Max pooling over cross-attended card tokens
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 4. Global Scalars & Flags Encoder (GLOBAL_SIZE features -> 128)
        self.global_proj = nn.Sequential(
            nn.Linear(self.GLOBAL_SIZE, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 5. History Sequence Encoder (16 tokens x 32 features -> 128)
        self.hist_conv = None if not self.use_history else nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # 6. Fusion Trunk (256 [board] + 256 [cards] + 256 [cross] + 128 [global] + 128 [hist] = 1024 -> hidden_dim)
        self.fusion_in = nn.Sequential(
            # 1024 with history: board 256 + card 256 + cross-attended card 256 + global
            # 128 + history 128. Dropping the history branch removes its 128.
            nn.Linear(1024 if self.use_history else 896, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_res_blocks)])

        # 7. Masked Policy Head (212 Actions)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.ACTION_SPACE_SIZE),
        )

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

        # Auxiliary DEFCON-risk head: logit of "the player to move loses to DEFCON 1 (or a
        # Cuban Missile Crisis coup) within the next few of its own plies". See the note in
        # coldwar_net.py; inert unless --defcon-coef is set. Emits a logit.
        self.defcon_risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def extract_features(self, obs: torch.Tensor, return_attn_weights: bool = False):
        """Extracts fused latent state representation and optional cross-attention maps."""
        if obs.shape[-1] != self.TOTAL_OBS_SIZE:
            raise ValueError(
                f"observation is {obs.shape[-1]} floats wide; this model reads "
                f"{self.TOTAL_OBS_SIZE}. Every slice below is taken at a fixed offset, so a "
                f"mismatched vector does not fail -- it is misread, and the network plays "
                f"near-randomly while returning perfectly ordinary-looking numbers. That went "
                f"unnoticed four times before this check existed.")
        batch_size = obs.shape[0]

        # 1. Board Graph Features: (B, 84, 28)
        board_raw = obs[:, self.BOARD_OFFSET : self.BOARD_OFFSET + self.BOARD_SIZE]
        board_nodes = board_raw.view(batch_size, 84, self.board_features)
        h_board = self.gconv1(board_nodes, self.norm_adj)
        h_board = self.gconv2(h_board, self.norm_adj)  # (B, 84, 64)
        board_mean = torch.mean(h_board, dim=1)  # (B, 64)
        board_max, _ = torch.max(h_board, dim=1)  # (B, 64)
        e_board = self.board_proj(torch.cat([board_mean, board_max], dim=-1))  # (B, 256)

        # 2. Card Features: (B, 110, card_features)
        card_raw = obs[:, self.CARD_OFFSET : self.CARD_OFFSET + self.CARD_SIZE]
        card_nodes = card_raw.view(batch_size, 110, self.card_features)
        h_cards = self.card_fc(card_nodes)  # (B, 110, 64)
        card_mean = torch.mean(h_cards, dim=1)  # (B, 64)
        card_max, _ = torch.max(h_cards, dim=1)  # (B, 64)
        e_card = self.card_proj(torch.cat([card_mean, card_max], dim=-1))  # (B, 256)

        # 3. Card <-> Country Multi-Head Cross-Attention
        # Cards query Countries: Q = h_cards, K = h_board, V = h_board
        attn_out, attn_weights = self.cross_attn(h_cards, h_board, h_board)  # (B, 110, 64), (B, 110, 84)
        h_cards_cross = self.cross_attn_ln(h_cards + attn_out)  # Residual connection + LayerNorm
        cross_mean = torch.mean(h_cards_cross, dim=1)  # (B, 64)
        cross_max, _ = torch.max(h_cards_cross, dim=1)  # (B, 64)
        e_cross = self.cross_card_proj(torch.cat([cross_mean, cross_max], dim=-1))  # (B, 256)

        # 4. Global Features: (B, 76)
        global_raw = obs[:, self.GLOBAL_OFFSET : self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
        e_global = self.global_proj(global_raw)  # (B, 128)

        # 5. History Features: (B, 16, 32) -> transpose to (B, 32, 16) for Conv1D
        if self.use_history and self.hist_conv is not None:
            hist_raw = obs[:, self.HIST_OFFSET : self.HIST_OFFSET + self.HIST_SIZE]
            hist_tokens = hist_raw.view(batch_size, 16, 32).transpose(1, 2)
            e_hist = self.hist_conv(hist_tokens)  # (B, 128)
            fused = torch.cat([e_board, e_card, e_cross, e_global, e_hist], dim=-1)  # (B, 1024)
        else:
            # Same trunk minus the history embedding. e_cross must stay: the cross-attention
            # branch is the whole point of this architecture, and dropping it from the concat
            # leaves it computed and discarded.
            fused = torch.cat([e_board, e_card, e_cross, e_global], dim=-1)  # (B, 896)
        h = self.fusion_in(fused)
        for block in self.res_blocks:
            h = block(h)

        if return_attn_weights:
            return h, attn_weights
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
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            masked_logits = torch.where(
                mask_bool, raw_logits, torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype)
            )
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
        other caller depends on, is untouched.
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


def create_coldwar_net_v2(device: torch.device | str = "cpu",
                          card_features: int = ColdWarNetV2.CARD_FEATURES,
                          use_history: bool = False,
                          global_features: int = ColdWarNetV2.GLOBAL_SIZE,
                          has_tail: bool = False,
                          board_features: int = ColdWarNetV2.BOARD_FEATURES) -> ColdWarNetV2:
    """Factory helper to instantiate ColdWarNetV2 on specified device.

    The defaults are observation layout v2.3, which is the only layout the engine emits.
    """
    model = ColdWarNetV2(hidden_dim=512, num_res_blocks=4, num_attn_heads=4,
                         card_features=card_features, use_history=use_history,
                         global_features=global_features, has_tail=has_tail,
                         board_features=board_features)
    return model.to(device)


def create_like(model: nn.Module, device: torch.device | str = "cpu") -> ColdWarNetV2:
    """A network shaped exactly like `model`.

    Every dimension is read off the model rather than passed in. A frozen evaluation copy built by
    listing arguments has to list *all* of them, and twice now it has not: arm E died on its first
    snapshot when card_features and use_history were left at factory defaults, and arm F died the
    same way on board_features after the other three had been fixed. Reading them from the source
    removes the class of error rather than the instance.
    """
    return create_coldwar_net_v2(
        device,
        card_features=int(getattr(model, "card_features", ColdWarNetV2.CARD_FEATURES)),
        use_history=bool(getattr(model, "use_history", False)),
        global_features=int(getattr(model, "GLOBAL_SIZE", ColdWarNetV2.GLOBAL_SIZE)),
        has_tail=bool(getattr(model, "has_tail", False)),
        board_features=int(getattr(model, "board_features", ColdWarNetV2.BOARD_FEATURES)))


def check_checkpoint_layout(state_dict: dict) -> None:
    """Refuses a checkpoint that was not trained on the engine's one observation layout.

    Derived from the weights rather than from a recorded name, because a checkpoint is a bare
    state dict. A model handed the wrong width does not fail on its own -- it reads fixed slices,
    so the observation is silently misread and the network merely plays badly.

    Checkpoints from the retired layouts cannot be run and are not being converted: legacy, v2.1
    and v2.2 all predate the starred-card fix, so they were trained against a different game and
    are not comparable to anything measured now.
    """
    cards = card_features_of(state_dict)
    g = state_dict.get("global_proj.0.weight")
    globals_ = int(g.shape[1]) if g is not None else ColdWarNetV2.GLOBAL_SIZE
    has_hist = any(k.startswith("hist_conv.") for k in state_dict)
    if (cards, globals_, has_hist) != (ColdWarNetV2.CARD_FEATURES, ColdWarNetV2.GLOBAL_SIZE, False):
        raise ValueError(
            f"this checkpoint has card_features={cards}, global_features={globals_}, "
            f"history={has_hist}; layout v2.3 is card_features={ColdWarNetV2.CARD_FEATURES}, "
            f"global_features={ColdWarNetV2.GLOBAL_SIZE}, history=False. Checkpoints from the "
            f"retired layouts cannot be run.")


def card_features_of(state_dict: dict) -> int:
    """How wide a checkpoint's card block is, read off its own weights.

    The first card layer is Linear(card_features, 64), so its weight is (64, card_features) and
    the width is not something a loader has to be told. That matters because the two layouts are
    otherwise indistinguishable from a file, and loading a 12-feature checkpoint as 13 would
    reinterpret every card feature by one position without any shape error to warn about --
    silently, and only after the numbers came out wrong.
    """
    for key in ("card_fc.0.weight", "module.card_fc.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise KeyError("no card_fc.0.weight in the checkpoint; cannot tell the layout")
