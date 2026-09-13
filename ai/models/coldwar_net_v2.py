"""ColdWarNetV2: Graph-Card Cross-Attention Neural Network for Twilight Struggle."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ts_engine as ts


#: Final VP is clamped to +/-20 by the engine, and every abrupt ending is normalised to exactly
#: that, so the categorical head's support is the true range rather than a hyperparameter.
VP_LIMIT: int = 20
#: One atom per integer VP across [-20, +20]. 41 puts an atom exactly on 0, which is what makes
#: P(VP > 0) - P(VP < 0) a clean read of the win probability.
VALUE_ATOMS: int = 2 * VP_LIMIT + 1


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
    """Batched Graph Convolution layer.

    With `self_transform=False` this is textbook GCN, `D^-1/2 (A+I) D^-1/2 @ (W x)`: **one weight
    matrix for a country and for its neighbours alike**, and the only thing preserving a
    country's own value is the self-loop, whose weight is `1/(deg+1)`. That is a low-pass filter,
    and exact per-country influence is the high-frequency part of the signal.

    The consequence is measurable: a linear probe recovers a country's exact influence far more
    often from its raw observation slots than from its post-GCN token, and the loss tracks degree
    -- a country with one neighbour passes through almost untouched, one with five loses most of
    it.

    `self_transform=True` adds a second weight matrix applied to the node itself, so the layer
    can hold a country at full strength instead of being forced to average it with its
    neighbours. Adjacency matters in this game -- placement legality, realignment, superpower
    adjacency -- so the relation is kept; only the forced averaging goes.
    """

    def __init__(self, in_features: int, out_features: int, self_transform: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.self_linear = (nn.Linear(in_features, out_features, bias=False)
                            if self_transform else None)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # x: (B, 84, in_features)
        # norm_adj: (84, 84)
        support = self.linear(x)  # (B, 84, out_features)
        out = torch.matmul(norm_adj, support) + self.bias
        if self.self_linear is not None:
            out = out + self.self_linear(x)
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
    the policy head is dense off the fused vector -- there is no per-country output path, which
    costs the heads most of what they could know about individual countries.
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

    # Declared for the type checker: `value_support` is a registered buffer, and the two scalar
    # heads are None on a categorical instance (and the distribution head None on a scalar one).
    value_support: torch.Tensor
    keep_idx: torch.Tensor
    val_win_head: nn.Module | None
    val_vp_head: nn.Module | None
    value_dist_head: nn.Module | None
    card_identity: nn.Embedding | None
    country_identity: nn.Embedding | None

    def __init__(self, hidden_dim: int = 512, num_res_blocks: int = 4, num_attn_heads: int = 4,
                 card_features: int = CARD_FEATURES, use_history: bool = False,
                 global_features: int = GLOBAL_SIZE, has_tail: bool = False,
                 board_features: int = BOARD_FEATURES,
                 categorical_value: bool = False, value_atoms: int = VALUE_ATOMS,
                 identity_dim: int = 0, self_transform: bool = False,
                 attn_readout: int = 0, per_entity_heads: int = 0,
                 graph_layers: int = 2):
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

        # Identity embeddings. The card block encodes a card's *properties* -- side, Ops, a few
        # flags -- and never which card it is; identity exists only as position in the 110x14
        # block. The card branch applies one shared MLP per token and then mean+max pools, so
        # position is discarded and two cards with equal properties are literally the same
        # vector to everything downstream. Measured: 110 cards collapse to 46 signatures and 86%
        # of them collide with another card, in groups of up to six -- Tear Down this Wall is
        # indistinguishable from Chernobyl.
        #
        # A learned embedding indexed by position restores it, and is model-side only: the
        # observation is untouched, so this is not an observation change.
        self.identity_dim = int(identity_dim)
        if self.identity_dim > 0:
            self.card_identity = nn.Embedding(110, self.identity_dim)
            self.country_identity = nn.Embedding(84, self.identity_dim)
        else:
            self.card_identity = None
            self.country_identity = None

        # 1. Board Graph Encoder (84 nodes x 26 features -> 64)
        self.self_transform = bool(self_transform)
        # How far information travels on the map. Adjacency's *mechanical* uses are already
        # precomputed per country in the observation -- placement legality, coup legality and the
        # realignment modifier are all slots -- so what the graph adds is strategic reasoning
        # about neighbourhoods, and that is worth measuring rather than assuming. With a
        # self-transform the second layer consistently loses per-country influence and has not
        # been shown to buy anything, so 1 and 0 are the interesting settings. 0 keeps a
        # per-country encoder and drops adjacency entirely, which isolates the relation itself.
        self.graph_layers = int(graph_layers)
        if not 0 <= self.graph_layers <= 2:
            raise ValueError(f"graph_layers must be 0, 1 or 2; got {graph_layers}")
        board_in = self.board_features + self.identity_dim
        if self.graph_layers >= 1:
            self.gconv1 = GraphConvLayer(board_in, 64, self_transform=self.self_transform)
        if self.graph_layers >= 2:
            self.gconv2 = GraphConvLayer(64, 64, self_transform=self.self_transform)
        if self.graph_layers == 0:
            self.board_fc = nn.Sequential(nn.Linear(board_in, 64), nn.GELU())
        self.board_proj = nn.Sequential(
            nn.Linear(64 * 2, 256),  # Mean + Max pooling over 84 nodes
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 2. Card Registry Encoder (110 cards x 14 features -> 64)
        self.card_fc = nn.Sequential(
            nn.Linear(self.card_features + self.identity_dim, 64),
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

        # End-of-trunk attention read-out. The board branch pools across all 84 countries before
        # the trunk is formed, and that pooling is lossy: a linear probe recovers a country's
        # exact influence from its pre-pooling token far more often than from the 512-float
        # trunk, where it is essentially gone. Every head reads only the trunk, so the board is
        # gone by the time anything decides where to place.
        #
        # This lets the trunk, once it has an estimate of the situation, go back and look at
        # specific countries and cards. The keys and values carry each entity's **raw**
        # observation slots alongside its encoded token, because the token is itself already
        # damaged -- the graph convolution costs a third before pooling costs the rest -- so
        # attending only over tokens would inherit that loss.
        # Per-entity policy heads. No read-out ending in one fixed-size summary can carry 84
        # countries to a head: a single query over 84 countries returns one weighted average, so
        # the attention read-out above adds nothing beyond what better tokens give it. A
        # country's exact influence is almost entirely recoverable from its own token and almost
        # entirely absent from the pooled trunk, so the remaining way to get the board into a
        # decision is to stop routing it through the trunk -- a country's logit is computed from
        # that country's token.
        self.per_entity_heads = int(per_entity_heads)
        if self.per_entity_heads > 0:
            d = self.per_entity_heads
            self.pe_trunk = nn.Linear(hidden_dim, d)
            self.pe_country = nn.Sequential(
                nn.Linear(64 + self.board_features + self.identity_dim + d, d), nn.GELU(),
                nn.Linear(d, 1))
            self.pe_card = nn.Sequential(
                nn.Linear(64 + self.card_features + self.identity_dim + d, d), nn.GELU(),
                nn.Linear(d, 1))
            # Zero the output layers so the correction is exactly 0 at initialisation and the
            # network begins as the dense baseline rather than as a perturbation of it.
            for head in (self.pe_country, self.pe_card):
                out_layer = head[-1]
                assert isinstance(out_layer, nn.Linear)
                nn.init.zeros_(out_layer.weight)
                nn.init.zeros_(out_layer.bias)

        self.attn_readout = int(attn_readout)
        if self.attn_readout > 0:
            d = self.attn_readout
            self.ro_query = nn.Linear(hidden_dim, d)
            self.ro_country_kv = nn.Linear(64 + self.board_features + self.identity_dim, 2 * d)
            self.ro_card_kv = nn.Linear(64 + self.card_features + self.identity_dim, 2 * d)
            self.ro_out = nn.Sequential(
                nn.Linear(hidden_dim + 2 * d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        # 7. Masked Policy Head (212 Actions)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.ACTION_SPACE_SIZE),
        )

        # 8. Value head(s).
        #
        # Scalar (the default, and what every existing checkpoint carries): win/loss utility in
        # [-1, 1] and an auxiliary VP estimate in [-20, 20], each regressed with MSE.
        #
        # Categorical (P1): one distribution over final VP on integer atoms spanning [-20, +20],
        # trained by cross-entropy. Outcomes here are multimodal -- a coup hits or misses, the
        # scoring card is or is not in hand -- and MSE on a scalar regresses to the mean of the
        # modes, a value that is never observed. Both scalars are still exposed, so nash_pg, the
        # tournament code and every probe run unchanged -- but only one of them is derived:
        #   v_vp  = E[VP] / VP_LIMIT, from the distribution
        #   v_win = its own regressed scalar head, unchanged from the scalar variant
        # The support is exact rather than a modelling choice: the engine normalises every abrupt
        # ending (20 VP, DEFCON 1, Europe Control) to exactly +/-20.
        self.categorical_value = bool(categorical_value)
        self.value_atoms = int(value_atoms)
        if self.categorical_value:
            # The win head stays. It is the baseline GAE subtracts, and deriving it from the
            # distribution as P(VP>0) - P(VP<0) reads only the distribution's *sign*: a head
            # that merely leans already returns +/-1, so the baseline saturates long before it
            # is accurate. Measured on a 24-iteration net, the derived form put |v_win| above
            # 0.9 in 49.6% of states where the regressed head never passed 0.8, which inflated
            # returns_win, doubled adv_std_raw, and left the normalised advantage carrying
            # proportionally less of the action difference. The distribution is therefore
            # additive here -- it replaces the scalar *VP* head, not the win head.
            self.val_win_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Tanh(),
            )
            self.val_vp_head = None
            self.value_dist_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, self.value_atoms),
            )
            self.register_buffer(
                "value_support",
                torch.linspace(-VP_LIMIT, VP_LIMIT, self.value_atoms, dtype=torch.float32))
        else:
            self.value_dist_head = None
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

    def _value_scalars(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(v_win, v_vp), from whichever value head this instance has.

        Keeping the derivation here means every caller -- forward, forward_with_risk, the
        tournament code, the probes -- sees the same two tensors it always did, whichever head
        produced them.
        """
        if not self.categorical_value:
            assert self.val_win_head is not None and self.val_vp_head is not None
            return self.val_win_head(h), self.val_vp_head(h)
        assert self.value_dist_head is not None and self.val_win_head is not None
        probs = torch.softmax(self.value_dist_head(h), dim=-1)
        support = self.value_support.to(probs.dtype)
        # E[VP] divided by VP_LIMIT, because `v_vp` is a *normalised* quantity by contract --
        # rollout_buffer stores returns_vp in [-1, 1] and bootstraps with
        # `last_ret_vp = last_v_vp.clone()`, so a v_vp in real VP units injects a value twenty
        # times too large at the buffer boundary. The support stays in real VP: that is what
        # makes +/-20 land on the end atoms and the multimodality meaningful. Only what leaves
        # this method is rescaled, so the scalar and categorical heads present one contract.
        v_vp = (probs * support).sum(dim=-1, keepdim=True) / float(VP_LIMIT)
        # v_win comes from its own regressed head, not from the sign mass of this
        # distribution -- see the note where the heads are built.
        return self.val_win_head(h), v_vp

    def forward_with_value_logits(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """forward() plus the raw value-distribution logits, from one backbone pass.

        The cross-entropy loss needs the logits, and re-running the trunk to get them would
        double the cost of every update. `None` on a scalar-headed model.
        """
        h, _attn, tokens = self._encode(obs)
        raw_logits = self._policy_logits(h, tokens)
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            masked_logits = torch.where(
                mask_bool, raw_logits,
                torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype))
        else:
            masked_logits = raw_logits
        v_win, v_vp = self._value_scalars(h)
        value_logits = None
        if self.categorical_value:
            assert self.value_dist_head is not None
            value_logits = self.value_dist_head(h)
        return masked_logits, v_win, v_vp, value_logits

    def two_hot(self, vp: torch.Tensor) -> torch.Tensor:
        """Project VP-valued targets onto the atom support as a two-hot distribution.

        `vp` is (B,) or (B, 1) in VP units. Values outside [-20, +20] are clamped rather than
        dropped -- the engine cannot produce them, so anything outside is a bug upstream and
        clamping keeps it visible as a pile-up on the end atom rather than a crash.
        """
        support = self.value_support.to(vp.dtype)
        atoms = support.numel()
        step = (support[-1] - support[0]) / (atoms - 1)
        x = vp.reshape(-1).clamp(float(support[0]), float(support[-1]))
        pos = (x - support[0]) / step
        lower = pos.floor().clamp(0, atoms - 1).long()
        upper = (lower + 1).clamp(max=atoms - 1)
        upper_w = pos - lower.to(pos.dtype)
        target = torch.zeros(x.shape[0], atoms, device=vp.device, dtype=vp.dtype)
        target.scatter_add_(1, lower.unsqueeze(1), (1.0 - upper_w).unsqueeze(1))
        target.scatter_add_(1, upper.unsqueeze(1), upper_w.unsqueeze(1))
        return target

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
        """The backbone, returning the trunk *and* the per-entity tokens it pooled away.

        `extract_features` keeps its old contract and returns only the trunk, because every
        probe and every caller reads that. The tokens come back separately because the
        per-entity policy heads need them: a country's exact influence is almost entirely
        recoverable from its own token and almost entirely absent from the pooled trunk, so a
        head that reads only the trunk cannot see the board however it is pooled.
        """
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
        if self.country_identity is not None:
            ids = self.country_identity.weight.unsqueeze(0).expand(batch_size, -1, -1)
            board_nodes = torch.cat([board_nodes, ids], dim=-1)
        if self.graph_layers == 0:
            h_board = self.board_fc(board_nodes)       # (B, 84, 64), no adjacency at all
        else:
            h_board = self.gconv1(board_nodes, self.norm_adj)
            if self.graph_layers >= 2:
                h_board = self.gconv2(h_board, self.norm_adj)  # (B, 84, 64)
        board_mean = torch.mean(h_board, dim=1)  # (B, 64)
        board_max, _ = torch.max(h_board, dim=1)  # (B, 64)
        e_board = self.board_proj(torch.cat([board_mean, board_max], dim=-1))  # (B, 256)

        # 2. Card Features: (B, 110, card_features)
        card_raw = obs[:, self.CARD_OFFSET : self.CARD_OFFSET + self.CARD_SIZE]
        card_nodes = card_raw.view(batch_size, 110, self.card_features)
        if self.card_identity is not None:
            ids = self.card_identity.weight.unsqueeze(0).expand(batch_size, -1, -1)
            card_nodes = torch.cat([card_nodes, ids], dim=-1)
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

        if self.attn_readout > 0:
            d = self.attn_readout
            q = self.ro_query(h).unsqueeze(1)                       # (B, 1, d)
            reads = []
            for kv_proj, tokens, raw in ((self.ro_country_kv, h_board, board_nodes),
                                         (self.ro_card_kv, h_cards, card_nodes)):
                kv = kv_proj(torch.cat([tokens, raw], dim=-1))      # (B, N, 2d)
                k, v = kv[..., :d], kv[..., d:]
                w = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (d ** 0.5), dim=-1)
                reads.append(torch.matmul(w, v).squeeze(1))         # (B, d)
            h = self.ro_out(torch.cat([h, *reads], dim=-1))

        return h, attn_weights, (h_board, board_nodes, h_cards, card_nodes)

    def extract_features(self, obs: torch.Tensor, return_attn_weights: bool = False):
        """Extracts fused latent state representation and optional cross-attention maps."""
        h, attn_weights, _tokens = self._encode(obs)
        if return_attn_weights:
            return h, attn_weights
        return h

    def _policy_logits(self, h: torch.Tensor,
                       tokens: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        """The 212 action logits, per-entity where the action names an entity.

        Actions 0..109 are cards and 119..202 are countries; the other 18 -- play mode, timing,
        op mode, branch, confirm -- name no entity and stay dense off the trunk.

        Without `per_entity_heads` every logit comes from the 512-float trunk. Each card and
        country still has its own output row, so it can be *preferred*, but the only channel
        carrying which country is in what state into that choice is a vector that 21.13 measured
        as holding almost none of it. Here a country's logit is computed from that country's own
        token and raw slots, conditioned on the trunk.
        """
        base = self.policy_head(h)
        if not self.per_entity_heads:
            return base
        if tokens is None:
            raise RuntimeError(
                "per-entity heads need the entity tokens, and this backbone produced none. A "
                "backbone that reads the flat observation has no tokens to read, which is why "
                "the constructor refuses the combination rather than reaching here.")
        h_board, board_nodes, h_cards, card_nodes = tokens
        ctx = self.pe_trunk(h).unsqueeze(1)
        card_in = torch.cat([h_cards, card_nodes, ctx.expand(-1, 110, -1)], dim=-1)
        country_in = torch.cat([h_board, board_nodes, ctx.expand(-1, 84, -1)], dim=-1)
        # A *correction* on the dense logit, not a replacement for it.
        #
        # The replacing form was measured and it cost 219 Elo. Routing a logit through
        # `pe_trunk` alone made a 64-float projection the only path from the trunk to that logit,
        # where the dense head reads all 512 -- so each logit gained its own entity's detail and
        # lost seven eighths of its view of the situation. That trade is far worse than the
        # per-country blindness it was meant to fix.
        #
        # Added instead, with the correction's last layer zero-initialised, the network *starts*
        # as the dense baseline exactly and learns a per-entity refinement on top. The full trunk
        # still reaches every logit through `base`; the narrow context now limits only how much
        # situation the correction itself can see.
        zeros_9 = base[:, 110:119] * 0.0
        zeros_end = base[:, 203:212] * 0.0
        correction = torch.cat([
            self.pe_card(card_in).squeeze(-1),         # 0..109
            zeros_9,                                    # play mode, timing, op mode: dense only
            self.pe_country(country_in).squeeze(-1),    # 119..202
            zeros_end,                                  # branch, confirm: dense only
        ], dim=-1)
        return base + correction

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
        h, _attn, tokens = self._encode(obs)
        raw_logits = self._policy_logits(h, tokens)

        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            masked_logits = torch.where(
                mask_bool, raw_logits, torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype)
            )
        else:
            masked_logits = raw_logits

        v_win, v_vp = self._value_scalars(h)
        return masked_logits, v_win, v_vp

    def forward_with_risk(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward() plus the auxiliary DEFCON-risk logit, sharing one backbone pass.

        Kept separate from forward() so the existing three-value contract, which every
        other caller depends on, is untouched.
        """
        h, _attn, tokens = self._encode(obs)
        raw_logits = self._policy_logits(h, tokens)
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            masked_logits = torch.where(
                mask_bool, raw_logits,
                torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype),
            )
        else:
            masked_logits = raw_logits
        v_win, v_vp = self._value_scalars(h)
        return masked_logits, v_win, v_vp, self.defcon_risk_head(h)

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
                          board_features: int = ColdWarNetV2.BOARD_FEATURES,
                          categorical_value: bool = False,
                          identity_dim: int = 0,
                          self_transform: bool = False,
                          attn_readout: int = 0,
                          per_entity_heads: int = 0,
                          graph_layers: int = 2) -> ColdWarNetV2:
    """Factory helper to instantiate ColdWarNetV2 on specified device.

    The defaults are observation layout v2.3, which is the only layout the engine emits.
    `categorical_value` is the P1 value head: it changes the output side, not the input, but it
    does change the checkpoint shape, so a run must be loaded with the setting it trained with.
    """
    model = ColdWarNetV2(hidden_dim=512, num_res_blocks=4, num_attn_heads=4,
                         identity_dim=identity_dim,
                         self_transform=self_transform, attn_readout=attn_readout,
                         per_entity_heads=per_entity_heads,
                         graph_layers=graph_layers,
                         card_features=card_features, use_history=use_history,
                         global_features=global_features, has_tail=has_tail,
                         board_features=board_features,
                         categorical_value=categorical_value)
    return model.to(device)


#: Per-entity observation slots that never change during a game. A shared-weight encoder needs
#: them -- its tokens are permutation-equivalent, so static properties are what tell Poland from
#: Brazil. A positional reader does not: slot i*26+3 is Poland's stability every time, so the
#: value adds a constant that the bias already supplies while occupying input width.
#:
#: Board: stability, battleground, the six region one-hots, and the Europe/SE-Asia sub-flags.
#: Card: Ops, era, one-time, is-scoring. Card slot 9 is the card's side *relative to the viewer*
#: and flips with perspective, so it is not static and stays.
STATIC_BOARD_SLOTS: tuple[int, ...] = (3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18)
STATIC_CARD_SLOTS: tuple[int, ...] = (8, 10, 11, 12)


def static_input_mask(board_features: int = ColdWarNetV2.BOARD_FEATURES,
                      card_features: int = ColdWarNetV2.CARD_FEATURES,
                      total: int = ColdWarNetV2.TOTAL_OBS_SIZE) -> torch.Tensor:
    """True where the observation dimension never varies within a game."""
    mask = torch.zeros(total, dtype=torch.bool)
    for c in range(84):
        for slot in STATIC_BOARD_SLOTS:
            mask[c * board_features + slot] = True
    off = 84 * board_features
    for i in range(110):
        for slot in STATIC_CARD_SLOTS:
            mask[off + i * card_features + slot] = True
    return mask


class ColdWarNetMLP(ColdWarNetV2):
    """The same heads and the same observation, on a plain MLP trunk.

    A control for the backbone, not a candidate. Everything structured about v2 -- the graph
    convolution over the map, the per-card encoder, the card-to-country cross-attention -- is
    replaced by two dense layers over the flat 3,824 floats. The heads, the action space, the
    value contract and the training recipe are untouched, so the difference between this and v2
    at matched steps is what the structure is worth.

    It is expected to be much faster per step and much weaker. If it is *not* much weaker, the
    structure is not earning its cost and the comparison is worth more than the arm.
    """

    def __init__(self, hidden_dim: int = 512, mlp_width: int = 1024,
                 drop_static: bool = False, **kwargs):
        if int(kwargs.get("per_entity_heads", 0)) > 0:
            raise ValueError(
                "ColdWarNetMLP cannot carry per-entity policy heads: it reads the flat "
                "observation and never forms entity tokens for them to read. Refused here "
                "rather than silently producing a model with dense heads under a flag that "
                "says otherwise.")
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        # Optionally drop the static per-entity slots. They are the price a shared-weight
        # encoder pays to tell its tokens apart; a positional reader gets identity from the
        # offset and the same values only inflate the first layer.
        self.drop_static = bool(drop_static)
        # Replaced, not merely bypassed: leaving them registered would put a few hundred
        # thousand never-updated parameters in the checkpoint and in any count of "how big is
        # this network", which is the one number this control exists to make honest.
        for name in ("gconv1", "gconv2", "board_proj", "card_fc", "card_proj",
                     "cross_attn", "cross_attn_ln", "cross_card_proj", "global_proj",
                     "hist_conv"):
            setattr(self, name, None)
        self.mlp_width = int(mlp_width)
        if self.drop_static:
            keep = ~static_input_mask(self.board_features, self.card_features,
                                      self.TOTAL_OBS_SIZE)
            # Not persistent: it is derived from drop_static, not learned, and putting it in
            # the state dict breaks every checkpoint saved before it existed -- which is exactly
            # what happened to the E3-09 MLP arms the moment this buffer was added.
            self.register_buffer("keep_idx", torch.nonzero(keep).squeeze(-1),
                                 persistent=False)
            in_width = int(keep.sum())
        else:
            self.register_buffer("keep_idx", torch.arange(self.TOTAL_OBS_SIZE),
                                 persistent=False)
            in_width = self.TOTAL_OBS_SIZE
        self.mlp_in = nn.Sequential(
            nn.Linear(in_width, self.mlp_width),
            nn.LayerNorm(self.mlp_width),
            nn.GELU(),
            nn.Linear(self.mlp_width, self.mlp_width),
            nn.LayerNorm(self.mlp_width),
            nn.GELU(),
        )
        self.fusion_in = nn.Sequential(
            nn.Linear(self.mlp_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
        """No entity tokens: this backbone reads the flat vector and never forms any.

        It therefore cannot carry per-entity policy heads, and `per_entity_heads` is refused in
        the constructor rather than silently ignored here.
        """
        if obs.shape[-1] != self.TOTAL_OBS_SIZE:
            raise ValueError(
                f"observation is {obs.shape[-1]} floats wide; this model reads "
                f"{self.TOTAL_OBS_SIZE}.")
        h = self.fusion_in(self.mlp_in(torch.index_select(obs, 1, self.keep_idx)))
        for block in self.res_blocks:
            h = block(h)
        return h, None, None

    def extract_features(self, obs: torch.Tensor, return_attn_weights: bool = False):
        h, attn, _tokens = self._encode(obs)
        if return_attn_weights:
            return h, attn
        return h


def create_coldwar_net_mlp(device: torch.device | str,
                           categorical_value: bool = False,
                           drop_static: bool = False) -> ColdWarNetMLP:
    """The MLP backbone control. Same heads, same observation, no structure."""
    dev = torch.device(device) if isinstance(device, str) else device
    return ColdWarNetMLP(categorical_value=categorical_value,
                         drop_static=drop_static).to(dev)


def create_like(model: nn.Module, device: torch.device | str = "cpu") -> ColdWarNetV2:
    """A network shaped exactly like `model`.

    Every dimension is read off the model rather than passed in. A frozen evaluation copy built by
    listing arguments has to list *all* of them, and twice a run has died at its first snapshot
    because one was left at a factory default -- the second time on a different argument, after
    the first had been fixed. Reading them from the source removes the class of error rather than
    the instance.
    """
    # The backbone is a dimension too. An MLP-trunk model copied as a v2 has a different state
    # dict and fails to load at the first snapshot -- the same failure the paragraph above is
    # about, one level up.
    if isinstance(model, ColdWarNetMLP):
        return ColdWarNetMLP(
            hidden_dim=int(getattr(model, "hidden_dim", 512)),
            mlp_width=int(getattr(model, "mlp_width", 1024)),
            drop_static=bool(getattr(model, "drop_static", False)),
            card_features=int(getattr(model, "card_features", ColdWarNetV2.CARD_FEATURES)),
            use_history=bool(getattr(model, "use_history", False)),
            global_features=int(getattr(model, "GLOBAL_SIZE", ColdWarNetV2.GLOBAL_SIZE)),
            has_tail=bool(getattr(model, "has_tail", False)),
            board_features=int(getattr(model, "board_features", ColdWarNetV2.BOARD_FEATURES)),
            categorical_value=bool(getattr(model, "categorical_value", False)),
            identity_dim=int(getattr(model, "identity_dim", 0)),
        ).to(torch.device(device) if isinstance(device, str) else device)
    return create_coldwar_net_v2(
        device,
        card_features=int(getattr(model, "card_features", ColdWarNetV2.CARD_FEATURES)),
        use_history=bool(getattr(model, "use_history", False)),
        global_features=int(getattr(model, "GLOBAL_SIZE", ColdWarNetV2.GLOBAL_SIZE)),
        has_tail=bool(getattr(model, "has_tail", False)),
        board_features=int(getattr(model, "board_features", ColdWarNetV2.BOARD_FEATURES)),
        # The P1 head must be carried too: a frozen evaluation copy built without it would have
        # a different state dict from the model it is copying, which is how arms E and F died.
        categorical_value=bool(getattr(model, "categorical_value", False)),
        identity_dim=int(getattr(model, "identity_dim", 0)),
        self_transform=bool(getattr(model, "self_transform", False)),
        attn_readout=int(getattr(model, "attn_readout", 0)),
        per_entity_heads=int(getattr(model, "per_entity_heads", 0)),
        graph_layers=int(getattr(model, "graph_layers", 2)))


def check_checkpoint_layout(state_dict: dict) -> None:
    """Refuses a checkpoint that was not trained on the engine's one observation layout.

    Derived from the weights rather than from a recorded name, because a checkpoint is a bare
    state dict. A model handed the wrong width does not fail on its own -- it reads fixed slices,
    so the observation is silently misread and the network merely plays badly.

    Checkpoints from the retired layouts cannot be run and are not being converted: legacy, v2.1
    and v2.2 all predate the starred-card fix, so they were trained against a different game and
    are not comparable to anything measured now.
    """
    # An MLP-trunk checkpoint has no per-card block to read the layout off. It has something
    # better: the first dense layer takes the whole observation, so its input width *is* the
    # layout, with nothing to infer.
    mlp_w = state_dict.get("mlp_in.0.weight")
    if mlp_w is not None:
        width = int(mlp_w.shape[1])
        # --drop-static narrows this layer by the static per-entity slots, so the narrowed
        # width is as valid a v2.3 checkpoint as the full one.
        narrowed = ColdWarNetV2.TOTAL_OBS_SIZE - int(static_input_mask().sum())
        if width not in (ColdWarNetV2.TOTAL_OBS_SIZE, narrowed):
            raise ValueError(
                f"this checkpoint reads {width} floats; layout v2.3 is "
                f"{ColdWarNetV2.TOTAL_OBS_SIZE}. Checkpoints from the retired layouts cannot "
                f"be run.")
        return
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
    # Identity embeddings widen this layer by identity_dim, so subtract what they added --
    # otherwise a 14-feature checkpoint with a 16-wide embedding reads as 30 and is refused as a
    # retired layout.
    ident = state_dict.get("card_identity.weight")
    extra = int(ident.shape[1]) if ident is not None else 0
    for key in ("card_fc.0.weight", "module.card_fc.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1]) - extra
    raise KeyError("no card_fc.0.weight in the checkpoint; cannot tell the layout")


