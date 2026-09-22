"""LadderNet — one backbone that can express every rung of P21.

`research/plans/P21_architecture_ladder.md` builds the architecture up from a flat MLP one
mechanism at a time, so that each mechanism gets a matched control by construction. That needs a
backbone whose structure is *configuration*, not a class hierarchy:

    M0  input_mode="flat"                                     a plain MLP
    M1  input_mode="grouped"                                  per-block dense projections
    M2  input_mode="entity",  aggregation="flatten"           shared encoder, position kept
    M3  ... + card_self_attention=True                        cards attend to cards
    M4  ... + cross_attention=True                            cards attend to countries
    M5  ... aggregation="pool"                                the removal that is measured last

and orthogonally `per_entity_heads` and `identity_dim`, which exist to repair the pooling loss
and are therefore tested as a 2x2 against `aggregation` rather than as rungs.

**Every axis is required at construction. There are no defaults.** A defaulted `layout` argument
was the mechanism behind five separate instances of one bug in this repository, because handing a
model the wrong variant returns a number instead of raising. The same discipline applies here: the
configuration is explicit, it is recorded on the model by `ladder_config()`, and `create_like`
rebuilds a copy from that record rather than from a caller's argument list.

`ColdWarNetMLP` already implements the M0 shape and predates this file; `input_mode="flat"`
reproduces it. The older class is left alone so E3-era MLP checkpoints keep loading.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn

from ai.models.coldwar_net_v2 import (STATIC_BOARD_SLOTS, STATIC_CARD_SLOTS,
                                      ColdWarNetV2, static_input_mask)
from bindings.action_encoder import ActionEncoder

#: How the observation is read before the trunk.
INPUT_MODES: Tuple[str, ...] = ("flat", "grouped", "entity")
#: How per-entity tokens become a fixed-size vector. Only meaningful for `input_mode="entity"`.
AGGREGATIONS: Tuple[str, ...] = ("flatten", "pool")
#: Which per-entity heads exist. The card-collision finding predicts `country` carries most of
#: M2's +352.8 Elo, since `pe_card` cannot distinguish 95 of 110 cards without identity.
HEAD_ENTITIES: Tuple[str, ...] = ("both", "country", "card")


class LadderNet(ColdWarNetV2):
    """A P21 rung. All structural axes are explicit; see the module docstring."""

    # The M2 family may switch either head off (`head_entities`) or remove the trunk context
    # (`head_context`), so these are optional here where the base class always builds them.
    pe_trunk: nn.Linear | None          # type: ignore[assignment]
    pe_country: nn.Sequential | None    # type: ignore[assignment]
    pe_card: nn.Sequential | None       # type: ignore[assignment]

    def __init__(self, *,
                 input_mode: str,
                 aggregation: str,
                 entity_dim: int,
                 card_self_attention: bool,
                 cross_attention: bool,
                 per_entity_heads: int,
                 head_context: bool,
                 head_static: bool,
                 head_entities: str,
                 identity_dim: int,
                 card_lookup: bool,
                 card_lookup_heads: int,
                 card_lookup_dim: int,
                 card_lookup_identity_dim: int,
                 drop_static: bool,
                 hidden_dim: int,
                 num_res_blocks: int,
                 entity_proj_dim: int,
                 num_attn_heads: int,
                 categorical_value: bool,
                 **kwargs: Any) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}; got {input_mode!r}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}; got {aggregation!r}")
        if head_entities not in HEAD_ENTITIES:
            raise ValueError(f"head_entities must be one of {HEAD_ENTITIES}; "
                             f"got {head_entities!r}")
        if not per_entity_heads and (not head_context or not head_static
                                     or head_entities != "both"):
            raise ValueError(
                "head_context / head_static / head_entities only mean anything with "
                "--per-entity-heads > 0. Refused rather than silently ignored.")

        tokenless = input_mode in ("flat", "grouped")
        # `grouped` keeps every entity's RAW slots at a fixed offset, so a per-entity head can
        # read country i's own features and reach country i's logit with no learned token space
        # in between -- `pe_country` already takes the raw slots alongside the token. That is the
        # lookup mechanism on its own, uncontaminated by the lossy 26->d compression that a
        # shared encoder imposes.
        raw_tokens = (input_mode == "grouped") and bool(per_entity_heads)
        if tokenless:
            # Refused rather than ignored. A model built with heads it cannot feed would train
            # happily and silently be a different architecture from the one that was asked for.
            if per_entity_heads and input_mode == "flat":
                raise ValueError(
                    "input_mode='flat' never reshapes the observation into entities, so "
                    "per-entity heads have nothing to read. Use 'grouped' for raw-feature "
                    "per-entity heads.")
            if card_self_attention or cross_attention:
                raise ValueError(
                    f"input_mode={input_mode!r} forms no attention tokens. Cross-attention "
                    f"between 14-wide card rows and 26-wide country rows needs kdim/vdim "
                    f"plumbing and a head count dividing 14; that is a separate rung.")
            if identity_dim and not raw_tokens:
                raise ValueError(
                    f"input_mode={input_mode!r} reads each entity at a fixed offset, so position "
                    f"already identifies it for the trunk. An identity embedding is only useful "
                    f"where a function is SHARED across entities -- add --per-entity-heads and "
                    f"it will be given to that head.")
        else:
            # P21: static per-entity slots are what a *shared* encoder uses to tell its tokens
            # apart. Dropping them is only correct where the reader is positional.
            if drop_static:
                raise ValueError(
                    "drop_static is for positional readers (flat/grouped), where a constant "
                    "input is exactly a bias. With a shared per-entity encoder the static slots "
                    "are how it knows what kind of entity it is looking at.")

        super().__init__(hidden_dim=hidden_dim, num_res_blocks=num_res_blocks,
                         num_attn_heads=num_attn_heads, categorical_value=categorical_value,
                         # Built for a raw-token head too: there the identity vector is the
                         # head's only way to tell two same-typed entities apart.
                         identity_dim=(identity_dim if (not tokenless or raw_tokens) else 0),
                         per_entity_heads=0,          # rebuilt below at the right token width
                         graph_layers=0,              # P21 runs without graph convolution
                         self_transform=False,
                         attn_readout=0,
                         **kwargs)

        self.input_mode = str(input_mode)
        self.aggregation = str(aggregation)
        # Canonical: the token width is meaningless without tokens, so a tokenless
        # rung records 0. That makes `ladder_config()` exactly recoverable from the
        # weights, which is what `ladder_config_from_state_dict` relies on.
        self.entity_dim = 0 if tokenless else int(entity_dim)
        self.card_self_attention = bool(card_self_attention)
        self.cross_attention = bool(cross_attention)
        self.drop_static = bool(drop_static)
        self.raw_tokens = bool(raw_tokens)
        self.head_context = bool(head_context)
        self.head_static = bool(head_static)
        self.head_entities = str(head_entities)
        self.entity_proj_dim = int(entity_proj_dim)
        self.ladder_num_attn_heads = int(num_attn_heads)
        self.ladder_per_entity_heads = int(per_entity_heads)
        # Kept explicitly rather than read back off `fusion_in`, so the config
        # record cannot drift from what was asked for.
        self.ladder_hidden_dim = int(hidden_dim)

        # Everything v2 builds that this configuration does not use is *replaced*, not bypassed:
        # a registered-but-unused module puts never-updated parameters into the checkpoint and
        # into any count of how big the network is, which is the number the ladder exists to keep
        # honest.
        for name in ("gconv1", "gconv2", "board_fc", "board_proj", "card_fc", "card_proj",
                     "cross_attn", "cross_attn_ln", "cross_card_proj", "hist_conv",
                     "ro_query", "ro_country_kv", "ro_card_kv", "ro_out"):
            if hasattr(self, name):
                setattr(self, name, None)

        d = int(entity_dim)
        p = self.entity_proj_dim
        # The trunk reads raw slots in tokenless modes; identity, when present, goes to the
        # head alone. In `entity` mode it is concatenated before the shared encoder as usual.
        board_in = self.board_features + (0 if tokenless else self.identity_dim)
        card_in = self.card_features + (0 if tokenless else self.identity_dim)

        if self.input_mode == "flat":
            keep = ~static_input_mask(self.board_features, self.card_features,
                                      self.TOTAL_OBS_SIZE) if self.drop_static \
                else torch.ones(self.TOTAL_OBS_SIZE, dtype=torch.bool)
            self.register_buffer("keep_idx", torch.nonzero(keep).squeeze(-1), persistent=False)
            self.lad_in = _mlp(int(keep.sum()), p)
            fused_width = p

        elif self.input_mode == "grouped":
            if self.drop_static:
                keep = ~static_input_mask(self.board_features, self.card_features,
                                          self.TOTAL_OBS_SIZE)
            else:
                keep = torch.ones(self.TOTAL_OBS_SIZE, dtype=torch.bool)
            b_keep = torch.nonzero(keep[:self.BOARD_SIZE]).squeeze(-1)
            c_keep = torch.nonzero(
                keep[self.CARD_OFFSET:self.CARD_OFFSET + self.CARD_SIZE]).squeeze(-1)
            self.register_buffer("board_keep_idx", b_keep, persistent=False)
            self.register_buffer("card_keep_idx", c_keep, persistent=False)
            self.lad_board = _mlp(int(b_keep.numel()), p)
            self.lad_card = _mlp(int(c_keep.numel()), p)
            fused_width = 2 * p + 128

        else:  # entity
            self.lad_board_enc = nn.Sequential(nn.Linear(board_in, d), nn.GELU())
            self.lad_card_enc = nn.Sequential(nn.Linear(card_in, d), nn.GELU())
            if self.card_self_attention:
                self.lad_self_attn = nn.MultiheadAttention(
                    embed_dim=d, num_heads=num_attn_heads, batch_first=True)
                self.lad_self_ln = nn.LayerNorm(d)
            if self.cross_attention:
                self.lad_cross_attn = nn.MultiheadAttention(
                    embed_dim=d, num_heads=num_attn_heads, batch_first=True)
                self.lad_cross_ln = nn.LayerNorm(d)
            board_agg = 84 * d if self.aggregation == "flatten" else 2 * d
            card_agg = 110 * d if self.aggregation == "flatten" else 2 * d
            self.lad_board = _mlp(board_agg, p)
            self.lad_card = _mlp(card_agg, p)
            if self.cross_attention:
                self.lad_cross = _mlp(card_agg, p)
            fused_width = (3 if self.cross_attention else 2) * p + 128

        # P22: identity-keyed card lookup. A parallel path over the RAW card rows -- the trunk's
        # own card projection is untouched, so this adds a mechanism rather than replacing one.
        self.card_lookup = bool(card_lookup)
        self.card_lookup_heads = int(card_lookup_heads)
        self.card_lookup_dim = int(card_lookup_dim)
        self.card_lookup_identity_dim = int(card_lookup_identity_dim)
        self.cl_identity: nn.Parameter | None = None
        self.cl_key: nn.Linear | None = None
        self.cl_value: nn.Linear | None = None
        self.cl_query: nn.Linear | None = None
        self.cl_out: nn.Linear | None = None
        if self.card_lookup:
            if self.card_lookup_heads <= 0 or self.card_lookup_dim <= 0:
                raise ValueError(
                    f"card_lookup needs positive heads and dim, got "
                    f"{self.card_lookup_heads} and {self.card_lookup_dim}.")
            hk = self.card_lookup_heads * self.card_lookup_dim
            n_props = self.card_features - len(CARD_LOCATION_SLOTS)      # 14 - 8 = 6
            if self.card_lookup_identity_dim:
                # Identity is what makes a card addressable. Without it, Europe Scoring and Asia
                # Scoring are identical on every property -- same ops, same era, both scoring --
                # so a query retrieves an average over the cards it needed to tell apart. The
                # identity-free variant is kept reachable because it is the ablation that
                # attributes the gain, not because it is expected to work.
                self.cl_identity = nn.Parameter(
                    torch.randn(110, self.card_lookup_identity_dim) * 0.02)
            self.cl_key = nn.Linear(self.card_lookup_identity_dim + n_props, hk)
            self.cl_value = nn.Linear(self.card_features, hk)
            # Queried from the PRE-fusion vector, not the trunk: the output is concatenated into
            # fusion_in's input, so querying the trunk would be circular.
            self.cl_query = nn.Linear(fused_width, hk)
            self.cl_out = nn.Linear(hk, hk)
            fused_width += hk

        self.fusion_in = nn.Sequential(
            nn.Linear(fused_width, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        # Per-entity heads at *this* token width. v2 hard-codes 64, which would silently build
        # the wrong shape for any other entity_dim.
        self.per_entity_heads = int(per_entity_heads)
        if self.per_entity_heads:
            k = self.per_entity_heads
            # `ctx` is the ONLY path from the trunk into the correction. Without it the head is a
            # pure per-entity map and the whole mechanism is embarrassingly parallel (arm M2a).
            ctx_w = k if self.head_context else 0
            self.pe_trunk = nn.Linear(hidden_dim, k) if self.head_context else None
            # With raw tokens the raw slots arrive in the `raw` slot, so the token slot carries
            # the identity vector instead -- width 0 when there is no identity.
            tok_w = self.identity_dim if self.raw_tokens else d
            # Dropping the constant slots leaves the dynamic ones, which is where the mechanism's
            # content has to be: constants alone would be a fixed per-TYPE bias (arm M2b).
            b_raw = board_in - (0 if self.head_static else len(STATIC_BOARD_SLOTS))
            c_raw = card_in - (0 if self.head_static else len(STATIC_CARD_SLOTS))
            self.pe_country = (nn.Sequential(nn.Linear(tok_w + b_raw + ctx_w, k), nn.GELU(),
                                             nn.Linear(k, 1))
                               if self.head_entities in ("both", "country") else None)
            self.pe_card = (nn.Sequential(nn.Linear(tok_w + c_raw + ctx_w, k), nn.GELU(),
                                          nn.Linear(k, 1))
                            if self.head_entities in ("both", "card") else None)
            for head in (self.pe_country, self.pe_card):
                if head is None:
                    continue
                out = head[-1]
                assert isinstance(out, nn.Linear)
                nn.init.zeros_(out.weight)
                nn.init.zeros_(out.bias)


    def _card_lookup(self, card_raw: torch.Tensor, pre: torch.Tensor,
                     b: int) -> torch.Tensor:
        """Content-addressed retrieval over the 110 card rows.

        Answers "where is the card I am asking about", where the question is state-dependent: the
        query comes from `pre`, so the lookup can be conditional -- *Europe is negative, therefore
        check Europe Scoring* -- rather than computing all 110 lookups unconditionally the way the
        dense card projection does.
        """
        assert self.cl_key is not None and self.cl_value is not None
        assert self.cl_query is not None and self.cl_out is not None
        rows = card_raw.view(b, 110, self.card_features)
        props = rows[:, :, len(CARD_LOCATION_SLOTS):]                 # (B,110,6)
        if self.cl_identity is not None:
            ident = self.cl_identity.unsqueeze(0).expand(b, -1, -1)
            k_in = torch.cat([ident, props], dim=-1)
        else:
            k_in = props
        nh, dk = self.card_lookup_heads, self.card_lookup_dim
        k = self.cl_key(k_in).view(b, 110, nh, dk).transpose(1, 2)    # (B,nh,110,dk)
        v = self.cl_value(rows).view(b, 110, nh, dk).transpose(1, 2)  # (B,nh,110,dk)
        q = self.cl_query(pre).view(b, nh, 1, dk)                     # (B,nh,1,dk)
        w = torch.softmax((q @ k.transpose(-2, -1)) / (dk ** 0.5), dim=-1)
        return self.cl_out((w @ v).view(b, nh * dk))

    # ------------------------------------------------------------------ config

    def ladder_config(self) -> Dict[str, Any]:
        """The full structural configuration, for the checkpoint and for `create_like`.

        Read off the model, never from a caller's arguments: a frozen evaluation copy built by
        listing arguments has to list all of them, and this repository has twice lost a run at
        its first snapshot because one was left at a factory default.
        """
        return dict(
            input_mode=self.input_mode,
            aggregation=self.aggregation,
            entity_dim=self.entity_dim,
            card_self_attention=self.card_self_attention,
            cross_attention=self.cross_attention,
            per_entity_heads=self.per_entity_heads,
            head_context=self.head_context,
            head_static=self.head_static,
            head_entities=self.head_entities,
            identity_dim=self.identity_dim,
            card_lookup=self.card_lookup,
            card_lookup_heads=self.card_lookup_heads,
            card_lookup_dim=self.card_lookup_dim,
            card_lookup_identity_dim=self.card_lookup_identity_dim,
            drop_static=self.drop_static,
            hidden_dim=self.ladder_hidden_dim,
            num_res_blocks=len(self.res_blocks),
            entity_proj_dim=self.entity_proj_dim,
            num_attn_heads=self.ladder_num_attn_heads,
            categorical_value=bool(self.categorical_value),
        )


    # ----------------------------------------------------- the per-entity correction

    #: Kept slot indices when `head_static` is off, as tensors registered lazily in `_encode`.
    def _dynamic_slice(self, nodes: torch.Tensor, static: Tuple[int, ...]) -> torch.Tensor:
        keep = [i for i in range(nodes.shape[-1]) if i not in static]
        return nodes[..., keep]

    def _policy_logits(self, h: torch.Tensor,
                       tokens: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        """The dense logits plus a per-entity correction, honouring the M2-family axes.

        The inherited version assumes both heads exist and that a trunk context is always fed.
        Here `head_entities` may switch one head off and `head_context` may remove the only path
        from the trunk into the correction -- so the concatenation is built rather than fixed.
        """
        base = self.policy_head(h)
        if not self.per_entity_heads or tokens is None:
            return base

        h_board, board_nodes, h_cards, card_nodes = tokens
        if not self.head_static:
            board_nodes = self._dynamic_slice(board_nodes, STATIC_BOARD_SLOTS)
            card_nodes = self._dynamic_slice(card_nodes, STATIC_CARD_SLOTS)

        parts_country = [h_board, board_nodes]
        parts_card = [h_cards, card_nodes]
        if self.head_context:
            assert self.pe_trunk is not None
            ctx = self.pe_trunk(h).unsqueeze(1)
            parts_country.append(ctx.expand(-1, 84, -1))
            parts_card.append(ctx.expand(-1, 110, -1))

        _A = ActionEncoder
        card_block = base[:, :_A.PLAY_MODE_OFFSET]
        gap_mid = base[:, _A.PLAY_MODE_OFFSET:_A.NODE_OFFSET] * 0.0
        country_block = base[:, _A.NODE_OFFSET:_A.BRANCH_OFFSET]
        gap_end = base[:, _A.BRANCH_OFFSET:] * 0.0

        card_corr = (self.pe_card(torch.cat(parts_card, dim=-1)).squeeze(-1)
                     if self.pe_card is not None else card_block * 0.0)
        country_corr = (self.pe_country(torch.cat(parts_country, dim=-1)).squeeze(-1)
                        if self.pe_country is not None else country_block * 0.0)
        return base + torch.cat([card_corr, gap_mid, country_corr, gap_end], dim=-1)

    # ----------------------------------------------------------------- encoder

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None,
                                                  tuple[torch.Tensor, ...] | None]:
        if obs.shape[-1] != self.TOTAL_OBS_SIZE:
            raise ValueError(
                f"observation is {obs.shape[-1]} floats wide; this model reads "
                f"{self.TOTAL_OBS_SIZE}. Every slice below is taken at a fixed offset, so a "
                f"mismatched vector is misread rather than rejected.")
        b = obs.shape[0]
        attn_weights: torch.Tensor | None = None
        tokens: tuple[torch.Tensor, ...] | None = None

        if self.input_mode == "flat":
            h = self.fusion_in(self.lad_in(
                torch.index_select(obs, 1, cast(torch.Tensor, self.keep_idx))))

        elif self.input_mode == "grouped":
            board_raw = obs[:, :self.BOARD_SIZE]
            card_raw = obs[:, self.CARD_OFFSET:self.CARD_OFFSET + self.CARD_SIZE]
            glob = obs[:, self.GLOBAL_OFFSET:self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
            e_board = self.lad_board(torch.index_select(
                board_raw, 1, cast(torch.Tensor, self.board_keep_idx)))
            e_card = self.lad_card(torch.index_select(
                card_raw, 1, cast(torch.Tensor, self.card_keep_idx)))
            pre = torch.cat([e_board, e_card, self.global_proj(glob)], dim=-1)
            if self.card_lookup:
                pre = torch.cat([pre, self._card_lookup(card_raw, pre, b)], dim=-1)
            h = self.fusion_in(pre)
            if self.raw_tokens:
                # The tokens are the raw slots themselves. `_policy_logits` concatenates
                # (token, raw, trunk context); with raw tokens the first two would be the same
                # vector, so an empty token slice is passed and the head is sized for it.
                b_nodes = board_raw.view(b, 84, self.board_features)
                c_nodes = card_raw.view(b, 110, self.card_features)
                if self.country_identity is not None and self.card_identity is not None:
                    tok_b = self.country_identity.weight.unsqueeze(0).expand(b, -1, -1)
                    tok_c = self.card_identity.weight.unsqueeze(0).expand(b, -1, -1)
                else:
                    tok_b = b_nodes.new_zeros(b, 84, 0)
                    tok_c = c_nodes.new_zeros(b, 110, 0)
                tokens = (tok_b, b_nodes, tok_c, c_nodes)

        else:  # entity
            board_nodes = obs[:, :self.BOARD_SIZE].view(b, 84, self.board_features)
            card_nodes = obs[:, self.CARD_OFFSET:self.CARD_OFFSET + self.CARD_SIZE] \
                .view(b, 110, self.card_features)
            if self.country_identity is not None:
                board_nodes = torch.cat(
                    [board_nodes, self.country_identity.weight.unsqueeze(0).expand(b, -1, -1)],
                    dim=-1)
            if self.card_identity is not None:
                card_nodes = torch.cat(
                    [card_nodes, self.card_identity.weight.unsqueeze(0).expand(b, -1, -1)],
                    dim=-1)

            h_board = self.lad_board_enc(board_nodes)          # (B, 84, d)
            h_cards = self.lad_card_enc(card_nodes)            # (B, 110, d)

            if self.card_self_attention:
                sa, _ = self.lad_self_attn(h_cards, h_cards, h_cards)
                h_cards = self.lad_self_ln(h_cards + sa)

            parts = [self.lad_board(_agg(h_board, self.aggregation)),
                     self.lad_card(_agg(h_cards, self.aggregation))]
            if self.cross_attention:
                ca, attn_weights = self.lad_cross_attn(h_cards, h_board, h_board)
                h_cross = self.lad_cross_ln(h_cards + ca)
                parts.append(self.lad_cross(_agg(h_cross, self.aggregation)))

            glob = obs[:, self.GLOBAL_OFFSET:self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
            parts.append(self.global_proj(glob))
            h = self.fusion_in(torch.cat(parts, dim=-1))
            tokens = (h_board, board_nodes, h_cards, card_nodes)

        for block in self.res_blocks:
            h = block(h)
        return h, attn_weights, tokens

    def extract_features(self, obs: torch.Tensor, return_attn_weights: bool = False):
        h, attn, _tokens = self._encode(obs)
        if return_attn_weights:
            return h, attn
        return h


#: Slots 0..7 of a card row are the location one-hot; 8..13 are ops, rel_side, era, one_time,
#: is_scoring and ACTIVE_CARD. The split is what makes a lookup natural: 8..13 say what a card IS
#: (key material), 0..7 say where it currently is (the value being retrieved).
CARD_LOCATION_SLOTS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)


def _mlp(in_width: int, out_width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_width, out_width), nn.LayerNorm(out_width), nn.GELU())


def _agg(tokens: torch.Tensor, how: str) -> torch.Tensor:
    """(B, N, d) -> (B, N*d) keeping position, or (B, 2d) discarding it."""
    if how == "flatten":
        return tokens.reshape(tokens.shape[0], -1)
    return torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)


def ladder_config_from_state_dict(sd: Dict[str, Any]) -> Dict[str, Any] | None:
    """Recover a rung's configuration from its weights, or None if this is not a LadderNet.

    Checkpoints in this repository are bare state dicts and the architecture is detected by
    weight name, never from a recorded config. That is the stronger contract: a saved config can
    drift from the weights it claims to describe, while shapes cannot.

    **Must be tried before the v2 detection.** `lad_cross_attn.*` contains the substring
    `cross_attn`, so a cross-attention rung matches v2's test and would be rebuilt as a
    ColdWarNetV2 -- loading most tensors, silently dropping the rest, and rating a different
    network than the one that trained.
    """
    if "lad_in.0.weight" in sd:
        input_mode = "flat"
    elif "lad_board_enc.0.weight" in sd:
        input_mode = "entity"
    elif "lad_board.0.weight" in sd:
        input_mode = "grouped"
    else:
        return None

    fusion = sd["fusion_in.0.weight"]
    hidden_dim = int(fusion.shape[0])
    blocks = {k.split(".")[1] for k in sd if k.startswith("res_blocks.")}
    ident = sd.get("country_identity.weight")
    pe = sd.get("pe_trunk.weight")

    if input_mode == "flat":
        proj = sd["lad_in.0.weight"]
        in_width = int(proj.shape[1])
        entity_dim, aggregation = 0, "flatten"
    else:
        proj = sd["lad_board.0.weight"]
        if input_mode == "grouped":
            in_width = int(proj.shape[1]) + int(sd["lad_card.0.weight"].shape[1])
            entity_dim, aggregation = 0, "flatten"
        else:
            entity_dim = int(sd["lad_board_enc.0.weight"].shape[0])
            # 84*d if position was kept, 2*d if mean+max discarded it
            aggregation = "flatten" if int(proj.shape[1]) == 84 * entity_dim else "pool"
            in_width = 0

    drop_static = False
    if input_mode in ("flat", "grouped"):
        mask = static_input_mask()
        static = int(mask.sum())
        # `flat` reads the whole observation; `grouped` reads only board+card, since the global
        # block goes through `global_proj`. Comparing a grouped width against the full-observation
        # total reported drop_static=False for a run that had it on.
        full = (int(mask.numel()) if input_mode == "flat"
                else ColdWarNetV2.BOARD_SIZE + ColdWarNetV2.CARD_SIZE)
        drop_static = (in_width == full - static) and (in_width != full)

    # The M2-family head axes, all recovered from the weights like everything else here.
    has_country = "pe_country.0.weight" in sd
    has_card = "pe_card.0.weight" in sd
    head_entities = ("both" if has_country and has_card
                     else "country" if has_country
                     else "card" if has_card else "both")
    # With no heads at all the three axes are inert, and the constructor refuses anything but
    # these, so they must be recovered as the canonical values rather than inferred from weights
    # that were never built.
    head_context = pe is not None                      # pe_trunk exists only with a context
    # P22 card lookup. Recovered from shapes like everything else: cl_key exists iff the lookup
    # is built, cl_out is square at heads*dim, and the identity width is cl_key's input minus the
    # six property slots. A saved config could disagree with the weights; shapes cannot.
    cl_key = sd.get("cl_key.weight")
    if cl_key is not None:
        card_lookup = True
        hk = int(sd["cl_out.weight"].shape[0])
        cl_ident = sd.get("cl_identity")
        card_lookup_identity_dim = int(cl_ident.shape[1]) if cl_ident is not None else 0
        n_props = int(cl_key.shape[1]) - card_lookup_identity_dim
        # heads*dim is recoverable but not their factorisation; dim is pinned by convention to
        # the value the ladder uses, and heads follow. Recorded in metadata either way.
        card_lookup_dim = 32 if hk % 32 == 0 else hk
        card_lookup_heads = hk // card_lookup_dim
        del n_props
    else:
        card_lookup = False
        card_lookup_heads = card_lookup_dim = card_lookup_identity_dim = 0

    identity_dim = int(ident.shape[1]) if ident is not None else 0
    per_entity_heads = (int(pe.shape[0]) if pe is not None
                        else int(sd["pe_country.0.weight"].shape[0]) if has_country
                        else int(sd["pe_card.0.weight"].shape[0]) if has_card else 0)
    # head_static from the head's input width: token + raw + context, where the raw part is the
    # full slot count or the dynamic remainder.
    head_static = True
    if not (has_country or has_card):
        head_entities, head_context, head_static = "both", True, True
    if has_country or has_card:
        tok_w = identity_dim if input_mode != "entity" else entity_dim
        ctx_w = per_entity_heads if head_context else 0
        if has_country:
            raw_w = int(sd["pe_country.0.weight"].shape[1]) - tok_w - ctx_w
            head_static = raw_w == ColdWarNetV2.BOARD_FEATURES
        else:
            raw_w = int(sd["pe_card.0.weight"].shape[1]) - tok_w - ctx_w
            head_static = raw_w == ColdWarNetV2.CARD_FEATURES

    return dict(
        input_mode=input_mode,
        aggregation=aggregation,
        entity_dim=entity_dim,
        card_self_attention=any(k.startswith("lad_self_attn.") for k in sd),
        cross_attention=any(k.startswith("lad_cross_attn.") for k in sd),
        per_entity_heads=per_entity_heads,
        head_context=head_context,
        head_static=head_static,
        head_entities=head_entities,
        identity_dim=identity_dim,
        card_lookup=card_lookup,
        card_lookup_heads=card_lookup_heads,
        card_lookup_dim=card_lookup_dim,
        card_lookup_identity_dim=card_lookup_identity_dim,
        drop_static=drop_static,
        hidden_dim=hidden_dim,
        num_res_blocks=len(blocks),
        entity_proj_dim=int(proj.shape[0]),
        num_attn_heads=4,
        categorical_value=any(k.startswith("value_dist_head") for k in sd),
    )

def create_ladder_net(device: torch.device | str, **config: Any) -> LadderNet:
    """Build a rung. Every structural axis must be named; see `LadderNet`."""
    dev = torch.device(device) if isinstance(device, str) else device
    return LadderNet(**config).to(dev)
