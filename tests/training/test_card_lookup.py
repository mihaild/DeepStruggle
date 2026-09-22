"""P22: the identity-keyed card lookup.

The properties asserted here are the ones that would fail silently. A lookup that never reads
location, or whose config cannot be recovered from its weights, trains perfectly well and is
simply a different experiment from the one the run's name claims.
"""

import torch

from ai.models.ladder_net import (CARD_LOCATION_SLOTS, create_ladder_net,
                                  ladder_config_from_state_dict)

BASE = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
            hidden_dim=480, num_res_blocks=4, drop_static=True, per_entity_heads=64,
            head_entities="country", identity_dim=0, card_self_attention=False,
            cross_attention=False, num_attn_heads=4, categorical_value=False,
            head_context=True, head_static=True)
LOOKUP = dict(card_lookup=True, card_lookup_heads=4, card_lookup_dim=32,
              card_lookup_identity_dim=16)
OFF = dict(card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
           card_lookup_identity_dim=0)


def _obs(n: int = 2) -> torch.Tensor:
    return torch.randn(n, 3824)


class TestShape:
    def test_lookup_is_additive_and_small(self) -> None:
        """It must not be a capacity story. The width probe spent +26.7% parameters and LOST
        129-291 Elo, so a gain from +5% would be hard to attribute to capacity -- but only if the
        delta really is +5%."""
        off = create_ladder_net("cpu", **BASE, **OFF)
        on = create_ladder_net("cpu", **BASE, **LOOKUP)
        n_off = sum(p.numel() for p in off.parameters())
        n_on = sum(p.numel() for p in on.parameters())
        assert n_on > n_off
        assert (n_on / n_off) - 1.0 < 0.07, f"lookup grew the model {(n_on/n_off)-1:.1%}"

    def test_trunk_card_projection_is_untouched(self) -> None:
        """The lookup is a parallel path over the RAW rows. If it changed the trunk's own card
        projection it would be two changes at once, which is what the ladder exists to avoid."""
        off = create_ladder_net("cpu", **BASE, **OFF)
        on = create_ladder_net("cpu", **BASE, **LOOKUP)
        assert on.lad_card[0].in_features == off.lad_card[0].in_features
        assert on.lad_card[0].out_features == off.lad_card[0].out_features

    def test_forward_runs_and_is_finite(self) -> None:
        m = create_ladder_net("cpu", **BASE, **LOOKUP)
        out = m(_obs())
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.shape[-1] == 220
        assert torch.isfinite(logits).all()


class TestItActuallyLooksUpLocation:
    def test_output_responds_to_a_card_moving(self) -> None:
        """Values are the full row, so changing a card's LOCATION must change the retrieval.
        A lookup keyed and valued only on static properties would be a constant."""
        torch.manual_seed(0)
        m = create_ladder_net("cpu", **BASE, **LOOKUP).eval()
        obs = _obs(1)
        card0 = m.CARD_OFFSET
        with torch.no_grad():
            a = m(obs)[0].clone()
            moved = obs.clone()
            # Card 0: clear its location one-hot and put it somewhere else.
            for s in CARD_LOCATION_SLOTS:
                moved[0, card0 + s] = 0.0
            moved[0, card0 + 3] = 1.0          # DISCARD
            b = m(moved)[0]
        assert not torch.allclose(a, b), "moving a card changed nothing; the lookup is inert"

    def test_identity_changes_what_is_addressable(self) -> None:
        """With identity the keys separate cards that share every property; without it they
        cannot -- Europe Scoring and Asia Scoring are identical on ops, era and is_scoring."""
        with_id = create_ladder_net("cpu", **BASE, **LOOKUP)
        without = create_ladder_net("cpu", **BASE, **dict(LOOKUP, card_lookup_identity_dim=0))
        assert with_id.cl_identity is not None
        assert without.cl_identity is None
        assert with_id.cl_key is not None and without.cl_key is not None
        assert with_id.cl_key.in_features == without.cl_key.in_features + 16


class TestBothQuestionShapesAreAddressable:
    """Keys carry the full row, so a query can match an identity OR a location.

    These check the mechanism at INITIALISATION -- whether a query exists that reaches the
    intended cards. Whether SGD finds it is what the arm measures. They are here because the
    first build failed this: identity initialised at 0.02 against 0/1 row slots, ~50x smaller,
    and a query built from card 12's own identity ranked it 107th of 110.
    """

    MY_HAND = 1
    IS_SCORING = 12

    def _rows(self) -> torch.Tensor:
        rows = torch.zeros(1, 110, 14)
        rows[0, :, 0] = 1.0                                  # everything in the deck
        for c in (10, 11, 12):
            rows[0, c, 0] = 0.0
            rows[0, c, self.MY_HAND] = 1.0
        rows[0, 12, self.IS_SCORING] = 1.0
        return rows

    def _rank(self, m, rows, q_in: torch.Tensor, target: int) -> int:
        assert m.cl_identity is not None and m.cl_key is not None
        ident = m.cl_identity.detach()
        keys = m.cl_key(torch.cat([ident.unsqueeze(0), rows], dim=-1))
        nh, dk = m.card_lookup_heads, m.card_lookup_dim
        kh = keys.view(1, 110, nh, dk).transpose(1, 2)
        qh = m.cl_key(q_in).view(1, 1, nh, dk).transpose(1, 2)
        w = torch.softmax((qh @ kh.transpose(-2, -1)) / (dk ** 0.5), dim=-1)[0, :, 0, :].mean(0)
        return int((w > w[target]).sum().item())

    def test_an_identity_query_reaches_its_own_card(self) -> None:
        """Uniform attention would rank it ~55th of 110."""
        rows = self._rows()
        ranks = []
        for seed in range(8):
            torch.manual_seed(seed)
            m = create_ladder_net("cpu", **BASE, **LOOKUP).eval()
            assert m.cl_identity is not None
            q = torch.cat([m.cl_identity.detach()[12].view(1, 1, -1), torch.zeros(1, 1, 14)],
                          dim=-1)
            ranks.append(self._rank(m, rows, q, 12))
        assert max(ranks) <= 10, f"identity addressing is not working: ranks {ranks}"

    def test_a_location_query_reaches_the_hand(self) -> None:
        """This is the question that was impossible before location entered the keys: with it
        only in the values there is nothing for a location query to match against."""
        rows = self._rows()
        ranks = []
        for seed in range(8):
            torch.manual_seed(seed)
            m = create_ladder_net("cpu", **BASE, **LOOKUP).eval()
            q = torch.zeros(1, 1, 16 + 14)
            q[0, 0, 16 + self.MY_HAND] = 1.0
            ranks.append(self._rank(m, rows, q, 12))      # 12 is one of the three hand cards
        assert max(ranks) <= 4, f"location addressing is not working: ranks {ranks}"

    def test_keys_carry_location(self) -> None:
        m = create_ladder_net("cpu", **BASE, **LOOKUP)
        assert m.cl_key is not None
        assert m.cl_key.in_features == 16 + 14, "keys must carry the FULL row, location included"


class TestConfigSurvivesTheCheckpoint:
    def test_config_is_recovered_from_weights(self) -> None:
        """Checkpoints here are bare state dicts and the architecture is detected by weight name.
        A lookup that cannot be recovered would be rebuilt without one and silently misread."""
        m = create_ladder_net("cpu", **BASE, **LOOKUP)
        cfg = ladder_config_from_state_dict(m.state_dict())
        assert cfg is not None
        assert cfg["card_lookup"] is True
        assert cfg["card_lookup_identity_dim"] == 16
        assert cfg["card_lookup_heads"] * cfg["card_lookup_dim"] == 4 * 32

    def test_a_lookup_checkpoint_round_trips(self) -> None:
        m = create_ladder_net("cpu", **BASE, **LOOKUP)
        cfg = ladder_config_from_state_dict(m.state_dict())
        assert cfg is not None
        rebuilt = create_ladder_net("cpu", **cfg)
        rebuilt.load_state_dict(m.state_dict(), strict=True)

    def test_a_lookupless_checkpoint_still_recovers(self) -> None:
        m = create_ladder_net("cpu", **BASE, **OFF)
        cfg = ladder_config_from_state_dict(m.state_dict())
        assert cfg is not None
        assert cfg["card_lookup"] is False
        create_ladder_net("cpu", **cfg).load_state_dict(m.state_dict(), strict=True)
