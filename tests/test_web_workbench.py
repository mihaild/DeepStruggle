import os
import json
import pytest
from starlette.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_map_metadata_completeness():
    """Validates that /api/metadata/map provides complete 84-country Deluxe Map graph with coordinates."""
    res = client.get("/api/metadata/map")
    assert res.status_code == 200, f"Failed to fetch map metadata: {res.status_code}"
    data = res.json()

    countries = data.get("countries", [])
    assert len(countries) == 84, f"Expected 84 countries, got {len(countries)}"

    country_ids = set()
    country_names = set()

    for c in countries:
        cid = c.get("id")
        name = c.get("name")
        assert cid is not None and isinstance(cid, int) and 0 <= cid < 84
        assert name and isinstance(name, str)
        assert cid not in country_ids, f"Duplicate country id: {cid}"
        assert name not in country_names, f"Duplicate country name: {name}"
        country_ids.add(cid)
        country_names.add(name)

        pos = c.get("pos")
        assert pos and len(pos) == 2 and isinstance(pos[0], (int, float)) and isinstance(pos[1], (int, float)),             f"Invalid pos coordinates for {name}: {pos}"

        stab = c.get("stability")
        assert stab is not None and isinstance(stab, int) and 1 <= stab <= 5, f"Invalid stability for {name}: {stab}"

        assert "battleground" in c and isinstance(c["battleground"], bool)
        assert "region" in c
        assert "neighbours" in c and isinstance(c["neighbours"], list)

    regions = data.get("regions", {})
    assert len(regions) >= 6, f"Expected at least 6 regions, got {len(regions)}"
    for r_name, r_info in regions.items():
        assert "region_id" in r_info
        assert "color" in r_info
        assert "battlegrounds" in r_info

def test_cards_metadata_completeness():
    """Validates that /api/metadata/cards returns all 110 Deluxe Edition cards."""
    res = client.get("/api/metadata/cards")
    assert res.status_code == 200, f"Failed to fetch cards metadata: {res.status_code}"
    cards = res.json()
    if isinstance(cards, dict):
        cards = cards.get("cards", [])

    assert len(cards) == 110, f"Expected 110 cards, got {len(cards)}"

    card_ids = set()
    for c in cards:
        cid = c.get("id")
        assert cid is not None and 1 <= cid <= 110
        assert cid not in card_ids, f"Duplicate card id: {cid}"
        card_ids.add(cid)

        name = c.get("name")
        assert name and isinstance(name, str)
        assert "ops" in c and 0 <= c["ops"] <= 4
        assert c.get("side") in ("us", "ussr", "neutral")
        assert c.get("age") in ("early war", "mid war", "middle war", "late war")
        assert "description" in c and len(c["description"]) > 0

def test_frontend_workbench_html_structure():
    """Validates that index.html contains all critical UI elements and SVG map canvas."""
    res = client.get("/")
    assert res.status_code == 200, f"Failed to fetch frontend index: {res.status_code}"
    html = res.text

    critical_ids = [
        "ts-map-svg",
        "map-canvas-wrapper",
        "map-container",
        "global-tracks",
        "decision-panel",
        "decision-body",
        "decision-type-badge",
        "tab-ussr-hand",
        "tab-us-hand",
        "tab-board-cards",
        "tab-all-cards",
        "china-card-container",
        "active-flags-container",
        "context-stack-container",
        "action-log-stream",
        "replay-toolbar",
        "rep-timeline-slider",
        "rep-server-select",
        "btn-rep-play",
        "btn-toggle-replay",
        "modal-container"
    ]

    for elem_id in critical_ids:
        assert f"id=\"{elem_id}\"" in html or f"id='{elem_id}'" in html, f"Missing critical element #{elem_id} in index.html"

def test_replay_state_snapshot_full_fidelity():
    """Validates that replay snapshots deliver complete game state with all 84 countries for map rendering."""
    res = client.get("/api/replays")
    assert res.status_code == 200
    replays = res.json()
    assert len(replays) > 0, "No replays available to test"

    target_filename = "llm_match_with_commentary.tslog.json"
    rep_res = client.get(f"/api/replays/{target_filename}")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()

    steps = rep_data.get("steps", [])
    assert len(steps) > 0, "Replay has 0 steps"

    # Validate first step state snapshot has all country nodes
    first_snapshot = steps[0].get("state_snapshot", {})
    assert "victory_points" in first_snapshot
    assert "defcon" in first_snapshot
    assert "countries" in first_snapshot
    countries = first_snapshot["countries"]
    assert len(countries) == 84, f"Expected 84 countries in state_snapshot, got {len(countries)}"

    # Validate decision context and hands
    assert "decision_context" in first_snapshot
    assert "hands" in first_snapshot
