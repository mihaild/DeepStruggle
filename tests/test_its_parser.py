"""Unit tests for ITS / RTSL tournament match extractor and parser."""

import gzip
import json
import os
import tempfile
import pytest

from tools.extract_its_games import init_its_db, insert_its_match, show_its_database_stats
from tools.parse_its_to_warmup import parse_playdek_line, parse_its_database
from bindings.action_encoder import ActionEncoder


def test_parse_playdek_line():
    # Headline line
    res = parse_playdek_line("USSR Headlines: Socialist Governments")
    assert len(res) == 1
    assert res[0][0] == "USSR"
    assert res[0][1] == 6  # Card 7 (Socialist Governments) -> index 6

    # Coup line
    res = parse_playdek_line("US Coups in Italy with Duck and Cover")
    assert len(res) == 2
    assert res[0] == ("US", ActionEncoder.OP_MODE_OFFSET + 1)
    assert res[1] == ("US", ActionEncoder.NODE_OFFSET + 10)  # Italy = 10

    # Space race line
    res = parse_playdek_line("USSR Space Race attempt with Nazi Scientist")
    assert len(res) == 1
    assert res[0] == ("USSR", ActionEncoder.PLAY_MODE_OFFSET + 2)


def test_its_db_and_parsing_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_its.sqlite")
        out_gz = os.path.join(tmpdir, "test_its_warmup.jsonl.gz")

        conn = init_its_db(db_path)

        match1 = {
            "game_id": 47900,
            "tournament": "RTSL Season 18",
            "season": "Season 18",
            "us_player": "PlayerAlpha",
            "ussr_player": "PlayerBeta",
            "winner": "US",
            "victory_type": "VP Track",
            "end_turn": 4,
            "final_vp": 20,
            "handicap": 2,
            "optional_cards": 1,
            "log_text": "USSR Headlines: Socialist Governments\nUS Headlines: Duck and Cover\nUSSR Coups in Italy with Comecon",
        }

        assert insert_its_match(conn, match1) is True
        conn.commit()
        conn.close()

        stats = show_its_database_stats(db_path)
        assert stats["total_matches"] == 1
        assert "RTSL Season 18" in stats["tournaments"]

        parse_stats = parse_its_database(db_path=db_path, output_path=out_gz)
        assert parse_stats["total_games_found"] == 1
        assert parse_stats["valid_games_written"] == 1

        with gzip.open(out_gz, "rt", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            assert len(lines) == 1
            assert lines[0]["game_id"] == "its_47900"
            assert lines[0]["winner"] == "US"
            assert len(lines[0]["actions"]) > 0
