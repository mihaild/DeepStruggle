#!/usr/bin/env python3
"""ACTS Tournament Game Journal Crawler & Database Builder.

Crawls and archives real, authentic human tournament game journals from
the Automated Card Tracking System (ACTS - acts.warhorsesim.com) into
raw storage (data/raw_acts_journals/) and SQLite (data/acts_games.sqlite).
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

DEFAULT_ACTS_DIR = os.path.join(_root, "data", "raw_acts_journals")
DEFAULT_ACTS_DB = os.path.join(_root, "data", "acts_games.sqlite")

CREATE_ACTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS acts_journals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id INTEGER UNIQUE,
  title TEXT DEFAULT "",
  us_player TEXT DEFAULT "",
  ussr_player TEXT DEFAULT "",
  total_entries INTEGER DEFAULT 0,
  raw_html_path TEXT DEFAULT "",
  journal_text TEXT DEFAULT "",
  created_at INTEGER DEFAULT 0
);
"""


def init_acts_db(db_path: str = DEFAULT_ACTS_DB) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(CREATE_ACTS_TABLE_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_acts_game_id ON acts_journals(game_id);")
    conn.commit()
    return conn


def extract_acts_journal_text(html: str) -> Dict[str, Any]:
    """Parses raw HTML from ACTS journal page and extracts structured text log."""
    # Check if this is a Twilight Struggle game
    is_ts = "Twilight Struggle" in html or "twilight" in html.lower() or "USSR" in html or "US " in html

    # Extract title / players
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""

    # Strip HTML tags to get clean chronological text journal
    text_content = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text_content = re.sub(r"<style[^>]*>.*?</style>", "", text_content, flags=re.DOTALL | re.IGNORECASE)
    text_content = re.sub(r"<br\s*/?>", "\n", text_content, flags=re.IGNORECASE)
    text_content = re.sub(r"</div>", "\n", text_content, flags=re.IGNORECASE)
    text_content = re.sub(r"</tr>", "\n", text_content, flags=re.IGNORECASE)
    text_content = re.sub(r"</td>", "\t", text_content, flags=re.IGNORECASE)
    text_content = re.sub(r"<[^>]+>", "", text_content)

    clean_lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    journal_text = "\n".join(clean_lines)

    return {
        "is_ts": is_ts,
        "title": title,
        "total_lines": len(clean_lines),
        "journal_text": journal_text,
    }


def fetch_acts_journals(
    start_id: int = 1,
    end_id: int = 500,
    raw_dir: str = DEFAULT_ACTS_DIR,
    db_path: str = DEFAULT_ACTS_DB,
    delay_sec: float = 0.2,
) -> int:
    """Download authentic tournament game journals from ACTS."""
    os.makedirs(raw_dir, exist_ok=True)
    conn = init_acts_db(db_path)
    cursor = conn.cursor()
    downloaded_count = 0

    print(f"Connecting to ACTS (acts.warhorsesim.com) for game IDs {start_id}..{end_id}...")

    for gid in range(start_id, end_id + 1):
        if gid % 25 == 0:
            print(f"  ...{gid}: {downloaded_count} saved", flush=True)
        url = f"https://acts.warhorsesim.com/dynamic/journal.asp?id={gid}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    html = resp.read().decode("utf-8", errors="ignore")
                    if len(html) > 500 and ("journal" in html.lower() or "turn" in html.lower()):
                        parsed = extract_acts_journal_text(html)
                        if parsed["is_ts"]:
                            # Save raw unedited HTML file
                            raw_path = os.path.join(raw_dir, f"acts_game_{gid}.html")
                            with open(raw_path, "w", encoding="utf-8") as f:
                                f.write(html)

                            # Save to SQLite
                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO acts_journals
                                (game_id, title, total_entries, raw_html_path, journal_text, created_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    gid,
                                    parsed["title"],
                                    parsed["total_lines"],
                                    raw_path,
                                    parsed["journal_text"],
                                    int(time.time()),
                                ),
                            )
                            conn.commit()
                            downloaded_count += 1
                            print(f" [✓] Downloaded Game #{gid}: {parsed['title']} ({parsed['total_lines']} lines)")
        except urllib.error.HTTPError as he:
            if he.code != 404:
                print(f" [-] Game #{gid}: HTTP {he.code}")
        except Exception as e:
            pass

        time.sleep(delay_sec)

    conn.close()
    print(f"\nCompleted! Total real human Twilight Struggle games downloaded: {downloaded_count}")
    return downloaded_count


def show_acts_stats(db_path: str = DEFAULT_ACTS_DB):
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return

    conn = init_acts_db(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM acts_journals")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT game_id, title, total_entries FROM acts_journals LIMIT 10")
    samples = cursor.fetchall()
    conn.close()

    print("=" * 80)
    print(f"ACTS HUMAN GAME CORPUS: {db_path}")
    print("=" * 80)
    print(f" Total Real Tournament Matches Stored: {total}")
    for gid, title, lines in samples:
        print(f"   • Game #{gid:5d}: {title:40s} ({lines} lines)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ACTS Human Game Journal Downloader")
    parser.add_argument("--start-id", type=int, default=1, help="Starting ACTS game ID")
    parser.add_argument("--end-id", type=int, default=200, help="Ending ACTS game ID")
    parser.add_argument("--raw-dir", type=str, default=DEFAULT_ACTS_DIR, help="Directory to save raw HTML files")
    parser.add_argument("--db-path", type=str, default=DEFAULT_ACTS_DB, help="SQLite database path")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    if args.stats:
        show_acts_stats(args.db_path)
    else:
        fetch_acts_journals(
            start_id=args.start_id,
            end_id=args.end_id,
            raw_dir=args.raw_dir,
            db_path=args.db_path,
        )


if __name__ == "__main__":
    main()
