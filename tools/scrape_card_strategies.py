#!/usr/bin/env python3
"""
Scrapes and formats Twilight Struggle card strategy guides from Twilight Strategy (twilightstrategy.com),
combining them into structured, comprehensive Markdown documents grouped by card and war period,
with complete metadata, official card text, historical context, strategic analysis, and references.
"""

import os
import re
import json
import bs4  # type: ignore
from typing import Dict, Any, List, Tuple

CACHE_DIR = "scratch/twilightstrategy_cache"
CARDS_JSON = "rules/cards.json"
DOCS_DIR = "docs"
STRATEGY_DIR = "docs/strategy"

# Load official card definitions
with open(CARDS_JSON, "r", encoding="utf-8") as f:
    cards_list = json.load(f)
    cards_dict: Dict[int, Dict[str, Any]] = {c["id"]: c for c in cards_list}

# Load list page to extract canonical URLs if present
list_page_path = os.environ.get("STRATEGY_LIST_PAGE", os.path.join(CACHE_DIR, "card_list.html"))
card_urls: Dict[int, str] = {}
if os.path.exists(list_page_path):
    with open(list_page_path, "r", encoding="utf-8") as f:
        soup = bs4.BeautifulSoup(f.read(), "html.parser")

    entry_content = soup.find("div", class_="entry-content")
    if entry_content:
        tables = entry_content.find_all("table")
        for table in tables:
            for tr in table.find_all("tr"):
                tds = tr.find_all(["td", "th"])
                if not tds:
                    continue
                row_text = [td.get_text(strip=True) for td in tds]
                a = tr.find("a")
                if a and a.get("href"):
                    match = re.match(r"^#?(\d+)", row_text[0])
                    if match:
                        card_urls[int(match.group(1))] = a.get("href")


def clean_inner_html_to_markdown(elem: bs4.PageElement) -> str:
    """Converts a BeautifulSoup HTML element into clean Markdown text."""
    if elem is None:
        return ""
    if isinstance(elem, str):
        return str(elem)
    
    tag_name = elem.name.lower()
    
    # Ignore widgets / scripts / sharing
    if tag_name in ["script", "style", "noscript"]:
        return ""
    if elem.get("class") and any(c in ["sharedaddy", "sd-sharing-enabled", "wpcnt", "jp-relatedposts"] for c in elem.get("class")):
        return ""
    if elem.get("id") in ["jp-post-flair"]:
        return ""

    if tag_name in ["strong", "b"]:
        inner = "".join(clean_inner_html_to_markdown(c) for c in elem.contents).strip()
        return f"**{inner}**" if inner else ""
    elif tag_name in ["em", "i"]:
        inner = "".join(clean_inner_html_to_markdown(c) for c in elem.contents).strip()
        return f"*{inner}*" if inner else ""
    elif tag_name == "a":
        href = elem.get("href", "").strip()
        inner = "".join(clean_inner_html_to_markdown(c) for c in elem.contents).strip()
        if href and inner:
            return f"[{inner}]({href})"
        return inner
    elif tag_name == "br":
        return "\n"
    elif tag_name == "code":
        inner = "".join(clean_inner_html_to_markdown(c) for c in elem.contents).strip()
        return f"`{inner}`" if inner else ""
    else:
        return "".join(clean_inner_html_to_markdown(c) for c in elem.contents)


def parse_card_article(cid: int) -> Dict[str, Any]:
    """Parses an individual card's cached HTML article from Twilight Strategy."""
    fpath = os.path.join(CACHE_DIR, f"card_{cid:03d}.html")
    with open(fpath, "rb") as f:
        soup = bs4.BeautifulSoup(f.read(), "html.parser", from_encoding="utf-8")
    
    entry = soup.find("div", class_="entry-content")
    if not entry:
        raise ValueError(f"Could not find entry-content for card {cid}")
    
    # Remove widgets
    for tag in entry.find_all(["div", "span", "script", "style"], class_=["sharedaddy", "sd-sharing-enabled", "wpcnt", "jp-relatedposts"]):
        tag.decompose()
    for tag in entry.find_all(id=["jp-post-flair"]):
        tag.decompose()

    meta = cards_dict[cid]
    name = meta["name"]
    clean_name = name.rstrip("*")
    ops = meta["ops"]
    side = meta["side"].upper()
    age = meta["age"].title()
    one_time = "Yes (Removed on Event)" if meta["one_time"] else "No (Recurring)"
    desc = meta["description"]
    url = card_urls.get(cid, "https://twilightstrategy.com/card-list/")

    # 1. Historical quote / blockquote
    bq = entry.find("blockquote")
    flavor_text = ""
    if bq:
        flavor_text = clean_inner_html_to_markdown(bq).strip()
        flavor_text = re.sub(r'^[“"«]\s*', '', flavor_text)
        flavor_text = re.sub(r'\s*[”"»]$', '', flavor_text)
        flavor_text = re.sub(r'\s+', ' ', flavor_text)
        bq.decompose()

    # 2. Extract Year / Period
    year = ""
    for p in list(entry.find_all("p"))[:4]:
        p_text = p.get_text(strip=True)
        if re.match(r"^\d{4}(\s*[-–—]\s*\d{4})?$", p_text):
            year = p_text
            p.decompose()
            break
        m = re.match(r"^[\(\[]?(\d{4}(?:\s*[-–—]\s*\d{4})?)[\)\]]?$", p_text)
        if m:
            year = m.group(1)
            p.decompose()
            break

    # 3. Check for redundant metadata paragraph (Time: ... Side: ...) or card title paragraph
    for p in list(entry.find_all("p"))[:4]:
        p_text = p.get_text(strip=True)
        if "Time:" in p_text and ("Side:" in p_text or "Battlegrounds" in p_text or "Ops:" in p_text or "Countries" in p_text):
            p.decompose()
            break
        if p_text.lower() in [name.lower(), clean_name.lower(), f"{cid}. {name.lower()}", f"{cid}. {clean_name.lower()}"]:
            p.decompose()
            break

    # 4. If flavor_text is still empty, check if first paragraph looks like flavor text
    if not flavor_text:
        first_p = entry.find("p")
        if first_p:
            p_text = first_p.get_text(strip=True)
            if ("While " in p_text or "In " in p_text or "After " in p_text or "During " in p_text or "The " in p_text) and len(p_text) > 80 and not any(kw in p_text.lower() for kw in ["as ussr", "as us", "you should", "i usually", "play this", "headline"]):
                flavor_text = clean_inner_html_to_markdown(first_p).strip()
                flavor_text = re.sub(r'\s+', ' ', flavor_text)
                first_p.decompose()

    # 5. Parse remaining body elements into structured sections
    sections: List[Tuple[str, str]] = []
    current_heading = "Overview"
    current_content: List[str] = []

    def flush_section():
        nonlocal current_heading, current_content
        text = "\n\n".join(c.strip() for c in current_content if c.strip())
        if text:
            sections.append((current_heading, text))
        current_content = []

    for child in entry.children:
        if not isinstance(child, bs4.Tag):
            continue
        
        tag_name = child.name.lower()

        # Check for headings or pseudo-heading paragraphs
        is_heading = False
        heading_text = ""

        if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            is_heading = True
            heading_text = child.get_text(strip=True)
        elif tag_name == "p":
            p_raw = child.get_text(strip=True)
            strong_tag = child.find(["strong", "b"])
            # If paragraph contains only strong tag and is short
            if strong_tag and strong_tag.get_text(strip=True) == p_raw and len(p_raw) < 70:
                is_heading = True
                heading_text = p_raw
            # Known standard section titles
            elif p_raw in ["As USSR", "As US", "General Considerations", "Early War", "Mid War", "Late War", "Headline", "Space Race", "Turn 1 considerations", "Where should I Decolonize into?"]:
                is_heading = True
                heading_text = p_raw
            # Numbered subsection header e.g. "1. 5 Ops in Asia"
            elif re.match(r"^\d+\.\s+[A-Z0-9]", p_raw) and len(p_raw) < 70:
                is_heading = True
                heading_text = p_raw
            elif p_raw.endswith("?") and len(p_raw) < 70:
                is_heading = True
                heading_text = p_raw

        if is_heading and heading_text:
            flush_section()
            current_heading = heading_text
            continue

        # Otherwise parse paragraph / list / table / blockquote
        if tag_name == "p":
            p_md = clean_inner_html_to_markdown(child).strip()
            # Clean up redundant metadata if it slipped through
            if "Time:" in p_md and ("Side:" in p_md or "Ops:" in p_md or "Battlegrounds" in p_md):
                continue
            if p_md:
                current_content.append(p_md)
        elif tag_name == "ul":
            items = []
            for li in child.find_all("li", recursive=False):
                li_md = clean_inner_html_to_markdown(li).strip()
                if li_md:
                    items.append(f"- {li_md}")
            if items:
                current_content.append("\n".join(items))
        elif tag_name == "ol":
            items = []
            for idx, li in enumerate(child.find_all("li", recursive=False), 1):
                li_md = clean_inner_html_to_markdown(li).strip()
                if li_md:
                    items.append(f"{idx}. {li_md}")
            if items:
                current_content.append("\n".join(items))
        elif tag_name == "blockquote":
            bq_md = clean_inner_html_to_markdown(child).strip()
            if bq_md:
                lines = bq_md.split("\n")
                quoted = "\n".join(f"> {l}" if l.strip() else ">" for l in lines)
                current_content.append(quoted)
        elif tag_name == "table":
            rows = []
            for tr in child.find_all("tr"):
                cells = [clean_inner_html_to_markdown(c).strip() for c in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                max_cols = max(len(r) for r in rows)
                rows = [r + [""] * (max_cols - len(r)) for r in rows]
                header = "| " + " | ".join(rows[0]) + " |"
                sep = "| " + " | ".join(["---"] * max_cols) + " |"
                body_rows = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
                current_content.append(f"{header}\n{sep}\n{body_rows}")

    flush_section()

    return {
        "id": cid,
        "name": name,
        "clean_name": clean_name,
        "ops": ops,
        "side": side,
        "age": age,
        "one_time": one_time,
        "desc": desc,
        "url": url,
        "year": year,
        "flavor_text": flavor_text,
        "sections": sections
    }


def format_card_markdown(card_data: Dict[str, Any], heading_level: int = 2) -> str:
    """Formats a single card's structured data into clean, comprehensive Markdown."""
    cid = card_data["id"]
    name = card_data["name"]
    ops = card_data["ops"]
    side = card_data["side"]
    age = card_data["age"]
    one_time = card_data["one_time"]
    desc = card_data["desc"]
    url = card_data["url"]
    year = card_data["year"]
    flavor = card_data["flavor_text"]
    sections = card_data["sections"]

    h2 = "#" * heading_level
    h3 = "#" * (heading_level + 1)
    h4 = "#" * (heading_level + 2)

    lines = []
    anchor = f"card-{cid}-{re.sub(r'[^a-zA-Z0-9]+', '-', name.lower()).strip('-')}"
    lines.append(f'<a id="{anchor}"></a>')
    lines.append(f'{h2} #{cid}: {name}\n')

    # Card Properties Table
    lines.append(
        f"| Property | Value |\n"
        f"|---|---|\n"
        f"| **War Period** | {age} |\n"
        f"| **Side** | {side} |\n"
        f"| **Operations (Ops)** | {ops} |\n"
        f"| **Removed After Event** | {one_time} |\n"
        f"| **Official Card Text** | *{desc}* |\n"
        f"| **Reference Source** | [Twilight Strategy: {name}]({url}) |\n"
    )

    # Historical Context
    if year or flavor:
        ctx_title = f"{h3} Historical Context" + (f" ({year})" if year else "")
        lines.append(ctx_title + "\n")
        if flavor:
            lines.append(f'> *"{flavor}"*\n')

    # Strategic Analysis Sections
    lines.append(f"{h3} Strategic Analysis\n")
    if not sections:
        lines.append("*No extended strategic commentary available for this card.*\n")
    else:
        for sec_title, sec_content in sections:
            if sec_title.lower() == "overview":
                lines.append(f"{sec_content}\n")
            else:
                lines.append(f"{h4} {sec_title}\n\n{sec_content}\n")

    lines.append("\n---\n")
    return "\n".join(lines)


def generate_period_summary_table(cards_in_period: List[Dict[str, Any]]) -> str:
    """Generates a summary markdown table for a collection of cards."""
    rows = [
        "| # | Card Name | Ops | Side | Event Type | Strategy Anchor |",
        "|---|---|---|---|---|---|"
    ]
    for c in cards_in_period:
        cid = c["id"]
        name = c["name"]
        ops = str(c["ops"])
        side = c["side"]
        ev = "Removed (*)" if "Yes" in c["one_time"] else "Recurring"
        anchor = f"#card-{cid}-{re.sub(r'[^a-zA-Z0-9]+', '-', name.lower()).strip('-')}"
        rows.append(f"| {cid} | {name} | {ops} | {side} | {ev} | [View Strategy]({anchor}) |")
    return "\n".join(rows) + "\n\n"


def main():
    print("Parsing all 110 Twilight Strategy card articles...")
    all_parsed_cards = []
    for cid in range(1, 111):
        c_data = parse_card_article(cid)
        all_parsed_cards.append(c_data)

    early_war_cards = [c for c in all_parsed_cards if (1 <= c["id"] <= 35) or (103 <= c["id"] <= 106)]
    mid_war_cards = [c for c in all_parsed_cards if (36 <= c["id"] <= 81) or (107 <= c["id"] <= 108)]
    late_war_cards = [c for c in all_parsed_cards if (82 <= c["id"] <= 102) or (109 <= c["id"] <= 110)]

    print(f"Total Early War Cards: {len(early_war_cards)}")
    print(f"Total Mid War Cards:   {len(mid_war_cards)}")
    print(f"Total Late War Cards:  {len(late_war_cards)}")

    # 1. Generate master unified file: docs/card_strategies.md
    print("Writing master file: docs/card_strategies.md...")
    master_lines = [
        "# Twilight Struggle: Complete Card Strategy Guide & Encyclopedia\n",
        "Comprehensive strategic analysis and operational guidelines for all 110 cards in the Deluxe Edition of **Twilight Struggle**.",
        "Scraped, synthesized, and categorized from the authoritative [Twilight Strategy](https://twilightstrategy.com/) guide by Chirag Asnani (Theory), combined with official rulebook card primitives and regional scoring mechanics.\n",
        "## Table of Contents\n",
        "- [Introduction & Core Strategic Concepts](#introduction--core-strategic-concepts)",
        "- [Early War Cards (Turns 1–3)](#early-war-cards-turns-13)",
        "- [Mid War Cards (Turns 4–7)](#mid-war-cards-turns-47)",
        "- [Late War Cards (Turns 8–10)](#late-war-cards-turns-810)",
        "- [Optional & Promo Cards (#103–#110)](#optional--promo-cards-103110)\n",
        "## Introduction & Core Strategic Concepts\n",
        "Twilight Struggle cards serve dual purposes: **Operations (Ops)** and **Events**. Mastering card play requires understanding:\n",
        "1. **DEFCON Management & Suicide Avoidance**: Opponent events triggered by your Ops play execute under your action round. If an event reduces DEFCON to 1 (e.g., *Duck and Cover*, *CIA Created*, *Lone Gunman*, *Olympic Games*), you lose immediately.",
        "2. **Event Timing & Deck Flow**: Asterisked events are removed from the game when triggered; unstarred cards return upon reshuffle (Turns 3 and 7). Filtering or triggering unstarred opponent events at minimal detriment is a core skill.",
        "3. **Headline Priority**: Higher Ops cards take priority during Headline Phase simultaneous reveals. In case of ties, US resolves first.",
        "4. **Space Race Disposal**: Space Race offers safe dumping of hazardous opponent events without triggering their text, but is rate-limited per turn.\n",
        "---\n",
        "## Early War Cards (Turns 1–3)\n",
        "The Early War comprises cards #1 to #35, plus optional cards #103 to #106.\n",
        generate_period_summary_table([c for c in all_parsed_cards if 1 <= c["id"] <= 35])
    ]

    for c in all_parsed_cards:
        if c["id"] == 36:
            master_lines.append("## Mid War Cards (Turns 4–7)\n")
            master_lines.append("The Mid War comprises cards #36 to #81, plus optional cards #107 to #108.\n")
            master_lines.append(generate_period_summary_table([card for card in all_parsed_cards if 36 <= card["id"] <= 81]))
        elif c["id"] == 82:
            master_lines.append("## Late War Cards (Turns 8–10)\n")
            master_lines.append("The Late War comprises cards #82 to #102, plus optional cards #109 to #110.\n")
            master_lines.append(generate_period_summary_table([card for card in all_parsed_cards if 82 <= card["id"] <= 102]))
        elif c["id"] == 103:
            master_lines.append("## Optional & Promo Cards (#103–#110)\n")
            master_lines.append("Deluxe Edition optional / promo cards spanning Early, Mid, and Late War.\n")
            master_lines.append(generate_period_summary_table([card for card in all_parsed_cards if 103 <= card["id"] <= 110]))

        master_lines.append(format_card_markdown(c, heading_level=3))

    with open(os.path.join(DOCS_DIR, "card_strategies.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(master_lines))

    # 2. Generate modular files in docs/strategy/
    print("Writing modular strategy files in docs/strategy/...")
    
    # README.md
    readme_content = [
        "# Twilight Struggle Card Strategy Encyclopedia\n",
        "A structured, searchable strategic compendium covering all 110 cards in **Twilight Struggle Deluxe Edition**.",
        "Based on analysis from [Twilight Strategy](https://twilightstrategy.com/) by Theory with official card specifications.\n",
        "## Guides by War Period\n",
        "- [Early War Strategy Guide (Cards #1–35, #103–106)](early_war.md)",
        "- [Mid War Strategy Guide (Cards #36–81, #107–108)](mid_war.md)",
        "- [Late War Strategy Guide (Cards #82–102, #109–110)](late_war.md)",
        "- [Complete Consolidated Encyclopedia (All 110 Cards)](../card_strategies.md)\n",
        "## Quick Reference Card Breakdown\n",
        "### Early War Cards Summary",
        generate_period_summary_table(early_war_cards),
        "### Mid War Cards Summary",
        generate_period_summary_table(mid_war_cards),
        "### Late War Cards Summary",
        generate_period_summary_table(late_war_cards)
    ]
    with open(os.path.join(STRATEGY_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme_content))

    # early_war.md
    early_content = [
        "# Early War Card Strategy Guide (Turns 1–3)\n",
        "Contains detailed strategic commentary, US/USSR tactical play, headline ratings, and historical background for all Early War cards (#1–#35, plus Optional Early War #103–#106).\n",
        "## Summary Table",
        generate_period_summary_table(early_war_cards),
        "---\n"
    ]
    for c in early_war_cards:
        early_content.append(format_card_markdown(c, heading_level=2))
    with open(os.path.join(STRATEGY_DIR, "early_war.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(early_content))

    # mid_war.md
    mid_content = [
        "# Mid War Card Strategy Guide (Turns 4–7)\n",
        "Contains detailed strategic commentary, US/USSR tactical play, headline ratings, and historical background for all Mid War cards (#36–#81, plus Optional Mid War #107–#108).\n",
        "## Summary Table",
        generate_period_summary_table(mid_war_cards),
        "---\n"
    ]
    for c in mid_war_cards:
        mid_content.append(format_card_markdown(c, heading_level=2))
    with open(os.path.join(STRATEGY_DIR, "mid_war.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(mid_content))

    # late_war.md
    late_content = [
        "# Late War Card Strategy Guide (Turns 8–10)\n",
        "Contains detailed strategic commentary, US/USSR tactical play, headline ratings, and historical background for all Late War cards (#82–#102, plus Optional Late War #109–#110).\n",
        "## Summary Table",
        generate_period_summary_table(late_war_cards),
        "---\n"
    ]
    for c in late_war_cards:
        late_content.append(format_card_markdown(c, heading_level=2))
    with open(os.path.join(STRATEGY_DIR, "late_war.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(late_content))

    print("All strategy documents generated successfully!")


if __name__ == "__main__":
    main()
