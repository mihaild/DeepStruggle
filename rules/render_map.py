#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def load_map_data(filepath="rules/map.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def render_map(data, output_path="rules/map_rendered.png"):
    countries = data["countries"]
    regions_info = data["regions"]
    name_to_country = {c["name"]: c for c in countries}

    # Superpower locations
    superpowers = {
        "USA": {"pos": [100, 230], "color": "#1E3A8A", "label": "U.S.A."},
        "USSR": {"pos": [650, 140], "color": "#991B1B", "label": "U.S.S.R."}
    }

    # Region colors
    region_colors = {
        "Europe": "#4A6FA5",
        "Middle East": "#0284C7",
        "Asia": "#EA580C",
        "Africa": "#D97706",
        "Central America": "#16A34A",
        "South America": "#059669"
    }

    box_w = 34.0
    box_h = 19.0

    fig, ax = plt.subplots(figsize=(25, 16), dpi=200, facecolor="#F4EADB")
    ax.set_facecolor("#EDE2CD")

    # Title & Subtitle
    ax.text(500, 20, "TWILIGHT STRUGGLE — DELUXE EDITION", 
            fontsize=24, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(500, 36, "Global Board Topology Graph (84 Countries, Authentic TS Board Layout)", 
            fontsize=14, fontstyle="italic", ha="center", va="center", color="#334155")

    # 1. Draw Connection Edges
    drawn_edges = set()
    for c in countries:
        u_name = c["name"]
        u_pos = c["pos"]
        u_reg = c["region"]

        # Inter-country edges
        for v_name in c["neighbours"]:
            edge_key = tuple(sorted([u_name, v_name]))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)

            v_c = name_to_country.get(v_name)
            if not v_c:
                continue
            v_pos = v_c["pos"]
            v_reg = v_c["region"]

            # Line styling: brown solid for same region, red dashed for inter-region
            if u_reg == v_reg:
                ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]],
                        color="#785028", linewidth=1.8, zorder=1, alpha=0.75)
            else:
                ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]],
                        color="#DC2626", linewidth=2.0, linestyle="--", zorder=1, alpha=0.9)

        # Superpower edges
        if c["superpower_adjacent"]:
            sp_name = c["superpower_adjacent"]
            if sp_name in superpowers:
                sp_pos = superpowers[sp_name]["pos"]
                ax.plot([u_pos[0], sp_pos[0]], [u_pos[1], sp_pos[1]],
                        color="#09090B", linewidth=2.4, linestyle="-", zorder=2, alpha=0.95)

    # 2. Draw Superpower Boxes
    for sp_key, sp in superpowers.items():
        x, y = sp["pos"]
        sp_box_w = 48.0
        sp_box_h = 28.0
        box = mpatches.FancyBboxPatch((x - sp_box_w/2, y - sp_box_h/2), sp_box_w, sp_box_h,
                                      boxstyle="round,pad=0.5,rounding_size=2.0",
                                      ec="#09090B", fc=sp["color"], lw=3.0, zorder=4)
        ax.add_patch(box)
        ax.text(x, y, sp["label"], color="white", fontsize=15, fontweight="bold",
                ha="center", va="center", zorder=5)

    # 3. Draw Country Rectangles
    for c in countries:
        c_name = c["name"]
        x, y = c["pos"]
        is_bg = c["battleground"]
        reg = c["region"]

        border_color = "#6D28D9" if is_bg else "#475569"
        border_width = 2.4 if is_bg else 1.2
        bg_fill = "#FEF3C7" if is_bg else "#FFFFFF"

        # Main node rectangle
        node_box = mpatches.FancyBboxPatch((x - box_w/2, y - box_h/2),
                                           box_w, box_h,
                                           boxstyle="round,pad=0.2,rounding_size=1.5",
                                           ec=border_color, fc=bg_fill, lw=border_width, zorder=3)
        ax.add_patch(node_box)

        # Top banner for Battleground countries
        if is_bg:
            banner_box = mpatches.FancyBboxPatch((x - box_w/2 + 0.6, y - box_h/2 + 0.5),
                                                 box_w - 1.2, 3.8,
                                                 boxstyle="round,pad=0.1,rounding_size=0.8",
                                                 ec="none", fc="#7C3AED", lw=0, zorder=4)
            ax.add_patch(banner_box)
            ax.text(x - 3.5, y - box_h/2 + 2.4, "BATTLEGROUND",
                    color="white", fontsize=6.0, fontweight="black", ha="center", va="center", zorder=5)

        # Stability badge (Upper-right corner)
        badge_color = "#5B21B6" if is_bg else "#334155"
        badge_w = 6.8
        badge_h = 5.6
        badge_x = x + box_w/2 - badge_w/2 - 0.8
        badge_y = y - box_h/2 + badge_h/2 + 0.8
        
        badge_box = mpatches.FancyBboxPatch((badge_x - badge_w/2, badge_y - badge_h/2),
                                            badge_w, badge_h,
                                            boxstyle="round,pad=0.1,rounding_size=0.8",
                                            ec=border_color, fc=badge_color, lw=1.0, zorder=6)
        ax.add_patch(badge_box)
        ax.text(badge_x, badge_y, str(c["stability"]),
                color="white", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=7)

        # Country label
        text_color = "#4C1D95" if is_bg else "#0F172A"
        font_weight = "bold" if is_bg else "semibold"
        font_size = 7.5 if len(c_name) <= 13 else 6.5

        label_y = y + 1.2 if is_bg else y
        label_text = c_name
        if len(label_text) > 13 and " " in label_text:
            parts = label_text.split(" ", 1)
            label_text = f"{parts[0]}\n{parts[1]}"
            font_size = 6.5

        ax.text(x - 2.5, label_y, label_text,
                color=text_color, fontsize=font_size, fontweight=font_weight,
                ha="center", va="center", zorder=6)

    # Set canvas boundaries (Invert Y so top is 0)
    ax.set_xlim(0, 1000)
    ax.set_ylim(640, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ Map successfully rendered and saved to: {output_path}")

if __name__ == "__main__":
    map_file = sys.argv[1] if len(sys.argv) > 1 else "rules/map.json"
    out_img = sys.argv[2] if len(sys.argv) > 2 else "rules/map_rendered.png"

    print(f"Loading map dataset from {map_file}...")
    map_data = load_map_data(map_file)
    render_map(map_data, out_img)
