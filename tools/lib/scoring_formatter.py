import ts_engine

def format_regional_scoring_breakdown(state_dict: dict, card_id: int) -> str:
    """
    Produces an exhaustive, formatted breakdown of country control and scoring calculus
    for any scoring card (Europe, Asia, Middle East, Southeast Asia, Africa, Central America, South America).
    """
    countries = state_dict.get("countries", {})

    # Southeast Asia Scoring (Card #38)
    if card_id == 38:
        se_countries = ["Burma", "Laos/Cambodia", "Vietnam", "Malaysia", "Indonesia", "Philippines"]
        thai = "Thailand"

        us_pts = 0
        ussr_pts = 0
        lines = []
        lines.append("┌─── 📊 REGIONAL SCORING AUDIT: SOUTHEAST ASIA (Card #38) ───")
        lines.append("│ • Southeast Asian Countries Breakdown (1 VP each, Thailand 2 VP):")

        for c_name in se_countries:
            c_data = countries.get(c_name, {})
            c_info = ts_engine.MapData.get_country_info(c_data["id"]) if "id" in c_data else {}
            stab = c_info.get("stability", 1)
            us_inf = c_data.get("us_influence", 0)
            ussr_inf = c_data.get("ussr_influence", 0)
            ctrl = c_data.get("controlled_by", "NONE")
            if ctrl == "US": us_pts += 1
            elif ctrl == "USSR": ussr_pts += 1
            lines.append(f"│     - {c_name:<16} [US:{us_inf} USSR:{ussr_inf} Stab:{stab}] -> Control: {ctrl} (1 VP)")

        t_data = countries.get(thai, {})
        t_info = ts_engine.MapData.get_country_info(t_data["id"]) if "id" in t_data else {}
        t_stab = t_info.get("stability", 2)
        t_us = t_data.get("us_influence", 0)
        t_ussr = t_data.get("ussr_influence", 0)
        t_ctrl = t_data.get("controlled_by", "NONE")
        if t_ctrl == "US": us_pts += 2
        elif t_ctrl == "USSR": ussr_pts += 2
        lines.append(f"│     - ★ Thailand [BG] [US:{t_us} USSR:{t_ussr} Stab:{t_stab}] -> Control: {t_ctrl} (2 VP)")

        net = us_pts - ussr_pts
        lines.append(f"│ • US Total SE Asia Score:   {us_pts} VP")
        lines.append(f"│ • USSR Total SE Asia Score: {ussr_pts} VP")
        lines.append(f"│ • Net VP Outcome: {net:+d} VP ({'US gains ' + str(net) + ' VP' if net > 0 else ('USSR gains ' + str(-net) + ' VP' if net < 0 else 'No VP swing')})")
        lines.append("└─────────────────────────────────────────────────────────────")
        return "\n".join(lines)

    scoring_cards_map = {
        1: (1, "ASIA"),
        2: (0, "EUROPE"),
        3: (2, "MIDDLE EAST"),
        79: (3, "AFRICA"),
        81: (5, "SOUTH AMERICA"),
        88: (4, "CENTRAL AMERICA"),
    }
    if card_id not in scoring_cards_map:
        return ""

    region_id, r_name = scoring_cards_map[card_id]

    us_controlled = []
    ussr_controlled = []
    us_bg_count = 0
    ussr_bg_count = 0
    total_bg_count = 0

    for c_name, c_data in countries.items():
        nid = c_data["id"]
        c_info = ts_engine.MapData.get_country_info(nid)
        if c_info["region"] != region_id:
            continue

        is_bg = c_info["battleground"]
        if is_bg:
            total_bg_count += 1

        ctrl = c_data["controlled_by"]
        us_inf = c_data["us_influence"]
        ussr_inf = c_data["ussr_influence"]
        stab = c_info["stability"]

        entry = f"{c_name} (US:{us_inf}/USSR:{ussr_inf}, Stab:{stab})"
        if is_bg:
            entry = f"★ [BG] {entry}"

        if ctrl == "US":
            us_controlled.append(entry)
            if is_bg:
                us_bg_count += 1
        elif ctrl == "USSR":
            ussr_controlled.append(entry)
            if is_bg:
                ussr_bg_count += 1

    us_total = len(us_controlled)
    ussr_total = len(ussr_controlled)
    us_non_bg = us_total - us_bg_count
    ussr_non_bg = ussr_total - ussr_bg_count

    # Evaluate Status
    def eval_status(my_tot, opp_tot, my_bg, opp_bg, my_non_bg):
        if my_tot > opp_tot and my_bg == total_bg_count:
            return "CONTROL"
        elif my_tot > opp_tot and my_bg > opp_bg and my_bg >= 1 and my_non_bg >= 1:
            return "DOMINATION"
        elif my_tot >= 1:
            return "PRESENCE"
        else:
            return "NONE"

    us_status = eval_status(us_total, ussr_total, us_bg_count, ussr_bg_count, us_non_bg)
    ussr_status = eval_status(ussr_total, us_total, ussr_bg_count, us_bg_count, ussr_non_bg)

    base_vps = {
        0: {"PRESENCE": 3, "DOMINATION": 7, "CONTROL": 0}, # Europe (Control = Instant Win)
        1: {"PRESENCE": 3, "DOMINATION": 7, "CONTROL": 9}, # Asia
        2: {"PRESENCE": 3, "DOMINATION": 5, "CONTROL": 7}, # Middle East
        3: {"PRESENCE": 1, "DOMINATION": 4, "CONTROL": 6}, # Africa
        4: {"PRESENCE": 1, "DOMINATION": 3, "CONTROL": 5}, # Central America
        5: {"PRESENCE": 2, "DOMINATION": 5, "CONTROL": 6}, # South America
    }

    reg_vps = base_vps.get(region_id, {"PRESENCE": 0, "DOMINATION": 0, "CONTROL": 0})
    us_base = reg_vps.get(us_status, 0)
    ussr_base = reg_vps.get(ussr_status, 0)

    us_pts = us_base + us_bg_count
    ussr_pts = ussr_base + ussr_bg_count
    net = us_pts - ussr_pts

    lines = []
    lines.append(f"┌─── 📊 REGIONAL SCORING AUDIT: {r_name} ───")
    lines.append(f"│ • Total Regional Battlegrounds: {total_bg_count}")
    lines.append(f"│ • US Controlled ({us_total} total, {us_bg_count} BG, {us_non_bg} Non-BG):")
    if us_controlled:
        for c in us_controlled:
            lines.append(f"│     - {c}")
    else:
        lines.append("│     - (None)")
    lines.append(f"│ • USSR Controlled ({ussr_total} total, {ussr_bg_count} BG, {ussr_non_bg} Non-BG):")
    if ussr_controlled:
        for c in ussr_controlled:
            lines.append(f"│     - {c}")
    else:
        lines.append("│     - (None)")
    lines.append(f"│ • US Standing:   {us_status:<10} -> Base: {us_base} VP + {us_bg_count} BGs = {us_pts} VP")
    lines.append(f"│ • USSR Standing: {ussr_status:<10} -> Base: {ussr_base} VP + {ussr_bg_count} BGs = {ussr_pts} VP")
    lines.append(f"│ • Net VP Outcome: {net:+d} VP ({'US gains ' + str(net) + ' VP' if net > 0 else ('USSR gains ' + str(-net) + ' VP' if net < 0 else 'No VP swing')})")
    lines.append("└──────────────────────────────────────────────")
    return "\n".join(lines)
