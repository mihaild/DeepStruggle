/**
 * A region's scoring, side by side: the US's summary on the left, the USSR's on the right, the
 * rows between them -- standing, battlegrounds, the other countries, those next to the enemy
 * superpower, the total -- as the Playdek app's scoring screen lays it out. Shown on hovering a
 * region's badge and, pinned in the modal, on clicking (or tapping) it.
 *
 * Every number is the engine's (WasmEngine.regionScores): what the scoring counted and what it
 * awards. Nothing here adds up a score.
 */
import type { RegionScoreAudit } from "./map_view";
import type { RegionalStatus, RegionSideScore } from "./types";
import { escapeHtml } from "./html";
import { regionName, sideName, t } from "./i18n";

function standingName(status: RegionalStatus): string {
  switch (status) {
    case "PRESENCE": return t("region:standing.presence");
    case "DOMINATION": return t("region:standing.domination");
    case "CONTROL": return t("region:standing.control");
    default: return t("region:standing.none");
  }
}

/** The same, as a map badge has room for. */
export function standingShort(status: RegionalStatus): string {
  switch (status) {
    case "PRESENCE": return t("region:standingShort.presence");
    case "DOMINATION": return t("region:standingShort.domination");
    case "CONTROL": return t("region:standingShort.control");
    default: return t("region:standingShort.none");
  }
}

const title = (s: string): string => s.toLowerCase().replace(/\b\w/g, c => c.toUpperCase());

export type NetSide = "us" | "ussr" | "even";

/** "+3 USSR", "+2 US", "even", or who wins outright -- in full, and as short as a map badge needs. */
export function regionNet(a: RegionScoreAudit): { text: string; short: string; side: NetSide } {
  if (a.wins) {
    const side = sideName(a.wins);
    return { text: t("region:net.winsGame", { side }), short: t("region:net.winsShort", { side }), side: a.wins === "US" ? "us" : "ussr" };
  }
  const vp = (text: string, side: NetSide) => ({ text, short: text, side });
  if (a.netVpUs > 0) return vp(t("region:net.vp", { vp: a.netVpUs, side: sideName("US") }), "us");
  if (a.netVpUs < 0) return vp(t("region:net.vp", { vp: -a.netVpUs, side: sideName("USSR") }), "ussr");
  return vp(t("region:net.even"), "even");
}

function row(label: string, us: string, ussr: string, cls = ""): string {
  return `<div class="rs-row ${cls}"><span class="rs-us">${us}</span><span class="rs-label">${label}</span><span class="rs-ussr">${ussr}</span></div>`;
}

/** A count and the VP it brings: "2 · +2". */
const counted = (n: number, vp: number): string => `${n}<small>${vp ? ` · +${vp}` : ""}</small>`;

/** A standing and what it alone is worth: "Domination · 7". */
function standing(s: RegionSideScore): string {
  const name = standingName(s.status ?? "NONE");
  return `${name}<small>${s.base_vp ? ` · ${s.base_vp}` : ""}</small>`;
}

export function regionSummaryHtml(a: RegionScoreAudit): string {
  const net = regionNet(a);
  let rows: string;
  if (a.southeastAsia) {
    // Southeast Asia Scoring has no standings: each country is worth its VP (Thailand 2).
    const worth = (s: RegionSideScore, vp: number) => s.controlled.filter(c => c.vp === vp).length;
    rows = row(t("region:row.countriesOneVp"), String(worth(a.us, 1)), String(worth(a.ussr, 1)))
      + row(t("region:row.thailandTwoVp"), worth(a.us, 2) ? "✓" : "—", worth(a.ussr, 2) ? "✓" : "—");
  } else {
    const other = (s: RegionSideScore) => String(s.countries - s.battlegrounds);
    rows = row(t("region:row.standing"), standing(a.us), standing(a.ussr), "rs-standing")
      + row(t("region:row.battlegrounds", { total: a.totalBattlegrounds }),
            counted(a.us.battlegrounds, a.us.battlegrounds), counted(a.ussr.battlegrounds, a.ussr.battlegrounds))
      + row(t("region:row.otherCountries"), other(a.us), other(a.ussr))
      + row(t("region:row.nextToEnemy"), counted(a.us.adjacent, a.us.adjacent), counted(a.ussr.adjacent, a.ussr.adjacent));
  }
  rows += row(t("region:row.total"), String(a.us.vp), String(a.ussr.vp), "rs-total");
  const list = (names: string[]): string => escapeHtml(names.length ? names.join(", ") : t("region:none"));
  const uncounted = a.shuttleDiplomacy
    ? `<div class="rs-foot">${t("region:shuttleDiplomacy")}</div>`
    : "";
  return `
    <div class="rs" style="--rs-accent: ${escapeHtml(a.color)}" data-region="${escapeHtml(a.regionName)}">
      <div class="rs-head"><span class="rs-title">${t("region:title", { region: escapeHtml(regionName(title(a.regionName))) })}</span><span class="rs-net rs-${net.side}">${net.text}</span></div>
      <div class="rs-sides"><span class="rs-us">${sideName("US")}</span><span></span><span class="rs-ussr">${sideName("USSR")}</span></div>
      ${rows}
      <div class="rs-list"><span class="rs-us">${t("region:controls", { side: sideName("US") })}</span> ${list(a.usControlled)}</div>
      <div class="rs-list"><span class="rs-ussr">${t("region:controls", { side: sideName("USSR") })}</span> ${list(a.ussrControlled)}</div>
      ${uncounted}
      <div class="rs-foot">${escapeHtml(a.scoringCardText)}</div>
      <div class="rs-foot">${t("region:cardWhere", { card: escapeHtml(a.scoringCardName), where: escapeHtml(a.scoringCard.where) })}</div>
    </div>`;
}
