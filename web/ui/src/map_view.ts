import { GameState, LastTurnCountry, MapMetadata, RecentRoll, RegionSideScore } from "./types";
import { renderRollTray } from "./roll_tray";
import { NetSide, regionNet, regionSummaryHtml, standingShort } from "./region_summary";
import { CARDS_METADATA, MAP_METADATA } from "./metadata";
import { engineLayout, namesByValue, titleCase } from "./engine/layout";
import { escapeHtml } from "./html";
import { cardDescription, cardName, countryName, onLanguageChange, regionName, sideName, t } from "./i18n";

/** A scoring card's badge and summary: the engine's scoring (RegionScore) with names and colours. */
export interface RegionScoreAudit {
  /** The Region (Europe 0 .. South America 5), or 6 for Southeast Asia. */
  regionId: number;
  regionName: string;
  badgePos: [number, number]; // [x, y]
  color: string;
  scoringCardName: string;
  /** The card's terms, as rules/cards.json prints them. */
  scoringCardText: string;
  /** Where the region's scoring card is, which says whether it can still score this cycle. */
  scoringCard: ScoringCardWhereabouts;
  southeastAsia: boolean;
  totalBattlegrounds: number;
  us: RegionSideScore;
  ussr: RegionSideScore;
  netVpUs: number;
  /** The side this scoring would hand the game (Control of Europe). */
  wins: "US" | "USSR" | null;
  /** The countries each side controls there, a battleground (in Southeast Asia, Thailand) starred. */
  usControlled: string[];
  ussrControlled: string[];
  /** Whether Shuttle Diplomacy took one USSR battleground out of this scoring's count. */
  shuttleDiplomacy: boolean;
}

/** The engine's slot for Southeast Asia Scoring, after the six regions (ScoringTap::kSoutheastAsia). */
const SOUTHEAST_ASIA_SLOT = 6;

/** Where each scoring badge sits, by the region's engine name ("SE_ASIA": Southeast Asia's card). */
const REGION_LOOK: Record<string, { pos: [number, number] }> = {
  EUROPE: { pos: [288, 29] },
  ASIA: { pos: [830, 179] },
  MIDDLE_EAST: { pos: [640, 221] },
  AFRICA: { pos: [424, 535] },
  CENTRAL_AMERICA: { pos: [130, 263] },
  SOUTH_AMERICA: { pos: [270, 441] },
  SE_ASIA: { pos: [894, 433] },
};

const CARD_META = new Map(CARDS_METADATA.map(c => [c.id, c]));

/**
 * A scoring card is `live` while it can still be played before the next reshuffle -- in the draw
 * deck or a hand (or face down in a headline, or being looked at by an event) -- and `out` once
 * discarded or removed. `later` is a Mid War scoring card before the Mid War deck comes in.
 */
export interface ScoringCardWhereabouts {
  status: "live" | "out" | "later";
  /** The engine's location, in words. */
  where: string;
}

/** Where the engine's card location leaves a scoring card, in words; null for a location not listed. */
function scoringCardWhere(loc: string): ScoringCardWhereabouts | null {
  switch (loc) {
    case "DRAW_DECK": return { status: "live", where: t("map:where.drawDeck") };
    case "HAND_US_UNKNOWN": return { status: "live", where: t("map:where.usHand") };
    case "HAND_US_KNOWN": return { status: "live", where: t("map:where.usHandKnown") };
    case "HAND_USSR_UNKNOWN": return { status: "live", where: t("map:where.ussrHand") };
    case "HAND_USSR_KNOWN": return { status: "live", where: t("map:where.ussrHandKnown") };
    case "HEADLINE_COMMITTED": return { status: "live", where: t("map:where.headline") };
    case "PEEKED_TEMP": return { status: "live", where: t("map:where.peeked") };
    case "DISCARD_PILE": return { status: "out", where: t("map:where.discarded") };
    case "REMOVED_FROM_GAME": return { status: "out", where: t("map:where.removed") };
    case "ONGOING_EVENT": return { status: "out", where: t("map:where.ongoing") };
    case "UNAVAILABLE": return { status: "later", where: t("map:where.unavailable") };
    // A player's view (game/view.ts): the deck and the other side's unseen cards are one pile.
    case "UNSEEN": return { status: "live", where: t("map:where.unseen") };
    default: return null;
  }
}

export function scoringCardWhereabouts(state: GameState, cardId: number): ScoringCardWhereabouts {
  const loc = state.card_locations?.[String(cardId)] ?? "";
  return scoringCardWhere(loc) ?? { status: "later", where: loc ? loc.toLowerCase() : t("map:where.unknown") };
}

const SCORING_CARD_TAG: Record<ScoringCardWhereabouts["status"], { fill: string; stroke: string; ink: string }> = {
  live: { fill: "rgba(22, 101, 52, 0.95)", stroke: "#4ADE80", ink: "#DCFCE7" },
  out: { fill: "rgba(51, 65, 85, 0.95)", stroke: "#64748B", ink: "#94A3B8" },
  later: { fill: "rgba(15, 23, 42, 0.90)", stroke: "#475569", ink: "#64748B" },
};

function scoringCardTagText(status: ScoringCardWhereabouts["status"]): string {
  return status === "live" ? t("map:tag.live") : status === "out" ? t("map:tag.out") : t("map:tag.later");
}

/**
 * Roughly how wide `text` is set, in map units: `narrow` a Latin glyph, `wide` a CJK one -- so a
 * translated name is squeezed into its box as an English one is.
 */
function textWidth(text: string, narrow: number, wide: number): number {
  let n = 0, w = 0;
  for (const ch of text) {
    if (ch.codePointAt(0)! >= 0x2e80) w++; else n++;
  }
  return n * narrow + w * wide;
}

const NS = "http://www.w3.org/2000/svg";

/** A country's box on the map, in map units, and its parts. */
const BOX_W = 38;
const BOX_H = 22;
const BAND_H = 7.4;
const STAB_W = 6.4;
const NAME_FONT = 4.4;
const MARKER_W = 12.2;
const MARKER_H = 12.2;
/** More countries than this changing at once is a jump (a new game, a link), not a move. */
const MAX_FLASHED = 12;
/** The view's width bounds in map units: 8x in, half out. */
const MIN_VIEW_W = 125;
const MAX_VIEW_W = 2000;
/** CSS pixels a press may move and still be a click. */
const DRAG_SLOP = 6;

/** Names too long for the band even squeezed; the hover card gives the full one. */
const SHORT_NAMES: Record<string, string> = {
  "United Kingdom": "UK",
  "West Germany": "W. Germany",
  "East Germany": "E. Germany",
  "Czechoslovakia": "Czechoslov.",
  "West African States": "W. African St.",
  "SE African States": "SE African St.",
  "Dominican Rep": "Dominican R.",
};

/**
 * Each region's colour: a dark `band` behind a country's name, and the brighter `accent` its
 * scoring badge is outlined in. None of them is the sides' blue and red or the battlegrounds'
 * purple, so a colour on the map always means one thing.
 */
export const REGION_STYLE: Record<string, { band: string; accent: string }> = {
  "Europe": { band: "#475569", accent: "#94A3B8" },
  "Middle East": { band: "#0E7490", accent: "#22D3EE" },
  "Asia": { band: "#C2410C", accent: "#FB923C" },
  "Southeast Asia": { band: "#7C2D12", accent: "#F59E0B" },
  "Africa": { band: "#A16207", accent: "#FACC15" },
  "Central America": { band: "#15803D", accent: "#4ADE80" },
  "South America": { band: "#065F46", accent: "#34D399" },
};

function regionStyle(region: string, subregions?: string[]): { band: string; accent: string } {
  if (subregions?.includes("Southeast Asia")) return REGION_STYLE["Southeast Asia"];
  return REGION_STYLE[region] ?? REGION_STYLE["Europe"];
}

/** A region's scoring badge, in map units; its position is chosen clear of every country box. */
/** The rolls made in a country, which mark it; a Space Race roll's country_id is the box it went for. */
const ROLLED_IN_A_COUNTRY = new Set(["COUP", "REALIGNMENT", "WAR_EVENT"]);

/** A badge's net VP, in the colour of the side it favours. */
const NET_COLOR: Record<NetSide, string> = { us: "#60A5FA", ussr: "#F87171", even: "#CBD5E1" };

const REGION_BADGE_W = 104;
const REGION_BADGE_H = 34;

/**
 * What it takes `side` to control a country (or break the other side's control), in influence and
 * in Ops: a point placed in a country the other side controls costs 2 Ops until control breaks.
 */
export function controlLine(side: "US" | "USSR", own: number, opp: number, stability: number,
                            controlledBy: string): string {
  const tag = `<span class="ct-${side.toLowerCase()}">${sideName(side)}</span>`;
  if (controlledBy === side) {
    const spare = own - opp - stability;
    return spare > 0 ? t("map:control.controlsSpare", { side: tag, spare }) : t("map:control.controlsNoSpare", { side: tag });
  }
  const control = Math.max(0, opp + stability - own);
  if (controlledBy === "NONE") return t("map:control.controlsWith", { side: tag, count: control });
  const brk = opp - stability - own + 1;
  const breakOps = 2 * brk;
  const controlOps = breakOps + (control - brk);
  return t("map:control.breaksWith", { side: tag, brk, breakOps, control, controlOps });
}

type SvgAttrs = Record<string, string | number | null | undefined>;

function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: SvgAttrs): SVGElementTagNameMap[K] {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  return el;
}

function svgText(text: string, attrs: SvgAttrs): SVGTextElement {
  const t = svgEl("text", attrs);
  t.textContent = text;
  return t;
}

/**
 * A country a recent roll was made in (a coup, a realignment, a war): its die just above the box,
 * in the roller's colour -- both dice for a realignment, US first. The roll tray says the rest.
 */
function markRoll(g: SVGGElement, r: RecentRoll, left: number, top: number) {
  g.classList.add("rolled");
  g.setAttribute("data-roll", r.type);
  const dice: Array<[number, string]> = r.type === "REALIGNMENT"
    ? [[r.roll1 ?? 0, "us"], [r.roll2 ?? 0, "ussr"]]
    : [[r.roll1 ?? 0, (r.roller ?? "").toLowerCase()]];
  dice.forEach(([v, side], i) => {
    const x = left + i * 8.4;
    const y = top - 8.6;
    const die = svgEl("g", { class: `roll-die ${side}` });
    die.appendChild(svgEl("rect", { x, y, width: 7.6, height: 7.6, rx: 1.6 }));
    die.appendChild(svgText(String(v), { x: x + 3.8, y: y + 5.8, "text-anchor": "middle" }));
    g.appendChild(die);
  });
}

/**
 * One side's influence space. With influence it is a marker: its number in the side's colour on
 * white, or -- flipped, when that influence controls the country -- white on the side's colour,
 * outlined in white. Empty, it is a faint dashed space with no number.
 */
function influenceMarker(side: "us" | "ussr", inf: number, x: number, y: number, controls: boolean): SVGGElement {
  const colour = side === "us" ? "#2563EB" : "#DC2626";
  const ink = side === "us" ? "#1D4ED8" : "#B91C1C";
  const g = svgEl("g", { class: `inf-marker inf-marker-${side}` });
  g.appendChild(svgEl("rect", {
    class: `inf-box inf-${side}${inf > 0 ? "" : " empty"}`, x, y, width: MARKER_W, height: MARKER_H, rx: 1.4,
    fill: inf === 0 ? "rgba(148, 163, 184, 0.06)" : controls ? colour : "#F8FAFC",
    stroke: controls ? "#FFFFFF" : inf > 0 ? colour : "rgba(148, 163, 184, 0.35)",
    "stroke-width": controls ? 1.1 : inf > 0 ? 0.9 : 0.5,
    "stroke-dasharray": inf === 0 ? "1.2,1" : null,
  }));
  if (inf > 0) {
    g.appendChild(svgText(inf.toString(), {
      class: `inf-num inf-num-${side}`, x: x + MARKER_W / 2, y: y + MARKER_H / 2 + 3.3, "text-anchor": "middle",
      fill: controls ? "#FFFFFF" : ink, "font-size": inf >= 10 ? 7.6 : 9.2, "font-weight": 900,
    }));
  }
  return g;
}

export class MapView {
  private svg: SVGSVGElement;
  private mapData: MapMetadata | null = null;
  private lastState: GameState | null = null;
  private onCountryClick: (countryId: number) => void;
  private tooltipEl: HTMLElement | null = null;
  /** Each country's "us/ussr" influence at the last render, to flash what changed. */
  private lastInfluence: Map<number, string> | null = null;

  // Pan and zoom state
  private viewBox = { x: 0, y: 0, w: 1000, h: 650 };
  /** The pointers pressed on the map, by id, at their latest position. */
  private pointers = new Map<number, { x: number; y: number }>();
  private pressedAt = { x: 0, y: 0 };
  /** The current press has moved: it is a pan or a pinch, and its click is not a click. */
  private dragged = false;
  /** The latest press's pointer type: "mouse" hovers, "touch" and "pen" tap to see a card. */
  private lastPointer = "mouse";
  /** The region whose scoring summary was last pinned in the modal, to redo in a new language. */
  private summaryRegion: number | null = null;

  constructor(svgElement: SVGSVGElement, onCountryClick: (countryId: number) => void) {
    this.svg = svgElement;
    this.onCountryClick = onCountryClick;
    this.tooltipEl = document.getElementById("card-tooltip");
    this.setupPanAndZoom();
    this.updateViewBox();
    this.fetchMetadata();
    onLanguageChange(() => this.refreshRegionSummary());
  }

  public fetchMetadata() {
    if (this.mapData) return;
    // Bundled from rules/map.json (metadata.ts): the page needs no server to draw the board.
    this.mapData = MAP_METADATA;
    if (this.lastState) {
      this.render(this.lastState);
    }
  }

  public setMapData(data: MapMetadata) {
    this.mapData = data;
    if (this.lastState) {
      this.render(this.lastState);
    }
  }

  private updateViewBox() {
    this.svg.setAttribute(
      "viewBox",
      `${this.viewBox.x} ${this.viewBox.y} ${this.viewBox.w} ${this.viewBox.h}`
    );
  }

  /** A client point (CSS pixels) in map units, through the SVG's own transform, letterboxing included. */
  private toMap(clientX: number, clientY: number): { x: number; y: number } {
    const m = this.svg.getScreenCTM();
    if (!m) return { x: this.viewBox.x + this.viewBox.w / 2, y: this.viewBox.y + this.viewBox.h / 2 };
    const p = new DOMPoint(clientX, clientY).matrixTransform(m.inverse());
    return { x: p.x, y: p.y };
  }

  /** Zoom by `factor` (below 1 is in) about the map point under a client point, which stays put. */
  private zoomAt(factor: number, clientX: number, clientY: number) {
    const p = this.toMap(clientX, clientY);
    const w = Math.min(MAX_VIEW_W, Math.max(MIN_VIEW_W, this.viewBox.w * factor));
    const f = w / this.viewBox.w;
    this.viewBox = {
      x: p.x - (p.x - this.viewBox.x) * f, y: p.y - (p.y - this.viewBox.y) * f,
      w, h: this.viewBox.h * f,
    };
    this.updateViewBox();
  }

  public zoom(factor: number) {
    const r = this.svg.getBoundingClientRect();
    this.zoomAt(factor, r.left + r.width / 2, r.top + r.height / 2);
  }

  public resetView() {
    this.viewBox = { x: 0, y: 0, w: 1000, h: 650 };
    this.updateViewBox();
  }

  /**
   * Frame a region: "Europe", "Asia", ..., or "Southeast Asia" (a subregion). The box is its
   * countries' and its scoring badge's, padded; the SVG letterboxes it to the map's shape.
   */
  public zoomToRegion(region: string) {
    const cs = (this.mapData?.countries ?? []).filter(c => c.region === region || (c.subregions ?? []).includes(region));
    if (!cs.length) return;
    const xs = cs.map(c => c.pos[0]), ys = cs.map(c => c.pos[1]);
    const pad = 34;
    let x0 = Math.min(...xs) - BOX_W / 2 - pad, x1 = Math.max(...xs) + BOX_W / 2 + pad;
    let y0 = Math.min(...ys) - BOX_H / 2 - pad, y1 = Math.max(...ys) + BOX_H / 2 + pad;
    // The view keeps the map's aspect, so zooming in and out from here does not stretch.
    const w = Math.max(x1 - x0, (y1 - y0) * (1000 / 650)), h = w * (650 / 1000);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.viewBox = { x: cx - w / 2, y: cy - h / 2, w, h };
    this.updateViewBox();
  }

  /** The two fingers' distance and midpoint, while pinching. */
  private pinch(): { span: number; x: number; y: number } {
    const [a, b] = [...this.pointers.values()];
    return { span: Math.hypot(a.x - b.x, a.y - b.y) || 1, x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  /**
   * Pan by dragging, with a mouse or a finger and from anywhere, a country included: zoomed in on
   * a phone the map is mostly countries. A press that moves less than DRAG_SLOP is a click (a
   * tap); one that moved is a pan, and its click is swallowed before it reaches a country. Two
   * fingers pinch-zoom about their midpoint; the wheel zooms about the cursor.
   */
  private setupPanAndZoom() {
    this.svg.addEventListener("pointerdown", (e) => {
      this.lastPointer = e.pointerType;
      if (e.pointerType === "mouse" && e.button !== 0) return;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pointers.size === 1) {
        this.dragged = false;
        this.pressedAt = { x: e.clientX, y: e.clientY };
      }
    });

    const release = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      // After the click this release may cause (same task), a drag is over: the next click counts.
      if (!this.pointers.size && this.dragged) setTimeout(() => { if (!this.pointers.size) this.dragged = false; }, 0);
    };

    // A real mouse moving brings hover back on a device with both (emulated hover makes no pointer event).
    this.svg.addEventListener("pointermove", (e) => { if (e.pointerType === "mouse") this.lastPointer = "mouse"; });
    window.addEventListener("pointermove", (e) => {
      const prev = this.pointers.get(e.pointerId);
      if (!prev) return;
      // A mouse released outside the window sent no pointerup here: it comes back with no button down.
      if (e.pointerType === "mouse" && e.buttons === 0) {
        release(e);
        return;
      }
      if (!this.dragged) {
        if (Math.hypot(e.clientX - this.pressedAt.x, e.clientY - this.pressedAt.y) < DRAG_SLOP) return;
        this.dragged = true;
        this.hideRegionTooltip();
      }
      if (this.pointers.size === 2) {
        const before = this.pinch();
        this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        const after = this.pinch();
        const a = this.toMap(before.x, before.y), b = this.toMap(after.x, after.y);
        this.viewBox.x -= b.x - a.x;
        this.viewBox.y -= b.y - a.y;
        this.updateViewBox();
        this.zoomAt(before.span / after.span, after.x, after.y);
        return;
      }
      const a = this.toMap(prev.x, prev.y), b = this.toMap(e.clientX, e.clientY);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.viewBox.x -= b.x - a.x;
      this.viewBox.y -= b.y - a.y;
      this.updateViewBox();
    });

    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);

    // Capture: runs before the country's own click handler, so a pan never selects a country.
    this.svg.addEventListener("click", (e) => {
      if (this.dragged) {
        e.stopPropagation();
        e.preventDefault();
      }
    }, true);
    // A tap on the open map puts a tapped country's card away.
    this.svg.addEventListener("click", (e) => {
      if (this.lastPointer === "mouse" || (e.target as Element).closest(".svg-country-node, .svg-region-badge")) return;
      this.focusCountry(null, []);
      this.hideRegionTooltip();
    });

    this.svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoomAt(e.deltaY > 0 ? 1.1 : 0.9, e.clientX, e.clientY);
    }, { passive: false });
  }

  /**
   * Each scoring card's badge and summary, from what the engine says it would score
   * (`state.region_scores`, WasmEngine.regionScores): the page adds names, colours and where the
   * card is, and no arithmetic. Empty until the engine has scored the state.
   */
  public calculateRegionalScoring(state: GameState): RegionScoreAudit[] {
    const scores = state.region_scores;
    const layout = engineLayout();
    if (!scores || !layout || !this.mapData) return [];
    const names = new Map((this.mapData.countries || []).map(c => [c.id, c.name]));
    const regionNames = namesByValue(layout.regions);
    return scores.regions.map(r => {
      const se = r.slot === SOUTHEAST_ASIA_SLOT;
      const key = se ? "SE_ASIA" : regionNames[r.slot];
      const look = REGION_LOOK[key];
      const cardId = layout.card_ids[`${key}_SCORING`] ?? 0;
      const card = CARD_META.get(cardId);
      const listed = (s: RegionSideScore) => s.controlled.map(c => {
        const star = se ? (c.vp ?? 0) > 1 : !!c.battleground;
        const name = names.get(c.id);
        return `${name ? countryName(c.id, name) : `#${c.id}`}${star ? " ★" : ""}`;
      });
      return {
        regionId: r.slot,
        regionName: se ? "SOUTHEAST ASIA" : titleCase(key).toUpperCase(),
        badgePos: look?.pos ?? [0, 0],
        color: (se ? REGION_STYLE["Southeast Asia"] : REGION_STYLE[titleCase(key)] ?? REGION_STYLE["Europe"]).accent,
        scoringCardName: card ? `${cardName(cardId, card.name)} (#${cardId})` : t("map:cardNumber", { id: cardId }),
        // A translation keeps the card's whole text: only the English is cut, by its own words.
        scoringCardText: cardDescription(cardId, (card?.description ?? "").replace(/;?\s*MAY NOT BE HELD!?\s*$/i, "")),
        scoringCard: scoringCardWhereabouts(state, cardId),
        southeastAsia: se,
        totalBattlegrounds: r.battlegrounds,
        us: r.us,
        ussr: r.ussr,
        netVpUs: r.net,
        wins: r.wins,
        usControlled: listed(r.us),
        ussrControlled: listed(r.ussr),
        shuttleDiplomacy: r.ussr.shuttle_diplomacy === true,
      };
    });
  }

  public render(state: GameState) {
    this.lastState = state;
    if (!this.mapData) {
      this.fetchMetadata();
      return;
    }

    const countries = this.mapData.countries || [];
    const countryMap = new Map<string, any>();
    countries.forEach(c => countryMap.set(c.name, c));

    // The nodes a hover lit are about to go: the hover goes with them (on a phone, until the next tap).
    this.svg.classList.remove("country-hovering");
    this.hideRegionTooltip();
    this.svg.innerHTML = "";

    // 1. Connection Adjacency Lines
    const linesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    linesGroup.setAttribute("id", "adjacency-lines");
    const drawnEdges = new Set<string>();

    const superpowers = [
      { id: "USA", label: t("map:superpower.usa"), pos: [85, 230], color: "#1E3A8A", stroke: "#3B82F6" },
      { id: "USSR", label: t("map:superpower.ussr"), pos: [635, 120], color: "#7F1D1D", stroke: "#EF4444" }
    ];

    // Country-to-Country Edges
    countries.forEach(c => {
      (c.neighbours || []).forEach(nName => {
        const n = countryMap.get(nName);
        if (!n) return;
        const edgeKey = [Math.min(c.id, n.id), Math.max(c.id, n.id)].join("-");
        if (drawnEdges.has(edgeKey)) return;
        drawnEdges.add(edgeKey);

        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", "adj-line");
        line.setAttribute("data-a", c.name);
        line.setAttribute("data-b", n.name);
        line.setAttribute("x1", c.pos[0].toString());
        line.setAttribute("y1", c.pos[1].toString());
        line.setAttribute("x2", n.pos[0].toString());
        line.setAttribute("y2", n.pos[1].toString());
        const isInterRegion = (c.region !== n.region);
        line.setAttribute("stroke", isInterRegion ? "#DC2626" : "#475569");
        line.setAttribute("stroke-width", isInterRegion ? "1.6" : "1.2");
        line.setAttribute("stroke-dasharray", isInterRegion ? "3,3" : "2,2");
        linesGroup.appendChild(line);
      });

      // Superpower Adjacency Edges
      if (c.superpower_adjacent) {
        const sp = superpowers.find(s => s.id === c.superpower_adjacent);
        if (sp) {
          if (c.name === "Japan") {
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", `M ${c.pos[0]} ${c.pos[1]} C 960 210, 960 500, 480 600 S 70 320, ${sp.pos[0]} ${sp.pos[1]}`);
            path.setAttribute("stroke", "#3B82F6");
            path.setAttribute("stroke-width", "2.0");
            path.setAttribute("stroke-dasharray", "4,4");
            path.setAttribute("fill", "none");
            path.setAttribute("opacity", "0.8");
            linesGroup.appendChild(path);
          } else {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", c.pos[0].toString());
            line.setAttribute("y1", c.pos[1].toString());
            line.setAttribute("x2", sp.pos[0].toString());
            line.setAttribute("y2", sp.pos[1].toString());
            line.setAttribute("stroke", sp.stroke);
            line.setAttribute("stroke-width", "2.0");
            line.setAttribute("stroke-dasharray", "4,4");
            line.setAttribute("opacity", "0.8");
            linesGroup.appendChild(line);
          }
        }
      }
    });
    this.svg.appendChild(linesGroup);

    // 2. Superpower Nodes
    const spGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    spGroup.setAttribute("id", "superpower-nodes");

    superpowers.forEach(sp => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const w = 90, h = 24;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", (sp.pos[0] - w/2).toString());
      rect.setAttribute("y", (sp.pos[1] - h/2).toString());
      rect.setAttribute("width", w.toString());
      rect.setAttribute("height", h.toString());
      rect.setAttribute("rx", "4");
      rect.setAttribute("fill", sp.color);
      rect.setAttribute("stroke", sp.stroke);
      rect.setAttribute("stroke-width", "2.0");
      g.appendChild(rect);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", sp.pos[0].toString());
      text.setAttribute("y", (sp.pos[1] + 3.5).toString());
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill", "#FFFFFF");
      text.setAttribute("font-size", "10");
      text.setAttribute("font-weight", "900");
      text.setAttribute("letter-spacing", "1");
      text.textContent = sp.label;
      g.appendChild(text);
      spGroup.appendChild(g);
    });
    this.svg.appendChild(spGroup);

    // 3. Regional Scoring Overlay Badges (Showing current net points for US)
    this.renderRegionalScoringBadges(state);

    // 4. Country Nodes
    const legalNodes = new Set(
      state.legal_actions && state.legal_actions.decision_type === engineLayout()?.decision_types.POINT_NODE
        ? (state.legal_actions.valid_ids || []).filter((id: number) => id < 84)
        : []
    );

    const nodesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    nodesGroup.setAttribute("id", "country-nodes");
    // A player's view: the countries the other side's latest turn changed or targeted.
    const lastTurn = new Map((state.last_turn?.countries || []).map(lc => [lc.id, lc]));
    this.renderLastTurnSummary(state);
    const rolls = state.recent_rolls ?? [];
    const tray = document.getElementById("map-rolls");
    if (tray) renderRollTray(tray, rolls);
    const rolledIn = new Map<number, RecentRoll>();
    for (const r of rolls) {
      if (ROLLED_IN_A_COUNTRY.has(r.type) && (r.country_id ?? 255) < 84) rolledIn.set(r.country_id!, r);
    }

    const changed = this.influenceChanges(state);
    countries.forEach(c => {
      const stateC = state.countries ? state.countries[c.name] : null;
      const usInf = stateC ? stateC.us_influence : 0;
      const ussrInf = stateC ? stateC.ussr_influence : 0;
      const controlledBy = stateC ? stateC.controlled_by : "NONE";
      const isLegal = legalNodes.has(c.id);
      const x = c.pos[0];
      const y = c.pos[1];
      const left = x - BOX_W / 2;
      const top = y - BOX_H / 2;

      const g = svgEl("g", {
        class: `svg-country-node${isLegal ? " legal-target" : ""}${c.battleground ? " battleground" : ""}`
          + (changed.has(c.id) ? " inf-changed" : ""),
        "data-id": c.id,
        "data-name": c.name,
        "data-control": controlledBy,
      });

      // Control is carried by the fill, not only the border: a country that is a legal target has
      // its border overridden by the pulsing highlight (style.css), which is exactly when it
      // matters most who holds it.
      const fill = controlledBy === "US" ? "#1E40AF"
        : controlledBy === "USSR" ? "#991B1B"
        : c.battleground ? "#1E1B4B" : "#0F172A";
      const border = controlledBy === "US" ? "#3B82F6"
        : controlledBy === "USSR" ? "#EF4444"
        : c.battleground ? "#A855F7" : "#475569";
      g.appendChild(svgEl("rect", {
        class: "country-card-bg", x: left, y: top, width: BOX_W, height: BOX_H, rx: 2.5,
        fill, stroke: border, "stroke-width": controlledBy !== "NONE" ? 2.2 : c.battleground ? 1.6 : 1.0,
      }));

      // The name band, in the region's colour, as the board colours its regions.
      const r = 2.2, bl = left + 0.6, br = left + BOX_W - 0.6, bt = top + 0.6, bb = top + BAND_H;
      g.appendChild(svgEl("path", {
        class: "country-band",
        d: `M ${bl} ${bb} V ${bt + r} Q ${bl} ${bt} ${bl + r} ${bt} H ${br - r} Q ${br} ${bt} ${br} ${bt + r} V ${bb} Z`,
        fill: regionStyle(c.region, c.subregions).band,
      }));

      // Stability, at the band's right end: solid purple on a battleground.
      const stabX = left + BOX_W - STAB_W - 1.2;
      const stabY = top + 1.2;
      g.appendChild(svgEl("rect", {
        class: "country-stability", x: stabX, y: stabY, width: STAB_W, height: BAND_H - 1.8, rx: 1.0,
        fill: c.battleground ? "#7C3AED" : "#E2E8F0",
        stroke: c.battleground ? "#DDD6FE" : "none", "stroke-width": 0.5,
      }));
      g.appendChild(svgText(c.stability.toString(), {
        x: stabX + STAB_W / 2, y: stabY + (BAND_H - 1.8) / 2 + 1.75, "text-anchor": "middle",
        fill: c.battleground ? "#FFFFFF" : "#0F172A", "font-size": 5.0, "font-weight": 900,
      }));

      // The name, squeezed rather than cut when it is long.
      const label = countryName(c.id, SHORT_NAMES[c.name] ?? c.name);
      const room = stabX - (left + 1.6) - 0.8;
      const name = svgText(label, {
        class: "country-name", x: left + 1.6, y: top + BAND_H / 2 + 1.75,
        fill: "#FFFFFF", "font-size": NAME_FONT, "font-weight": 800,
      });
      if (textWidth(label, NAME_FONT * 0.56, NAME_FONT) > room) {
        name.setAttribute("textLength", room.toString());
        name.setAttribute("lengthAdjust", "spacingAndGlyphs");
      }
      g.appendChild(name);

      // Influence: US left, USSR right, as on the board. A marker that controls the country is
      // flipped -- white on its colour -- and an empty space shows no number at all.
      const markerY = top + BAND_H + (BOX_H - BAND_H - MARKER_H) / 2;
      const usX = left + 1.4;
      const ussrX = usX + MARKER_W + 0.9;
      g.appendChild(influenceMarker("us", usInf, usX, markerY, controlledBy === "US"));
      g.appendChild(influenceMarker("ussr", ussrInf, ussrX, markerY, controlledBy === "USSR"));

      const lc = lastTurn.get(c.id);
      if (lc) this.markLastTurn(g, lc, x, y, ussrX, usX);
      const roll = rolledIn.get(c.id);
      if (roll) markRoll(g, roll, left, top);

      g.addEventListener("click", (e) => {
        // A finger has no hover: tapping a country that is not a choice shows its card instead.
        // (The browser's emulated hover around a tap is ignored: the taps alone open and close it.)
        if (this.lastPointer !== "mouse") {
          if (!legalNodes.has(c.id)) {
            this.focusCountry(c.name, c.neighbours || []);
            this.showCountryTooltip(e, c, stateC, state, lc);
            return;
          }
          this.focusCountry(null, []);
          this.hideRegionTooltip();
        }
        this.onCountryClick(c.id);
      });
      g.addEventListener("mouseenter", (e) => {
        if (this.lastPointer !== "mouse") return;
        this.focusCountry(c.name, c.neighbours || []);
        this.showCountryTooltip(e, c, stateC, state, lc);
      });
      g.addEventListener("mousemove", (e) => this.placeTooltip(e));
      g.addEventListener("mouseleave", () => {
        if (this.lastPointer !== "mouse") return;
        this.focusCountry(null, []);
        this.hideRegionTooltip();
      });

      nodesGroup.appendChild(g);
    });

    this.svg.appendChild(nodesGroup);
  }

  /**
   * Mark a country the other side's latest turn touched: a dashed ring around it and the signed
   * change under each influence marker (the hover card says it in words).
   */
  private markLastTurn(g: SVGGElement, lc: LastTurnCountry, x: number, y: number, ussrX: number, usX: number) {
    g.classList.add("last-turn");
    g.insertBefore(svgEl("rect", {
      class: "last-turn-ring", x: x - BOX_W / 2 - 1.6, y: y - BOX_H / 2 - 1.6,
      width: BOX_W + 3.2, height: BOX_H + 3.2, rx: 3.5,
    }), g.firstChild);
    const chip = (delta: number, boxX: number, side: "us" | "ussr") => {
      if (!delta) return;
      g.appendChild(svgText(delta > 0 ? `+${delta}` : `\u2212${-delta}`, {
        class: `last-turn-delta ${side}`, x: boxX + MARKER_W / 2, y: y + BOX_H / 2 + 5.4, "text-anchor": "middle",
      }));
    };
    chip(lc.ussr, ussrX, "ussr");
    chip(lc.us, usX, "us");
  }

  /** What changed since the last render: the countries to flash (none on a jump). */
  private influenceChanges(state: GameState): Set<number> {
    const now = new Map<number, string>();
    for (const c of this.mapData?.countries || []) {
      const sc = state.countries?.[c.name];
      now.set(c.id, `${sc?.us_influence ?? 0}/${sc?.ussr_influence ?? 0}`);
    }
    const before = this.lastInfluence;
    this.lastInfluence = now;
    const changed = new Set<number>();
    if (!before) return changed;
    now.forEach((v, id) => { if (before.get(id) !== v) changed.add(id); });
    // A new game, a loaded link or a long jump in a replay moves too much to point at.
    return changed.size <= MAX_FLASHED ? changed : new Set();
  }

  /** Light a country and its neighbours (and the lines between them); `null` clears it. */
  private focusCountry(name: string | null, neighbours: string[]) {
    this.svg.querySelectorAll(".hover-focus, .hover-neighbour, .hover-edge").forEach(el =>
      el.classList.remove("hover-focus", "hover-neighbour", "hover-edge"));
    this.svg.classList.toggle("country-hovering", name !== null);
    if (name === null) return;
    const node = (n: string) => this.svg.querySelector(`.svg-country-node[data-name="${CSS.escape(n)}"]`);
    node(name)?.classList.add("hover-focus");
    neighbours.forEach(n => node(n)?.classList.add("hover-neighbour"));
    this.svg.querySelectorAll(`line.adj-line[data-a="${CSS.escape(name)}"], line.adj-line[data-b="${CSS.escape(name)}"]`)
      .forEach(l => l.classList.add("hover-edge"));
  }

  /**
   * A country's card on hover: its standing, what it takes each side to control it or break the
   * other's control (in influence and in Ops, at 2 Ops a point while the other side controls it),
   * the coup number, its neighbours, and what the other side's latest turn did there.
   */
  private showCountryTooltip(e: MouseEvent, c: MapMetadata["countries"][number],
                             sc: { us_influence: number; ussr_influence: number; controlled_by: string } | null,
                             state: GameState, lc: LastTurnCountry | undefined) {
    if (!this.tooltipEl) return;
    // Numbers, as a replay's snapshot may say anything.
    const us = Number(sc?.us_influence ?? 0), ussr = Number(sc?.ussr_influence ?? 0);
    const ctrl = sc?.controlled_by ?? "NONE";
    const lines = [controlLine("US", us, ussr, c.stability, ctrl), controlLine("USSR", ussr, us, c.stability, ctrl)];
    const ids = new Map((this.mapData?.countries || []).map(x => [x.name, x.id]));
    const adjacent = (c.neighbours || []).map(n => { const id = ids.get(n); return id === undefined ? n : countryName(id, n); });
    if (c.superpower_adjacent) adjacent.push(c.superpower_adjacent === "USA" ? t("map:tooltip.theUs") : t("map:tooltip.theUssr"));
    const standing = ctrl === "US" ? `<b class="ct-us">${t("map:tooltip.usControls")}</b>`
      : ctrl === "USSR" ? `<b class="ct-ussr">${t("map:tooltip.ussrControls")}</b>` : t("map:tooltip.uncontrolled");
    // Whether a coup or realignment is allowed here is the engine's to say (the legal targets lit on
    // the map); the card gives only the board's numbers.
    this.tooltipEl.innerHTML = `
      <div class="ct-head"><span class="ct-name">${escapeHtml(countryName(c.id, c.name))}</span>${c.battleground ? `<span class="ct-bg">${t("map:tooltip.battleground")}</span>` : ""}</div>
      <div class="ct-sub">${t("map:tooltip.sub", { regions: escapeHtml([c.region, ...(c.subregions || [])].map(regionName).join(" \u00b7 ")), stability: c.stability })}</div>
      <div class="ct-inf"><span class="ct-us">${sideName("US")} ${us}</span><span class="ct-ussr">${sideName("USSR")} ${ussr}</span><span>${standing}</span></div>
      <div class="ct-rows">${lines.map(l => `<div>${l}</div>`).join("")}
        <div>${t("map:tooltip.coup", { n: 2 * c.stability })}</div>
        ${lc && state.last_turn ? `<div class="ct-last">${t("map:tooltip.lastTurn", { player: escapeHtml(sideName(state.last_turn.player)), what: escapeHtml(describeLastTurnCountry(lc)) })}</div>` : ""}
      </div>
      <div class="ct-adj">${t("map:tooltip.adjacent", { list: escapeHtml(adjacent.join(", ") || t("map:tooltip.none")) })}</div>`;
    this.tooltipEl.classList.remove("hidden");
    this.placeTooltip(e);
  }

  private placeTooltip(e: MouseEvent) {
    if (!this.tooltipEl || this.tooltipEl.classList.contains("hidden")) return;
    const box = this.tooltipEl.getBoundingClientRect();
    const x = e.clientX + 16 + box.width > window.innerWidth ? e.clientX - box.width - 12 : e.clientX + 16;
    const y = Math.min(window.innerHeight - box.height - 8, e.clientY + 14);
    this.tooltipEl.style.left = `${Math.max(4, x)}px`;
    this.tooltipEl.style.top = `${Math.max(4, y)}px`;
  }

  /** The line under the map naming what the other side's latest turn did. */
  private renderLastTurnSummary(state: GameState) {
    const el = document.getElementById("map-last-turn");
    if (!el) return;
    const lt = state.last_turn;
    if (!lt) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    el.classList.remove("hidden");
    const what = lt.countries.length
      ? lt.countries.map(lc => `${countryName(lc.id, lc.name)} ${describeLastTurnCountry(lc)}`).join(" · ")
      : t("map:noCountryChanged");
    el.textContent = t("map:lastTurn", { player: sideName(lt.player), when: lt.label, what });
  }

  /**
   * One badge per scoring region, in the region's colour: what the scoring card would give now
   * (net VP and each side's standing, as the engine scores it) and whether the card can still come
   * this cycle. Which countries can be couped or realigned is the engine's legal mask, shown as the
   * lit targets of an open decision -- never a rule of the page's.
   */
  private renderRegionalScoringBadges(state: GameState) {
    const audits = this.calculateRegionalScoring(state);
    if (!audits.length) return;

    const overlayGroup = svgEl("g", { id: "regional-scoring-overlay" });

    audits.forEach(audit => {
      const g = svgEl("g", { class: "svg-region-badge", "data-region": audit.regionName });
      g.style.cursor = "pointer";
      const [x, y] = audit.badgePos;
      const left = x - REGION_BADGE_W / 2;
      const top = y - REGION_BADGE_H / 2;
      const right = left + REGION_BADGE_W;
      const accent = audit.color;

      g.appendChild(svgEl("rect", {
        class: "region-badge-bg", x: left, y: top, width: REGION_BADGE_W, height: REGION_BADGE_H,
        rx: 3, fill: "rgba(15, 23, 42, 0.92)", stroke: accent, "stroke-width": 1,
        filter: "drop-shadow(0px 2px 4px rgba(0,0,0,0.5))",
      }));
      g.appendChild(svgEl("rect", { x: left + 1.2, y: top + 1.2, width: 2.6, height: REGION_BADGE_H - 2.4, rx: 1.2, fill: accent }));

      // Row 1: the region and what its scoring card is worth to whom, right now.
      const regionTitle = regionName(audit.regionName.toLowerCase().replace(/\b\w/g, ch => ch.toUpperCase())).toUpperCase();
      const title = svgText(regionTitle, {
        class: "region-badge-title", x: left + 6, y: top + 9.2, fill: accent,
        "font-size": 6, "font-weight": 800, "letter-spacing": 0.3,
      });
      if (textWidth(regionTitle, 6 * 0.68, 6) > 62) {
        title.setAttribute("textLength", "62");
        title.setAttribute("lengthAdjust", "spacingAndGlyphs");
      }
      g.appendChild(title);
      const net = regionNet(audit);
      g.appendChild(svgText(net.short, {
        class: "region-badge-vp", x: right - 3.5, y: top + 9.4, "text-anchor": "end", fill: NET_COLOR[net.side],
        "font-size": 6.6, "font-weight": 900,
      }));

      // Row 2: each side's standing and its total (Southeast Asia has no standings, only VP).
      const line = (side: string, s: RegionSideScore) =>
        s.status === null ? t("map:badgeVp", { side, vp: s.vp }) : `${side} ${standingShort(s.status)} ${s.vp}`;
      const usLine = line(sideName("US"), audit.us);
      const ussrLine = line(sideName("USSR"), audit.ussr);
      g.appendChild(svgText(usLine, {
        class: "region-badge-us", x: left + 6, y: top + 18.6, fill: "#93C5FD", "font-size": 4.9, "font-weight": 700,
      }));
      g.appendChild(svgText(ussrLine, {
        class: "region-badge-ussr", x: right - 3.5, y: top + 18.6, "text-anchor": "end", fill: "#FCA5A5",
        "font-size": 4.9, "font-weight": 700,
      }));

      // Row 3: can the region's scoring card still come this cycle? Inside the badge, because
      // country nodes are drawn over the badges and would hide anything sticking out.
      const tag = { ...SCORING_CARD_TAG[audit.scoringCard.status], text: scoringCardTagText(audit.scoringCard.status) };
      const tagH = 6;
      const tagY = top + REGION_BADGE_H - tagH - 2.6;
      const tagG = svgEl("g", { class: "svg-region-card-tag", "data-status": audit.scoringCard.status });
      tagG.appendChild(svgText(t("map:tag.scoring"), {
        x: left + 6, y: tagY + tagH / 2 + 1.4, fill: "#64748B", "font-size": 3.8, "font-weight": 800, "letter-spacing": 0.4,
      }));
      const tagX = left + 28;
      const tagW = textWidth(tag.text, 2.9, 4.2) + 4.4;
      const tagRect = svgEl("rect", {
        x: tagX, y: tagY, width: tagW, height: tagH, rx: 1.6,
        fill: tag.fill, stroke: tag.stroke, "stroke-width": 0.6,
      });
      if (audit.scoringCard.status === "later") tagRect.setAttribute("stroke-dasharray", "1,0.8");
      tagG.appendChild(tagRect);
      tagG.appendChild(svgText(tag.text, {
        x: tagX + tagW / 2, y: tagY + tagH / 2 + 1.5, "text-anchor": "middle", fill: tag.ink,
        "font-size": 4.2, "font-weight": 800, "letter-spacing": 0.3,
      }));
      g.appendChild(tagG);

      g.addEventListener("mouseenter", (e) => { if (this.lastPointer === "mouse") this.showRegionTooltip(e, audit); });
      g.addEventListener("mouseleave", () => { if (this.lastPointer === "mouse") this.hideRegionTooltip(); });
      g.addEventListener("click", () => this.openRegionSummary(audit));

      overlayGroup.appendChild(g);
    });

    this.svg.appendChild(overlayGroup);
  }

  private showRegionTooltip(e: MouseEvent, audit: RegionScoreAudit) {
    if (!this.tooltipEl) return;
    this.tooltipEl.innerHTML = regionSummaryHtml(audit);
    this.tooltipEl.classList.add("region-tip");
    this.tooltipEl.classList.remove("hidden");
    this.placeTooltip(e);
  }

  /** The region's scoring summary, pinned in the modal: what a click (or a tap) on a badge opens. */
  private openRegionSummary(audit: RegionScoreAudit) {
    this.hideRegionTooltip();
    const container = document.getElementById("modal-container");
    const title = document.getElementById("modal-title");
    const body = document.getElementById("modal-body");
    if (!container || !title || !body) return;
    container.querySelector(".modal-content")?.classList.remove("modal-lg");
    title.textContent = audit.scoringCardName;
    body.innerHTML = regionSummaryHtml(audit);
    container.classList.remove("hidden");
    this.summaryRegion = audit.regionId;
  }

  /** The pinned summary, redone in a new language while it is still the modal's. */
  private refreshRegionSummary() {
    const container = document.getElementById("modal-container");
    const body = document.getElementById("modal-body");
    if (this.summaryRegion === null || !container || container.classList.contains("hidden") || !body?.querySelector(".rs")) {
      this.summaryRegion = null;
      return;
    }
    const audit = this.lastState && this.calculateRegionalScoring(this.lastState).find(a => a.regionId === this.summaryRegion);
    if (audit) this.openRegionSummary(audit);
  }

  private hideRegionTooltip() {
    if (!this.tooltipEl) return;
    this.tooltipEl.classList.add("hidden");
    this.tooltipEl.classList.remove("region-tip");
  }
}

/** "USSR +2, US −1", or what a target that moved nothing was: "targeted, no change". */
export function describeLastTurnCountry(lc: LastTurnCountry): string {
  const signed = (v: number) => (v > 0 ? `+${v}` : `\u2212${-v}`);
  const parts: string[] = [];
  if (lc.ussr) parts.push(`${sideName("USSR")} ${signed(lc.ussr)}`);
  if (lc.us) parts.push(`${sideName("US")} ${signed(lc.us)}`);
  return parts.length ? parts.join(", ") : t("map:targetedNoChange");
}
