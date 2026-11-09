/**
 * Showing what the model believed, next to what it did.
 *
 * Four views over the same per-step trace (see `ai/eval/policy_readout.py`):
 *   - the value ribbon, the calibrated critic as P(US wins) across the whole game, under the
 *     timeline;
 *   - a probability chip on each log row, and a chip for the calibrated value's move;
 *   - **a probability on every choosable thing** -- each card in hand, each mode button, each
 *     country on the map -- for the decision about to be made from the position on screen;
 *   - the readout panel: each side's P(victory) and expected VP from its own point of view, the
 *     side to move shown plainly and the other dimmed, since only its reading is trained.
 *
 * **Which decision the board's numbers belong to.** A replay step's snapshot is the position
 * *after* its action, so the board on screen is the node the *next* step was decided at -- its
 * `decision_context` is that decision, and the HUD is already rendering its buttons. The
 * probabilities painted on the board therefore come from step N+1, while the critic numbers
 * describe the position itself, which is step N's `critic`. Pairing them the other way would
 * label a hand that no longer holds the card that was played.
 *
 * Every view is optional at runtime: a replay from a heuristic bot, a human game, or one
 * recorded before the trace existed has no `policy`/`critic` keys, and the views hide rather
 * than render zeros.
 */
import { ReplayStep, PolicyTrace, CriticTrace } from "./replay_controls";
import { GameState } from "./types";
import { engineLayout } from "./engine/layout";
import { escapeHtml, num } from "./html";
import { sideName, t } from "./i18n";

const FLAG_CONFIRM_DONE = 0x80;

/**
 * The flat action index for a decision + primary id, or null when it does not map to one. The
 * layout is the engine's own (engine/layout.ts): a second copy maintained here would eventually
 * disagree with the engine, and every probability would then be painted on the wrong card or
 * country while still looking entirely plausible.
 */
function flatIndex(decisionType: number, primaryId: number, flags: number): number | null {
  const actionSpace = engineLayout();
  if (!actionSpace) return null;
  const dt = actionSpace.decision_types;
  const off = actionSpace.offsets;
  if (flags & FLAG_CONFIRM_DONE) return actionSpace.confirm_done_index;
  switch (decisionType) {
    case dt.SELECT_CARD:
      return primaryId >= 1 && primaryId <= 110 ? off.card + primaryId - 1 : null;
    case dt.SELECT_PLAY_MODE:
      return off.play_mode + primaryId;
    // CHOOSE_TIMING_BRANCH is retired (P17) and has no flat slots: nothing to paint.
    case dt.SELECT_OP_MODE:
      return off.op_mode + primaryId;
    case dt.POINT_NODE:
      return off.node + primaryId;
    case dt.CHOOSE_BRANCH:
      return off.branch + primaryId;
    default:
      return null;
  }
}

/** Colour for a probability: the redder, the less the policy expected that move. */
export function probColor(p: number): string {
  if (p >= 0.5) return "var(--success)";
  if (p >= 0.2) return "var(--warning)";
  if (p >= 0.05) return "#f97316";
  return "var(--danger)";
}

/** Three decimals, but never a bare "0.000" for something that is merely unlikely. */
export function fmtP(p: number): string {
  if (p >= 0.0005) return p.toFixed(3);
  return p > 0 ? "<.001" : "0";
}


// -- which critic reading to believe ------------------------------------------------------------
//
// The critic is read from both sides' observations, but the value head is trained only on the
// observation of the player who is deciding. That side's reading is the calibrated one; the
// other side's is an untrained extrapolation and can be far off (+0.99 where the truth is about
// -0.35 has been seen). Every view below therefore shows each side's own P(win) = (1 + v)/2,
// highlights the decider's and dims the other, and plots only the decider's -- converted to the
// US point of view -- as the value of a position. A terminal position has no decider.

export type Side = "US" | "USSR";

/** `v_vp` is the predicted final VP margin divided by this (the automatic-victory threshold). */
export const VP_LIMIT = 20;

export function asSide(p: string | undefined | null): Side | null {
  return p === "US" || p === "USSR" ? p : null;
}

/** An expected result v in [-1, 1] as a probability of winning (draws are rare). */
export function pWin(v: number): number {
  return Math.max(0, Math.min(1, (1 + v) / 2));
}

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function signed1(v: number): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
}

/**
 * Who decides at the position replay step `i`'s critic was read on.
 *
 * `critic.at === "after"` (every writer today) reads the state after the step's action -- the
 * position the *next* step is decided from, so the decider is the next step's player. The last
 * step has no next one; its snapshot's own decision context stands in, unless the game is over.
 * A terminal snapshot has no decider even though its stale decision context still names one.
 */
export function replayDecider(steps: ReplayStep[], i: number): Side | null {
  const step = steps[i];
  if (!step) return null;
  if (step.critic?.at === "before") return asSide(step.player);
  if (step.state_snapshot?.is_terminal) return null;
  return asSide(steps[i + 1]?.player) ?? asSide(step.state_snapshot?.decision_context?.decision_player);
}

/** The calibrated and the untrained reading of one position, both as P(US wins). */
export interface CalibratedValue {
  decider: Side;
  pUs: number;
  pUsOther: number | null;
}

/** The decider's reading converted to the US point of view, or null when there is no decider. */
export function calibratedValue(c: CriticTrace | undefined, decider: Side | null): CalibratedValue | null {
  if (!c || !decider) return null;
  const vUs = c.v_win_us;
  const vUssr = c.v_win_ussr;
  if (decider === "US") {
    if (vUs === undefined) return null;
    return { decider, pUs: pWin(vUs), pUsOther: vUssr === undefined ? null : 1 - pWin(vUssr) };
  }
  if (vUssr === undefined) return null;
  return { decider, pUs: 1 - pWin(vUssr), pUsOther: vUs === undefined ? null : pWin(vUs) };
}

const seriesCache = new WeakMap<ReplayStep[], Array<CalibratedValue | null>>();

/** Every step's calibrated value (memoised per replay: each log row asks for two of them). */
export function calibratedSeries(steps: ReplayStep[]): Array<CalibratedValue | null> {
  let series = seriesCache.get(steps);
  if (!series) {
    series = steps.map((s, i) => calibratedValue(s.critic, replayDecider(steps, i)));
    seriesCache.set(steps, series);
  }
  return series;
}

/** How a finished game came out, from the terminal display state. */
export function resultText(state: GameState | null | undefined): string {
  if (!state) return t("trace:result.gameOver");
  const u = num(state.terminal_utility);
  const outcome = u > 0 ? t("trace:result.usWon") : u < 0 ? t("trace:result.ussrWon") : t("trace:result.drawn");
  const vp = state.victory_points === undefined ? undefined : num(state.victory_points);
  return vp !== undefined ? t("trace:result.outcomeVp", { outcome, vp: `${vp > 0 ? "+" : ""}${vp}` })
    : t("trace:result.outcome", { outcome });
}

/**
 * The critic, for both sides: each side's P(victory) = (1 + v_win)/2 and its expected final VP
 * margin, v_vp × 20, both from that side's own point of view. The side to move is shown plainly
 * and the other dimmed (only the decider's reading is the one the value head is trained on). A
 * finished game has no side to move: both are dimmed and the result is given below.
 */
export function valueSidesHtml(c: CriticTrace, decider: Side | null,
                               terminal?: GameState | null): string {
  const side = (s: Side): string => {
    const v = s === "US" ? c.v_win_us : c.v_win_ussr;
    const vp = s === "US" ? c.v_vp_us : c.v_vp_ussr;
    const cls = s.toLowerCase();
    const status = decider === s ? "calibrated" : "uncalibrated";
    const p = v === undefined ? null : pWin(v);
    const evp = vp === undefined ? null : vp * VP_LIMIT;
    return `
      <div class="value-side ${cls} ${status}" data-side="${s}" data-p="${p === null ? "" : p.toFixed(6)}" data-vp="${evp === null ? "" : evp.toFixed(4)}"
           title="${decider && decider !== s
             ? t("trace:sides.titleOtherToMove", { side: sideName(s), limit: VP_LIMIT, decider: sideName(decider) })
             : t("trace:sides.title", { side: sideName(s), limit: VP_LIMIT })}">
        <div class="value-side-name">${sideName(s)}</div>
        <div class="value-side-p">${p === null ? "–" : pct(p)}</div>
        <div class="value-side-vp">${t("trace:sides.vp", { vp: evp === null ? "–" : signed1(evp) })}</div>
      </div>`;
  };
  const parts = [`<div class="value-sides" data-decider="${decider ?? "none"}">${side("US")}${side("USSR")}</div>`];
  if (!decider && terminal) parts.push(`<div class="value-result">${resultText(terminal)}</div>`);
  return parts.join("");
}

/** The small `p=0.42` chip that goes on a replay log row, and the calibrated value's move. */
export function policyChipHtml(steps: ReplayStep[], idx: number): string {
  const step = steps[idx];
  const pol = step.policy;
  const parts: string[] = [];
  if (pol && pol.source === "forced") {
    parts.push(`<span class="trace-chip trace-chip-forced" title="${t("trace:chip.forcedTitle")}">${t("trace:chip.forced")}</span>`);
  } else if (pol && pol.p_chosen !== undefined) {
    const p = num(pol.p_chosen);
    const offMode = pol.argmax_idx !== undefined && pol.argmax_idx !== pol.chosen_idx;
    const best = pol.p_max !== undefined ? num(pol.p_max).toFixed(3) : undefined;
    const title = best !== undefined
      ? t(offMode ? "trace:chip.titleBestNotMode" : "trace:chip.titleBest", { best })
      : t(offMode ? "trace:chip.titleNotMode" : "trace:chip.title");
    parts.push(
      `<span class="trace-chip" style="border-color:${probColor(p)};color:${probColor(p)}" title="${title}">`
      + `p=${p.toFixed(3)}${offMode ? " ✳" : ""}</span>`);
  }
  const series = calibratedSeries(steps);
  const cur = series[idx];
  const prev = idx > 0 ? series[idx - 1] : null;
  if (cur && prev) {
    const dp = cur.pUs - prev.pUs;
    // Only a move worth noticing gets a badge; every step nudges the value a little. 2.5 points
    // of probability is the 0.05 of v_win this chip used to be keyed on.
    if (Math.abs(dp) >= 0.025) {
      const cls = dp > 0 ? "trace-dv-us" : "trace-dv-ussr";
      parts.push(`<span class="trace-chip ${cls}" title="${t("trace:chip.deltaTitle", { before: sideName(prev.decider), after: sideName(cur.decider) })}">ΔP ${dp > 0 ? "+" : "−"}${Math.abs(dp * 100).toFixed(0)}%</span>`);
    }
  }
  return parts.join("");
}

/**
 * The value ribbon: one column per step, the calibrated critic as P(US wins) -- each position
 * read from the side to move there, never the untrained side -- with 50% through the middle.
 * The untrained reading of the same positions is the faint dashed line, for comparison only.
 * Clicking seeks. Drawn as inline SVG -- the repo draws its map the same way, and a charting
 * dependency for one sparkline is not worth the bytes.
 */
export function renderValueRibbon(
  container: HTMLElement,
  steps: ReplayStep[],
  currentIndex: number,
  onSeek: (index: number) => void,
): void {
  const series = calibratedSeries(steps);
  const values: Array<number | null> = series.map(c => (c ? c.pUs : null));
  const others: Array<number | null> = series.map(c => (c ? c.pUsOther : null));
  if (!values.some(v => v !== null)) {
    container.classList.add("hidden");
    container.innerHTML = "";
    return;
  }
  container.classList.remove("hidden");

  const W = 1000, H = 60, mid = H / 2;
  const n = values.length;
  const x = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * W);
  const y = (p: number) => mid - (Math.max(0, Math.min(1, p)) * 2 - 1) * (mid - 3);

  // One path per contiguous run of values, so gaps (a replay traced only at decision nodes, the
  // terminal position) stay gaps instead of being bridged by a line that was never measured.
  const runs = (vals: Array<number | null>): Array<Array<{ i: number; v: number }>> => {
    const out: Array<Array<{ i: number; v: number }>> = [];
    let run: Array<{ i: number; v: number }> = [];
    vals.forEach((v, i) => {
      if (v === null) { if (run.length) out.push(run); run = []; } else { run.push({ i, v }); }
    });
    if (run.length) out.push(run);
    return out;
  };
  const linePath = (run: Array<{ i: number; v: number }>) =>
    run.map((pt, k) => `${k === 0 ? "M" : "L"}${x(pt.i).toFixed(1)},${y(pt.v).toFixed(1)}`).join("");

  const paths: string[] = [];
  for (const run of runs(others)) {
    paths.push(`<path class="ribbon-uncalibrated" d="${linePath(run)}" fill="none" stroke="var(--text-muted)" stroke-opacity="0.45" stroke-dasharray="3 3" stroke-width="1" vector-effect="non-scaling-stroke"/>`);
  }
  for (const run of runs(values)) {
    const line = linePath(run);
    const close = `L${x(run[run.length - 1].i).toFixed(1)},${mid} L${x(run[0].i).toFixed(1)},${mid} Z`;
    paths.push(`<path d="${line}${close}" fill="url(#ribbon-fill)" stroke="none"/>`);
    paths.push(`<path class="ribbon-calibrated" d="${line}" fill="none" stroke="var(--us-blue-light)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`);
  }

  // VP changes: the events the value curve should be explaining.
  const ticks: string[] = [];
  steps.forEach((s, i) => {
    const prevVp = i > 0 ? steps[i - 1].state_snapshot?.victory_points : undefined;
    const vp = s.state_snapshot?.victory_points;
    if (prevVp !== undefined && vp !== undefined && vp !== prevVp) {
      ticks.push(`<line x1="${x(i).toFixed(1)}" y1="0" x2="${x(i).toFixed(1)}" y2="${H}" stroke="var(--text-dim)" stroke-dasharray="2 3" stroke-width="1" vector-effect="non-scaling-stroke"/>`);
    }
  });

  const cx = x(Math.max(0, Math.min(n - 1, currentIndex)));
  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="value-ribbon-svg">
      <defs>
        <linearGradient id="ribbon-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--us-blue-light)" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="var(--us-blue-light)" stop-opacity="0.05"/>
        </linearGradient>
      </defs>
      ${ticks.join("")}
      <line x1="0" y1="${mid}" x2="${W}" y2="${mid}" stroke="var(--text-dim)" stroke-width="1" vector-effect="non-scaling-stroke"/>
      ${paths.join("")}
      <line x1="${cx.toFixed(1)}" y1="0" x2="${cx.toFixed(1)}" y2="${H}" stroke="var(--warning)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
    </svg>
    <span class="ribbon-label ribbon-label-top">${sideName("US")}</span>
    <span class="ribbon-label ribbon-label-bottom">${sideName("USSR")}</span>`;

  container.onclick = (ev: MouseEvent) => {
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0 || n <= 1) return;
    const frac = (ev.clientX - rect.left) / rect.width;
    onSeek(Math.round(Math.max(0, Math.min(1, frac)) * (n - 1)));
  };
}

// -- probabilities on the things you would click ----------------------------------------------

export function clearDecorations(): void {
  document.querySelectorAll(".trace-choice-badge").forEach(el => el.remove());
  document.querySelectorAll(".trace-choice-played").forEach(el =>
    el.classList.remove("trace-choice-played"));
}

/**
 * How a badge is singled out: `played` is the move a replay went on to make (◀), `favourite`
 * the live model's own most likely move (★). Both get the same highlight -- it is the one to
 * compare everything else against -- and a different glyph, because they are different claims.
 */
export type BadgeMark = "played" | "favourite" | null;

export function htmlBadge(p: number, mark: BadgeMark, note?: string): HTMLElement {
  const span = document.createElement("span");
  span.className = "trace-choice-badge" + (mark ? " played" : "");
  span.style.color = probColor(p);
  span.style.borderColor = probColor(p);
  span.textContent = fmtP(p) + (mark === "played" ? " ◀" : mark === "favourite" ? " ★" : "");
  span.title = t(mark === "played" ? "trace:badge.titlePlayed" : mark === "favourite" ? "trace:badge.titleFavourite"
    : "trace:badge.title", { p }) + (note ? `\n${note}` : "");
  return span;
}

/**
 * A country's probability, small, in the free corner right of its influence markers: placed from
 * the rightmost marker (`rect.inf-box`, map_view.ts) so it can never cover a number.
 */
export function svgBadge(g: Element, p: number, played: boolean, note?: string): void {
  const rect = g.querySelector("rect.country-card-bg");
  const boxes = [...g.querySelectorAll("rect.inf-box")];
  if (!rect || !boxes.length) return;
  const num = (e: Element, a: string) => parseFloat(e.getAttribute(a) || "0");
  const inf = boxes.reduce((a, b) => (num(b, "x") > num(a, "x") ? b : a));
  const right = num(rect, "x") + num(rect, "width") - 0.9;
  const x = num(inf, "x") + num(inf, "width") + 1.0;
  const w = right - x, h = 5.6;
  const y = num(inf, "y") + (num(inf, "height") - h) / 2;

  const NS = "http://www.w3.org/2000/svg";
  const box = document.createElementNS(NS, "rect");
  box.setAttribute("class", "trace-choice-badge");
  box.setAttribute("x", x.toString());
  box.setAttribute("y", y.toString());
  box.setAttribute("width", w.toString());
  box.setAttribute("height", h.toString());
  box.setAttribute("rx", "0.8");
  box.setAttribute("fill", played ? "#422006" : "#0F172A");
  box.setAttribute("stroke", played ? "var(--warning)" : probColor(p));
  box.setAttribute("stroke-width", played ? "0.7" : "0.45");
  g.appendChild(box);

  const text = document.createElementNS(NS, "text");
  text.setAttribute("class", "trace-choice-badge");
  text.setAttribute("x", (x + w / 2).toString());
  text.setAttribute("y", (y + h / 2 + 1.15).toString());
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("fill", played ? "#FDE68A" : "#E2E8F0");
  text.textContent = fmtP(p);
  // As large as fits: bold digits run to about 0.62 em. (Not textLength, which squeezes the glyphs
  // but leaves the text's measured box at its natural width.)
  text.setAttribute("font-size", Math.min(3.1, (w - 0.4) / (text.textContent.length * 0.62)).toFixed(2));
  text.setAttribute("font-weight", "bold");
  if (note) {
    const title = document.createElementNS(NS, "title");
    title.textContent = `${fmtP(p)} -- ${note}`;
    text.appendChild(title);
  }
  g.appendChild(text);
}

/**
 * Put each legal action's probability on the thing you would click to take it.
 *
 * `policy` is the trace of the step decided *from the position now on screen* (step N+1 while
 * viewing step N), and `state` is that position -- its `decision_context` names which kind of
 * choice is open, which is what decides whether the numbers belong on the cards, the HUD
 * buttons or the map.
 */
export function decorateChoices(policy: PolicyTrace | undefined, state: GameState | null): void {
  clearDecorations();
  const entries = policy?.top;
  const actionSpace = engineLayout();
  if (!entries || entries.length === 0 || !state || !actionSpace) return;

  const dType = state.decision_context?.decision_type;
  if (dType === undefined) return;

  const p = new Map<number, number>();
  entries.forEach(e => p.set(e.idx, e.p));
  const chosen = policy?.chosen_idx;
  const dt = actionSpace.decision_types;

  // HUD buttons: play mode, op mode, timing, branch, the target list, and pass/confirm.
  document.querySelectorAll<HTMLElement>("#decision-body [data-primary]").forEach(btn => {
    const primary = parseInt(btn.getAttribute("data-primary") || "", 10);
    if (Number.isNaN(primary)) return;
    const flags = parseInt(btn.getAttribute("data-flags") || "0", 10) || 0;
    const idx = flatIndex(dType, primary, flags);
    if (idx === null || !p.has(idx)) return;
    const prob = p.get(idx)!;
    const played = idx === chosen;
    if (played) btn.classList.add("trace-choice-played");
    btn.appendChild(htmlBadge(prob, played ? "played" : null));
  });

  // Cards in hand, when a card is what is being chosen.
  if (dType === dt.SELECT_CARD) {
    document.querySelectorAll<HTMLElement>(".card-item[data-card-id]").forEach(el => {
      const cid = parseInt(el.getAttribute("data-card-id") || "", 10);
      const idx = flatIndex(dType, cid, 0);
      if (idx === null || !p.has(idx)) return;
      const prob = p.get(idx)!;
      const played = idx === chosen;
      if (played) el.classList.add("trace-choice-played");
      el.appendChild(htmlBadge(prob, played ? "played" : null));
    });
  }

  // Countries on the map, when a country is what is being chosen.
  if (dType === dt.POINT_NODE) {
    document.querySelectorAll(".svg-country-node[data-id]").forEach(g => {
      const cid = parseInt(g.getAttribute("data-id") || "", 10);
      const idx = flatIndex(dType, cid, 0);
      if (idx === null || !p.has(idx)) return;
      const prob = p.get(idx)!;
      const played = idx === chosen;
      if (played) g.classList.add("trace-choice-played");
      svgBadge(g, prob, played);
    });
  }
}

// -- the right-rail panel ---------------------------------------------------------------------

function nextDecisionHtml(pol: PolicyTrace | undefined, index: number): string {
  if (!pol) {
    return `<div class="trace-empty">${t("trace:next.none")}</div>`;
  }
  if (pol.source === "forced") {
    return `<div class="trace-empty">${t("trace:next.forced")}</div>`;
  }
  if (pol.source === "scripted") {
    return `<div class="trace-empty">${t("trace:next.scripted")}</div>`;
  }
  // A replay's trace is the file's: every field is checked or escaped on its way into the HTML.
  const offMode = pol.argmax_idx !== undefined && pol.argmax_idx !== pol.chosen_idx;
  const played = (pol.top || []).find(e => e.idx === pol.chosen_idx);
  const pChosen = num(pol.p_chosen ?? 0);
  const nLegal = pol.n_legal !== undefined ? num(pol.n_legal) : undefined;
  return `
    <div class="trace-head">
      <span class="trace-head-item" title="${t("trace:next.playsTitle")}">${t("trace:next.plays", { p: `<b style="color:${probColor(pChosen)}">${pChosen.toFixed(5)}</b>` })}</span>
      <span class="trace-head-item" title="${t("trace:next.bestTitle")}">${t("trace:next.best", { p: num(pol.p_max ?? 0).toFixed(5) })}</span>
      ${pol.entropy !== undefined ? `<span class="trace-head-item" title="${t("trace:next.entropyTitle")}">H ${num(pol.entropy).toFixed(3)}</span>` : ""}
      ${nLegal !== undefined ? `<span class="trace-head-item">${t("trace:next.legal", { n: nLegal })}</span>` : ""}
      ${offMode ? `<span class="trace-head-item trace-offmode" title="${t("trace:next.offModeTitle")}">${t("trace:next.offMode")}</span>` : ""}
    </div>
    <div class="trace-note">${t("trace:next.step", { n: index + 2, move: escapeHtml(played?.name ?? `#${num(pol.chosen_idx)}`) })}</div>
    <div class="trace-note trace-hint">${nLegal !== undefined ? t("trace:next.hintCount", { n: nLegal }) : t("trace:next.hintAll")}</div>`;
}

/**
 * Fill the readout panel: the critic's prediction for the position on screen -- each side's own
 * chance, the side to move highlighted -- then a summary of the decision whose probabilities are
 * painted on the board.
 */
export function renderTracePanel(steps: ReplayStep[], index: number): void {
  const panel = document.getElementById("trace-panel");
  const body = document.getElementById("trace-panel-body");
  const badge = document.getElementById("trace-source-badge");
  if (!panel || !body) return;

  const current = steps[index];
  const next = steps[index + 1];
  const critic = current?.critic;
  const nextPolicy = next?.policy;
  if (!critic && !nextPolicy) {
    panel.classList.add("hidden");
    body.innerHTML = "";
    return;
  }
  panel.classList.remove("hidden");
  if (badge) badge.textContent = nextPolicy?.source ?? t("trace:panel.criticOnly");

  const sections: string[] = [];
  if (critic) {
    const decider = replayDecider(steps, index);
    const terminal = current?.state_snapshot?.is_terminal ? current.state_snapshot : null;
    sections.push(`<div class="trace-section-label">${decider
      ? t("trace:panel.positionToMove", { n: index + 1, side: `<span class="analysis-who ${decider.toLowerCase()}">${sideName(decider)}</span>` })
      : t("trace:panel.position", { n: index + 1 })}</div>`);
    sections.push(valueSidesHtml(critic, decider, terminal));
  }
  sections.push(`<div class="trace-section-label">${t("trace:panel.next")}</div>`);
  sections.push(nextDecisionHtml(nextPolicy, index));
  body.innerHTML = sections.join("");
}
