/**
 * Showing what the model believed, next to what it did.
 *
 * Four views over the same per-step trace (see `ai/eval/policy_readout.py`):
 *   - the value ribbon, the critic's win value across the whole game, under the timeline;
 *   - a probability chip on each log row;
 *   - **a probability on every choosable thing** -- each card in hand, each mode button, each
 *     country on the map -- for the decision about to be made from the position on screen;
 *   - the readout panel, with the critic's prediction for both sides to five decimals.
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
import { ActionLayout } from "./engine/wasm_engine";

let actionSpace: ActionLayout | null = null;

/**
 * The flat action space's layout, from the engine running in the page (WasmEngine.layout).
 *
 * A second copy maintained here would eventually disagree with the engine, and every probability
 * would then be painted on the wrong card or country while still looking entirely plausible.
 */
export function setActionSpace(layout: ActionLayout): void {
  actionSpace = layout;
}

const FLAG_CONFIRM_DONE = 0x80;

/** The flat action index for a decision + primary id, or null when it does not map to one. */
function flatIndex(decisionType: number, primaryId: number, flags: number): number | null {
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

/** Fixed-width signed value, the precision the panel promises. */
function fmt5(v: number | undefined): string {
  return v === undefined ? "–" : (v >= 0 ? "+" : "") + v.toFixed(5);
}

/** The small `p=0.42` chip that goes on a replay log row. */
export function policyChipHtml(step: ReplayStep, prev?: ReplayStep): string {
  const pol = step.policy;
  const parts: string[] = [];
  if (pol && pol.source === "forced") {
    parts.push(`<span class="trace-chip trace-chip-forced" title="One legal action -- played by the loop, not chosen">forced</span>`);
  } else if (pol && pol.p_chosen !== undefined) {
    const p = pol.p_chosen;
    const offMode = pol.argmax_idx !== undefined && pol.argmax_idx !== pol.chosen_idx;
    const title = `policy probability of the move it played (T=1)`
      + (pol.p_max !== undefined ? `; best was ${pol.p_max.toFixed(3)}` : "")
      + (offMode ? " -- not the mode" : "");
    parts.push(
      `<span class="trace-chip" style="border-color:${probColor(p)};color:${probColor(p)}" title="${title}">`
      + `p=${p.toFixed(3)}${offMode ? " ✳" : ""}</span>`);
  }
  const v = step.critic?.v_win_us;
  const pv = prev?.critic?.v_win_us;
  if (v !== undefined && pv !== undefined) {
    const dv = v - pv;
    // Only a move worth noticing gets a badge; every step nudges the value a little.
    if (Math.abs(dv) >= 0.05) {
      const cls = dv > 0 ? "trace-dv-us" : "trace-dv-ussr";
      parts.push(`<span class="trace-chip ${cls}" title="critic v_win (US) moved by this much across this step">Δv ${dv > 0 ? "+" : ""}${dv.toFixed(2)}</span>`);
    }
  }
  return parts.join("");
}

/**
 * The value ribbon: one column per step, the critic's US win value, zero line through the
 * middle. Clicking seeks. Drawn as inline SVG -- the repo draws its map the same way, and a
 * charting dependency for one sparkline is not worth the bytes.
 */
export function renderValueRibbon(
  container: HTMLElement,
  steps: ReplayStep[],
  currentIndex: number,
  onSeek: (index: number) => void,
): void {
  const values: Array<number | null> = steps.map(s =>
    s.critic?.v_win_us !== undefined ? s.critic.v_win_us : null);
  if (!values.some(v => v !== null)) {
    container.classList.add("hidden");
    container.innerHTML = "";
    return;
  }
  container.classList.remove("hidden");

  const W = 1000, H = 60, mid = H / 2;
  const n = values.length;
  const x = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * W);
  const y = (v: number) => mid - Math.max(-1, Math.min(1, v)) * (mid - 3);

  // One area per contiguous run of values, so gaps (a replay traced only at decision nodes)
  // stay gaps instead of being bridged by a line that was never measured.
  const areas: string[] = [];
  let run: Array<{ i: number; v: number }> = [];
  const flush = () => {
    if (run.length === 0) return;
    const line = run.map((pt, k) => `${k === 0 ? "M" : "L"}${x(pt.i).toFixed(1)},${y(pt.v).toFixed(1)}`).join("");
    const close = `L${x(run[run.length - 1].i).toFixed(1)},${mid} L${x(run[0].i).toFixed(1)},${mid} Z`;
    areas.push(`<path d="${line}${close}" fill="url(#ribbon-fill)" stroke="none"/>`);
    areas.push(`<path d="${line}" fill="none" stroke="var(--us-blue-light)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`);
    run = [];
  };
  values.forEach((v, i) => { if (v === null) { flush(); } else { run.push({ i, v }); } });
  flush();

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
      ${areas.join("")}
      <line x1="${cx.toFixed(1)}" y1="0" x2="${cx.toFixed(1)}" y2="${H}" stroke="var(--warning)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
    </svg>
    <span class="ribbon-label ribbon-label-top">US +1</span>
    <span class="ribbon-label ribbon-label-bottom">USSR −1</span>`;

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
  span.title = `policy probability ${p} (temperature 1)`
    + (mark === "played" ? " -- the move it played" : mark === "favourite" ? " -- the model's favourite" : "")
    + (note ? `\n${note}` : "");
  return span;
}

export function svgBadge(g: Element, p: number, played: boolean, note?: string): void {
  const rect = g.querySelector("rect.country-card-bg");
  if (!rect) return;
  const bx = parseFloat(rect.getAttribute("x") || "0");
  const by = parseFloat(rect.getAttribute("y") || "0");
  const bw = parseFloat(rect.getAttribute("width") || "0");
  const bh = parseFloat(rect.getAttribute("height") || "0");
  const w = 12.4, h = 5.6;
  const x = bx + bw - w - 1.2;
  const y = by + bh - h - 1.2;

  const NS = "http://www.w3.org/2000/svg";
  const box = document.createElementNS(NS, "rect");
  box.setAttribute("class", "trace-choice-badge");
  box.setAttribute("x", x.toString());
  box.setAttribute("y", y.toString());
  box.setAttribute("width", w.toString());
  box.setAttribute("height", h.toString());
  box.setAttribute("rx", "1.0");
  box.setAttribute("fill", played ? "#422006" : "#0F172A");
  box.setAttribute("stroke", played ? "var(--warning)" : probColor(p));
  box.setAttribute("stroke-width", played ? "1.0" : "0.7");
  g.appendChild(box);

  const text = document.createElementNS(NS, "text");
  text.setAttribute("class", "trace-choice-badge");
  text.setAttribute("x", (x + w / 2).toString());
  text.setAttribute("y", (y + h / 2 + 1.2).toString());
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("fill", played ? "#FDE68A" : "#E2E8F0");
  text.setAttribute("font-size", "3.4");
  text.setAttribute("font-weight", "bold");
  text.textContent = fmtP(p);
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

export function criticTableHtml(critic: CriticTrace): string {
  return `
    <table class="trace-critic">
      <tr><th>prediction</th><th>v_win</th><th>v_vp</th></tr>
      <tr class="row-us"><td>US</td><td>${fmt5(critic.v_win_us)}</td><td>${fmt5(critic.v_vp_us)}</td></tr>
      <tr class="row-ussr"><td>USSR</td><td>${fmt5(critic.v_win_ussr)}</td><td>${fmt5(critic.v_vp_ussr)}</td></tr>
      <tr class="trace-residual" title="A zero-sum critic must put these at zero; whatever is left is model error">
        <td>residual</td><td>${fmt5(critic.win_residual)}</td><td>${fmt5(critic.vp_residual)}</td></tr>
    </table>`;
}

function nextDecisionHtml(pol: PolicyTrace | undefined, index: number): string {
  if (!pol) {
    return `<div class="trace-empty">No recorded decision from this position.</div>`;
  }
  if (pol.source === "forced") {
    return `<div class="trace-empty">Next step is forced — one legal action, played by the loop.</div>`;
  }
  if (pol.source === "scripted") {
    return `<div class="trace-empty">Next step is a scripted opening — the policy was overruled.</div>`;
  }
  const offMode = pol.argmax_idx !== undefined && pol.argmax_idx !== pol.chosen_idx;
  const played = (pol.top || []).find(e => e.idx === pol.chosen_idx);
  return `
    <div class="trace-head">
      <span class="trace-head-item" title="probability the policy put on the move it went on to play, at temperature 1">plays <b style="color:${probColor(pol.p_chosen ?? 0)}">${(pol.p_chosen ?? 0).toFixed(5)}</b></span>
      <span class="trace-head-item" title="probability of its most likely move">best ${(pol.p_max ?? 0).toFixed(5)}</span>
      ${pol.entropy !== undefined ? `<span class="trace-head-item" title="entropy of the distribution, in nats">H ${pol.entropy.toFixed(3)}</span>` : ""}
      ${pol.n_legal !== undefined ? `<span class="trace-head-item">${pol.n_legal} legal</span>` : ""}
      ${offMode ? `<span class="trace-head-item trace-offmode" title="the sampler did not take the policy's own best move">off-mode</span>` : ""}
    </div>
    <div class="trace-note">step ${index + 2}: ${played?.name ?? `#${pol.chosen_idx}`}</div>
    <div class="trace-note trace-hint">probabilities for all ${pol.n_legal ?? "legal"} options are on the cards, buttons and map</div>`;
}

/**
 * Fill the right-rail panel: the critic's prediction for the position on screen, then a summary
 * of the decision whose probabilities are painted on the board.
 */
export function renderTracePanel(current: ReplayStep | undefined, next: ReplayStep | undefined,
                                 index: number): void {
  const panel = document.getElementById("trace-panel");
  const body = document.getElementById("trace-panel-body");
  const badge = document.getElementById("trace-source-badge");
  if (!panel || !body) return;

  const critic = current?.critic;
  const nextPolicy = next?.policy;
  if (!critic && !nextPolicy) {
    panel.classList.add("hidden");
    body.innerHTML = "";
    return;
  }
  panel.classList.remove("hidden");
  if (badge) badge.textContent = nextPolicy?.source ?? "critic only";

  const sections: string[] = [];
  if (critic) {
    sections.push(`<div class="trace-section-label">POSITION ON SCREEN — after step ${index + 1}</div>`);
    sections.push(criticTableHtml(critic));
  }
  sections.push(`<div class="trace-section-label">NEXT DECISION — from this position</div>`);
  sections.push(nextDecisionHtml(nextPolicy, index));
  body.innerHTML = sections.join("");
}
