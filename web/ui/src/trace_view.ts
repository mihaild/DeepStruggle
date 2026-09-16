/**
 * Showing what the model believed, next to what it did.
 *
 * Three views over the same per-step trace (see `ai/eval/policy_readout.py`):
 *   - the value ribbon, the critic's win value across the whole game, under the timeline;
 *   - a probability chip on each log row;
 *   - the readout panel, the distribution at the selected step.
 *
 * Every one of them is optional at runtime: a replay from a heuristic bot, a human game, or one
 * recorded before the trace existed simply has no `policy`/`critic` keys, and the views hide
 * rather than render zeros.
 */
import { ReplayStep, PolicyTrace, CriticTrace } from "./replay_controls";

/** Colour for a probability: the redder, the less the policy expected its own move. */
function probColor(p: number): string {
  if (p >= 0.5) return "var(--success)";
  if (p >= 0.2) return "var(--warning)";
  if (p >= 0.05) return "#f97316";
  return "var(--danger)";
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

function criticRowsHtml(critic: CriticTrace): string {
  const cell = (v: number | undefined, nd = 3) => (v === undefined ? "–" : v.toFixed(nd));
  return `
    <table class="trace-critic">
      <tr><th></th><th>US</th><th>USSR</th></tr>
      <tr><td>v_win</td><td>${cell(critic.v_win_us)}</td><td>${cell(critic.v_win_ussr)}</td></tr>
      <tr><td>v_vp</td><td>${cell(critic.v_vp_us, 2)}</td><td>${cell(critic.v_vp_ussr, 2)}</td></tr>
      <tr class="trace-residual" title="A zero-sum critic must put these at zero; what is left is model error">
        <td>residual</td><td>${cell(critic.win_residual)}</td><td>${cell(critic.vp_residual, 2)}</td></tr>
    </table>`;
}

function policyBarsHtml(pol: PolicyTrace): string {
  const top = pol.top || [];
  if (top.length === 0) {
    return `<div class="trace-empty">No distribution recorded for this step (${pol.source ?? "unknown"}).</div>`;
  }
  const max = Math.max(...top.map(e => e.p), 1e-6);
  const rows = top.map(e => {
    const isChosen = e.idx === pol.chosen_idx;
    const isMode = e.idx === pol.argmax_idx;
    const width = Math.max(1, (e.p / max) * 100);
    return `
      <div class="trace-bar-row ${isChosen ? "chosen" : ""}">
        <div class="trace-bar-label" title="${e.name ?? `#${e.idx}`}">${e.name ?? `#${e.idx}`}</div>
        <div class="trace-bar-track"><div class="trace-bar-fill" style="width:${width.toFixed(1)}%;background:${isChosen ? probColor(e.p) : "var(--us-blue-light)"}"></div></div>
        <div class="trace-bar-p">${e.p.toFixed(3)}${isChosen ? " ◀" : isMode ? " ✳" : ""}</div>
      </div>`;
  }).join("");
  const tail = (pol.p_tail ?? 0) > 0.0005
    ? `<div class="trace-tail">+ ${pol.p_tail!.toFixed(3)} spread over the remaining ${(pol.n_legal ?? 0) - top.length} legal actions</div>`
    : "";
  return rows + tail;
}

/** Fill the right-rail readout panel for one step, or hide it when there is nothing to show. */
export function renderTracePanel(step: ReplayStep | undefined): void {
  const panel = document.getElementById("trace-panel");
  const body = document.getElementById("trace-panel-body");
  const badge = document.getElementById("trace-source-badge");
  if (!panel || !body) return;

  const pol = step?.policy;
  const critic = step?.critic;
  if (!step || (!pol && !critic)) {
    panel.classList.add("hidden");
    body.innerHTML = "";
    return;
  }
  panel.classList.remove("hidden");
  if (badge) badge.textContent = pol?.source ?? "critic only";

  const head: string[] = [];
  if (pol && pol.p_chosen !== undefined) {
    const offMode = pol.argmax_idx !== undefined && pol.argmax_idx !== pol.chosen_idx;
    head.push(`<div class="trace-head">
      <span class="trace-head-item" title="probability the policy put on the move it played, at temperature 1">played <b style="color:${probColor(pol.p_chosen)}">${pol.p_chosen.toFixed(3)}</b></span>
      <span class="trace-head-item" title="probability of its most likely move">best ${(pol.p_max ?? 0).toFixed(3)}</span>
      ${pol.entropy !== undefined ? `<span class="trace-head-item" title="entropy of the distribution, in nats">H ${pol.entropy.toFixed(2)}</span>` : ""}
      ${pol.n_legal !== undefined ? `<span class="trace-head-item">${pol.n_legal} legal</span>` : ""}
      ${offMode ? `<span class="trace-head-item trace-offmode" title="the sampler did not take the policy's own best move">off-mode</span>` : ""}
    </div>`);
  }
  if (pol && pol.temperature !== undefined && pol.p_chosen_sampled !== undefined
      && pol.temperature !== 1.0) {
    head.push(`<div class="trace-note">sampled at T=${pol.temperature} → ${pol.p_chosen_sampled.toFixed(3)}</div>`);
  }

  body.innerHTML = head.join("")
    + (pol ? policyBarsHtml(pol) : "")
    + (critic ? criticRowsHtml(critic) : "");
}
