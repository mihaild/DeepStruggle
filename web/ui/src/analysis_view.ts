/**
 * Live model analysis: pick a model, and on every position see what it would do and what its
 * critic thinks, while you keep playing either side by hand.
 *
 * Everything runs in the page: the engine as WebAssembly, the model as ONNX in onnxruntime-web
 * (analysis/model.ts), the readout in analysis/readout.ts. This panel picks the model -- from the
 * local server's checkpoints, a Hugging Face repo, or a file -- and shows the readout.
 *
 * Each choice carries the MicroAction a click would send (`decision_type`/`primary_id`/`flags`),
 * so a probability is attached to a button, card or country by matching what that element sends
 * -- no flat offsets are re-derived here. A *composed* choice belongs to a model that decides in
 * the E4.1 merged-influence view: "Ops -> Influence, first point in X" is one action for the
 * model and two clicks for a person, so it is painted on the country X, and the influence button
 * carries the sum of all of them -- the probability the model puts on choosing influence at all.
 */
import { GameState } from "./types";
import { engineLayout } from "./engine/layout";
import {
  BadgeMark, clearDecorations, fmtP, htmlBadge, probColor, svgBadge, valueSidesHtml,
} from "./trace_view";
import { CriticTrace, PolicyTrace } from "./replay_controls";
import {
  DEFAULT_HF_REPO, DEFAULT_HF_REVISION, defaultHfCatalog, groupHfModels, HfCatalog, hfCatalog, HfModelFile, ModelSource,
  sourceLabel,
} from "./analysis/model";
import { cachedModelCount, clearModelCache } from "./analysis/model_cache";
import { SearchResult } from "./analysis/rollout";
import { onLanguageChange, sideName, t } from "./i18n";

export interface AnalysisChoice {
  idx: number;
  p: number;
  name?: string;
  decision_type?: number;
  primary_id?: number;
  flags?: number;
  composed?: boolean;
  country_id?: number;
}

export interface LiveAnalysis {
  /** Which loaded model produced it (the app's key); a readout for another is dropped. */
  model: string;
  label?: string;
  merged_influence?: boolean;
  /** P42: the model decides in the macro (until-control) view. */
  macro_influence?: boolean;
  decision_player?: "US" | "USSR";
  decision_type?: number;
  critic?: CriticTrace;
  policy?: PolicyTrace;
  choices: AnalysisChoice[];
  /** The search's decision at this position, when the search select asks for one (analysis/rollout.ts). */
  search?: SearchResult;
  /** The search is running, or failed with this reason. */
  search_pending?: boolean;
  search_error?: string;
}

/** Which side auto-play moves for; "" is off. */
export type AutoSide = "" | "US" | "USSR";

/** A model the user picked: where from, and for a dropped file, its bytes. */
export interface ModelPick {
  source: ModelSource;
  bytes?: Uint8Array;
}

/** What the panel says about the loaded model. */
export interface LoadedInfo {
  key: string;
  label: string;
  merged: boolean;
  checkpoint: string;
  /** The file came from the browser's model cache, not a download. */
  fromCache?: boolean;
  /** Set when the model was exported next to another engine build than the page runs. */
  engineWarning?: string;
}

interface ModelList {
  root: string;
  runs: Array<{ run: string; snapshots: string[] }>;
  loose: string[];
}

const FLAG_CONFIRM_DONE = 0x80;

/** The run select's value for checkpoints outside any run (its label is `analysis:local.looseGroup`). */
const LOOSE_GROUP = "(loose checkpoints)";
const TOP_LISTED = 8;

/** `snapshot_240058368steps.pt` -> `240M`; the run name already says which run it is. */
function snapshotLabel(name: string): string {
  const m = name.match(/^snapshot_(\d+)steps\.pt$/);
  if (m) {
    const n = parseInt(m[1], 10);
    return n >= 1e6 ? t("analysis:snapshot.millionSteps", { m: Math.round(n / 1e6) }) : t("analysis:snapshot.steps", { n });
  }
  if (name === "snapshot_final.pt") return t("analysis:snapshot.final");
  if (name === "snapshot_0s.pt") return t("analysis:snapshot.start");
  return name;
}

/** What a HUD button with these attributes sends, as a lookup key. */
function clickKey(primary: number, flags: number): string {
  return (flags & FLAG_CONFIRM_DONE) ? "done" : `p${primary}`;
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
}

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

export class AnalysisPanel {
  private models: ModelList | null = null;
  private status: "off" | "loading" | "ready" | "error" = "off";
  private pending: ModelSource | null = null;
  private loaded: LoadedInfo | null = null;
  private error: string | null = null;
  private lastAnalysis: LiveAnalysis | null = null;
  private auto: AutoSide = "";
  /**
   * A player's view (game/view.ts): the side the model plays. Its decisions' readout is never
   * shown -- its options, probabilities and critic are read from its own hand -- whatever
   * `readoutShown` says. The model still reads every position: auto-play plays `favourite()` for
   * its side.
   */
  private hiddenSide: "US" | "USSR" | null = null;
  /**
   * The readout -- the critic's win chances, the policy, the search -- and its badges on the board
   * are on screen (Show analysis in the technical bar, main.ts). They would be hints to a player:
   * the Developer view turns them on, a player's view off, and the toggle overrides either.
   */
  private readoutShown = true;
  /** No local server to list checkpoints (a static host): the local source is disabled. */
  private localUnavailable = false;
  /** The Hugging Face file select holds only its "listing…" placeholder. */
  private hfListing = false;
  /** The position of the last `render`, for a redraw in another language. */
  private lastState: GameState | null = null;

  private sourceSelect = el<HTMLSelectElement>("analysis-source-select");
  private runSelect = el<HTMLSelectElement>("analysis-run-select");
  private snapSelect = el<HTMLSelectElement>("analysis-snapshot-select");
  private hfRepo = el<HTMLInputElement>("analysis-hf-repo");
  private hfRevision = el<HTMLInputElement>("analysis-hf-revision");
  private hfFile = el<HTMLSelectElement>("analysis-hf-file");
  private hfClear = el<HTMLButtonElement>("analysis-hf-clear");
  private fileInput = el<HTMLInputElement>("analysis-file-input");
  private favButton = el<HTMLButtonElement>("btn-play-favourite");
  private autoSelect = el<HTMLSelectElement>("analysis-autoplay-select");
  private body = el<HTMLElement>("analysis-body");
  private badge = el<HTMLElement>("analysis-status");
  private info = el<HTMLElement>("analysis-model-info");
  /** The model section's one-line state of the model: off, loading, error -- or nothing, when
   *  `info` names the loaded model beside it. */
  private activeStatus = el<HTMLElement>("model-active-status");
  /** The model in a few words, for the technical bar's model · mode label (main.ts). */
  public onModelLabel: (text: string, status: "off" | "loading" | "ready" | "error") => void = () => {};

  constructor(
    private onPick: (pick: ModelPick | null) => void,
    private onPlayFlat: (flatIdx: number) => void,
    private onAutoSideChange: (side: AutoSide) => void = () => {},
  ) {
    this.autoSelect.addEventListener("change", () => {
      this.setAutoSide(this.autoSelect.value as AutoSide);
      this.onAutoSideChange(this.auto);
    });
    this.sourceSelect.addEventListener("change", () => {
      // Switching source loads nothing by itself: a checkpoint's first use is an export, and a
      // model the user did not pick should not cost that or replace the one on screen.
      this.showSourceControls();
      if (this.sourceSelect.value === "") this.pick(null);
    });
    this.runSelect.addEventListener("change", () => {
      this.fillSnapshots(this.runSelect.value, null);
      this.pickLocal();
    });
    this.snapSelect.addEventListener("change", () => this.pickLocal());
    el<HTMLButtonElement>("analysis-hf-list").addEventListener("click", () => {
      this.listHf().catch(e => this.setError(e instanceof Error ? e.message : String(e)));
    });
    this.hfFile.addEventListener("change", () => {
      if (this.hfFile.value) {
        this.pick({ source: { kind: "hf", repo: this.hfRepo.value.trim(), revision: this.hfRevision.value.trim() || "main", path: this.hfFile.value } });
      }
    });
    this.hfClear.addEventListener("click", async () => {
      await clearModelCache();
      this.refreshCacheButton();
    });
    this.fileInput.addEventListener("change", async () => {
      const f = this.fileInput.files?.[0];
      if (f) this.pick({ source: { kind: "file", name: f.name }, bytes: new Uint8Array(await f.arrayBuffer()) });
    });
    this.favButton.addEventListener("click", () => this.playFavourite());
    this.body.addEventListener("click", (ev) => {
      const row = (ev.target as HTMLElement).closest<HTMLElement>("[data-flat-idx]");
      if (row) this.onPlayFlat(parseInt(row.getAttribute("data-flat-idx")!, 10));
    });
    if (!this.hfRepo.value) this.hfRepo.value = DEFAULT_HF_REPO;
    this.showSourceControls();
    this.loadLocalModels();
    this.refreshCacheButton();
    onLanguageChange(() => this.relabel());
  }

  /**
   * The panel in another language: the pickers' own labels (filled once, from the lists), the
   * cache button, the auto-play lock, and the body. The values selected stay as they are.
   */
  private relabel(): void {
    if (this.localUnavailable) this.markLocalUnavailable();
    for (const o of this.runSelect.options) {
      if (o.value === "") o.textContent = t("analysis:local.chooseRun");
      else if (o.value === LOOSE_GROUP) o.textContent = t("analysis:local.looseGroup");
      else if (o.dataset.unlisted !== undefined) o.textContent = t("analysis:local.unlisted", { run: o.value });
    }
    for (const o of this.snapSelect.options) o.textContent = snapshotLabel(o.value);
    const placeholder = this.hfFile.querySelector<HTMLOptionElement>('option[value=""]');
    if (placeholder) placeholder.textContent = this.hfListing ? t("analysis:hf.listing") : t("analysis:hf.chooseModel");
    this.refreshCacheButton();
    if (this.hiddenSide) this.autoSelect.title = t("analysis:autoLocked", { side: sideName(this.hiddenSide) });
    this.renderBody(this.lastState);
  }

  /** The Hugging Face models this browser keeps (analysis/model_cache.ts), on the clear button. */
  private async refreshCacheButton(): Promise<void> {
    const n = await cachedModelCount().catch(() => 0);
    this.hfClear.disabled = n === 0;
    this.hfClear.title = n === 0 ? t("analysis:cache.none") : t("analysis:cache.delete", { count: n });
  }

  /** A model file dropped anywhere on the page. */
  public async dropFile(f: File): Promise<void> {
    this.sourceSelect.value = "file";
    this.showSourceControls();
    this.pick({ source: { kind: "file", name: f.name }, bytes: new Uint8Array(await f.arrayBuffer()) });
  }

  private pick(p: ModelPick | null): void {
    this.lastAnalysis = null;
    this.error = null;
    this.onPick(p);
  }

  private pickLocal(): void {
    const run = this.runSelect.value;
    const snap = this.snapSelect.value;
    if (!run || !snap) return;
    this.pick({ source: { kind: "local", path: run === LOOSE_GROUP ? snap : `${run}/${snap}` } });
  }

  private showSourceControls(): void {
    const kind = this.sourceSelect.value;
    for (const k of ["local", "hf", "file"]) {
      el<HTMLElement>(`analysis-src-${k}`).classList.toggle("hidden", kind !== k);
    }
  }

  // ---- what the app tells the panel ------------------------------------------------------------

  /** Show a source as selected (from the URL) without loading anything. */
  public showSource(s: ModelSource | null): void {
    if (!s) {
      this.sourceSelect.value = "";
    } else if (s.kind === "local") {
      this.sourceSelect.value = "local";
      this.selectLocal(s.path);
    } else if (s.kind === "hf") {
      this.sourceSelect.value = "hf";
      this.hfRepo.value = s.repo;
      this.hfRevision.value = s.revision;
      this.hfFile.innerHTML = `<option value="${esc(s.path)}">${esc(s.path)}</option>`;
      this.hfFile.value = s.path;
    } else {
      this.sourceSelect.value = "file";
    }
    this.showSourceControls();
  }

  public setLoading(s: ModelSource): void {
    this.status = "loading";
    this.pending = s;
    this.loaded = null;
    this.error = null;
    this.lastAnalysis = null;
    this.renderBody();
  }

  public setLoaded(info: LoadedInfo): void {
    this.status = "ready";
    this.loaded = info;
    this.pending = null;
    this.error = null;
    this.renderBody();
    this.refreshCacheButton();   // a download may just have been kept
  }

  public setOff(): void {
    this.status = "off";
    this.loaded = null;
    this.pending = null;
    this.error = null;
    this.lastAnalysis = null;
    this.renderBody();
  }

  public setError(message: string): void {
    this.status = "error";
    this.error = message;
    this.loaded = null;
    this.lastAnalysis = null;
    this.renderBody();
  }

  /**
   * The move the model would make in the position now on screen: the search's pick when the
   * search select asked for one and it has answered, else the policy's argmax. Null with nothing
   * to play -- and while a requested search is still running, so auto-play waits for it.
   */
  public favourite(): number | null {
    const a = this.lastAnalysis;
    if (!a) return null;
    if (a.search) return a.search.pick;
    if (a.search_pending) return null;
    const idx = a.policy?.argmax_idx;
    return idx === undefined ? null : idx;
  }

  public playFavourite(): void {
    if (!this.readoutOnScreen()) return;   // a hint with the readout off, or the model's own move
    const idx = this.favourite();
    if (idx !== null) this.onPlayFlat(idx);
  }

  /**
   * Set the player's view: `side` is the model's side (auto-play is locked to it), or null for
   * the developer's view, which unlocks auto-play. Whether the readout shows is `readoutShown`.
   */
  public setHiddenSide(side: "US" | "USSR" | null): void {
    this.hiddenSide = side;
    if (side) this.setAutoSide(side);
    this.autoSelect.disabled = side !== null;
    this.autoSelect.title = side ? t("analysis:autoLocked", { side: sideName(side) }) : "";
    this.renderBody();
  }

  /** The analysis on screen is the hidden side's decision. */
  private hidingDecision(): boolean {
    return !!this.hiddenSide && this.lastAnalysis?.decision_player === this.hiddenSide;
  }

  /** Show the readout and its badges, or keep them off (Show analysis). */
  public setReadoutShown(on: boolean): void {
    this.readoutShown = on;
    this.renderBody();
  }

  /** A decision by `side` would have its readout on screen. */
  public showsReadoutFor(side: "US" | "USSR"): boolean {
    return this.readoutShown && side !== this.hiddenSide;
  }

  /** The readout of the analysis on screen is shown. */
  private readoutOnScreen(): boolean {
    return this.readoutShown && !this.hidingDecision();
  }

  /** The side that plays the model's favourite by itself ("" = nobody). */
  public get autoSide(): AutoSide {
    return this.auto;
  }

  /** Set the auto-play side programmatically (from the URL); does not notify. */
  public setAutoSide(side: AutoSide): void {
    this.auto = side === "US" || side === "USSR" ? side : "";
    this.autoSelect.value = this.auto;
    this.autoSelect.closest(".analysis-autoplay")?.classList.toggle("active", this.auto !== "");
    this.renderBody();
  }

  /**
   * The move auto-play should make in the position on screen: the favourite, when a decision
   * is open and it belongs to the auto-play side. Null otherwise -- including when no model is
   * loaded, since there is then nothing to play.
   */
  public autoPlayMove(): number | null {
    const a = this.lastAnalysis;
    if (!this.auto || !a || a.decision_player !== this.auto) return null;
    return this.favourite();
  }

  public show(visible: boolean): void {
    document.getElementById("analysis-panel")?.classList.toggle("hidden", !visible);
  }

  // ---- sources -----------------------------------------------------------------------------------

  private async loadLocalModels(): Promise<void> {
    try {
      const res = await fetch("/api/local/models");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.models = await res.json();
    } catch {
      // A static host (GitHub Pages): there is no local server to list checkpoints.
      this.models = null;
      this.localUnavailable = true;
      this.markLocalUnavailable();
      return;
    }
    const opts: string[] = [`<option value="">${t("analysis:local.chooseRun")}</option>`];
    for (const r of this.models!.runs) opts.push(`<option value="${esc(r.run)}">${esc(r.run)}</option>`);
    if (this.models!.loose.length) opts.push(`<option value="${LOOSE_GROUP}">${t("analysis:local.looseGroup")}</option>`);
    const keep = this.runSelect.value;
    this.runSelect.innerHTML = opts.join("");
    if (keep) this.selectLocal(this.currentLocalPath(keep));
    else this.fillSnapshots(this.runSelect.value, null);
  }

  /** The local source's option, disabled and saying why. Its text is the code's from here on, so
   *  the markup's message key (index.html) is dropped. */
  private markLocalUnavailable(): void {
    const option = el<HTMLOptionElement>("analysis-source-local-option");
    option.disabled = true;
    delete option.dataset.i18n;
    option.textContent = t("analysis:local.unavailable");
  }

  private currentLocalPath(run: string): string {
    const snap = this.snapSelect.value;
    return run === LOOSE_GROUP ? snap : `${run}/${snap}`;
  }

  private selectLocal(path: string): void {
    const slash = path.indexOf("/");
    const run = slash >= 0 ? path.slice(0, slash) : LOOSE_GROUP;
    const snap = slash >= 0 ? path.slice(slash + 1) : path;
    if (![...this.runSelect.options].some(o => o.value === run)) {
      // Named by a link but not listed here (another machine's checkpoint tree): keep it
      // visible rather than silently showing a different model.
      this.runSelect.insertAdjacentHTML("beforeend",
        `<option value="${esc(run)}" data-unlisted>${t("analysis:local.unlisted", { run: esc(run) })}</option>`);
    }
    this.runSelect.value = run;
    this.fillSnapshots(run, snap);
    if (this.snapSelect.value !== snap) {
      this.snapSelect.insertAdjacentHTML("beforeend", `<option value="${esc(snap)}">${esc(snapshotLabel(snap))}</option>`);
      this.snapSelect.value = snap;
    }
  }

  private fillSnapshots(run: string, want: string | null): void {
    let snaps: string[] = [];
    if (run === LOOSE_GROUP) snaps = this.models?.loose ?? [];
    else snaps = this.models?.runs.find(r => r.run === run)?.snapshots ?? [];
    this.snapSelect.innerHTML = snaps.map(s => `<option value="${esc(s)}">${esc(snapshotLabel(s))}</option>`).join("");
    this.snapSelect.disabled = snaps.length === 0;
    // Default to the run's latest weights: the list is ordered by training step, final last.
    const pick = want && snaps.includes(want) ? want : snaps[snaps.length - 1];
    if (pick) this.snapSelect.value = pick;
  }

  /** List the Hugging Face repo in the inputs into the file select: one group per directory (a run
   *  published whole), newest upload first. */
  private async listHf(read?: () => Promise<HfCatalog>): Promise<HfCatalog> {
    const repo = this.hfRepo.value.trim();
    const rev = this.hfRevision.value.trim() || "main";
    this.hfFile.innerHTML = `<option value="">${t("analysis:hf.listing")}</option>`;
    this.hfListing = true;
    try {
      const catalog = await (read ? read() : hfCatalog(repo, rev));
      const files = catalog.files;
      if (files.length === 0) throw new Error("no .onnx files in the repo (export them with tools/export_onnx.py)");
      const option = (f: HfModelFile, text: string) =>
        `<option value="${esc(f.path)}">${esc(text)}${f.date ? ` · ${esc(f.date.slice(0, 10))}` : ""}</option>`;
      this.hfFile.innerHTML = `<option value="">${t("analysis:hf.chooseModel")}</option>`
        + groupHfModels(files).map(g => g.dir
          ? `<optgroup label="${esc(g.dir)}">${g.files.map(f => option(f, f.path.slice(g.dir.length + 1))).join("")}</optgroup>`
          : g.files.map(f => option(f, f.path)).join("")).join("");
      return catalog;
    } catch (e) {
      this.hfFile.innerHTML = "";
      throw new Error(t("analysis:hf.listFailed", { repo, error: e instanceof Error ? e.message : String(e) }));
    } finally {
      this.hfListing = false;
    }
  }

  /**
   * The model a link that names none gets, shown as selected: the one the default repo names
   * (models.json, or default.json before it), or its newest upload when it names none. Throws if
   * the repo cannot be listed or names a file it does not hold; the caller decides whether that
   * is still worth showing, since the user may have picked a model while it was listing.
   */
  public async defaultHf(): Promise<ModelSource> {
    this.sourceSelect.value = "hf";
    this.hfRepo.value = DEFAULT_HF_REPO;
    this.hfRevision.value = DEFAULT_HF_REVISION;
    this.showSourceControls();
    // The page's one read of the default repo, shared with the default search (main.ts).
    const { files, defaultModel: named } = await this.listHf(defaultHfCatalog);
    if (named !== null && !files.some(f => f.path === named)) {
      throw new Error(`${DEFAULT_HF_REPO}'s default model ${named} is not in the repo`);
    }
    const path = named ?? files[0].path;
    this.hfFile.value = path;
    return { kind: "hf", repo: DEFAULT_HF_REPO, revision: DEFAULT_HF_REVISION, path };
  }

  // ---- rendering ----------------------------------------------------------------------------------

  /** Redraw for the position on screen. `analysis` is undefined until it arrives. */
  public render(analysis: LiveAnalysis | undefined, state: GameState | null): void {
    this.lastAnalysis = analysis && this.loaded && analysis.model === this.loaded.key ? analysis : null;
    this.lastState = state;
    this.renderBody(state);
  }

  private renderInfo(): void {
    if (this.status === "ready" && this.loaded) {
      const l = this.loaded;
      this.info.innerHTML = `<b>${esc(l.label)}</b> <span class="analysis-info-dim">${l.merged ? t("analysis:view.e41") : t("analysis:view.e4")}${l.checkpoint ? ` · ${esc(l.checkpoint)}` : ""}${l.fromCache ? ` · ${t("analysis:info.cached")}` : ""}</span>`
        + (l.engineWarning ? `<span class="analysis-warning"> · ${esc(l.engineWarning)}</span>` : "");
      this.info.classList.remove("hidden");
      this.activeStatus.textContent = "";
      this.activeStatus.className = "model-active-status hidden";
      this.onModelLabel(l.label, this.status);
      return;
    }
    this.info.innerHTML = "";
    this.info.classList.add("hidden");
    const what = this.pending ? sourceLabel(this.pending) : this.loaded?.label ?? "";
    const text = this.status === "loading" ? t("analysis:status.loading", { what })
      : this.status === "error" ? t("analysis:status.failed")
      : t("analysis:status.none");
    this.activeStatus.textContent = text;
    this.activeStatus.className = `model-active-status ${this.status}`;
    this.activeStatus.title = this.status === "error" ? this.error ?? "" : "";
    this.onModelLabel(text, this.status);
  }

  private renderBody(state: GameState | null = null): void {
    const a = this.lastAnalysis;
    this.renderInfo();
    this.favButton.disabled = !this.readoutOnScreen() || this.favourite() === null;
    const fav = a?.choices.find(c => c.idx === a.policy?.argmax_idx);
    this.favButton.title = this.hidingDecision() ? t("analysis:favourite.hiddenSide", { side: sideName(this.hiddenSide!) })
      : !this.readoutShown ? t("analysis:favourite.readoutHidden")
      : fav ? t("analysis:favourite.play", { move: fav.name ?? "#" + fav.idx }) : t("analysis:favourite.none");

    if (this.status === "error") {
      this.badge.textContent = t("analysis:badge.error");
      this.body.innerHTML = `<div class="analysis-error">${esc(this.error ?? "")}</div>`;
      return;
    }
    if (this.status === "off") {
      this.badge.textContent = t("analysis:badge.off");
      this.body.innerHTML = `<div class="trace-empty">${t("analysis:body.choose")}</div>`
        + (this.auto ? `<div class="analysis-error">${t("analysis:body.autoNeedsModel", { side: sideName(this.auto) })}</div>` : "");
      return;
    }
    if (this.status === "loading" || !a) {
      this.badge.textContent = t("analysis:badge.loading");
      const what = esc(this.pending ? sourceLabel(this.pending) : this.loaded?.label ?? "");
      this.body.innerHTML = `<div class="trace-empty">${this.pending?.kind === "local"
        ? t("analysis:body.loadingLocal", { what }) : t("analysis:body.loading", { what })}</div>`;
      return;
    }
    this.badge.textContent = a.merged_influence ? t("analysis:view.e41") : a.macro_influence ? t("analysis:view.macro") : t("analysis:view.e4");
    if (!this.readoutOnScreen()) {
      this.body.innerHTML = this.readoutOffNote(a);
      return;
    }

    const parts: string[] = [];
    if (a.critic) {
      // The readout is on the position itself (`at: "position"`), so the side to move is the one
      // whose reading the value head was trained on. A finished game has none.
      const decider = state?.is_terminal ? null : a.decision_player ?? null;
      parts.push(valueSidesHtml(a.critic, decider, state?.is_terminal ? state : null));
    }

    // The search's decision, above the policy it searched from.
    if (a.search || a.search_pending || a.search_error) {
      const s = a.search;
      const spec = s ? t("analysis:search.spec", { k: s.spec.k, worlds: s.spec.worlds, horizon: s.spec.horizon, rule: s.spec.rule }) : "";
      parts.push(`<div class="trace-section-label">${t("analysis:search.label")}${spec ? ` <span class="analysis-info-dim">${esc(spec)}</span>` : ""}</div>`);
      if (a.search_error) {
        parts.push(`<div class="analysis-error">${esc(a.search_error)}</div>`);
      } else if (!s) {
        parts.push(`<div class="trace-empty">${t("analysis:search.searching")}</div>`);
      } else {
        const name = (idx: number) => esc(a.choices.find(c => c.idx === idx)?.name ?? "#" + idx);
        const rows = s.candidates.map((idx, i) => {
          const picked = idx === s.pick;
          const own = i === 0;
          const lead = own ? "" : `${s.lead[i] >= 0 ? "+" : ""}${(s.lead[i] * 100).toFixed(1)} ± ${(s.lead_se[i] * 100).toFixed(1)}`;
          return `
            <div class="analysis-choice analysis-search-row${picked ? " favourite" : ""}" data-flat-idx="${idx}" title="${own ? t("analysis:search.ownTitle") : t("analysis:search.leadTitle")}">
              <span class="analysis-choice-name">${picked ? "★ " : ""}${name(idx)}${own ? ` <span class="analysis-info-dim">${t("analysis:search.policyTag")}</span>` : ""}</span>
              <span class="analysis-search-q" title="${t("analysis:search.qTitle")}">${s.q[i] >= 0 ? "+" : ""}${s.q[i].toFixed(3)}</span>
              <span class="analysis-search-lead">${lead}</span>
            </div>`;
        });
        parts.push(`<div class="analysis-choices">${rows.join("")}</div>`);
        parts.push(`<div class="trace-note trace-hint">${t(s.pick === s.candidates[0] ? "analysis:search.summaryStays" : "analysis:search.summaryDeparts",
          { worlds: s.worlds, rows: s.rows, ms: s.ms })}</div>`);
      }
    }

    const pol = a.policy;
    if (!pol || a.choices.length === 0) {
      parts.push(`<div class="trace-section-label">${t("analysis:next.label")}</div>`);
      parts.push(`<div class="trace-empty">${state?.is_terminal ? t("analysis:next.gameOver") : t("analysis:next.noDecision")}</div>`);
    } else {
      const who = a.decision_player ?? "";
      const autoTag = this.auto && this.auto === who ? `<span class="analysis-auto-tag">${t("analysis:next.autoPlaying")}</span>` : "";
      parts.push(`<div class="trace-section-label">${t("analysis:next.toChoose", {
        who: `<span class="analysis-who ${who.toLowerCase()}">${who ? sideName(who) : ""}</span>`, auto: autoTag })}</div>`);
      parts.push(`
        <div class="trace-head">
          <span class="trace-head-item" title="${t("analysis:next.bestTitle")}">${t("analysis:next.best", {
            p: `<b style="color:${probColor(pol.p_max ?? 0)}">${(pol.p_max ?? 0).toFixed(3)}</b>` })}</span>
          ${pol.entropy !== undefined ? `<span class="trace-head-item" title="${t("analysis:next.entropyTitle")}">H ${pol.entropy.toFixed(3)}</span>` : ""}
          <span class="trace-head-item">${t("analysis:next.legal", { n: pol.n_legal ?? a.choices.length })}</span>
        </div>`);
      const rows = a.choices.slice(0, TOP_LISTED).map(c => {
        const isFav = c.idx === pol.argmax_idx;
        const pct = Math.max(0, Math.min(100, c.p * 100));
        return `
          <div class="analysis-choice${isFav ? " favourite" : ""}" data-flat-idx="${c.idx}" title="${c.composed ? t("analysis:next.playComposed") : t("analysis:next.play")}">
            <span class="analysis-choice-bar" style="width:${pct.toFixed(1)}%;background:${probColor(c.p)}"></span>
            <span class="analysis-choice-name">${isFav ? "★ " : ""}${esc(c.name ?? "#" + c.idx)}</span>
            <span class="analysis-choice-p" style="color:${probColor(c.p)}">${fmtP(c.p)}</span>
          </div>`;
      });
      parts.push(`<div class="analysis-choices">${rows.join("")}</div>`);
      if (a.choices.length > TOP_LISTED) {
        parts.push(`<div class="trace-note trace-hint">${t("analysis:next.more", { n: a.choices.length - TOP_LISTED })}</div>`);
      }
    }
    this.body.innerHTML = parts.join("");
  }

  /**
   * What the panel says with the readout off. On the model's own move, that it is choosing -- and
   * whether its search is running or has failed, since a failed search leaves it playing the
   * policy's move rather than the one asked for. Otherwise, how to turn the readout on.
   */
  private readoutOffNote(a: LiveAnalysis): string {
    const who = a.decision_player;
    if (who && who === this.auto) {
      const side = esc(sideName(who));
      const failed = a.search_error
        ? `<div class="analysis-error search-failed">${t("analysis:off.searchFailed", { error: esc(a.search_error) })}</div>`
        : "";
      return `<div class="trace-empty">${a.search_pending ? t("analysis:off.choosingSearching", { side })
        : t("analysis:off.choosing", { side })}</div>${failed}`;
    }
    const me = this.hiddenSide === "US" ? "USSR" : "US";
    return `<div class="trace-empty readout-off-note">${this.hiddenSide
      ? t("analysis:off.hiddenPlayer", { side: sideName(me) }) : t("analysis:off.hidden")}</div>`;
  }

  /**
   * Paint every legal action's probability on the thing you would click to take it.
   * Must run after the HUD, hands and map have rendered the same `state`.
   */
  public decorate(state: GameState | null): void {
    clearDecorations();
    if (!this.readoutOnScreen()) return;   // no hints with the readout off, nor the model's own options
    const a = this.lastAnalysis;
    if (!a || !a.policy || !state || a.choices.length === 0) return;
    const dType = state.decision_context?.decision_type;
    if (dType === undefined || a.decision_type !== dType) return;
    const favIdx = this.favourite() ?? a.policy.argmax_idx;
    const DT = engineLayout()?.decision_types;
    if (!DT) return;

    type Cell = { p: number; fav: boolean; note?: string };
    const add = (m: Map<string | number, Cell>, k: string | number, p: number, fav: boolean, note?: string) => {
      const cur = m.get(k);
      m.set(k, { p: (cur?.p ?? 0) + p, fav: (cur?.fav ?? false) || fav, note: cur?.note ?? note });
    };
    const buttons = new Map<string | number, Cell>();
    const cards = new Map<string | number, Cell>();
    const countries = new Map<string | number, Cell>();

    for (const c of a.choices) {
      const fav = c.idx === favIdx;
      const key = clickKey(c.primary_id ?? -1, c.flags ?? 0);
      if (c.composed) {
        add(buttons, key, c.p, fav, t("analysis:badge.composedSum"));
        if (c.country_id !== undefined) countries.set(c.country_id, { p: c.p, fav, note: t("analysis:badge.composedHere") });
        continue;
      }
      add(buttons, key, c.p, fav);
      if (dType === DT.SELECT_CARD && !(c.flags! & FLAG_CONFIRM_DONE)) cards.set(c.primary_id!, { p: c.p, fav });
      if (dType === DT.POINT_NODE && !(c.flags! & FLAG_CONFIRM_DONE)) countries.set(c.primary_id!, { p: c.p, fav });
    }
    const mark = (cell: Cell): BadgeMark => (cell.fav ? "favourite" : null);

    document.querySelectorAll<HTMLElement>("#decision-body [data-primary]").forEach(btn => {
      const primary = parseInt(btn.getAttribute("data-primary") || "", 10);
      if (Number.isNaN(primary)) return;
      const flags = parseInt(btn.getAttribute("data-flags") || "0", 10) || 0;
      const cell = buttons.get(clickKey(primary, flags));
      if (!cell) return;
      if (cell.fav) btn.classList.add("trace-choice-played");
      btn.appendChild(htmlBadge(cell.p, mark(cell), cell.note));
    });

    if (cards.size) {
      document.querySelectorAll<HTMLElement>(".card-item[data-card-id]").forEach(el => {
        const cell = cards.get(parseInt(el.getAttribute("data-card-id") || "", 10));
        if (!cell) return;
        if (cell.fav) el.classList.add("trace-choice-played");
        el.appendChild(htmlBadge(cell.p, mark(cell)));
      });
    }

    if (countries.size) {
      document.querySelectorAll(".svg-country-node[data-id]").forEach(g => {
        const cell = countries.get(parseInt(g.getAttribute("data-id") || "", 10));
        if (!cell) return;
        if (cell.fav) g.classList.add("trace-choice-played");
        svgBadge(g, cell.p, cell.fav, cell.note);
      });
    }
  }
}
