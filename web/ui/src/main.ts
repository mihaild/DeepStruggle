/**
 * The workbench: watch replays, test the engine by playing it, and play with a model.
 *
 * Everything runs in the page. The engine is the WebAssembly build of engine/ + bindings/
 * (engine/wasm_engine.ts), the game -- stepping, dice, undo, the action log -- is
 * game/session.ts, and models run as ONNX in onnxruntime-web (analysis/). A server is optional:
 * the local one (web/server/main.py) only lists this machine's checkpoints and replays, and the
 * same page works from GitHub Pages with models from Hugging Face or a dropped file.
 */
import { GameState, MicroAction } from "./types";
import { MapView } from "./map_view";
import { TracksView } from "./tracks_view";
import { CardsView } from "./cards_view";
import { ActionHud } from "./action_hud";
import { ReplayControls, ReplayStep } from "./replay_controls";
import { DebugPanel } from "./debug_panel";
import { decorateChoices, policyChipHtml, renderTracePanel, renderValueRibbon, setActionSpace } from "./trace_view";
import { AnalysisPanel, AutoSide, LiveAnalysis, ModelPick } from "./analysis_view";
import { WasmEngine } from "./engine/wasm_engine";
import { GameSession } from "./game/session";
import { decodePosition, encodePosition } from "./game/position";
import { Model, sourceFromParam, sourceToParam } from "./analysis/model";
import { analyze } from "./analysis/readout";

/** Pause before an auto-played move, long enough to see each one land. */
const AUTO_PLAY_DELAY_MS = 350;

export class TSApp {
  private state: GameState | null = null;
  private isReplayMode: boolean = false;
  private replaySteps: ReplayStep[] = [];
  private replayCurrentStep: number = 0;
  private isVpFilterActive: boolean = false;

  private mapView: MapView;
  private tracksView: TracksView;
  private cardsView: CardsView;
  private actionHud: ActionHud;
  public replayControls: ReplayControls;
  private debugPanel: DebugPanel;
  private analysisPanel: AnalysisPanel;

  private engine: WasmEngine | null = null;
  private session: GameSession | null = null;
  private model: Model | null = null;
  private modelKey = "";
  private modelLoads = 0;
  /**
   * Analysis was turned off on purpose, by the user or by the link. The link then says
   * `model=off`, so that opening it does not load the default model instead.
   */
  private analysisOff = false;

  /** The live position, its readout and its link token, kept while a replay is on screen. */
  private liveState: GameState | null = null;
  private liveAnalysis: LiveAnalysis | undefined = undefined;
  private positionToken: string | null = null;
  /** Bumped on every change of position; async work for an older one is dropped. */
  private version = 0;

  private urlPosition: string | null = null;
  private urlModel: string | null = null;
  private urlAuto: AutoSide = "";
  private autoPlayTimer: number | null = null;

  constructor() {
    this.parseQueryParams();

    const tracksContainer = document.getElementById("global-tracks")!;
    const mapSvg = document.getElementById("ts-map-svg") as unknown as SVGSVGElement;

    this.tracksView = new TracksView(tracksContainer);
    this.mapView = new MapView(mapSvg, (countryId: number) => this.handleCountryClick(countryId));
    this.cardsView = new CardsView((cardId: number) => this.handleCardClick(cardId));
    this.actionHud = new ActionHud((action: MicroAction) => this.sendAction(action));
    this.debugPanel = new DebugPanel((override: any) => this.sendDebugOverride(override));

    this.replayControls = new ReplayControls((replayState: GameState, stepIndex: number, allSteps: ReplayStep[]) => {
      this.setReplayMode(true);
      this.state = replayState;
      this.replaySteps = allSteps;
      this.replayCurrentStep = stepIndex;
      this.renderState();
      this.renderTrace();
    });
    this.replayControls.liveReplay = () => this.session?.replay() ?? null;

    this.analysisPanel = new AnalysisPanel(
      (pick: ModelPick | null) => this.loadModel(pick),
      (flatIdx: number) => this.sendFlatAction(flatIdx),
      () => {
        this.syncUrl();
        this.maybeAutoPlay();
      },
    );
    this.analysisPanel.setAutoSide(this.urlAuto);
    this.analysisPanel.show(!this.isReplayMode);
    this.actionHud.onRerender = () => this.redecorate();

    this.setupGlobalControls();
    this.setupBottomResizer();
    this.setupFileDrop();
    this.boot();
  }

  private badge(text: string, cls: string, title = ""): void {
    const el = document.getElementById("connection-status")!;
    el.textContent = text;
    el.className = `status-badge ${cls}`;
    el.title = title;
  }

  /**
   * Load the engine, start a game -- or open the position a link names -- and the link's model.
   */
  private async boot() {
    try {
      this.engine = await WasmEngine.load();
    } catch (e) {
      this.badge("NO ENGINE", "disconnected", String(e));
      window.alert(String(e));
      return;
    }
    setActionSpace(this.engine.layout);
    this.session = new GameSession(this.engine);
    this.session.newGame(Math.floor(Math.random() * 1_000_000));
    if (this.urlPosition) {
      try {
        this.session.loadPosition(await decodePosition(this.urlPosition), this.urlPosition);
      } catch (e) {
        window.alert(`Could not open the position in the link -- starting a new game instead.\n\n${e}`);
      }
    }
    this.checkEngineFreshness();
    this.refresh();
    const src = this.urlModel && this.urlModel !== "off" ? sourceFromParam(this.urlModel) : null;
    if (this.urlModel === "off") {
      this.analysisOff = true;
      this.analysisPanel.showSource(null);
    } else if (src && src.kind !== "file") {
      this.analysisPanel.showSource(src);
      this.loadModel({ source: src });
    } else if (!this.isReplayMode) {
      this.loadDefaultModel();
    }
  }

  /**
   * A link that names no model gets the newest upload in the default Hugging Face repo. Dropped
   * -- result and error alike -- if a model was picked while the repo was being listed.
   */
  private async loadDefaultModel() {
    const before = this.modelLoads;
    try {
      const source = await this.analysisPanel.newestDefaultHf();
      if (this.modelLoads === before) this.loadModel({ source });
    } catch (e) {
      if (this.modelLoads === before) this.analysisPanel.setError(e instanceof Error ? e.message : String(e));
    }
  }

  /** Against the local server's sources: a page built from other sources plays other rules. */
  private async checkEngineFreshness() {
    const fp = this.engine!.fingerprint;
    try {
      const res = await fetch("/api/local/info");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const info = await res.json();
      if (info.engine_fingerprint !== fp) {
        this.badge("ENGINE STALE", "stale",
          `The page's engine (${fp.slice(0, 12)}) is not what the sources here build (${String(info.engine_fingerprint).slice(0, 12)}). `
          + "Rebuild it with tools/scripts/build_web.sh.");
        return;
      }
      this.badge(`ENGINE ${fp.slice(0, 8)}`, "connected", `engine fingerprint ${fp} -- matches the local sources`);
    } catch {
      this.badge(`ENGINE ${fp.slice(0, 8)}`, "connected", `engine fingerprint ${fp} (no local server to compare with)`);
    }
  }

  private parseQueryParams() {
    const params = new URLSearchParams(window.location.search);
    if (params.has("replay")) {
      this.isReplayMode = true;
    }
    this.urlPosition = params.get("pos");
    this.urlModel = params.get("model");
    const auto = (params.get("auto") || "").toUpperCase();
    this.urlAuto = auto === "US" || auto === "USSR" ? auto : "";
  }

  /**
   * Keep the address bar naming the board on screen -- the position itself, the model and the
   * auto-play side -- so it can be copied and shared at any moment. `replaceState`, never
   * `pushState`: every move would otherwise add a history entry, and Back would walk the game
   * move by move instead of leaving the page. A dropped model file has no address and is left out.
   */
  private syncUrl() {
    if (this.isReplayMode || !this.positionToken) return;
    const params = new URLSearchParams(window.location.search);
    params.delete("game_id");
    params.delete("role");
    const model = this.model ? sourceToParam(this.model.source) : null;
    if (model) params.set("model", model);
    else if (this.analysisOff) params.set("model", "off");
    else params.delete("model");
    const auto = this.analysisPanel.autoSide;
    if (auto) params.set("auto", auto.toLowerCase()); else params.delete("auto");
    params.set("pos", this.positionToken);
    const next = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
    if (next !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.history.replaceState(window.history.state, "", next);
    }
  }

  /**
   * The position changed (a move, an undo, a new game, a loaded link): redraw it, then -- off the
   * critical path -- its link token, the model's readout, and auto-play.
   */
  private refresh() {
    if (!this.session) return;
    const version = ++this.version;
    this.liveState = this.session.state();
    this.liveAnalysis = undefined;
    if (!this.isReplayMode) {
      this.state = this.liveState;
      this.renderState();
    }
    encodePosition(this.engine!.saveJson()).then(token => {
      if (version !== this.version) return;
      this.positionToken = token;
      this.liveState!.position = token;
      this.syncUrl();
    });
    this.runAnalysis(version);
  }

  private async runAnalysis(version: number) {
    const model = this.model;
    if (!model || !this.engine) {
      this.maybeAutoPlay();
      return;
    }
    const key = this.modelKey;
    try {
      const a = await analyze(this.engine, model, key);
      if (version !== this.version || key !== this.modelKey) return;   // the board moved on
      this.liveAnalysis = a;
      if (!this.isReplayMode) {
        this.analysisPanel.render(a, this.state);
        this.analysisPanel.decorate(this.state);
      }
      this.maybeAutoPlay();
    } catch (e) {
      if (key === this.modelKey) this.analysisPanel.setError(`The model failed on this position: ${e}`);
    }
  }

  private async loadModel(pick: ModelPick | null) {
    const load = ++this.modelLoads;
    this.analysisOff = !pick;
    this.liveAnalysis = undefined;
    if (!pick) {
      this.model = null;
      this.modelKey = "";
      this.analysisPanel.setOff();
      this.renderState();
      this.syncUrl();
      return;
    }
    this.analysisPanel.setLoading(pick.source);
    try {
      const m = await Model.fetch(pick.source, this.engine!.obsSize, this.engine!.actionSize, pick.bytes);
      if (load !== this.modelLoads) return;   // another pick superseded this one
      this.model = m;
      this.modelKey = `m${load}`;
      const fp = m.meta.engineFingerprint;
      this.analysisPanel.setLoaded({
        key: this.modelKey,
        label: m.meta.label,
        merged: m.meta.mergedInfluence,
        checkpoint: m.meta.checkpoint,
        engineWarning: fp && fp !== this.engine!.fingerprint
          ? `Exported next to another engine build (${fp.slice(0, 8)}; this page runs ${this.engine!.fingerprint.slice(0, 8)}).`
          : undefined,
      });
    } catch (e) {
      if (load !== this.modelLoads) return;
      this.model = null;
      this.modelKey = "";
      this.analysisPanel.setError(e instanceof Error ? e.message : String(e));
    }
    this.syncUrl();
    this.runAnalysis(this.version);
  }

  /** Drop an .onnx anywhere to analyse with it, or a .tslog.json to watch it. */
  private setupFileDrop() {
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
      document.body.classList.add("drop-target");
    });
    window.addEventListener("dragleave", () => document.body.classList.remove("drop-target"));
    window.addEventListener("drop", async (e) => {
      e.preventDefault();
      document.body.classList.remove("drop-target");
      const f = e.dataTransfer?.files?.[0];
      if (!f) return;
      if (f.name.endsWith(".onnx")) {
        this.analysisPanel.dropFile(f);
      } else if (f.name.endsWith(".json")) {
        try {
          this.replayControls.loadReplayData(JSON.parse(await f.text()));
        } catch (err) {
          window.alert(`Not a replay: ${err}`);
        }
      }
    });
  }

  private setReplayMode(on: boolean) {
    this.isReplayMode = on;
    this.replayControls.watching = on;
    const replayBtn = document.getElementById("btn-toggle-replay");
    if (replayBtn) {
      replayBtn.textContent = on ? "Live Mode" : "Replay Mode";
      replayBtn.className = on ? "btn btn-warning btn-sm" : "btn btn-primary btn-sm";
    }
    this.analysisPanel?.show(!on);
  }

  private renderState() {
    if (!this.state) return;
    this.tracksView.render(this.state);
    this.mapView.render(this.state);
    this.cardsView.render(this.state);
    this.actionHud.render(this.state);
    this.renderLogStream();
    if (!this.isReplayMode) {
      this.analysisPanel.render(this.liveAnalysis, this.state);
      this.analysisPanel.decorate(this.state);
    }
  }

  /** Re-apply the probability badges after part of the board re-rendered on its own. */
  private redecorate() {
    if (this.isReplayMode) {
      const next = this.replaySteps[this.replayCurrentStep + 1];
      decorateChoices(next?.policy, this.state);
    } else {
      this.analysisPanel.decorate(this.state);
    }
  }

  private getFilteredIndices(
    totalCount: number,
    isVpChanging: (index: number) => boolean
  ): { targetIndices: Set<number>; includedIndices: number[] } {
    const targetIndices = new Set<number>();
    for (let i = 0; i < totalCount; i++) {
      if (isVpChanging(i)) {
        targetIndices.add(i);
      }
    }

    if (!this.isVpFilterActive) {
      const all: number[] = [];
      for (let i = 0; i < totalCount; i++) all.push(i);
      return { targetIndices, includedIndices: all };
    }

    const includedSet = new Set<number>();
    targetIndices.forEach(targetIdx => {
      for (let offset = -2; offset <= 2; offset++) {
        const idx = targetIdx + offset;
        if (idx >= 0 && idx < totalCount) {
          includedSet.add(idx);
        }
      }
    });

    const sorted = Array.from(includedSet).sort((a, b) => a - b);
    return { targetIndices, includedIndices: sorted };
  }

  /**
   * The replay-only views of the model's trace.
   *
   * The board on screen is the position *after* the current step, which is the node the NEXT
   * step was decided at -- so the probabilities painted on the cards, buttons and map come from
   * step N+1, while the critic numbers describe the position itself and come from step N.
   */
  private renderTrace() {
    const ribbon = document.getElementById("rep-value-ribbon");
    if (ribbon) {
      renderValueRibbon(ribbon, this.replaySteps, this.replayCurrentStep,
                        (idx: number) => this.replayControls.goToStep(idx));
    }
    const current = this.replaySteps[this.replayCurrentStep];
    const next = this.replaySteps[this.replayCurrentStep + 1];
    renderTracePanel(current, next, this.replayCurrentStep);
    decorateChoices(next?.policy, this.state);
  }

  private renderLogStream() {
    const logContainer = document.getElementById("action-log-stream");
    const logCountBadge = document.getElementById("log-count");
    if (!logContainer) return;

    if (this.isReplayMode && this.replaySteps && this.replaySteps.length > 0) {
      const isVpChangingReplay = (idx: number): boolean => {
        const s = this.replaySteps[idx];
        const prev = idx > 0 ? this.replaySteps[idx - 1] : null;
        const vpDiff = prev ? (s.state_snapshot?.victory_points !== prev.state_snapshot?.victory_points) : false;
        const descHasVp = Boolean(s.description && (s.description.includes("Victory Points") || s.description.includes(" VP")));
        return Boolean(vpDiff || descHasVp);
      };

      const { targetIndices, includedIndices } = this.getFilteredIndices(this.replaySteps.length, isVpChangingReplay);

      if (logCountBadge) {
        logCountBadge.textContent = this.isVpFilterActive
          ? `${includedIndices.length} filtered (${targetIndices.size} VP events)`
          : `${this.replaySteps.length} steps (Current: #${this.replayCurrentStep + 1})`;
      }

      logContainer.innerHTML = "";
      if (includedIndices.length === 0) {
        logContainer.innerHTML = `<div class="log-empty-msg">No VP-affecting events found in replay.</div>`;
        return;
      }

      includedIndices.forEach((idx, k) => {
        const step = this.replaySteps[idx];
        const isCurrent = idx === this.replayCurrentStep;
        const isVpTarget = targetIndices.has(idx);

        if (k > 0 && idx > includedIndices[k - 1] + 1) {
          const gap = idx - includedIndices[k - 1] - 1;
          const sep = document.createElement("div");
          sep.className = "log-context-separator";
          sep.textContent = `••• ${gap} step${gap > 1 ? "s" : ""} skipped •••`;
          logContainer.appendChild(sep);
        }

        const item = document.createElement("div");
        item.className = `log-item ${isCurrent ? "active-replay-step" : ""} ${isVpTarget ? "vp-target-event" : ""}`;

        const playerClass = (step.player || "SYSTEM").toUpperCase();

        // Format die roll details if available in snapshot
        let dieRollHtml = "";
        const dieRoll = step.state_snapshot?.die_roll;
        if (dieRoll && dieRoll.type && dieRoll.type !== "NONE") {
          const rType = dieRoll.type;
          const rPlayer = dieRoll.roller || step.player;
          const cName = dieRoll.country_name || (dieRoll.country_id !== undefined && dieRoll.country_id < 84 ? `Country #${dieRoll.country_id}` : "");
          const r1 = dieRoll.roll1 ?? 0;
          const tot1 = dieRoll.total1 ?? r1;
          const r2 = dieRoll.roll2 ?? 0;
          const tot2 = dieRoll.total2 ?? r2;
          const succ = dieRoll.success;
          const net = dieRoll.net_delta ?? 0;

          let detailText = "";
          if (rType === "COUP") {
            const defTarget = dieRoll.mod2 ?? 0;
            const res = succ ? `Net +${net} Inf (Success)` : "Coup Failed";
            detailText = `🎲 Coup in ${cName}: ${rPlayer} rolls ${r1} (Total ${tot1}) vs ${defTarget} Def -> ${res}`;
          } else if (rType === "REALIGNMENT") {
            detailText = `🎲 Realignment in ${cName}: US rolls ${r1} (Tot ${tot1}), USSR rolls ${r2} (Tot ${tot2}) -> Net ${net}`;
          } else if (rType === "SPACE_RACE") {
            const status = succ ? "SUCCESS (Advanced)" : "FAILED";
            detailText = `🎲 Space Race Attempt: ${rPlayer} rolls ${r1} -> ${status}`;
          } else if (rType === "WAR_EVENT") {
            const cStr = dieRoll.card_name || "War Event";
            const targetStr = cName ? ` in ${cName}` : "";
            const res = succ ? "Success (Target Captured & VP)" : "Failed";
            detailText = `🎲 ${cStr}${targetStr}: ${rPlayer} rolls ${r1} (Tot ${tot1}) -> ${res}`;
          } else {
            detailText = `🎲 ${rType} Roll: ${rPlayer} rolls ${r1}`;
          }
          dieRollHtml = `<div class="log-detail-line log-die-line">${detailText}</div>`;
        }

        item.innerHTML = `
          <div>
            <div style="display: flex; gap: 6px; align-items: baseline; width: 100%;">
              <span class="log-index" style="font-size: 10px; color: var(--text-dim); min-width: 28px;">#${idx + 1}</span>
              <span class="log-step">[T${step.turn} AR${step.ar}]</span>
              <span class="log-player ${playerClass}">${playerClass}:</span>
              <span class="log-text" style="flex: 1;">${step.description || `Action type=${step.action?.decision_type}`}</span>
              ${policyChipHtml(step, idx > 0 ? this.replaySteps[idx - 1] : undefined)}
              ${isVpTarget ? '<span class="log-vp-badge">🎯 VP CHANGE</span>' : ''}
            </div>
            ${dieRollHtml}
          </div>
        `;

        item.addEventListener("click", () => {
          this.replayControls.goToStep(idx);
        });

        logContainer.appendChild(item);
      });

      // Smooth scroll active step into view
      requestAnimationFrame(() => {
        const activeEl = logContainer.querySelector(".active-replay-step") as HTMLElement;
        if (activeEl) {
          activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      });
      return;
    }

    if (!this.state?.action_logs) return;

    const logs = this.state.action_logs;
    const isVpChangingLive = (idx: number) => {
      const log = logs[idx];
      const hasVpDelta = log.vp_delta !== undefined && log.vp_delta !== 0;
      const hasVpDetail = (log.details || []).some(d => d.includes("Victory Points:") || d.includes(" VP"));
      const hasVpText = log.text.includes("Victory Points:") || log.text.includes(" VP");
      return hasVpDelta || hasVpDetail || hasVpText;
    };

    const { targetIndices, includedIndices } = this.getFilteredIndices(logs.length, isVpChangingLive);

    if (logCountBadge) {
      logCountBadge.textContent = this.isVpFilterActive
        ? `${includedIndices.length} filtered (${targetIndices.size} VP events)`
        : `${logs.length} events`;
    }

    logContainer.innerHTML = "";
    if (includedIndices.length === 0) {
      logContainer.innerHTML = `<div class="log-empty-msg">No VP-affecting events recorded yet.</div>`;
      return;
    }

    includedIndices.forEach((idx, k) => {
      const log = logs[idx];
      const isVpTarget = targetIndices.has(idx);

      if (k > 0 && idx > includedIndices[k - 1] + 1) {
        const gap = idx - includedIndices[k - 1] - 1;
        const sep = document.createElement("div");
        sep.className = "log-context-separator";
        sep.textContent = `••• ${gap} step${gap > 1 ? "s" : ""} skipped •••`;
        logContainer.appendChild(sep);
      }

      const item = document.createElement("div");
      item.className = `log-item ${isVpTarget ? "vp-target-event" : ""}`;

      let detailsHtml = "";
      if (log.details && log.details.length > 0) {
        detailsHtml = log.details.map(d => {
          let lineClass = "log-detail-line";
          if (d.includes("🎲")) lineClass += " log-die-line";
          if (d.includes("Victory Points:")) lineClass += " log-vp-line";
          return `<div class="${lineClass}">${d}</div>`;
        }).join("");
      }

      let vpDeltaBadge = "";
      if (log.vp_delta !== undefined && log.vp_delta !== 0) {
        const deltaClass = log.vp_delta > 0 ? "us" : "ussr";
        const sign = log.vp_delta > 0 ? `+${log.vp_delta}` : `${log.vp_delta}`;
        vpDeltaBadge = `<span class="vp-delta-pill ${deltaClass}">${sign} VP</span>`;
      }

      item.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
          <div>
            <span class="log-step">[T${log.turn} AR${log.ar}]</span>
            <span class="log-player ${log.player}">${log.player}:</span>
            <span class="log-main-text">${log.text}</span>
          </div>
          ${vpDeltaBadge}
        </div>
        ${detailsHtml}
      `;
      logContainer.appendChild(item);
    });

    // Ensure auto-scrolling to the latest bottom line
    requestAnimationFrame(() => {
      logContainer.scrollTop = logContainer.scrollHeight;
    });
  }

  private handleCountryClick(countryId: number) {
    if (this.isReplayMode) return;
    if (!this.state || !this.state.legal_actions) return;

    // If active decision is POINT_NODE (5), send action
    if (this.state.decision_context.decision_type === 5) {
      if (this.state.legal_actions.valid_ids.includes(countryId)) {
        const roll = this.actionHud ? this.actionHud.selectedDieRoll : 0;
        this.sendAction({
          decision_type: 5,
          primary_id: countryId,
          secondary_id: roll,
          flags: 0
        });
      }
    }
  }

  private handleCardClick(cardId: number) {
    if (this.isReplayMode) return;
    if (!this.state || !this.state.legal_actions) return;

    if (this.state.decision_context.decision_type === 1) {
      if (this.state.legal_actions.valid_ids.includes(cardId) || this.state.legal_actions.valid_ids.length === 0) {
        this.sendAction({
          decision_type: 1,
          primary_id: cardId,
          secondary_id: 0,
          flags: 0
        });
      }
    }
  }

  /** A refused action changes nothing; say so without interrupting play. */
  private refused(why: string) {
    console.warn("Engine refused:", why);
    const el = document.getElementById("connection-status");
    if (!el) return;
    const prev = { text: el.textContent, cls: el.className, title: el.title };
    this.badge("REFUSED", "disconnected", why);
    window.setTimeout(() => {
      el.textContent = prev.text;
      el.className = prev.cls;
      el.title = prev.title;
    }, 1500);
  }

  /**
   * A click. `secondary_id` carries the HUD's manual die (0 = roll, 1..6 = force), the
   * workbench's affordance for testing the engine, as it did on the server.
   */
  private sendAction(action: MicroAction) {
    if (this.isReplayMode || !this.session) return;
    const why = this.session.apply(action, action.secondary_id ?? 0);
    if (why) {
      this.refused(why);
      return;
    }
    this.refresh();
  }

  /**
   * Play a flat action in the loaded model's own action view (the favourite button, a row of the
   * panel's list, or auto-play). A composed E4.1 action is applied as its two E4 steps.
   */
  private sendFlatAction(flatIdx: number, forcedDie: number = this.actionHud.selectedDieRoll) {
    if (this.isReplayMode || !this.session) return;
    const why = this.session.applyFlat(flatIdx, this.model?.meta.mergedInfluence ?? false, forcedDie);
    if (why) {
      this.refused(why);
      return;
    }
    this.refresh();
  }

  /**
   * Auto-play: when the side to move is the auto-play side, play the model's favourite after a
   * short pause (so a person can follow the moves). Re-armed on every readout; a pending move is
   * dropped if the position changes before it fires.
   */
  private maybeAutoPlay() {
    if (this.autoPlayTimer !== null) {
      window.clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    if (this.isReplayMode) return;
    const idx = this.analysisPanel.autoPlayMove();
    if (idx === null) return;
    const version = this.version;
    this.autoPlayTimer = window.setTimeout(() => {
      this.autoPlayTimer = null;
      if (this.isReplayMode || version !== this.version) return;
      // Always the engine's own die: the manual die selector is for the moves you make.
      this.sendFlatAction(idx, 0);
    }, AUTO_PLAY_DELAY_MS);
  }

  /**
   * Take back the last move. With auto-play on, keep taking back until the decision is yours
   * again -- otherwise auto-play would replay the move just taken back, and Cancel could never
   * get past it.
   */
  private cancelAction() {
    if (this.isReplayMode || !this.session || !this.engine) return;
    if (this.autoPlayTimer !== null) {
      window.clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    if (!this.session.undo()) return;
    const auto = this.analysisPanel.autoSide;
    const side = () => (this.engine!.decisionPlayer() > 0 ? "US" : this.engine!.decisionPlayer() < 0 ? "USSR" : "");
    while (auto && side() === auto && this.session.canUndo) {
      this.session.undo();
    }
    this.refresh();
  }

  private sendDebugOverride(override: any) {
    if (!this.session) return;
    let why: string | null = null;
    if (override.op === "set_country") why = this.session.setCountry(override.country_id, override.us, override.ussr);
    else if (override.op === "set_defcon") why = this.session.setDefcon(override.defcon);
    else if (override.op === "set_vp") why = this.session.setVp(override.vp);
    if (why) {
      this.refused(why);
      return;
    }
    this.refresh();
  }
  private setupBottomResizer() {
    const resizer = document.getElementById("bottom-resizer");
    const bottomPanel = document.getElementById("bottom-panel");
    if (!resizer || !bottomPanel) return;

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    resizer.addEventListener("mousedown", (e) => {
      isResizing = true;
      startY = e.clientY;
      startHeight = bottomPanel.offsetHeight;
      resizer.classList.add("active");
      document.body.style.cursor = "ns-resize";
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!isResizing) return;
      const deltaY = startY - e.clientY;
      const newHeight = Math.max(100, Math.min(window.innerHeight * 0.65, startHeight + deltaY));
      bottomPanel.style.height = `${newHeight}px`;
    });

    window.addEventListener("mouseup", () => {
      if (isResizing) {
        isResizing = false;
        resizer.classList.remove("active");
        document.body.style.cursor = "";
      }
    });
  }

  private setupGlobalControls() {
    // Undo / Cancel Action button
    document.getElementById("btn-undo-action")?.addEventListener("click", () => {
      this.cancelAction();
    });

    // Global keyboard shortcuts (Ctrl+Z / Cmd+Z / Escape for undo)
    window.addEventListener("keydown", (e) => {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !typing) {
        e.preventDefault();
        this.cancelAction();
      } else if (e.key === "Escape" && !typing) {
        this.cancelAction();
      } else if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (!typing && !this.isReplayMode) {
          e.preventDefault();
          this.analysisPanel.playFavourite();
        }
      }
    });

    // Zoom buttons
    document.getElementById("btn-zoom-in")?.addEventListener("click", () => this.mapView.zoom(0.85));
    document.getElementById("btn-zoom-out")?.addEventListener("click", () => this.mapView.zoom(1.15));
    document.getElementById("btn-zoom-reset")?.addEventListener("click", () => this.mapView.resetView());

    // New Game: a fresh seed, in the page. The link then names the new position.
    document.getElementById("btn-new-game")?.addEventListener("click", () => {
      if (!this.session) return;
      this.session.newGame(Math.floor(Math.random() * 1_000_000));
      if (this.isReplayMode) this.setReplayMode(false);
      renderTracePanel(undefined, undefined, 0);
      this.refresh();
    });

    // Debug Tools button
    document.getElementById("btn-toggle-debug")?.addEventListener("click", () => {
      if (this.state && !this.isReplayMode) this.debugPanel.openDebugModal(this.state);
    });

    // Replay Mode Toggle button
    const replayBtn = document.getElementById("btn-toggle-replay")!;
    replayBtn.addEventListener("click", () => {
      this.setReplayMode(!this.isReplayMode);
      if (!this.isReplayMode && this.liveState) {
        // Back to the live game: the board still shows the replay's last position.
        this.state = this.liveState;
        renderTracePanel(undefined, undefined, 0);
        this.renderState();
        if (this.liveAnalysis) {
          this.analysisPanel.render(this.liveAnalysis, this.state);
          this.analysisPanel.decorate(this.state);
        }
        this.syncUrl();
        this.maybeAutoPlay();
        return;
      }
      this.renderLogStream();
    });

    // VP Filter Checkbox
    const vpFilterCheckbox = document.getElementById("chk-filter-vp") as HTMLInputElement;
    if (vpFilterCheckbox) {
      vpFilterCheckbox.addEventListener("change", (e) => {
        this.isVpFilterActive = (e.target as HTMLInputElement).checked;
        this.renderLogStream();
      });
    }

    // Discard & Removed pile modals
    document.getElementById("btn-view-discard")?.addEventListener("click", () => {
      if (this.state) this.cardsView.showPileModal("Discard Pile", this.state.discard_pile || []);
    });
    document.getElementById("btn-view-removed")?.addEventListener("click", () => {
      if (this.state) this.cardsView.showPileModal("Removed from Game Pile", this.state.removed_pile || []);
    });

    // Modal Close button & ESC key
    const closeModal = () => {
      document.getElementById("modal-container")?.classList.add("hidden");
    };
    document.getElementById("modal-close")?.addEventListener("click", closeModal);
    document.getElementById("modal-container")?.addEventListener("click", (e) => {
      if ((e.target as HTMLElement).id === "modal-container") {
        closeModal();
      }
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeModal();
      }
    });
  }
}

// Boot application
window.addEventListener("DOMContentLoaded", () => {
  (window as any).__wb = new TSApp();
});
