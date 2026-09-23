import { GameState, MicroAction } from "./types";
import { MapView } from "./map_view";
import { TracksView } from "./tracks_view";
import { CardsView } from "./cards_view";
import { ActionHud } from "./action_hud";
import { ReplayControls, ReplayStep } from "./replay_controls";
import { DebugPanel } from "./debug_panel";
import { decorateChoices, loadActionSpace, policyChipHtml, renderTracePanel, renderValueRibbon } from "./trace_view";
import { AnalysisPanel, AutoSide, LiveAnalysis } from "./analysis_view";

/** Pause before an auto-played move, long enough to see each one land. */
const AUTO_PLAY_DELAY_MS = 350;

export class TSApp {
  private state: GameState | null = null;
  private ws: WebSocket | null = null;
  private gameId: string = "game-1";
  private role: string = "OBSERVER";
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

  /** The latest live position and its readout, kept while a replay is on screen. */
  private liveState: GameState | null = null;
  private liveAnalysis: LiveAnalysis | undefined = undefined;
  /** A position named by the address bar (a shared link), to be put on the board at boot. */
  private urlPosition: string | null = null;
  private urlModel: string | null = null;
  private urlAuto: AutoSide = "";
  /** A pending auto-play move, and whether a Cancel is still unwinding auto-played moves. */
  private autoPlayTimer: number | null = null;
  private undoingAutoMoves: boolean = false;

  constructor() {
    this.parseQueryParams();

    const tracksContainer = document.getElementById("global-tracks")!;
    const mapSvg = document.getElementById("ts-map-svg") as unknown as SVGSVGElement;

    this.tracksView = new TracksView(tracksContainer);
    this.mapView = new MapView(mapSvg, (countryId: number) => this.handleCountryClick(countryId));
    this.cardsView = new CardsView((cardId: number) => this.handleCardClick(cardId));
    this.actionHud = new ActionHud((action: MicroAction) => this.sendAction(action));
    this.debugPanel = new DebugPanel((override: any) => this.sendDebugOverride(override));

    this.loadGlobalMetadata();
    // The flat-action offsets, needed before a probability can be attached to a card or a
    // country. It arrives after the first render, so the trace views are redrawn once it does.
    loadActionSpace().then(() => { if (this.isReplayMode) this.renderTrace(); });

    this.replayControls = new ReplayControls((replayState: GameState, stepIndex: number, allSteps: ReplayStep[]) => {
      this.isReplayMode = true;
      const replayBtn = document.getElementById("btn-toggle-replay");
      if (replayBtn) {
        replayBtn.textContent = "Live Mode";
        replayBtn.className = "btn btn-warning btn-sm";
      }
      this.analysisPanel?.show(false);
      this.state = replayState;
      this.replaySteps = allSteps;
      this.replayCurrentStep = stepIndex;
      this.renderState();
      this.renderTrace();
    });

    this.analysisPanel = new AnalysisPanel(
      (rel: string | null) => {
        this.liveAnalysis = undefined;
        this.sendWs({ type: "SET_ANALYSIS_MODEL", model: rel });
        this.syncUrl();
      },
      (flatIdx: number) => this.sendFlatAction(flatIdx),
      () => {
        this.syncUrl();
        this.maybeAutoPlay();
      },
    );
    this.analysisPanel.setModel(this.urlModel);
    this.analysisPanel.setAutoSide(this.urlAuto);
    this.analysisPanel.show(!this.isReplayMode);
    this.actionHud.onRerender = () => this.redecorate();

    this.setupGlobalControls();
    this.setupBottomResizer();
    this.boot();
  }

  /**
   * Put a shared position on the board before connecting, so the first STATE_UPDATE is already
   * that position. The server ignores a position the game already holds, which is what makes a
   * plain reload keep the game's history.
   */
  private async boot() {
    if (this.urlPosition && !this.isReplayMode) {
      try {
        const res = await fetch(`/api/games/${encodeURIComponent(this.gameId)}/position`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ position: this.urlPosition }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `HTTP ${res.status}`);
        }
      } catch (e) {
        console.warn("Could not open the position in the link:", e);
        window.alert(`Could not open the position in the link -- showing the current game instead.\n\n${e}`);
      }
    }
    this.connectWebSocket();
  }

  private async loadGlobalMetadata() {
    try {
      const [mapRes, cardsRes] = await Promise.all([
        fetch("/api/metadata/map"),
        fetch("/api/metadata/cards")
      ]);
      if (mapRes.ok) {
        const mapData = await mapRes.json();
        this.mapView.setMapData(mapData);
        this.debugPanel.setMapData(mapData);
      }
      if (cardsRes.ok) {
        const cardsData = await cardsRes.json();
        this.cardsView.setCardsMetadata(cardsData);
      }
      // Both views re-render themselves once their metadata lands, dropping any badges.
      this.redecorate();
    } catch (e) {
      console.warn("loadGlobalMetadata failed:", e);
    }
  }

  private parseQueryParams() {
    const params = new URLSearchParams(window.location.search);
    if (params.has("game_id")) {
      this.gameId = params.get("game_id")!;
    }
    if (params.has("role")) {
      this.role = params.get("role")!.toUpperCase();
    }
    if (params.has("replay")) {
      this.isReplayMode = true;
    }
    this.urlPosition = params.get("pos");
    this.urlModel = params.get("model");
    const auto = (params.get("auto") || "").toUpperCase();
    this.urlAuto = auto === "US" || auto === "USSR" ? auto : "";
  }

  /**
   * Keep the address bar naming the board on screen -- game, role, analysis model and the
   * position itself -- so it can be copied and shared at any moment. `replaceState`, never
   * `pushState`: every move would otherwise add a history entry, and Back would walk the game
   * move by move instead of leaving the page.
   */
  private syncUrl() {
    if (this.isReplayMode || !this.liveState?.position) return;
    const params = new URLSearchParams(window.location.search);
    params.set("game_id", this.gameId);
    if (this.role !== "OBSERVER") params.set("role", this.role);
    const model = this.analysisPanel.model;
    if (model) params.set("model", model); else params.delete("model");
    const auto = this.analysisPanel.autoSide;
    if (auto) params.set("auto", auto.toLowerCase()); else params.delete("auto");
    params.set("pos", this.liveState.position);
    const next = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
    if (next !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.history.replaceState(window.history.state, "", next);
    }
  }

  private connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/game/${this.gameId}?role=${this.role}`;

    const statusBadge = document.getElementById("connection-status")!;
    statusBadge.textContent = "CONNECTING...";
    statusBadge.className = "status-badge";

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      statusBadge.textContent = "CONNECTED";
      statusBadge.className = "status-badge connected";
      // Analysis is per connection, so a reconnect has to ask again.
      if (this.analysisPanel.model) {
        this.sendWs({ type: "SET_ANALYSIS_MODEL", model: this.analysisPanel.model });
      }
    };

    this.ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.type === "STATE_UPDATE") {
        this.liveState = data.state;
        // A readout for a model this page has since switched away from is dropped by the panel.
        this.liveAnalysis = data.analysis;
        if (!this.isReplayMode) {
          this.state = data.state;
          this.renderState();
          this.syncUrl();
          if (!this.continueAutoUndo(data.state)) this.maybeAutoPlay();
        }
      } else if (data.type === "ANALYSIS_ERROR") {
        this.analysisPanel.setError(data.message || "analysis failed");
        this.syncUrl();
      }
    };

    this.ws.onclose = () => {
      statusBadge.textContent = "DISCONNECTED";
      statusBadge.className = "status-badge disconnected";
      setTimeout(() => this.connectWebSocket(), 2000);
    };

    this.ws.onerror = () => {
      statusBadge.textContent = "ERROR";
      statusBadge.className = "status-badge disconnected";
    };
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

  private sendAction(action: MicroAction) {
    if (this.isReplayMode) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    this.ws.send(JSON.stringify({
      type: "PLAY_ACTION",
      action: action
    }));
  }

  /**
   * Play a flat action in the analysis model's own action view (the favourite button, or a row
   * of the panel's list). The server decodes it, so a composed E4.1 action needs no client code.
   */
  private sendFlatAction(flatIdx: number, forcedDie: number = this.actionHud.selectedDieRoll) {
    if (this.isReplayMode) return;
    // The position it was chosen in: the server ignores it if the board has moved on, since the
    // same index at the next node can be legal and mean something else.
    this.sendWs({ type: "PLAY_FLAT", flat_idx: flatIdx, forced_die: forcedDie,
                  expect_position: this.liveState?.position });
  }

  /**
   * Auto-play: when the side to move is the auto-play side, play the model's favourite after a
   * short pause (so a person can follow the moves). Re-armed on every live update; a pending
   * move is dropped if the position changes before it fires.
   */
  private maybeAutoPlay() {
    if (this.autoPlayTimer !== null) {
      window.clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    if (this.isReplayMode || this.undoingAutoMoves) return;
    const idx = this.analysisPanel.autoPlayMove();
    const pos = this.liveState?.position;
    if (idx === null || !pos) return;
    this.autoPlayTimer = window.setTimeout(() => {
      this.autoPlayTimer = null;
      if (this.isReplayMode || this.liveState?.position !== pos) return;
      // Always the engine's own die: the manual die selector is for the moves you make.
      this.sendFlatAction(idx, 0);
    }, AUTO_PLAY_DELAY_MS);
  }

  /**
   * While undoing with auto-play on, keep undoing until the decision is yours again --
   * otherwise auto-play would immediately replay the move just taken back, and Cancel could
   * never get past it. Returns true while the chain is still running.
   */
  private continueAutoUndo(state: GameState): boolean {
    if (!this.undoingAutoMoves) return false;
    const auto = this.analysisPanel.autoSide;
    if (auto && state.decision_context?.decision_player === auto && state.can_undo) {
      this.sendWs({ type: "CANCEL_ACTION" });
      return true;
    }
    this.undoingAutoMoves = false;
    return false;
  }

  private sendWs(msg: Record<string, unknown>) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify(msg));
  }

  private cancelAction() {
    if (this.isReplayMode) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (this.analysisPanel.autoSide) {
      this.undoingAutoMoves = true;
      this.maybeAutoPlay();  // disarms any pending auto move
    }

    this.ws.send(JSON.stringify({
      type: "CANCEL_ACTION"
    }));
  }

  private sendDebugOverride(override: any) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    this.ws.send(JSON.stringify({
      type: "DEBUG_OVERRIDE",
      override: override
    }));
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
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        this.cancelAction();
      } else if (e.key === "Escape") {
        this.cancelAction();
      } else if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const t = e.target as HTMLElement | null;
        const typing = t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable);
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

    // New Game button
    document.getElementById("btn-new-game")?.addEventListener("click", async () => {
      const seed = Math.floor(Math.random() * 1000000);
      try {
        await fetch("/api/games/new", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ game_id: this.gameId, seed })
        });
        // Not a plain reload: the address bar names the old position, and booting from it
        // would put that position straight back on the new game's board.
        const params = new URLSearchParams(window.location.search);
        params.delete("pos");
        window.location.assign(`${window.location.pathname}?${params.toString()}`);
      } catch (e) {
        console.error(e);
      }
    });

    // Debug Tools button
    document.getElementById("btn-toggle-debug")?.addEventListener("click", () => {
      if (this.state) this.debugPanel.openDebugModal(this.state);
    });

    // Replay Mode Toggle button
    const replayBtn = document.getElementById("btn-toggle-replay")!;
    replayBtn.addEventListener("click", () => {
      this.isReplayMode = !this.isReplayMode;
      replayBtn.textContent = this.isReplayMode ? "Live Mode" : "Replay Mode";
      replayBtn.className = this.isReplayMode ? "btn btn-warning btn-sm" : "btn btn-primary btn-sm";
      this.analysisPanel.show(!this.isReplayMode);
      if (!this.isReplayMode && this.liveState) {
        // Back to the live game: the board still shows the replay's last position.
        this.state = this.liveState;
        renderTracePanel(undefined, undefined, 0);
        this.renderState();
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
