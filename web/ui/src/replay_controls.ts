import { DieRollState, EventTrackState, GameState } from "./types";
import { onLanguageChange, sideName, t } from "./i18n";

/** One legal action and the probability the policy put on it. */
export interface PolicyChoice {
  idx: number;
  p: number;
  name?: string;
}

/**
 * What the policy believed at the node this step was played from.
 *
 * Optional throughout: only a neural side records one, and `source` says how the move came
 * about -- "policy" (sampled), "forced" (one legal action, played by the loop), "scripted" (a
 * forced opening overrode the choice, so there is no distribution) or "annotated" (a checkpoint
 * asked after the fact, about a game it may not have played).
 */
export interface PolicyTrace {
  source?: string;
  temperature?: number;
  n_legal?: number;
  chosen_idx?: number;
  p_chosen?: number;
  p_chosen_sampled?: number;
  argmax_idx?: number;
  p_max?: number;
  entropy?: number;
  top?: PolicyChoice[];
  p_tail?: number;
  v_win?: number;
  v_vp?: number;
}

/** Both value heads from both perspectives, read on the state this step's snapshot shows. */
export interface CriticTrace {
  v_win_us?: number;
  v_win_ussr?: number;
  v_vp_us?: number;
  v_vp_ussr?: number;
  win_residual?: number;
  vp_residual?: number;
  at?: string;
}

export interface ReplayStep {
  step_index: number;
  turn: number;
  ar: number;
  phase: string;
  player: string;
  action: any;
  description: string;
  state_snapshot: GameState;
  /** Every die the step rolled, in order (the live session's; a replay file's step has only its snapshot). */
  rolls?: DieRollState[];
  /** What each Event of the step did to VP and DEFCON, over the same engine steps as `rolls`. */
  event_tracks?: EventTrackState[];
  policy?: PolicyTrace;
  critic?: CriticTrace;
}

/** One file of the local server's replay directory (web/server/replay.py); `error` when it could not be read. */
interface ServerReplay {
  filename: string;
  game_id?: string;
  total_steps?: number;
  result?: { winner?: string } | null;
  error?: string;
}

export interface ReplayFile {
  version?: string;
  metadata?: {
    game_id?: string;
    seed?: number;
    total_steps?: number;
    result?: any;
  };
  steps?: ReplayStep[];
  logs?: any[];
  events?: any[];
  game_id?: string;
  seed?: number;
  winner?: string;
  final_vp?: number;
  final_turn?: number;
}

export class ReplayControls {
  public steps: ReplayStep[] = [];
  public currentStep = 0;
  private isPlaying = false;
  private timer: number | null = null;
  private onStepChange: (state: GameState, stepIndex: number, allSteps: ReplayStep[]) => void;

  private sliderEl: HTMLInputElement;
  private labelEl: HTMLElement;
  private playBtn: HTMLButtonElement;
  private speedSelect: HTMLSelectElement;
  private serverSelect: HTMLSelectElement | null;
  /** The local server's replays, once listed: the select is redrawn from them in a new language. */
  private serverReplays: ServerReplay[] = [];

  constructor(onStepChange: (state: GameState, stepIndex: number, allSteps: ReplayStep[]) => void) {
    this.onStepChange = onStepChange;
    this.sliderEl = document.getElementById("rep-timeline-slider") as HTMLInputElement;
    this.labelEl = document.getElementById("rep-step-label") as HTMLElement;
    this.playBtn = document.getElementById("btn-rep-play") as HTMLButtonElement;
    this.speedSelect = document.getElementById("rep-speed-select") as HTMLSelectElement;
    this.serverSelect = document.getElementById("rep-server-select") as HTMLSelectElement | null;

    this.setupListeners();
    this.updateUI();
    onLanguageChange(() => {
      this.updateUI();
      this.renderPlayButton();
      this.renderServerList();
    });
    this.fetchServerReplays();
  }

  public async fetchServerReplays() {
    if (!this.serverSelect) return;
    try {
      // The local server's replay directory. On a static host (GitHub Pages) there is none:
      // the list is hidden and replays come in through "Load Replay".
      const res = await fetch("/api/local/replays");
      if (!res.ok) {
        this.serverSelect.classList.add("hidden");
        return;
      }
      // A file the server could not read comes with its `error` instead (web/server/replay.py).
      this.serverReplays = await res.json();
      this.renderServerList();

      // Check URL query param ?replay=
      const urlParams = new URLSearchParams(window.location.search);
      const replayParam = urlParams.get("replay");
      if (replayParam) {
        this.loadServerReplay(replayParam);
      }
    } catch (e) {
      this.serverSelect.classList.add("hidden");
      console.warn("No local replay list (static host?):", e);
    }
  }

  /** The replay select: its prompt, then each listed file (the choice kept across a redraw). */
  private renderServerList() {
    const select = this.serverSelect;
    if (!select) return;
    const chosen = select.value;
    select.innerHTML = "";
    const prompt = new Option(t("replay:localReplays"), "");
    prompt.dataset.i18n = "replay:localReplays";
    select.add(prompt);
    for (const r of this.serverReplays) {
      const opt = document.createElement("option");
      opt.value = r.filename;
      if (r.error !== undefined) {
        opt.disabled = true;
        opt.textContent = t("replay:unreadable", { file: r.filename, error: r.error });
      } else {
        const values = { game: String(r.game_id), steps: String(r.total_steps) };
        opt.textContent = r.result?.winner
          ? t("replay:entryWithWinner", { ...values, winner: sideName(r.result.winner) })
          : t("replay:entry", values);
      }
      select.appendChild(opt);
    }
    select.value = chosen;
  }

  public async loadServerReplay(filename: string) {
    try {
      const res = await fetch(`/api/local/replays/${encodeURIComponent(filename)}`);
      if (!res.ok) return;
      const data = await res.json();
      this.loadReplayData(data);
      if (this.serverSelect) {
        this.serverSelect.value = filename;
      }
    } catch (e) {
      console.error("Failed to load replay:", e);
    }
  }

  public loadReplayData(replayData: ReplayFile) {
    if (replayData.steps && replayData.steps.length > 0) {
      this.steps = replayData.steps;
    } else if (replayData.logs && replayData.logs.length > 0) {
      this.steps = replayData.logs.map((l, i) => ({
        step_index: l.step || i + 1,
        turn: l.turn || 1,
        ar: l.ar || 1,
        phase: l.phase || "ACTION",
        player: l.phasing_player || l.decision?.player || "SYSTEM",
        action: l.action_executed || {},
        description: l.commentary || l.strategy || `Step ${i + 1}`,
        state_snapshot: l.state_snapshot || {}
      }));
    } else if (replayData.events && replayData.events.length > 0) {
      this.steps = replayData.events.map((ev, i) => ({
        step_index: ev.step !== undefined ? ev.step + 1 : i + 1,
        turn: ev.turn || ev.state?.turn || 1,
        ar: ev.ar || ev.state?.action_round || 0,
        phase: ev.phase || ev.state?.current_phase_name || ev.state?.phase_name || "ACTION",
        player: ev.player || "SYSTEM",
        action: {
          decision_type: ev.decision_type,
          primary_id: ev.primary_id,
          secondary_id: ev.secondary_id,
          flat_action: ev.flat_action,
        },
        description: ev.description || `Action (type=${ev.decision_type}, flat=${ev.flat_action})`,
        state_snapshot: ev.state || ev.state_snapshot || {}
      }));
    } else {
      this.steps = [];
    }

    this.sliderEl.max = Math.max(0, this.steps.length - 1).toString();
    this.sliderEl.value = "0";
    this.currentStep = 0;
    this.updateUI();
    if (this.steps.length > 0 && this.steps[0].state_snapshot && Object.keys(this.steps[0].state_snapshot).length > 0) {
      this.onStepChange(this.steps[0].state_snapshot, 0, this.steps);
    }
  }

  private setupListeners() {
    this.sliderEl.addEventListener("input", (e) => {
      const idx = parseInt((e.target as HTMLInputElement).value, 10);
      this.goToStep(idx);
    });

    this.serverSelect?.addEventListener("change", (e) => {
      const val = (e.target as HTMLSelectElement).value;
      if (val) {
        this.loadServerReplay(val);
      }
    });

    document.getElementById("btn-rep-first")?.addEventListener("click", () => this.goToStep(0));
    document.getElementById("btn-rep-last")?.addEventListener("click", () => this.goToStep(this.steps.length - 1));
    document.getElementById("btn-rep-prev")?.addEventListener("click", () => this.goToStep(this.currentStep - 1));
    document.getElementById("btn-rep-next")?.addEventListener("click", () => this.goToStep(this.currentStep + 1));

    this.playBtn.addEventListener("click", () => {
      this.togglePlay();
    });

    document.getElementById("btn-rep-export")?.addEventListener("click", () => {
      this.exportReplay();
    });

    const fileInput = document.getElementById("input-replay-file") as HTMLInputElement;
    fileInput?.addEventListener("change", (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const json = JSON.parse(evt.target?.result as string);
          this.loadReplayData(json);
        } catch (err) {
          alert(t("replay:parseFailed"));
        }
      };
      reader.readAsText(file);
    });
  }

  public goToStep(stepIndex: number) {
    if (this.steps.length === 0) return;
    this.currentStep = Math.max(0, Math.min(this.steps.length - 1, stepIndex));
    this.sliderEl.value = this.currentStep.toString();
    this.updateUI();
    const step = this.steps[this.currentStep];
    if (step && step.state_snapshot && Object.keys(step.state_snapshot).length > 0) {
      this.onStepChange(step.state_snapshot, this.currentStep, this.steps);
    }
  }

  public togglePlay() {
    if (this.isPlaying) {
      this.stop();
    } else {
      this.play();
    }
  }

  public play() {
    if (this.steps.length === 0) return;
    this.isPlaying = true;
    this.renderPlayButton();
    const interval = parseInt(this.speedSelect.value, 10) || 500;
    this.timer = window.setInterval(() => {
      if (this.currentStep >= this.steps.length - 1) {
        this.stop();
        return;
      }
      this.goToStep(this.currentStep + 1);
    }, interval);
  }

  public stop() {
    this.isPlaying = false;
    this.renderPlayButton();
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private updateUI() {
    this.labelEl.textContent = t("replay:stepLabel", { step: this.currentStep, last: Math.max(0, this.steps.length - 1) });
  }

  /** Play or Pause; its key follows, so the static markup's redraw on a language switch agrees. */
  private renderPlayButton() {
    const key = this.isPlaying ? "replay:pause" : "replay:play";
    this.playBtn.dataset.i18n = key;
    this.playBtn.textContent = t(key);
  }

  /**
   * Set by the app: the live game as a replay document, exported when no replay is being watched
   * -- so a game played in the page (engine testing, a game against a model) can be saved.
   */
  public liveReplay: (() => object | null) | null = null;
  public watching = false;

  private exportReplay() {
    let doc: object | null = null;
    let name = "ts_replay.tslog.json";
    if (this.watching && this.steps.length > 0) {
      doc = { steps: this.steps };
      name = `ts_replay_step_${this.currentStep}.tslog.json`;
    } else if (this.liveReplay) {
      doc = this.liveReplay();
      name = `ts_workbench_${new Date().toISOString().replace(/[:.]/g, "-")}.tslog.json`;
    }
    if (!doc) {
      alert(t("replay:nothingToExport"));
      return;
    }
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }
}
