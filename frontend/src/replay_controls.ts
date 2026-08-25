import { GameState } from "./types";

export interface ReplayStep {
  step_index: number;
  turn: number;
  ar: number;
  phase: string;
  player: string;
  action: any;
  description: string;
  state_snapshot: GameState;
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

  constructor(onStepChange: (state: GameState, stepIndex: number, allSteps: ReplayStep[]) => void) {
    this.onStepChange = onStepChange;
    this.sliderEl = document.getElementById("rep-timeline-slider") as HTMLInputElement;
    this.labelEl = document.getElementById("rep-step-label") as HTMLElement;
    this.playBtn = document.getElementById("btn-rep-play") as HTMLButtonElement;
    this.speedSelect = document.getElementById("rep-speed-select") as HTMLSelectElement;
    this.serverSelect = document.getElementById("rep-server-select") as HTMLSelectElement | null;

    this.setupListeners();
    this.fetchServerReplays();
  }

  public async fetchServerReplays() {
    if (!this.serverSelect) return;
    try {
      const res = await fetch("/api/replays");
      if (!res.ok) return;
      const replays: Array<{ filename: string; game_id: string; total_steps: number; result: any }> = await res.json();
      
      this.serverSelect.innerHTML = "<option value=\"\">Select Server Replay...</option>";
      replays.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r.filename;
        const winner = r.result?.winner ? ` [Winner: ${r.result.winner}]` : "";
        opt.textContent = `${r.game_id} (${r.total_steps} steps)${winner}`;
        this.serverSelect?.appendChild(opt);
      });

      // Check URL query param ?replay=
      const urlParams = new URLSearchParams(window.location.search);
      const replayParam = urlParams.get("replay");
      if (replayParam) {
        this.loadServerReplay(replayParam);
      }
    } catch (e) {
      console.warn("Could not fetch server replays:", e);
    }
  }

  public async loadServerReplay(filename: string) {
    try {
      const res = await fetch(`/api/replays/${filename}`);
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
          alert("Failed to parse replay file.");
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
    this.playBtn.textContent = "⏸ Pause";
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
    this.playBtn.textContent = "▶ Play";
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private updateUI() {
    this.labelEl.textContent = `Step ${this.currentStep} / ${Math.max(0, this.steps.length - 1)}`;
  }

  private exportReplay() {
    if (this.steps.length === 0) {
      alert("No replay data to export.");
      return;
    }
    const blob = new Blob([JSON.stringify({ steps: this.steps }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ts_replay_step_${this.currentStep}.tslog.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
}
