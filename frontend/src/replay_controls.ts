import { GameState } from './types';

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
  version: string;
  metadata: {
    game_id: string;
    seed: number;
    total_steps: number;
    result: any;
  };
  steps: ReplayStep[];
}

export class ReplayControls {
  private steps: ReplayStep[] = [];
  private currentStep = 0;
  private isPlaying = false;
  private timer: number | null = null;
  private onStepChange: (state: GameState, stepIndex: number) => void;

  private sliderEl: HTMLInputElement;
  private labelEl: HTMLElement;
  private playBtn: HTMLButtonElement;
  private speedSelect: HTMLSelectElement;

  constructor(onStepChange: (state: GameState, stepIndex: number) => void) {
    this.onStepChange = onStepChange;
    this.sliderEl = document.getElementById('rep-timeline-slider') as HTMLInputElement;
    this.labelEl = document.getElementById('rep-step-label') as HTMLElement;
    this.playBtn = document.getElementById('btn-rep-play') as HTMLButtonElement;
    this.speedSelect = document.getElementById('rep-speed-select') as HTMLSelectElement;

    this.setupListeners();
  }

  public loadReplayData(replayData: ReplayFile) {
    this.steps = replayData.steps || [];
    this.sliderEl.max = Math.max(0, this.steps.length - 1).toString();
    this.sliderEl.value = '0';
    this.currentStep = 0;
    this.updateUI();
    if (this.steps.length > 0) {
      this.onStepChange(this.steps[0].state_snapshot, 0);
    }
  }

  private setupListeners() {
    this.sliderEl.addEventListener('input', (e) => {
      const idx = parseInt((e.target as HTMLInputElement).value, 10);
      this.goToStep(idx);
    });

    document.getElementById('btn-rep-first')?.addEventListener('click', () => this.goToStep(0));
    document.getElementById('btn-rep-last')?.addEventListener('click', () => this.goToStep(this.steps.length - 1));
    document.getElementById('btn-rep-prev')?.addEventListener('click', () => this.goToStep(this.currentStep - 1));
    document.getElementById('btn-rep-next')?.addEventListener('click', () => this.goToStep(this.currentStep + 1));

    this.playBtn.addEventListener('click', () => {
      this.togglePlay();
    });

    document.getElementById('btn-rep-export')?.addEventListener('click', () => {
      this.exportReplay();
    });

    const fileInput = document.getElementById('input-replay-file') as HTMLInputElement;
    fileInput?.addEventListener('change', (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const json = JSON.parse(evt.target?.result as string);
          this.loadReplayData(json);
        } catch (err) {
          alert('Failed to parse replay file.');
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
    if (step && step.state_snapshot) {
      this.onStepChange(step.state_snapshot, this.currentStep);
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
    this.playBtn.textContent = '⏸ Pause';
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
    this.playBtn.textContent = '▶ Play';
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
      alert('No replay data to export.');
      return;
    }
    const blob = new Blob([JSON.stringify({ steps: this.steps }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ts_replay_step_${this.currentStep}.tslog.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
}
