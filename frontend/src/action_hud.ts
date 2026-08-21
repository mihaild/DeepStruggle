import { GameState, MicroAction } from './types';

export class ActionHUD {
  private container: HTMLElement;
  private badgeEl: HTMLElement;
  private onAction: (action: MicroAction) => void;

  constructor(container: HTMLElement, badgeEl: HTMLElement, onAction: (action: MicroAction) => void) {
    this.container = container;
    this.badgeEl = badgeEl;
    this.onAction = onAction;
  }

  public render(state: GameState) {
    const ctx = state.decision_context;
    const legal = state.legal_actions;
    if (!ctx || !legal) {
      this.badgeEl.textContent = 'WAITING';
      this.container.innerHTML = '<div style="color: var(--text-dim);">Waiting for engine...</div>';
      return;
    }

    const dType = ctx.decision_type;
    const player = ctx.decision_player;
    const validIds = legal.valid_ids || [];
    const allowEarlyStop = legal.allow_early_stop;

    const typeNames: Record<number, string> = {
      0: 'NONE',
      1: 'SELECT_CARD',
      2: 'SELECT_PLAY_MODE',
      3: 'CHOOSE_TIMING_BRANCH',
      4: 'SELECT_OP_MODE',
      5: 'POINT_NODE',
      6: 'CHOOSE_BRANCH'
    };
    this.badgeEl.textContent = typeNames[dType] || `TYPE ${dType}`;

    let promptText = '';
    let buttonsHtml = '';

    if (state.is_terminal) {
      this.container.innerHTML = `
        <div class="decision-prompt" style="color: #34D399;">
          GAME TERMINATED
        </div>
        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">
          Utility: ${state.terminal_utility > 0 ? 'US Victory' : (state.terminal_utility < 0 ? 'USSR Victory' : 'Draw')}
        </div>
        <button id="btn-restart-game" class="btn btn-primary btn-block">Start New Game</button>
      `;
      document.getElementById('btn-restart-game')?.addEventListener('click', () => {
        window.location.reload();
      });
      return;
    }

    switch (dType) {
      case 1: // SELECT_CARD
        promptText = `${player}: Select a card from your hand (or valid selection below):`;
        if (validIds.includes(6)) { // The China Card
          buttonsHtml += `<button class="btn btn-danger btn-block btn-hud-action" data-primary="6">Play The China Card (#6)</button>`;
        }
        break;

      case 2: // SELECT_PLAY_MODE
        promptText = `${player}: Choose how to play Card #${ctx.pending_op_card}:`;
        const modeLabels: Record<number, { text: string; class: string }> = {
          0: { text: 'Play as Event', class: 'btn-primary' },
          1: { text: 'Play for Operations', class: 'btn-success' },
          2: { text: 'Space Race Attempt', class: 'btn-secondary' },
          3: { text: 'Pass', class: 'btn-danger' }
        };
        validIds.forEach(m => {
          const cfg = modeLabels[m] || { text: `Mode ${m}`, class: 'btn-secondary' };
          buttonsHtml += `<button class="btn ${cfg.class} btn-block btn-hud-action" data-primary="${m}">${cfg.text}</button>`;
        });
        break;

      case 3: // CHOOSE_TIMING_BRANCH
        promptText = `${player}: Opponent Card Timing Priority:`;
        if (validIds.includes(0)) {
          buttonsHtml += `<button class="btn btn-primary btn-block btn-hud-action" data-primary="0">1. Operations First (Event Second)</button>`;
        }
        if (validIds.includes(1)) {
          buttonsHtml += `<button class="btn btn-secondary btn-block btn-hud-action" data-primary="1">2. Event First (Operations Second)</button>`;
        }
        break;

      case 4: // SELECT_OP_MODE
        promptText = `${player}: Choose Operation Mode (${ctx.pending_ops_value} Ops Available):`;
        const opLabels: Record<number, { text: string; class: string }> = {
          0: { text: 'Place Influence', class: 'btn-primary' },
          1: { text: 'Conduct Coup Attempt', class: 'btn-danger' },
          2: { text: 'Conduct Realignment', class: 'btn-secondary' }
        };
        validIds.forEach(m => {
          const cfg = opLabels[m] || { text: `Op ${m}`, class: 'btn-secondary' };
          buttonsHtml += `<button class="btn ${cfg.class} btn-block btn-hud-action" data-primary="${m}">${cfg.text}</button>`;
        });
        break;

      case 5: // POINT_NODE
        const remaining = ctx.remaining_steps > 0 ? ` (Remaining: ${ctx.remaining_steps})` : '';
        promptText = `${player}: Click a highlighted country on the map to target${remaining}`;
        if (allowEarlyStop) {
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0">Confirm Done / Pass Remaining</button>`;
        }
        break;

      case 6: // CHOOSE_BRANCH
        promptText = `${player}: Choose branch option:`;
        validIds.forEach(b => {
          buttonsHtml += `<button class="btn btn-secondary btn-block btn-hud-action" data-primary="${b}">Option ${b}</button>`;
        });
        if (allowEarlyStop) {
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0">Confirm / Pass</button>`;
        }
        break;

      default:
        promptText = `Waiting for decision...`;
        break;
    }

    this.container.innerHTML = `
      <div class="decision-prompt">${promptText}</div>
      <div class="decision-actions">${buttonsHtml}</div>
    `;

    // Attach click listeners to action buttons
    const buttons = this.container.querySelectorAll('.btn-hud-action');
    buttons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const primary = parseInt(target.getAttribute('data-primary') || '0', 10);
        const secondary = parseInt(target.getAttribute('data-secondary') || '0', 10);
        const flags = parseInt(target.getAttribute('data-flags') || '0', 10);

        this.onAction({
          decision_type: dType,
          primary_id: primary,
          secondary_id: secondary,
          flags: flags
        });
      });
    });
  }
}
