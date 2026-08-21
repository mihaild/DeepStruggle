import { GameState, MicroAction } from './types';

export class ActionHUD {
  private container: HTMLElement;
  private badgeEl: HTMLElement;
  private onAction: (action: MicroAction) => void;
  public selectedDieRoll = 0; // 0 = Auto roll, 1..6 = Manual roll
  public selectedOppDieRoll = 0; // For Realignment opponent roll

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
    const dTypeName = ctx.decision_type_name || this.getDecisionTypeName(dType);
    const player = ctx.decision_player;
    const validIds = legal.valid_ids || [];
    const allowEarlyStop = legal.allow_early_stop;
    const phaseName = state.current_phase_name || state.phase_name || '';

    // Update Decision Badge with human-readable type
    this.badgeEl.textContent = dTypeName;

    let promptText = '';
    let buttonsHtml = '';
    let showDieSelector = false;

    if (state.is_terminal) {
      const winner = state.terminal_utility > 0 ? 'US Victory' : (state.terminal_utility < 0 ? 'USSR Victory' : 'Draw');
      this.container.innerHTML = `
        <div class="decision-prompt" style="color: #34D399; font-weight: bold;">
          GAME OVER (${winner})
        </div>
        <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
          Final VP: ${state.victory_points > 0 ? '+' + state.victory_points : state.victory_points}
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
        if (phaseName === 'HEADLINE') {
          promptText = `<strong>${player} Headline:</strong> Select a card from your hand to play as your secret Headline event:`;
        } else if (phaseName === 'DISCARD') {
          promptText = `<strong>${player} Discard:</strong> Select a card from your hand to discard:`;
        } else {
          promptText = `<strong>${player} [Turn ${state.turn} AR ${state.action_round}]:</strong> Select a card from your hand to play:`;
        }
        if (validIds.includes(6)) { // The China Card
          buttonsHtml += `<button class="btn btn-danger btn-block btn-hud-action" data-primary="6" style="margin-top: 8px;">Play The China Card (4 Ops)</button>`;
        }
        break;

      case 2: // SELECT_PLAY_MODE
        const cardName = ctx.pending_op_card_name || `Card #${ctx.pending_op_card}`;
        promptText = `<strong>${player}:</strong> Choose how to play <em>${cardName}</em>:`;
        const modeLabels: Record<number, { text: string; class: string }> = {
          0: { text: '1. Play as Event', class: 'btn-primary' },
          1: { text: '2. Play for Operations', class: 'btn-success' },
          2: { text: '3. Space Race Attempt', class: 'btn-secondary' },
          3: { text: '4. Pass / Discard', class: 'btn-danger' }
        };
        validIds.forEach(m => {
          const cfg = modeLabels[m] || { text: `Mode ${m}`, class: 'btn-secondary' };
          buttonsHtml += `<button class="btn ${cfg.class} btn-block btn-hud-action" data-primary="${m}" style="margin-bottom: 6px;">${cfg.text}</button>`;
        });
        if (validIds.includes(2)) {
          showDieSelector = true;
        }
        break;

      case 3: // CHOOSE_TIMING_BRANCH
        const oppCardName = ctx.pending_op_card_name || `Card #${ctx.pending_op_card}`;
        promptText = `<strong>${player}:</strong> Opponent card <em>${oppCardName}</em> played for Ops. Choose execution order:`;
        if (validIds.includes(0)) {
          buttonsHtml += `<button class="btn btn-primary btn-block btn-hud-action" data-primary="0" style="margin-bottom: 6px;">1. Operations First (Opponent Event Second)</button>`;
        }
        if (validIds.includes(1)) {
          buttonsHtml += `<button class="btn btn-secondary btn-block btn-hud-action" data-primary="1" style="margin-bottom: 6px;">2. Opponent Event First (Operations Second)</button>`;
        }
        break;

      case 4: // SELECT_OP_MODE
        promptText = `<strong>${player}:</strong> Select Operations action with <strong>${ctx.pending_ops_value} Ops</strong> available:`;
        const opLabels: Record<number, { text: string; desc: string; class: string }> = {
          0: { text: 'Place Influence', desc: 'Place influence markers in adjacent countries', class: 'btn-primary' },
          1: { text: 'Conduct Coup Attempt', desc: 'Roll die to remove enemy influence and add own', class: 'btn-danger' },
          2: { text: 'Conduct Realignment', desc: 'Opposed die rolls to remove enemy influence', class: 'btn-secondary' }
        };
        validIds.forEach(m => {
          const cfg = opLabels[m] || { text: `Op Mode ${m}`, desc: '', class: 'btn-secondary' };
          buttonsHtml += `
            <button class="btn ${cfg.class} btn-block btn-hud-action" data-primary="${m}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
              <div style="font-weight: bold;">${cfg.text}</div>
              <div style="font-size: 11px; opacity: 0.85;">${cfg.desc}</div>
            </button>
          `;
        });
        break;

      case 5: // POINT_NODE
        if (phaseName === 'SETUP') {
          const region = player === 'USSR' ? 'Eastern Europe' : 'Western Europe';
          promptText = `<strong>${player} Setup Phase:</strong> Click highlighted countries in ${region} to place influence (<strong>${ctx.remaining_steps} remaining</strong>).`;
        } else if (ctx.resolving_card > 0) {
          const resCard = ctx.resolving_card_name || `Card #${ctx.resolving_card}`;
          promptText = `<strong>${player}:</strong> Resolving event <em>${resCard}</em>. Click a highlighted target country (<strong>${ctx.remaining_steps} remaining</strong>):`;
          showDieSelector = true;
        } else {
          const remaining = ctx.remaining_steps > 0 ? ` (${ctx.remaining_steps} Ops remaining)` : '';
          promptText = `<strong>${player}:</strong> Click a highlighted country on the map to target${remaining}:`;
          showDieSelector = true;
        }

        if (allowEarlyStop) {
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">✓ Done / Pass Remaining</button>`;
        }
        break;

      case 6: // CHOOSE_BRANCH
        promptText = `<strong>${player}:</strong> Choose an option for this event:`;
        const actionLabels = legal.valid_action_labels || {};
        validIds.forEach(b => {
          const label = actionLabels[b.toString()] || `Option ${b + 1}`;
          buttonsHtml += `<button class="btn btn-secondary btn-block btn-hud-action" data-primary="${b}" style="margin-bottom: 6px;">${label}</button>`;
        });
        if (allowEarlyStop) {
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">✓ Confirm / Pass</button>`;
        }
        break;

      default:
        promptText = `Waiting for decision...`;
        break;
    }

    let dieSelectorHtml = '';
    if (showDieSelector) {
      dieSelectorHtml = `
        <div class="die-roll-selector-container">
          <div class="die-roll-header">
            <span>🎲 Die Roll Input:</span>
            <span class="die-selected-text">${this.selectedDieRoll === 0 ? 'Auto Roll (PRNG)' : 'Manual: ' + this.selectedDieRoll}</span>
          </div>
          <div class="die-buttons-row">
            <button class="btn-die-opt ${this.selectedDieRoll === 0 ? 'active' : ''}" data-roll="0">🎲 Auto</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 1 ? 'active' : ''}" data-roll="1">⚀ 1</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 2 ? 'active' : ''}" data-roll="2">⚁ 2</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 3 ? 'active' : ''}" data-roll="3">⚂ 3</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 4 ? 'active' : ''}" data-roll="4">⚃ 4</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 5 ? 'active' : ''}" data-roll="5">⚄ 5</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 6 ? 'active' : ''}" data-roll="6">⚅ 6</button>
          </div>
        </div>
      `;
    }

    this.container.innerHTML = `
      <div class="decision-prompt" style="font-size: 13px; line-height: 1.4; margin-bottom: 10px;">${promptText}</div>
      ${dieSelectorHtml}
      <div class="decision-actions">${buttonsHtml}</div>
    `;

    // Attach click listeners to die buttons
    const dieButtons = this.container.querySelectorAll('.btn-die-opt');
    dieButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const roll = parseInt((e.currentTarget as HTMLElement).getAttribute('data-roll') || '0', 10);
        this.selectedDieRoll = roll;
        this.render(state);
      });
    });

    // Attach click listeners to action buttons
    const buttons = this.container.querySelectorAll('.btn-hud-action');
    buttons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const primary = parseInt(target.getAttribute('data-primary') || '0', 10);
        const secondary = parseInt(target.getAttribute('data-secondary') || this.selectedDieRoll.toString(), 10);
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

  private getDecisionTypeName(dType: number): string {
    const typeNames: Record<number, string> = {
      0: 'NONE',
      1: 'SELECT_CARD',
      2: 'SELECT_PLAY_MODE',
      3: 'CHOOSE_TIMING_BRANCH',
      4: 'SELECT_OP_MODE',
      5: 'POINT_NODE',
      6: 'CHOOSE_BRANCH'
    };
    return typeNames[dType] || `TYPE_${dType}`;
  }
}
