import { GameState } from './types';

export interface SpaceBoxMetadata {
  box: number;
  name: string;
  minOps: number;
  maxRoll: number;
  vpFirst: number;
  vpSecond: number;
  ability: string;
}

export const SPACE_RACE_BOXES: SpaceBoxMetadata[] = [
  { box: 0, name: "Start / Launchpad", minOps: 0, maxRoll: 0, vpFirst: 0, vpSecond: 0, ability: "None" },
  { box: 1, name: "Earth Satellite", minOps: 2, maxRoll: 3, vpFirst: 2, vpSecond: 1, ability: "None" },
  { box: 2, name: "Animal in Space", minOps: 2, maxRoll: 4, vpFirst: 0, vpSecond: 0, ability: "May attempt Space Race twice per turn" },
  { box: 3, name: "Man in Space", minOps: 2, maxRoll: 3, vpFirst: 2, vpSecond: 0, ability: "None" },
  { box: 4, name: "Lunar Orbit", minOps: 2, maxRoll: 4, vpFirst: 0, vpSecond: 0, ability: "Opponent must select & reveal headline first" },
  { box: 5, name: "Lunar Probe", minOps: 3, maxRoll: 3, vpFirst: 3, vpSecond: 1, ability: "None" },
  { box: 6, name: "Space Walk", minOps: 3, maxRoll: 4, vpFirst: 0, vpSecond: 0, ability: "May discard 1 held card at end of turn" },
  { box: 7, name: "Space Station", minOps: 3, maxRoll: 3, vpFirst: 4, vpSecond: 2, ability: "None" },
  { box: 8, name: "Moon Landing", minOps: 4, maxRoll: 2, vpFirst: 2, vpSecond: 0, ability: "May play 8 Action Rounds per turn" }
];

export class TracksView {
  private container: HTMLElement;
  private currentState: GameState | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public render(state: GameState) {
    this.currentState = state;
    const vp = state.victory_points;
    const defcon = state.defcon;
    const turn = state.turn;
    const ar = state.action_round;
    const phase = state.current_phase;
    const phasing = state.phasing_player;

    const usMilOps = state.mil_ops?.US ?? 0;
    const ussrMilOps = state.mil_ops?.USSR ?? 0;
    const usSpace = state.space?.US ?? 0;
    const ussrSpace = state.space?.USSR ?? 0;

    const usTurnsUsed = state.space_turns_used?.US ?? 0;
    const ussrTurnsUsed = state.space_turns_used?.USSR ?? 0;
    const maxUsAttempts = (usSpace >= 2 && ussrSpace < 2) ? 2 : 1;
    const maxUssrAttempts = (ussrSpace >= 2 && usSpace < 2) ? 2 : 1;

    // Needle pos in percentage: -20 (USSR) is 0%, 0 is 50%, +20 (US) is 100%
    const needlePercent = Math.max(0, Math.min(100, ((vp + 20) / 40) * 100));

    let vpLabel = 'TIE 0';
    let vpClass = 'tie';
    if (vp > 0) {
      vpLabel = `US +${vp}`;
      vpClass = 'us';
    } else if (vp < 0) {
      vpLabel = `USSR +${Math.abs(vp)}`;
      vpClass = 'ussr';
    }

    const eraName = turn <= 3 ? 'EARLY WAR' : (turn <= 7 ? 'MID WAR' : 'LATE WAR');

    this.container.innerHTML = `
      <!-- DEFCON Track -->
      <div class="track-group">
        <span class="track-label">DEFCON</span>
        <div class="defcon-scale">
          ${[5, 4, 3, 2, 1].map(lvl => `
            <div class="defcon-box ${defcon === lvl ? 'active' : ''}" data-level="${lvl}">
              ${lvl}
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Victory Points Meter -->
      <div class="track-group">
        <span class="track-label">VICTORY POINTS</span>
        <div class="vp-meter-container">
          <div class="vp-bar">
            <div class="vp-fill-ussr" style="width: 50%;"></div>
            <div class="vp-center-line"></div>
            <div class="vp-needle" style="left: ${needlePercent}%;"></div>
          </div>
          <span class="vp-val-badge ${vpClass}">${vpLabel}</span>
        </div>
      </div>

      <!-- Turn & Era -->
      <div class="track-group">
        <span class="track-label">TURN & ERA</span>
        <div class="turn-ar-badge">
          <span>T<strong>${turn}</strong>/10</span>
          <span style="color: var(--text-dim);">•</span>
          <span>${eraName}</span>
        </div>
      </div>

      <!-- Action Round -->
      <div class="track-group">
        <span class="track-label">ACTION ROUND</span>
        <div class="turn-ar-badge">
          ${phase === 0 ? '<span>SETUP</span>' : (phase === 1 ? '<span>HEADLINE</span>' : `<span>AR <strong>${ar}</strong> (${phasing})</span>`)}
        </div>
      </div>

      <!-- Military Operations -->
      <div class="track-group">
        <span class="track-label">MILITARY OPS (REQ: ${defcon})</span>
        <div class="superpower-track">
          <span style="color: var(--ussr-red-light); font-weight: bold;">USSR:</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= ussrMilOps ? 'filled-ussr' : ''}"></div>`).join('')}
          </div>
          <span style="color: var(--text-dim); margin: 0 4px;">|</span>
          <span style="color: var(--us-blue-light); font-weight: bold;">US:</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= usMilOps ? 'filled-us' : ''}"></div>`).join('')}
          </div>
        </div>
      </div>

      <!-- Space Race Visual Track -->
      <div class="track-group">
        <span class="track-label">SPACE RACE (CLICK TO VIEW)</span>
        <div class="space-race-track-widget" id="space-race-widget" title="Click to view full Space Race Track and perks">
          <div class="space-mini-steps">
            ${[0, 1, 2, 3, 4, 5, 6, 7, 8].map(step => {
              const hasUs = usSpace === step;
              const hasUssr = ussrSpace === step;
              const isOccupied = hasUs || hasUssr;
              return `
                <div class="space-mini-step ${isOccupied ? 'active' : ''}" title="Box ${step}: ${SPACE_RACE_BOXES[step].name}">
                  <span>${step}</span>
                  <div class="space-step-dots">
                    ${hasUssr ? '<div class="space-token-dot ussr" title="USSR at Box ' + step + '"></div>' : ''}
                    ${hasUs ? '<div class="space-token-dot us" title="US at Box ' + step + '"></div>' : ''}
                  </div>
                </div>
              `;
            }).join('')}
          </div>
          <div class="space-summary-labels">
            <span class="ussr-label">🔴 USSR: #${ussrSpace} (${ussrTurnsUsed}/${maxUssrAttempts})</span>
            <span class="us-label">🔵 US: #${usSpace} (${usTurnsUsed}/${maxUsAttempts})</span>
          </div>
        </div>
      </div>
    `;

    // Hook up click to view detailed space race modal
    document.getElementById('space-race-widget')?.addEventListener('click', () => {
      if (this.currentState) this.showSpaceRaceModal(this.currentState);
    });

    // Update active player badge in top-right
    const activeBadge = document.getElementById('active-player-badge');
    if (activeBadge) {
      const activePlayer = state.decision_context?.decision_player || phasing || 'NONE';
      activeBadge.textContent = `${activePlayer} TURN`;
      activeBadge.className = `player-badge ${activePlayer.toLowerCase()}`;
    }
  }

  public showSpaceRaceModal(state: GameState) {
    const modalContainer = document.getElementById('modal-container');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    if (!modalContainer || !modalTitle || !modalBody) return;

    modalTitle.textContent = "SPACE RACE PROGRESS & TRACK";

    const usSpace = state.space?.US ?? 0;
    const ussrSpace = state.space?.USSR ?? 0;
    const usTurnsUsed = state.space_turns_used?.US ?? 0;
    const ussrTurnsUsed = state.space_turns_used?.USSR ?? 0;

    const maxUsAttempts = (usSpace >= 2 && ussrSpace < 2) ? 2 : 1;
    const maxUssrAttempts = (ussrSpace >= 2 && usSpace < 2) ? 2 : 1;

    let leadText = "TIED IN SPACE RACE";
    let leadClass = "tie";
    if (usSpace > ussrSpace) {
      leadText = `US LEADS BY ${usSpace - ussrSpace} SPACES`;
      leadClass = "us";
    } else if (ussrSpace > usSpace) {
      leadText = `USSR LEADS BY ${ussrSpace - usSpace} SPACES`;
      leadClass = "ussr";
    }

    modalBody.innerHTML = `
      <div class="space-modal-container">
        <!-- Summary Header -->
        <div class="space-modal-summary-header">
          <div class="space-summary-card">
            <h4 style="color: var(--ussr-red-light);">🔴 USSR Space Program</h4>
            <div>Current Position: <strong>Box #${ussrSpace} (${SPACE_RACE_BOXES[ussrSpace].name})</strong></div>
            <div>Attempts this Turn: <strong>${ussrTurnsUsed} / ${maxUssrAttempts}</strong> ${ussrTurnsUsed >= maxUssrAttempts ? '<span class="badge" style="background:#EF4444;">Max Reached</span>' : '<span class="badge" style="background:#10B981;">Available</span>'}</div>
            <div>Active Perks: <em>${this.getActivePerkSummary(ussrSpace, usSpace)}</em></div>
          </div>

          <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;">
            <span class="badge ${leadClass}" style="font-size: 13px; padding: 6px 12px;">${leadText}</span>
            <span style="font-size: 11px; color: var(--text-dim);">Discard cards for Ops ≥ Box Min to advance</span>
          </div>

          <div class="space-summary-card">
            <h4 style="color: var(--us-blue-light);">🔵 US Space Program</h4>
            <div>Current Position: <strong>Box #${usSpace} (${SPACE_RACE_BOXES[usSpace].name})</strong></div>
            <div>Attempts this Turn: <strong>${usTurnsUsed} / ${maxUsAttempts}</strong> ${usTurnsUsed >= maxUsAttempts ? '<span class="badge" style="background:#EF4444;">Max Reached</span>' : '<span class="badge" style="background:#10B981;">Available</span>'}</div>
            <div>Active Perks: <em>${this.getActivePerkSummary(usSpace, ussrSpace)}</em></div>
          </div>
        </div>

        <!-- 9 Boxes Grid -->
        <div class="space-ladder-grid">
          ${SPACE_RACE_BOXES.map(box => {
            const isUsHere = usSpace === box.box;
            const isUssrHere = ussrSpace === box.box;
            let occupancyClass = "";
            if (isUsHere && isUssrHere) occupancyClass = "occupied-both";
            else if (isUsHere) occupancyClass = "occupied-us";
            else if (isUssrHere) occupancyClass = "occupied-ussr";

            const isCompletedByUs = usSpace >= box.box && box.box > 0;
            const isCompletedByUssr = ussrSpace >= box.box && box.box > 0;

            return `
              <div class="space-box-card ${occupancyClass}">
                <div class="space-box-header">
                  <span class="space-box-num">BOX #${box.box}</span>
                  <span class="space-box-name">${box.name}</span>
                  <div class="space-box-tokens">
                    ${isUssrHere ? '<span class="space-player-token ussr">🔴 USSR</span>' : ''}
                    ${isUsHere ? '<span class="space-player-token us">🔵 US</span>' : ''}
                  </div>
                </div>

                <div class="space-box-details">
                  <div>Required Ops: <strong>${box.minOps > 0 ? `≥ ${box.minOps} Ops` : '—'}</strong></div>
                  <div>Roll Needed: <strong>${box.maxRoll > 0 ? `1–${box.maxRoll}` : '—'}</strong></div>
                  <div>1st Player VP: <strong>${box.vpFirst > 0 ? `+${box.vpFirst} VP` : '—'}</strong></div>
                  <div>2nd Player VP: <strong>${box.vpSecond > 0 ? `+${box.vpSecond} VP` : '—'}</strong></div>
                </div>

                ${box.ability !== 'None' ? `
                  <div class="space-box-ability">
                    ✨ <strong>Privilege:</strong> ${box.ability}
                  </div>
                ` : ''}

                <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim);">
                  <span>USSR: ${isCompletedByUssr ? '✓ Cleared' : (isUssrHere ? '📍 Current' : 'Pending')}</span>
                  <span>US: ${isCompletedByUs ? '✓ Cleared' : (isUsHere ? '📍 Current' : 'Pending')}</span>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;

    modalContainer.classList.remove('hidden');
  }

  private getActivePerkSummary(myStep: number, oppStep: number): string {
    const perks: string[] = [];
    if (myStep >= 2 && oppStep < 2) perks.push("2 attempts/turn");
    if (myStep >= 4 && oppStep < 4) perks.push("Opponent reveals headline first");
    if (myStep >= 6 && oppStep < 6) perks.push("Discard held card at turn end");
    if (myStep >= 8 && oppStep < 8) perks.push("8th Action Round");
    return perks.length > 0 ? perks.join(", ") : "None";
  }
}
