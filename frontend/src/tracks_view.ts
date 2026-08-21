import { GameState } from './types';

export class TracksView {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public render(state: GameState) {
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

      <!-- Military Ops & Space -->
      <div class="track-group">
        <span class="track-label">MIL-OPS & SPACE</span>
        <div class="superpower-track">
          <span style="color: var(--ussr-red-light); font-weight: bold;">USSR:</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= ussrMilOps ? 'filled-ussr' : ''}"></div>`).join('')}
          </div>
          <span style="color: var(--text-dim);">S:${ussrSpace}</span>
          <span style="color: var(--text-dim); margin: 0 2px;">|</span>
          <span style="color: var(--us-blue-light); font-weight: bold;">US:</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= usMilOps ? 'filled-us' : ''}"></div>`).join('')}
          </div>
          <span style="color: var(--text-dim);">S:${usSpace}</span>
        </div>
      </div>
    `;

    // Update active player badge in top-right
    const activeBadge = document.getElementById('active-player-badge');
    if (activeBadge) {
      const activePlayer = state.decision_context?.decision_player || phasing || 'NONE';
      activeBadge.textContent = `${activePlayer} TURN`;
      activeBadge.className = `player-badge ${activePlayer.toLowerCase()}`;
    }
  }
}
