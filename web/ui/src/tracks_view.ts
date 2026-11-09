import { GameState } from './types';
import { engineLayout } from './engine/layout';
import { escapeHtml, num } from './html';
import { sideName, t } from './i18n';

export interface SpaceBoxMetadata {
  box: number;
  name: string;
  minOps: number;
  maxRoll: number;
  vpFirst: number;
  vpSecond: number;
  /** Whether the box gives its first holder a privilege (`ability` says which). */
  privilege: boolean;
  ability: string;
  abilityShort: string;
}

/** The boxes' names and privileges, as message ids; what each box asks and pays is the engine's. */
const SPACE_RACE_TEXT = [
  { name: "tracks:spaceBox.launchpad" },
  { name: "tracks:spaceBox.earthSatellite" },
  { name: "tracks:spaceBox.animalInSpace", ability: "tracks:privilege.animalInSpace", abilityShort: "tracks:privilegeShort.animalInSpace" },
  { name: "tracks:spaceBox.manInSpace" },
  { name: "tracks:spaceBox.manInEarthOrbit", ability: "tracks:privilege.manInEarthOrbit", abilityShort: "tracks:privilegeShort.manInEarthOrbit" },
  { name: "tracks:spaceBox.lunarOrbit" },
  { name: "tracks:spaceBox.eagleBearHasLanded", ability: "tracks:privilege.eagleBearHasLanded", abilityShort: "tracks:privilegeShort.eagleBearHasLanded" },
  { name: "tracks:spaceBox.spaceShuttle" },
  { name: "tracks:spaceBox.spaceStation", ability: "tracks:privilege.spaceStation", abilityShort: "tracks:privilegeShort.spaceStation" },
] as const;

/** A box's name, privilege and short privilege in the page's language ("None" for no privilege). */
function spaceBoxText(box: number): Pick<SpaceBoxMetadata, "name" | "privilege" | "ability" | "abilityShort"> {
  const entry: (typeof SPACE_RACE_TEXT)[number] | undefined = SPACE_RACE_TEXT[box];
  if (!entry) return { name: t("tracks:spaceBox.fallback", { box }), privilege: false, ability: t("tracks:none"), abilityShort: t("tracks:none") };
  return "ability" in entry
    ? { name: t(entry.name), privilege: true, ability: t(entry.ability), abilityShort: t(entry.abilityShort) }
    : { name: t(entry.name), privilege: false, ability: t("tracks:none"), abilityShort: t("tracks:none") };
}

/**
 * The track: each box's Ops, roll and VP as the engine plays them (SpaceRace::get_box_info, exported
 * as WasmEngine.layout.space_boxes), with its name and privilege. Empty until the engine has loaded.
 */
export function spaceRaceBoxes(): SpaceBoxMetadata[] {
  const boxes = engineLayout()?.space_boxes ?? [];
  return boxes.map((b, box) => ({
    box, minOps: b.min_ops, maxRoll: b.max_roll, vpFirst: b.vp_first, vpSecond: b.vp_second,
    ...spaceBoxText(box),
  }));
}

/** A box's name, safe for any number a replay's state holds. */
function boxName(box: number): string {
  return escapeHtml(spaceBoxText(box).name);
}

/** What the next box asks, as the app's "NEXT PROJECT: 2 OPS: Roll 1-3": a card of N Ops, a roll. */
function nextProject(at: number): string {
  const boxes = spaceRaceBoxes();
  if (!boxes.length) return "";
  const next = boxes[at + 1];
  if (!next) return `<span class="pill-next" title="${escapeHtml(t("tracks:next.doneTitle"))}">${t("tracks:next.done")}</span>`;
  return `<span class="pill-next" title="${escapeHtml(t("tracks:next.title", { box: next.box, name: next.name, ops: next.minOps, roll: next.maxRoll }))}">`
    + `${t("tracks:next.text", { ops: next.minOps, roll: next.maxRoll })}</span>`;
}

export class TracksView {
  private container: HTMLElement;
  private currentState: GameState | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public render(state: GameState) {
    this.currentState = state;
    // Numbers and names, as a replay snapshot may hold anything.
    const vp = num(state.victory_points);
    const defcon = num(state.defcon);
    const turn = num(state.turn);
    const ar = num(state.action_round);
    const phase = num(state.current_phase);
    const phasing = escapeHtml(state.phasing_player);

    const usMilOps = num(state.mil_ops?.US ?? 0);
    const ussrMilOps = num(state.mil_ops?.USSR ?? 0);
    const usSpace = num(state.space?.US ?? 0);
    const ussrSpace = num(state.space?.USSR ?? 0);

    const usTurnsUsed = num(state.space_turns_used?.US ?? 0);
    const ussrTurnsUsed = num(state.space_turns_used?.USSR ?? 0);
    const maxUsAttempts = (usSpace >= 2 && ussrSpace < 2) ? 2 : 1;
    const maxUssrAttempts = (ussrSpace >= 2 && usSpace < 2) ? 2 : 1;

    // Needle pos in percentage: -20 (USSR) is 0%, 0 is 50%, +20 (US) is 100%
    const needlePercent = Math.max(0, Math.min(100, ((vp + 20) / 40) * 100));

    let vpLabel = t('tracks:vp.tie');
    let vpClass = 'tie';
    if (vp > 0) {
      vpLabel = t('tracks:vp.lead', { side: sideName('US'), vp });
      vpClass = 'us';
    } else if (vp < 0) {
      vpLabel = t('tracks:vp.lead', { side: sideName('USSR'), vp: Math.abs(vp) });
      vpClass = 'ussr';
    }

    const eraName = turn <= 3 ? t('tracks:era.early') : (turn <= 7 ? t('tracks:era.mid') : t('tracks:era.late'));

    this.container.innerHTML = `
      <!-- DEFCON Track -->
      <div class="track-group">
        <span class="track-label">${t('tracks:label.defcon')}</span>
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
        <span class="track-label">${t('tracks:label.victoryPoints')}</span>
        <div class="vp-meter-container">
          <div class="vp-bar">
            <div class="vp-fill-ussr" style="width: ${needlePercent}%;"></div>
            <div class="vp-center-line"></div>
            <div class="vp-needle" style="left: ${needlePercent}%;"></div>
          </div>
          <span class="vp-val-badge ${vpClass}">${vpLabel}</span>
        </div>
      </div>

      <!-- Turn & Era -->
      <div class="track-group">
        <span class="track-label">${t('tracks:label.turnAndEra')}</span>
        <div class="turn-ar-badge">
          <strong>${t('tracks:turn', { turn })}</strong>
          <span style="color: var(--text-dim);">•</span>
          <span>${eraName}</span>
        </div>
      </div>

      <!-- Action Round -->
      <div class="track-group">
        <span class="track-label">${t('tracks:label.actionRound')}</span>
        <div class="turn-ar-badge">
          <strong>${(phase === 0 || state.current_phase_name === 'SETUP') ? t('tracks:round.setup') : ((phase === 1 || state.current_phase_name === 'HEADLINE') ? t('tracks:round.headline') : ((phase === 3 || state.current_phase_name === 'GAME_OVER') ? t('tracks:round.gameOver') : t('tracks:round.actionRound', { ar })))}</strong>
        </div>
      </div>

      <!-- Military Ops Track -->
      <div class="track-group">
        <span class="track-label">${t('tracks:label.militaryOps', { defcon })}</span>
        <div class="superpower-track">
          <span style="color: var(--ussr-red-light); font-weight: bold;">${t('tracks:sideColon', { side: sideName('USSR') })}</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= ussrMilOps ? 'filled-ussr' : ''}"></div>`).join('')}
          </div>
          <span style="color: var(--text-dim); margin: 0 4px;">|</span>
          <span style="color: var(--us-blue-light); font-weight: bold;">${t('tracks:sideColon', { side: sideName('US') })}</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= usMilOps ? 'filled-us' : ''}"></div>`).join('')}
          </div>
        </div>
      </div>

      <!-- Space Race Visual Track Widget -->
      <div class="track-group space-race-track-group">
        <span class="track-label">${t('tracks:label.spaceRace')}</span>
        <div class="space-race-header-widget" id="space-race-widget" role="button" tabindex="0" title="${escapeHtml(t('tracks:widget.title', { ussr: ussrSpace, us: usSpace }))}">
          <!-- USSR Pill -->
          <div class="space-pill ussr" title="${t('tracks:widget.pillTitle', { side: sideName('USSR'), box: ussrSpace, name: boxName(ussrSpace), used: ussrTurnsUsed, max: maxUssrAttempts })}">
            <span class="pill-dot ussr"></span>
            <span class="pill-side">${t('tracks:sideColon', { side: sideName('USSR') })}</span>
            <span class="pill-box-num">#${ussrSpace}</span>
            <span class="pill-attempts">(${ussrTurnsUsed}/${maxUssrAttempts})</span>
            ${nextProject(ussrSpace)}
          </div>

          <!-- Mini Progress Ladder (Boxes 0 to 8) -->
          <div class="space-mini-ladder">
            ${[0, 1, 2, 3, 4, 5, 6, 7, 8].map(step => {
              const hasUs = usSpace === step;
              const hasUssr = ussrSpace === step;
              let occ = '';
              if (hasUs && hasUssr) occ = 'both';
              else if (hasUs) occ = 'us';
              else if (hasUssr) occ = 'ussr';

              return `
                <div class="mini-box ${occ ? 'active ' + occ : ''}" data-step="${step}" title="${t('tracks:widget.boxTitle', { box: step, name: boxName(step) })}">
                  <span class="mini-box-num">${step}</span>
                  <div class="mini-box-tokens">
                    ${hasUssr ? `<div class="mini-token ussr" title="${escapeHtml(t('tracks:widget.tokenTitle', { side: sideName('USSR'), box: step }))}"></div>` : ''}
                    ${hasUs ? `<div class="mini-token us" title="${escapeHtml(t('tracks:widget.tokenTitle', { side: sideName('US'), box: step }))}"></div>` : ''}
                  </div>
                </div>
              `;
            }).join('')}
          </div>

          <!-- US Pill -->
          <div class="space-pill us" title="${t('tracks:widget.pillTitle', { side: sideName('US'), box: usSpace, name: boxName(usSpace), used: usTurnsUsed, max: maxUsAttempts })}">
            <span class="pill-dot us"></span>
            <span class="pill-side">${t('tracks:sideColon', { side: sideName('US') })}</span>
            <span class="pill-box-num">#${usSpace}</span>
            <span class="pill-attempts">(${usTurnsUsed}/${maxUsAttempts})</span>
            ${nextProject(usSpace)}
          </div>

          <!-- Expand Icon Button -->
          <span class="space-view-btn" title="${escapeHtml(t('tracks:widget.details'))}">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
              <polyline points="15 3 21 3 21 9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
          </span>
        </div>
      </div>
    `;

    // Hook up click & keyboard access to view detailed space race modal
    const widget = document.getElementById('space-race-widget');
    if (widget) {
      widget.addEventListener('click', () => {
        if (this.currentState) this.showSpaceRaceModal(this.currentState);
      });
      widget.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (this.currentState) this.showSpaceRaceModal(this.currentState);
        }
      });
    }

    // Update active player badge in top-right
    const activeBadge = document.getElementById('active-player-badge');
    if (activeBadge) {
      const activePlayer = state.decision_context?.decision_player || phasing || 'NONE';
      activeBadge.textContent = t('tracks:playerTurn', { side: sideName(activePlayer) });
      activeBadge.className = `player-badge ${activePlayer.toLowerCase()}`;
    }
  }

  public showSpaceRaceModal(state: GameState) {
    const modalContainer = document.getElementById('modal-container');
    const modalContent = modalContainer?.querySelector('.modal-content');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    if (!modalContainer || !modalTitle || !modalBody) return;

    modalTitle.innerHTML = `🚀 <span>${escapeHtml(t('tracks:modal.title'))}</span>`;
    modalContent?.classList.add('modal-lg');

    const usSpace = num(state.space?.US ?? 0);
    const ussrSpace = num(state.space?.USSR ?? 0);
    const usTurnsUsed = num(state.space_turns_used?.US ?? 0);
    const ussrTurnsUsed = num(state.space_turns_used?.USSR ?? 0);

    const maxUsAttempts = (usSpace >= 2 && ussrSpace < 2) ? 2 : 1;
    const maxUssrAttempts = (ussrSpace >= 2 && usSpace < 2) ? 2 : 1;

    let leadText = t('tracks:modal.tied');
    let leadClass = "tie";
    if (usSpace > ussrSpace) {
      leadText = `🔵 ${t('tracks:modal.leads', { side: sideName('US'), spaces: usSpace - ussrSpace })}`;
      leadClass = "us";
    } else if (ussrSpace > usSpace) {
      leadText = `🔴 ${t('tracks:modal.leads', { side: sideName('USSR'), spaces: ussrSpace - usSpace })}`;
      leadClass = "ussr";
    }

    modalBody.innerHTML = `
      <div class="space-modal-container">
        <!-- Summary Header Dashboard -->
        <div class="space-modal-summary-header">
          <!-- USSR Card -->
          <div class="space-summary-card ussr-border">
            <div class="card-header-row">
              <h4 style="color: var(--ussr-red-light); margin: 0;">🔴 ${t('tracks:modal.program', { side: sideName('USSR') })}</h4>
              <span class="badge ${ussrTurnsUsed >= maxUssrAttempts ? 'badge-danger' : 'badge-success'}">
                ${ussrTurnsUsed >= maxUssrAttempts ? t('tracks:modal.maxAttemptsUsed') : t('tracks:modal.attemptAvailable')}
              </span>
            </div>
            <div class="summary-stat-row">
              <span>${t('tracks:modal.position')}</span>
              <strong>${t('tracks:modal.positionValue', { box: ussrSpace, name: boxName(ussrSpace) })}</strong>
            </div>
            <div class="summary-stat-row">
              <span>${t('tracks:modal.attemptsThisTurn')}</span>
              <strong>${t('tracks:modal.attemptsValue', { used: ussrTurnsUsed, max: maxUssrAttempts })}</strong>
            </div>
            <div class="summary-stat-row">
              <span>${t('tracks:modal.activePerks')}</span>
              <em class="perk-highlight">${this.getActivePerkSummary(ussrSpace, usSpace)}</em>
            </div>
          </div>

          <!-- Center Lead Status -->
          <div class="space-lead-indicator">
            <span class="lead-badge ${leadClass}">${leadText}</span>
            <div class="lead-subtext">${t('tracks:modal.leadSubtext')}</div>
          </div>

          <!-- US Card -->
          <div class="space-summary-card us-border">
            <div class="card-header-row">
              <h4 style="color: var(--us-blue-light); margin: 0;">🔵 ${t('tracks:modal.program', { side: sideName('US') })}</h4>
              <span class="badge ${usTurnsUsed >= maxUsAttempts ? 'badge-danger' : 'badge-success'}">
                ${usTurnsUsed >= maxUsAttempts ? t('tracks:modal.maxAttemptsUsed') : t('tracks:modal.attemptAvailable')}
              </span>
            </div>
            <div class="summary-stat-row">
              <span>${t('tracks:modal.position')}</span>
              <strong>${t('tracks:modal.positionValue', { box: usSpace, name: boxName(usSpace) })}</strong>
            </div>
            <div class="summary-stat-row">
              <span>${t('tracks:modal.attemptsThisTurn')}</span>
              <strong>${t('tracks:modal.attemptsValue', { used: usTurnsUsed, max: maxUsAttempts })}</strong>
            </div>
            <div class="summary-stat-row">
              <span>${t('tracks:modal.activePerks')}</span>
              <em class="perk-highlight">${this.getActivePerkSummary(usSpace, ussrSpace)}</em>
            </div>
          </div>
        </div>

        <!-- 9 Milestone Boxes Grid -->
        <div class="space-ladder-grid">
          ${spaceRaceBoxes().map(box => {
            const isUsHere = usSpace === box.box;
            const isUssrHere = ussrSpace === box.box;
            let occupancyClass = "";
            if (isUsHere && isUssrHere) occupancyClass = "occupied-both";
            else if (isUsHere) occupancyClass = "occupied-us";
            else if (isUssrHere) occupancyClass = "occupied-ussr";

            const isPassedByUs = usSpace > box.box && box.box > 0;
            const isPassedByUssr = ussrSpace > box.box && box.box > 0;

            const isUsActivePerk = (box.box === 2 && usSpace >= 2 && ussrSpace < 2) ||
                                  (box.box === 4 && usSpace >= 4 && ussrSpace < 4) ||
                                  (box.box === 6 && usSpace >= 6 && ussrSpace < 6) ||
                                  (box.box === 8 && usSpace >= 8 && ussrSpace < 8);

            const isUssrActivePerk = (box.box === 2 && ussrSpace >= 2 && usSpace < 2) ||
                                    (box.box === 4 && ussrSpace >= 4 && usSpace < 4) ||
                                    (box.box === 6 && ussrSpace >= 6 && usSpace < 6) ||
                                    (box.box === 8 && ussrSpace >= 8 && usSpace < 8);

            return `
              <div class="space-box-card ${occupancyClass}" data-box="${box.box}">
                <div class="space-box-header">
                  <div class="box-title-group">
                    <span class="space-box-num">${t('tracks:modal.boxNumber', { box: box.box })}</span>
                    <span class="space-box-name">${box.name}</span>
                  </div>
                  <div class="space-box-tokens">
                    ${isUssrHere ? `<span class="space-player-token ussr">🔴 ${sideName('USSR')}</span>` : ''}
                    ${isUsHere ? `<span class="space-player-token us">🔵 ${sideName('US')}</span>` : ''}
                  </div>
                </div>

                <div class="space-box-details">
                  <div class="detail-item">
                    <span class="detail-label">${t('tracks:modal.requiredOps')}</span>
                    <strong class="detail-val">${box.minOps > 0 ? t('tracks:modal.requiredOpsValue', { ops: box.minOps }) : t('tracks:none')}</strong>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">${t('tracks:modal.rollToAdvance')}</span>
                    <strong class="detail-val">${box.maxRoll > 0 ? `🎲 1–${box.maxRoll}` : t('tracks:none')}</strong>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">${t('tracks:modal.firstVp')}</span>
                    <strong class="detail-val vp-first">${box.vpFirst > 0 ? t('tracks:modal.vpValue', { vp: box.vpFirst }) : '—'}</strong>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">${t('tracks:modal.secondVp')}</span>
                    <strong class="detail-val vp-second">${box.vpSecond > 0 ? t('tracks:modal.vpValue', { vp: box.vpSecond }) : '—'}</strong>
                  </div>
                </div>

                ${box.privilege ? `
                  <div class="space-box-ability ${(isUsActivePerk || isUssrActivePerk) ? 'active-privilege' : ''}">
                    <div class="ability-title">${t('tracks:modal.privilegeTitle')}</div>
                    <div class="ability-desc">${box.ability}</div>
                    ${isUssrActivePerk ? `<div class="perk-holder ussr">🔴 ${t('tracks:modal.activeFor', { side: sideName('USSR') })}</div>` : ''}
                    ${isUsActivePerk ? `<div class="perk-holder us">🔵 ${t('tracks:modal.activeFor', { side: sideName('US') })}</div>` : ''}
                    ${(!isUsActivePerk && !isUssrActivePerk && usSpace >= box.box && ussrSpace >= box.box && box.box > 0) ? `<div class="perk-holder cancelled">${t('tracks:modal.cancelled')}</div>` : ''}
                  </div>
                ` : ''}

                <div class="space-box-footer">
                  <span class="${isUssrHere ? 'status-current ussr' : (isPassedByUssr ? 'status-passed' : 'status-pending')}">
                    ${t('tracks:modal.status', { side: sideName('USSR'), status: isUssrHere ? t('tracks:modal.here') : (isPassedByUssr ? t('tracks:modal.cleared') : t('tracks:modal.pending')) })}
                  </span>
                  <span class="${isUsHere ? 'status-current us' : (isPassedByUs ? 'status-passed' : 'status-pending')}">
                    ${t('tracks:modal.status', { side: sideName('US'), status: isUsHere ? t('tracks:modal.here') : (isPassedByUs ? t('tracks:modal.cleared') : t('tracks:modal.pending')) })}
                  </span>
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
    if (myStep >= 2 && oppStep < 2) perks.push(t("tracks:perk.twoAttempts"));
    if (myStep >= 4 && oppStep < 4) perks.push(t("tracks:perk.opponentRevealsHeadline"));
    if (myStep >= 6 && oppStep < 6) perks.push(t("tracks:perk.discardHeldCard"));
    if (myStep >= 8 && oppStep < 8) perks.push(t("tracks:perk.eighthActionRound"));
    return perks.length > 0 ? perks.join(", ") : t("tracks:none");
  }
}
