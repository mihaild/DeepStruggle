import { GameState, CardMetadata } from './types';

export class CardsView {
  private cardsMeta: Map<number, CardMetadata> = new Map();
  private onCardClick: (cardId: number) => void;
  private tooltipEl: HTMLElement;
  private lastState: GameState | null = null;
  private searchFilter = '';
  private locationFilter = 'ALL';

  constructor(onCardClick: (cardId: number) => void) {
    this.onCardClick = onCardClick;
    this.tooltipEl = document.getElementById('card-tooltip')!;
    this.setupTabs();
    this.setupAllCardsControls();
  }

  public setCardsMetadata(metaList: CardMetadata[]) {
    this.cardsMeta.clear();
    metaList.forEach(c => this.cardsMeta.set(c.id, c));
  }

  private setupTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const targetTab = target.getAttribute('data-tab')!;

        tabs.forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        target.classList.add('active');
        document.getElementById(targetTab)?.classList.add('active');

        if (targetTab === 'tab-all-cards' && this.lastState) {
          this.renderAllCardsList(this.lastState);
        }
      });
    });
  }

  private setupAllCardsControls() {
    const searchInput = document.getElementById('all-cards-search') as HTMLInputElement;
    const filterSelect = document.getElementById('all-cards-filter') as HTMLSelectElement;

    searchInput?.addEventListener('input', (e) => {
      this.searchFilter = (e.target as HTMLInputElement).value.toLowerCase();
      if (this.lastState) this.renderAllCardsList(this.lastState);
    });

    filterSelect?.addEventListener('change', (e) => {
      this.locationFilter = (e.target as HTMLSelectElement).value;
      if (this.lastState) this.renderAllCardsList(this.lastState);
    });
  }

  public render(state: GameState) {
    this.lastState = state;
    const ussrHand = state.hands?.USSR || [];
    const usHand = state.hands?.US || [];
    const legalCards = new Set(
      state.legal_actions && state.legal_actions.decision_type === 1 // SELECT_CARD
        ? state.legal_actions.valid_ids
        : []
    );

    // Update Tab count badges
    const ussrCountEl = document.getElementById('ussr-hand-count');
    const usCountEl = document.getElementById('us-hand-count');
    const discardCountEl = document.getElementById('discard-count');
    const removedCountEl = document.getElementById('removed-count');
    const drawCountEl = document.getElementById('draw-count');

    if (ussrCountEl) ussrCountEl.textContent = ussrHand.length.toString();
    if (usCountEl) usCountEl.textContent = usHand.length.toString();
    if (discardCountEl) discardCountEl.textContent = (state.discard_pile?.length || 0).toString();
    if (removedCountEl) removedCountEl.textContent = (state.removed_pile?.length || 0).toString();
    if (drawCountEl) drawCountEl.textContent = (state.draw_deck_count || 0).toString();

    // Render Hands
    this.renderCardList('ussr-cards-list', ussrHand, legalCards, state.decision_context?.pending_op_card);
    this.renderCardList('us-cards-list', usHand, legalCards, state.decision_context?.pending_op_card);

    // Render China Card
    this.renderChinaCard(state);

    // Render Active Flags
    this.renderFlags(state);

    // Render Context Stack
    this.renderContextStack(state);

    // Render All Cards Directory
    this.renderAllCardsList(state);
  }

  private renderCardList(containerId: string, cardIds: number[], legalCards: Set<number>, selectedId?: number) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    if (cardIds.length === 0) {
      container.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 20px;">No cards in hand</div>';
      return;
    }

    cardIds.forEach(id => {
      const meta = this.cardsMeta.get(id) || {
        id,
        name: `Card #${id}`,
        ops: 0,
        side: 'neutral',
        age: 'early war',
        description: '',
        one_time: false
      };

      const isLegal = legalCards.has(id);
      const isSelected = selectedId === id;

      const cardEl = document.createElement('div');
      cardEl.className = `card-item ${isLegal ? 'legal' : ''} ${isSelected ? 'selected' : ''}`;

      const eraClass = meta.age === 'early war' ? 'era-early' : (meta.age === 'mid war' ? 'era-mid' : 'era-late');

      cardEl.innerHTML = `
        <div class="card-item-era-bar ${eraClass}"></div>
        <div class="card-item-left">
          <div class="card-ops-badge ${meta.side}">
            ${meta.ops}
          </div>
          <div class="card-info">
            <div class="card-name">
              #${meta.id} ${meta.name} ${meta.one_time ? '★' : ''}
            </div>
            <div class="card-meta">
              ${meta.age.toUpperCase()} • ${meta.side.toUpperCase()}
            </div>
          </div>
        </div>
      `;

      cardEl.addEventListener('mouseenter', (e) => this.showTooltip(e, meta));
      cardEl.addEventListener('mouseleave', () => this.hideTooltip());
      cardEl.addEventListener('click', () => {
        if (isLegal || legalCards.size === 0) {
          this.onCardClick(id);
        }
      });

      container.appendChild(cardEl);
    });
  }

  private renderAllCardsList(state: GameState) {
    const container = document.getElementById('all-cards-list');
    if (!container) return;

    const locations = (state.card_locations || {}) as Record<string, string>;
    container.innerHTML = '';

    let matchedCount = 0;
    for (let id = 1; id <= 110; id++) {
      const meta = this.cardsMeta.get(id) || {
        id,
        name: `Card #${id}`,
        ops: 0,
        side: 'neutral',
        age: 'early war',
        description: '',
        one_time: false
      };

      const loc = locations[id.toString()] || 'DRAW_DECK';

      // Apply Location Filter
      if (this.locationFilter !== 'ALL' && loc !== this.locationFilter) {
        continue;
      }

      // Apply Search Filter
      if (this.searchFilter) {
        const nameMatch = meta.name.toLowerCase().includes(this.searchFilter);
        const idMatch = id.toString() === this.searchFilter;
        if (!nameMatch && !idMatch) continue;
      }

      matchedCount++;
      const itemEl = document.createElement('div');
      itemEl.className = 'all-cards-item';

      const locBadgeClass = this.getLocationBadgeClass(loc);
      const locLabel = this.getLocationLabel(loc);
      const eraClass = meta.age === 'early war' ? 'era-early' : (meta.age === 'mid war' ? 'era-mid' : 'era-late');

      itemEl.innerHTML = `
        <div class="card-item-era-bar ${eraClass}"></div>
        <div class="all-cards-item-left">
          <div class="card-ops-badge ${meta.side}">${meta.ops}</div>
          <div class="all-cards-info">
            <div class="card-name">#${meta.id} ${meta.name} ${meta.one_time ? '★' : ''}</div>
            <div class="card-meta">${meta.age.toUpperCase()} • ${meta.side.toUpperCase()}</div>
          </div>
        </div>
        <div class="card-loc-badge ${locBadgeClass}">${locLabel}</div>
      `;

      itemEl.addEventListener('mouseenter', (e) => this.showTooltip(e, meta));
      itemEl.addEventListener('mouseleave', () => this.hideTooltip());
      itemEl.addEventListener('click', () => {
        this.showPileModal(`Card #${meta.id}: ${meta.name}`, [id]);
      });

      container.appendChild(itemEl);
    }

    if (matchedCount === 0) {
      container.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 20px;">No cards match your filter.</div>';
    }
  }

  private getLocationBadgeClass(loc: string): string {
    switch (loc) {
      case 'HAND_US': return 'loc-us';
      case 'HAND_USSR': return 'loc-ussr';
      case 'DRAW_DECK': return 'loc-deck';
      case 'DISCARD_PILE': return 'loc-discard';
      case 'REMOVED_FROM_GAME': return 'loc-removed';
      case 'ONGOING_EVENT': return 'loc-ongoing';
      case 'UNAVAILABLE': return 'loc-future';
      default: return 'loc-default';
    }
  }

  private getLocationLabel(loc: string): string {
    switch (loc) {
      case 'HAND_US': return 'US Hand';
      case 'HAND_USSR': return 'USSR Hand';
      case 'DRAW_DECK': return 'Draw Deck';
      case 'DISCARD_PILE': return 'Discard';
      case 'REMOVED_FROM_GAME': return 'Removed';
      case 'ONGOING_EVENT': return 'Ongoing';
      case 'UNAVAILABLE': return 'Future Era';
      default: return loc;
    }
  }

  private renderChinaCard(state: GameState) {
    const chinaContainer = document.getElementById('china-card-container');
    if (!chinaContainer) return;

    const china = state.china_card || { holder: 'USSR', playable: true };
    const holder = china.holder || 'USSR';
    const isPlayable = china.playable;

    chinaContainer.innerHTML = `
      <div class="china-card-box ${isPlayable ? 'face-up' : 'face-down'}">
        <div>
          <div style="font-weight: 800; font-size: 13px;">#6 THE CHINA CARD (4 Ops)</div>
          <div style="font-size: 11px; opacity: 0.9;">Holder: <strong>${holder}</strong> (${isPlayable ? 'Face Up / Playable' : 'Face Down'})</div>
        </div>
        <div style="font-weight: bold; font-size: 11px; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">
          +1 Op in Asia
        </div>
      </div>
    `;
  }

  private renderFlags(state: GameState) {
    const flagsContainer = document.getElementById('active-flags-container');
    if (!flagsContainer) return;

    flagsContainer.innerHTML = '';
    const flags = state.flags || [];
    if (flags.length === 0) {
      flagsContainer.innerHTML = '<span style="font-size: 10px; color: var(--text-dim);">No active persistent flags</span>';
      return;
    }

    flags.forEach(f => {
      const pill = document.createElement('span');
      pill.className = 'flag-pill';
      pill.textContent = f.replace(/_ACTIVE|_PLAYED/, '');
      flagsContainer.appendChild(pill);
    });
  }

  private renderContextStack(state: GameState) {
    const stackContainer = document.getElementById('context-stack-container');
    if (!stackContainer) return;

    const ctx = state.decision_context;
    if (!ctx) return;

    stackContainer.innerHTML = `
      <div><strong>Reentrancy Depth:</strong> ${ctx.stack_depth || 0} / 2</div>
      <div><strong>Decision:</strong> ${ctx.decision_type_name || ctx.decision_type} (${ctx.decision_player})</div>
      ${ctx.pending_op_card ? `<div><strong>Pending Card:</strong> #${ctx.pending_op_card} (Ops: ${ctx.pending_ops_value})</div>` : ''}
      ${ctx.resolving_card ? `<div><strong>Resolving Card:</strong> #${ctx.resolving_card} (${ctx.resolving_card_name || ''})</div>` : ''}
      ${ctx.remaining_steps ? `<div><strong>Remaining Steps:</strong> ${ctx.remaining_steps}</div>` : ''}
    `;
  }

  private showTooltip(e: MouseEvent, meta: CardMetadata) {
    this.tooltipEl.innerHTML = `
      <div style="font-weight: bold; font-size: 13px; color: #F8FAFC; margin-bottom: 4px;">
        #${meta.id} ${meta.name} ${meta.one_time ? '★' : ''}
      </div>
      <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">
        ${meta.age.toUpperCase()} • ${meta.side.toUpperCase()} • ${meta.ops} OPS
      </div>
      <div style="font-size: 12px; color: #E2E8F0; line-height: 1.4;">
        ${meta.description}
      </div>
    `;

    this.tooltipEl.classList.remove('hidden');
    const x = Math.min(window.innerWidth - 300, e.clientX + 15);
    const y = Math.min(window.innerHeight - 200, e.clientY + 15);
    this.tooltipEl.style.left = `${x}px`;
    this.tooltipEl.style.top = `${y}px`;
  }

  private hideTooltip() {
    this.tooltipEl.classList.add('hidden');
  }

  public showPileModal(title: string, cardIds: number[]) {
    const modal = document.getElementById('modal-container');
    const titleEl = document.getElementById('modal-title');
    const bodyEl = document.getElementById('modal-body');
    if (!modal || !titleEl || !bodyEl) return;

    titleEl.textContent = `${title} (${cardIds.length} cards)`;
    bodyEl.innerHTML = '';

    if (cardIds.length === 0) {
      bodyEl.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 20px;">No cards in this pile</div>';
    } else {
      cardIds.forEach(id => {
        const meta = this.cardsMeta.get(id);
        if (!meta) return;
        const row = document.createElement('div');
        row.style.background = '#111827';
        row.style.padding = '8px 12px';
        row.style.borderRadius = '4px';
        row.style.border = '1px solid rgba(255,255,255,0.06)';
        row.innerHTML = `
          <div style="font-weight: bold; color: #F8FAFC;">#${meta.id} ${meta.name} (${meta.ops} Ops, ${meta.side.toUpperCase()}, ${meta.age.toUpperCase()})</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px; line-height: 1.35;">${meta.description}</div>
        `;
        bodyEl.appendChild(row);
      });
    }

    modal.classList.remove('hidden');
  }
}
