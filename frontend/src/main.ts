import { GameState, MapMetadata, CardMetadata, MicroAction } from './types';
import { MapView } from './map_view';
import { TracksView } from './tracks_view';
import { CardsView } from './cards_view';
import { ActionHUD } from './action_hud';
import { DebugPanel } from './debug_panel';
import { ReplayControls } from './replay_controls';

class TSApp {
  private ws: WebSocket | null = null;
  private state: GameState | null = null;
  private mapData: MapMetadata | null = null;
  private cardsData: CardMetadata[] = [];

  private mapView!: MapView;
  private tracksView!: TracksView;
  private cardsView!: CardsView;
  private actionHud!: ActionHUD;
  private debugPanel!: DebugPanel;
  public replayControls!: ReplayControls;

  private isReplayMode = false;
  private gameId = 'game-1';
  private userRole = 'OBSERVER';

  constructor() {
    this.parseQueryParams();
    this.init();
  }

  private parseQueryParams() {
    const params = new URLSearchParams(window.location.search);
    if (params.has('game_id')) this.gameId = params.get('game_id')!;
    if (params.has('role')) this.userRole = params.get('role')!.toUpperCase();
  }

  private async init() {
    // 1. Fetch metadata
    await this.loadMetadata();

    // 2. Initialize Views
    const svgEl = document.getElementById('ts-map-svg') as unknown as SVGSVGElement;
    this.mapView = new MapView(svgEl, (countryId) => this.handleCountryClick(countryId));
    if (this.mapData) this.mapView.setMapData(this.mapData);

    const tracksEl = document.getElementById('global-tracks')!;
    this.tracksView = new TracksView(tracksEl);

    this.cardsView = new CardsView((cardId) => this.handleCardClick(cardId));
    this.cardsView.setCardsMetadata(this.cardsData);

    const decisionBodyEl = document.getElementById('decision-body')!;
    const decisionBadgeEl = document.getElementById('decision-type-badge')!;
    this.actionHud = new ActionHUD(decisionBodyEl, decisionBadgeEl, (action) => this.sendAction(action));

    this.debugPanel = new DebugPanel((override) => this.sendDebugOverride(override));
    if (this.mapData) this.debugPanel.setMapData(this.mapData);

    this.replayControls = new ReplayControls((snapshotState, _stepIdx) => {
      this.state = snapshotState;
      this.renderState();
    });

    this.setupGlobalControls();
    this.setupBottomResizer();
    this.connectWebSocket();
  }

  private async loadMetadata() {
    try {
      const [mapRes, cardsRes] = await Promise.all([
        fetch('/api/metadata/map'),
        fetch('/api/metadata/cards')
      ]);
      this.mapData = await mapRes.json();
      this.cardsData = await cardsRes.json();
    } catch (e) {
      console.error('Failed to load API metadata, attempting fallback:', e);
      try {
        const [mapRes, cardsRes] = await Promise.all([
          fetch('/map.json'),
          fetch('/cards.json')
        ]);
        this.mapData = await mapRes.json();
        this.cardsData = await cardsRes.json();
      } catch (err) {
        console.error('Failed to load static metadata files:', err);
      }
    }
  }

  private connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/game/${this.gameId}?role=${this.userRole}`;

    const statusBadge = document.getElementById('connection-status')!;
    statusBadge.textContent = 'CONNECTING...';
    statusBadge.className = 'status-badge';

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      statusBadge.textContent = 'CONNECTED';
      statusBadge.className = 'status-badge connected';
    };

    this.ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.type === 'STATE_UPDATE') {
        if (!this.isReplayMode) {
          this.state = data.state;
          this.renderState();
        }
      }
    };

    this.ws.onclose = () => {
      statusBadge.textContent = 'DISCONNECTED';
      statusBadge.className = 'status-badge disconnected';
      setTimeout(() => this.connectWebSocket(), 2000);
    };

    this.ws.onerror = () => {
      statusBadge.textContent = 'ERROR';
      statusBadge.className = 'status-badge disconnected';
    };
  }

  private renderState() {
    if (!this.state) return;
    this.tracksView.render(this.state);
    this.mapView.render(this.state);
    this.cardsView.render(this.state);
    this.actionHud.render(this.state);
    this.renderLogStream();
  }

  private renderLogStream() {
    const logContainer = document.getElementById('action-log-stream');
    const logCountBadge = document.getElementById('log-count');
    if (!logContainer || !this.state?.action_logs) return;

    if (logCountBadge) {
      logCountBadge.textContent = `${this.state.action_logs.length} events`;
    }

    logContainer.innerHTML = '';
    this.state.action_logs.forEach(log => {
      const item = document.createElement('div');
      item.className = 'log-item';
      
      let detailsHtml = '';
      if (log.details && log.details.length > 0) {
        detailsHtml = log.details.map(d => `<div class="log-detail-line">${d}</div>`).join('');
      }

      item.innerHTML = `
        <div>
          <span class="log-step">[T${log.turn} AR${log.ar}]</span>
          <span class="log-player ${log.player}">${log.player}:</span>
          <span>${log.text}</span>
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

    // If active decision is SELECT_CARD (1), send action
    if (this.state.decision_context.decision_type === 1) {
      if (this.state.legal_actions.valid_ids.includes(cardId)) {
        this.sendAction({
          decision_type: 1,
          primary_id: cardId,
          secondary_id: 0,
          flags: 0
        });
      }
    }
  }

  public sendAction(action: MicroAction) {
    if (this.isReplayMode) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    this.ws.send(JSON.stringify({
      type: 'PLAY_ACTION',
      action: action
    }));
  }

  public cancelAction() {
    if (this.isReplayMode) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    this.ws.send(JSON.stringify({
      type: 'CANCEL_ACTION'
    }));
  }

  private sendDebugOverride(override: any) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({
      type: 'DEBUG_OVERRIDE',
      override: override
    }));
  }

  private setupBottomResizer() {
    const resizer = document.getElementById('bottom-resizer');
    const bottomPanel = document.getElementById('bottom-panel');
    if (!resizer || !bottomPanel) return;

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    resizer.addEventListener('mousedown', (e) => {
      isResizing = true;
      startY = e.clientY;
      startHeight = bottomPanel.offsetHeight;
      resizer.classList.add('active');
      document.body.style.cursor = 'ns-resize';
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!isResizing) return;
      const deltaY = startY - e.clientY;
      const newHeight = Math.max(100, Math.min(window.innerHeight * 0.65, startHeight + deltaY));
      bottomPanel.style.height = `${newHeight}px`;
    });

    window.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        resizer.classList.remove('active');
        document.body.style.cursor = '';
      }
    });
  }

  private setupGlobalControls() {
    // Undo / Cancel Action button
    document.getElementById('btn-undo-action')?.addEventListener('click', () => {
      this.cancelAction();
    });

    // Global keyboard shortcuts (Ctrl+Z / Cmd+Z / Escape for undo)
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        this.cancelAction();
      } else if (e.key === 'Escape') {
        this.cancelAction();
      }
    });

    // Zoom buttons
    document.getElementById('btn-zoom-in')?.addEventListener('click', () => this.mapView.zoom(0.85));
    document.getElementById('btn-zoom-out')?.addEventListener('click', () => this.mapView.zoom(1.15));
    document.getElementById('btn-zoom-reset')?.addEventListener('click', () => this.mapView.resetView());

    // New Game button
    document.getElementById('btn-new-game')?.addEventListener('click', async () => {
      const seed = Math.floor(Math.random() * 1000000);
      try {
        await fetch('/api/games/new', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ game_id: this.gameId, seed })
        });
        window.location.reload();
      } catch (e) {
        console.error(e);
      }
    });

    // Debug Tools button
    document.getElementById('btn-toggle-debug')?.addEventListener('click', () => {
      if (this.state) this.debugPanel.openDebugModal(this.state);
    });

    // Replay Mode Toggle button
    const replayBtn = document.getElementById('btn-toggle-replay')!;
    replayBtn.addEventListener('click', () => {
      this.isReplayMode = !this.isReplayMode;
      replayBtn.textContent = this.isReplayMode ? 'Live Mode' : 'Replay Mode';
      replayBtn.className = this.isReplayMode ? 'btn btn-warning btn-sm' : 'btn btn-primary btn-sm';
    });

    // Discard & Removed pile modals
    document.getElementById('btn-view-discard')?.addEventListener('click', () => {
      if (this.state) this.cardsView.showPileModal('Discard Pile', this.state.discard_pile || []);
    });
    document.getElementById('btn-view-removed')?.addEventListener('click', () => {
      if (this.state) this.cardsView.showPileModal('Removed from Game Pile', this.state.removed_pile || []);
    });

    // Modal Close button
    document.getElementById('modal-close')?.addEventListener('click', () => {
      document.getElementById('modal-container')?.classList.add('hidden');
    });
    document.getElementById('modal-container')?.addEventListener('click', (e) => {
      if ((e.target as HTMLElement).id === 'modal-container') {
        document.getElementById('modal-container')?.classList.add('hidden');
      }
    });
  }
}

// Boot application
window.addEventListener('DOMContentLoaded', () => {
  new TSApp();
});
