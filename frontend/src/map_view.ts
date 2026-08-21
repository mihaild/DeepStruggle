import { GameState, MapMetadata } from './types';

export class MapView {
  private svg: SVGSVGElement;
  private mapData: MapMetadata | null = null;
  private onCountryClick: (countryId: number) => void;

  // ViewBox pan & zoom state
  private viewBox = { x: -180, y: -130, w: 330, h: 255 };
  private isPanning = false;
  private startPoint = { x: 0, y: 0 };

  constructor(svgElement: SVGSVGElement, onCountryClick: (countryId: number) => void) {
    this.svg = svgElement;
    this.onCountryClick = onCountryClick;
    this.setupPanZoom();
  }

  public setMapData(data: MapMetadata) {
    this.mapData = data;
  }

  private setupPanZoom() {
    this.svg.addEventListener('mousedown', (e) => {
      // Only pan on background click, not country click
      if ((e.target as HTMLElement).tagName === 'svg' || (e.target as HTMLElement).id === 'bg-rect' || (e.target as HTMLElement).tagName === 'line') {
        this.isPanning = true;
        this.startPoint = { x: e.clientX, y: e.clientY };
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isPanning) return;
      const dx = (e.clientX - this.startPoint.x) * (this.viewBox.w / this.svg.clientWidth);
      const dy = (e.clientY - this.startPoint.y) * (this.viewBox.h / this.svg.clientHeight);
      this.viewBox.x -= dx;
      this.viewBox.y -= dy;
      this.startPoint = { x: e.clientX, y: e.clientY };
      this.updateViewBox();
    });

    window.addEventListener('mouseup', () => {
      this.isPanning = false;
    });

    this.svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
      this.zoom(zoomFactor);
    });
  }

  public zoom(factor: number) {
    const newW = this.viewBox.w * factor;
    const newH = this.viewBox.h * factor;
    this.viewBox.x += (this.viewBox.w - newW) / 2;
    this.viewBox.y += (this.viewBox.h - newH) / 2;
    this.viewBox.w = newW;
    this.viewBox.h = newH;
    this.updateViewBox();
  }

  public resetView() {
    this.viewBox = { x: -180, y: -130, w: 330, h: 255 };
    this.updateViewBox();
  }

  private updateViewBox() {
    this.svg.setAttribute('viewBox', `${this.viewBox.x} ${this.viewBox.y} ${this.viewBox.w} ${this.viewBox.h}`);
  }

  public render(state: GameState) {
    if (!this.mapData) return;

    this.svg.innerHTML = '';
    const countries = this.mapData.countries;
    const nameToCountry = new Map(countries.map(c => [c.name, c]));

    // Superpowers
    const superpowers: Record<string, { pos: [number, number]; label: string; color: string }> = {
      'USA': { pos: [-155, 48], label: 'U.S.A.', color: '#1E3A8A' },
      'USSR': { pos: [28, 98], label: 'U.S.S.R.', color: '#991B1B' }
    };

    // Background rect to catch mouse events for panning
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('id', 'bg-rect');
    bg.setAttribute('x', '-300');
    bg.setAttribute('y', '-250');
    bg.setAttribute('width', '600');
    bg.setAttribute('height', '500');
    bg.setAttribute('fill', 'transparent');
    this.svg.appendChild(bg);

    // 1. Connection lines
    const drawnEdges = new Set<string>();
    const linesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    linesGroup.setAttribute('id', 'connection-lines');

    countries.forEach(u => {
      const uPos = u.pos;
      u.neighbours.forEach(vName => {
        const edgeKey = [u.name, vName].sort().join('|');
        if (drawnEdges.has(edgeKey)) return;
        drawnEdges.add(edgeKey);

        const v = nameToCountry.get(vName);
        if (!v) return;
        const vPos = v.pos;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', uPos[0].toString());
        line.setAttribute('y1', uPos[1].toString());
        line.setAttribute('x2', vPos[0].toString());
        line.setAttribute('y2', vPos[1].toString());

        if (u.region === v.region) {
          line.setAttribute('stroke', '#64748B');
          line.setAttribute('stroke-width', '1.2');
          line.setAttribute('opacity', '0.6');
        } else {
          line.setAttribute('stroke', '#EF4444');
          line.setAttribute('stroke-width', '1.4');
          line.setAttribute('stroke-dasharray', '3,3');
          line.setAttribute('opacity', '0.7');
        }
        linesGroup.appendChild(line);
      });

      // Superpower adjacency lines
      if (u.superpower_adjacent && superpowers[u.superpower_adjacent]) {
        const sp = superpowers[u.superpower_adjacent];
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', uPos[0].toString());
        line.setAttribute('y1', uPos[1].toString());
        line.setAttribute('x2', sp.pos[0].toString());
        line.setAttribute('y2', sp.pos[1].toString());
        line.setAttribute('stroke', '#F59E0B');
        line.setAttribute('stroke-width', '1.8');
        line.setAttribute('opacity', '0.8');
        linesGroup.appendChild(line);
      }
    });
    this.svg.appendChild(linesGroup);

    // 2. Superpower Boxes
    const spGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    Object.entries(superpowers).forEach(([_key, sp]) => {
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      const w = 18, h = 9;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', (sp.pos[0] - w/2).toString());
      rect.setAttribute('y', (sp.pos[1] - h/2).toString());
      rect.setAttribute('width', w.toString());
      rect.setAttribute('height', h.toString());
      rect.setAttribute('rx', '2');
      rect.setAttribute('fill', sp.color);
      rect.setAttribute('stroke', '#F8FAFC');
      rect.setAttribute('stroke-width', '1');
      g.appendChild(rect);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', sp.pos[0].toString());
      text.setAttribute('y', (sp.pos[1] + 1.2).toString());
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#FFFFFF');
      text.setAttribute('font-size', '3.5');
      text.setAttribute('font-weight', 'bold');
      text.textContent = sp.label;
      g.appendChild(text);
      spGroup.appendChild(g);
    });
    this.svg.appendChild(spGroup);

    // 3. Country Nodes
    const legalNodes = new Set(
      state.legal_actions && state.legal_actions.decision_type === 5
        ? state.legal_actions.valid_ids
        : []
    );

    const nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    nodesGroup.setAttribute('id', 'country-nodes');

    countries.forEach(c => {
      const stateC = state.countries ? state.countries[c.name] : null;
      const usInf = stateC ? stateC.us_influence : 0;
      const ussrInf = stateC ? stateC.ussr_influence : 0;
      const controlledBy = stateC ? stateC.controlled_by : 'NONE';
      const isLegal = legalNodes.has(c.id);

      const boxW = 15;
      const boxH = 5.2;
      const x = c.pos[0];
      const y = c.pos[1];

      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', `svg-country-node ${isLegal ? 'legal-target' : ''}`);
      g.setAttribute('data-id', c.id.toString());
      g.setAttribute('data-name', c.name);

      // Card Body
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('class', 'country-card-bg');
      rect.setAttribute('x', (x - boxW/2).toString());
      rect.setAttribute('y', (y - boxH/2).toString());
      rect.setAttribute('width', boxW.toString());
      rect.setAttribute('height', boxH.toString());
      rect.setAttribute('rx', '1');

      // Base fill & border depending on Battleground & Control
      let borderColor = c.battleground ? '#A855F7' : '#475569';
      let strokeWidth = c.battleground ? '1.5' : '0.8';

      if (controlledBy === 'US') {
        borderColor = '#3B82F6';
        strokeWidth = '2';
      } else if (controlledBy === 'USSR') {
        borderColor = '#EF4444';
        strokeWidth = '2';
      }

      rect.setAttribute('fill', c.battleground ? '#1E1B4B' : '#0F172A');
      rect.setAttribute('stroke', borderColor);
      rect.setAttribute('stroke-width', strokeWidth);
      g.appendChild(rect);

      // Stability Badge (Right box)
      const badgeW = 2.4, badgeH = 2.2;
      const badgeX = x + boxW/2 - badgeW - 0.3;
      const badgeY = y - boxH/2 + 0.3;
      const stabRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      stabRect.setAttribute('x', badgeX.toString());
      stabRect.setAttribute('y', badgeY.toString());
      stabRect.setAttribute('width', badgeW.toString());
      stabRect.setAttribute('height', badgeH.toString());
      stabRect.setAttribute('rx', '0.5');
      stabRect.setAttribute('fill', c.battleground ? '#6D28D9' : '#334155');
      g.appendChild(stabRect);

      const stabText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      stabText.setAttribute('x', (badgeX + badgeW/2).toString());
      stabText.setAttribute('y', (badgeY + badgeH/2 + 0.5).toString());
      stabText.setAttribute('text-anchor', 'middle');
      stabText.setAttribute('fill', '#FFFFFF');
      stabText.setAttribute('font-size', '1.6');
      stabText.setAttribute('font-weight', 'bold');
      stabText.textContent = c.stability.toString();
      g.appendChild(stabText);

      // Country Name Text
      const nameText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      nameText.setAttribute('x', (x - boxW/2 + 1.0).toString());
      nameText.setAttribute('y', (y - boxH/2 + 2.0).toString());
      nameText.setAttribute('fill', c.battleground ? '#E9D5FF' : '#E2E8F0');
      nameText.setAttribute('font-size', '1.5');
      nameText.setAttribute('font-weight', 'bold');
      nameText.textContent = c.name.length > 13 ? c.name.substring(0, 11) + '..' : c.name;
      g.appendChild(nameText);

      // Influence Counters (Bottom bar)
      // USSR (Red) Box on Left
      const infBoxW = 3.2, infBoxH = 2.0;
      const ussrBoxX = x - boxW/2 + 1.0;
      const ussrBoxY = y + boxH/2 - infBoxH - 0.4;
      const ussrRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      ussrRect.setAttribute('x', ussrBoxX.toString());
      ussrRect.setAttribute('y', ussrBoxY.toString());
      ussrRect.setAttribute('width', infBoxW.toString());
      ussrRect.setAttribute('height', infBoxH.toString());
      ussrRect.setAttribute('rx', '0.4');
      ussrRect.setAttribute('fill', ussrInf > 0 ? '#DC2626' : '#1E293B');
      ussrRect.setAttribute('stroke', '#EF4444');
      ussrRect.setAttribute('stroke-width', '0.4');
      g.appendChild(ussrRect);

      const ussrText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      ussrText.setAttribute('x', (ussrBoxX + infBoxW/2).toString());
      ussrText.setAttribute('y', (ussrBoxY + infBoxH/2 + 0.45).toString());
      ussrText.setAttribute('text-anchor', 'middle');
      ussrText.setAttribute('fill', '#FFFFFF');
      ussrText.setAttribute('font-size', '1.4');
      ussrText.setAttribute('font-weight', 'bold');
      ussrText.textContent = ussrInf.toString();
      g.appendChild(ussrText);

      // US (Blue) Box on Right of bottom
      const usBoxX = ussrBoxX + infBoxW + 0.8;
      const usBoxY = ussrBoxY;
      const usRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      usRect.setAttribute('x', usBoxX.toString());
      usRect.setAttribute('y', usBoxY.toString());
      usRect.setAttribute('width', infBoxW.toString());
      usRect.setAttribute('height', infBoxH.toString());
      usRect.setAttribute('rx', '0.4');
      usRect.setAttribute('fill', usInf > 0 ? '#2563EB' : '#1E293B');
      usRect.setAttribute('stroke', '#3B82F6');
      usRect.setAttribute('stroke-width', '0.4');
      g.appendChild(usRect);

      const usText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      usText.setAttribute('x', (usBoxX + infBoxW/2).toString());
      usText.setAttribute('y', (usBoxY + infBoxH/2 + 0.45).toString());
      usText.setAttribute('text-anchor', 'middle');
      usText.setAttribute('fill', '#FFFFFF');
      usText.setAttribute('font-size', '1.4');
      usText.setAttribute('font-weight', 'bold');
      usText.textContent = usInf.toString();
      g.appendChild(usText);

      // Click Event
      g.addEventListener('click', () => {
        this.onCountryClick(c.id);
      });

      nodesGroup.appendChild(g);
    });

    this.svg.appendChild(nodesGroup);
  }
}
