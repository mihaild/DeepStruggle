import { GameState, MapMetadata } from "./types";

export class MapView {
  private svg: SVGSVGElement;
  private mapData: MapMetadata | null = null;
  private lastState: GameState | null = null;
  private onCountryClick: (countryId: number) => void;

  // Pan and zoom state
  private viewBox = { x: 0, y: 0, w: 1000, h: 650 };
  private isPanning = false;
  private startPoint = { x: 0, y: 0 };

  constructor(svgElement: SVGSVGElement, onCountryClick: (countryId: number) => void) {
    this.svg = svgElement;
    this.onCountryClick = onCountryClick;
    this.setupPanAndZoom();
    this.updateViewBox();
    this.fetchMetadata();
  }

  public async fetchMetadata() {
    if (this.mapData) return;
    try {
      const res = await fetch("/api/metadata/map");
      if (res.ok) {
        this.mapData = await res.json();
        if (this.lastState) {
          this.render(this.lastState);
        }
      }
    } catch (e) {
      console.warn("Could not fetch map metadata from /api/metadata/map:", e);
    }
  }

  public setMapData(data: MapMetadata) {
    this.mapData = data;
    if (this.lastState) {
      this.render(this.lastState);
    }
  }

  private updateViewBox() {
    this.svg.setAttribute(
      "viewBox",
      `${this.viewBox.x} ${this.viewBox.y} ${this.viewBox.w} ${this.viewBox.h}`
    );
  }

  public zoom(factor: number) {
    const cx = this.viewBox.x + this.viewBox.w / 2;
    const cy = this.viewBox.y + this.viewBox.h / 2;
    this.viewBox.w *= factor;
    this.viewBox.h *= factor;
    this.viewBox.x = cx - this.viewBox.w / 2;
    this.viewBox.y = cy - this.viewBox.h / 2;
    this.updateViewBox();
  }

  public resetView() {
    this.viewBox = { x: 0, y: 0, w: 1000, h: 650 };
    this.updateViewBox();
  }

  private setupPanAndZoom() {
    this.svg.addEventListener("mousedown", (e) => {
      if ((e.target as HTMLElement).closest(".svg-country-node")) return;
      this.isPanning = true;
      this.startPoint = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isPanning) return;
      const dx = ((e.clientX - this.startPoint.x) / this.svg.clientWidth) * this.viewBox.w;
      const dy = ((e.clientY - this.startPoint.y) / this.svg.clientHeight) * this.viewBox.h;
      this.viewBox.x -= dx;
      this.viewBox.y -= dy;
      this.startPoint = { x: e.clientX, y: e.clientY };
      this.updateViewBox();
    });

    window.addEventListener("mouseup", () => {
      this.isPanning = false;
    });

    this.svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
      this.zoom(zoomFactor);
    }, { passive: false });
  }

  public render(state: GameState) {
    this.lastState = state;
    if (!this.mapData) {
      this.fetchMetadata();
      return;
    }

    const countries = this.mapData.countries || [];
    const countryMap = new Map<string, any>();
    countries.forEach(c => countryMap.set(c.name, c));

    this.svg.innerHTML = "";

    // 1. Connection Adjacency Lines
    const linesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    linesGroup.setAttribute("id", "adjacency-lines");
    const drawnEdges = new Set<string>();

    const superpowers = [
      { id: "USA", label: "UNITED STATES", pos: [85, 230], color: "#1E3A8A", stroke: "#3B82F6" },
      { id: "USSR", label: "SOVIET UNION", pos: [635, 120], color: "#7F1D1D", stroke: "#EF4444" }
    ];

    // Country-to-Country Edges
    countries.forEach(c => {
      (c.neighbours || []).forEach(nName => {
        const n = countryMap.get(nName);
        if (!n) return;
        const edgeKey = [Math.min(c.id, n.id), Math.max(c.id, n.id)].join("-");
        if (drawnEdges.has(edgeKey)) return;
        drawnEdges.add(edgeKey);

        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", c.pos[0].toString());
        line.setAttribute("y1", c.pos[1].toString());
        line.setAttribute("x2", n.pos[0].toString());
        line.setAttribute("y2", n.pos[1].toString());
        const isInterRegion = (c.region !== n.region);
        line.setAttribute("stroke", isInterRegion ? "#DC2626" : "#475569");
        line.setAttribute("stroke-width", isInterRegion ? "1.6" : "1.2");
        line.setAttribute("stroke-dasharray", isInterRegion ? "3,3" : "2,2");
        linesGroup.appendChild(line);
      });

      // Superpower Adjacency Edges
      if (c.superpower_adjacent) {
        const sp = superpowers.find(s => s.id === c.superpower_adjacent);
        if (sp) {
          if (c.name === "Japan") {
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", `M ${c.pos[0]} ${c.pos[1]} C 960 210, 960 500, 480 600 S 70 320, ${sp.pos[0]} ${sp.pos[1]}`);
            path.setAttribute("stroke", "#3B82F6");
            path.setAttribute("stroke-width", "2.0");
            path.setAttribute("stroke-dasharray", "4,4");
            path.setAttribute("fill", "none");
            path.setAttribute("opacity", "0.8");
            linesGroup.appendChild(path);
          } else {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", c.pos[0].toString());
            line.setAttribute("y1", c.pos[1].toString());
            line.setAttribute("x2", sp.pos[0].toString());
            line.setAttribute("y2", sp.pos[1].toString());
            line.setAttribute("stroke", sp.stroke);
            line.setAttribute("stroke-width", "2.2");
            line.setAttribute("opacity", "0.9");
            linesGroup.appendChild(line);
          }
        }
      }
    });
    this.svg.appendChild(linesGroup);

    // 2. Superpower Connection Boxes (US on west, USSR on east)
    const spGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    spGroup.setAttribute("id", "superpowers");

    superpowers.forEach(sp => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const w = 110, h = 28;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", (sp.pos[0] - w/2).toString());
      rect.setAttribute("y", (sp.pos[1] - h/2).toString());
      rect.setAttribute("width", w.toString());
      rect.setAttribute("height", h.toString());
      rect.setAttribute("rx", "4");
      rect.setAttribute("fill", sp.color);
      rect.setAttribute("stroke", sp.stroke);
      rect.setAttribute("stroke-width", "2.0");
      g.appendChild(rect);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", sp.pos[0].toString());
      text.setAttribute("y", (sp.pos[1] + 3.5).toString());
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill", "#FFFFFF");
      text.setAttribute("font-size", "10");
      text.setAttribute("font-weight", "900");
      text.setAttribute("letter-spacing", "1");
      text.textContent = sp.label;
      g.appendChild(text);
      spGroup.appendChild(g);
    });
    this.svg.appendChild(spGroup);

    // 3. Country Nodes
    const legalNodes = new Set(
      state.legal_actions && state.legal_actions.decision_type === 5
        ? (state.legal_actions.valid_ids || []).filter((id: number) => id < 84)
        : []
    );

    const nodesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    nodesGroup.setAttribute("id", "country-nodes");

    countries.forEach(c => {
      const stateC = state.countries ? state.countries[c.name] : null;
      const usInf = stateC ? stateC.us_influence : 0;
      const ussrInf = stateC ? stateC.ussr_influence : 0;
      const controlledBy = stateC ? stateC.controlled_by : "NONE";
      const isLegal = legalNodes.has(c.id);

      const boxW = 34;
      const boxH = 19;
      const x = c.pos[0];
      const y = c.pos[1];

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", `svg-country-node ${isLegal ? "legal-target" : ""}`);
      g.setAttribute("data-id", c.id.toString());
      g.setAttribute("data-name", c.name);

      // Card Background
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", "country-card-bg");
      rect.setAttribute("x", (x - boxW/2).toString());
      rect.setAttribute("y", (y - boxH/2).toString());
      rect.setAttribute("width", boxW.toString());
      rect.setAttribute("height", boxH.toString());
      rect.setAttribute("rx", "2.5");

      // Base border depending on Battleground & Control
      let borderColor = c.battleground ? "#A855F7" : "#334155";
      let strokeWidth = c.battleground ? "1.8" : "1.0";

      if (controlledBy === "US") {
        borderColor = "#3B82F6";
        strokeWidth = "2.4";
      } else if (controlledBy === "USSR") {
        borderColor = "#EF4444";
        strokeWidth = "2.4";
      }

      rect.setAttribute("fill", c.battleground ? "#1E1B4B" : "#0F172A");
      rect.setAttribute("stroke", borderColor);
      rect.setAttribute("stroke-width", strokeWidth);
      g.appendChild(rect);

      // Battleground Purple Top Stripe
      if (c.battleground) {
        const bgBanner = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        bgBanner.setAttribute("x", (x - boxW/2 + 0.8).toString());
        bgBanner.setAttribute("y", (y - boxH/2 + 0.8).toString());
        bgBanner.setAttribute("width", (boxW - 1.6).toString());
        bgBanner.setAttribute("height", "3.0");
        bgBanner.setAttribute("rx", "1.2");
        bgBanner.setAttribute("fill", "#7C3AED");
        g.appendChild(bgBanner);
      }

      // Stability Badge (Upper Right Corner)
      const badgeW = 6.2, badgeH = 5.2;
      const badgeX = x + boxW/2 - badgeW - 0.8;
      const badgeY = y - boxH/2 + 0.8;
      const stabRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      stabRect.setAttribute("x", badgeX.toString());
      stabRect.setAttribute("y", badgeY.toString());
      stabRect.setAttribute("width", badgeW.toString());
      stabRect.setAttribute("height", badgeH.toString());
      stabRect.setAttribute("rx", "1.2");
      stabRect.setAttribute("fill", c.battleground ? "#5B21B6" : "#1E293B");
      stabRect.setAttribute("stroke", c.battleground ? "#A855F7" : "#475569");
      stabRect.setAttribute("stroke-width", "0.6");
      g.appendChild(stabRect);

      const stabText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      stabText.setAttribute("x", (badgeX + badgeW/2).toString());
      stabText.setAttribute("y", (badgeY + badgeH/2 + 1.2).toString());
      stabText.setAttribute("text-anchor", "middle");
      stabText.setAttribute("fill", "#FFFFFF");
      stabText.setAttribute("font-size", "3.6");
      stabText.setAttribute("font-weight", "bold");
      stabText.textContent = c.stability.toString();
      g.appendChild(stabText);

      // Country Name Text
      const nameText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      nameText.setAttribute("x", (x - boxW/2 + 1.6).toString());
      nameText.setAttribute("y", (y - boxH/2 + (c.battleground ? 7.2 : 5.4)).toString());
      nameText.setAttribute("fill", c.battleground ? "#FAF5FF" : "#F1F5F9");
      nameText.setAttribute("font-size", c.name.length > 12 ? "2.8" : "3.2");
      nameText.setAttribute("font-weight", "bold");
      nameText.textContent = c.name.length > 13 ? c.name.substring(0, 11) + ".." : c.name;
      g.appendChild(nameText);

      // Influence Counters (Bottom Bar)
      // USSR (Red) Box on Left
      const infBoxW = 8.8, infBoxH = 5.6;
      const ussrBoxX = x - boxW/2 + 1.6;
      const ussrBoxY = y + boxH/2 - infBoxH - 1.2;
      const ussrRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      ussrRect.setAttribute("x", ussrBoxX.toString());
      ussrRect.setAttribute("y", ussrBoxY.toString());
      ussrRect.setAttribute("width", infBoxW.toString());
      ussrRect.setAttribute("height", infBoxH.toString());
      ussrRect.setAttribute("rx", "1.0");
      ussrRect.setAttribute("fill", ussrInf > 0 ? "#DC2626" : "#0F172A");
      ussrRect.setAttribute("stroke", "#EF4444");
      ussrRect.setAttribute("stroke-width", ussrInf > 0 ? "1.0" : "0.6");
      g.appendChild(ussrRect);

      const ussrText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      ussrText.setAttribute("x", (ussrBoxX + infBoxW/2).toString());
      ussrText.setAttribute("y", (ussrBoxY + infBoxH/2 + 1.2).toString());
      ussrText.setAttribute("text-anchor", "middle");
      ussrText.setAttribute("fill", "#FFFFFF");
      ussrText.setAttribute("font-size", "3.4");
      ussrText.setAttribute("font-weight", "bold");
      ussrText.textContent = ussrInf.toString();
      g.appendChild(ussrText);

      // US (Blue) Box on Right of bottom
      const usBoxX = ussrBoxX + infBoxW + 2.0;
      const usBoxY = ussrBoxY;
      const usRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      usRect.setAttribute("x", usBoxX.toString());
      usRect.setAttribute("y", usBoxY.toString());
      usRect.setAttribute("width", infBoxW.toString());
      usRect.setAttribute("height", infBoxH.toString());
      usRect.setAttribute("rx", "1.0");
      usRect.setAttribute("fill", usInf > 0 ? "#2563EB" : "#0F172A");
      usRect.setAttribute("stroke", "#3B82F6");
      usRect.setAttribute("stroke-width", usInf > 0 ? "1.0" : "0.6");
      g.appendChild(usRect);

      const usText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      usText.setAttribute("x", (usBoxX + infBoxW/2).toString());
      usText.setAttribute("y", (usBoxY + infBoxH/2 + 1.2).toString());
      usText.setAttribute("text-anchor", "middle");
      usText.setAttribute("fill", "#FFFFFF");
      usText.setAttribute("font-size", "3.4");
      usText.setAttribute("font-weight", "bold");
      usText.textContent = usInf.toString();
      g.appendChild(usText);

      // Click Event Handler
      g.addEventListener("click", () => {
        this.onCountryClick(c.id);
      });

      nodesGroup.appendChild(g);
    });

    this.svg.appendChild(nodesGroup);
  }
}
