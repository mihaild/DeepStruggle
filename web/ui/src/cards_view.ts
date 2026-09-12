import { GameState, CardMetadata } from "./types";

export interface EffectInfo {
  name: string;
  cardId?: number;
  cardName: string;
  side: "US" | "USSR" | "BOTH" | "NEUTRAL";
  duration: "PERMANENT" | "TURN" | "CONDITIONAL" | "SPACE_PERK";
  description: string;
}

export const EFFECT_INFO_MAP: Record<string, EffectInfo> = {
  NATO_ACTIVE: {
    name: "NATO Active",
    cardId: 21,
    cardName: "NATO",
    side: "US",
    duration: "PERMANENT",
    description: "USSR cannot Coup or Realign in US-controlled Western European countries."
  },
  NATO_CANCELED_FRANCE: {
    name: "NATO (France Excluded)",
    cardId: 17,
    cardName: "De Gaulle Leads France",
    side: "NEUTRAL",
    duration: "PERMANENT",
    description: "NATO no longer protects France."
  },
  NATO_CANCELED_WEST_GERMANY: {
    name: "NATO (W. Germany Excluded)",
    cardId: 55,
    cardName: "Willy Brandt",
    side: "NEUTRAL",
    duration: "CONDITIONAL",
    description: "NATO no longer protects West Germany (canceled by Tear Down This Wall)."
  },
  MARSHALL_PLAN_PLAYED: {
    name: "Marshall Plan Played",
    cardId: 23,
    cardName: "Marshall Plan",
    side: "US",
    duration: "PERMANENT",
    description: "Prerequisite fulfilled for US to activate NATO."
  },
  WARSAW_PACT_PLAYED: {
    name: "Warsaw Pact Played",
    cardId: 16,
    cardName: "Warsaw Pact Formed",
    side: "USSR",
    duration: "PERMANENT",
    description: "Prerequisite fulfilled for US to activate NATO."
  },
  US_JAPAN_PACT_ACTIVE: {
    name: "US/Japan Defense Pact",
    cardId: 27,
    cardName: "US/Japan Mutual Defense Pact",
    side: "US",
    duration: "PERMANENT",
    description: "USSR cannot Coup or Realign in Japan. Japan is US-controlled."
  },
  CONTAINMENT_ACTIVE: {
    name: "Containment Active",
    cardId: 25,
    cardName: "Containment",
    side: "US",
    duration: "TURN",
    description: "All US Operations receive +1 Op (capped at 4 Ops)."
  },
  PURGE_US_ACTIVE: {
    name: "Red Scare (US)",
    cardId: 31,
    cardName: "Red Scare/Purge",
    side: "US",
    duration: "TURN",
    description: "All US Operations receive -1 Op (minimum 1 Op)."
  },
  PURGE_USSR_ACTIVE: {
    name: "Red Scare (USSR)",
    cardId: 31,
    cardName: "Red Scare/Purge",
    side: "USSR",
    duration: "TURN",
    description: "All USSR Operations receive -1 Op (minimum 1 Op)."
  },
  VIETNAM_REVOLTS_ACTIVE: {
    name: "Vietnam Revolts",
    cardId: 9,
    cardName: "Vietnam Revolts",
    side: "USSR",
    duration: "TURN",
    description: "All USSR Operations in Southeast Asia receive +1 Op for remainder of turn."
  },
  FORMOSAN_RESOLUTION_ACTIVE: {
    name: "Formosan Resolution",
    cardId: 35,
    cardName: "Formosan Resolution",
    side: "US",
    duration: "CONDITIONAL",
    description: "Taiwan counts as a Battleground country for Asia Scoring until China Card played."
  },
  CMC_ACTIVE_US: {
    name: "Cuban Missile Crisis (US)",
    cardId: 40,
    cardName: "Cuban Missile Crisis",
    side: "US",
    duration: "TURN",
    description: "US cannot perform Coups anywhere unless removing 2 influence from W. Germany/Turkey."
  },
  CMC_ACTIVE_USSR: {
    name: "Cuban Missile Crisis (USSR)",
    cardId: 40,
    cardName: "Cuban Missile Crisis",
    side: "USSR",
    duration: "TURN",
    description: "USSR cannot perform Coups anywhere unless removing 2 influence from Cuba."
  },
  NUCLEAR_SUBS_ACTIVE: {
    name: "Nuclear Subs Active",
    cardId: 41,
    cardName: "Nuclear Subs",
    side: "US",
    duration: "TURN",
    description: "US Coup Attempts in battleground countries do not degrade DEFCON this turn."
  },
  QUAGMIRE_ACTIVE: {
    name: "Quagmire Active",
    cardId: 42,
    cardName: "Quagmire",
    side: "US",
    duration: "CONDITIONAL",
    description: "US must discard a 2+ Ops card and roll 5-6 on each Action Round to exit."
  },
  BEAR_TRAP_ACTIVE: {
    name: "Bear Trap Active",
    cardId: 44,
    cardName: "Bear Trap",
    side: "USSR",
    duration: "CONDITIONAL",
    description: "USSR must discard a 2+ Ops card and roll 5-6 on each Action Round to exit."
  },
  SALT_ACTIVE: {
    name: "SALT Negotiations",
    cardId: 43,
    cardName: "SALT Negotiations",
    side: "BOTH",
    duration: "TURN",
    description: "All Coup Attempts by both players suffer -1 to die rolls for remainder of turn."
  },
  WE_WILL_BURY_YOU_PENDING: {
    name: "We Will Bury You (Pending)",
    cardId: 50,
    cardName: "We Will Bury You",
    side: "USSR",
    duration: "TURN",
    description: "USSR gains 3 VP at Turn End unless UN Intervention is played by US."
  },
  BREZHNEV_DOCTRINE_ACTIVE: {
    name: "Brezhnev Doctrine",
    cardId: 51,
    cardName: "Brezhnev Doctrine",
    side: "USSR",
    duration: "TURN",
    description: "All USSR Operations receive +1 Op (capped at 4 Ops)."
  },
  FLOWER_POWER_ACTIVE: {
    name: "Flower Power Active",
    cardId: 59,
    cardName: "Flower Power",
    side: "USSR",
    duration: "CONDITIONAL",
    description: "USSR gains 2 VP for each US War card played for Operations (canceled by An Evil Empire)."
  },
  U2_INCIDENT_ACTIVE: {
    name: "U-2 Incident",
    cardId: 60,
    cardName: "U-2 Incident",
    side: "USSR",
    duration: "TURN",
    description: "USSR gains 1 VP immediately; +1 additional VP if UN Intervention played this turn."
  },
  SHUTTLE_DIPLOMACY_ACTIVE: {
    name: "Shuttle Diplomacy",
    cardId: 73,
    cardName: "Shuttle Diplomacy",
    side: "US",
    duration: "CONDITIONAL",
    description: "Subtracts 1 USSR Battleground from next Asia or Middle East Scoring."
  },
  DEATH_SQUADS_US: {
    name: "Death Squads (US)",
    cardId: 69,
    cardName: "Latin American Death Squads",
    side: "US",
    duration: "TURN",
    description: "US receives +1 to all Realignment rolls in Central and South America this turn."
  },
  DEATH_SQUADS_USSR: {
    name: "Death Squads (USSR)",
    cardId: 69,
    cardName: "Latin American Death Squads",
    side: "USSR",
    duration: "TURN",
    description: "USSR receives +1 to all Realignment rolls in Central and South America this turn."
  },
  CAMP_DAVID_PLAYED: {
    name: "Camp David Accords",
    cardId: 65,
    cardName: "Camp David Accords",
    side: "US",
    duration: "PERMANENT",
    description: "Prevents Arab-Israeli War from being played as an Event."
  },
  NORTH_SEA_OIL_ACTIVE: {
    name: "North Sea Oil",
    cardId: 86,
    cardName: "North Sea Oil",
    side: "US",
    duration: "TURN",
    description: "Grants US an 8th Action Round for the current turn only."
  },
  THE_REFORMER_PLAYED: {
    name: "The Reformer",
    cardId: 87,
    cardName: "The Reformer",
    side: "US",
    duration: "PERMANENT",
    description: "USSR can no longer Coup in Europe. Enables #90 Glasnost to be used for Operations."
  },
  IRAN_CONTRA_ACTIVE: {
    name: "Iran-Contra Scandal",
    cardId: 93,
    cardName: "Iran-Contra Scandal",
    side: "USSR",
    duration: "TURN",
    description: "All US Realignment rolls receive -1 to their die roll for remainder of turn."
  },
  EVIL_EMPIRE_PLAYED: {
    name: "An Evil Empire",
    cardId: 97,
    cardName: "“An Evil Empire”",
    side: "US",
    duration: "PERMANENT",
    description: "Cancels and prevents the effects of #59 Flower Power."
  },
  ALDRICH_AMES_ACTIVE: {
    name: "Aldrich Ames Remix",
    cardId: 98,
    cardName: "Aldrich Ames Remix",
    side: "USSR",
    duration: "TURN",
    description: "US hand of cards remains revealed face-up to USSR for remainder of turn."
  },
  JOHN_PAUL_II_PLAYED: {
    name: "John Paul II Elected Pope",
    cardId: 68,
    cardName: "John Paul II Elected Pope",
    side: "US",
    duration: "PERMANENT",
    description: "Prerequisite fulfilled enabling #101 Solidarity to be played as an Event."
  },
  NORAD_ACTIVE: {
    name: "NORAD Active",
    cardId: 106,
    cardName: "NORAD",
    side: "US",
    duration: "CONDITIONAL",
    description: "When Canada is US-controlled, US adds 1 Influence whenever DEFCON reaches 2 in an AR."
  },
  YURI_AND_SAMANTHA_ACTIVE: {
    name: "Yuri and Samantha",
    cardId: 109,
    cardName: "Yuri and Samantha",
    side: "USSR",
    duration: "TURN",
    description: "USSR gains 1 VP for each US Coup Attempt performed during remainder of turn."
  },
  AWACS_PLAYED: {
    name: "AWACS Sale to Saudis",
    cardId: 110,
    cardName: "AWACS Sale to Saudis",
    side: "US",
    duration: "PERMANENT",
    description: "Prevents #56 Muslim Revolution from being played as an Event."
  },
  IRANIAN_HOSTAGE_CRISIS_PLAYED: {
    name: "Iranian Hostage Crisis",
    cardId: 82,
    cardName: "Iranian Hostage Crisis",
    side: "USSR",
    duration: "PERMANENT",
    description: "Forces US to randomly discard 2 cards instead of 1 when #92 Terrorism is played."
  },
  WILLY_BRANDT_PLAYED: {
    name: "Willy Brandt Played",
    cardId: 55,
    cardName: "Willy Brandt",
    side: "USSR",
    duration: "CONDITIONAL",
    description: "Cancels NATO for West Germany (canceled by Tear Down This Wall)."
  },
  TEAR_DOWN_THIS_WALL_PLAYED: {
    name: "Tear Down this Wall",
    cardId: 96,
    cardName: "Tear Down this Wall",
    side: "US",
    duration: "PERMANENT",
    description: "Cancels and prevents the effects of #55 Willy Brandt."
  },
  CHERNOBYL_ACTIVE: {
    name: "Chernobyl Active",
    cardId: 94,
    cardName: "Chernobyl",
    side: "US",
    duration: "TURN",
    description: "USSR cannot add Influence using Operations points in the forbidden region this turn."
  }
};

export class CardsView {
  private cardsMeta: Map<number, CardMetadata> = new Map();
  private onCardClick: (cardId: number) => void;
  private tooltipEl: HTMLElement;
  private locationFilter: string = "ALL";
  private searchFilter: string = "";
  private lastState: GameState | null = null;

  constructor(onCardClick: (cardId: number) => void) {
    this.onCardClick = onCardClick;
    this.tooltipEl = document.getElementById("card-tooltip")!;
    this.fetchMetadata();
    this.setupTabs();
    this.setupFilters();
  }

  public setCardsMetadata(data: any) {
    const cards: CardMetadata[] = Array.isArray(data) ? data : (data.cards || []);
    cards.forEach((c: CardMetadata) => {
      this.cardsMeta.set(c.id, c);
    });
    if (this.lastState) {
      this.render(this.lastState);
    }
  }

  public async fetchMetadata() {
    try {
      const res = await fetch("/api/metadata/cards");
      if (!res.ok) return;
      const data = await res.json();
      this.setCardsMetadata(data);
    } catch (e) {
      console.warn("Could not fetch card metadata:", e);
    }
  }

  private setupTabs() {
    const tabButtons = document.querySelectorAll(".tabs .tab-btn");
    tabButtons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        const targetTab = (e.currentTarget as HTMLElement).getAttribute("data-tab");
        tabButtons.forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));

        (e.currentTarget as HTMLElement).classList.add("active");
        if (targetTab) {
          document.getElementById(targetTab)?.classList.add("active");
        }
      });
    });
  }

  private setupFilters() {
    const searchInput = document.getElementById("all-cards-search") as HTMLInputElement;
    const filterSelect = document.getElementById("all-cards-filter") as HTMLSelectElement;

    searchInput?.addEventListener("input", (e) => {
      this.searchFilter = (e.target as HTMLInputElement).value.trim().toLowerCase();
      const state = (window as any).__CURRENT_STATE__;
      if (state) this.renderAllCardsList(state);
    });

    filterSelect?.addEventListener("change", (e) => {
      this.locationFilter = (e.target as HTMLSelectElement).value;
      const state = (window as any).__CURRENT_STATE__;
      if (state) this.renderAllCardsList(state);
    });
  }

  public render(state: GameState) {
    this.lastState = state;
    (window as any).__CURRENT_STATE__ = state;

    // Render Hand Counts in Tab Titles
    const ussrCount = (state.hands?.USSR || []).length;
    const usCount = (state.hands?.US || []).length;
    const discardCount = (state.discard_pile || []).length;
    const removedCount = (state.removed_pile || []).length;

    const china = state.china_card || { holder: "USSR", playable: true };
    const ussrCountEl = document.getElementById("ussr-hand-count");
    if (ussrCountEl) {
      ussrCountEl.textContent = china.holder === "USSR" ? `${ussrCount} + 🇨🇳` : ussrCount.toString();
    }

    const usCountEl = document.getElementById("us-hand-count");
    if (usCountEl) {
      usCountEl.textContent = china.holder === "US" ? `${usCount} + 🇨🇳` : usCount.toString();
    }

    const discardCountEl = document.getElementById("discard-count");
    if (discardCountEl) discardCountEl.textContent = discardCount.toString();

    const removedCountEl = document.getElementById("removed-count");
    if (removedCountEl) removedCountEl.textContent = removedCount.toString();

    const drawCountEl = document.getElementById("draw-count");
    if (drawCountEl) drawCountEl.textContent = (state.draw_deck_count || 0).toString();

    // Render Hand Lists
    this.renderHand("tab-ussr-hand", state.hands?.USSR || [], "USSR", state);
    this.renderHand("tab-us-hand", state.hands?.US || [], "US", state);

    // Render China Card Box
    this.renderChinaCard(state);

    // Render All Cards Explorer List
    this.renderAllCardsList(state);

    // Render Active Effects and Context Stack
    this.renderFlags(state);
    this.renderContextStack(state);
  }

  private renderHand(containerId: string, cardIds: number[], player: "US" | "USSR", state: GameState) {
    const tabEl = document.getElementById(containerId);
    if (!tabEl) return;
    const container = tabEl.querySelector(".card-list");
    if (!container) return;

    container.innerHTML = "";

    const china = state.china_card || { holder: "USSR", playable: true };
    const holdsChina = china.holder === player;
    const isCurrentDecision = state.decision_context?.decision_player === player && state.decision_context?.decision_type === 1;
    const legalCards = new Set(state.legal_actions?.valid_ids || []);

    // 1. Render China Card at top of hand if held by this player (Single-line)
    if (holdsChina) {
      const isChinaLegal = isCurrentDecision && (legalCards.has(6) || legalCards.size === 0) && china.playable;
      const chinaEl = document.createElement("div");
      chinaEl.className = `china-hand-card ${china.playable ? "face-up" : "face-down"} ${isChinaLegal ? "playable" : ""}`;

      const meta = this.cardsMeta.get(6) || {
        id: 6,
        name: "The China Card",
        ops: 4,
        side: "neutral",
        age: "early war",
        description: "May be played as a normal 4 Ops card. If all Ops are spent in Asia, receive +1 Op (5 Ops total). Pass to opponent face down when played.",
        one_time: false
      };

      chinaEl.innerHTML = `
        <div class="card-item-era-bar era-early" style="background: #F59E0B;"></div>
        <div class="card-ops-badge neutral">4</div>
        <div class="card-name-single">
          🇨🇳 #6 The China Card
        </div>
        <span class="china-status-pill ${china.playable ? "playable" : "face-down"}">
          ${china.playable ? "✓ Face Up" : "✕ Face Down"}
        </span>
      `;

      chinaEl.addEventListener("mouseenter", (e) => this.showTooltip(e, meta));
      chinaEl.addEventListener("mouseleave", () => this.hideTooltip());
      if (isChinaLegal) {
        chinaEl.addEventListener("click", () => {
          this.onCardClick(6);
        });
      }

      container.appendChild(chinaEl);
    }

    if (cardIds.length === 0 && !holdsChina) {
      container.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 20px; font-size: 11px;">Hand is empty</div>`;
      return;
    }

    cardIds.forEach(id => {
      const meta = this.cardsMeta.get(id) || {
        id,
        name: `Card #${id}`,
        ops: 0,
        side: "neutral",
        age: "early war",
        description: "",
        one_time: false
      };

      const isLegal = isCurrentDecision && legalCards.has(id);
      const cardEl = document.createElement("div");
      cardEl.className = `card-item ${meta.side} ${isLegal ? "playable" : ""}`;

      const eraClass = meta.age === "early war" ? "era-early" : (meta.age === "mid war" ? "era-mid" : "era-late");

      cardEl.innerHTML = `
        <div class="card-item-era-bar ${eraClass}"></div>
        <div class="card-ops-badge ${meta.side}">
          ${meta.ops}${meta.one_time ? '<span class="card-star" title="Remove after event">★</span>' : ''}
        </div>
        <div class="card-name-single">
          #${meta.id} ${meta.name}
        </div>
      `;

      cardEl.addEventListener("mouseenter", (e) => this.showTooltip(e, meta));
      cardEl.addEventListener("mouseleave", () => this.hideTooltip());
      cardEl.addEventListener("click", () => {
        if (isLegal || legalCards.size === 0) {
          this.onCardClick(id);
        }
      });

      container.appendChild(cardEl);
    });
  }

  private renderAllCardsList(state: GameState) {
    const container = document.getElementById("all-cards-list");
    if (!container) return;

    const locations = (state.card_locations || {}) as Record<string, string>;
    container.innerHTML = "";

    let matchedCount = 0;
    for (let id = 1; id <= 110; id++) {
      const meta = this.cardsMeta.get(id) || {
        id,
        name: `Card #${id}`,
        ops: 0,
        side: "neutral",
        age: "early war",
        description: "",
        one_time: false
      };

      const loc = locations[id.toString()] || "DRAW_DECK";

      if (this.locationFilter !== "ALL" && loc !== this.locationFilter) {
        continue;
      }

      if (this.searchFilter) {
        const nameMatch = meta.name.toLowerCase().includes(this.searchFilter);
        const idMatch = id.toString() === this.searchFilter;
        if (!nameMatch && !idMatch) continue;
      }

      matchedCount++;
      const itemEl = document.createElement("div");
      itemEl.className = "all-cards-item";

      const locBadgeClass = this.getLocationBadgeClass(loc);
      const locLabel = this.getLocationLabel(loc);
      const eraClass = meta.age === "early war" ? "era-early" : (meta.age === "mid war" ? "era-mid" : "era-late");

      itemEl.innerHTML = `
        <div class="card-item-era-bar ${eraClass}"></div>
        <div class="all-cards-item-left">
          <div class="card-ops-badge ${meta.side}">${meta.ops}</div>
          <div class="all-cards-info">
            <div class="card-name">#${meta.id} ${meta.name} ${meta.one_time ? "★" : ""}</div>
            <div class="card-meta">${meta.age.toUpperCase()} • ${meta.side.toUpperCase()}</div>
          </div>
        </div>
        <div class="card-loc-badge ${locBadgeClass}">${locLabel}</div>
      `;

      itemEl.addEventListener("mouseenter", (e) => this.showTooltip(e, meta));
      itemEl.addEventListener("mouseleave", () => this.hideTooltip());
      itemEl.addEventListener("click", () => {
        this.showPileModal(`Card #${meta.id}: ${meta.name}`, [id]);
      });

      container.appendChild(itemEl);
    }

    if (matchedCount === 0) {
      container.innerHTML = "<div style=\"color: var(--text-dim); text-align: center; padding: 20px;\">No cards match your filter.</div>";
    }
  }

  private getLocationBadgeClass(loc: string): string {
    switch (loc) {
      case "HAND_US": return "loc-us";
      case "HAND_USSR": return "loc-ussr";
      case "DRAW_DECK": return "loc-deck";
      case "DISCARD_PILE": return "loc-discard";
      case "REMOVED_FROM_GAME": return "loc-removed";
      case "ONGOING_EVENT": return "loc-ongoing";
      case "UNAVAILABLE": return "loc-future";
      default: return "loc-default";
    }
  }

  private getLocationLabel(loc: string): string {
    switch (loc) {
      case "HAND_US": return "US Hand";
      case "HAND_USSR": return "USSR Hand";
      case "DRAW_DECK": return "Draw Deck";
      case "DISCARD_PILE": return "Discard";
      case "REMOVED_FROM_GAME": return "Removed";
      case "ONGOING_EVENT": return "Ongoing";
      case "UNAVAILABLE": return "Future Era";
      default: return loc;
    }
  }

  private renderChinaCard(state: GameState) {
    const chinaContainer = document.getElementById("china-card-container");
    if (!chinaContainer) return;

    const china = state.china_card || { holder: "USSR", playable: true };
    const holder = china.holder || "USSR";
    const isPlayable = china.playable;

    chinaContainer.innerHTML = `
      <div class="china-card-box ${isPlayable ? "face-up" : "face-down"}">
        <div>
          <div style="font-weight: 800; font-size: 13px;">#6 THE CHINA CARD (4 Ops)</div>
          <div style="font-size: 11px; opacity: 0.9;">Holder: <strong>${holder}</strong> (${isPlayable ? "Face Up / Playable" : "Face Down"})</div>
        </div>
        <div style="font-weight: bold; font-size: 11px; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">
          +1 Op in Asia
        </div>
      </div>
    `;
  }

  private renderFlags(state: GameState) {
    const flagsContainer = document.getElementById("active-flags-container");
    if (!flagsContainer) return;

    flagsContainer.innerHTML = "";
    const flags = state.flags || [];
    const effectCards: Array<{ title: string; side: string; badge: string; desc: string }> = [];

    // 1. Check persistent flags
    flags.forEach(f => {
      const info = EFFECT_INFO_MAP[f];
      if (info) {
        effectCards.push({
          title: info.name,
          side: info.side,
          badge: info.duration === "TURN" ? "Turn Only" : (info.duration === "PERMANENT" ? "Permanent" : "Conditional"),
          desc: info.description
        });
      } else {
        effectCards.push({
          title: f.replace(/_ACTIVE|_PLAYED/, ""),
          side: "NEUTRAL",
          badge: "Active",
          desc: `Continuous effect for ${f}`
        });
      }
    });

    // 2. Space Race special perks
    const usSpace = state.space?.US || 0;
    const ussrSpace = state.space?.USSR || 0;

    // A space perk is held only while the opponent has not reached the same box: once
    // both sides are on it, neither side has the ability. The engine already works this
    // way (SpaceRace::has_animal_in_space and friends all require opp_track < box), so
    // testing only one's own track showed perks that were not actually in effect --
    // "US may attempt Space Race twice" with the USSR level with them, for instance.
    if (usSpace >= 2 && ussrSpace < 2) effectCards.push({ title: "US: Animal in Space (#2)", side: "US", badge: "Space Perk", desc: "US may attempt Space Race twice per turn." });
    if (usSpace >= 4 && ussrSpace < 4) effectCards.push({ title: "US: Lunar Orbit (#4)", side: "US", badge: "Space Perk", desc: "USSR must select & reveal headline card before US chooses." });
    if (usSpace >= 6 && ussrSpace < 6) effectCards.push({ title: "US: Space Walk (#6)", side: "US", badge: "Space Perk", desc: "US may discard 1 held card at end of turn." });
    if (usSpace >= 8 && ussrSpace < 8) effectCards.push({ title: "US: Moon Landing (#8)", side: "US", badge: "Space Perk", desc: "US gains an 8th Action Round each turn." });

    if (ussrSpace >= 2 && usSpace < 2) effectCards.push({ title: "USSR: Animal in Space (#2)", side: "USSR", badge: "Space Perk", desc: "USSR may attempt Space Race twice per turn." });
    if (ussrSpace >= 4 && usSpace < 4) effectCards.push({ title: "USSR: Lunar Orbit (#4)", side: "USSR", badge: "Space Perk", desc: "US must select & reveal headline card before USSR chooses." });
    if (ussrSpace >= 6 && usSpace < 6) effectCards.push({ title: "USSR: Space Walk (#6)", side: "USSR", badge: "Space Perk", desc: "USSR may discard 1 held card at end of turn." });
    if (ussrSpace >= 8 && usSpace < 8) effectCards.push({ title: "USSR: Moon Landing (#8)", side: "USSR", badge: "Space Perk", desc: "USSR gains an 8th Action Round each turn." });

    if (effectCards.length === 0) {
      flagsContainer.innerHTML = "<div style=\"font-size: 11px; color: var(--text-dim); text-align: center; padding: 10px;\">No active continuous effects</div>";
      return;
    }

    effectCards.forEach(eff => {
      const cardEl = document.createElement("div");
      const sideClass = eff.side.toLowerCase();
      cardEl.className = `active-effect-card side-${sideClass}`;
      cardEl.innerHTML = `
        <div class="effect-header">
          <span class="effect-title">${eff.title}</span>
          <span class="effect-badge badge-${sideClass}">${eff.badge}</span>
        </div>
        <div class="effect-desc">${eff.desc}</div>
      `;
      flagsContainer.appendChild(cardEl);
    });
  }

  private renderContextStack(state: GameState) {
    const stackContainer = document.getElementById("context-stack-container");
    if (!stackContainer) return;

    const ctx = state.decision_context;
    if (!ctx) return;

    stackContainer.innerHTML = `
      <div><strong>Reentrancy Depth:</strong> ${ctx.stack_depth || 0} / 2</div>
      <div><strong>Decision:</strong> ${ctx.decision_type_name || ctx.decision_type} (${ctx.decision_player})</div>
      ${ctx.pending_op_card ? `<div><strong>Pending Card:</strong> #${ctx.pending_op_card} (Ops: ${ctx.pending_ops_value})</div>` : ""}
      ${ctx.resolving_card ? `<div><strong>Resolving Card:</strong> #${ctx.resolving_card} (${ctx.resolving_card_name || ""})</div>` : ""}
      ${ctx.remaining_steps ? `<div><strong>Remaining Steps:</strong> ${ctx.remaining_steps}</div>` : ""}
    `;
  }

  private showTooltip(e: MouseEvent, meta: CardMetadata) {
    this.tooltipEl.innerHTML = `
      <div style="font-weight: bold; font-size: 13px; color: #F8FAFC; margin-bottom: 4px;">
        #${meta.id} ${meta.name} ${meta.one_time ? "★" : ""}
      </div>
      <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">
        ${meta.age.toUpperCase()} • ${meta.side.toUpperCase()} • ${meta.ops} OPS
      </div>
      <div style="font-size: 12px; color: #E2E8F0; line-height: 1.4;">
        ${meta.description}
      </div>
    `;

    this.tooltipEl.classList.remove("hidden");
    const x = Math.min(window.innerWidth - 300, e.clientX + 15);
    const y = Math.min(window.innerHeight - 200, e.clientY + 15);
    this.tooltipEl.style.left = `${x}px`;
    this.tooltipEl.style.top = `${y}px`;
  }

  private hideTooltip() {
    this.tooltipEl.classList.add("hidden");
  }

  public showPileModal(title: string, cardIds: number[]) {
    const modal = document.getElementById("modal-container");
    const titleEl = document.getElementById("modal-title");
    const bodyEl = document.getElementById("modal-body");
    if (!modal || !titleEl || !bodyEl) return;

    titleEl.textContent = `${title} (${cardIds.length} cards)`;
    bodyEl.innerHTML = "";

    if (cardIds.length === 0) {
      bodyEl.innerHTML = "<div style=\"color: var(--text-muted); text-align: center; padding: 20px;\">No cards in this pile</div>";
    } else {
      cardIds.forEach(id => {
        const meta = this.cardsMeta.get(id);
        if (!meta) return;
        const row = document.createElement("div");
        row.style.background = "#111827";
        row.style.padding = "8px 12px";
        row.style.borderRadius = "4px";
        row.style.border = "1px solid rgba(255,255,255,0.06)";
        row.innerHTML = `
          <div style="font-weight: bold; color: #F8FAFC;">#${meta.id} ${meta.name} (${meta.ops} Ops, ${meta.side.toUpperCase()}, ${meta.age.toUpperCase()})</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px; line-height: 1.35;">${meta.description}</div>
        `;
        bodyEl.appendChild(row);
      });
    }

    modal.classList.remove("hidden");
  }
}
