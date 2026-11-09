import { GameState, CardMetadata } from "./types";
import { CARDS_METADATA } from "./metadata";
import { UNSEEN } from "./game/view";
import { engineLayout } from "./engine/layout";
import { escapeHtml, num } from "./html";
import { cardDescription, cardName, sideName, t } from "./i18n";

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
    name: "Cuban Missile Crisis (played by US)",
    cardId: 40,
    cardName: "Cuban Missile Crisis",
    side: "US",
    duration: "TURN",
    description: "A USSR Coup loses the game this turn, unless the USSR cancels the crisis by removing 2 influence from Cuba."
  },
  CMC_ACTIVE_USSR: {
    name: "Cuban Missile Crisis (played by USSR)",
    cardId: 40,
    cardName: "Cuban Missile Crisis",
    side: "USSR",
    duration: "TURN",
    description: "A US Coup loses the game this turn, unless the US cancels the crisis by removing 2 influence from West Germany or Turkey."
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
  // The engine's own name for the bit (bindings/state_json.cpp), shortened there: keyed as
  // `..._PLAYED` this entry never matched and the panel showed the raw flag instead.
  IRANIAN_HOSTAGE_CRISIS_PLAY: {
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

const PILE_SIDE_ORDER: Record<CardMetadata["side"], number> = { ussr: 0, us: 1, neutral: 2 };

/** A card's era as the tooltip and the card text say it ("EARLY WAR"). */
function ageLabel(age: string): string {
  switch (age) {
    case "early war": return t("cards:age.earlyWar");
    case "mid war": return t("cards:age.midWar");
    case "late war": return t("cards:age.lateWar");
    default: return age.toUpperCase();
  }
}

/** A card's side as the tooltip and the card text say it ("USSR"). */
function cardSideLabel(side: string): string {
  switch (side) {
    case "us": return t("cards:side.us");
    case "ussr": return t("cards:side.ussr");
    case "neutral": return t("cards:side.neutral");
    default: return side.toUpperCase();
  }
}

/** A pile's order: by side (USSR, US, Neutral), then highest Ops first, then by name. */
export function sortPile(metas: CardMetadata[]): CardMetadata[] {
  return [...metas].sort((a, b) =>
    PILE_SIDE_ORDER[a.side] - PILE_SIDE_ORDER[b.side] || b.ops - a.ops || a.name.localeCompare(b.name));
}

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

  public fetchMetadata() {
    // Bundled from rules/cards.json (metadata.ts).
    this.setCardsMetadata(CARDS_METADATA);
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

    // Render Hand Counts in Tab Titles. A player's view withholds the other hand's unseen cards
    // (game/view.ts) but not how many there are: hand sizes are public.
    const hidden = state.hidden_cards || { US: 0, USSR: 0 };
    const ussrCount = (state.hands?.USSR || []).length + hidden.USSR;
    const usCount = (state.hands?.US || []).length + hidden.US;
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

    // In a player's view the draw deck and the other side's unseen cards are one pile.
    const opp = this.opponent(state);
    const drawCountEl = document.getElementById("draw-count");
    if (drawCountEl) drawCountEl.textContent = (opp ? state.unseen_count ?? 0 : state.draw_deck_count || 0).toString();
    const drawNameEl = document.getElementById("draw-count-name");
    if (drawNameEl) drawNameEl.textContent = opp ? t("cards:drawDeckPlusHand", { side: sideName(opp) }) : t("cards:drawDeck");
    const deckOption = document.querySelector<HTMLOptionElement>('#all-cards-filter option[value="DRAW_DECK"]');
    if (deckOption) deckOption.textContent = opp ? t("cards:drawDeckOrHand", { side: sideName(opp) }) : t("cards:drawDeck");

    // Render Hand Lists
    this.renderHand("tab-ussr-hand", state.hands?.USSR || [], "USSR", state);
    this.renderHand("tab-us-hand", state.hands?.US || [], "US", state);

    // Render China Card Box
    this.renderChinaCard(state);

    // The piles, drawn as the hands are
    this.renderPile("discard-cards-list", state.discard_pile || []);
    this.renderPile("removed-cards-list", state.removed_pile || []);

    // Render All Cards Explorer List
    this.renderAllCardsList(state);

    // Render Active Effects and Context Stack
    this.renderFlags(state);
    this.renderTurnEffects(state);
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
    const isCurrentDecision = state.decision_context?.decision_player === player && state.decision_context?.decision_type === engineLayout()?.decision_types.SELECT_CARD;
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
          🇨🇳 #6 ${cardName(6, meta.name)}
        </div>
        <span class="china-status-pill ${china.playable ? "playable" : "face-down"}">
          ${china.playable ? t("cards:faceUp") : t("cards:faceDown")}
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

    const hiddenCount = state.hidden_cards?.[player] ?? 0;
    if (cardIds.length === 0 && !holdsChina && hiddenCount === 0) {
      container.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 20px; font-size: 11px;">${t("cards:handEmpty")}</div>`;
      return;
    }

    cardIds.forEach(id => {
      const meta = this.metaOf(id);
      const isLegal = isCurrentDecision && legalCards.has(id);
      const cardEl = this.cardItem(meta, isLegal ? "playable" : "");
      // The replay trace paints this card's probability onto it; that needs the id on the node.
      cardEl.setAttribute("data-card-id", id.toString());
      cardEl.addEventListener("click", () => {
        if (isLegal || legalCards.size === 0) {
          this.onCardClick(id);
        }
      });

      container.appendChild(cardEl);
    });

    // A player's view: the other side's cards it has not seen, face down.
    for (let i = 0; i < hiddenCount; i++) {
      const el = document.createElement("div");
      el.className = "card-item card-hidden";
      el.title = t("cards:hiddenCardTitle");
      el.innerHTML = `
        <div class="card-item-era-bar"></div>
        <div class="card-ops-badge neutral">?</div>
        <div class="card-name-single">${t("cards:hiddenCard")}</div>
      `;
      container.appendChild(el);
    }
  }

  /** A card's metadata, or a stand-in for an id the rules do not list. */
  private metaOf(id: number): CardMetadata {
    return this.cardsMeta.get(id) || {
      id: num(id),
      name: t("cards:cardFallback", { id: num(id) }),
      ops: 0,
      side: "neutral",
      age: "early war",
      description: "",
      one_time: false
    };
  }

  /** A card's line as the hand tabs draw it -- era bar, Ops in the side's colour, number and name --
   *  with its text on hover. */
  private cardItem(meta: CardMetadata, extraClass = ""): HTMLElement {
    const cardEl = document.createElement("div");
    cardEl.className = `card-item ${meta.side} ${extraClass}`;
    const eraClass = meta.age === "early war" ? "era-early" : (meta.age === "mid war" ? "era-mid" : "era-late");
    cardEl.innerHTML = `
      <div class="card-item-era-bar ${eraClass}"></div>
      <div class="card-ops-badge ${meta.side}">
        ${meta.ops}${meta.one_time ? `<span class="card-star" title="${escapeHtml(t("cards:removeAfterEvent"))}">★</span>` : ''}
      </div>
      <div class="card-name-single">
        #${meta.id} ${cardName(meta.id, meta.name)}
      </div>
    `;
    cardEl.addEventListener("mouseenter", (e) => this.showTooltip(e, meta));
    cardEl.addEventListener("mouseleave", () => this.hideTooltip());
    return cardEl;
  }

  /**
   * A pile (the discards, the removed cards) in the Piles tab: each card as the hands draw it,
   * packed, by side -- USSR, US, Neutral, as the tabs run -- then highest Ops first. No
   * `data-card-id`: the model's probabilities are painted on the hands, not on a pile.
   */
  private renderPile(containerId: string, cardIds: number[]) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";
    if (cardIds.length === 0) {
      container.innerHTML = `<div class="pile-empty">${t("cards:pileEmpty")}</div>`;
      return;
    }
    for (const meta of sortPile(cardIds.map(id => this.metaOf(id)))) {
      container.appendChild(this.cardItem(meta, "pile-card"));
    }
  }

  /** The side a player's view hides (null for the developer's view). */
  private opponent(state: GameState): "US" | "USSR" | null {
    return state.view === "US" ? "USSR" : state.view === "USSR" ? "US" : null;
  }

  private renderAllCardsList(state: GameState) {
    const container = document.getElementById("all-cards-list");
    if (!container) return;

    const locations = (state.card_locations || {}) as Record<string, string>;
    container.innerHTML = "";

    let matchedCount = 0;
    for (let id = 1; id <= 110; id++) {
      const meta = this.cardsMeta.get(id) || {
        id: num(id),
        name: t("cards:cardFallback", { id: num(id) }),
        ops: 0,
        side: "neutral",
        age: "early war",
        description: "",
        one_time: false
      };

      // The engine says whether the opponent has seen a hand card (HAND_US_KNOWN / _UNKNOWN);
      // the filter and badges only care whose hand it is. Compared raw, "US Hand" and "USSR
      // Hand" never matched a single card.
      const raw = locations[id.toString()] || "DRAW_DECK";
      const loc = raw.startsWith("HAND_USSR") ? "HAND_USSR" : raw.startsWith("HAND_US") ? "HAND_US" : raw;

      // In a player's view UNSEEN (the draw deck or the other side's unseen cards) answers to
      // the draw-deck filter.
      const filterLoc = loc === UNSEEN ? "DRAW_DECK" : loc;
      if (this.locationFilter !== "ALL" && filterLoc !== this.locationFilter) {
        continue;
      }

      // A name matches in English (the rules') or as the page shows it.
      const shownName = cardName(meta.id, meta.name);
      if (this.searchFilter) {
        const nameMatch = meta.name.toLowerCase().includes(this.searchFilter)
          || shownName.toLowerCase().includes(this.searchFilter);
        const idMatch = id.toString() === this.searchFilter;
        if (!nameMatch && !idMatch) continue;
      }

      matchedCount++;
      const itemEl = document.createElement("div");
      itemEl.className = "all-cards-item";

      const locBadgeClass = this.getLocationBadgeClass(loc === UNSEEN ? "DRAW_DECK" : loc);
      const opp = this.opponent(state);
      const locLabel = loc === UNSEEN
        ? (opp ? t("cards:deckOrHand", { side: sideName(opp) }) : t("cards:deckOrOtherHand"))
        : this.getLocationLabel(loc);
      const eraClass = meta.age === "early war" ? "era-early" : (meta.age === "mid war" ? "era-mid" : "era-late");

      itemEl.innerHTML = `
        <div class="card-item-era-bar ${eraClass}"></div>
        <div class="all-cards-item-left">
          <div class="card-ops-badge ${meta.side}">${meta.ops}</div>
          <div class="all-cards-info">
            <div class="card-name">#${meta.id} ${shownName} ${meta.one_time ? "★" : ""}</div>
          </div>
        </div>
        <div class="card-loc-badge ${locBadgeClass}">${locLabel}</div>
      `;

      // No description here -- neither an era/side line nor the hover tooltip the hand tabs
      // show: the list is for finding a card and where it is. Its text is one click away.
      itemEl.title = t("cards:clickForText");
      itemEl.addEventListener("click", () => {
        this.showPileModal(t("cards:cardModalTitle", { id: meta.id, name: cardName(meta.id, meta.name) }), [id]);
      });

      container.appendChild(itemEl);
    }

    if (matchedCount === 0) {
      container.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 20px;">${t("cards:noMatch")}</div>`;
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
      case "PEEKED_TEMP": return "loc-ongoing";
      default: return "loc-default";
    }
  }

  private getLocationLabel(loc: string): string {
    switch (loc) {
      case "HAND_US": return t("cards:location.handUs");
      case "HAND_USSR": return t("cards:location.handUssr");
      case "DRAW_DECK": return t("cards:location.drawDeck");
      case "DISCARD_PILE": return t("cards:location.discard");
      case "REMOVED_FROM_GAME": return t("cards:location.removed");
      case "ONGOING_EVENT": return t("cards:location.ongoing");
      case "UNAVAILABLE": return t("cards:location.futureEra");
      case "PEEKED_TEMP": return t("cards:location.peeked");
      default: return escapeHtml(loc);
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
          <div style="font-weight: 800; font-size: 13px;">${t("cards:china.title", { name: cardName(6, this.cardsMeta.get(6)?.name ?? "The China Card").toUpperCase() })}</div>
          <div style="font-size: 11px; opacity: 0.9;">${t("cards:china.holder", { holder: escapeHtml(sideName(holder)), status: isPlayable ? t("cards:china.playable") : t("cards:china.faceDown") })}</div>
        </div>
        <div style="font-weight: bold; font-size: 11px; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">
          ${t("cards:china.asiaBonus")}
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
          badge: info.duration === "TURN" ? t("cards:effect.turnOnly") : (info.duration === "PERMANENT" ? t("cards:effect.permanent") : t("cards:effect.conditional")),
          desc: info.description
        });
      } else {
        effectCards.push({
          title: escapeHtml(f.replace(/_ACTIVE|_PLAYED/, "")),
          side: "NEUTRAL",
          badge: t("cards:effect.active"),
          desc: t("cards:effect.continuous", { flag: escapeHtml(f) })
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
    const perk = (side: "US" | "USSR", title: string, desc: string) => effectCards.push({ title, side, badge: t("cards:effect.spacePerk"), desc });
    const us = sideName("US"), ussr = sideName("USSR");
    if (usSpace >= 2 && ussrSpace < 2) perk("US", t("cards:perk.animalInSpace", { side: us }), t("cards:perk.animalInSpaceDesc", { side: us }));
    if (usSpace >= 4 && ussrSpace < 4) perk("US", t("cards:perk.manInEarthOrbit", { side: us }), t("cards:perk.manInEarthOrbitDesc", { side: us, opponent: ussr }));
    if (usSpace >= 6 && ussrSpace < 6) perk("US", t("cards:perk.eagleHasLanded", { side: us }), t("cards:perk.landedDesc", { side: us }));
    if (usSpace >= 8 && ussrSpace < 8) perk("US", t("cards:perk.spaceStation", { side: us }), t("cards:perk.spaceStationDesc", { side: us }));

    if (ussrSpace >= 2 && usSpace < 2) perk("USSR", t("cards:perk.animalInSpace", { side: ussr }), t("cards:perk.animalInSpaceDesc", { side: ussr }));
    if (ussrSpace >= 4 && usSpace < 4) perk("USSR", t("cards:perk.manInEarthOrbit", { side: ussr }), t("cards:perk.manInEarthOrbitDesc", { side: ussr, opponent: us }));
    if (ussrSpace >= 6 && usSpace < 6) perk("USSR", t("cards:perk.bearHasLanded", { side: ussr }), t("cards:perk.landedDesc", { side: ussr }));
    if (ussrSpace >= 8 && usSpace < 8) perk("USSR", t("cards:perk.spaceStation", { side: ussr }), t("cards:perk.spaceStationDesc", { side: ussr }));

    if (effectCards.length === 0) {
      flagsContainer.innerHTML = `<div style="font-size: 11px; color: var(--text-dim); text-align: center; padding: 10px;">${t("cards:effect.none")}</div>`;
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

  /**
   * The effects that last for this turn only, on top of the map where they are always in view:
   * every flag whose `EFFECT_INFO_MAP` duration is TURN. The Space Race attempt flags are not in
   * the map and so never appear here; permanent and conditional effects stay in the side panel.
   */
  private renderTurnEffects(state: GameState) {
    const strip = document.getElementById("map-turn-effects");
    if (!strip) return;
    const esc = (s: string) => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
    const chips = (state.flags || [])
      .map(f => ({ flag: f, info: EFFECT_INFO_MAP[f] }))
      .filter(e => e.info && e.info.duration === "TURN")
      .map(({ flag, info }) =>
        `<span class="turn-effect side-${info.side.toLowerCase()}" data-flag="${flag}" title="${esc(`${info.cardName}: ${info.description}`)}">${esc(info.name.replace(/ Active$/, ""))}</span>`);
    strip.innerHTML = `<span class="turn-effects-label">${t("cards:effect.thisTurn")}</span>`
      + (chips.length ? chips.join("") : `<span class="turn-effects-none">${t("cards:effect.noEffects")}</span>`);
  }

  private renderContextStack(state: GameState) {
    const stackContainer = document.getElementById("context-stack-container");
    if (!stackContainer) return;

    const ctx = state.decision_context;
    if (!ctx) return;

    stackContainer.innerHTML = `
      <div>${t("cards:context.depth", { depth: num(ctx.stack_depth || 0) })}</div>
      <div>${t("cards:context.decision", { decision: escapeHtml(ctx.decision_type_name || ctx.decision_type), player: escapeHtml(ctx.decision_player) })}</div>
      ${ctx.pending_op_card ? `<div>${t("cards:context.pendingCard", { card: num(ctx.pending_op_card), ops: num(ctx.pending_ops_value) })}</div>` : ""}
      ${ctx.resolving_card ? `<div>${t("cards:context.resolvingCard", { card: num(ctx.resolving_card), name: escapeHtml(ctx.resolving_card_name || "") })}</div>` : ""}
      ${ctx.remaining_steps ? `<div>${t("cards:context.remainingSteps", { steps: num(ctx.remaining_steps) })}</div>` : ""}
    `;
  }

  private showTooltip(e: MouseEvent, meta: CardMetadata) {
    this.tooltipEl.innerHTML = `
      <div style="font-weight: bold; font-size: 13px; color: #F8FAFC; margin-bottom: 4px;">
        #${meta.id} ${cardName(meta.id, meta.name)} ${meta.one_time ? "★" : ""}
      </div>
      <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">
        ${t("cards:tooltipMeta", { age: ageLabel(meta.age), side: cardSideLabel(meta.side), ops: meta.ops })}
      </div>
      <div style="font-size: 12px; color: #E2E8F0; line-height: 1.4;">
        ${cardDescription(meta.id, meta.description)}
      </div>
    `;

    this.tooltipEl.classList.remove("hidden");
    const x = Math.min(window.innerWidth - 300, e.clientX + 15);
    // Below the pointer, or above it where the card would run off the bottom (the action log).
    const h = this.tooltipEl.offsetHeight;
    const y = e.clientY + 15 + h <= window.innerHeight ? e.clientY + 15 : Math.max(8, e.clientY - 15 - h);
    this.tooltipEl.style.left = `${x}px`;
    this.tooltipEl.style.top = `${y}px`;
  }

  /** A card's tooltip by its id (the action log's card names, main.ts). */
  public showCardTooltip(e: MouseEvent, cardId: number) {
    const meta = this.cardsMeta.get(cardId);
    if (meta) this.showTooltip(e, meta);
  }

  public hideTooltip() {
    this.tooltipEl.classList.add("hidden");
  }

  public showPileModal(title: string, cardIds: number[]) {
    const modal = document.getElementById("modal-container");
    const titleEl = document.getElementById("modal-title");
    const bodyEl = document.getElementById("modal-body");
    if (!modal || !titleEl || !bodyEl) return;

    titleEl.textContent = t("cards:pileModalTitle", { title, n: cardIds.length });
    bodyEl.innerHTML = "";

    if (cardIds.length === 0) {
      bodyEl.innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 20px;">${t("cards:pileModalEmpty")}</div>`;
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
          <div style="font-weight: bold; color: #F8FAFC;">${t("cards:pileCard", { id: meta.id, name: cardName(meta.id, meta.name), ops: meta.ops, side: cardSideLabel(meta.side), age: ageLabel(meta.age) })}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px; line-height: 1.35;">${cardDescription(meta.id, meta.description)}</div>
        `;
        bodyEl.appendChild(row);
      });
    }

    modal.classList.remove("hidden");
  }
}
