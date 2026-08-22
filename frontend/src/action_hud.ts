import { GameState, MicroAction } from "./types";

interface BranchMeta {
  title: string;
  desc: string;
}

const CARD_BRANCHES: Record<number, Record<number, BranchMeta>> = {
  16: { // Warsaw Pact Formed
    0: { title: "Remove US Influence", desc: "Remove all US influence from up to 4 countries in Eastern Europe." },
    1: { title: "Add USSR Influence", desc: "Add 5 USSR influence in Eastern Europe (max 2 per country)." }
  },
  20: { // Olympic Games
    0: { title: "Participate in Olympic Games", desc: "Contest the Olympics with 1d6 roll (+1 modifier for sponsor). Highest roll receives 2 VP." },
    1: { title: "Boycott Olympic Games", desc: "Boycott: Sponsor receives +2 VP, DEFCON degrades by 1, and sponsor conducts 4 Operations." }
  },
  22: { // Independent Reds
    0: { title: "Match Influence", desc: "Add Influence to Yugoslavia, Romania, Bulgaria, Hungary, or Czechoslovakia equal to USSR influence." },
    1: { title: "Remove USSR Influence", desc: "Remove all USSR Influence from Yugoslavia, Romania, Bulgaria, Hungary, or Czechoslovakia." }
  },
  7: { // Socialist Governments
    0: { title: "Remove US Influence", desc: "Remove 3 US influence total from Western Europe (max 2 per country)." }
  },
  23: { // Marshall Plan
    0: { title: "Place Influence", desc: "Add 1 US influence in each of 7 Western European countries (non-USSR controlled)." }
  },
  28: { // Suez Crisis
    0: { title: "Remove US Influence", desc: "Remove 4 US influence total from UK, France, and Israel (max 2 per country)." }
  },
  30: { // Decolonization
    0: { title: "Place Influence", desc: "Add 1 USSR influence in each of any 4 countries in Africa and/or SE Asia." }
  },
  33: { // De-Stalinization
    0: { title: "Redeploy Influence", desc: "Remove up to 4 USSR influence and redeploy to non-US controlled countries (max 2 per country)." }
  },
  40: { // Cuban Missile Crisis
    0: { title: "Cancel Crisis", desc: "Remove 2 influence from Cuba (USSR) or W. Germany/Turkey (US) to cancel crisis." }
  },
  43: { // SALT Negotiations
    0: { title: "Retrieve Discarded Card", desc: "Search the discard pile and retrieve 1 card into hand." }
  },
  46: { // How I Learned to Stop Worrying
    0: { title: "Set DEFCON to 5 (Peace)", desc: "Reset DEFCON track to 5." },
    1: { title: "Set DEFCON to 4", desc: "Set DEFCON track to 4." },
    2: { title: "Set DEFCON to 3", desc: "Set DEFCON track to 3." },
    3: { title: "Set DEFCON to 2", desc: "Set DEFCON track to 2." },
    4: { title: "Set DEFCON to 1 (Warning: Loss)", desc: "Set DEFCON track to 1 (triggers DEFCON suicide loss for phasing player!)." }
  },
  47: { // Junta
    0: { title: "Place 2 Influence", desc: "Add 2 Influence in Central America or South America." },
    1: { title: "Free Coup / Realignment", desc: "Conduct a free Coup or Realignment in Central or South America using 2 Ops." }
  },
  74: { // Voice of America
    0: { title: "Remove USSR Influence", desc: "Remove 4 USSR influence from non-European countries (max 2 per country)." }
  },
  77: { // Ask Not What Your Country Can Do For You
    0: { title: "Discard and Redraw Hand", desc: "Discard selected cards from hand and draw replacements from the draw deck." },
    1: { title: "Keep Current Hand (Pass)", desc: "Do not discard any cards; retain full hand." }
  },
  80: { // One Small Step
    0: { title: "Advance Space Track (+2)", desc: "Advance 2 spaces on the Space Race track." }
  },
  86: { // North Sea Oil
    0: { title: "Gain 8th Action Round", desc: "Grants US an extra 8th Action Round for this turn." }
  },
  90: { // Glasnost
    0: { title: "Gain 2 VP & 4 Ops", desc: "Gain 2 VP and conduct 4 Operations (The Reformer active)." }
  },
  91: { // Ortega Elected in Nicaragua
    0: { title: "Free Coup in Neighbor", desc: "Conduct a free Coup with 2 Ops in a country adjacent to Nicaragua." }
  },
  96: { // Tear Down This Wall
    0: { title: "Free Coup / Realignment in Europe", desc: "Conduct a free Coup or Realignment in Europe with 3 Ops." }
  },
  100: { // Wargames
    0: { title: "End Game (Give Opponent 6 VP)", desc: "Award opponent 6 VP and immediately terminate the game with current score." },
    1: { title: "Cancel / Continue Play", desc: "Do not trigger victory condition; continue normal play." }
  },
  104: { // Cambridge Five
    0: { title: "Reveal US Hand & Place Influence", desc: "Reveal US hand and add 1 USSR influence in a region from a revealed scoring card." },
    1: { title: "Pass / No Scoring Cards", desc: "No scoring cards held in US hand." }
  },
  105: { // Special Relationship
    0: { title: "Add 1 Influence & Gain 2 VP (NATO)", desc: "Add 1 US influence in adjacent country and score 2 VP (NATO active)." },
    1: { title: "Add 2 Influence (No NATO)", desc: "Add 2 US influence in adjacent country (NATO inactive)." }
  },
  107: { // Che
    0: { title: "Free Coup in Non-Battleground", desc: "Conduct free Coup in non-battleground country in CA, SA, or Africa." }
  }
};

export class ActionHud {
  private container: HTMLElement;
  private onAction: (action: MicroAction) => void;
  public selectedDieRoll: number = 0; // 0 = auto

  constructor(onAction: (action: MicroAction) => void) {
    this.container = document.getElementById("decision-body")!;
    this.onAction = onAction;
  }

  public render(state: GameState) {
    const ctx = state.decision_context;
    const legal = state.legal_actions;
    const badgeEl = document.getElementById("decision-type-badge");

    if (!ctx || !legal || state.is_terminal) {
      if (badgeEl) badgeEl.textContent = state.is_terminal ? "TERMINAL" : "WAITING";
      this.container.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 20px;">No active decision required.</div>`;
      return;
    }

    const dType = ctx.decision_type;
    const dTypeName = this.getDecisionTypeName(dType);
    const player = ctx.decision_player || "NONE";
    const phaseName = state.current_phase_name || state.phase_name || "ACTION";

    if (badgeEl) {
      badgeEl.textContent = dTypeName;
      badgeEl.className = `badge badge-${player.toLowerCase()}`;
    }

    let promptText = "";
    let buttonsHtml = "";
    let showDieSelector = false;

    const validIds = legal.valid_ids || [];
    const allowEarlyStop = legal.allow_early_stop || ctx.allow_early_stop;

    switch (dType) {
      case 0: // NONE
        promptText = `Simulation idle or transitioning.`;
        break;

      case 1: // SELECT_CARD
        const resolvingCard = ctx.resolving_card || 0;
        if (resolvingCard === 10) { // Blockade
          promptText = `<strong>${player}:</strong> <em>Blockade</em> is resolving! Select and discard a card with <strong>Operations ≥ 3</strong> from hand to maintain US influence in West Germany (or confirm/pass to forfeit influence):`;
        } else if (resolvingCard === 43) { // SALT Negotiations
          promptText = `<strong>${player}:</strong> <em>SALT Negotiations</em>: Select a non-scoring card from the Discard Pile to return to hand:`;
        } else if (resolvingCard === 5) { // Five Year Plan
          promptText = `<strong>${player}:</strong> <em>Five Year Plan</em>: Select a card from hand to discard:`;
        } else if (resolvingCard === 49) { // Missile Envy
          promptText = `<strong>${player}:</strong> <em>Missile Envy</em>: Select your highest Ops card to transfer to opponent:`;
        } else if (resolvingCard === 98) { // Aldrich Ames
          promptText = `<strong>${player}:</strong> <em>Aldrich Ames</em>: Select a card from US hand to discard:`;
        } else if (resolvingCard === 67) { // Grain Sales
          promptText = `<strong>${player}:</strong> <em>Grain Sales</em>: Select a card from USSR hand:`;
        } else if (resolvingCard === 77) { // Ask Not
          promptText = `<strong>${player}:</strong> <em>Ask Not What Your Country Can Do For You</em>: Select cards from hand to discard:`;
        } else if (resolvingCard === 92) { // Terrorism
          promptText = `<strong>${player}:</strong> <em>Terrorism</em>: Select a card from hand to discard:`;
        } else if (resolvingCard > 0) {
          promptText = `<strong>${player}:</strong> Resolving Event (Card #${resolvingCard}): Select a card:`;
        } else if (phaseName === "HEADLINE") {
          promptText = `<strong>${player} Headline Phase:</strong> Select a card from your hand to play as your Headline:`;
        } else {
          promptText = `<strong>${player} Action Round ${state.action_round}:</strong> Select a card from your hand to play:`;
        }

        if (allowEarlyStop) {
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">✓ Pass / Confirm (Do not discard)</button>`;
        }
        break;

      case 2: // SELECT_PLAY_MODE
        const cardName = ctx.pending_op_card_name || `Card #${ctx.pending_op_card}`;
        promptText = `<strong>${player}:</strong> Choose how to play <em>${cardName}</em>:`;
        const modeLabels: Record<number, { title: string; desc: string; class: string }> = {
          0: { title: "Mode 0: Play as Event", desc: "Trigger printed card event effect", class: "btn-primary" },
          1: { title: "Mode 1: Play for Operations", desc: `Use Ops for Influence, Coup, or Realignment (${ctx.pending_ops_value} Ops)`, class: "btn-success" },
          2: { title: "Mode 2: Space Race Attempt", desc: "Attempt to advance on Space Race track", class: "btn-secondary" },
          3: { title: "Mode 3: Pass / Discard", desc: "Discard without effect", class: "btn-danger" }
        };
        validIds.forEach(m => {
          const cfg = modeLabels[m] || { title: `Mode ${m}`, desc: "", class: "btn-secondary" };
          buttonsHtml += `
            <button class="btn ${cfg.class} btn-block btn-hud-action btn-choice-card" data-primary="${m}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
              <div style="font-weight: bold; font-size: 13px;">${cfg.title}</div>
              <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${cfg.desc}</div>
            </button>
          `;
        });
        if ([9, 11].includes(ctx.pending_op_card)) {
          showDieSelector = true;
        }
        break;

      case 3: // CHOOSE_TIMING_BRANCH
        const oppCardName = ctx.pending_op_card_name || `Card #${ctx.pending_op_card}`;
        promptText = `<strong>${player}:</strong> Opponent card <em>${oppCardName}</em> played for Ops. Choose execution order:`;
        if (validIds.includes(0)) {
          buttonsHtml += `
            <button class="btn btn-primary btn-block btn-hud-action btn-choice-card" data-primary="0" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
              <div style="font-weight: bold; font-size: 13px;">Branch 0: Operations First</div>
              <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">Conduct Operations now. Opponent event triggers afterwards.</div>
            </button>
          `;
        }
        if (validIds.includes(1)) {
          buttonsHtml += `
            <button class="btn btn-secondary btn-block btn-hud-action btn-choice-card" data-primary="1" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
              <div style="font-weight: bold; font-size: 13px;">Branch 1: Opponent Event First</div>
              <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">Trigger opponent event first. Conduct your Operations afterwards.</div>
            </button>
          `;
        }
        break;

      case 4: // SELECT_OP_MODE
        promptText = `<strong>${player}:</strong> Select Operations action with <strong>${ctx.pending_ops_value} Ops</strong> available:`;
        const opLabels: Record<number, { text: string; desc: string; class: string }> = {
          0: { text: "Place Influence", desc: "Place influence markers in adjacent countries", class: "btn-primary" },
          1: { text: "Conduct Coup Attempt", desc: "Roll die to remove enemy influence and add own", class: "btn-danger" },
          2: { text: "Conduct Realignment", desc: "Opposed die rolls to remove enemy influence", class: "btn-secondary" }
        };
        validIds.forEach(m => {
          const cfg = opLabels[m] || { text: `Op Mode ${m}`, desc: "", class: "btn-secondary" };
          buttonsHtml += `
            <button class="btn ${cfg.class} btn-block btn-hud-action btn-choice-card" data-primary="${m}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
              <div style="font-weight: bold;">${cfg.text}</div>
              <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${cfg.desc}</div>
            </button>
          `;
        });
        break;

      case 5: // POINT_NODE
        const isWarEvent = [9, 11, 23, 36, 84, 18, 70, 72, 97, 106].includes(ctx.resolving_card);
        const isCoupOrRealign = (legal as any).op_mode === 1 || (legal as any).op_mode === 2 || (ctx as any).op_mode === 1 || (ctx as any).op_mode === 2;

        if (phaseName === "SETUP") {
          const region = player === "USSR" ? "Eastern Europe" : "Western Europe";
          promptText = `<strong>${player} Setup Phase:</strong> Click highlighted countries in ${region} to place influence (<strong>${ctx.remaining_steps} remaining</strong>).`;
        } else if (ctx.resolving_card > 0) {
          const resCard = ctx.resolving_card_name || `Card #${ctx.resolving_card}`;
          promptText = `<strong>${player}:</strong> Resolving event <em>${resCard}</em>. Click a highlighted target country (<strong>${ctx.remaining_steps} remaining</strong>):`;
          if (isWarEvent) showDieSelector = true;
        } else {
          const remaining = ctx.remaining_steps > 0 ? ` (${ctx.remaining_steps} Ops remaining)` : "";
          promptText = `<strong>${player}:</strong> Click a highlighted country on the map to target${remaining}:`;
          if (isCoupOrRealign) showDieSelector = true;
        }

        if (allowEarlyStop) {
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">✓ Done / Pass Remaining</button>`;
        }
        break;

      case 6: // CHOOSE_BRANCH
        const resolvingCardId = ctx.resolving_card || ctx.pending_op_card;
        const resolvingCardName = ctx.resolving_card_name || (resolvingCardId ? `Card #${resolvingCardId}` : "Event");
        promptText = `<strong>${player}:</strong> Choose option for <em>${resolvingCardName}</em>:`;
        const actionLabels = legal.valid_action_labels || {};

        validIds.forEach(b => {
          const branchMeta = CARD_BRANCHES[resolvingCardId]?.[b];
          const branchTitle = branchMeta ? branchMeta.title : (actionLabels[b.toString()] || `Option ${b + 1}`);
          const branchDesc = branchMeta ? branchMeta.desc : "";

          buttonsHtml += `
            <button class="btn btn-secondary btn-block btn-hud-action btn-choice-card" data-primary="${b}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; font-size: 13px;">Branch ${b}: ${branchTitle}</span>
                <span class="badge" style="font-size: 10px; background: rgba(255,255,255,0.15);">#${b}</span>
              </div>
              ${branchDesc ? `<div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${branchDesc}</div>` : ""}
            </button>
          `;
        });

        if (allowEarlyStop) {
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">✓ Confirm / Pass</button>`;
        }
        break;

      default:
        promptText = `Waiting for decision...`;
        break;
    }

    let dieSelectorHtml = "";
    if (showDieSelector) {
      dieSelectorHtml = `
        <div class="die-roll-selector-container">
          <div class="die-roll-header">
            <span>🎲 Die Roll Input:</span>
            <span class="die-selected-text">${this.selectedDieRoll === 0 ? "Auto Roll (PRNG)" : "Manual: " + this.selectedDieRoll}</span>
          </div>
          <div class="die-buttons-row">
            <button class="btn-die-opt ${this.selectedDieRoll === 0 ? "active" : ""}" data-roll="0">🎲 Auto</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 1 ? "active" : ""}" data-roll="1">⚀ 1</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 2 ? "active" : ""}" data-roll="2">⚁ 2</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 3 ? "active" : ""}" data-roll="3">⚂ 3</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 4 ? "active" : ""}" data-roll="4">⚃ 4</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 5 ? "active" : ""}" data-roll="5">⚄ 5</button>
            <button class="btn-die-opt ${this.selectedDieRoll === 6 ? "active" : ""}" data-roll="6">⚅ 6</button>
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
    const dieButtons = this.container.querySelectorAll(".btn-die-opt");
    dieButtons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        const roll = parseInt((e.currentTarget as HTMLElement).getAttribute("data-roll") || "0", 10);
        this.selectedDieRoll = roll;
        this.render(state);
      });
    });

    // Attach click listeners to action buttons
    const buttons = this.container.querySelectorAll(".btn-hud-action");
    buttons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        const target = e.currentTarget as HTMLElement;
        const primary = parseInt(target.getAttribute("data-primary") || "0", 10);
        const secondary = parseInt(target.getAttribute("data-secondary") || this.selectedDieRoll.toString(), 10);
        const flags = parseInt(target.getAttribute("data-flags") || "0", 10);

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
      0: "NONE",
      1: "SELECT_CARD",
      2: "SELECT_PLAY_MODE",
      3: "CHOOSE_TIMING_BRANCH",
      4: "SELECT_OP_MODE",
      5: "POINT_NODE",
      6: "CHOOSE_BRANCH"
    };
    return typeNames[dType] || `TYPE_${dType}`;
  }
}
