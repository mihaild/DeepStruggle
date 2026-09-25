import { GameState, MicroAction } from "./types";
import { CARDS_METADATA } from "./metadata";

const CARD_META = new Map(CARDS_METADATA.map(c => [c.id, c]));

/** Where a card an event offers sits, in words ("Drawn", "Discard pile", "US hand", ...). */
function whereLabel(loc: string | undefined): string {
  if (!loc) return "";
  if (loc === "PEEKED_TEMP") return "Drawn";
  if (loc === "DISCARD_PILE") return "Discard pile";
  if (loc.startsWith("HAND_USSR")) return "USSR hand";
  if (loc.startsWith("HAND_US")) return "US hand";
  if (loc === "REMOVED_FROM_GAME") return "Removed";
  if (loc === "DRAW_DECK") return "Draw deck";
  return loc;
}

function escAttr(s: string): string {
  return s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
}

/**
 * One button per card an event lets the player choose. The cards are often in no hand the
 * player can see -- Our Man in Tehran's five drawn cards, the discard pile for Star Wars and SALT
 * Negotiations -- so without these the choice could not be made from the page at all.
 */
function cardChoiceButtons(state: GameState, ids: number[]): string {
  const locs = (state.card_locations || {}) as Record<string, string>;
  return ids.filter(id => id >= 1 && id <= 110).map(id => {
    const meta = CARD_META.get(id);
    const name = meta?.name ?? `Card #${id}`;
    const side = meta?.side ?? "neutral";
    const where = whereLabel(locs[String(id)]);
    const text = meta?.description ? escAttr(meta.description) : "";
    return `
      <button class="btn btn-secondary btn-block btn-hud-action btn-choice-card card-choice side-${side}" data-primary="${id}" title="${text}" style="margin-bottom: 6px; text-align: left; padding: 7px 10px;">
        <div style="display: flex; justify-content: space-between; gap: 8px; align-items: baseline;">
          <span style="font-weight: bold;">#${id} ${escAttr(name)}</span>
          <span style="font-size: 11px; opacity: 0.8; white-space: nowrap;">${meta ? `${meta.ops} Ops · ${side.toUpperCase()}` : ""}</span>
        </div>
        ${where ? `<div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">${where}</div>` : ""}
      </button>`;
  }).join("");
}

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
  45: { // Summit
    0: { title: "Improve DEFCON (+1)", desc: "Improve the DEFCON track by 1 level." },
    1: { title: "Degrade DEFCON (-1)", desc: "Degrade the DEFCON track by 1 level." },
    2: { title: "Keep DEFCON Unchanged", desc: "Do not alter the DEFCON track." }
  },
  46: { // How I Learned to Stop Worrying
    1: { title: "Set DEFCON to 1 (Warning: Loss)", desc: "Set DEFCON track to 1 (triggers DEFCON suicide loss for phasing player!)." },
    2: { title: "Set DEFCON to 2", desc: "Set DEFCON track to 2." },
    3: { title: "Set DEFCON to 3", desc: "Set DEFCON track to 3." },
    4: { title: "Set DEFCON to 4", desc: "Set DEFCON track to 4." },
    5: { title: "Set DEFCON to 5 (Peace)", desc: "Reset DEFCON track to 5." }
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
  /** Called when the HUD re-renders itself (die selector), so decorations can be re-applied. */
  public onRerender: (() => void) | null = null;

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
        } else if (resolvingCard === 108) { // Our Man in Tehran
          promptText = `<strong>${player}:</strong> <em>Our Man in Tehran</em>: these are the cards drawn from the deck. Discard any of them, one at a time; <em>Done</em> returns the rest to the deck:`;
        } else if (resolvingCard === 85) { // Star Wars
          promptText = `<strong>${player}:</strong> <em>Star Wars</em>: Select a non-scoring card from the Discard Pile -- its Event is played immediately:`;
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
        } else if (resolvingCard === 250) { // Space Walk (Box 6)
          promptText = `<strong>${player}:</strong> <em>Space Walk (Space Race Box 6)</em>: You may discard 1 card from hand at turn end (or click Pass to retain full hand):`;
        } else if (resolvingCard > 0) {
          promptText = `<strong>${player}:</strong> Resolving Event (Card #${resolvingCard}): Select a card:`;
        } else if (phaseName === "HEADLINE") {
          let oppHeadlineBanner = "";
          if (player === "US" && state.headline_ussr_card) {
            oppHeadlineBanner = `<div class="headline-reveal-box" style="background: rgba(220,53,69,0.15); border: 1px solid rgba(220,53,69,0.4); padding: 6px 10px; border-radius: 4px; margin-bottom: 8px; font-size: 13px;">📡 <strong>Space Race Privilege (Box 4):</strong> USSR revealed Headline: Card #${state.headline_ussr_card}</div>`;
          } else if (player === "USSR" && state.headline_us_card) {
            oppHeadlineBanner = `<div class="headline-reveal-box" style="background: rgba(13,110,253,0.15); border: 1px solid rgba(13,110,253,0.4); padding: 6px 10px; border-radius: 4px; margin-bottom: 8px; font-size: 13px;">📡 <strong>Space Race Privilege (Box 4):</strong> US revealed Headline: Card #${state.headline_us_card}</div>`;
          }
          promptText = `${oppHeadlineBanner}<strong>${player} Headline Phase:</strong> Select a card from your hand to play as your Headline:`;
        } else {
          promptText = `<strong>${player} Action Round ${state.action_round}:</strong> Select a card from your hand to play:`;
        }

        // An event's card choice lists its cards here: they are often in no hand on screen.
        if (resolvingCard > 0) {
          buttonsHtml += cardChoiceButtons(state, validIds);
        }
        // Offered whenever the engine offers it (id 0 in valid_ids), not only when the context
        // allows an early stop: a choice with no legal card at all -- Star Wars facing a discard
        // pile of events the US cannot trigger -- is left by passing, and without this button the
        // page had nothing to click.
        if (allowEarlyStop || validIds.includes(0)) {
          const passLabel = resolvingCard === 108 ? "✓ Done — return the rest to the deck"
            : resolvingCard === 43 ? "✓ Take no card"
            : !validIds.some(id => id >= 1 && id <= 110) ? "✓ No card can be chosen — continue"
            : "✓ Pass / Confirm (Do not discard)";
          buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">${passLabel}</button>`;
        }
        break;

      case 2: // SELECT_PLAY_MODE
        const cardName = ctx.pending_op_card_name || `Card #${ctx.pending_op_card}`;
        promptText = `<strong>${player}:</strong> Choose how to play <em>${cardName}</em>:`;
        // Resolution, NOT the retired PlayMode. Source of truth: `enum class Resolution` in
        // engine/include/ts/types.hpp, mirrored by `mode_names` in bindings/action_encoder.py.
        // P17 merged PlayMode -> CHOOSE_TIMING_BRANCH -> SELECT_OP_MODE into this one decision,
        // and these labels were left on the old enum: 1/2/3 named the wrong action entirely and
        // 4 had no label, so an opponent card rendered as
        // "Event / Space Race / Pass-Discard / Mode 4" for EVENT / INFLUENCE / COUP / REALIGN.
        const ops = ctx.pending_ops_value;
        const modeLabels: Record<number, { title: string; desc: string; class: string }> = {
          0: { title: "Resolve the Event", desc: "Trigger the printed event. On an opponent's card this is the event-first branch — you choose how to spend the Ops afterwards.", class: "btn-primary" },
          1: { title: "Space Race Attempt", desc: `Discard the card to attempt an advance on the Space Race track (${ops} Ops).`, class: "btn-secondary" },
          2: { title: "Operations: Place Influence", desc: `Spend ${ops} Ops placing influence.`, class: "btn-success" },
          3: { title: "Operations: Coup Attempt", desc: `Spend ${ops} Ops on a coup — rolls a die.`, class: "btn-danger" },
          4: { title: "Operations: Realignment", desc: `Spend ${ops} Ops on realignment rolls — opposed dice.`, class: "btn-warning" }
        };
        validIds.forEach(m => {
          // Loud, not bland. A silent `Mode ${m}` is what let this drift sit unnoticed; anything
          // outside Resolution::COUNT means the enum grew and this table did not.
          const cfg = modeLabels[m] || {
            title: `⚠ Unlabelled resolution mode ${m}`,
            desc: "The viewer has no label for this mode — the engine's Resolution enum has changed and web/ui/src/action_hud.ts was not updated.",
            class: "btn-warning"
          };
          buttonsHtml += `
            <button class="btn ${cfg.class} btn-block btn-hud-action btn-choice-card" data-primary="${m}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
              <div style="font-weight: bold; font-size: 13px;">${cfg.title}</div>
              <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${cfg.desc}</div>
            </button>
          `;
        });
        // A die is rolled by SPACE(1), OPS_COUP(3) and OPS_REALIGN(4). OPS_INFLUENCE(2) rolls
        // nothing. This read `validIds.includes(2)`, which was SPACE under the retired PlayMode
        // enum, so the manual die input appeared for influence and was missing for coups.
        if ([9, 11].includes(ctx.pending_op_card) || validIds.some(m => m === 1 || m === 3 || m === 4)) {
          showDieSelector = true;
        }
        break;

      // RETIRED by P17 and unreachable: on an opponent card, Resolution::EVENT *is* the
      // event-first branch, so no separate timing decision is ever emitted. Kept because
      // a replay recorded before P17 can still contain one.
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
        const isCoup = (legal as any).op_mode === 1 || (ctx as any).op_mode === 1;
        const isRealign = (legal as any).op_mode === 2 || (ctx as any).op_mode === 2;
        const isCoupOrRealign = isCoup || isRealign;

        if (phaseName === "SETUP") {
          const region = player === "USSR" ? "Eastern Europe" : "Western Europe";
          promptText = `<strong>${player} Setup Phase:</strong> Click highlighted countries in ${region} to place influence (<strong>${ctx.remaining_steps} remaining</strong>):`;
        } else if (ctx.resolving_card > 0) {
          const resCard = ctx.resolving_card_name || `Card #${ctx.resolving_card}`;
          promptText = `<strong>${player}:</strong> Resolving event <em>${resCard}</em>. Select target country (<strong>${ctx.remaining_steps} remaining</strong>):`;
          if (isWarEvent) showDieSelector = true;
        } else if (isCoup) {
          promptText = `<strong>${player}:</strong> Select target country to <strong>COUP</strong> with <strong>${ctx.pending_ops_value} Ops</strong> (or click on map):`;
          showDieSelector = true;
        } else if (isRealign) {
          promptText = `<strong>${player}:</strong> Select target country for <strong>REALIGNMENT</strong> (or click on map):`;
          showDieSelector = true;
        } else {
          const remaining = ctx.remaining_steps > 0 ? ` (${ctx.remaining_steps} Ops remaining)` : "";
          promptText = `<strong>${player}:</strong> Select a target country${remaining}:`;
        }

        // Render Candidate Targets List
        if (validIds.length > 0 && (isCoupOrRealign || isWarEvent || validIds.length <= 12)) {
          let targetsHtml = "";
          validIds.filter(id => id < 84).forEach(cid => {
            const country = Object.values(state.countries || {}).find(c => c.id === cid) || (state.countries ? (state.countries as any)[cid] : null);
            const cName = country?.name || `Country #${cid}`;
            const cStab = country?.stability || 1;
            const isBg = country?.battleground || false;
            const usInf = country?.us_influence || 0;
            const ussrInf = country?.ussr_influence || 0;

            let badgeHtml = "";
            let calculusHtml = "";

            if (isCoup) {
              const ops = ctx.pending_ops_value;
              const def = cStab * 2;
              const minRoll = Math.max(1, def - ops + 1);
              const prob = minRoll <= 1 ? 100 : (minRoll > 6 ? 0 : Math.round(((7 - minRoll) / 6) * 100));

              if (isBg) {
                if (state.defcon === 2) {
                  badgeHtml = `<span class="badge" style="background: #DC2626; color: #FFF; font-weight: bold; font-size: 10px; margin-left: 6px;">🚨 DEFCON 1 SUICIDE</span>`;
                } else {
                  badgeHtml = `<span class="badge" style="background: #EF4444; color: #FFF; font-size: 10px; margin-left: 6px;">DEFCON ${state.defcon}→${state.defcon - 1}</span>`;
                }
              }

              if (this.selectedDieRoll > 0) {
                const total = ops + this.selectedDieRoll;
                const net = Math.max(0, total - def);
                calculusHtml = net > 0 
                  ? `<span style="color: #34D399; font-weight: bold;">🎲 Roll ${this.selectedDieRoll}: Ops ${ops} + ${this.selectedDieRoll} vs ${def} Def -> Net +${net} Inf (Success)</span>`
                  : `<span style="color: #F87171;">🎲 Roll ${this.selectedDieRoll}: Ops ${ops} + ${this.selectedDieRoll} vs ${def} Def -> 0 Net (Fails)</span>`;
              } else {
                calculusHtml = `<span>🎲 Auto: Roll ≥ ${minRoll} needed (${prob}% chance) • Def: ${def} (2×${cStab})</span>`;
              }
            } else if (isRealign) {
              calculusHtml = `<span>🎲 Opposed Die Rolls • US:${usInf} vs USSR:${ussrInf} (Stab ${cStab})</span>`;
            } else if (isWarEvent) {
              calculusHtml = `<span>🎲 War Event: Roll vs Target Threshold</span>`;
            } else {
              calculusHtml = `<span>Stab ${cStab} • US:${usInf} / USSR:${ussrInf}</span>`;
            }

            targetsHtml += `
              <button class="btn btn-secondary btn-block btn-hud-action btn-target-card" data-primary="${cid}" data-secondary="${this.selectedDieRoll}" style="margin-bottom: 6px; text-align: left; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.12);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <span style="font-weight: bold; font-size: 13px;">${isBg ? '★ ' : ''}${cName} (Stab ${cStab}) ${badgeHtml}</span>
                  <span style="font-size: 11px; opacity: 0.85;">US:<strong>${usInf}</strong> / USSR:<strong>${ussrInf}</strong></span>
                </div>
                <div style="font-size: 11px; color: var(--text-dim); margin-top: 3px;">${calculusHtml}</div>
              </button>
            `;
          });

          if (targetsHtml) {
            buttonsHtml += `
              <div class="hud-target-list-header" style="font-size: 11px; font-weight: 700; color: var(--text-dim); margin: 8px 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">
                Available Target Actions (${validIds.length}):
              </div>
              <div class="hud-targets-scroll-list" style="max-height: 200px; overflow-y: auto; padding-right: 2px;">
                ${targetsHtml}
              </div>
            `;
          }
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
        // The re-render dropped any probability badges painted on the buttons.
        this.onRerender?.();
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
