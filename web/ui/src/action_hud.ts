import { GameState, MicroAction } from "./types";
import { CARDS_METADATA } from "./metadata";
import { branchMeta } from "./game/branches";
import { ActionLayout, engineLayout, namesByValue } from "./engine/layout";
import { escapeHtml, num } from "./html";
import { cardDescription, cardName, countryName, regionName, render, sideName, t } from "./i18n";
import { enumLabel } from "./i18n/enums";

const CARD_META = new Map(CARDS_METADATA.map(c => [c.id, c]));

/** Where a card an event offers sits, in words ("Drawn", "Discard pile", "US hand", ...). */
function whereLabel(loc: string | undefined): string {
  if (!loc) return "";
  if (loc === "PEEKED_TEMP") return t("hud:where.drawn");
  if (loc === "DISCARD_PILE") return t("hud:where.discardPile");
  if (loc.startsWith("HAND_USSR")) return t("hud:where.ussrHand");
  if (loc.startsWith("HAND_US")) return t("hud:where.usHand");
  if (loc === "REMOVED_FROM_GAME") return t("hud:where.removed");
  if (loc === "DRAW_DECK") return t("hud:where.drawDeck");
  return loc;
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
    const name = meta ? cardName(id, meta.name) : t("hud:cardNumber", { id });
    const side = meta?.side ?? "neutral";
    const where = whereLabel(locs[String(id)]);
    const text = meta?.description ? escapeHtml(cardDescription(id, meta.description)) : "";
    return `
      <button class="btn btn-secondary btn-block btn-hud-action btn-choice-card card-choice side-${escapeHtml(side)}" data-primary="${id}" title="${text}" style="margin-bottom: 6px; text-align: left; padding: 7px 10px;">
        <div style="display: flex; justify-content: space-between; gap: 8px; align-items: baseline;">
          <span style="font-weight: bold;">#${id} ${escapeHtml(name)}</span>
          <span style="font-size: 11px; opacity: 0.8; white-space: nowrap;">${meta ? t("hud:cardOpsSide", { ops: meta.ops, side: escapeHtml(cardSide(side)) }) : ""}</span>
        </div>
        ${where ? `<div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">${escapeHtml(where)}</div>` : ""}
      </button>`;
  }).join("");
}

/** A card's side on its choice button: "US", "USSR" or "NEUTRAL". */
function cardSide(side: string): string {
  if (side === "us") return t("hud:cardSide.us");
  if (side === "ussr") return t("hud:cardSide.ussr");
  if (side === "neutral") return t("hud:cardSide.neutral");
  return side.toUpperCase();
}

/**
 * What an event's card choice asks for, by the card resolving (its ts::card_ids name); null for a
 * card with no words of its own. `who` is the deciding side, already HTML.
 */
function cardChoicePrompt(key: string, who: string): string | null {
  switch (key) {
    case "BLOCKADE": return t("hud:cardChoice.blockade", { who });
    case "SALT_NEGOTIATIONS": return t("hud:cardChoice.saltNegotiations", { who });
    case "OUR_MAN_IN_TEHRAN": return t("hud:cardChoice.ourManInTehran", { who });
    case "STAR_WARS": return t("hud:cardChoice.starWars", { who });
    case "FIVE_YEAR_PLAN": return t("hud:cardChoice.fiveYearPlan", { who });
    case "MISSILE_ENVY": return t("hud:cardChoice.missileEnvy", { who });
    case "ALDRICH_AMES": return t("hud:cardChoice.aldrichAmes", { who });
    case "GRAIN_SALES_TO_SOVIETS": return t("hud:cardChoice.grainSales", { who });
    case "ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU": return t("hud:cardChoice.askNot", { who });
    case "TERRORISM": return t("hud:cardChoice.terrorism", { who });
    case "EAGLE_BEAR_LANDED_DISCARD": return t("hud:cardChoice.eagleBearLanded", { who });
    default: return null;
  }
}

/** What a CHOOSE_BRANCH event asks, by its card's ts::card_ids name; anything else "choose an option". */
function branchAsk(key: string): string {
  switch (key) {
    case "CHERNOBYL": return t("hud:branchAsk.chernobyl");
    case "SUMMIT": return t("hud:branchAsk.summit");
    case "HOW_I_LEARNED_TO_STOP_WORRYING": return t("hud:branchAsk.howILearned");
    default: return t("hud:branchAsk.default");
  }
}

/**
 * The play modes, by the engine's Resolution names (WasmEngine.layout.resolutions). P17 merged
 * PlayMode -> CHOOSE_TIMING_BRANCH -> SELECT_OP_MODE into this one decision; on an opponent's card
 * EVENT is the event-first branch.
 */
function modeLabel(mode: string, ops: string): { title: string; desc: string; class: string } | null {
  switch (mode) {
    case "EVENT": return { title: t("hud:mode.event.title"), desc: t("hud:mode.event.desc"), class: "btn-primary" };
    case "SPACE": return { title: t("hud:mode.space.title"), desc: t("hud:mode.space.desc", { ops }), class: "btn-secondary" };
    case "OPS_INFLUENCE": return { title: t("hud:mode.opsInfluence.title"), desc: t("hud:mode.opsInfluence.desc", { ops }), class: "btn-success" };
    case "OPS_COUP": return { title: t("hud:mode.opsCoup.title"), desc: t("hud:mode.opsCoup.desc", { ops }), class: "btn-danger" };
    case "OPS_REALIGN": return { title: t("hud:mode.opsRealign.title"), desc: t("hud:mode.opsRealign.desc", { ops }), class: "btn-warning" };
    default: return null;
  }
}

/** The deferred Ops choice, by the engine's OpMode names (WasmEngine.layout.op_modes). */
function opModeLabel(mode: string): { text: string; desc: string; class: string } | null {
  switch (mode) {
    case "INFLUENCE": return { text: t("hud:opMode.influence.text"), desc: t("hud:opMode.influence.desc"), class: "btn-primary" };
    case "COUP": return { text: t("hud:opMode.coup.text"), desc: t("hud:opMode.coup.desc"), class: "btn-danger" };
    case "REALIGN": return { text: t("hud:opMode.realign.text"), desc: t("hud:opMode.realign.desc"), class: "btn-secondary" };
    default: return null;
  }
}

/** The ts::card_ids name of `id`, among those the engine exports; "" for any other card. */
function cardKey(L: ActionLayout, id: number): string {
  if (!id) return "";
  for (const [name, v] of Object.entries(L.card_ids)) if (v === id) return name;
  return "";
}

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

  /**
   * The open decision's options -- exactly the engine's legal actions (`legal_actions`, the engine's
   * mask), worded. The HUD says nothing the engine did not: no odds, no rule of its own.
   *
   * In a replay the state is the file's snapshot, so every string from it is escaped.
   */
  public render(state: GameState) {
    const ctx = state.decision_context;
    const legal = state.legal_actions;
    const badgeEl = document.getElementById("decision-type-badge");
    const L = engineLayout();

    if (!ctx || !legal || state.is_terminal || !L) {
      if (badgeEl) badgeEl.textContent = state.is_terminal ? t("hud:badge.terminal") : t("hud:badge.waiting");
      this.container.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 20px;">${L ? t("hud:noDecision") : t("hud:loadingEngine")}</div>`;
      return;
    }

    const dType = num(ctx.decision_type);
    const dTypeName = ctx.decision_type_name || namesByValue(L.decision_types)[dType] || `TYPE_${dType}`;
    const playerRaw = String(ctx.decision_player || "NONE");
    const player = escapeHtml(sideName(playerRaw));
    const phaseName = String(state.current_phase_name || state.phase_name || "ACTION");
    const card = (name: string) => L.card_ids[name] ?? -1;

    if (badgeEl) {
      badgeEl.textContent = enumLabel("decision", dTypeName);
      badgeEl.className = `badge badge-${playerRaw.toLowerCase().replace(/[^a-z]/g, "")}`;
    }

    // A player's view while the other side decides (game/view.ts): its options are its own.
    if (state.view_hidden_decision) {
      this.container.innerHTML = `<div class="hud-waiting">
        <div>${t("hud:hidden.toMove", { player, phase: escapeHtml(enumLabel("phase", phaseName)) })}</div>
        <div class="hud-waiting-note">${t("hud:hidden.note", { view: escapeHtml(sideName(String(state.view))) })}</div>
      </div>`;
      return;
    }

    let promptText = "";
    let buttonsHtml = "";
    let showDieSelector = false;

    const validIds = (legal.valid_ids || []).map(num);
    const allowEarlyStop = legal.allow_early_stop || ctx.allow_early_stop;
    const DT = L.decision_types;

    if (dType === DT.SELECT_CARD) {
      const resolvingCard = num(ctx.resolving_card || 0);
      const prompt = cardChoicePrompt(cardKey(L, resolvingCard), player);
      if (prompt) {
        promptText = prompt;
      } else if (resolvingCard > 0) {
        promptText = t("hud:prompt.resolvingEventCard", { player, card: resolvingCard });
      } else if (phaseName === "HEADLINE") {
        let oppHeadlineBanner = "";
        if (playerRaw === "US" && state.headline_ussr_card) {
          oppHeadlineBanner = `<div class="headline-reveal-box" style="background: rgba(220,53,69,0.15); border: 1px solid rgba(220,53,69,0.4); padding: 6px 10px; border-radius: 4px; margin-bottom: 8px; font-size: 13px;">${t("hud:prompt.headlineRevealed", { side: sideName("USSR"), card: num(state.headline_ussr_card) })}</div>`;
        } else if (playerRaw === "USSR" && state.headline_us_card) {
          oppHeadlineBanner = `<div class="headline-reveal-box" style="background: rgba(13,110,253,0.15); border: 1px solid rgba(13,110,253,0.4); padding: 6px 10px; border-radius: 4px; margin-bottom: 8px; font-size: 13px;">${t("hud:prompt.headlineRevealed", { side: sideName("US"), card: num(state.headline_us_card) })}</div>`;
        }
        promptText = `${oppHeadlineBanner}${t("hud:prompt.headline", { player })}`;
      } else {
        promptText = t("hud:prompt.actionRound", { player, round: num(state.action_round) });
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
        const passLabel = resolvingCard === card("OUR_MAN_IN_TEHRAN") ? t("hud:pass.ourManInTehran")
          : resolvingCard === card("SALT_NEGOTIATIONS") ? t("hud:pass.salt")
          : !validIds.some(id => id >= 1 && id <= 110) ? t("hud:pass.noCard")
          : t("hud:pass.default");
        buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">${passLabel}</button>`;
      }
    } else if (dType === DT.SELECT_PLAY_MODE) {
      const pendingId = num(ctx.pending_op_card);
      const playedName = escapeHtml(ctx.pending_op_card_name ? cardName(pendingId, ctx.pending_op_card_name) : t("hud:cardNumber", { id: pendingId }));
      // Grain Sales to Soviets resolves the card it drew on this node: play it, or hand it back to
      // the USSR and conduct Grain Sales' own 2 Ops. The engine offers the return as CONFIRM_DONE
      // (allow_early_stop), outside the play modes in valid_ids -- so without the button below the
      // page could only play the drawn card.
      const grainSales = num(ctx.resolving_card || 0) === card("GRAIN_SALES_TO_SOVIETS");
      promptText = grainSales
        ? t("hud:prompt.grainSales", { player, card: playedName })
        : t("hud:prompt.choosePlay", { player, card: playedName });
      // The Ops are known only once granted; at Grain Sales' node the drawn card has none yet.
      const opsValue = num(ctx.pending_ops_value);
      const ops = opsValue > 0 ? t("hud:mode.ops", { ops: opsValue }) : t("hud:mode.cardsOps");
      const modeNames = namesByValue(L.resolutions);
      validIds.forEach(m => {
        // Loud, not bland: a mode the engine has and this table has not means the Resolution enum
        // grew. The engine's own name for it is still shown.
        const cfg = modeLabel(modeNames[m] ?? "", ops) ?? {
          title: t("hud:mode.unlabelled.title", { mode: m, name: escapeHtml(modeNames[m] ?? "unknown") }),
          desc: t("hud:mode.unlabelled.desc"),
          class: "btn-warning",
        };
        buttonsHtml += `
          <button class="btn ${cfg.class} btn-block btn-hud-action btn-choice-card" data-primary="${m}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
            <div style="font-weight: bold; font-size: 13px;">${cfg.title}</div>
            <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${cfg.desc}</div>
          </button>
        `;
      });
      // Any of these can roll a die -- a Space Race attempt, a coup, a realignment, an event that
      // rolls -- and a forced die is used only if one is rolled.
      showDieSelector = true;
      if (grainSales && allowEarlyStop) {
        // The engine's own decoding of this node's CONFIRM_DONE: primary 255, the done flag.
        buttonsHtml += `
          <button class="btn btn-warning btn-block btn-hud-action" data-primary="255" data-flags="128" style="margin-top: 4px; text-align: left; padding: 8px 12px;">
            <div style="font-weight: bold; font-size: 13px;">${t("hud:grainSalesReturn.title", { card: playedName })}</div>
            <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${t("hud:grainSalesReturn.desc")}</div>
          </button>
        `;
      }
    } else if (dType === DT.CHOOSE_TIMING_BRANCH) {
      // RETIRED by P17 and unreachable: on an opponent card, Resolution::EVENT *is* the
      // event-first branch, so no separate timing decision is ever emitted. Kept because
      // a replay recorded before P17 can still contain one.
      const oppId = num(ctx.pending_op_card);
      const oppCardName = escapeHtml(ctx.pending_op_card_name ? cardName(oppId, ctx.pending_op_card_name) : t("hud:cardNumber", { id: oppId }));
      promptText = t("hud:prompt.timing", { player, card: oppCardName });
      if (validIds.includes(0)) {
        buttonsHtml += `
          <button class="btn btn-primary btn-block btn-hud-action btn-choice-card" data-primary="0" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
            <div style="font-weight: bold; font-size: 13px;">${t("hud:timing.opsFirst.title")}</div>
            <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${t("hud:timing.opsFirst.desc")}</div>
          </button>
        `;
      }
      if (validIds.includes(1)) {
        buttonsHtml += `
          <button class="btn btn-secondary btn-block btn-hud-action btn-choice-card" data-primary="1" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
            <div style="font-weight: bold; font-size: 13px;">${t("hud:timing.eventFirst.title")}</div>
            <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${t("hud:timing.eventFirst.desc")}</div>
          </button>
        `;
      }
    } else if (dType === DT.SELECT_OP_MODE) {
      promptText = t("hud:prompt.opMode", { player, ops: num(ctx.pending_ops_value) });
      const opNames = namesByValue(L.op_modes);
      validIds.forEach(m => {
        const cfg = opModeLabel(opNames[m] ?? "") ?? { text: t("hud:opMode.unknown", { mode: m }), desc: "", class: "btn-secondary" };
        buttonsHtml += `
          <button class="btn ${cfg.class} btn-block btn-hud-action btn-choice-card" data-primary="${m}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
            <div style="font-weight: bold;">${cfg.text}</div>
            <div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${cfg.desc}</div>
          </button>
        `;
      });
    } else if (dType === DT.POINT_NODE) {
      const opMode = ctx.op_mode;
      const isCoup = opMode === L.op_modes.COUP;
      const isRealign = opMode === L.op_modes.REALIGN;
      const resolving = num(ctx.resolving_card || 0);
      const remaining = num(ctx.remaining_steps);

      if (phaseName === "SETUP") {
        const region = regionName(playerRaw === "USSR" ? "Eastern Europe" : "Western Europe");
        promptText = t("hud:prompt.setup", { player, region, remaining });
      } else if (resolving > 0) {
        const resCard = escapeHtml(ctx.resolving_card_name ? cardName(resolving, ctx.resolving_card_name) : t("hud:cardNumber", { id: resolving }));
        promptText = t("hud:prompt.resolvingEvent", { player, card: resCard, remaining });
        // An event's target may be rolled for (a war, a coup the event grants).
        showDieSelector = true;
      } else if (isCoup) {
        promptText = t("hud:prompt.coup", { player, ops: num(ctx.pending_ops_value) });
        showDieSelector = true;
      } else if (isRealign) {
        promptText = t("hud:prompt.realign", { player });
        showDieSelector = true;
      } else {
        promptText = remaining > 0 ? t("hud:prompt.targetOpsLeft", { player, remaining }) : t("hud:prompt.target", { player });
      }

      // The targets the engine offers (its legal mask): the board's numbers for each, no more.
      if (validIds.length > 0 && (isCoup || isRealign || resolving > 0 || validIds.length <= 12)) {
        const byId = new Map(Object.values(state.countries || {}).map(c => [num(c.id), c]));
        let targetsHtml = "";
        validIds.filter(id => id < 84).forEach(cid => {
          const country = byId.get(cid);
          const cName = escapeHtml(country?.name ? countryName(cid, country.name) : t("hud:countryNumber", { id: cid }));
          const cStab = num(country?.stability ?? 0);
          const isBg = !!country?.battleground;
          const usInf = num(country?.us_influence ?? 0);
          const ussrInf = num(country?.ussr_influence ?? 0);
          targetsHtml += `
            <button class="btn btn-secondary btn-block btn-hud-action btn-target-card" data-primary="${cid}" data-secondary="${this.selectedDieRoll}" style="margin-bottom: 6px; text-align: left; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.12);">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; font-size: 13px;">${isBg ? '★ ' : ''}${t("hud:target.name", { name: cName, stab: cStab })}</span>
                <span style="font-size: 11px; opacity: 0.85;">${t("hud:target.influence", { us: usInf, ussr: ussrInf })}</span>
              </div>
            </button>
          `;
        });

        if (targetsHtml) {
          buttonsHtml += `
            <div class="hud-target-list-header" style="font-size: 11px; font-weight: 700; color: var(--text-dim); margin: 8px 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px;">
              ${t("hud:target.header", { n: validIds.length })}
            </div>
            <div class="hud-targets-scroll-list" style="max-height: 200px; overflow-y: auto; padding-right: 2px;">
              ${targetsHtml}
            </div>
          `;
        }
      }

      if (allowEarlyStop) {
        buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">${t("hud:pass.donePassRemaining")}</button>`;
      } else if (legal.can_confirm_done) {
        // No eligible target is left for what remains (the engine's flat mask offers only
        // CONFIRM_DONE): without this the panel was empty and the game could not go on.
        promptText += ` ${t("hud:prompt.noTargetLeft")}`;
        buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">${t("hud:pass.doneNoTarget")}</button>`;
      }
    } else if (dType === DT.CHOOSE_BRANCH) {
      const resolvingCardId = num(ctx.resolving_card || ctx.pending_op_card);
      const resolvingCardName = escapeHtml(ctx.resolving_card_name ? cardName(resolvingCardId, ctx.resolving_card_name)
        : resolvingCardId ? t("hud:cardNumber", { id: resolvingCardId }) : t("hud:event"));
      // What each branch means comes from game/branches.ts (a region for Chernobyl, the DEFCON level
      // for Summit and How I Learned...), the same text the action log uses -- never a bare number.
      promptText = t("hud:prompt.branch", { player, card: resolvingCardName, ask: branchAsk(cardKey(L, resolvingCardId)) });
      const actionLabels = legal.valid_action_labels || {};

      validIds.forEach(b => {
        const meta = branchMeta(resolvingCardId, b, state);
        // An event this page has no words for still shows the engine's own label, never just "Option N".
        const engineLabel = actionLabels[b.toString()];
        const branchTitle = escapeHtml(meta ? render(meta.title) : engineLabel ? `${engineLabel} ${b}` : t("hud:optionNumber", { n: b }));
        const branchDesc = escapeHtml(meta ? render(meta.desc) : "");

        buttonsHtml += `
          <button class="btn btn-secondary btn-block btn-hud-action btn-choice-card" data-primary="${b}" style="margin-bottom: 8px; text-align: left; padding: 8px 12px;">
            <div style="font-weight: bold; font-size: 13px;">${branchTitle}</div>
            ${branchDesc ? `<div style="font-size: 11px; opacity: 0.85; margin-top: 2px;">${branchDesc}</div>` : ""}
          </button>
        `;
      });

      if (allowEarlyStop) {
        buttonsHtml += `<button class="btn btn-warning btn-block btn-hud-action" data-flags="128" data-primary="0" style="margin-top: 8px;">${t("hud:pass.confirmPass")}</button>`;
      }
    } else if (dType === DT.NONE) {
      promptText = t("hud:prompt.idle");
    } else {
      promptText = t("hud:prompt.waiting");
    }

    let dieSelectorHtml = "";
    if (showDieSelector) {
      dieSelectorHtml = `
        <div class="die-roll-selector-container">
          <div class="die-roll-header">
            <span>${t("hud:die.input")}</span>
            <span class="die-selected-text">${this.selectedDieRoll === 0 ? t("hud:die.autoRoll") : t("hud:die.manual", { roll: this.selectedDieRoll })}</span>
          </div>
          <div class="die-buttons-row">
            <button class="btn-die-opt ${this.selectedDieRoll === 0 ? "active" : ""}" data-roll="0">${t("hud:die.auto")}</button>
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
}
