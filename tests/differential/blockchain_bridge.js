/**
 * Headless Node.js bridge for ts-blockchain (twilight.js).
 * Communicates via JSON-RPC over stdin/stdout.
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const Module = require('module');

// Silence twilight.js console.log output to keep stdout clean for JSON-RPC
const originalConsoleLog = console.log;
console.log = function(...args) {
  // Redirect to stderr for debugging if needed
};

// 1. Hook Module.prototype.require to shim Saito framework dependencies
const originalRequire = Module.prototype.require;

function MockGame(app) {
  this.name = "Game";
  this.game = {
    id: "blockchain_differential_test",
    player: 1,
    players: ["1", "2"],
    opponents: ["2"],
    status: "",
    over: 0,
    winner: 0,
    turn: [],
    queue: [],
    deck: [{
      cards: {},
      hand: ["china"],
      discards: {},
      removed: {},
      crypt: []
    }],
    options: {},
    dice: "",
    log: []
  };
  this.moves = [];
}

let lastStatusHtml = "";
let cardClickHandler = null;
let cardClickOptions = [];

MockGame.prototype.updateStatus = function(msg) { lastStatusHtml = msg || ""; };
MockGame.prototype.updateStatusAndListCards = function() {};
MockGame.prototype.updateLog = function() {};
MockGame.prototype.displayModal = function() {};
MockGame.prototype.saveGame = function() {};
MockGame.prototype.saveGamePreference = function() {};
MockGame.prototype.initializeDice = function() {};
MockGame.prototype.sendMessage = function() {};
MockGame.prototype.scale = function(v) { return v; };
MockGame.prototype.returnNextPlayer = function(p) { return p === 1 ? 2 : 1; };
MockGame.prototype.resignGame = function() {};
MockGame.prototype.updateVictoryPoints = function() {};
MockGame.prototype.updateDefcon = function() {};
MockGame.prototype.lowerDefcon = function() { this.game.state.defcon--; };
MockGame.prototype.endTurn = function() {};
MockGame.prototype.addMove = function(mv) {
  if (typeof mv === "string") {
    const parts = mv.split("	");
    if (parts[0] === "milops") {
      const p = parts[1].toLowerCase();
      const amt = parseInt(parts[2], 10);
      if (p === "us") this.game.state.milops_us += amt;
      if (p === "ussr") this.game.state.milops_ussr += amt;
    }
    if (parts[0] === "vp") {
      const p = parts[1].toLowerCase();
      const amt = parseInt(parts[2], 10);
      if (p === "us") this.game.state.vp += amt;
      if (p === "ussr") this.game.state.vp -= amt;
    }
  }
};
MockGame.prototype.playerFinishedPlacingInfluence = function() {};
MockGame.prototype.rollDice = function(sides = 6) { return Math.floor(Math.random() * sides) + 1; };

// Proxy for jQuery DOM operations with click handler interception
const dummyProxy = new Proxy(function() { return dummyProxy; }, {
  get: (target, prop) => {
    if (prop === Symbol.toPrimitive || prop === "toString") return () => "";
    return dummyProxy;
  },
  apply: () => dummyProxy
});

const registeredClickHandlers = {};

global.$ = function(selector) {
  if (selector && typeof selector === "object" && selector.id) {
    selector = "#" + selector.id;
  }
  if (typeof selector === "string") {
    return new Proxy(function() {}, {
      get: (target, prop) => {
        if (prop === "on") {
          return (evt, handler) => {
            if (evt === "click") {
              if (selector === ".card") {
                cardClickHandler = handler;
                cardClickOptions = [];
                const regex = /id=["']([^"']+)["'][^>]*>([^<]+)<\/li>/gi;
                let match;
                const statusHtml = (gameInstance && gameInstance.game && gameInstance.game.status) || "";
                while ((match = regex.exec(statusHtml)) !== null) {
                  cardClickOptions.push({ id: match[1], name: match[2].trim() });
                }
              } else {
                registeredClickHandlers[selector] = handler;
              }
            }
            return dummyProxy;
          };
        }
        if (prop === "off") {
          return () => {
            delete registeredClickHandlers[selector];
            return dummyProxy;
          };
        }
        if (prop === "attr") {
          return (attrName) => (attrName === "id" ? selector.replace("#", "") : "");
        }
        return dummyProxy;
      },
      apply: () => dummyProxy
    });
  }
  return dummyProxy;
};
global.document = { getElementById: () => null };
global.confirm = () => true;
global.alert = () => {};

Module.prototype.require = function(reqPath) {
  if (reqPath.includes('lib/saito/saito')) return {};
  if (reqPath.includes('lib/templates/game')) return MockGame;
  return originalRequire.apply(this, arguments);
};

// 2. Load Twilight from external submodule
const twilightPath = path.resolve(__dirname, '../external/ts-blockchain/twilight.js');
const Twilight = require(twilightPath);

const origAddMove = Twilight.prototype.addMove;
Twilight.prototype.addMove = function(mv) {
  if (typeof mv === "string") {
    const parts = mv.split("	");
    if (parts[0] === "milops") {
      const p = parts[1].toLowerCase();
      const amt = parseInt(parts[2], 10);
      if (p === "us") this.game.state.milops_us += amt;
      if (p === "ussr") this.game.state.milops_ussr += amt;
    }
    if (parts[0] === "vp") {
      const p = parts[1].toLowerCase();
      const amt = parseInt(parts[2], 10);
      if (p === "us") this.game.state.vp += amt;
      if (p === "ussr") this.game.state.vp -= amt;
    }
  }
  return origAddMove.call(this, mv);
};

let gameInstance = null;
let customDiceRolls = [];

function createGameInstance() {
  const app = {
    wallet: { returnPublicKey: () => "pk1" },
    options: {}
  };
  const t = new Twilight(app);
  t.rollDice = function(sides = 6) {
    if (customDiceRolls.length > 0) {
      return customDiceRolls.shift();
    }
    return Math.floor(Math.random() * sides) + 1;
  };
  t.initializeGame("blockchain_differential_test");
  t.game.deck[0].cards = Object.assign({}, t.returnEarlyWarCards(), t.returnMidWarCards(), t.returnLateWarCards());
  t.game.deck[0].cards["china"] = t.returnChinaCard();
  return t;
}

gameInstance = createGameInstance();

// Helper to determine coup target legality matching twilight.js logic
function isLegalCoupTarget(t, player, countryname, ops, card) {
  const c = t.countries[countryname];
  if (!c) return false;

  // Must have opponent influence
  if (player === "us") {
    if (c.ussr <= 0) return false;
  } else {
    if (c.us <= 0) return false;
  }

  // Coup Restrictions
  if (t.game.state.events.usjapan === 1 && countryname === "japan") {
    return false;
  }

  if (t.game.state.limit_ignoredefcon === 0) {
    if (t.game.state.limit_region && t.game.state.limit_region.indexOf(c.region) > -1) {
      return false;
    }
    if (c.region === "europe" && t.game.state.defcon < 5) return false;
    if (c.region === "asia" && t.game.state.defcon < 4) return false;
    if (c.region === "seasia" && t.game.state.defcon < 4) return false;
    if (c.region === "mideast" && t.game.state.defcon < 3) return false;
  }

  // NATO
  if (c.region === "europe" && t.game.state.events.nato === 1 && player === "ussr") {
    if (t.isControlled("us", countryname) === 1) {
      if ((countryname === "westgermany" && t.game.state.events.nato_westgermany === 0) ||
          (countryname === "france" && t.game.state.events.nato_france === 0)) {
        // exempt
      } else {
        return false;
      }
    }
  }

  return true;
}

// Helper to determine realignment target legality matching twilight.js logic
function isLegalRealignTarget(t, player, countryname, card) {
  const c = t.countries[countryname];
  if (!c) return false;

  // Opponent influence check
  if (player === "ussr") {
    if (c.us < 1) return false;
  } else {
    if (c.ussr < 1) return false;
  }

  if (t.game.state.limit_region && t.game.state.limit_region.indexOf(c.region) > -1) {
    return false;
  }

  if (t.game.state.limit_ignoredefcon === 0) {
    if (c.region === "europe" && t.game.state.defcon < 5) return false;
    if (c.region === "asia" && t.game.state.defcon < 4) return false;
    if (c.region === "seasia" && t.game.state.defcon < 4) return false;
    if (c.region === "mideast" && t.game.state.defcon < 3) return false;
  }

  if (card === "junta" && (c.region !== "camerica" && c.region !== "samerica")) {
    return false;
  }

  if (t.game.state.events.usjapan === 1 && countryname === "japan" && player === "ussr") {
    return false;
  }

  return true;
}

// 3. Command dispatcher
function handleCommand(req) {
  const { cmd, args = {} } = req;
  const t = gameInstance;

  switch (cmd) {
    case "init_game": {
      customDiceRolls.length = 0;
      cardClickHandler = null;
      cardClickOptions = [];
      lastStatusHtml = "";
      for (const k of Object.keys(registeredClickHandlers)) delete registeredClickHandlers[k];
      gameInstance = createGameInstance();
      if (args.defcon !== undefined) gameInstance.game.state.defcon = args.defcon;
      if (args.turn !== undefined) gameInstance.game.state.turn = args.turn;
      if (args.vp !== undefined) gameInstance.game.state.vp = args.vp;
      return { status: "ok" };
    }

    case "get_state": {
      const countries = {};
      for (const [k, v] of Object.entries(t.countries)) {
        countries[k] = {
          us: v.us,
          ussr: v.ussr,
          control: v.control,
          bg: v.bg,
          region: v.region,
          neighbours: v.neighbours,
          name: v.name
        };
      }
      return {
        status: "ok",
        state: {
          countries,
          defcon: t.game.state.defcon,
          vp: t.game.state.vp,
          milops_us: t.game.state.milops_us,
          milops_ussr: t.game.state.milops_ussr,
          space_race_us: t.game.state.space_race_us,
          space_race_ussr: t.game.state.space_race_ussr,
          turn: t.game.state.turn,
          round: t.game.state.round,
          china_card_holder: t.whoHasTheChinaCard(),
          china_card_playable: t.game.state.events.china_card_eligible,
          events: t.game.state.events,
          discards: Object.keys(t.game.deck[0].discards || {}),
          hands: {
            us: (t.game.player === 2 ? t.game.deck[0].hand : (t.game.state.us_hand || [])),
            ussr: (t.game.player === 1 ? t.game.deck[0].hand : (t.game.state.ussr_hand || []))
          }
        }
      };
    }

    case "sync_state": {
      if (args.countries) {
        for (const [k, v] of Object.entries(args.countries)) {
          if (t.countries[k]) {
            if (v.us !== undefined) t.countries[k].us = v.us;
            if (v.ussr !== undefined) t.countries[k].ussr = v.ussr;
          }
        }
      }
      if (args.defcon !== undefined) t.game.state.defcon = args.defcon;
      if (args.vp !== undefined) t.game.state.vp = args.vp;
      if (args.milops_us !== undefined) t.game.state.milops_us = args.milops_us;
      if (args.milops_ussr !== undefined) t.game.state.milops_ussr = args.milops_ussr;
      if (args.space_race_us !== undefined) t.game.state.space_race_us = args.space_race_us;
      if (args.space_race_ussr !== undefined) t.game.state.space_race_ussr = args.space_race_ussr;
      if (args.turn !== undefined) t.game.state.turn = args.turn;
      if (args.round !== undefined) t.game.state.round = args.round;
      if (args.headline !== undefined) t.game.state.headline = args.headline;
      if (args.china_card_holder !== undefined) {
        t.game.deck[0].hand = t.game.deck[0].hand.filter(c => c !== "china");
        if (args.china_card_holder === "ussr") {
          t.game.state.events.china_card = 1;
          if (t.game.player === 1) t.game.deck[0].hand.push("china");
        } else {
          t.game.state.events.china_card = 2;
          if (t.game.player === 2) t.game.deck[0].hand.push("china");
        }
      }
      if (args.china_card_playable !== undefined) {
        t.game.state.events.china_card_eligible = args.china_card_playable ? 1 : 0;
      }
      if (args.hands) {
        if (args.hands.us) {
          t.game.state.us_hand = [...args.hands.us];
          if (t.game.player === 2) t.game.deck[0].hand = [...args.hands.us];
        }
        if (args.hands.ussr) {
          t.game.state.ussr_hand = [...args.hands.ussr];
          if (t.game.player === 1) t.game.deck[0].hand = [...args.hands.ussr];
        }
      }
      if (args.discards) {
        t.game.deck[0].discards = {};
        for (const c of args.discards) {
          if (t.game.deck[0].cards[c]) t.game.deck[0].discards[c] = t.game.deck[0].cards[c];
        }
      }
      if (args.events) {
        Object.assign(t.game.state.events, args.events);
      }
      return { status: "ok" };
    }

    case "get_legal_placements": {
      const player = args.player.toLowerCase();
      t.prePlayerPlaceInfluence(player);
      const placeable = [];
      for (const [k, v] of Object.entries(t.game.countries)) {
        if (v.place === 1) placeable.push(k);
      }
      return { status: "ok", countries: placeable };
    }

    case "get_legal_coups": {
      const player = args.player.toLowerCase();
      const ops = args.ops || 1;
      const card = args.card || "";
      const valid = [];
      for (const k of Object.keys(t.countries)) {
        if (isLegalCoupTarget(t, player, k, ops, card)) {
          valid.push(k);
        }
      }
      return { status: "ok", countries: valid };
    }

    case "get_legal_realignments": {
      const player = args.player.toLowerCase();
      const card = args.card || "";
      const valid = [];
      for (const k of Object.keys(t.countries)) {
        if (isLegalRealignTarget(t, player, k, card)) {
          valid.push(k);
        }
      }
      return { status: "ok", countries: valid };
    }

    case "get_legal_setup": {
      const player = args.player.toLowerCase();
      if (player === "ussr") {
        return {
          status: "ok",
          countries: ["finland", "eastgermany", "poland", "austria", "czechoslovakia", "hungary", "romania", "yugoslavia", "bulgaria"]
        };
      } else {
        return {
          status: "ok",
          countries: ["canada", "uk", "benelux", "france", "italy", "westgermany", "greece", "spain", "turkey", "austria", "norway", "denmark", "sweden", "finland"]
        };
      }
    }

    case "place_influence": {
      const { country, inf, player } = args;
      t.placeInfluence(country, inf, player.toLowerCase());
      return { status: "ok" };
    }

    case "remove_influence": {
      const { country, inf, player } = args;
      t.removeInfluence(country, inf, player.toLowerCase());
      return { status: "ok" };
    }

    case "resolve_coup": {
      const { player, country, ops, roll } = args;
      customDiceRolls.push(roll);
      t.playCoup(player.toLowerCase(), country, ops);
      const p = player.toLowerCase();
      if (p === "us") {
        t.game.state.milops_us = Math.min(5, (t.game.state.milops_us || 0) + ops);
      } else {
        t.game.state.milops_ussr = Math.min(5, (t.game.state.milops_ussr || 0) + ops);
      }
      return { status: "ok" };
    }

    case "resolve_realignment": {
      const { country, us_roll, ussr_roll } = args;
      customDiceRolls.push(us_roll, ussr_roll);
      t.playRealign(country);
      return { status: "ok" };
    }

    case "score_region": {
      const { region } = args;
      const vpBefore = t.game.state.vp;
      t.scoreRegion(region);
      const vpDelta = t.game.state.vp - vpBefore;
      return { status: "ok", vp_delta: vpDelta, vp_after: t.game.state.vp };
    }

    case "calculate_scoring": {
      const { region } = args;
      const breakdown = t.calculateScoring(region);
      return { status: "ok", breakdown };
    }

    case "lower_defcon": {
      t.lowerDefcon();
      return { status: "ok", defcon: t.game.state.defcon };
    }

    case "get_pending_decision": {
      // 0. Card option click handlers (e.g. Indo-Pakistani War):
      if (cardClickHandler && cardClickOptions.length > 0) {
        return {
          status: "ok",
          pending: true,
          type: "country",
          targets: cardClickOptions.map(o => o.name)
        };
      }
      // 1. NORAD bonus:
      if (t.game.state.us_defcon_bonus === 1 && t.isControlled("us", "canada") === 1) {
        const placeable = [];
        for (const [k, v] of Object.entries(t.countries)) {
          if (v.us > 0) placeable.push(k);
        }
        return {
          status: "ok",
          pending: true,
          type: "country",
          targets: placeable
        };
      }
      // 2. Interactive event targets (countries with place === 1):
      const placeable = [];
      for (const [k, v] of Object.entries(t.countries)) {
        if (v.place === 1) placeable.push(k);
      }
      if (placeable.length > 0) {
        return {
          status: "ok",
          pending: true,
          type: "country",
          targets: placeable
        };
      }
      // 3. Registered click handlers (e.g. Truman):
      const clickTargets = Object.keys(registeredClickHandlers).map(s => s.replace("#", ""));
      if (clickTargets.length > 0) {
        return {
          status: "ok",
          pending: true,
          type: "country",
          targets: clickTargets
        };
      }
      return {
        status: "ok",
        pending: false,
        type: "none",
        targets: []
      };
    }

    case "resolve_decision": {
      const { choice } = args;
      // 0. Card option click handlers:
      if (cardClickHandler && cardClickOptions.length > 0) {
        const targetOption = cardClickOptions.find(o =>
          o.name.toLowerCase() === String(choice).toLowerCase() ||
          o.id.toLowerCase().includes(String(choice).toLowerCase())
        );
        if (targetOption) {
          const fn = cardClickHandler;
          cardClickHandler = null;
          cardClickOptions = [];
          fn.call({ id: targetOption.id });
          return { status: "ok" };
        }
      }
      // 1. NORAD bonus resolution:
      if (t.game.state.us_defcon_bonus === 1 && t.isControlled("us", "canada") === 1) {
        const c = String(choice).toLowerCase().replace(/[^a-z]/g, "");
        if (t.countries[c]) {
          t.placeInfluence(c, 1, "us");
          t.game.state.us_defcon_bonus = 0;
          return { status: "ok" };
        }
      }
      // 2. Interactive event click resolution:
      const c = String(choice).toLowerCase().replace(/[^a-z]/g, "");
      if (registeredClickHandlers['#' + c]) {
        const fn = registeredClickHandlers['#' + c];
        delete registeredClickHandlers['#' + c];
        fn.call({ id: c });
        if (t.countries[c]) t.countries[c].place = 0;
        return { status: "ok" };
      }
      if (t.countries[c] && t.countries[c].place === 1) {
        t.countries[c].place = 0;
        return { status: "ok" };
      }
      // 3. Dice roll resolution:
      if (typeof choice === "number" || (!isNaN(Number(choice)) && Number(choice) >= 1 && Number(choice) <= 6)) {
        customDiceRolls.push(Number(choice));
        return { status: "ok" };
      }
      return { status: "ok" };
    }

    case "advance_space_race": {
      const { player } = args;
      t.advanceSpaceRace(player.toLowerCase());
      return {
        status: "ok",
        space_race_us: t.game.state.space_race_us,
        space_race_ussr: t.game.state.space_race_ussr,
        vp: t.game.state.vp
      };
    }

    case "play_event": {
      const { player, card } = args;
      const p = player.toLowerCase();
      t.game.player = (p === "us" ? 2 : 1);
      t.playEvent(p, card.toLowerCase());
      return { status: "ok" };
    }

    case "get_card_info": {
      const early = t.returnEarlyWarCards();
      const mid = t.returnMidWarCards();
      const late = t.returnLateWarCards();
      const all = Object.assign({}, early, mid, late);
      all["china"] = t.returnChinaCard();
      return { status: "ok", cards: all };
    }

    default:
      return { status: "error", message: `Unknown command: ${cmd}` };
  }
}

// 4. Line reader loop for stdio IPC
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  try {
    const req = JSON.parse(trimmed);
    const res = handleCommand(req);
    process.stdout.write(JSON.stringify(res) + '\n');
  } catch (err) {
    process.stdout.write(JSON.stringify({ status: "error", message: err.message, stack: err.stack }) + '\n');
  }
});
