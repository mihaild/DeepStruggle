/**
 * The Elo leaderboard page (leaderboard.html). Its data, leaderboard.json, is fitted from the
 * records in leaderboard/ by `tools/leaderboard.py fit` when the page is built
 * (tools/scripts/build_web.sh), so the page computes nothing: a view -- the main players, or the
 * main players plus some lineages -- is a filter over that one fit, and a rating is the same
 * number in every view. The view is kept in the link, so a filtered table can be shared.
 *
 * A second tab lists the architecture and recipe codes the players' names are written in, with each
 * code's description and flags, read from research/run_codes.json by the same fit (`codes`).
 *
 * A player's `play` link opens the workbench with its network and its search (`workbench_search`,
 * written by the fit). The player that is the Hugging Face repo's default -- its default model, played
 * with its default search (models.json, set by `tools/publish_hf.py default --player`) -- is tagged,
 * read from the repo when the page opens rather than fitted in, so the tag follows the repo without a
 * rebuild. So are the players twilightstruggle.app offers as its AI's levels (site_models.json in the
 * same repo): each is tagged with its level, and its card adds the site's description of it. Either
 * read failing leaves the page without those tags, and nothing else.
 *
 * When the build could read the stats database, leaderboard.json also holds each player's estimate on
 * Playdek's scale (tools/lib/playdek_estimate.py), and a switch shows those in the table. It is one
 * estimate per player whatever the epoch shown, and is not the anchored Elo shifted: games against
 * people move a player and, more loosely, its network's other players away from their leaderboard gaps.
 */

import { SiteLevel, fetchSiteLevels, hfCatalog } from "./analysis/hf_catalog";
import { bindLanguageSelects, initLanguage, onLanguageChange, t } from "./i18n";
import { DEFAULT_ROLLOUT, RolloutSpec, parseSearchParam, rolloutParam, specEqual } from "./analysis/rollout_spec";

interface Seat { w: number; l: number; d: number }

interface Network {
  file: string;
  sha256: string;
  description: string;
  old_names: string[];
  /** The .pt's path in the Hugging Face repo once published; its .onnx sits beside it. */
  hf: string | null;
  /** The behaviour report, a path in the GitHub repository. */
  report: string | null;
}

interface Player {
  network: string | null;
  lineage: string | null;
  kind: string;
  inference: Record<string, unknown>;
  description: string;
  /** The report on this player's play (tools/lib/leaderboard.py player_report): its own, or for the
   * greedy policy its network's; null for a search or a sampled policy that has none of its own. */
  report: string | null;
  /** How the workbench plays it, as its `search=` value ("policy" or a rollout spec, every field
   *  spelled out); null for a search it cannot run (Gumbel, PUCT, a sampled policy, another leaf network). */
  workbench_search: string | null;
}

interface Rating {
  player: string;
  elo: number;
  se: number;
  main: boolean;
  games: number;
  score: number;
  opponents: number;
  main_opponents: number;
}

interface Pair { a: string; b: string; a_as_us: Seat; a_as_ussr: Seat; records: string[] }

interface Epoch {
  description: string;
  /** The id of the epoch's engine behaviour digest (tools/lib/engine_behaviour.py). */
  behaviour: string;
  anchor: string;
  anchor_elo: number;
  main: string[];
  seat_us: number;
  seat_us_se: number;
  unrated: string[];
  ratings: Rating[];
  pairs: Pair[];
}

/** One architecture or recipe code (research/run_codes.json). */
interface CodeInfo {
  description: string;
  flags: Record<string, unknown>;
  /** The first run directory trained under it. */
  first: string | null;
  /** The networks on the board trained under it (the last code of its kind in the name). */
  networks: string[];
}

interface Codes {
  architectures: Record<string, CodeInfo>;
  recipes: Record<string, CodeInfo>;
  by_network: Record<string, { architecture: string | null; recipe: string | null }>;
}

/** One player on Playdek's scale; its record is the model's, against people. */
interface PlayerEstimate {
  rating: number;
  se: number;
  lo: number;
  hi: number;
  games: number;
  humans: number;
  w: number;
  d: number;
  l: number;
}

interface PlaydekEstimate {
  as_of: string;
  games: number;
  humans: number;
  excluded: Record<string, number>;
  offset: number;
  network_sd: number;
  config_sd: number;
  reference_epoch: string;
  players: Record<string, PlayerEstimate>;
}

interface LeaderboardData {
  schema: number;
  commit: string | null;
  hf_repo: string;
  github: string;
  networks: Record<string, Network>;
  players: Record<string, Player>;
  epochs: Record<string, Epoch>;
  /** Absent from a leaderboard.json fitted before the codes tab existed. */
  codes?: Codes;
  /** Absent when the build could not read the stats database. */
  playdek?: PlaydekEstimate;
}

type View = "main" | "lineage" | "all";
type Tab = "ratings" | "codes";
type Mode = "anchor" | "playdek";

/** `--flag value` as the command line spells it: `--x` for true, `--no-x` for false, lists spaced. */
function flagText(name: string, value: unknown): string {
  const flag = "--" + name.replace(/_/g, "-");
  if (value === true) return flag;
  if (value === false) return `--no-${name.replace(/_/g, "-")}`;
  if (Array.isArray(value)) return `${flag} ${value.join(" ")}`;
  return `${flag} ${String(value)}`;
}

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

/** The page opens on the newest epoch with a game played in it: a new engine epoch starts with its
 *  anchor alone and nothing to show, and the one before it stays the board until games are played. */
function defaultEpoch(epochs: Record<string, { pairs: unknown[] }>): string {
  const names = Object.keys(epochs);
  const played = names.filter(n => epochs[n].pairs.length > 0);
  return played.length ? played[played.length - 1] : names[names.length - 1];
}

const hfTag = () => `<span class="lb-tag hf" title="${esc(t("leaderboard:tag.hfDefaultTitle"))}">${t("leaderboard:tag.hfDefault")}</span>`;

const SITE_URL = "https://twilightstruggle.app/ai/";

function siteTag(level: SiteLevel): string {
  return `<span class="lb-tag site" title="${esc(t("leaderboard:tag.siteTitle", { level: level.name }))}">${
    t("leaderboard:tag.site", { level: esc(level.name.toUpperCase()) })}</span>`;
}

/** A player's kind as its tag says it; none for the greedy policy, the kind itself for one not listed. */
function kindLabel(kind: string): string {
  switch (kind) {
    case "policy": return "";
    case "gumbel": return t("leaderboard:kind.gumbel");
    case "rollout": return t("leaderboard:kind.rollout");
    case "search": return t("leaderboard:kind.search");
    case "bot": return t("leaderboard:kind.bot");
    default: return kind;
  }
}

class LeaderboardPage {
  private epoch = "";
  private view: View = "main";
  private lineages = new Set<string>();
  private selected: string | null = null;
  private tab: Tab = "ratings";
  private allCodes = false;
  private mode: Mode = "anchor";
  /** The Hugging Face repo's default model (an .onnx path), once read; null until then or if none. */
  private hfDefault: string | null = null;
  /** Its default search (null: the policy): the repo's, else the workbench's built-in one. */
  private hfDefaultSearch: RolloutSpec | null = DEFAULT_ROLLOUT;
  /** twilightstruggle.app's levels, once read; empty until then or if they cannot be. */
  private siteLevels: SiteLevel[] = [];

  constructor(private readonly data: LeaderboardData) {
    const params = new URLSearchParams(location.search);
    const names = Object.keys(data.epochs);
    this.epoch = names.includes(params.get("epoch") ?? "") ? params.get("epoch")! : defaultEpoch(data.epochs);
    const v = params.get("view");
    this.view = v === "lineage" || v === "all" ? v : "main";
    for (const l of (params.get("lineage") ?? "").split(",")) if (l) this.lineages.add(l);
    this.selected = params.get("player");
    this.tab = params.get("tab") === "codes" ? "codes" : "ratings";
    this.allCodes = params.get("codes") === "all";
    this.mode = data.playdek && params.get("scale") === "playdek" ? "playdek" : "anchor";
    $("lb-scale-mode").hidden = !data.playdek;
    for (const r of document.querySelectorAll<HTMLInputElement>("input[name=lb-mode]")) {
      r.checked = r.value === this.mode;
      r.addEventListener("change", () => { if (r.checked) { this.mode = r.value as Mode; this.render(); } });
    }

    const sel = $<HTMLSelectElement>("lb-epoch");
    sel.innerHTML = names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
    sel.value = this.epoch;
    sel.addEventListener("change", () => { this.epoch = sel.value; this.selected = null; this.render(); });
    for (const r of document.querySelectorAll<HTMLInputElement>("input[name=lb-view]")) {
      r.checked = r.value === this.view;
      r.addEventListener("change", () => { if (r.checked) { this.view = r.value as View; this.render(); } });
    }
    $("lb-lineages").addEventListener("change", (ev) => {
      const box = ev.target as HTMLInputElement;
      if (box.checked) this.lineages.add(box.value); else this.lineages.delete(box.value);
      this.render();
    });
    for (const b of document.querySelectorAll<HTMLButtonElement>(".lb-tabs button")) {
      b.addEventListener("click", () => { this.tab = b.dataset.tab as Tab; this.render(); });
    }
    // A code named anywhere on the page opens the codes tab at its row.
    document.addEventListener("click", (ev) => {
      const a = (ev.target as HTMLElement).closest<HTMLElement>("a[data-code]");
      if (!a) return;
      ev.preventDefault();
      this.tab = "codes";
      this.render();
      document.getElementById(`code-${a.dataset.code}`)?.scrollIntoView({ block: "center" });
    });
    $("lb-tab-codes").addEventListener("change", (ev) => {
      const box = ev.target as HTMLInputElement;
      if (box.id === "lb-codes-all") { this.allCodes = box.checked; this.render(); }
    });
    $("lb-body").addEventListener("click", (ev) => {
      if ((ev.target as HTMLElement).closest("a")) return;
      const row = (ev.target as HTMLElement).closest<HTMLElement>("tr[data-player]");
      if (!row) return;
      const p = row.dataset.player!;
      this.selected = this.selected === p ? null : p;
      this.render();
    });
    this.render();
    void this.loadHfDefault();
    void this.loadSiteLevels();
  }

  /** twilightstruggle.app's levels, read after the first render as the default is: without them the
   *  page has no site tags. */
  private async loadSiteLevels(): Promise<void> {
    try {
      this.siteLevels = await fetchSiteLevels(this.data.hf_repo);
    } catch (e) {
      console.warn(`leaderboard: twilightstruggle.app's levels could not be read: ${e instanceof Error ? e.message : e}`);
      return;
    }
    if (this.siteLevels.length) this.render();
  }

  /** The site's levels that are `pid`: its network the level's model (the .onnx beside the .pt), played
   *  with the level's search. */
  private siteLevelsOf(pid: string): SiteLevel[] {
    const pl = this.data.players[pid];
    const hf = pl?.network ? this.data.networks[pl.network]?.hf : null;
    if (!hf || !pl.workbench_search) return [];
    const search = parseSearchParam(pl.workbench_search);
    if (search === undefined) return [];
    return this.siteLevels.filter(l => {
      if (l.path !== hf.replace(/\.pt$/, ".onnx")) return false;
      const want = parseSearchParam(l.search);
      return want !== undefined && (search === null || want === null ? search === want : specEqual(search, want));
    });
  }

  /** The repo's default model, read after the first render: the table does not wait for Hugging
   *  Face, and without it (offline, the repo unreachable) the page simply has no tag. */
  private async loadHfDefault(): Promise<void> {
    try {
      const catalog = await hfCatalog(this.data.hf_repo, "main");
      const search = catalog.defaultSearch === null ? DEFAULT_ROLLOUT : parseSearchParam(catalog.defaultSearch);
      if (search === undefined) throw new Error(`its default search ${catalog.defaultSearch} is not a search= value`);
      this.hfDefault = catalog.defaultModel;
      this.hfDefaultSearch = search;
    } catch (e) {
      console.warn(`leaderboard: the Hugging Face default model could not be read: ${e instanceof Error ? e.message : e}`);
      return;
    }
    this.render();
  }

  /** Whether `pid` is the repo's default: its network the default model (the .onnx beside the .pt),
   *  played with the default search. */
  private isHfDefault(pid: string): boolean {
    const pl = this.data.players[pid];
    const hf = pl?.network ? this.data.networks[pl.network]?.hf : null;
    if (!hf || !this.hfDefault || hf.replace(/\.pt$/, ".onnx") !== this.hfDefault || !pl.workbench_search) return false;
    const search = parseSearchParam(pl.workbench_search);
    const want = this.hfDefaultSearch;
    return search !== undefined && (search === null || want === null ? search === want : specEqual(search, want));
  }

  private get ep(): Epoch { return this.data.epochs[this.epoch]; }

  private ratingsShown(): Rating[] {
    const ep = this.ep;
    const shown = this.view === "all" ? ep.ratings
      : this.view === "main" ? ep.ratings.filter(r => r.main)
      : ep.ratings.filter(r => r.main || this.lineages.has(this.data.players[r.player]?.lineage ?? ""));
    if (this.mode !== "playdek") return shown;
    // Its own order: games against people can reorder players. One it cannot place goes last.
    const at = (r: Rating) => this.estimateOf(r.player)?.rating ?? -Infinity;
    return [...shown].sort((x, y) => at(y) - at(x));
  }

  private estimateOf(pid: string): PlayerEstimate | undefined {
    return this.data.playdek?.players[pid];
  }

  private links(pid: string): string {
    const pl = this.data.players[pid];
    const net = pl?.network ? this.data.networks[pl.network] : undefined;
    if (!net) return "";
    const out: string[] = [];
    // The player's report, never simply its network's: a network's report is of its greedy play,
    // and says nothing about the same weights under a search.
    if (pl.report) {
      const what = pl.report === net.report ? t("leaderboard:links.reportOfNetworkTitle") : t("leaderboard:links.reportOfPlayerTitle");
      out.push(`<a href="https://github.com/${esc(this.data.github)}/blob/${esc(this.data.commit ?? "main")}/${esc(pl.report)}"
        title="${esc(what)}">${t("leaderboard:links.report")}</a>`);
    }
    if (net.hf) {
      const onnx = net.hf.replace(/\.pt$/, ".onnx");
      out.push(`<a href="https://huggingface.co/${esc(this.data.hf_repo)}/blob/main/${esc(net.hf)}" title="${esc(t("leaderboard:links.weightsTitle"))}">${t("leaderboard:links.weights")}</a>`);
      // The player as the workbench plays it; a search it cannot run opens the network's policy, and says so.
      const search = pl.workbench_search ?? "policy";
      const what = pl.workbench_search ? t("leaderboard:links.playTitle", { search: pl.workbench_search })
        : t("leaderboard:links.playPolicyTitle");
      out.push(`<a href="./?model=${encodeURIComponent(`hf:${this.data.hf_repo}@main:${onnx}`)}&search=${encodeURIComponent(search)}"
        title="${esc(what)}">${t("leaderboard:links.play")}</a>`);
    }
    return out.join("");
  }

  private renderLineages(): void {
    const box = $("lb-lineages");
    box.hidden = this.view !== "lineage";
    const all = new Set<string>();
    for (const r of this.ep.ratings) {
      const l = this.data.players[r.player]?.lineage;
      if (l) all.add(l);
    }
    box.innerHTML = [...all].sort().map(l =>
      `<label><input type="checkbox" value="${esc(l)}" ${this.lineages.has(l) ? "checked" : ""}> ${esc(l)}</label>`).join("")
      || `<span class="lb-dim">${t("leaderboard:noLineages")}</span>`;
  }

  private renderTable(): void {
    const rows = this.ratingsShown();
    const playdek = this.mode === "playdek";
    $("lb-th-elo").textContent = playdek ? t("leaderboard:table.eloPlaydek") : t("leaderboard:table.elo");
    $("lb-th-games").textContent = playdek ? t("leaderboard:table.gamesPlaydek") : t("leaderboard:table.games");
    $("lb-th-score").textContent = playdek ? t("leaderboard:table.scorePlaydek") : t("leaderboard:table.score");
    $("lb-body").innerHTML = rows.map((r, k) => {
      const pl = this.data.players[r.player];
      const kind = kindLabel(pl?.kind ?? "");
      const pct = r.games ? (100 * r.score / r.games).toFixed(1) + "%" : "";
      const est = this.estimateOf(r.player);
      const eloCell = !playdek ? r.elo.toFixed(0) : est ? est.rating.toFixed(0) : "–";
      const seCell = !playdek ? (r.player === this.ep.anchor ? t("leaderboard:table.anchor") : r.se.toFixed(0)) : est ? est.se.toFixed(0) : "";
      const gamesCell = !playdek ? r.games.toLocaleString("en-US") : est?.games ? String(est.games) : "–";
      const scoreCell = !playdek ? pct : est?.games ? `${est.w}-${est.d}-${est.l}` : "";
      return `<tr data-player="${esc(r.player)}" class="${r.player === this.selected ? "selected" : ""}">
        <td class="num lb-dim">${k + 1}</td>
        <td><span class="lb-player">${esc(r.player)}</span>${r.main ? `<span class="lb-tag main">${t("leaderboard:tag.main")}</span>` : ""}${kind ? `<span class="lb-tag kind">${esc(kind)}</span>` : ""}${this.isHfDefault(r.player) ? hfTag() : ""}${this.siteLevelsOf(r.player).map(siteTag).join("")}</td>
        <td class="num lb-elo">${eloCell}</td>
        <td class="num lb-dim lb-hide-narrow">${seCell}</td>
        <td class="num lb-hide-narrow">${gamesCell}</td>
        <td class="num">${scoreCell}</td>
        <td class="lb-links lb-hide-narrow">${this.links(r.player)}</td>
      </tr>`;
    }).join("");
  }

  private renderDetail(): void {
    const box = $("lb-detail");
    const pid = this.selected;
    const rating = pid ? this.ep.ratings.find(r => r.player === pid) : undefined;
    if (!pid || !rating) { box.hidden = true; return; }
    box.hidden = false;
    const pl = this.data.players[pid];
    const net = pl?.network ? this.data.networks[pl.network] : undefined;
    const fmtSeat = (s: Seat) => {
      const n = s.w + s.l + s.d;
      return n ? `${(100 * (s.w + s.d / 2) / n).toFixed(1)}%` : "–";
    };
    const h2h = this.ep.pairs.filter(p => p.a === pid || p.b === pid).map(p => {
      // From this player's side: the pair is stored from `a`'s.
      const flip = p.b === pid;
      const opp = flip ? p.a : p.b;
      const asUs: Seat = flip ? { w: p.a_as_ussr.l, l: p.a_as_ussr.w, d: p.a_as_ussr.d } : p.a_as_us;
      const asUssr: Seat = flip ? { w: p.a_as_us.l, l: p.a_as_us.w, d: p.a_as_us.d } : p.a_as_ussr;
      const all: Seat = { w: asUs.w + asUssr.w, l: asUs.l + asUssr.l, d: asUs.d + asUssr.d };
      const oppRating = this.ep.ratings.find(r => r.player === opp);
      return { opp, all, asUs, asUssr, oppElo: oppRating?.elo };
    }).sort((x, y) => (y.oppElo ?? 0) - (x.oppElo ?? 0));
    const desc = [pl?.description, net?.description].filter(Boolean).map(s => esc(s!)).join(" ");
    const old = net?.old_names.length ? " " + t("leaderboard:detail.formerly", { names: net.old_names.map(n => `<code>${esc(n)}</code>`).join(", ") }) : "";
    const links = this.links(pid);
    const nc = pl?.network ? this.data.codes?.by_network[pl.network] : undefined;
    const codeLine = nc && (nc.architecture || nc.recipe)
      ? `<p class="lb-codes-line">${nc.architecture ? t("leaderboard:detail.architecture", { code: this.codeLink(nc.architecture) }) : ""}${nc.architecture && nc.recipe ? " · " : ""}${nc.recipe ? t("leaderboard:detail.recipe", { code: this.codeLink(nc.recipe) }) : ""}</p>`
      : "";
    const est = this.estimateOf(pid);
    const estLine = !est ? "" : `<p>${t("leaderboard:detail.estimate", {
      rating: est.rating.toFixed(0), se: est.se.toFixed(0), lo: est.lo.toFixed(0), hi: est.hi.toFixed(0),
      games: est.games ? t("leaderboard:detail.estimateGames", { games: est.games, count: est.humans, record: `${est.w}-${est.d}-${est.l}` })
        : t("leaderboard:detail.estimateNoGames") })}</p>`;
    box.innerHTML = `
      <h2>${esc(pid)}</h2>
      <p>${desc}${old}</p>
      ${this.isHfDefault(pid) ? `<p>${hfTag()} ${t("leaderboard:detail.hfDefault")}</p>` : ""}
      ${this.siteLevelsOf(pid).map(l => {
        const site = `<a href="${SITE_URL}">twilightstruggle.app</a>`;
        return `<p>${siteTag(l)} ${l.description
          ? t("leaderboard:detail.siteLevelDescribed", { level: esc(l.name), site, description: esc(l.description) })
          : t("leaderboard:detail.siteLevel", { level: esc(l.name), site })}</p>`;
      }).join("")}
      ${codeLine}
      ${links ? `<p class="lb-links">${links}</p>` : ""}
      <p>${t("leaderboard:detail.elo", {
        elo: rating.elo.toFixed(0),
        uncertainty: pid !== this.ep.anchor ? ` ± ${rating.se.toFixed(0)}` : this.mode === "playdek" ? "" : t("leaderboard:detail.theAnchor"),
        games: rating.games.toLocaleString("en-US"), opponents: rating.opponents, main: rating.main_opponents })}</p>
      ${estLine}
      <div class="lb-table-wrap"><table class="lb-table">
        <thead><tr><th>${t("leaderboard:detail.against")}</th><th class="num">${t("leaderboard:table.elo")}</th><th class="num">${t("leaderboard:table.games")}</th><th class="num">${t("leaderboard:table.score")}</th>
          <th class="num lb-hide-narrow">${t("leaderboard:detail.asUs")}</th><th class="num lb-hide-narrow">${t("leaderboard:detail.asUssr")}</th></tr></thead>
        <tbody>${h2h.map(h => `<tr>
          <td><span class="lb-player">${esc(h.opp)}</span></td>
          <td class="num lb-dim">${h.oppElo === undefined ? "–" : h.oppElo.toFixed(0)}</td>
          <td class="num">${(h.all.w + h.all.l + h.all.d).toLocaleString("en-US")}</td>
          <td class="num">${fmtSeat(h.all)}</td>
          <td class="num us lb-hide-narrow">${fmtSeat(h.asUs)}</td>
          <td class="num ussr lb-hide-narrow">${fmtSeat(h.asUssr)}</td></tr>`).join("")}</tbody>
      </table></div>`;
  }

  private codeLink(code: string): string {
    const kind = code.startsWith("A") ? "architectures" : "recipes";
    const info = this.data.codes?.[kind][code];
    return `<a href="?tab=codes" data-code="${esc(code)}" title="${esc(info?.description ?? "")}"><code>${esc(code)}</code></a>`;
  }

  private renderCodes(): void {
    const box = $("lb-tab-codes");
    const codes = this.data.codes;
    if (!codes) {
      box.innerHTML = `<p class="lb-dim">${t("leaderboard:codes.predates")}</p>`;
      return;
    }
    const src = `https://github.com/${esc(this.data.github)}/blob/${esc(this.data.commit ?? "main")}`;
    const table = (kind: "architectures" | "recipes", title: string, inline: boolean): string => {
      const rows = Object.entries(codes[kind]).filter(([, c]) => this.allCodes || c.networks.length);
      const body = rows.map(([code, c]) => {
        const flags = Object.entries(c.flags).map(([k, v]) => `<code class="lb-flag">${esc(flagText(k, v))}</code>`).join(" ");
        const flagCell = !flags ? `<span class="lb-dim">${t("leaderboard:codes.defaults")}</span>`
          : inline ? flags : `<details><summary>${t("leaderboard:codes.flags", { count: Object.keys(c.flags).length })}</summary>${flags}</details>`;
        const nets = c.networks.length
          ? c.networks.map(n => `<span class="lb-player">${esc(n)}</span>`).join("<br>")
          : `<span class="lb-dim">${c.first ? t("leaderboard:codes.firstRun", { run: esc(c.first) }) : "–"}</span>`;
        return `<tr id="code-${esc(code)}"><td class="lb-code"><code>${esc(code)}</code></td>
          <td>${esc(c.description) || `<span class="lb-dim">${t("leaderboard:codes.noDescription")}</span>`}</td>
          <td class="lb-flags">${flagCell}</td><td class="lb-hide-narrow">${nets}</td></tr>`;
      }).join("");
      return `<h2>${title}</h2>
        <div class="lb-table-wrap"><table class="lb-table lb-codes">
          <thead><tr><th>${t("leaderboard:codes.thCode")}</th><th>${t("leaderboard:codes.thWhat")}</th><th>${t("leaderboard:codes.thFlags")}</th><th class="lb-hide-narrow">${t("leaderboard:codes.thOnBoard")}</th></tr></thead>
          <tbody>${body}</tbody></table></div>`;
    };
    box.innerHTML = `
      <p class="lb-note">${t("leaderboard:codes.note", { src })}</p>
      <label class="lb-codes-toggle"><input type="checkbox" id="lb-codes-all" ${this.allCodes ? "checked" : ""}>
        ${t("leaderboard:codes.showAll")}</label>
      ${table("recipes", t("leaderboard:codes.recipes"), true)}
      ${table("architectures", t("leaderboard:codes.architectures"), false)}`;
  }

  private renderScale(): void {
    const ep = this.ep;
    const pd = this.data.playdek;
    $("lb-note").textContent = this.mode === "playdek" && pd
      ? t("leaderboard:note.playdek", { games: pd.games, humans: pd.humans, asOf: pd.as_of })
      : t("leaderboard:note.anchor") + (pd ? " " + t("leaderboard:note.estimatePointer") : "");
    const anchor = this.mode === "playdek" ? "" : t("leaderboard:scale.anchor", { anchor: esc(ep.anchor), elo: ep.anchor_elo.toFixed(0) }) + " ";
    $("lb-scale").innerHTML = `${anchor}${t("leaderboard:scale.clickRow")}`
      + (ep.unrated.length ? ` <span class="lb-dim">${t("leaderboard:scale.unrated", { players: ep.unrated.map(esc).join(", ") })}</span>` : "");
    const d = this.data;
    const records = `https://github.com/${esc(d.github)}/blob/${esc(d.commit ?? "main")}/leaderboard/matches/${esc(this.epoch)}.jsonl`;
    $("lb-footer").innerHTML = t("leaderboard:footer.fitted", {
      records,
      atCommit: d.commit ? t("leaderboard:footer.atCommit", { commit: esc(d.commit.slice(0, 10)) }) : "",
      weights: `<a href="https://huggingface.co/${esc(d.hf_repo)}">${esc(d.hf_repo)}</a>`,
      defaultModel: this.hfDefault ? t("leaderboard:footer.defaultModel", {
        model: esc(this.hfDefault),
        search: esc(this.hfDefaultSearch ? rolloutParam(this.hfDefaultSearch) : "policy"),
        notOnBoard: Object.keys(this.data.players).some(p => this.isHfDefault(p)) ? "" : t("leaderboard:footer.notOnBoard"),
      }) : "",
    });
  }

  render(): void {
    $("lb-tab-ratings").hidden = this.tab !== "ratings";
    $("lb-tab-codes").hidden = this.tab !== "codes";
    for (const b of document.querySelectorAll<HTMLButtonElement>(".lb-tabs button")) {
      const on = b.dataset.tab === this.tab;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", String(on));
    }
    this.renderLineages();
    this.renderScale();
    this.renderTable();
    this.renderDetail();
    this.renderCodes();
    const params = new URLSearchParams();
    if (this.tab === "codes") params.set("tab", "codes");
    if (this.allCodes) params.set("codes", "all");
    params.set("epoch", this.epoch);
    if (this.view !== "main") params.set("view", this.view);
    if (this.view === "lineage" && this.lineages.size) params.set("lineage", [...this.lineages].sort().join(","));
    if (this.selected) params.set("player", this.selected);
    if (this.mode === "playdek") params.set("scale", "playdek");
    const lang = new URLSearchParams(location.search).get("lang");
    if (lang) params.set("lang", lang);
    history.replaceState(null, "", `?${params}`);
  }
}

async function main(): Promise<void> {
  await initLanguage();
  bindLanguageSelects();
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}leaderboard.json`);
    if (!res.ok) throw new Error(`leaderboard.json: HTTP ${res.status} (build it with tools/scripts/build_web.sh)`);
    const data = await res.json() as LeaderboardData;
    if (data.schema !== 1) throw new Error(`leaderboard.json schema ${data.schema}; this page reads schema 1`);
    const page = new LeaderboardPage(data);
    onLanguageChange(() => page.render());
  } catch (e) {
    const box = $("lb-error");
    box.hidden = false;
    const show = () => { box.textContent = t("leaderboard:loadFailed", { error: e instanceof Error ? e.message : String(e) }); };
    show();
    onLanguageChange(show);
  }
}

main();
