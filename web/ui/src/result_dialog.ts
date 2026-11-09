/**
 * The game-over dialog (index.html #result-dialog): the result, how the game was played -- the
 * record's summary (game/record.ts) in words -- and the offer to add it to the stats database
 * with the player's Playdek login and rating, which estimate the models' strength on Playdek's
 * scale (tools/game_stats.py). Offered once when a game ends, and again from the Result button.
 * Nothing is sent unless the player presses Add; the record can also be downloaded.
 *
 * The login and rating are remembered in this browser (localStorage) to prefill the next game.
 */
import { RecordSummary, Submission } from "./game/record";
import { STATS_ENDPOINT } from "./build_info";
import { onLanguageChange, sideName, t } from "./i18n";

const PLAYDEK_KEY = "ts.stats.playdek.v1";

export interface ResultView {
  winner: string;
  margin: number;
  endTurn: number;
  reason: string;
  summary: RecordSummary;
  /** The model's name and how it searched, for the auto-played side ("" if none). */
  model: string;
  submitted: boolean;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

/** The summary's departures from a straight game against one model, in words (the page's language). */
export function summaryNotes(s: RecordSummary): string[] {
  const notes: string[] = [];
  if (s.model_side === null) notes.push(t("result:note.noModelSide"));
  if (s.model_side === "mixed") notes.push(t("result:note.bothSides"));
  if (s.human_moves_for_model) notes.push(t("result:note.movesForModel", { count: s.human_moves_for_model }));
  if (s.dev_view_moves) notes.push(t("result:note.devViewMoves", { count: s.dev_view_moves }));
  if (s.analysis_moves) notes.push(t("result:note.analysisMoves", { count: s.analysis_moves }));
  if (s.suggested_moves) notes.push(t("result:note.suggestedMoves", { count: s.suggested_moves }));
  if (s.models.includes("")) notes.push(t("result:note.noModelLoaded"));
  if (s.models.length > 1) notes.push(t("result:note.modelChanged"));
  if (s.searches.length > 1) notes.push(t("result:note.searchChanged"));
  if (s.forced_dice) notes.push(t("result:note.forcedDice", { count: s.forced_dice }));
  if (s.debug_overrides) notes.push(t("result:note.debugOverrides", { count: s.debug_overrides }));
  if (s.loaded_position) notes.push(t("result:note.loadedPosition"));
  return notes;
}

interface Playdek {
  login: string;
  rating: number | null;
}

function loadPlaydek(): Playdek {
  try {
    const v = JSON.parse(window.localStorage.getItem(PLAYDEK_KEY) ?? "null");
    if (v && typeof v.login === "string") return { login: v.login, rating: typeof v.rating === "number" ? v.rating : null };
  } catch {
    // nothing remembered
  }
  return { login: "", rating: null };
}

function storePlaydek(p: Playdek): void {
  try {
    window.localStorage.setItem(PLAYDEK_KEY, JSON.stringify(p));
  } catch {
    // not remembered; asked again next time
  }
}

export class ResultDialog {
  private readonly el = document.getElementById("result-dialog")!;
  private readonly form = document.getElementById("result-form") as HTMLFormElement;
  private readonly login = document.getElementById("result-login") as HTMLInputElement;
  private readonly rating = document.getElementById("result-rating") as HTMLInputElement;
  private readonly note = document.getElementById("result-note") as HTMLTextAreaElement;
  private readonly submit = document.getElementById("result-submit") as HTMLButtonElement;
  private readonly status = document.getElementById("result-status")!;
  private sending = false;
  /** The game shown, while the dialog is open: redrawn in a new language, the form as filled. */
  private view: ResultView | null = null;
  /** The status line, as a message to redraw in a new language. */
  private statusText: () => string = () => "";

  /**
   * `makeSubmission` builds the game's record with the form's Playdek fields and note;
   * `onSubmitted` is told once the database has it.
   */
  constructor(
    private readonly makeSubmission: (playdek: { login: string; rating: number }, note: string) => Promise<Submission>,
    private readonly onSubmitted: () => void,
  ) {
    document.getElementById("result-dialog-close")?.addEventListener("click", () => this.close());
    document.getElementById("result-skip")?.addEventListener("click", () => this.close());
    this.el.addEventListener("click", (ev) => { if (ev.target === this.el) this.close(); });
    this.el.addEventListener("keydown", (ev) => { if (ev.key === "Escape") this.close(); });
    this.form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      void this.send();
    });
    document.getElementById("result-download")?.addEventListener("click", () => void this.download());
    onLanguageChange(() => {
      if (!this.isOpen || !this.view) return;
      this.render(this.view);
      this.status.textContent = this.statusText();
    });
  }

  get isOpen(): boolean {
    return !this.el.classList.contains("hidden");
  }

  open(v: ResultView): void {
    this.view = v;
    this.render(v);

    const p = loadPlaydek();
    this.login.value = p.login;
    this.rating.value = p.rating === null ? "" : String(p.rating);
    this.note.value = "";
    this.setStatus(() => "", "");
    this.sending = false;
    if (v.submitted) {
      this.submit.disabled = true;
      this.setStatus(() => t("result:status.already"), "ok");
    } else if (!STATS_ENDPOINT) {
      this.submit.disabled = true;
      this.setStatus(() => t("result:status.noDatabase"), "");
    } else {
      this.submit.disabled = false;
    }
    this.el.classList.remove("hidden");
    (p.login ? this.rating : this.login).focus();
  }

  close(): void {
    this.el.classList.add("hidden");
  }

  /** The result, the facts and the notes: everything but the form. */
  private render(v: ResultView): void {
    const s = v.summary;
    const outcome = document.getElementById("result-dialog-outcome")!;
    const verdict = v.winner === "DRAW" ? t("result:draw") : t("result:wins", { side: sideName(v.winner) });
    const you = s.human_side === null ? "" : v.winner === "DRAW" ? "" : v.winner === s.human_side ? t("result:youWon") : t("result:youLost");
    outcome.innerHTML = t("result:outcome", {
      verdict, you, margin: `${v.margin > 0 ? "+" : ""}${v.margin}`, turn: v.endTurn,
      reason: v.reason ? t("result:reason", { reason: escapeHtml(v.reason) }) : "",
    });
    const facts = document.getElementById("result-dialog-facts")!;
    facts.innerHTML = [
      s.human_side ? `<dt>${t("result:facts.youPlayed")}</dt><dd>${sideName(s.human_side)}</dd>` : "",
      v.model ? `<dt>${t("result:facts.against")}</dt><dd>${escapeHtml(v.model)}</dd>` : "",
      s.undos ? `<dt>${t("result:facts.takenBack")}</dt><dd>${t("result:facts.moves", { count: s.undos })}</dd>` : "",
    ].join("");
    const notes = summaryNotes(s);
    const list = document.getElementById("result-dialog-notes")!;
    list.innerHTML = notes.length
      ? `<p class="result-notes-intro">${t("result:notesIntro")}</p><ul>${notes.map(n => `<li>${escapeHtml(n)}</li>`).join("")}</ul>`
      : `<p class="result-notes-intro">${t("result:clean")}</p>`;
  }

  /** The status line; `text` is called again when the language changes. */
  private setStatus(text: () => string, cls: "" | "ok" | "error"): void {
    this.statusText = text;
    this.status.textContent = text();
    this.status.className = `result-status ${cls}`;
  }

  /** The form's Playdek fields, or null (with the reason shown) when they are not filled in. */
  private playdek(require: boolean): { login: string; rating: number } | null {
    const login = this.login.value.trim();
    const rating = Number(this.rating.value);
    const ratingOk = this.rating.value.trim() !== "" && Number.isFinite(rating) && rating >= 1 && rating <= 5000;
    if (require && (!login || !ratingOk)) {
      this.setStatus(() => !login ? t("result:status.needLogin") : t("result:status.needRating"), "error");
      (!login ? this.login : this.rating).focus();
      return null;
    }
    return { login, rating: ratingOk ? Math.round(rating) : 0 };
  }

  private async send(): Promise<void> {
    if (this.sending || !STATS_ENDPOINT) return;
    const p = this.playdek(true);
    if (!p) return;
    storePlaydek(p);
    this.sending = true;
    this.submit.disabled = true;
    this.setStatus(() => t("result:status.sending"), "");
    try {
      const body = await this.makeSubmission(p, this.note.value.trim());
      const res = await fetch(`${STATS_ENDPOINT.replace(/\/+$/, "")}/games`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const reply = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(reply?.error ?? `HTTP ${res.status}`);
      this.onSubmitted();
      const stored = reply?.stored !== false;
      this.setStatus(() => stored ? t("result:status.added") : t("result:status.already"), "ok");
    } catch (e) {
      this.submit.disabled = false;
      const error = e instanceof Error ? e.message : String(e);
      this.setStatus(() => t("result:status.notAdded", { error }), "error");
    } finally {
      this.sending = false;
    }
  }

  /** The record as the database would get it, as a file. */
  private async download(): Promise<void> {
    const p = this.playdek(false)!;
    try {
      const body = await this.makeSubmission(p, this.note.value.trim());
      const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `ts-game-${body.id}.json`;
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      this.setStatus(() => t("result:status.recordFailed", { error }), "error");
    }
  }
}
