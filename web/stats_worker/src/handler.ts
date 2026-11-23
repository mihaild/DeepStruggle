/**
 * The stats database's Worker: takes a finished game from the workbench page (POST /games, the
 * shape of web/ui/src/game/record.ts `Submission`) and keeps it in D1. Nothing reads games back
 * over HTTP -- tools/game_stats.py exports the table with wrangler -- so the Playdek logins stay
 * with the database's owner.
 *
 * The checks are for junk and accidents, not for cheating: anyone can POST anything, and the
 * offline verification (re-driving each game in the engine) is what tells a game from a story.
 * So: the shape and size of a submission, the page's origin, and a per-address rate limit.
 * An id stored once is not stored again (a double click, a retried request).
 */
import type { Submission } from "../../ui/src/game/record";

/** The D1 binding, as far as this Worker uses it. */
export interface D1Statement {
  bind(...values: unknown[]): D1Statement;
  run(): Promise<{ meta: { changes?: number } }>;
  first<T>(): Promise<T | null>;
}
export interface D1Database {
  prepare(sql: string): D1Statement;
}

export interface Env {
  DB: D1Database;
  /** Comma-separated origins allowed to submit (the page's); unset or "*": any. */
  ALLOWED_ORIGINS?: string;
  /** Salt for the hashed client address the rate limit counts by (a secret). */
  IP_SALT?: string;
  /** Submissions per address per hour (default 30). */
  RATE_PER_HOUR?: string;
}

const MAX_BODY_BYTES = 64 * 1024;
const MAX_LOG_CHARS = 48 * 1024;
const SCHEMA = 1;
const EVENT_KINDS = ["undo", "debug", "load", "resume"];

function allowedOrigin(req: Request, env: Env): string | null {
  const origin = req.headers.get("Origin");
  const allowed = (env.ALLOWED_ORIGINS ?? "*").split(",").map(s => s.trim()).filter(Boolean);
  if (allowed.length === 0 || allowed.includes("*")) return origin ?? "*";
  return origin && allowed.includes(origin) ? origin : null;
}

function reply(status: number, body: object, origin: string | null): Response {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (origin) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
  }
  return new Response(JSON.stringify(body), { status, headers });
}

const isStr = (v: unknown, max: number): v is string => typeof v === "string" && v.length <= max;
const isInt = (v: unknown, lo: number, hi: number): v is number => Number.isInteger(v) && (v as number) >= lo && (v as number) <= hi;
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const SIDES = ["US", "USSR"];
const isStrList = (v: unknown, n: number, max: number): v is string[] =>
  Array.isArray(v) && v.length <= n && v.every(x => isStr(x, max));

/** A setting the page moved under (record.ts Settings). */
function validSetting(v: unknown): boolean {
  return isObj(v) && ["DEV", "US", "USSR"].includes(v.view as string) && ["", "US", "USSR"].includes(v.auto as string)
    && isStr(v.model, 64) && isStr(v.search, 256);
}

/** A model the page moved with, by its sha256 (record.ts ModelInfo). */
function validModel(v: unknown): boolean {
  return isObj(v) && isStr(v.label, 200) && isStr(v.source, 1024) && isStr(v.checkpoint, 1024)
    && (v.obs_features === undefined || isInt(v.obs_features, 0, 0xffffffff));
}

/** What happened besides moves (record.ts RecordEvent), a debug override's change included. */
function validEvent(v: unknown): boolean {
  if (!isObj(v) || !isInt(v.step, 0, 20000) || !EVENT_KINDS.includes(v.kind as string) || !isStr(v.text, 500)) return false;
  if (v.undone !== undefined && typeof v.undone !== "boolean") return false;
  const o = v.override;
  if (o === undefined) return true;
  if (!isObj(o)) return false;
  if (o.kind === "country") return isInt(o.id, 0, 83) && isInt(o.us, 0, 255) && isInt(o.ussr, 0, 255);
  if (o.kind === "defcon" || o.kind === "vp") return isInt(o.value, -20, 20);
  return false;
}

/** Why `v` is not a submission this Worker keeps, or null. Every field the row binds is checked for
 *  its type and length here, so the INSERT never binds something the table does not expect. */
export function invalid(v: unknown): string | null {
  if (!isObj(v)) return "not a JSON object";
  if (v.schema !== SCHEMA) return `schema must be ${SCHEMA}`;
  if (!isStr(v.id, 64) || !/^[A-Za-z0-9-]{8,64}$/.test(v.id)) return "bad id";
  if (!isStr(v.started_at, 40) || !isStr(v.ended_at, 40)) return "bad timestamps";
  const b = v.build;
  if (!isObj(b) || !isStr(b.commit, 64) || typeof b.dirty !== "boolean" || !isStr(b.engine_fingerprint, 128)) return "bad build";
  const p = v.playdek;
  if (!isObj(p) || !isStr(p.login, 64) || !p.login.trim()) return "a Playdek login is needed";
  if (!isInt(p.rating, 1, 5000)) return "a Playdek rating (1-5000) is needed";
  if (!isStr(v.note, 1000)) return "bad note";
  const r = v.result;
  if (!isObj(r) || !["US", "USSR", "DRAW"].includes(r.winner as string) || !isInt(r.margin, -100, 100)
      || !isInt(r.end_turn, 0, 20) || !isStr(r.reason, 80)) return "bad result";
  const s = v.summary;
  if (!isObj(s) || typeof s.clean !== "boolean" || !(s.human_side === null || SIDES.includes(s.human_side as string))
      || !isStrList(s.models, 255, 64) || !isStrList(s.searches, 255, 256)) return "bad summary";
  if (!Array.isArray(v.settings) || v.settings.length > 255 || !v.settings.every(validSetting)) return "bad settings";
  const models = v.models;
  if (!isObj(models) || Object.keys(models).length > 255
      || !Object.entries(models).every(([k, m]) => k.length <= 64 && validModel(m))) return "bad models";
  if (!Array.isArray(v.events) || v.events.length > 5000 || !v.events.every(validEvent)) return "bad events";
  if (!(v.start_position === null || isStr(v.start_position, 8192))) return "bad start position";
  if (!isStr(v.log, MAX_LOG_CHARS) || !/^[A-Za-z0-9_-]+$/.test(v.log)) return "bad log";
  if (!isInt(v.n_steps, 1, 20000)) return "bad step count";
  if (!isStr(v.final_hash, 64) || !/^[0-9a-f]{64}$/.test(v.final_hash)) return "bad final hash";
  return null;
}

async function sha256Hex(text: string): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(d), x => x.toString(16).padStart(2, "0")).join("");
}

export async function handle(req: Request, env: Env): Promise<Response> {
  const url = new URL(req.url);
  const origin = allowedOrigin(req, env);
  if (req.method === "OPTIONS") {
    if (!origin) return new Response(null, { status: 403 });
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin",
      },
    });
  }
  if (url.pathname === "/health" && req.method === "GET") return reply(200, { ok: true }, origin);
  if (url.pathname !== "/games") return reply(404, { error: "not found" }, origin);
  if (req.method !== "POST") return reply(405, { error: "POST a game" }, origin);
  if (!origin) return reply(403, { error: "origin not allowed" }, null);

  // The cap is on bytes, not characters (a character can be four bytes), and is checked before the
  // body is decoded: a declared length over it is refused unread.
  const declared = Number(req.headers.get("Content-Length") ?? "0");
  if (declared > MAX_BODY_BYTES) return reply(413, { error: "too large" }, origin);
  const raw = await req.arrayBuffer();
  if (raw.byteLength > MAX_BODY_BYTES) return reply(413, { error: "too large" }, origin);
  let text: string;
  let body: unknown;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
    body = JSON.parse(text);
  } catch {
    return reply(400, { error: "not JSON" }, origin);
  }
  const why = invalid(body);
  if (why) return reply(400, { error: why }, origin);
  const sub = body as Submission;

  const ipHash = await sha256Hex(`${env.IP_SALT ?? ""}|${req.headers.get("CF-Connecting-IP") ?? ""}`);
  const limit = Number(env.RATE_PER_HOUR ?? "30");
  const since = new Date(Date.now() - 3600_000).toISOString();
  const recent = await env.DB.prepare("SELECT COUNT(*) AS n FROM games WHERE ip_hash = ? AND received_at > ?")
    .bind(ipHash, since).first<{ n: number }>();
  if ((recent?.n ?? 0) >= limit) return reply(429, { error: "too many games from here this hour" }, origin);

  const s = sub.summary;
  const single = (xs: string[]) => (xs.length === 1 ? xs[0] : "");
  const sha = single(s.models);
  const label = sub.models[sha]?.label ?? "";
  const res = await env.DB.prepare(
    `INSERT OR IGNORE INTO games (id, received_at, schema, commit_sha, dirty, engine_fp, playdek_login,
       playdek_rating, human_side, model_sha256, model_label, search, winner, margin, end_turn, reason,
       n_steps, clean, ip_hash, body)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).bind(
    sub.id, new Date().toISOString(), sub.schema, sub.build.commit, sub.build.dirty ? 1 : 0, sub.build.engine_fingerprint,
    sub.playdek.login.trim(), sub.playdek.rating, s.human_side, sha, label, single(s.searches), sub.result.winner,
    sub.result.margin, sub.result.end_turn, sub.result.reason, sub.n_steps, s.clean ? 1 : 0, ipHash, text,
  ).run();
  return reply(200, { ok: true, stored: (res.meta.changes ?? 0) > 0 }, origin);
}
