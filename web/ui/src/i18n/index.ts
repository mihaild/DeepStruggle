/**
 * The page's languages, with i18next. English is the default and the source: every message is an id
 * in locales/en/<namespace>.json (`t("hud:pass")`), and another language is the same ids in
 * locales/<lang>/, loaded only when it is chosen, with English for whatever it lacks.
 *
 * Only what the page shows is translated. Engine names, enum values, saved games, replays, position
 * links and the stats record stay as they are; card and country names are looked up by id
 * (`cardName`), with the engine's English as the fallback.
 *
 * The language is `?lang=` when the link names one, else the one chosen last in this browser, else
 * English. Choosing one does not put it in the address, so a shared link keeps the reader's own.
 *
 * `?lang=en-XA` is the pseudo-language: English with every message marked `⟦…⟧`, so text the page
 * shows unmarked was never translated (tests/web/test_e2e_i18n.py). It is never offered in the select.
 */
import i18next from "i18next";
import type { ParseKeys } from "i18next";
import en from "./locales/en";

/** Each language by its own name, as the select offers it. A new one also goes in LOADERS. */
export const LANGUAGES = { en: "English", "zh-CN": "简体中文" } as const;
export const PSEUDO = "en-XA";
export type Language = keyof typeof LANGUAGES | typeof PSEUDO;

type Bundle = Record<string, object>;
/** A language's catalogs (locales/<lang>/index.ts), imported only when it is chosen. */
const LOADERS: Partial<Record<Language, () => Promise<{ default: Bundle }>>> = {
  "zh-CN": () => import("./locales/zh-CN"),
};
const STORAGE_KEY = "ts.language";

i18next.use({
  type: "postProcessor",
  name: "pseudo",
  process: (value: string) => (i18next.language === PSEUDO ? `⟦${value}⟧` : value),
});
void i18next.init({
  lng: "en",
  fallbackLng: "en",
  supportedLngs: [...Object.keys(LANGUAGES), PSEUDO],
  resources: { en },
  ns: Object.keys(en),
  defaultNS: "common",
  initAsync: false,
  // Messages go into innerHTML exactly as the templates they replaced did: a value is escaped by the
  // caller where the template escaped it, and never twice.
  interpolation: { escapeValue: false },
  returnEmptyString: false,
  postProcess: ["pseudo"],
});

/** The typed `t`: a key the English catalog lacks does not compile. */
export const t = i18next.t.bind(i18next) as typeof i18next.t;
/** `t` for an id built at run time (a card's, a country's); `fallback` when no language has it. */
const tDynamic = (key: string, fallback: string): string =>
  (i18next.t as unknown as (k: string, o: object) => string)(key, { defaultValue: fallback });

export function language(): Language {
  return i18next.language as Language;
}

const isLanguage = (v: unknown): v is Language =>
  typeof v === "string" && (Object.hasOwn(LANGUAGES, v) || v === PSEUDO);

function requested(): Language {
  try {
    const fromLink = new URLSearchParams(globalThis.location?.search ?? "").get("lang");
    if (isLanguage(fromLink)) return fromLink;
  } catch { /* no location: under node */ }
  try {
    const stored = globalThis.localStorage?.getItem(STORAGE_KEY);
    if (isLanguage(stored)) return stored;
  } catch { /* storage blocked */ }
  return "en";
}

async function use(lang: Language): Promise<void> {
  const load = LOADERS[lang];
  if (load && !i18next.hasResourceBundle(lang, "common")) {
    const { default: bundle } = await load();
    for (const [ns, messages] of Object.entries(bundle)) i18next.addResourceBundle(lang, ns, messages, true, true);
  }
  await i18next.changeLanguage(lang);
}

/** Before the page draws: the language the link or this browser asks for (English on failure). */
export async function initLanguage(): Promise<void> {
  try {
    await use(requested());
  } catch (e) {
    console.warn(`i18n: could not load the language, English stands: ${e instanceof Error ? e.message : e}`);
  }
  if (typeof document !== "undefined") applyStatic(document);
}

/**
 * Switch the page's language and remember it in this browser. Redraws through `onLanguageChange`.
 * A `?lang=` the page was opened with is dropped from the address, so a reload keeps the choice.
 */
export async function setLanguage(lang: Language): Promise<void> {
  try { globalThis.localStorage?.setItem(STORAGE_KEY, lang); } catch { /* storage blocked */ }
  try {
    const url = new URL(location.href);
    if (url.searchParams.has("lang")) {
      url.searchParams.delete("lang");
      history.replaceState(history.state, "", url);
    }
  } catch { /* no location: under node */ }
  await use(lang);
}

/** Called after every switch, once the static markup has been translated; returns an unsubscribe. */
export function onLanguageChange(listener: () => void): () => void {
  const wrapped = () => listener();
  i18next.on("languageChanged", wrapped);
  return () => i18next.off("languageChanged", wrapped);
}

const ATTRIBUTES = ["title", "placeholder", "aria-label"] as const;

/**
 * The static markup's messages: `data-i18n="ns:key"` sets the text, `data-i18n-title` (and
 * `-placeholder`, `-aria-label`) the attribute. An element whose text the code sets carries none.
 */
export function applyStatic(root: ParentNode): void {
  if (typeof document !== "undefined") document.documentElement.lang = language();
  root.querySelectorAll<HTMLElement>("[data-i18n]").forEach(el => {
    el.textContent = tDynamic(el.dataset.i18n!, el.textContent ?? "");
  });
  for (const attr of ATTRIBUTES) {
    root.querySelectorAll<HTMLElement>(`[data-i18n-${attr}]`).forEach(el => {
      el.setAttribute(attr, tDynamic(el.getAttribute(`data-i18n-${attr}`)!, el.getAttribute(attr) ?? ""));
    });
  }
  root.querySelectorAll<HTMLSelectElement>("select[data-language-select]").forEach(s => { s.value = language(); });
}
// First, so the static markup is in the new language before any component redraws.
i18next.on("languageChanged", () => { if (typeof document !== "undefined") applyStatic(document); });

/**
 * Fill and wire every `<select data-language-select>`: each language under its own name, and the
 * pseudo-language while it is on. With nothing to choose between, the select's label is hidden.
 */
export function bindLanguageSelects(root: ParentNode = document): void {
  const offered: [string, string][] = Object.entries(LANGUAGES);
  if (language() === PSEUDO) offered.push([PSEUDO, `⟦${LANGUAGES.en}⟧`]);
  root.querySelectorAll<HTMLSelectElement>("select[data-language-select]").forEach(select => {
    select.innerHTML = "";
    for (const [code, name] of offered) select.add(new Option(name, code));
    select.value = language();
    const label = select.closest<HTMLElement>(".language-control") ?? select;
    label.hidden = offered.length < 2;
    select.addEventListener("change", () => { if (isLanguage(select.value)) void setLanguage(select.value); });
  });
}

/** A card's name by id; `english` (the engine's or the rules') when the language has none. */
export function cardName(id: number, english: string): string {
  return tDynamic(`game:cards.${id}.name`, english);
}
/** A card's Event text by id; `english` (rules/cards.json) when the language has none. */
export function cardDescription(id: number, english: string): string {
  return tDynamic(`game:cards.${id}.description`, english);
}
export function countryName(id: number, english: string): string {
  return tDynamic(`game:countries.${id}`, english);
}
/** A region by its English name (rules/map.json, the engine's layout). */
export function regionName(english: string): string {
  return tDynamic(`game:regions.${english}`, english);
}
/** A side as the engine names it ("US", "USSR"); anything else unchanged. */
export function sideName(side: string): string {
  return tDynamic(`common:side.${side}`, side);
}

/** Any message id of the English catalog, `ns:dotted.id` (a plural by its base id). */
export type MessageKey = ParseKeys<(keyof typeof en)[]>;

/**
 * A message kept to be rendered later, in whatever language is current then: its id and its
 * parameters, never its text. Built where the words are decided (the action log, a branch's
 * meaning), rendered where they are shown -- so a language switch redraws it, and code that needs to
 * know what was said reads the id and the ids in its parameters, never the words.
 */
export interface Message {
  key: MessageKey;
  params?: Record<string, MessageArg>;
}
/**
 * A parameter: text or a number as it is, a nested message, or a card / country / region / side by
 * what the engine calls it, named in the current language when rendered (`name` is the English).
 */
export type MessageArg =
  | string | number | Message
  | { card: number; name: string }
  | { country: number; name: string }
  | { region: string }
  | { side: string };

export function message(key: MessageKey, params?: Record<string, MessageArg>): Message {
  return params ? { key, params } : { key };
}

/** How a rendering shows each kind of parameter; by default plain names and text as it is. */
export interface MessageFormat {
  /** Text and numbers: `escapeHtml` for innerHTML. */
  text?: (s: string) => string;
  /** A card by id and its name in the current language: the log's hoverable reference. */
  card?: (id: number, name: string) => string;
  country?: (id: number, name: string) => string;
}

const isMessage = (a: MessageArg): a is Message => typeof a === "object" && "key" in a;

/** What `renderEnglish` asks of every lookup: English, and unmarked even in the pseudo-language. */
const ENGLISH = { lng: "en", applyPostProcessor: false } as const;

/** A message id with its values, in the current language or (`english`) in English. */
function lookup(key: string, values: object, english: boolean): string {
  return (i18next.t as unknown as (k: string, o: object) => string)(key, english ? { ...values, ...ENGLISH } : values);
}

function formatArg(a: MessageArg, how: MessageFormat, english: boolean): string {
  const text = how.text ?? ((s: string) => s);
  if (typeof a === "string" || typeof a === "number") return text(String(a));
  if (isMessage(a)) return renderIn(a, how, english);
  if ("card" in a) {
    const name = english ? a.name : cardName(a.card, a.name);
    return how.card ? how.card(a.card, name) : text(name);
  }
  if ("country" in a) {
    const name = english ? a.name : countryName(a.country, a.name);
    return how.country ? how.country(a.country, name) : text(name);
  }
  if ("region" in a) return text(english ? a.region : regionName(a.region));
  return text(english ? lookup(`common:side.${a.side}`, { defaultValue: a.side }, true) : sideName(a.side));
}

function renderIn(m: Message, how: MessageFormat, english: boolean): string {
  const values: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(m.params ?? {})) {
    values[k] = k === "count" && typeof v === "number" ? v : formatArg(v, how, english);
  }
  return lookup(m.key, values, english);
}

/**
 * A message in the current language. `count` stays a number (it picks the plural form); every other
 * parameter is formatted by `how` -- for innerHTML, `{ text: escapeHtml }`, and the catalog's own
 * markup is kept.
 */
export function render(m: Message, how: MessageFormat = {}): string {
  return renderIn(m, how, false);
}

/**
 * A message in English whatever the page's language, unmarked even in the pseudo-language, with its
 * cards, countries and regions under their English names. Only for text kept as English data: a
 * replay's step descriptions and the stats record's event texts (game/session.ts), the model's choice
 * names (game/names.ts, analysis/readout.ts), a roll's and a turn's `when` / `label` (game/rolls.ts,
 * game/last_turn.ts), and the log's English for the tests (game/log.ts `lineEnglish`).
 */
export function renderEnglish(m: Message, how: MessageFormat = {}): string {
  return renderIn(m, how, true);
}

/**
 * A message id built at run time (a flag's, an enum name's) as a Message, or null when the English
 * catalog has no such id -- the caller then shows what it showed before there was a catalog.
 */
export function messageIfKnown(key: string, params?: Record<string, MessageArg>): Message | null {
  return i18next.exists(key, { lng: "en" }) ? message(key as MessageKey, params) : null;
}
