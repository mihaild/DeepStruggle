/**
 * Position tokens: the `pos=` in a shared link.
 *
 * The engine's save JSON (canonical: keys sorted, compact -- bindings/state_json.cpp), zlib-
 * deflated, base64url without padding. The browser's "deflate" CompressionStream *is* zlib, so a
 * token the old Python server wrote opens here and one written here opens in Python
 * (`ts.state_from_save_json(zlib.decompress(...))`). About 1 KB mid-game.
 */

/** Inflated tokens are refused past this: a real save is under 8 KB. */
const MAX_SAVE_BYTES = 64 * 1024;

function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(token: string): Uint8Array {
  const b64 = token.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (token.length % 4)) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream, limit: number): Promise<Uint8Array> {
  const reader = new Blob([new Uint8Array(bytes)]).stream().pipeThrough(stream).getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > limit) {
      await reader.cancel();
      throw new Error("position token inflates past the size limit");
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

export async function encodePosition(saveJson: string): Promise<string> {
  const packed = await pipe(new TextEncoder().encode(saveJson), new CompressionStream("deflate"), Number.MAX_SAFE_INTEGER);
  return toBase64Url(packed);
}

/** The save JSON a token holds. Throws on anything that is not one. */
export async function decodePosition(token: string): Promise<string> {
  let packed: Uint8Array;
  try {
    packed = fromBase64Url(token.trim());
  } catch {
    throw new Error("not a position token (bad base64)");
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(await pipe(packed, new DecompressionStream("deflate"), MAX_SAVE_BYTES));
  } catch (e) {
    throw new Error(`not a position token (${e instanceof Error ? e.message : e})`);
  }
}
