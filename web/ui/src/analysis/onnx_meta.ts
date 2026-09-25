/**
 * The key/value metadata inside an .onnx file (ModelProto.metadata_props), read without an ONNX
 * library: onnxruntime-web does not expose it, and it is where tools/export_onnx.py records what
 * the page must know before it runs a model -- its action view, its observation width, the
 * checkpoint it came from and the engine it was exported next to.
 *
 * Only the top level of the protobuf is walked; the graph (field 7, most of the file) is skipped
 * by its length, so this costs a scan of a few dozen field headers.
 */

const METADATA_PROPS = 14;   // ModelProto.metadata_props: repeated StringStringEntryProto

function varint(buf: Uint8Array, pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  for (;;) {
    if (pos >= buf.length) throw new Error("truncated varint");
    const b = buf[pos++];
    // Past 2^53 a length is not a length in any file we can hold; stop rather than wrap.
    result += (b & 0x7f) * Math.pow(2, shift);
    if (!(b & 0x80)) return [result, pos];
    shift += 7;
    if (shift > 56) throw new Error("varint too long");
  }
}

function skip(buf: Uint8Array, pos: number, wire: number): number {
  switch (wire) {
    case 0: return varint(buf, pos)[1];
    case 1: return pos + 8;
    case 2: { const [len, p] = varint(buf, pos); return p + len; }
    case 5: return pos + 4;
    default: throw new Error(`unsupported protobuf wire type ${wire}`);
  }
}

function entry(buf: Uint8Array): [string, string] {
  const dec = new TextDecoder();
  let key = "";
  let value = "";
  let pos = 0;
  while (pos < buf.length) {
    const [tag, p] = varint(buf, pos);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (wire === 2 && (field === 1 || field === 2)) {
      const [len, q] = varint(buf, p);
      const s = dec.decode(buf.subarray(q, q + len));
      if (field === 1) key = s; else value = s;
      pos = q + len;
    } else {
      pos = skip(buf, p, wire);
    }
  }
  return [key, value];
}

export function onnxMetadata(bytes: Uint8Array): Record<string, string> {
  const out: Record<string, string> = {};
  let pos = 0;
  while (pos < bytes.length) {
    const [tag, p] = varint(bytes, pos);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (field === METADATA_PROPS && wire === 2) {
      const [len, q] = varint(bytes, p);
      const [k, v] = entry(bytes.subarray(q, q + len));
      out[k] = v;
      pos = q + len;
    } else {
      pos = skip(bytes, p, wire);
    }
    if (pos > bytes.length) throw new Error("not an ONNX model (truncated)");
  }
  return out;
}
