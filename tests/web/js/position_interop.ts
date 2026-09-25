// Bundled by tests/web/test_position_tokens.py: the page's position-token codec, under node.
//   node position_interop.mjs <case.json>   with {save_json, python_token}
// Prints the token this code makes for `save_json`, and what it decodes `python_token` to.
import { readFileSync } from "node:fs";
import { decodePosition, encodePosition } from "../../../web/ui/src/game/position";

const { save_json, python_token } = JSON.parse(readFileSync(process.argv[2], "utf8"));
const page_token = await encodePosition(save_json);
const decoded_python = await decodePosition(python_token);
let garbage_refused = false;
try {
  await decodePosition("not-a-token!!");
} catch {
  garbage_refused = true;
}
console.log(JSON.stringify({ page_token, decoded_python, garbage_refused }));
