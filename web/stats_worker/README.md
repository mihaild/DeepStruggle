# The stats database (`web/stats_worker/`)

Games people play against the models in the workbench, for estimating the models' strength on
Playdek's rating scale. The page (`web/ui/src/result_dialog.ts`) POSTs a finished game here; this
Cloudflare Worker checks it and keeps it in a D1 (SQLite) table; `tools/game_stats.py` exports the
table and does the rest (`tools/README.md` §2d). Nothing is self-hosted: the Worker and the
database run on Cloudflare's free plan (100k requests and 100k rows written a day, 5 GB).

* `POST /games` -- one submission (`web/ui/src/game/record.ts` `Submission`), at most 64 KB.
  Answers `{"ok": true, "stored": true}`, or `"stored": false` for an id it has already (a
  double click, a retried request). 400 for a malformed game, 403 for an origin not in
  `ALLOWED_ORIGINS`, 429 past `RATE_PER_HOUR` submissions from one address.
* `GET /health` -- `{"ok": true}`.
* Nothing reads games back over HTTP: the Playdek logins stay with the database's owner. The
  leaderboard publishes only aggregates fitted from them (below).

The checks are against junk and accidents, not cheating -- anyone can post anything, and
re-driving a game in the engine (`tools/game_stats.py verify`) is what tells a game from a story.
The client's address is kept only as a salted hash, for the rate limit.

## Setting it up (once)

Needs a Cloudflare account and node. From this directory:

```bash
npm install
npx wrangler login
npx wrangler d1 create ts-game-stats          # put the printed database_id into wrangler.toml
npx wrangler d1 execute ts-game-stats --remote --file schema.sql
npx wrangler secret put IP_SALT               # any random string
npx wrangler deploy                           # prints the Worker's URL
```

Then point the page at it: set the repository variable `TS_STATS_ENDPOINT` (Settings → Secrets and
variables → Actions → Variables) to the Worker's URL, which `.github/workflows/pages.yml` passes
to the build; locally, `TS_STATS_ENDPOINT=<url> tools/scripts/build_web.sh`. A build without it
shows the dialog with *Download record* only. `ALLOWED_ORIGINS` in `wrangler.toml` lists the
page's origins (GitHub Pages and the local servers); add any other place the page is served from.

## Testing locally

`wrangler dev` runs the Worker on this machine with its own local D1 (in `.wrangler/`, git-ignored)
and local secrets from `.dev.vars` (git-ignored), so nothing reaches the deployed database:

```bash
# terminal 1, in web/stats_worker/
echo "IP_SALT=local-test" > .dev.vars
npx wrangler d1 execute ts-game-stats --local --file schema.sql   # once
npx wrangler dev                                                  # http://localhost:8787

# terminal 2, in web/ui/: the page, pointed at it (localhost:5173 is in ALLOWED_ORIGINS)
TS_STATS_ENDPOINT=http://localhost:8787 npm run dev
```

Open `http://localhost:5173/`, play a game to its end, and add it from the dialog. To get to the end
quickly, finish the game from the browser console with first legal moves (the model's side as
auto-play, yours as clicks):

```js
const wb = __wb;
for (let n = 0; n < 5000 && !wb.state.is_terminal; n++) {
  const i = wb.engine.mask(false).findIndex(x => x), p = wb.engine.decisionPlayer();
  if ((p > 0 ? "US" : "USSR") === wb.analysisPanel.autoSide) wb.sendFlatAction(i, 0, true);
  else wb.sendAction(wb.engine.decodeFlat(i));
}
```

Then look at what arrived:

```bash
npx wrangler d1 execute ts-game-stats --local --command "SELECT id, playdek_login, clean FROM games"
PYTHONPATH=.:build/release python tools/game_stats.py fetch --local --output /tmp/local_games.json   # from the repo root
PYTHONPATH=.:build/release python tools/game_stats.py verify /tmp/local_games.json
```

The deployed Worker can be tried the same way (`TS_STATS_ENDPOINT=https://ts-game-stats.<subdomain>.workers.dev
npm run dev`): those games go into the real table. Remove them afterwards with
`npx wrangler d1 execute ts-game-stats --remote --command "DELETE FROM games WHERE playdek_login = '<the login you typed>'"`.

## The leaderboard's Playdek scale (GitHub Actions)

`.github/workflows/pages.yml` fits the leaderboard's players on Playdek's scale from this database
each time it publishes the page (each push to main, or "Run workflow"). It needs two repository
secrets (Settings -> Secrets and variables -> Actions); without them, or if the fit fails, the page
is published without that scale and the run shows a warning:

* `CLOUDFLARE_API_TOKEN`: an API token with the account's D1 read permission and nothing else.
* `CLOUDFLARE_ACCOUNT_ID`: the account's id, as `npx wrangler whoami` prints it.

Only that one step of the workflow holds them, and only aggregates leave it: the logins stay private.
The fit is `tools/playdek_estimate.py` (`tools/README.md`, "Games against people").

## Reading it

```bash
PYTHONPATH=.:build/release python tools/game_stats.py fetch --output data/game_stats/games.json
npx wrangler d1 execute ts-game-stats --remote --command \
    "SELECT received_at, playdek_login, playdek_rating, human_side, model_label, search, winner, clean FROM games"
```

The columns besides `body` are copies of the submission's fields, for queries like the second;
`body` is the submission exactly as received, and is what the tools read. `src/handler.ts` is
tested under node with a fake D1 by `tests/web/test_game_stats.py`.
