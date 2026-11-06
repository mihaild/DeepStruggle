# What J5's transformer attends to: 260M against 830M (2026-11-06)

J5 (`E8-A35-R102-S44`, 3 token layers x 128, 195 tokens: the global token, 84 countries, 110 cards) read on 24 greedy
self-play games per snapshot (3,483 positions at 260M, 3,865 at 830M), attention averaged over heads and positions.
Attention is where a token reads from, not what decides the move; read the shifts, not single cells. Data:
`data/reports/j5_attention_{260M,830M}.txt` (hubs, strongest pairs, by side), `j5_scoring_attention_{260M,830M}.txt`
(what each scoring card reads), `j5_attention_by_location_{260M,830M}.txt` (attention to cards by where they are, as a
multiple of uniform), `j5_{260M,830M}_layer1_top.md` (each card's and country's top card and country in layer 1).

## What held from 260M

* **Layer 1 is regional and the scoring cards summarise regions:** a country's strongest card is its region's scoring
  card (Egypt, Libya, Saudi Arabia, Lebanon, Israel, Iran -> Middle East Scoring 3.9-4.7%), a country's strongest
  country a neighbour (Syria -> Lebanon, Uruguay / Colombia / Bolivia -> Brazil).
* **West Germany is the layer-2 hub** (4.3% of all layer-2 attention at 830M, 4.5% at 260M), for both sides -- even
  though the US has placed nothing there at setup since 360M. Late-war cards read it most (Chernobyl, Aldrich Ames,
  Yuri and Samantha, Reagan Bombs Libya, Iran-Contra -> West Germany 3.1-3.8%), and the global token's top country is
  West Germany (2.8%; at 260M Italy, 2.4%).
* **Location gating:** in layer 2 the countries and the global token read the card in play; in layer 3 the cards read the
  mover's hand.

## What changed by 830M

* **The scoring cards became the network's backbone.** Layer-3 hubs: Europe Scoring 10.3% (260M 3.7%), Asia Scoring 7.2%
  (4.8%), South America 4.2%, Middle East 3.7%; countries have nearly left the layer-3 hub list (Italy, Poland, East
  Germany at 2.1-2.4% at 260M; at 830M only Honduras 2.0%, and Poland 3.5% when the US moves). The most-read token over
  the whole rollout is Europe Scoring (2.9%), then the global token.
* **The scoring cards read each other far more:** Middle East Scoring's layer-3 attention to Europe Scoring 33.7% (260M,
  Asia Scoring -> Europe Scoring 4.6%); Asia -> Europe 25.0%; every scoring card's top three in layers 2-3 are other
  scoring cards. Almost every held card's strongest read over the rollout is Europe Scoring (5-7%).
* **Layer-1 hub:** the global token (3.9%) instead of Europe Scoring; Poland enters the list (1.2-1.3%) -- the USSR's
  setup moved to Poland.
* **Layer-2 hubs:** Italy, Poland and East Germany (2.4-3.7% at 260M) are replaced by scoring cards (Europe 2.9, Asia
  2.6, Middle East 2.4%), Jordan (2.2%) and, when the USSR moves, The Cambridge Five (3.5%).
* **Gating sharpened in layer 2:** countries read the card in play at 31.5x uniform (260M 13.1x), the global token at
  40x (18.5x); cards read the opponent's known cards more (an opponent-known card reads others like it at 9.5x).
* **Layer 3 by location:** cards read the mover's hand at 7-10x uniform whatever their own location (260M 6-10x, the hand
  cards themselves only 6.5x, now 8.4x); countries moved from the hand (6.2x -> 1.8x) to the opponent's known cards
  (4.2x -> 4.5x) and the card in play (3.0x -> 4.6x).

## Reading

By 830M the deep layers compute mostly over the six scoring cards and Europe Scoring in particular -- a small,
tightly coupled sub-network that every card consults -- while country-level detail lives in layers 1-2. That fits the
play: scoring timing and Europe (the USSR's Europe-control wins, the US's empty West Germany) are where J5's games are
decided. Whether the concentration is a strength (a compact game-state summary) or a bottleneck (country detail
squeezed through six tokens) is not something attention alone can say; an ablation (masking the scoring-card tokens in
layer 3) would.

## Does anything look for a specific card? (830M)

`data/reports/j5_card_seek_830M.txt` (`j5_card_seek.py`): attention to card c averaged only over positions where c sits in
one place -- unseen by the mover (deck / opponent's unknown hand) or visible (own hand, opponent-known, discard,
removed) -- so the location gating above does not masquerade as interest, and a query's pull compared with what the other
queries of its kind give c there (lift). Scoring cards excluded as targets.

* **Rule-shaped pulls exist, a handful:** Pakistan, North Korea, Thailand -> The China Card (visible, layer 1, ~3x
  uniform, lift 3-4; the China Card's Asia bonus); Gulf States -> OPEC (visible, layer 1); Finland, Turkey, the United
  Kingdom -> Truman Doctrine while it is unseen (layer 2, 12-13x uniform; Truman strips USSR influence from an
  uncontrolled European country); Poland -> Warsaw Pact Formed while unseen (layer 2, 10x uniform, lift 10), and Europe
  Scoring -> Warsaw Pact Formed / Five Year Plan / SALT / Nuclear Test Ban while unseen (layer 3, lifts 22-51).
* **A few cards are tracked by everyone:** Wargames (unseen and visible; Southeast Asia Scoring reads it at 36x uniform
  when visible), The Cambridge Five, CIA Created, Aldrich Ames Remix (the most-read unseen card in layer 3, 6.3x), The
  Iron Lady (5.3x) -- hand-information and game-ending cards.
* **Most of the strongest pulls carry no rule:** Dominican Republic / Ecuador -> Wargames, Allende, AWACS; Terrorism ->
  Aldrich Ames / Pershing II / "An Evil Empire" (7-15x uniform); Turkey and Spain/Portugal -> Allende. These look like
  tokens of absent or irrelevant cards serving as scratch registers, not card knowledge; 24 games, so single cells are
  noisy.

## By side, and cards reading cards (830M)

`data/reports/j5_card_card_830M.txt` (`j5_card_card.py`; 1,950 US and 1,915 USSR positions).

* **The rule-shaped pulls are side-specific:** European countries (and Australia) read an unseen Truman Doctrine at 24-28x
  uniform when the US moves; Poland and North Korea read an unseen Warsaw Pact Formed at 20x when the USSR moves; Europe
  Scoring reads an unseen Warsaw Pact Formed when the US moves (lift 56) and an unseen Five Year Plan when the USSR moves
  (lift 25). Southeast Asia Scoring reading a visible Wargames (35-37x) is the same for both.
* **Held cards mostly do not read rule-related cards.** For fifteen cards whose rules name other cards (UN Intervention,
  The China Card, Missile Envy, Grain Sales, Star Wars, Ask Not, Five Year Plan, Blockade, Red Scare, CIA Created,
  Aldrich Ames, Defectors, Cambridge Five, Wargames, Lone Gunman), what each reads in its own hand, among unseen cards, in
  the discard and among removed cards is mostly unrelated cards (Star Wars, which retrieves from the discard, reads
  Wargames and The Iron Lady there; Grain Sales reads Colonial Rear Guards and Junta). 34-1,040 positions per card and
  side, so cells are noisy.
* **UN Intervention is the partial exception.** Held by the US, its strongest reads in its own hand in layer 3 are three
  USSR events -- Romanian Abdication (lift 10), Korean War (6.8), Fidel (2.4) -- the cards it exists to play; held by the
  USSR, Special Relationship (a US event, lift 3.7) and Captured Nazi Scientist. Over every held card, the attention a
  card gives the opponent's events beside it in hand relative to its own/neutral cards: UN Intervention 1.88 (US) and
  2.02 (USSR) in layer 2, against medians 0.81 and 1.13 (rank 14 of 104 and 24 of 100) -- above typical, not the top (the
  top ratios, e.g. Soviets Shoot Down KAL-007 at 118, come from a handful of hands).

## Dead card tokens as scratch space (830M)

Owner's observation: Southeast Asia Scoring (played by turn 7, then removed) reading Wargames (enters at turn 8) pairs two
tokens that almost never coexist as live cards -- the token of a dead card doing work. J5's trunk reads the global token
and the MEAN of the 110 card tokens, so a removed or not-yet-dealt card's token can carry computed state to the trunk.
`data/reports/j5_dead_tokens_830M.txt` (`j5_dead_tokens.py`; 2,555 positions, ~53 dead card tokens per position):

| intervention on card tokens (every block) | tokens | policy KL (mean / median) | top move changes | mean abs value change |
|:---|---:|---:|---:|---:|
| mean-ablate the dead ones (each replaced by its card's average dead state) | 52.7 | 0.046 / 0.017 | 12.9% | 0.113 |
| freeze the dead ones | 52.7 | 0.057 / 0.021 | 14.1% | 0.117 |
| freeze unseen ones (deck / opponent's unknown hand) | 26.7 | 0.043 / 0.012 | 11.0% | 0.071 |
| freeze the own hand | 5.1 | 0.049 / 0.009 | 12.6% | 0.045 |
| freeze 5 random dead ones | 5.1 | 0.004 / 0.000 | 3.0% | 0.016 |

Removing only the position-specific content of the dead tokens moves the policy about as much as freezing the whole hand
and the value 2.5x as much: J5 computes in its dead card tokens and reads the result through the card mean. The use is
spread thin -- five dead tokens matter a tenth of five hand tokens -- and it is not the high-norm artifact of vision
transformers (removed cards 93, deck 78, hand 97, countries 169 in norm before the final norm; no outliers), nor an
attention sink (attention *to* removed and absent cards stays at 0.3-1.2x uniform): the dead tokens work as queries
and write into the mean. This is the behaviour "Vision Transformers Need Registers" (Darcet et al., 2023) reports for
low-information patches, and what register / memory tokens (Burtsev et al., 2020) provide on purpose. The cost here
specific to the game: the dead set changes -- mid- and late-war cards enter at turns 4 and 8, cards are removed every
turn -- so scratch kept in a card's token must move when that card becomes live.
