# Where raw play loses: paired playouts by decision kind

Points of win probability a game, both sides' decisions summed (435.3 decisions with two or more legal moves a game): each playout row's paired gain of the best move over raw's, weighted back to the decisions raw play meets. 'k=8 @256 over raw' is what that searcher's own move recovers.

Screened census: the best move is chosen on the screen's pairs, and its gain measured on fresh pairs where the screen's lead held (escalated); other rows count as no loss, so the total is a lower bound. 'raw' is the move under study (annotate --player: the player's).

## total

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| escalated | 932 | 11.88% | +58.7 ± 3.4 | +43.8 ± 2.8 |
| not escalated | 7068 | 88.12% | +0.0 ± 0.0 | -137.9 ± 5.5 |

## all positions: decision kind, stratified bootstrap (2000 resamples)

Positions resampled within their (bank stratum, playout class) cell: the ± counts the sampling of positions as well as the playout noise the tables below print alone.

| kind | best over raw | share | k=8 @256 over raw | leaks the most in |
|:---|---:|---:|---:|---:|
| target (influence) | +23.1 ± 6.2 | 39% | -19.8 ± 6.9 | 60% |
| AR card choice | +20.9 ± 3.1 | 36% | -47.4 ± 5.3 | 40% |
| play mode | +7.6 ± 2.0 | 13% | -15.2 ± 2.0 | 0% |
| headline | +5.3 ± 0.6 | 9% | -6.8 ± 0.9 | 0% |
| target (coup) | +1.7 ± 1.4 | 3% | -0.2 ± 1.3 | 0% |
| op mode | +0.7 ± 0.3 | 1% | -4.7 ± 0.6 | 0% |
| choose_branch | +0.1 ± 0.1 | 0% | -0.1 ± 0.1 | 0% |
| target (realign) | -0.8 ± 0.7 | -1% | +0.2 ± 0.7 | 0% |
| all | +58.7 ± 7.5 |  | -94.1 ± 9.2 |  |

## all positions: decision kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| target (influence) | 1467 | 41.10% | +23.1 ± 3.0 | -19.8 ± 4.8 |
| AR card choice | 1734 | 23.44% | +20.9 ± 1.2 | -47.4 ± 3.3 |
| play mode | 1722 | 20.96% | +7.6 ± 0.9 | -15.2 ± 1.4 |
| headline | 1446 | 4.01% | +5.3 ± 0.2 | -6.8 ± 0.4 |
| target (coup) | 157 | 4.75% | +1.7 ± 0.6 | -0.2 ± 1.2 |
| op mode | 1125 | 3.21% | +0.7 ± 0.1 | -4.7 ± 0.3 |
| choose_branch | 239 | 0.29% | +0.1 ± 0.0 | -0.1 ± 0.1 |
| target (realign) | 110 | 2.24% | -0.8 ± 0.4 | +0.2 ± 0.7 |

## all positions: side x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| US target (influence) | 742 | 20.02% | +12.2 ± 2.1 | -14.3 ± 3.3 |
| US AR card choice | 885 | 11.86% | +12.1 ± 0.8 | -20.0 ± 2.3 |
| USSR target (influence) | 725 | 21.08% | +10.9 ± 2.2 | -5.6 ± 3.5 |
| USSR AR card choice | 849 | 11.58% | +8.8 ± 0.9 | -27.4 ± 2.4 |
| USSR play mode | 840 | 9.88% | +4.3 ± 0.7 | -7.9 ± 1.0 |
| US play mode | 882 | 11.08% | +3.3 ± 0.5 | -7.3 ± 0.9 |
| US headline | 745 | 2.06% | +2.8 ± 0.2 | -3.8 ± 0.3 |
| USSR headline | 701 | 1.96% | +2.5 ± 0.2 | -3.0 ± 0.3 |
| US target (coup) | 57 | 1.56% | +0.9 ± 0.6 | -0.0 ± 0.7 |
| USSR target (coup) | 100 | 3.19% | +0.8 ± 0.2 | -0.2 ± 0.9 |

## all positions: era x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| T4-7 AR card choice | 578 | 10.77% | +9.4 ± 0.9 | -20.0 ± 2.4 |
| T4-7 target (influence) | 466 | 14.70% | +8.4 ± 1.7 | -6.6 ± 2.7 |
| T8-10 target (influence) | 467 | 6.45% | +8.3 ± 0.5 | -5.6 ± 1.0 |
| T8-10 AR card choice | 578 | 4.48% | +6.9 ± 0.3 | -11.3 ± 0.7 |
| T1-3 target (influence) | 534 | 19.95% | +6.4 ± 2.4 | -7.7 ± 3.8 |
| T1-3 AR card choice | 578 | 8.19% | +4.6 ± 0.8 | -16.2 ± 2.1 |
| T8-10 play mode | 574 | 4.12% | +4.6 ± 0.3 | -4.3 ± 0.5 |
| T1-3 play mode | 574 | 7.71% | +2.0 ± 0.7 | -4.4 ± 0.7 |
| T8-10 headline | 383 | 0.80% | +2.0 ± 0.1 | -0.6 ± 0.1 |
| T4-7 headline | 537 | 1.82% | +1.7 ± 0.2 | -2.6 ± 0.3 |
| T1-3 headline | 526 | 1.40% | +1.6 ± 0.2 | -3.6 ± 0.3 |
| T4-7 target (coup) | 60 | 2.15% | +1.1 ± 0.3 | +0.1 ± 0.9 |

## all positions: DEFCON x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| D2 other | 2047 | 35.78% | +18.9 ± 2.5 | -19.1 ± 4.1 |
| D2 AR card choice | 1357 | 18.64% | +16.1 ± 1.1 | -37.9 ± 2.9 |
| D3 other | 1709 | 10.37% | +8.5 ± 1.3 | -9.3 ± 2.3 |
| D2 play mode | 1271 | 16.50% | +4.6 ± 0.6 | -11.1 ± 1.2 |
| D4+ other | 788 | 9.45% | +2.8 ± 1.3 | -3.1 ± 1.6 |
| D3 play mode | 233 | 2.46% | +2.7 ± 0.6 | -2.4 ± 0.5 |
| D4+ AR card choice | 153 | 2.05% | +2.5 ± 0.5 | -3.8 ± 1.1 |
| D3 AR card choice | 224 | 2.74% | +2.3 ± 0.4 | -5.8 ± 1.1 |
| D4+ play mode | 218 | 2.00% | +0.3 ± 0.3 | -1.7 ± 0.5 |

## all positions, AR card choice: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 3 -> opp 2 | 6 | 0.07% | +1.1 ± 0.1 | +0.2 ± 0.2 |
| neutral 2 -> opp 2 | 8 | 0.08% | +1.0 ± 0.2 | +0.2 ± 0.3 |
| own 2 -> opp 2 | 6 | 0.08% | +0.7 ± 0.2 | +0.2 ± 0.3 |
| neutral 2 -> opp 3 | 6 | 0.10% | +0.6 ± 0.2 | +0.6 ± 0.2 |
| opp 2 -> opp 3 | 7 | 0.05% | +0.4 ± 0.1 | +0.6 ± 0.2 |
| opp 1 -> opp 1 | 80 | 1.13% | +0.4 ± 0.2 | -3.3 ± 0.8 |
| opp 3 -> opp 3 | 225 | 2.97% | +0.4 ± 0.1 | -6.2 ± 1.3 |
| own 3 -> opp 1 | 6 | 0.09% | +0.3 ± 0.2 | +0.3 ± 0.2 |
| own 2 -> own 3 | 5 | 0.04% | +0.2 ± 0.1 | +0.7 ± 0.3 |
| opp 4 -> opp 4 | 59 | 1.01% | +0.2 ± 0.2 | -2.4 ± 0.5 |
| opp 2 -> opp 1 | 5 | 0.06% | +0.1 ± 0.2 | +0.0 ± 0.1 |
| neutral 3 -> neutral 3 | 32 | 0.40% | +0.1 ± 0.1 | -0.6 ± 0.4 |

## all positions, AR card choice: the card raw plays instead (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Muslim Revolution | 17 | 0.25% | +1.7 ± 0.0 | -0.5 ± 0.3 |
| Ussuri River Skirmish | 16 | 0.26% | +1.6 ± 0.2 | +0.3 ± 0.3 |
| Socialist Governments | 22 | 0.30% | +1.3 ± 0.2 | +0.0 ± 0.3 |
| The China Card | 45 | 0.63% | +1.2 ± 0.3 | -3.0 ± 0.7 |
| Brush War | 12 | 0.12% | +0.9 ± 0.2 | +0.4 ± 0.2 |
| Defectors | 16 | 0.22% | +0.9 ± 0.2 | -0.6 ± 0.4 |
| Iran-Iraq War | 15 | 0.14% | +0.7 ± 0.1 | +0.1 ± 0.1 |
| UN Intervention | 33 | 0.55% | +0.6 ± 0.3 | -0.0 ± 0.5 |
| How I Learned to Stop Worrying | 19 | 0.25% | +0.6 ± 0.1 | +0.0 ± 0.3 |
| Marshall Plan | 22 | 0.40% | +0.5 ± 0.2 | -1.0 ± 0.3 |
| Arms Race | 12 | 0.17% | +0.5 ± 0.2 | -0.1 ± 0.2 |
| South America Scoring | 14 | 0.23% | +0.5 ± 0.2 | -0.9 ± 0.3 |

## all positions, AR card choice: the card the best move plays (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| The China Card | 51 | 0.70% | +1.7 ± 0.3 | -3.2 ± 0.7 |
| Latin American Death Squads | 11 | 0.14% | +1.6 ± 0.0 | -0.1 ± 0.3 |
| De-Stalinization | 21 | 0.26% | +1.3 ± 0.2 | -1.9 ± 0.3 |
| pass | 12 | 0.13% | +0.9 ± 0.1 | +0.1 ± 0.2 |
| Latin American Debt Crisis | 12 | 0.08% | +0.8 ± 0.1 | -0.5 ± 0.1 |
| De Gaulle Leads France | 21 | 0.24% | +0.7 ± 0.2 | -0.6 ± 0.5 |
| Allende | 19 | 0.31% | +0.7 ± 0.2 | -1.2 ± 0.5 |
| Our Man in Tehran | 11 | 0.15% | +0.6 ± 0.2 | -0.3 ± 0.4 |
| East European Unrest | 35 | 0.46% | +0.5 ± 0.2 | +0.4 ± 0.5 |
| Portuguese Empire Crumbles | 22 | 0.26% | +0.5 ± 0.1 | -0.6 ± 0.4 |
| Missile Envy | 9 | 0.20% | +0.5 ± 0.2 | -0.3 ± 0.4 |
| Blockade | 20 | 0.30% | +0.5 ± 0.1 | -0.8 ± 0.4 |

## all positions, play mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own card: COUP -> INFL | 9 | 0.10% | +1.1 ± 0.3 | +0.4 ± 0.1 |
| opp card: INFL -> COUP | 4 | 0.08% | +0.9 ± 0.3 | +0.0 ± 0.0 |
| opp card: SPACE -> INFL | 7 | 0.03% | +0.7 ± 0.1 | +0.2 ± 0.1 |
| own card: INFL -> REALIGN | 6 | 0.04% | +0.5 ± 0.1 | +0.2 ± 0.1 |
| opp card: EVENT -> SPACE | 4 | 0.02% | +0.4 ± 0.0 | +0.3 ± 0.1 |
| opp card: INFL -> SPACE | 4 | 0.04% | +0.4 ± 0.1 | +0.2 ± 0.0 |
| own card: EVENT -> INFL | 11 | 0.09% | +0.4 ± 0.2 | +0.7 ± 0.1 |
| opp card: SPACE -> EVENT | 6 | 0.06% | +0.3 ± 0.2 | +0.0 ± 0.0 |
| opp card: COUP -> INFL | 14 | 0.06% | +0.3 ± 0.1 | +0.2 ± 0.1 |
| neutral card: COUP -> REALIGN | 4 | 0.02% | +0.3 ± 0.1 | +0.0 ± 0.0 |

## all positions, play mode: the card (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Blockade | 28 | 0.37% | +0.9 ± 0.3 | -0.7 ± 0.2 |
| “One Small Step…” | 13 | 0.12% | +0.8 ± 0.1 | -0.1 ± 0.2 |
| Defectors | 32 | 0.41% | +0.8 ± 0.3 | -0.6 ± 0.2 |
| The Iron Lady | 8 | 0.08% | +0.7 ± 0.1 | -0.2 ± 0.1 |
| Truman Doctrine | 20 | 0.32% | +0.5 ± 0.3 | -0.1 ± 0.1 |
| John Paul II Elected Pope | 20 | 0.22% | +0.4 ± 0.1 | -0.1 ± 0.2 |
| NORAD | 21 | 0.27% | +0.3 ± 0.1 | -0.4 ± 0.1 |
| Terrorism | 8 | 0.02% | +0.3 ± 0.0 | +0.1 ± 0.1 |
| US/Japan Mutual Defense Pact | 21 | 0.40% | +0.3 ± 0.1 | -0.1 ± 0.1 |
| “An Evil Empire” | 10 | 0.06% | +0.3 ± 0.0 | +0.3 ± 0.1 |

## all positions, op mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| COUP -> INFL | 19 | 0.04% | +0.4 ± 0.1 | +0.3 ± 0.0 |
| COUP -> REALIGN | 13 | 0.02% | +0.2 ± 0.0 | +0.2 ± 0.0 |
| INFL -> REALIGN | 15 | 0.04% | +0.2 ± 0.1 | +0.2 ± 0.0 |
| REALIGN -> INFL | 3 | 0.00% | +0.0 ± 0.0 | +0.0 ± 0.0 |
| COUP -> COUP | 478 | 1.04% | +0.0 ± 0.0 | -2.9 ± 0.2 |
| INFL -> INFL | 490 | 1.81% | +0.0 ± 0.0 | -2.1 ± 0.2 |

## all positions, influence targets: raw's region -> the best region

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Europe -> Europe | 395 | 12.28% | +4.0 ± 1.3 | -2.7 ± 2.2 |
| Europe -> Africa | 10 | 0.30% | +3.6 ± 0.6 | -1.3 ± 0.7 |
| Europe -> Asia | 12 | 0.40% | +3.6 ± 0.8 | +2.2 ± 0.7 |
| Europe -> Middle East | 8 | 0.24% | +3.2 ± 0.7 | +1.2 ± 0.5 |
| Asia -> Europe | 5 | 0.22% | +2.1 ± 0.7 | +0.3 ± 0.2 |
| Europe -> South America | 4 | 0.14% | +1.5 ± 0.5 | +0.2 ± 0.1 |
| Africa -> Africa | 179 | 4.31% | +0.8 ± 0.6 | -6.2 ± 1.9 |
| Central America -> Central America | 162 | 3.64% | +0.6 ± 0.2 | -6.7 ± 1.7 |
| South America -> Europe | 4 | 0.03% | +0.4 ± 0.1 | +0.2 ± 0.2 |
| Asia -> Middle East | 7 | 0.07% | +0.3 ± 0.3 | +1.3 ± 0.5 |

## all positions, headline: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 3 -> scoring 0 | 14 | 0.03% | +0.7 ± 0.1 | +0.1 ± 0.0 |
| own 3 -> own 2 | 13 | 0.03% | +0.5 ± 0.1 | +0.1 ± 0.0 |
| own 2 -> own 2 | 257 | 0.68% | +0.3 ± 0.1 | -1.3 ± 0.2 |
| neutral 2 -> own 2 | 9 | 0.02% | +0.3 ± 0.1 | +0.2 ± 0.0 |
| neutral 2 -> scoring 0 | 4 | 0.01% | +0.2 ± 0.0 | +0.0 ± 0.0 |
| own 3 -> neutral 2 | 6 | 0.02% | +0.2 ± 0.0 | +0.1 ± 0.0 |
| neutral 3 -> own 1 | 4 | 0.02% | +0.2 ± 0.1 | +0.0 ± 0.0 |
| own 2 -> own 3 | 7 | 0.01% | +0.2 ± 0.0 | +0.0 ± 0.0 |

## The 20 positions that cost raw the most a game (all positions)

| turn/AR | side | DEFCON | kind | raw | best | gain a decision | a game |
|:---|:---|---:|:---|:---|:---|---:|---:|
| T10 AR4 | US | 2 | target (influence) | PointNode #0 (Canada) | PointNode #50 (Saharan States) | +16.9 ± 1.0 | +2.31 |
| T3 AR2 | USSR | 2 | target (influence) | PointNode #36 (Thailand) | PointNode #11 (Greece) | +5.7 ± 1.3 | +2.24 |
| T10 AR4 | US | 2 | target (influence) | PointNode #10 (Italy) | PointNode #38 (Malaysia) | +13.9 ± 1.0 | +1.90 |
| T7 AR2 | US | 2 | AR card choice | SelectCard #56 (Muslim Revolution) | SelectCard #69 (Latin American Death Squads) | +66.7 ± 1.1 | +1.61 |
| T1 AR0 | US | 5 | target (influence) | PointNode #10 (Italy) | PointNode #7 (West Germany) | +4.1 ± 1.5 | +1.59 |
| T1 AR3 | USSR | 3 | target (influence) | PointNode #5 (Finland) | PointNode #22 (Syria) | +3.6 ± 1.5 | +1.42 |
| T7 AR5 | US | 2 | AR card choice | SelectCard #76 (Ussuri River Skirmish) | SelectCard #33 (De-Stalinization) | +11.9 ± 1.4 | +1.37 |
| T5 AR5 | USSR | 2 | target (influence) | PointNode #8 (France) | PointNode #70 (Panama) | +4.1 ± 1.2 | +1.34 |
| T6 AR3 | US | 2 | target (influence) | PointNode #15 (Poland) | PointNode #79 (Brazil) | +3.4 ± 1.4 | +1.12 |
| T6 AR1 | US | 3 | target (coup) | PointNode #57 (Zaire) | PointNode #22 (Syria) | +3.3 ± 0.7 | +1.10 |
| T7 AR7 | USSR | 2 | target (influence) | PointNode #29 (Egypt) | PointNode #57 (Zaire) | +3.3 ± 1.3 | +1.10 |
| T6 AR6 | USSR | 2 | target (influence) | PointNode #47 (Algeria) | PointNode #63 (South Africa) | +3.1 ± 1.2 | +1.04 |
| T1 AR0 | USSR | 5 | target (influence) | PointNode #5 (Finland) | PointNode #14 (East Germany) | +2.3 ± 1.4 | +0.90 |
| T5 AR3 | USSR | 2 | target (influence) | PointNode #15 (Poland) | PointNode #28 (Saudi Arabia) | +4.8 ± 1.4 | +0.87 |
| T9 AR4 | USSR | 2 | target (influence) | PointNode #14 (East Germany) | PointNode #22 (Syria) | +6.0 ± 0.8 | +0.82 |
| T3 AR2 | USSR | 3 | play mode | Resolution: OPS_COUP / OpMode: COUP | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | +4.1 ± 1.5 | +0.82 |
| T1 AR1 | US | 5 | target (influence) | PointNode #1 (United Kingdom) | PointNode #44 (South Korea) | +3.7 ± 1.5 | +0.81 |
| T10 AR5 | USSR | 3 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_COUP / OpMode: COUP | +10.1 ± 0.7 | +0.77 |
| T7 AR2 | US | 2 | target (influence) | PointNode #8 (France) | PointNode #70 (Panama) | +2.2 ± 1.3 | +0.73 |
| T5 AR7 | USSR | 2 | target (influence) | PointNode #5 (Finland) | PointNode #17 (Hungary) | +2.1 ± 1.3 | +0.69 |
