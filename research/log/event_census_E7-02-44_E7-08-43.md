# Event census of E7-02-44 and E7-08-43 at 1,200M, against the soup (2026-10-03)

Owner: repeat the [soup's event census](event_census_soup.md) for E7-02-44 and E7-08-43.

**Method.** `tools/scripts/event_play_census.py`, the same tool and size as the soup's census
(4,096 greedy self-play games per model; here seed 55,000 in batches of 512, the soup's were
8 × 512): one count per holding (the card entering
its owner's hand until it leaves), counting only holdings where the owner could have played the
event at one of its decisions; US/USSR cards by their owner only, neutral cards by whoever holds
them. Models: **E7-02-44@1,200M** (seed 44, the shallow trunk; one of the soup's four
ingredients, `snapshot_1200029696steps.pt`) and **E7-08-43@1,200M** (the same recipe, seed 43,
55 Elo below seed 44 at saturation). The soup is
`shallow_E7-02+03+04+05_1200M` from the earlier census. Dumps:
`data/eval/event_census/E7-02-44_1200M.json`, `.../E7-08-43_1200M.json`.

## Result

**How stable the census is.** Per card, a difference counts when it is at least 10 points and
more than 3 standard errors.

| pair | mean \|difference\| | cards that differ |
|:---|---:|---:|
| E7-02-44 vs E7-08-43 (two seeds of one recipe) | 5.9 pp | 20 of 109 |
| soup vs E7-02-44 (the soup vs one of its own ingredients) | 4.9 pp | 22 of 109 |
| soup vs E7-08-43 | 5.9 pp | 21 of 109 |

* **About a fifth of the cards' event rates move between models of the same recipe**, and the
  soup is as far from its own ingredient as two seeds are from each other. The cards at the top
  of the census (UN Intervention, scoring cards, Decolonization, Brush War, Junta, The Voice of
  America, Colonial Rear Guards: 91-98%) and the bottom are the same in all three; the middle is
  where they disagree. So a rate in the soup's census is a property of that model, not of the
  recipe, unless the other two agree with it.
* **Quagmire is the one large seed difference.** The seed-44 lineage events it almost always
  (E7-02-44 92%, the soup 89%, mostly headlined); **E7-08-43 only 16%** -- headlined 16%, never
  evented in a round, so it is played for Ops. The other seed-43 lows: Blockade 6% (E7-02-44
  39%), Star Wars 8% (34%), NORAD 1% (21%), Aldrich Ames 54% (80%), Ask Not 51% (77%), Lone Gunman
  8% (26%). Seed 43 events more of US/Japan Mutual Defense Pact (22% vs 4%), Puppet Governments,
  Solidarity, South African Unrest, Glasnost and Containment.
* **Where the soup differs from both raw models**, averaging the branches changed the behaviour
  rather than the seed: Korean War (soup 61%, E7-02-44 91%, E7-08-43 93%), Vietnam Revolts (56%
  vs 31%, 25%), De Gaulle Leads France (43% vs 62%, 62%), Sadat Expels Soviets (77% vs 57%, 59%),
  Soviets Shoot Down KAL-007 (51% vs 35%, 41%).
* Whether any of these differences costs games is not measured here; Quagmire is the obvious
  candidate to test first (it is a strong USSR event, and the weaker seed almost never uses it).

## The three censuses side by side

Evented share of holdings (holdings), sorted by the soup; * marks a largest gap above 3 standard
errors.

| card | side | soup | E7-02-44 | E7-08-43 | largest gap |
|:---|:---|---:|---:|---:|---:|
| UN Intervention | neutral | 98% (8,458) | 99% (8,467) | 98% (8,384) | 1 |
| Pershing II Deployed | ussr | 97% (1,024) | 95% (1,023) | 88% (938) | 9 * |
| The Reformer | ussr | 97% (1,038) | 95% (975) | 94% (965) | 3 * |
| Junta | neutral | 97% (5,372) | 96% (5,236) | 94% (5,269) | 3 * |
| Decolonization | ussr | 97% (4,128) | 96% (4,148) | 96% (4,060) | 1 |
| Brush War | neutral | 96% (5,477) | 96% (5,428) | 96% (5,372) | 0 |
| The Voice of America | us | 96% (2,647) | 97% (2,633) | 95% (2,533) | 2 * |
| Asia Scoring | neutral | 95% (9,107) | 95% (8,998) | 95% (9,005) | 0 |
| Liberation Theology | ussr | 95% (2,557) | 93% (2,478) | 92% (2,489) | 3 * |
| Colonial Rear Guards | us | 95% (2,656) | 93% (2,619) | 94% (2,689) | 2 * |
| Middle East Scoring | neutral | 93% (8,910) | 93% (8,925) | 94% (8,880) | 1 |
| Europe Scoring | neutral | 93% (9,008) | 92% (8,987) | 93% (8,976) | 1 |
| OAS Founded | us | 91% (1,789) | 84% (1,813) | 83% (1,826) | 8 * |
| Tear Down this Wall | us | 91% (1,037) | 91% (999) | 90% (999) | 1 |
| Southeast Asia Scoring | neutral | 90% (3,816) | 89% (3,777) | 90% (3,779) | 1 |
| South America Scoring | neutral | 90% (5,417) | 89% (5,339) | 90% (5,356) | 1 |
| Central America Scoring | neutral | 90% (5,433) | 89% (5,304) | 91% (5,404) | 2 * |
| Africa Scoring | neutral | 90% (5,392) | 89% (5,281) | 91% (5,417) | 2 * |
| Containment | us | 90% (1,882) | 77% (1,961) | 90% (1,920) | 13 * |
| Quagmire | ussr | 89% (2,169) | 92% (2,146) | 16% (2,187) | 76 * |
| ABM Treaty | neutral | 87% (5,535) | 88% (5,265) | 89% (5,394) | 2 * |
| Grain Sales to Soviets | us | 85% (2,636) | 81% (2,620) | 92% (2,610) | 11 * |
| Marine Barracks Bombing | ussr | 85% (1,083) | 78% (983) | 67% (954) | 18 * |
| De-Stalinization | ussr | 85% (2,952) | 86% (2,946) | 86% (2,964) | 1 |
| Captured Nazi Scientist | neutral | 84% (4,680) | 83% (4,822) | 86% (4,545) | 3 * |
| Allende | ussr | 82% (2,003) | 73% (2,042) | 78% (2,048) | 9 * |
| Defectors | us | 81% (4,579) | 71% (4,504) | 69% (4,563) | 12 * |
| Panama Canal Returned | us | 79% (1,849) | 67% (1,857) | 60% (1,879) | 19 * |
| Sadat Expels Soviets | us | 77% (1,835) | 57% (1,956) | 59% (1,925) | 20 * |
| Red Scare/Purge | neutral | 76% (9,116) | 80% (9,191) | 85% (9,076) | 9 * |
| Brezhnev Doctrine | ussr | 74% (2,029) | 71% (1,977) | 79% (2,075) | 8 * |
| Aldrich Ames Remix | ussr | 73% (1,047) | 80% (928) | 54% (1,014) | 26 * |
| Indo-Pakistani War | neutral | 72% (8,840) | 66% (8,771) | 70% (8,725) | 6 * |
| Nixon Plays the China Card | us | 69% (2,073) | 66% (2,104) | 68% (2,066) | 3 |
| “Ask Not What Your Country Can Do For You…” | us | 68% (2,120) | 77% (2,051) | 51% (2,139) | 26 * |
| Camp David Accords | us | 67% (2,183) | 57% (2,228) | 65% (2,195) | 10 * |
| Iran-Iraq War | neutral | 66% (2,103) | 58% (1,999) | 59% (1,989) | 8 * |
| Solidarity | us | 65% (527) | 52% (538) | 69% (493) | 17 * |
| Puppet Governments | us | 65% (2,109) | 45% (2,104) | 68% (2,040) | 23 * |
| South African Unrest | ussr | 63% (2,546) | 56% (2,455) | 71% (2,563) | 15 * |
| Korean War | ussr | 61% (3,055) | 91% (2,958) | 93% (2,933) | 32 * |
| Nasser | ussr | 61% (2,673) | 54% (2,761) | 36% (2,884) | 25 * |
| Socialist Governments | ussr | 60% (3,897) | 71% (4,021) | 67% (3,988) | 11 * |
| Missile Envy | neutral | 57% (5,287) | 47% (5,143) | 56% (5,240) | 10 * |
| Vietnam Revolts | ussr | 56% (2,745) | 31% (2,902) | 25% (2,904) | 31 * |
| Soviets Shoot Down KAL-007 | us | 51% (1,037) | 35% (968) | 41% (1,016) | 16 * |
| Ussuri River Skirmish | us | 48% (2,248) | 57% (2,211) | 61% (2,161) | 13 * |
| Fidel | ussr | 48% (3,083) | 52% (3,051) | 46% (3,108) | 6 * |
| De Gaulle Leads France | ussr | 43% (3,135) | 62% (2,880) | 62% (2,822) | 19 * |
| Suez Crisis | ussr | 38% (3,136) | 54% (2,988) | 31% (3,080) | 23 * |
| Cultural Revolution | ussr | 31% (2,217) | 41% (2,139) | 37% (2,250) | 10 * |
| CIA Created | us | 30% (2,612) | 31% (2,569) | 21% (2,643) | 10 * |
| OPEC | ussr | 27% (2,477) | 31% (2,422) | 26% (2,418) | 5 * |
| Blockade | ussr | 25% (3,039) | 39% (2,955) | 6% (3,154) | 33 * |
| Star Wars | us | 24% (604) | 34% (537) | 8% (494) | 26 * |
| Portuguese Empire Crumbles | ussr | 24% (2,329) | 34% (2,126) | 40% (2,267) | 16 * |
| Truman Doctrine | us | 24% (2,515) | 24% (2,691) | 28% (2,617) | 4 * |
| Bear Trap | us | 23% (2,081) | 26% (2,132) | 23% (2,144) | 3 |
| Latin American Debt Crisis | ussr | 21% (1,067) | 29% (989) | 25% (1,000) | 8 * |
| “Lone Gunman” | ussr | 21% (2,425) | 26% (2,251) | 8% (2,350) | 18 * |
| Duck and Cover | us | 20% (4,627) | 30% (4,560) | 23% (4,599) | 10 * |
| Alliance for Progress | us | 19% (2,343) | 26% (2,179) | 28% (2,184) | 9 * |
| Willy Brandt | ussr | 16% (2,269) | 12% (2,188) | 24% (2,301) | 12 * |
| SALT Negotiations | neutral | 16% (5,050) | 20% (4,891) | 14% (5,074) | 6 * |
| Muslim Revolution | ussr | 15% (2,477) | 25% (2,423) | 13% (2,479) | 12 * |
| NORAD | us | 14% (3,048) | 21% (2,993) | 1% (2,944) | 20 * |
| Marshall Plan | us | 13% (3,029) | 14% (2,845) | 17% (2,933) | 4 * |
| “We Will Bury You” | ussr | 12% (2,520) | 12% (2,406) | 9% (2,520) | 3 * |
| Che | ussr | 10% (2,536) | 13% (2,492) | 14% (2,520) | 4 * |
| Glasnost | ussr | 10% (1,058) | 14% (1,008) | 29% (928) | 19 * |
| Terrorism | neutral | 9% (2,096) | 20% (1,983) | 11% (1,996) | 11 * |
| John Paul II Elected Pope | us | 7% (2,278) | 19% (2,246) | 14% (2,291) | 12 * |
| The Cambridge Five | ussr | 6% (3,520) | 8% (3,378) | 8% (3,393) | 2 * |
| Kitchen Debates | us | 6% (2,523) | 5% (2,430) | 9% (2,575) | 4 * |
| US/Japan Mutual Defense Pact | us | 6% (3,157) | 4% (3,249) | 22% (2,870) | 18 * |
| Arab-Israeli War | ussr | 6% (3,506) | 8% (3,530) | 10% (3,474) | 4 * |
| Olympic Games | neutral | 5% (8,939) | 4% (8,887) | 3% (8,894) | 2 * |
| East European Unrest | us | 4% (4,638) | 8% (4,679) | 10% (4,502) | 6 * |
| How I Learned to Stop Worrying | neutral | 4% (5,215) | 3% (5,181) | 4% (5,162) | 1 |
| Iranian Hostage Crisis | ussr | 4% (1,056) | 9% (969) | 14% (1,000) | 10 * |
| Our Man in Tehran | us | 3% (2,352) | 5% (2,287) | 3% (2,274) | 2 * |
| “An Evil Empire” | us | 3% (1,041) | 4% (961) | 4% (979) | 1 |
| Romanian Abdication | ussr | 3% (3,140) | 5% (3,098) | 5% (3,090) | 2 * |
| Yuri and Samantha | ussr | 2% (1,027) | 4% (990) | 1% (979) | 3 * |
| Special Relationship | us | 2% (4,797) | 3% (4,692) | 2% (4,594) | 1 |
| Flower Power | ussr | 2% (2,166) | 3% (2,166) | 1% (2,180) | 2 * |
| Cuban Missile Crisis | neutral | 2% (5,272) | 2% (5,133) | 3% (5,209) | 1 |
| Shuttle Diplomacy | us | 1% (2,355) | 0% (2,323) | 0% (2,415) | 1 |
| Independent Reds | us | 1% (2,899) | 1% (2,999) | 1% (3,062) | 0 |
| AWACS Sale to Saudis | us | 1% (1,047) | 6% (958) | 1% (1,021) | 5 * |
| Iran-Contra Scandal | ussr | 1% (1,051) | 6% (976) | 3% (1,027) | 5 * |
| Five Year Plan | us | 1% (4,543) | 4% (4,633) | 3% (4,665) | 3 * |
| Reagan Bombs Libya | us | 1% (1,062) | 2% (966) | 1% (985) | 1 |
| Comecon | ussr | 0% (3,185) | 0% (3,162) | 1% (3,219) | 1 |
| Warsaw Pact Formed | ussr | 0% (3,204) | 1% (3,226) | 1% (3,184) | 1 |
| Summit | neutral | 0% (5,314) | 0% (5,118) | 1% (5,234) | 1 |
| Chernobyl | us | 0% (1,026) | 1% (1,000) | 2% (987) | 2 * |
| Nuclear Test Ban | neutral | 0% (9,016) | 1% (8,934) | 1% (8,979) | 1 |
| The Iron Lady | us | 0% (1,046) | 1% (1,021) | 1% (1,008) | 1 |
| U-2 Incident | ussr | 0% (2,246) | 2% (2,241) | 1% (2,278) | 2 * |
| Arms Race | neutral | 0% (5,271) | 1% (5,209) | 0% (5,229) | 1 |
| Latin American Death Squads | neutral | 0% (5,290) | 2% (5,149) | 3% (5,236) | 3 * |
| Formosan Resolution | us | 0% (2,984) | 0% (3,052) | 1% (3,053) | 1 |
| North Sea Oil | us | 0% (1,039) | 3% (966) | 1% (985) | 3 * |
| Wargames | neutral | 0% (2,082) | 1% (1,980) | 0% (1,981) | 1 |
| NATO | us | 0% (2,021) | 0% (2,078) | 0% (2,191) | 0 |
| Nuclear Subs | us | 0% (2,262) | 0% (2,182) | 1% (2,221) | 1 |
| Ortega Elected in Nicaragua | ussr | 0% (1,050) | 1% (961) | 0% (1,032) | 1 |
| “One Small Step…” | neutral | 0% (5,299) | 1% (5,174) | 1% (5,265) | 1 |

## E7-02-44@1,200M

| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |
|:---|:---|---:|---:|---:|---:|---:|
| UN Intervention | neutral | **99%** (8,467) | 0% | 99% | 99% (4,608) | 98% (3,859) |
| The Voice of America | us | **97%** (2,633) | 34% | 63% |  |  |
| Junta | neutral | **96%** (5,236) | 65% | 31% | 95% (2,726) | 97% (2,510) |
| Brush War | neutral | **96%** (5,428) | 35% | 60% | 96% (2,810) | 95% (2,618) |
| Decolonization | ussr | **96%** (4,148) | 18% | 78% |  |  |
| Asia Scoring | neutral | **95%** (8,998) | 22% | 73% | 95% (4,838) | 96% (4,160) |
| The Reformer | ussr | **95%** (975) | 55% | 40% |  |  |
| Pershing II Deployed | ussr | **95%** (1,023) | 51% | 44% |  |  |
| Middle East Scoring | neutral | **93%** (8,925) | 21% | 73% | 93% (4,825) | 93% (4,100) |
| Liberation Theology | ussr | **93%** (2,478) | 39% | 53% |  |  |
| Colonial Rear Guards | us | **93%** (2,619) | 19% | 74% |  |  |
| Europe Scoring | neutral | **92%** (8,987) | 14% | 79% | 92% (4,808) | 93% (4,179) |
| Quagmire | ussr | **92%** (2,146) | 76% | 16% |  |  |
| Korean War | ussr | **91%** (2,958) | 58% | 34% |  |  |
| Tear Down this Wall | us | **91%** (999) | 37% | 54% |  |  |
| Central America Scoring | neutral | **89%** (5,304) | 11% | 78% | 87% (2,768) | 92% (2,536) |
| Southeast Asia Scoring | neutral | **89%** (3,777) | 9% | 80% | 84% (1,809) | 93% (1,968) |
| Africa Scoring | neutral | **89%** (5,281) | 12% | 77% | 85% (2,649) | 93% (2,632) |
| South America Scoring | neutral | **89%** (5,339) | 8% | 81% | 85% (2,743) | 92% (2,596) |
| ABM Treaty | neutral | **88%** (5,265) | 59% | 29% | 95% (2,748) | 80% (2,517) |
| De-Stalinization | ussr | **86%** (2,946) | 16% | 70% |  |  |
| OAS Founded | us | **84%** (1,813) | 9% | 75% |  |  |
| Captured Nazi Scientist | neutral | **83%** (4,822) | 32% | 50% | 75% (2,179) | 89% (2,643) |
| Grain Sales to Soviets | us | **81%** (2,620) | 81% | 0% |  |  |
| Red Scare/Purge | neutral | **80%** (9,191) | 79% | 0% | 68% (5,032) | 94% (4,159) |
| Aldrich Ames Remix | ussr | **80%** (928) | 79% | 1% |  |  |
| Marine Barracks Bombing | ussr | **78%** (983) | 13% | 66% |  |  |
| “Ask Not What Your Country Can Do For You…” | us | **77%** (2,051) | 43% | 34% |  |  |
| Containment | us | **77%** (1,961) | 77% | 0% |  |  |
| Allende | ussr | **73%** (2,042) | 25% | 48% |  |  |
| Defectors | us | **71%** (4,504) | 71% | 0% |  |  |
| Brezhnev Doctrine | ussr | **71%** (1,977) | 70% | 1% |  |  |
| Socialist Governments | ussr | **71%** (4,021) | 41% | 30% |  |  |
| Panama Canal Returned | us | **67%** (1,857) | 11% | 56% |  |  |
| Nixon Plays the China Card | us | **66%** (2,104) | 7% | 59% |  |  |
| Indo-Pakistani War | neutral | **66%** (8,771) | 25% | 41% | 63% (4,611) | 70% (4,160) |
| De Gaulle Leads France | ussr | **62%** (2,880) | 42% | 20% |  |  |
| Iran-Iraq War | neutral | **58%** (1,999) | 19% | 39% | 70% (1,038) | 44% (961) |
| Camp David Accords | us | **57%** (2,228) | 10% | 47% |  |  |
| Sadat Expels Soviets | us | **57%** (1,956) | 8% | 49% |  |  |
| Ussuri River Skirmish | us | **57%** (2,211) | 24% | 33% |  |  |
| South African Unrest | ussr | **56%** (2,455) | 15% | 42% |  |  |
| Nasser | ussr | **54%** (2,761) | 32% | 22% |  |  |
| Suez Crisis | ussr | **54%** (2,988) | 22% | 31% |  |  |
| Solidarity | us | **52%** (538) | 2% | 50% |  |  |
| Fidel | ussr | **52%** (3,051) | 12% | 40% |  |  |
| Missile Envy | neutral | **47%** (5,143) | 47% | 0% | 61% (2,647) | 33% (2,496) |
| Puppet Governments | us | **45%** (2,104) | 18% | 27% |  |  |
| Cultural Revolution | ussr | **41%** (2,139) | 21% | 21% |  |  |
| Blockade | ussr | **39%** (2,955) | 1% | 38% |  |  |
| Soviets Shoot Down KAL-007 | us | **35%** (968) | 35% | 0% |  |  |
| Star Wars | us | **34%** (537) | 33% | 1% |  |  |
| Portuguese Empire Crumbles | ussr | **34%** (2,126) | 13% | 21% |  |  |
| Vietnam Revolts | ussr | **31%** (2,902) | 21% | 10% |  |  |
| CIA Created | us | **31%** (2,569) | 31% | 0% |  |  |
| OPEC | ussr | **31%** (2,422) | 18% | 13% |  |  |
| Duck and Cover | us | **30%** (4,560) | 30% | 0% |  |  |
| Latin American Debt Crisis | ussr | **29%** (989) | 6% | 23% |  |  |
| Alliance for Progress | us | **26%** (2,179) | 12% | 14% |  |  |
| Bear Trap | us | **26%** (2,132) | 26% | 0% |  |  |
| “Lone Gunman” | ussr | **26%** (2,251) | 23% | 3% |  |  |
| Muslim Revolution | ussr | **25%** (2,423) | 21% | 4% |  |  |
| Truman Doctrine | us | **24%** (2,691) | 8% | 16% |  |  |
| NORAD | us | **21%** (2,993) | 20% | 1% |  |  |
| SALT Negotiations | neutral | **20%** (4,891) | 18% | 3% | 28% (2,523) | 12% (2,368) |
| Terrorism | neutral | **20%** (1,983) | 18% | 2% | 9% (992) | 30% (991) |
| John Paul II Elected Pope | us | **19%** (2,246) | 10% | 9% |  |  |
| Marshall Plan | us | **14%** (2,845) | 14% | 0% |  |  |
| Glasnost | ussr | **14%** (1,008) | 10% | 4% |  |  |
| Che | ussr | **13%** (2,492) | 8% | 5% |  |  |
| Willy Brandt | ussr | **12%** (2,188) | 4% | 8% |  |  |
| “We Will Bury You” | ussr | **12%** (2,406) | 12% | 0% |  |  |
| Iranian Hostage Crisis | ussr | **9%** (969) | 6% | 4% |  |  |
| The Cambridge Five | ussr | **8%** (3,378) | 8% | 0% |  |  |
| Arab-Israeli War | ussr | **8%** (3,530) | 2% | 6% |  |  |
| East European Unrest | us | **8%** (4,679) | 7% | 1% |  |  |
| AWACS Sale to Saudis | us | **6%** (958) | 5% | 1% |  |  |
| Iran-Contra Scandal | ussr | **6%** (976) | 0% | 5% |  |  |
| Kitchen Debates | us | **5%** (2,430) | 3% | 2% |  |  |
| Our Man in Tehran | us | **5%** (2,287) | 3% | 2% |  |  |
| Romanian Abdication | ussr | **5%** (3,098) | 4% | 1% |  |  |
| US/Japan Mutual Defense Pact | us | **4%** (3,249) | 4% | 0% |  |  |
| Yuri and Samantha | ussr | **4%** (990) | 1% | 3% |  |  |
| Five Year Plan | us | **4%** (4,633) | 4% | 0% |  |  |
| “An Evil Empire” | us | **4%** (961) | 1% | 3% |  |  |
| Olympic Games | neutral | **4%** (8,887) | 4% | 0% | 7% (4,702) | 0% (4,185) |
| Flower Power | ussr | **3%** (2,166) | 2% | 1% |  |  |
| North Sea Oil | us | **3%** (966) | 0% | 3% |  |  |
| How I Learned to Stop Worrying | neutral | **3%** (5,181) | 2% | 1% | 5% (2,601) | 1% (2,580) |
| Special Relationship | us | **3%** (4,692) | 1% | 1% |  |  |
| Reagan Bombs Libya | us | **2%** (966) | 1% | 2% |  |  |
| Cuban Missile Crisis | neutral | **2%** (5,133) | 2% | 1% | 1% (2,639) | 4% (2,494) |
| Latin American Death Squads | neutral | **2%** (5,149) | 2% | 0% | 2% (2,588) | 2% (2,561) |
| U-2 Incident | ussr | **2%** (2,241) | 0% | 2% |  |  |
| Nuclear Test Ban | neutral | **1%** (8,934) | 1% | 0% | 2% (4,828) | 0% (4,106) |
| Ortega Elected in Nicaragua | ussr | **1%** (961) | 0% | 1% |  |  |
| The Iron Lady | us | **1%** (1,021) | 0% | 1% |  |  |
| Chernobyl | us | **1%** (1,000) | 0% | 1% |  |  |
| Arms Race | neutral | **1%** (5,209) | 0% | 1% | 1% (2,636) | 2% (2,573) |
| Wargames | neutral | **1%** (1,980) | 0% | 1% | 1% (1,002) | 1% (978) |
| Independent Reds | us | **1%** (2,999) | 1% | 0% |  |  |
| “One Small Step…” | neutral | **1%** (5,174) | 0% | 0% | 1% (2,693) | 1% (2,481) |
| Warsaw Pact Formed | ussr | **1%** (3,226) | 0% | 0% |  |  |
| Shuttle Diplomacy | us | **0%** (2,323) | 0% | 0% |  |  |
| Nuclear Subs | us | **0%** (2,182) | 0% | 0% |  |  |
| Comecon | ussr | **0%** (3,162) | 0% | 0% |  |  |
| NATO | us | **0%** (2,078) | 0% | 0% |  |  |
| Summit | neutral | **0%** (5,118) | 0% | 0% | 0% (2,606) | 0% (2,512) |
| Formosan Resolution | us | **0%** (3,052) | 0% | 0% |  |  |

## E7-08-43@1,200M

| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |
|:---|:---|---:|---:|---:|---:|---:|
| UN Intervention | neutral | **98%** (8,384) | 0% | 98% | 99% (4,527) | 96% (3,857) |
| Decolonization | ussr | **96%** (4,060) | 28% | 68% |  |  |
| Brush War | neutral | **96%** (5,372) | 38% | 58% | 97% (2,759) | 94% (2,613) |
| Asia Scoring | neutral | **95%** (9,005) | 19% | 76% | 95% (4,772) | 96% (4,233) |
| The Voice of America | us | **95%** (2,533) | 29% | 66% |  |  |
| Colonial Rear Guards | us | **94%** (2,689) | 29% | 65% |  |  |
| Junta | neutral | **94%** (5,269) | 67% | 27% | 93% (2,732) | 95% (2,537) |
| Middle East Scoring | neutral | **94%** (8,880) | 20% | 73% | 92% (4,657) | 96% (4,223) |
| The Reformer | ussr | **94%** (965) | 61% | 33% |  |  |
| Korean War | ussr | **93%** (2,933) | 56% | 37% |  |  |
| Europe Scoring | neutral | **93%** (8,976) | 13% | 80% | 91% (4,760) | 95% (4,216) |
| Liberation Theology | ussr | **92%** (2,489) | 43% | 49% |  |  |
| Grain Sales to Soviets | us | **92%** (2,610) | 92% | 0% |  |  |
| Africa Scoring | neutral | **91%** (5,417) | 14% | 77% | 90% (2,704) | 93% (2,713) |
| Central America Scoring | neutral | **91%** (5,404) | 6% | 85% | 89% (2,785) | 93% (2,619) |
| Tear Down this Wall | us | **90%** (999) | 32% | 59% |  |  |
| Southeast Asia Scoring | neutral | **90%** (3,779) | 12% | 78% | 85% (1,828) | 94% (1,951) |
| South America Scoring | neutral | **90%** (5,356) | 9% | 81% | 88% (2,676) | 92% (2,680) |
| Containment | us | **90%** (1,920) | 90% | 0% |  |  |
| ABM Treaty | neutral | **89%** (5,394) | 70% | 19% | 88% (2,835) | 91% (2,559) |
| Pershing II Deployed | ussr | **88%** (938) | 26% | 62% |  |  |
| De-Stalinization | ussr | **86%** (2,964) | 12% | 74% |  |  |
| Captured Nazi Scientist | neutral | **86%** (4,545) | 33% | 53% | 90% (2,041) | 82% (2,504) |
| Red Scare/Purge | neutral | **85%** (9,076) | 84% | 0% | 73% (4,978) | 99% (4,098) |
| OAS Founded | us | **83%** (1,826) | 13% | 70% |  |  |
| Brezhnev Doctrine | ussr | **79%** (2,075) | 79% | 0% |  |  |
| Allende | ussr | **78%** (2,048) | 21% | 57% |  |  |
| South African Unrest | ussr | **71%** (2,563) | 28% | 43% |  |  |
| Indo-Pakistani War | neutral | **70%** (8,725) | 33% | 37% | 68% (4,633) | 71% (4,092) |
| Solidarity | us | **69%** (493) | 4% | 65% |  |  |
| Defectors | us | **69%** (4,563) | 69% | 0% |  |  |
| Nixon Plays the China Card | us | **68%** (2,066) | 12% | 57% |  |  |
| Puppet Governments | us | **68%** (2,040) | 20% | 48% |  |  |
| Socialist Governments | ussr | **67%** (3,988) | 37% | 30% |  |  |
| Marine Barracks Bombing | ussr | **67%** (954) | 16% | 50% |  |  |
| Camp David Accords | us | **65%** (2,195) | 16% | 48% |  |  |
| De Gaulle Leads France | ussr | **62%** (2,822) | 35% | 27% |  |  |
| Ussuri River Skirmish | us | **61%** (2,161) | 16% | 45% |  |  |
| Panama Canal Returned | us | **60%** (1,879) | 18% | 42% |  |  |
| Iran-Iraq War | neutral | **59%** (1,989) | 18% | 41% | 82% (1,018) | 36% (971) |
| Sadat Expels Soviets | us | **59%** (1,925) | 7% | 52% |  |  |
| Missile Envy | neutral | **56%** (5,240) | 56% | 0% | 67% (2,661) | 45% (2,579) |
| Aldrich Ames Remix | ussr | **54%** (1,014) | 54% | 0% |  |  |
| “Ask Not What Your Country Can Do For You…” | us | **51%** (2,139) | 19% | 32% |  |  |
| Fidel | ussr | **46%** (3,108) | 16% | 30% |  |  |
| Soviets Shoot Down KAL-007 | us | **41%** (1,016) | 41% | 0% |  |  |
| Portuguese Empire Crumbles | ussr | **40%** (2,267) | 28% | 12% |  |  |
| Cultural Revolution | ussr | **37%** (2,250) | 27% | 10% |  |  |
| Nasser | ussr | **36%** (2,884) | 28% | 8% |  |  |
| Suez Crisis | ussr | **31%** (3,080) | 20% | 12% |  |  |
| Glasnost | ussr | **29%** (928) | 28% | 1% |  |  |
| Alliance for Progress | us | **28%** (2,184) | 8% | 20% |  |  |
| Truman Doctrine | us | **28%** (2,617) | 13% | 15% |  |  |
| OPEC | ussr | **26%** (2,418) | 14% | 12% |  |  |
| Latin American Debt Crisis | ussr | **25%** (1,000) | 10% | 15% |  |  |
| Vietnam Revolts | ussr | **25%** (2,904) | 20% | 4% |  |  |
| Willy Brandt | ussr | **24%** (2,301) | 7% | 17% |  |  |
| Bear Trap | us | **23%** (2,144) | 23% | 0% |  |  |
| Duck and Cover | us | **23%** (4,599) | 23% | 0% |  |  |
| US/Japan Mutual Defense Pact | us | **22%** (2,870) | 22% | 0% |  |  |
| CIA Created | us | **21%** (2,643) | 21% | 0% |  |  |
| Marshall Plan | us | **17%** (2,933) | 16% | 0% |  |  |
| Quagmire | ussr | **16%** (2,187) | 16% | 0% |  |  |
| Iranian Hostage Crisis | ussr | **14%** (1,000) | 13% | 1% |  |  |
| Che | ussr | **14%** (2,520) | 11% | 2% |  |  |
| SALT Negotiations | neutral | **14%** (5,074) | 13% | 1% | 21% (2,640) | 6% (2,434) |
| John Paul II Elected Pope | us | **14%** (2,291) | 6% | 8% |  |  |
| Muslim Revolution | ussr | **13%** (2,479) | 12% | 0% |  |  |
| Terrorism | neutral | **11%** (1,996) | 8% | 2% | 11% (1,020) | 10% (976) |
| Arab-Israeli War | ussr | **10%** (3,474) | 7% | 3% |  |  |
| East European Unrest | us | **10%** (4,502) | 9% | 1% |  |  |
| “We Will Bury You” | ussr | **9%** (2,520) | 9% | 0% |  |  |
| Kitchen Debates | us | **9%** (2,575) | 3% | 6% |  |  |
| The Cambridge Five | ussr | **8%** (3,393) | 7% | 1% |  |  |
| Star Wars | us | **8%** (494) | 7% | 0% |  |  |
| “Lone Gunman” | ussr | **8%** (2,350) | 7% | 0% |  |  |
| Blockade | ussr | **6%** (3,154) | 3% | 3% |  |  |
| Romanian Abdication | ussr | **5%** (3,090) | 3% | 1% |  |  |
| “An Evil Empire” | us | **4%** (979) | 3% | 1% |  |  |
| How I Learned to Stop Worrying | neutral | **4%** (5,162) | 3% | 1% | 6% (2,590) | 1% (2,572) |
| Iran-Contra Scandal | ussr | **3%** (1,027) | 3% | 1% |  |  |
| Latin American Death Squads | neutral | **3%** (5,236) | 3% | 0% | 1% (2,717) | 6% (2,519) |
| Five Year Plan | us | **3%** (4,665) | 3% | 0% |  |  |
| Olympic Games | neutral | **3%** (8,894) | 3% | 0% | 5% (4,719) | 0% (4,175) |
| Our Man in Tehran | us | **3%** (2,274) | 2% | 1% |  |  |
| Cuban Missile Crisis | neutral | **3%** (5,209) | 2% | 0% | 1% (2,625) | 4% (2,584) |
| Chernobyl | us | **2%** (987) | 2% | 0% |  |  |
| Special Relationship | us | **2%** (4,594) | 0% | 1% |  |  |
| Nuclear Test Ban | neutral | **1%** (8,979) | 1% | 0% | 2% (4,793) | 1% (4,186) |
| AWACS Sale to Saudis | us | **1%** (1,021) | 1% | 1% |  |  |
| Warsaw Pact Formed | ussr | **1%** (3,184) | 1% | 1% |  |  |
| Formosan Resolution | us | **1%** (3,053) | 1% | 0% |  |  |
| Independent Reds | us | **1%** (3,062) | 0% | 1% |  |  |
| U-2 Incident | ussr | **1%** (2,278) | 1% | 0% |  |  |
| Flower Power | ussr | **1%** (2,180) | 1% | 0% |  |  |
| North Sea Oil | us | **1%** (985) | 1% | 0% |  |  |
| Nuclear Subs | us | **1%** (2,221) | 0% | 0% |  |  |
| “One Small Step…” | neutral | **1%** (5,265) | 0% | 0% | 1% (2,710) | 0% (2,555) |
| Comecon | ussr | **1%** (3,219) | 0% | 0% |  |  |
| The Iron Lady | us | **1%** (1,008) | 0% | 0% |  |  |
| Summit | neutral | **1%** (5,234) | 1% | 0% | 1% (2,666) | 0% (2,568) |
| NORAD | us | **1%** (2,944) | 0% | 0% |  |  |
| Yuri and Samantha | ussr | **1%** (979) | 0% | 0% |  |  |
| Reagan Bombs Libya | us | **1%** (985) | 0% | 0% |  |  |
| Shuttle Diplomacy | us | **0%** (2,415) | 0% | 0% |  |  |
| Ortega Elected in Nicaragua | ussr | **0%** (1,032) | 0% | 0% |  |  |
| Arms Race | neutral | **0%** (5,229) | 0% | 0% | 0% (2,701) | 0% (2,528) |
| NATO | us | **0%** (2,191) | 0% | 0% |  |  |
| Wargames | neutral | **0%** (1,981) | 0% | 0% | 0% (1,026) | 0% (955) |
