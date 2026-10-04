# Event census of E7-21-44 (league) at 3,160M (2026-10-04)

Owner: repeat the events-vs-Ops analysis of
[event_census_E7-02-44_E7-08-43.md](event_census_E7-02-44_E7-08-43.md) on E7-21-44's latest
snapshot.

**Model.** E7-21-44@3,160M (`E7-21-44_20261004_190016/snapshot_3160014848steps.pt`; the run was
still training when it was taken). It is the seed-44 shallow line throughout: E7-02-44 to 1,200M,
continued with the same flags as E7-17-44 (2,000M), E7-19-44 (2,400M) and E7-20-44 (2,800M), then
the **P24 league** from E7-20-44's end state (league fraction 0.5, 4 published exploiters, no
setup credit) -- so 3,160M is 360M into the league. To tell the league from the extra 1,600M of
plain training, **E7-20-44@2,800M** (the league's starting point) was censused as well.

**Method.** `tools/scripts/event_play_census.py` exactly as for E7-02-44 and E7-08-43: 4,096
greedy self-play games per model, seed 55,000, batches of 512; one count per holding where the
owner could have played the event. Dumps: `data/eval/event_census/E7-21-44_3160M.json`,
`.../E7-20-44_2800M.json`.

## Result

How far apart the censuses are (a card counts when it differs by ≥ 10 points and > 3 standard
errors):

| pair | what separates them | mean \|difference\| | cards that differ |
|:---|:---|---:|---:|
| E7-20-44@2,800M vs E7-02-44@1,200M | 1,600M more plain training | 6.3 pp | 24 |
| E7-21-44@3,160M vs E7-20-44@2,800M | 360M of league | 4.6 pp | 18 |
| E7-21-44@3,160M vs E7-02-44@1,200M | both | 5.8 pp | 23 |
| E7-21-44@3,160M vs soup | | 6.4 pp | 25 |
| E7-21-44@3,160M vs E7-08-43@1,200M | | 7.6 pp | 28 |
| E7-02-44@1,200M vs E7-08-43@1,200M (earlier) | another seed | 5.9 pp | 20 |

**Reading.**

* **The largest changes came with the longer training, before the league.** Star Wars goes from
  34% evented (E7-02-44) to **86% at 2,800M** and 90% in the league -- now mostly as the event in
  an action round (E7-21-44: 72% in a round, 18% headlined; E7-02-44: 1% and 33%). John Paul II
  Elected Pope: 19% → 63% → **82%** (61% in a round). Truman Doctrine 24% → 58% → 43%, Marshall
  Plan 14% → 30% → 34%, East European Unrest 8% → 20% → 25%. The other way: Aldrich Ames Remix 80%
  → 50% → 57%, Ask Not 77% → 60% → 58%.
* **360M of league moved 18 cards**, some back toward E7-02-44: De Gaulle Leads France 62% → 16%
  → **49%**, Nasser 54% → 90% → **61%**, Vietnam Revolts 31% → 43% → 25%; and some further:
  Blockade 39% → 37% → **16%**, Truman Doctrine back down to 43%, Brezhnev Doctrine 71% → 72% →
  56%.
* **Stable through all of it:** Korean War (89-91% from 1,200M on), Quagmire (86-92%; seed 43's
  16% stays a seed difference), and the cards at the top and the bottom of the census. KAL-007's
  headline: 35% → 44% → 51%.
* As with the seeds, each window of training -- 1,600M plain, then 360M of league -- rewrites about
  a fifth of the cards' event rates. The census is a snapshot of one model, not a converged
  property of the recipe; a card's rate should be read against its neighbours on the line before
  it is called a preference. Whether the changes win games is not measured here.

## All five censuses

Evented share of holdings (holdings), sorted by E7-21-44; the last two columns are the change from
the longer training (E7-20-44 − E7-02-44) and from the league (E7-21-44 − E7-20-44), * where ≥ 10
points and > 3 standard errors.

| card | side | soup | E7-02-44@1,200M | E7-08-43@1,200M | E7-20-44@2,800M | E7-21-44@3,160M | longer training (E7-20 − E7-02) | league (E7-21 − E7-20) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| UN Intervention | neutral | 98% (8,458) | 99% (8,467) | 98% (8,384) | 97% (8,351) | 98% (8,514) | -2 | +1 |
| The Voice of America | us | 96% (2,647) | 97% (2,633) | 95% (2,533) | 95% (2,656) | 96% (2,810) | -2 | +1 |
| Decolonization | ussr | 97% (4,128) | 96% (4,148) | 96% (4,060) | 96% (4,050) | 96% (4,191) | +0 | +0 |
| Junta | neutral | 97% (5,372) | 96% (5,236) | 94% (5,269) | 96% (5,089) | 96% (5,289) | +0 | +0 |
| Brush War | neutral | 96% (5,477) | 96% (5,428) | 96% (5,372) | 95% (5,286) | 96% (5,514) | -1 | +1 |
| Tear Down this Wall | us | 91% (1,037) | 91% (999) | 90% (999) | 95% (925) | 94% (988) | +4 | -1 |
| The Reformer | ussr | 97% (1,038) | 95% (975) | 94% (965) | 96% (937) | 94% (1,045) | +1 | -2 |
| Europe Scoring | neutral | 93% (9,008) | 92% (8,987) | 93% (8,976) | 91% (8,834) | 92% (8,959) | -1 | +1 |
| Pershing II Deployed | ussr | 97% (1,024) | 95% (1,023) | 88% (938) | 95% (931) | 92% (998) | +0 | -3 |
| Liberation Theology | ussr | 95% (2,557) | 93% (2,478) | 92% (2,489) | 92% (2,328) | 92% (2,581) | -1 | +0 |
| Asia Scoring | neutral | 95% (9,107) | 95% (8,998) | 95% (9,005) | 93% (8,870) | 92% (9,033) | -2 | -1 |
| Middle East Scoring | neutral | 93% (8,910) | 93% (8,925) | 94% (8,880) | 92% (8,735) | 91% (8,898) | -1 | -1 |
| Colonial Rear Guards | us | 95% (2,656) | 93% (2,619) | 94% (2,689) | 92% (2,523) | 90% (2,668) | -1 | -2 |
| Africa Scoring | neutral | 90% (5,392) | 89% (5,281) | 91% (5,417) | 89% (5,063) | 90% (5,327) | +0 | +1 |
| Korean War | ussr | 61% (3,055) | 91% (2,958) | 93% (2,933) | 89% (2,957) | 90% (2,933) | -2 | +1 |
| Star Wars | us | 24% (604) | 34% (537) | 8% (494) | 86% (521) | 90% (535) | +52 * | +4 |
| Southeast Asia Scoring | neutral | 90% (3,816) | 89% (3,777) | 90% (3,779) | 89% (3,646) | 89% (3,762) | +0 | +0 |
| Central America Scoring | neutral | 90% (5,433) | 89% (5,304) | 91% (5,404) | 89% (5,125) | 89% (5,283) | +0 | +0 |
| South America Scoring | neutral | 90% (5,417) | 89% (5,339) | 90% (5,356) | 86% (5,109) | 88% (5,362) | -3 | +2 |
| Quagmire | ussr | 89% (2,169) | 92% (2,146) | 16% (2,187) | 92% (2,121) | 86% (2,209) | +0 | -6 |
| ABM Treaty | neutral | 87% (5,535) | 88% (5,265) | 89% (5,394) | 83% (5,279) | 86% (5,470) | -5 | +3 |
| Captured Nazi Scientist | neutral | 84% (4,680) | 83% (4,822) | 86% (4,545) | 88% (4,595) | 85% (4,742) | +5 | -3 |
| Defectors | us | 81% (4,579) | 71% (4,504) | 69% (4,563) | 82% (4,456) | 84% (4,683) | +11 * | +2 |
| Grain Sales to Soviets | us | 85% (2,636) | 81% (2,620) | 92% (2,610) | 81% (2,527) | 82% (2,641) | +0 | +1 |
| John Paul II Elected Pope | us | 7% (2,278) | 19% (2,246) | 14% (2,291) | 63% (2,051) | 82% (2,175) | +44 * | +19 * |
| Marine Barracks Bombing | ussr | 85% (1,083) | 78% (983) | 67% (954) | 83% (955) | 81% (957) | +5 | -2 |
| OAS Founded | us | 91% (1,789) | 84% (1,813) | 83% (1,826) | 82% (1,752) | 80% (1,852) | -2 | -2 |
| De-Stalinization | ussr | 85% (2,952) | 86% (2,946) | 86% (2,964) | 77% (2,897) | 78% (2,879) | -9 | +1 |
| Allende | ussr | 82% (2,003) | 73% (2,042) | 78% (2,048) | 81% (1,963) | 77% (2,077) | +8 | -4 |
| Red Scare/Purge | neutral | 76% (9,116) | 80% (9,191) | 85% (9,076) | 79% (8,970) | 77% (9,035) | -1 | -2 |
| Socialist Governments | ussr | 60% (3,897) | 71% (4,021) | 67% (3,988) | 76% (3,884) | 76% (3,928) | +5 | +0 |
| Nixon Plays the China Card | us | 69% (2,073) | 66% (2,104) | 68% (2,066) | 74% (1,918) | 70% (1,971) | +8 | -4 |
| Containment | us | 90% (1,882) | 77% (1,961) | 90% (1,920) | 64% (2,052) | 69% (2,064) | -13 * | +5 |
| Panama Canal Returned | us | 79% (1,849) | 67% (1,857) | 60% (1,879) | 70% (1,802) | 68% (1,841) | +3 | -2 |
| Indo-Pakistani War | neutral | 72% (8,840) | 66% (8,771) | 70% (8,725) | 63% (8,576) | 66% (8,793) | -3 | +3 |
| South African Unrest | ussr | 63% (2,546) | 56% (2,455) | 71% (2,563) | 49% (2,412) | 64% (2,474) | -7 | +15 * |
| Solidarity | us | 65% (527) | 52% (538) | 69% (493) | 61% (577) | 62% (664) | +9 | +1 |
| Nasser | ussr | 61% (2,673) | 54% (2,761) | 36% (2,884) | 90% (2,536) | 61% (2,709) | +36 * | -29 * |
| Sadat Expels Soviets | us | 77% (1,835) | 57% (1,956) | 59% (1,925) | 70% (1,742) | 59% (1,986) | +13 * | -11 * |
| “Ask Not What Your Country Can Do For You…” | us | 68% (2,120) | 77% (2,051) | 51% (2,139) | 60% (2,002) | 58% (2,167) | -17 * | -2 |
| Suez Crisis | ussr | 38% (3,136) | 54% (2,988) | 31% (3,080) | 55% (2,989) | 57% (3,047) | +1 | +2 |
| Missile Envy | neutral | 57% (5,287) | 47% (5,143) | 56% (5,240) | 61% (5,036) | 57% (5,283) | +14 * | -4 |
| Aldrich Ames Remix | ussr | 73% (1,047) | 80% (928) | 54% (1,014) | 50% (969) | 57% (999) | -30 * | +7 |
| Brezhnev Doctrine | ussr | 74% (2,029) | 71% (1,977) | 79% (2,075) | 72% (1,886) | 56% (2,128) | +1 | -16 * |
| Fidel | ussr | 48% (3,083) | 52% (3,051) | 46% (3,108) | 47% (3,149) | 54% (3,101) | -5 | +7 |
| Camp David Accords | us | 67% (2,183) | 57% (2,228) | 65% (2,195) | 58% (2,102) | 54% (2,152) | +1 | -4 |
| Soviets Shoot Down KAL-007 | us | 51% (1,037) | 35% (968) | 41% (1,016) | 44% (1,013) | 51% (1,008) | +9 | +7 |
| De Gaulle Leads France | ussr | 43% (3,135) | 62% (2,880) | 62% (2,822) | 16% (3,083) | 49% (2,905) | -46 * | +33 * |
| Iran-Iraq War | neutral | 66% (2,103) | 58% (1,999) | 59% (1,989) | 59% (1,855) | 48% (1,972) | +1 | -11 * |
| Puppet Governments | us | 65% (2,109) | 45% (2,104) | 68% (2,040) | 61% (1,929) | 47% (2,071) | +16 * | -14 * |
| Ussuri River Skirmish | us | 48% (2,248) | 57% (2,211) | 61% (2,161) | 55% (2,071) | 44% (2,196) | -2 | -11 * |
| Truman Doctrine | us | 24% (2,515) | 24% (2,691) | 28% (2,617) | 58% (2,115) | 43% (2,332) | +34 * | -15 * |
| OPEC | ussr | 27% (2,477) | 31% (2,422) | 26% (2,418) | 29% (2,329) | 37% (2,500) | -2 | +8 |
| Marshall Plan | us | 13% (3,029) | 14% (2,845) | 17% (2,933) | 30% (2,678) | 34% (2,657) | +16 * | +4 |
| “Lone Gunman” | ussr | 21% (2,425) | 26% (2,251) | 8% (2,350) | 31% (2,258) | 32% (2,347) | +5 | +1 |
| Portuguese Empire Crumbles | ussr | 24% (2,329) | 34% (2,126) | 40% (2,267) | 44% (2,081) | 32% (2,231) | +10 * | -12 * |
| SALT Negotiations | neutral | 16% (5,050) | 20% (4,891) | 14% (5,074) | 31% (4,572) | 31% (4,702) | +11 * | +0 |
| NORAD | us | 14% (3,048) | 21% (2,993) | 1% (2,944) | 26% (2,934) | 30% (2,895) | +5 | +4 |
| Bear Trap | us | 23% (2,081) | 26% (2,132) | 23% (2,144) | 16% (2,092) | 28% (2,115) | -10 * | +12 * |
| Cultural Revolution | ussr | 31% (2,217) | 41% (2,139) | 37% (2,250) | 46% (2,084) | 28% (2,169) | +5 | -18 * |
| Willy Brandt | ussr | 16% (2,269) | 12% (2,188) | 24% (2,301) | 32% (2,139) | 26% (2,210) | +20 * | -6 |
| East European Unrest | us | 4% (4,638) | 8% (4,679) | 10% (4,502) | 20% (4,398) | 25% (4,435) | +12 * | +5 |
| Vietnam Revolts | ussr | 56% (2,745) | 31% (2,902) | 25% (2,904) | 43% (2,777) | 25% (2,923) | +12 * | -18 * |
| Alliance for Progress | us | 19% (2,343) | 26% (2,179) | 28% (2,184) | 36% (2,115) | 22% (2,173) | +10 * | -14 * |
| CIA Created | us | 30% (2,612) | 31% (2,569) | 21% (2,643) | 19% (2,617) | 21% (2,680) | -12 * | +2 |
| Duck and Cover | us | 20% (4,627) | 30% (4,560) | 23% (4,599) | 17% (4,559) | 19% (4,636) | -13 * | +2 |
| Blockade | ussr | 25% (3,039) | 39% (2,955) | 6% (3,154) | 37% (2,896) | 16% (3,172) | -2 | -21 * |
| Muslim Revolution | ussr | 15% (2,477) | 25% (2,423) | 13% (2,479) | 5% (2,322) | 14% (2,474) | -20 * | +9 |
| “We Will Bury You” | ussr | 12% (2,520) | 12% (2,406) | 9% (2,520) | 25% (2,276) | 12% (2,386) | +13 * | -13 * |
| Terrorism | neutral | 9% (2,096) | 20% (1,983) | 11% (1,996) | 17% (1,909) | 12% (1,992) | -3 | -5 |
| Che | ussr | 10% (2,536) | 13% (2,492) | 14% (2,520) | 12% (2,382) | 11% (2,554) | -1 | -1 |
| Latin American Debt Crisis | ussr | 21% (1,067) | 29% (989) | 25% (1,000) | 26% (931) | 11% (1,030) | -3 | -15 * |
| Glasnost | ussr | 10% (1,058) | 14% (1,008) | 29% (928) | 17% (929) | 9% (1,012) | +3 | -8 |
| Arab-Israeli War | ussr | 6% (3,506) | 8% (3,530) | 10% (3,474) | 12% (3,483) | 7% (3,504) | +4 | -5 |
| Romanian Abdication | ussr | 3% (3,140) | 5% (3,098) | 5% (3,090) | 5% (3,060) | 7% (3,083) | +0 | +2 |
| Kitchen Debates | us | 6% (2,523) | 5% (2,430) | 9% (2,575) | 13% (2,377) | 7% (2,475) | +8 | -6 |
| Special Relationship | us | 2% (4,797) | 3% (4,692) | 2% (4,594) | 5% (4,541) | 7% (4,686) | +2 | +2 |
| US/Japan Mutual Defense Pact | us | 6% (3,157) | 4% (3,249) | 22% (2,870) | 8% (3,023) | 6% (3,130) | +4 | -2 |
| The Cambridge Five | ussr | 6% (3,520) | 8% (3,378) | 8% (3,393) | 6% (3,368) | 6% (3,378) | -2 | +0 |
| Reagan Bombs Libya | us | 1% (1,062) | 2% (966) | 1% (985) | 3% (941) | 6% (1,031) | +1 | +3 |
| How I Learned to Stop Worrying | neutral | 4% (5,215) | 3% (5,181) | 4% (5,162) | 6% (4,941) | 5% (5,113) | +3 | -1 |
| Olympic Games | neutral | 5% (8,939) | 4% (8,887) | 3% (8,894) | 3% (8,685) | 4% (8,891) | -1 | +1 |
| Five Year Plan | us | 1% (4,543) | 4% (4,633) | 3% (4,665) | 3% (4,413) | 3% (4,660) | -1 | +0 |
| Latin American Death Squads | neutral | 0% (5,290) | 2% (5,149) | 3% (5,236) | 4% (4,964) | 3% (5,219) | +2 | -1 |
| Iranian Hostage Crisis | ussr | 4% (1,056) | 9% (969) | 14% (1,000) | 3% (933) | 3% (960) | -6 | +0 |
| Iran-Contra Scandal | ussr | 1% (1,051) | 6% (976) | 3% (1,027) | 2% (935) | 2% (990) | -4 | +0 |
| The Iron Lady | us | 0% (1,046) | 1% (1,021) | 1% (1,008) | 3% (900) | 2% (986) | +2 | -1 |
| Chernobyl | us | 0% (1,026) | 1% (1,000) | 2% (987) | 2% (975) | 2% (969) | +1 | +0 |
| Cuban Missile Crisis | neutral | 2% (5,272) | 2% (5,133) | 3% (5,209) | 2% (4,953) | 2% (5,200) | +0 | +0 |
| “An Evil Empire” | us | 3% (1,041) | 4% (961) | 4% (979) | 6% (915) | 2% (968) | +2 | -4 |
| Our Man in Tehran | us | 3% (2,352) | 5% (2,287) | 3% (2,274) | 1% (2,224) | 1% (2,290) | -4 | +0 |
| Independent Reds | us | 1% (2,899) | 1% (2,999) | 1% (3,062) | 2% (2,958) | 1% (3,008) | +1 | -1 |
| Warsaw Pact Formed | ussr | 0% (3,204) | 1% (3,226) | 1% (3,184) | 2% (3,174) | 1% (3,235) | +1 | -1 |
| AWACS Sale to Saudis | us | 1% (1,047) | 6% (958) | 1% (1,021) | 3% (972) | 1% (999) | -3 | -2 |
| Comecon | ussr | 0% (3,185) | 0% (3,162) | 1% (3,219) | 5% (3,150) | 1% (3,190) | +5 | -4 |
| Yuri and Samantha | ussr | 2% (1,027) | 4% (990) | 1% (979) | 4% (952) | 1% (980) | +0 | -3 |
| Arms Race | neutral | 0% (5,271) | 1% (5,209) | 0% (5,229) | 1% (5,012) | 1% (5,179) | +0 | +0 |
| U-2 Incident | ussr | 0% (2,246) | 2% (2,241) | 1% (2,278) | 3% (2,150) | 1% (2,153) | +1 | -2 |
| North Sea Oil | us | 0% (1,039) | 3% (966) | 1% (985) | 1% (964) | 1% (1,001) | -2 | +0 |
| Summit | neutral | 0% (5,314) | 0% (5,118) | 1% (5,234) | 1% (5,005) | 0% (5,216) | +1 | -1 |
| Nuclear Subs | us | 0% (2,262) | 0% (2,182) | 1% (2,221) | 1% (2,105) | 0% (2,150) | +1 | -1 |
| Wargames | neutral | 0% (2,082) | 1% (1,980) | 0% (1,981) | 0% (1,856) | 0% (1,981) | -1 | +0 |
| Flower Power | ussr | 2% (2,166) | 3% (2,166) | 1% (2,180) | 2% (2,052) | 0% (2,170) | -1 | -2 |
| “One Small Step…” | neutral | 0% (5,299) | 1% (5,174) | 1% (5,265) | 1% (4,938) | 0% (5,226) | +0 | -1 |
| Shuttle Diplomacy | us | 1% (2,355) | 0% (2,323) | 0% (2,415) | 1% (2,339) | 0% (2,494) | +1 | -1 |
| Ortega Elected in Nicaragua | ussr | 0% (1,050) | 1% (961) | 0% (1,032) | 1% (996) | 0% (1,071) | +0 | -1 |
| NATO | us | 0% (2,021) | 0% (2,078) | 0% (2,191) | 0% (2,204) | 0% (2,215) | +0 | +0 |
| Formosan Resolution | us | 0% (2,984) | 0% (3,052) | 1% (3,053) | 0% (3,105) | 0% (3,049) | +0 | +0 |
| Nuclear Test Ban | neutral | 0% (9,016) | 1% (8,934) | 1% (8,979) | 2% (8,749) | 0% (9,045) | +1 | -2 |

## E7-21-44@3,160M

| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |
|:---|:---|---:|---:|---:|---:|---:|
| UN Intervention | neutral | **98%** (8,514) | 0% | 98% | 98% (4,522) | 98% (3,992) |
| Junta | neutral | **96%** (5,289) | 65% | 31% | 96% (2,759) | 97% (2,530) |
| Decolonization | ussr | **96%** (4,191) | 35% | 61% |  |  |
| The Voice of America | us | **96%** (2,810) | 32% | 64% |  |  |
| Brush War | neutral | **96%** (5,514) | 40% | 56% | 96% (2,786) | 95% (2,728) |
| Tear Down this Wall | us | **94%** (988) | 48% | 46% |  |  |
| The Reformer | ussr | **94%** (1,045) | 61% | 33% |  |  |
| Liberation Theology | ussr | **92%** (2,581) | 40% | 52% |  |  |
| Europe Scoring | neutral | **92%** (8,959) | 7% | 84% | 92% (4,799) | 91% (4,160) |
| Pershing II Deployed | ussr | **92%** (998) | 37% | 55% |  |  |
| Asia Scoring | neutral | **92%** (9,033) | 15% | 77% | 94% (4,784) | 88% (4,249) |
| Middle East Scoring | neutral | **91%** (8,898) | 13% | 78% | 91% (4,780) | 91% (4,118) |
| Colonial Rear Guards | us | **90%** (2,668) | 17% | 73% |  |  |
| Africa Scoring | neutral | **90%** (5,327) | 11% | 79% | 89% (2,750) | 91% (2,577) |
| Korean War | ussr | **90%** (2,933) | 38% | 52% |  |  |
| Star Wars | us | **90%** (535) | 18% | 72% |  |  |
| Southeast Asia Scoring | neutral | **89%** (3,762) | 9% | 81% | 86% (1,837) | 92% (1,925) |
| Central America Scoring | neutral | **89%** (5,283) | 6% | 83% | 87% (2,698) | 90% (2,585) |
| South America Scoring | neutral | **88%** (5,362) | 6% | 82% | 86% (2,741) | 91% (2,621) |
| Quagmire | ussr | **86%** (2,209) | 71% | 15% |  |  |
| ABM Treaty | neutral | **86%** (5,470) | 60% | 26% | 88% (2,911) | 84% (2,559) |
| Captured Nazi Scientist | neutral | **85%** (4,742) | 33% | 52% | 83% (2,147) | 86% (2,595) |
| Defectors | us | **84%** (4,683) | 84% | 0% |  |  |
| Grain Sales to Soviets | us | **82%** (2,641) | 82% | 0% |  |  |
| John Paul II Elected Pope | us | **82%** (2,175) | 21% | 61% |  |  |
| Marine Barracks Bombing | ussr | **81%** (957) | 18% | 63% |  |  |
| OAS Founded | us | **80%** (1,852) | 19% | 62% |  |  |
| De-Stalinization | ussr | **78%** (2,879) | 20% | 59% |  |  |
| Red Scare/Purge | neutral | **77%** (9,035) | 77% | 0% | 69% (4,841) | 86% (4,194) |
| Allende | ussr | **77%** (2,077) | 23% | 54% |  |  |
| Socialist Governments | ussr | **76%** (3,928) | 61% | 15% |  |  |
| Nixon Plays the China Card | us | **70%** (1,971) | 11% | 59% |  |  |
| Containment | us | **69%** (2,064) | 69% | 0% |  |  |
| Panama Canal Returned | us | **68%** (1,841) | 4% | 64% |  |  |
| Indo-Pakistani War | neutral | **66%** (8,793) | 18% | 48% | 62% (4,725) | 72% (4,068) |
| South African Unrest | ussr | **64%** (2,474) | 21% | 43% |  |  |
| Solidarity | us | **62%** (664) | 12% | 50% |  |  |
| Nasser | ussr | **61%** (2,709) | 41% | 20% |  |  |
| Sadat Expels Soviets | us | **59%** (1,986) | 15% | 44% |  |  |
| “Ask Not What Your Country Can Do For You…” | us | **58%** (2,167) | 28% | 29% |  |  |
| Aldrich Ames Remix | ussr | **57%** (999) | 57% | 0% |  |  |
| Missile Envy | neutral | **57%** (5,283) | 57% | 0% | 68% (2,759) | 44% (2,524) |
| Suez Crisis | ussr | **57%** (3,047) | 31% | 26% |  |  |
| Brezhnev Doctrine | ussr | **56%** (2,128) | 55% | 1% |  |  |
| Fidel | ussr | **54%** (3,101) | 18% | 37% |  |  |
| Camp David Accords | us | **54%** (2,152) | 11% | 43% |  |  |
| Soviets Shoot Down KAL-007 | us | **51%** (1,008) | 51% | 0% |  |  |
| De Gaulle Leads France | ussr | **49%** (2,905) | 26% | 23% |  |  |
| Iran-Iraq War | neutral | **48%** (1,972) | 12% | 37% | 66% (987) | 30% (985) |
| Puppet Governments | us | **47%** (2,071) | 8% | 39% |  |  |
| Ussuri River Skirmish | us | **44%** (2,196) | 20% | 24% |  |  |
| Truman Doctrine | us | **43%** (2,332) | 15% | 28% |  |  |
| OPEC | ussr | **37%** (2,500) | 17% | 20% |  |  |
| Marshall Plan | us | **34%** (2,657) | 34% | 0% |  |  |
| “Lone Gunman” | ussr | **32%** (2,347) | 31% | 1% |  |  |
| Portuguese Empire Crumbles | ussr | **32%** (2,231) | 12% | 19% |  |  |
| SALT Negotiations | neutral | **31%** (4,702) | 19% | 12% | 41% (2,422) | 20% (2,280) |
| NORAD | us | **30%** (2,895) | 30% | 0% |  |  |
| Cultural Revolution | ussr | **28%** (2,169) | 8% | 20% |  |  |
| Bear Trap | us | **28%** (2,115) | 28% | 0% |  |  |
| Willy Brandt | ussr | **26%** (2,210) | 6% | 19% |  |  |
| East European Unrest | us | **25%** (4,435) | 22% | 3% |  |  |
| Vietnam Revolts | ussr | **25%** (2,923) | 22% | 3% |  |  |
| Alliance for Progress | us | **22%** (2,173) | 11% | 11% |  |  |
| CIA Created | us | **21%** (2,680) | 21% | 0% |  |  |
| Duck and Cover | us | **19%** (4,636) | 19% | 0% |  |  |
| Blockade | ussr | **16%** (3,172) | 2% | 14% |  |  |
| Muslim Revolution | ussr | **14%** (2,474) | 13% | 0% |  |  |
| “We Will Bury You” | ussr | **12%** (2,386) | 12% | 0% |  |  |
| Terrorism | neutral | **12%** (1,992) | 9% | 2% | 5% (1,012) | 18% (980) |
| Che | ussr | **11%** (2,554) | 10% | 0% |  |  |
| Latin American Debt Crisis | ussr | **11%** (1,030) | 4% | 7% |  |  |
| Glasnost | ussr | **9%** (1,012) | 9% | 0% |  |  |
| Kitchen Debates | us | **7%** (2,475) | 4% | 3% |  |  |
| Romanian Abdication | ussr | **7%** (3,083) | 7% | 0% |  |  |
| Special Relationship | us | **7%** (4,686) | 4% | 4% |  |  |
| Arab-Israeli War | ussr | **7%** (3,504) | 4% | 2% |  |  |
| US/Japan Mutual Defense Pact | us | **6%** (3,130) | 6% | 0% |  |  |
| The Cambridge Five | ussr | **6%** (3,378) | 6% | 0% |  |  |
| Reagan Bombs Libya | us | **6%** (1,031) | 4% | 2% |  |  |
| How I Learned to Stop Worrying | neutral | **5%** (5,113) | 5% | 1% | 6% (2,608) | 5% (2,505) |
| Olympic Games | neutral | **4%** (8,891) | 4% | 0% | 6% (4,743) | 2% (4,148) |
| Iranian Hostage Crisis | ussr | **3%** (960) | 3% | 1% |  |  |
| Latin American Death Squads | neutral | **3%** (5,219) | 3% | 0% | 2% (2,671) | 3% (2,548) |
| Five Year Plan | us | **3%** (4,660) | 3% | 0% |  |  |
| Cuban Missile Crisis | neutral | **2%** (5,200) | 2% | 0% | 0% (2,684) | 4% (2,516) |
| The Iron Lady | us | **2%** (986) | 1% | 1% |  |  |
| Iran-Contra Scandal | ussr | **2%** (990) | 2% | 0% |  |  |
| “An Evil Empire” | us | **2%** (968) | 1% | 1% |  |  |
| Chernobyl | us | **2%** (969) | 1% | 1% |  |  |
| Comecon | ussr | **1%** (3,190) | 1% | 0% |  |  |
| Yuri and Samantha | ussr | **1%** (980) | 1% | 0% |  |  |
| Warsaw Pact Formed | ussr | **1%** (3,235) | 1% | 0% |  |  |
| Our Man in Tehran | us | **1%** (2,290) | 1% | 0% |  |  |
| U-2 Incident | ussr | **1%** (2,153) | 1% | 0% |  |  |
| AWACS Sale to Saudis | us | **1%** (999) | 0% | 1% |  |  |
| North Sea Oil | us | **1%** (1,001) | 0% | 0% |  |  |
| Independent Reds | us | **1%** (3,008) | 1% | 0% |  |  |
| Arms Race | neutral | **1%** (5,179) | 0% | 0% | 0% (2,651) | 1% (2,528) |
| Nuclear Subs | us | **0%** (2,150) | 0% | 0% |  |  |
| Shuttle Diplomacy | us | **0%** (2,494) | 0% | 0% |  |  |
| Nuclear Test Ban | neutral | **0%** (9,045) | 0% | 0% | 1% (4,826) | 0% (4,219) |
| “One Small Step…” | neutral | **0%** (5,226) | 0% | 0% | 0% (2,707) | 0% (2,519) |
| Summit | neutral | **0%** (5,216) | 0% | 0% | 0% (2,656) | 0% (2,560) |
| Formosan Resolution | us | **0%** (3,049) | 0% | 0% |  |  |
| Flower Power | ussr | **0%** (2,170) | 0% | 0% |  |  |
| NATO | us | **0%** (2,215) | 0% | 0% |  |  |
| Wargames | neutral | **0%** (1,981) | 0% | 0% | 0% (1,040) | 0% (941) |
| Ortega Elected in Nicaragua | ussr | **0%** (1,071) | 0% | 0% |  |  |

## E7-20-44@2,800M

| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |
|:---|:---|---:|---:|---:|---:|---:|
| UN Intervention | neutral | **97%** (8,351) | 0% | 97% | 96% (4,513) | 97% (3,838) |
| The Reformer | ussr | **96%** (937) | 68% | 28% |  |  |
| Decolonization | ussr | **96%** (4,050) | 29% | 67% |  |  |
| Junta | neutral | **96%** (5,089) | 66% | 30% | 94% (2,599) | 97% (2,490) |
| Brush War | neutral | **95%** (5,286) | 31% | 65% | 95% (2,716) | 95% (2,570) |
| The Voice of America | us | **95%** (2,656) | 28% | 68% |  |  |
| Tear Down this Wall | us | **95%** (925) | 46% | 49% |  |  |
| Pershing II Deployed | ussr | **95%** (931) | 59% | 36% |  |  |
| Asia Scoring | neutral | **93%** (8,870) | 18% | 75% | 96% (4,734) | 90% (4,136) |
| Middle East Scoring | neutral | **92%** (8,735) | 15% | 77% | 92% (4,705) | 92% (4,030) |
| Quagmire | ussr | **92%** (2,121) | 62% | 30% |  |  |
| Colonial Rear Guards | us | **92%** (2,523) | 32% | 60% |  |  |
| Liberation Theology | ussr | **92%** (2,328) | 24% | 68% |  |  |
| Europe Scoring | neutral | **91%** (8,834) | 10% | 81% | 91% (4,691) | 91% (4,143) |
| Nasser | ussr | **90%** (2,536) | 75% | 15% |  |  |
| Africa Scoring | neutral | **89%** (5,063) | 6% | 83% | 87% (2,587) | 92% (2,476) |
| Central America Scoring | neutral | **89%** (5,125) | 10% | 80% | 89% (2,660) | 90% (2,465) |
| Southeast Asia Scoring | neutral | **89%** (3,646) | 9% | 80% | 87% (1,784) | 91% (1,862) |
| Korean War | ussr | **89%** (2,957) | 37% | 51% |  |  |
| Captured Nazi Scientist | neutral | **88%** (4,595) | 31% | 57% | 90% (2,044) | 86% (2,551) |
| South America Scoring | neutral | **86%** (5,109) | 7% | 79% | 85% (2,645) | 88% (2,464) |
| Star Wars | us | **86%** (521) | 26% | 60% |  |  |
| ABM Treaty | neutral | **83%** (5,279) | 51% | 32% | 89% (2,804) | 77% (2,475) |
| Marine Barracks Bombing | ussr | **83%** (955) | 19% | 64% |  |  |
| Defectors | us | **82%** (4,456) | 82% | 0% |  |  |
| OAS Founded | us | **82%** (1,752) | 20% | 62% |  |  |
| Grain Sales to Soviets | us | **81%** (2,527) | 81% | 0% |  |  |
| Allende | ussr | **81%** (1,963) | 34% | 48% |  |  |
| Red Scare/Purge | neutral | **79%** (8,970) | 79% | 0% | 70% (4,883) | 90% (4,087) |
| De-Stalinization | ussr | **77%** (2,897) | 13% | 63% |  |  |
| Socialist Governments | ussr | **76%** (3,884) | 51% | 25% |  |  |
| Nixon Plays the China Card | us | **74%** (1,918) | 9% | 64% |  |  |
| Brezhnev Doctrine | ussr | **72%** (1,886) | 71% | 1% |  |  |
| Panama Canal Returned | us | **70%** (1,802) | 10% | 60% |  |  |
| Sadat Expels Soviets | us | **70%** (1,742) | 19% | 51% |  |  |
| Containment | us | **64%** (2,052) | 64% | 0% |  |  |
| Indo-Pakistani War | neutral | **63%** (8,576) | 22% | 41% | 63% (4,543) | 64% (4,033) |
| John Paul II Elected Pope | us | **63%** (2,051) | 18% | 44% |  |  |
| Solidarity | us | **61%** (577) | 3% | 59% |  |  |
| Puppet Governments | us | **61%** (1,929) | 26% | 35% |  |  |
| Missile Envy | neutral | **61%** (5,036) | 60% | 0% | 68% (2,603) | 52% (2,433) |
| “Ask Not What Your Country Can Do For You…” | us | **60%** (2,002) | 23% | 37% |  |  |
| Iran-Iraq War | neutral | **59%** (1,855) | 15% | 44% | 68% (898) | 50% (957) |
| Truman Doctrine | us | **58%** (2,115) | 25% | 33% |  |  |
| Camp David Accords | us | **58%** (2,102) | 9% | 49% |  |  |
| Ussuri River Skirmish | us | **55%** (2,071) | 14% | 41% |  |  |
| Suez Crisis | ussr | **55%** (2,989) | 25% | 29% |  |  |
| Aldrich Ames Remix | ussr | **50%** (969) | 50% | 0% |  |  |
| South African Unrest | ussr | **49%** (2,412) | 16% | 33% |  |  |
| Fidel | ussr | **47%** (3,149) | 12% | 35% |  |  |
| Cultural Revolution | ussr | **46%** (2,084) | 25% | 21% |  |  |
| Portuguese Empire Crumbles | ussr | **44%** (2,081) | 13% | 31% |  |  |
| Soviets Shoot Down KAL-007 | us | **44%** (1,013) | 44% | 0% |  |  |
| Vietnam Revolts | ussr | **43%** (2,777) | 29% | 13% |  |  |
| Blockade | ussr | **37%** (2,896) | 0% | 37% |  |  |
| Alliance for Progress | us | **36%** (2,115) | 12% | 24% |  |  |
| Willy Brandt | ussr | **32%** (2,139) | 9% | 22% |  |  |
| “Lone Gunman” | ussr | **31%** (2,258) | 29% | 2% |  |  |
| SALT Negotiations | neutral | **31%** (4,572) | 22% | 8% | 39% (2,300) | 22% (2,272) |
| Marshall Plan | us | **30%** (2,678) | 30% | 0% |  |  |
| OPEC | ussr | **29%** (2,329) | 15% | 14% |  |  |
| NORAD | us | **26%** (2,934) | 26% | 1% |  |  |
| Latin American Debt Crisis | ussr | **26%** (931) | 6% | 19% |  |  |
| “We Will Bury You” | ussr | **25%** (2,276) | 25% | 0% |  |  |
| East European Unrest | us | **20%** (4,398) | 18% | 2% |  |  |
| CIA Created | us | **19%** (2,617) | 19% | 0% |  |  |
| Glasnost | ussr | **17%** (929) | 15% | 2% |  |  |
| Terrorism | neutral | **17%** (1,909) | 12% | 5% | 15% (948) | 19% (961) |
| Duck and Cover | us | **17%** (4,559) | 17% | 0% |  |  |
| Bear Trap | us | **16%** (2,092) | 16% | 1% |  |  |
| De Gaulle Leads France | ussr | **16%** (3,083) | 5% | 11% |  |  |
| Kitchen Debates | us | **13%** (2,377) | 3% | 9% |  |  |
| Arab-Israeli War | ussr | **12%** (3,483) | 8% | 4% |  |  |
| Che | ussr | **12%** (2,382) | 10% | 2% |  |  |
| US/Japan Mutual Defense Pact | us | **8%** (3,023) | 8% | 0% |  |  |
| The Cambridge Five | ussr | **6%** (3,368) | 5% | 1% |  |  |
| “An Evil Empire” | us | **6%** (915) | 4% | 2% |  |  |
| How I Learned to Stop Worrying | neutral | **6%** (4,941) | 5% | 1% | 5% (2,526) | 6% (2,415) |
| Muslim Revolution | ussr | **5%** (2,322) | 5% | 1% |  |  |
| Comecon | ussr | **5%** (3,150) | 4% | 1% |  |  |
| Special Relationship | us | **5%** (4,541) | 4% | 1% |  |  |
| Romanian Abdication | ussr | **5%** (3,060) | 4% | 1% |  |  |
| Latin American Death Squads | neutral | **4%** (4,964) | 4% | 0% | 3% (2,499) | 4% (2,465) |
| Yuri and Samantha | ussr | **4%** (952) | 2% | 2% |  |  |
| Reagan Bombs Libya | us | **3%** (941) | 1% | 2% |  |  |
| Iranian Hostage Crisis | ussr | **3%** (933) | 2% | 2% |  |  |
| Five Year Plan | us | **3%** (4,413) | 3% | 0% |  |  |
| AWACS Sale to Saudis | us | **3%** (972) | 2% | 1% |  |  |
| Olympic Games | neutral | **3%** (8,685) | 3% | 0% | 4% (4,705) | 1% (3,980) |
| U-2 Incident | ussr | **3%** (2,150) | 1% | 1% |  |  |
| The Iron Lady | us | **3%** (900) | 2% | 1% |  |  |
| Iran-Contra Scandal | ussr | **2%** (935) | 0% | 2% |  |  |
| Flower Power | ussr | **2%** (2,052) | 2% | 0% |  |  |
| Cuban Missile Crisis | neutral | **2%** (4,953) | 2% | 0% | 0% (2,574) | 4% (2,379) |
| Independent Reds | us | **2%** (2,958) | 2% | 0% |  |  |
| Chernobyl | us | **2%** (975) | 0% | 2% |  |  |
| Nuclear Test Ban | neutral | **2%** (8,749) | 1% | 0% | 2% (4,661) | 1% (4,088) |
| Warsaw Pact Formed | ussr | **2%** (3,174) | 0% | 1% |  |  |
| Our Man in Tehran | us | **1%** (2,224) | 1% | 1% |  |  |
| North Sea Oil | us | **1%** (964) | 1% | 0% |  |  |
| Ortega Elected in Nicaragua | ussr | **1%** (996) | 0% | 1% |  |  |
| Summit | neutral | **1%** (5,005) | 1% | 0% | 0% (2,566) | 2% (2,439) |
| Arms Race | neutral | **1%** (5,012) | 0% | 1% | 0% (2,538) | 2% (2,474) |
| “One Small Step…” | neutral | **1%** (4,938) | 0% | 1% | 0% (2,555) | 1% (2,383) |
| Nuclear Subs | us | **1%** (2,105) | 1% | 0% |  |  |
| Shuttle Diplomacy | us | **1%** (2,339) | 1% | 0% |  |  |
| Wargames | neutral | **0%** (1,856) | 0% | 0% | 0% (931) | 0% (925) |
| NATO | us | **0%** (2,204) | 0% | 0% |  |  |
| Formosan Resolution | us | **0%** (3,105) | 0% | 0% |  |  |
