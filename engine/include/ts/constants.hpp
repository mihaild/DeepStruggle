#pragma once
#include <cstdint>
#include <cstddef>

namespace ts {

constexpr size_t TOTAL_COUNTRIES = 84;
constexpr size_t TOTAL_REGIONS   = 6;
constexpr size_t TOTAL_CARDS     = 110;
constexpr size_t CARD_ARRAY_SIZE = 111; // Index 0 unused, 1..110

// Country Indices constants for fast direct reference
namespace countries {
    constexpr uint8_t CANADA          = 0;
    constexpr uint8_t UNITED_KINGDOM  = 1;
    constexpr uint8_t NORWAY          = 2;
    constexpr uint8_t SWEDEN          = 3;
    constexpr uint8_t DENMARK         = 4;
    constexpr uint8_t FINLAND         = 5;
    constexpr uint8_t BENELUX         = 6;
    constexpr uint8_t WEST_GERMANY    = 7;
    constexpr uint8_t FRANCE          = 8;
    constexpr uint8_t SPAIN_PORTUGAL  = 9;
    constexpr uint8_t ITALY           = 10;
    constexpr uint8_t GREECE          = 11;
    constexpr uint8_t TURKEY          = 12;
    constexpr uint8_t AUSTRIA         = 13;
    constexpr uint8_t EAST_GERMANY    = 14;
    constexpr uint8_t POLAND          = 15;
    constexpr uint8_t CZECHOSLOVAKIA  = 16;
    constexpr uint8_t HUNGARY         = 17;
    constexpr uint8_t YUGOSLAVIA      = 18;
    constexpr uint8_t ROMANIA         = 19;
    constexpr uint8_t BULGARIA        = 20;

    constexpr uint8_t LEBANON         = 21;
    constexpr uint8_t SYRIA           = 22;
    constexpr uint8_t ISRAEL          = 23;
    constexpr uint8_t IRAQ            = 24;
    constexpr uint8_t IRAN            = 25;
    constexpr uint8_t JORDAN          = 26;
    constexpr uint8_t GULF_STATES     = 27;
    constexpr uint8_t SAUDI_ARABIA    = 28;
    constexpr uint8_t EGYPT           = 29;
    constexpr uint8_t LIBYA           = 30;

    constexpr uint8_t AFGHANISTAN     = 31;
    constexpr uint8_t PAKISTAN        = 32;
    constexpr uint8_t INDIA           = 33;
    constexpr uint8_t BURMA           = 34;
    constexpr uint8_t LAOS_CAMBODIA   = 35;
    constexpr uint8_t THAILAND        = 36;
    constexpr uint8_t VIETNAM         = 37;
    constexpr uint8_t MALAYSIA        = 38;
    constexpr uint8_t INDONESIA       = 39;
    constexpr uint8_t PHILIPPINES     = 40;
    constexpr uint8_t AUSTRALIA       = 41;
    constexpr uint8_t TAIWAN          = 42;
    constexpr uint8_t NORTH_KOREA     = 43;
    constexpr uint8_t SOUTH_KOREA     = 44;
    constexpr uint8_t JAPAN           = 45;

    constexpr uint8_t MOROCCO         = 46;
    constexpr uint8_t ALGERIA         = 47;
    constexpr uint8_t TUNISIA         = 48;
    constexpr uint8_t WEST_AFRICA     = 49;
    constexpr uint8_t SAHARAN_STATES  = 50;
    constexpr uint8_t SUDAN           = 51;
    constexpr uint8_t IVORY_COAST     = 52;
    constexpr uint8_t NIGERIA         = 53;
    constexpr uint8_t ETHIOPIA        = 54;
    constexpr uint8_t SOMALIA         = 55;
    constexpr uint8_t CAMEROON        = 56;
    constexpr uint8_t ZAIRE           = 57;
    constexpr uint8_t KENYA           = 58;
    constexpr uint8_t ANGOLA          = 59;
    constexpr uint8_t ZIMBABWE        = 60;
    constexpr uint8_t SE_AFRICAN_STS  = 61;
    constexpr uint8_t BOTSWANA        = 62;
    constexpr uint8_t SOUTH_AFRICA    = 63;

    constexpr uint8_t MEXICO          = 64;
    constexpr uint8_t GUATEMALA       = 65;
    constexpr uint8_t EL_SALVADOR     = 66;
    constexpr uint8_t HONDURAS        = 67;
    constexpr uint8_t COSTA_RICA      = 68;
    constexpr uint8_t NICARAGUA       = 69;
    constexpr uint8_t PANAMA          = 70;
    constexpr uint8_t CUBA            = 71;
    constexpr uint8_t HAITI           = 72;
    constexpr uint8_t DOMINICAN_REP   = 73;

    constexpr uint8_t COLOMBIA        = 74;
    constexpr uint8_t ECUADOR         = 75;
    constexpr uint8_t PERU            = 76;
    constexpr uint8_t VENEZUELA       = 77;
    constexpr uint8_t BOLIVIA         = 78;
    constexpr uint8_t BRAZIL          = 79;
    constexpr uint8_t PARAGUAY        = 80;
    constexpr uint8_t CHILE           = 81;
    constexpr uint8_t ARGENTINA       = 82;
    constexpr uint8_t URUGUAY         = 83;

    constexpr uint8_t SUPERPOWER_USA  = 253;
    constexpr uint8_t SUPERPOWER_USSR = 254;
    constexpr uint8_t INVALID_COUNTRY = 255;
}

// All 110 Card IDs
namespace card_ids {
    constexpr uint8_t ASIA_SCORING                        = 1;
    constexpr uint8_t EUROPE_SCORING                      = 2;
    constexpr uint8_t MIDDLE_EAST_SCORING                 = 3;
    constexpr uint8_t DUCK_AND_COVER                      = 4;
    constexpr uint8_t FIVE_YEAR_PLAN                      = 5;
    constexpr uint8_t THE_CHINA_CARD                      = 6;
    constexpr uint8_t SOCIALIST_GOVERNMENTS               = 7;
    constexpr uint8_t FIDEL                               = 8;
    constexpr uint8_t VIETNAM_REVOLTS                     = 9;
    constexpr uint8_t BLOCKADE                            = 10;
    constexpr uint8_t KOREAN_WAR                          = 11;
    constexpr uint8_t ROMANIAN_ABDICATION                 = 12;
    constexpr uint8_t ARAB_ISRAELI_WAR                    = 13;
    constexpr uint8_t COMECON                             = 14;
    constexpr uint8_t NASSER                              = 15;
    constexpr uint8_t WARSAW_PACT                         = 16;
    constexpr uint8_t WARSAW_PACT_FORMED                  = 16;
    constexpr uint8_t DE_GAULLE                           = 17;
    constexpr uint8_t DE_GAULLE_LEADS_FRANCE              = 17;
    constexpr uint8_t CAPTURED_NAZI_SCIENTIST             = 18;
    constexpr uint8_t TRUMAN_DOCTRINE                     = 19;
    constexpr uint8_t OLYMPIC_GAMES                       = 20;
    constexpr uint8_t NATO                                = 21;
    constexpr uint8_t INDEPENDENT_REDS                    = 22;
    constexpr uint8_t MARSHALL_PLAN                       = 23;
    constexpr uint8_t INDO_PAKISTANI_WAR                  = 24;
    constexpr uint8_t CONTAINMENT                         = 25;
    constexpr uint8_t CIA_CREATED                         = 26;
    constexpr uint8_t US_JAPAN_PACT                       = 27;
    constexpr uint8_t SUEZ_CRISIS                         = 28;
    constexpr uint8_t EAST_EUROPEAN_UNREST                = 29;
    constexpr uint8_t DECOLONIZATION                      = 30;
    constexpr uint8_t RED_SCARE_PURGE                     = 31;
    constexpr uint8_t UN_INTERVENTION                     = 32;
    constexpr uint8_t DE_STALINIZATION                    = 33;
    constexpr uint8_t NUCLEAR_TEST_BAN                    = 34;
    constexpr uint8_t FORMOSAN_RESOLUTION                 = 35;
    constexpr uint8_t BRUSH_WAR                           = 36;
    constexpr uint8_t CENTRAL_AMERICA_SCORING             = 37;
    constexpr uint8_t SE_ASIA_SCORING                     = 38;
    constexpr uint8_t ARMS_RACE                            = 39;
    constexpr uint8_t CUBAN_MISSILE_CRISIS                = 40;
    constexpr uint8_t NUCLEAR_SUBS                        = 41;
    constexpr uint8_t QUAGMIRE                            = 42;
    constexpr uint8_t SALT_NEGOTIATIONS                   = 43;
    constexpr uint8_t BEAR_TRAP                           = 44;
    constexpr uint8_t SUMMIT                              = 45;
    constexpr uint8_t HOW_I_LEARNED_TO_STOP_WORRYING      = 46;
    constexpr uint8_t JUNTA                               = 47;
    constexpr uint8_t KITCHEN_DEBATES                     = 48;
    constexpr uint8_t MISSILE_ENVY                        = 49;
    constexpr uint8_t WE_WILL_BURY_YOU                    = 50;
    constexpr uint8_t BREZHNEV_DOCTRINE                   = 51;
    constexpr uint8_t PORTUGUESE_EMPIRE_CRUMBLES          = 52;
    constexpr uint8_t SOUTH_AFRICAN_UNREST                = 53;
    constexpr uint8_t ALLENDE                             = 54;
    constexpr uint8_t WILLY_BRANDT                        = 55;
    constexpr uint8_t MUSLIM_REVOLUTION                   = 56;
    constexpr uint8_t ABM_TREATY                          = 57;
    constexpr uint8_t CULTURAL_REVOLUTION                 = 58;
    constexpr uint8_t FLOWER_POWER                        = 59;
    constexpr uint8_t U2_INCIDENT                         = 60;
    constexpr uint8_t OPEC                                = 61;
    constexpr uint8_t LONE_GUNMAN                         = 62;
    constexpr uint8_t COLONIAL_REAR_GUARDS                = 63;
    constexpr uint8_t PANAMA_CANAL_RETURNED               = 64;
    constexpr uint8_t CAMP_DAVID_ACCORDS                  = 65;
    constexpr uint8_t PUPPET_GOVERNMENTS                  = 66;
    constexpr uint8_t GRAIN_SALES                         = 67;
    constexpr uint8_t GRAIN_SALES_TO_SOVIETS              = 67;
    constexpr uint8_t JOHN_PAUL_II                        = 68;
    constexpr uint8_t JOHN_PAUL_II_ELECTED_POPE           = 68;
    constexpr uint8_t LATIN_DEATH_SQUADS                  = 69;
    constexpr uint8_t LATIN_AMERICAN_DEATH_SQUADS         = 69;
    constexpr uint8_t OAS_FOUNDED                         = 70;
    constexpr uint8_t NIXON_PLAYS_THE_CHINA_CARD          = 71;
    constexpr uint8_t SADAT_EXPELS_SOVIETS                = 72;
    constexpr uint8_t SHUTTLE_DIPLOMACY                   = 73;
    constexpr uint8_t THE_VOICE_OF_AMERICA                = 74;
    constexpr uint8_t LIBERATION_THEOLOGY                 = 75;
    constexpr uint8_t USSURI_RIVER_SKIRMISH               = 76;
    constexpr uint8_t ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU = 77;
    constexpr uint8_t ALLIANCE_FOR_PROGRESS               = 78;
    constexpr uint8_t AFRICA_SCORING                      = 79;
    constexpr uint8_t ONE_SMALL_STEP                      = 80;
    constexpr uint8_t SOUTH_AMERICA_SCORING               = 81;
    constexpr uint8_t IRANIAN_HOSTAGE_CRISIS              = 82;
    constexpr uint8_t THE_IRON_LADY                       = 83;
    constexpr uint8_t REAGAN_BOMBS_LIBYA                  = 84;
    constexpr uint8_t STAR_WARS                           = 85;
    constexpr uint8_t NORTH_SEA_OIL                       = 86;
    constexpr uint8_t THE_REFORMER                        = 87;
    constexpr uint8_t MARINE_BARRACKS_BOMBING             = 88;
    constexpr uint8_t SOVIETS_SHOOT_DOWN_KAL_007          = 89;
    constexpr uint8_t GLASNOST                            = 90;
    constexpr uint8_t ORTEGA_ELECTED_IN_NICARAGUA         = 91;
    constexpr uint8_t TERRORISM                           = 92;
    constexpr uint8_t IRAN_CONTRA                         = 93;
    constexpr uint8_t IRAN_CONTRA_SCANDAL                 = 93;
    constexpr uint8_t CHERNOBYL                           = 94;
    constexpr uint8_t LATIN_AMERICAN_DEBT_CRISIS          = 95;
    constexpr uint8_t TEAR_DOWN_THIS_WALL                 = 96;
    constexpr uint8_t AN_EVIL_EMPIRE                      = 97;
    constexpr uint8_t ALDRICH_AMES                        = 98;
    constexpr uint8_t ALDRICH_AMES_REMIX                  = 98;
    constexpr uint8_t PERSHING_II_DEPLOYED                = 99;
    constexpr uint8_t WARGAMES                            = 100;
    constexpr uint8_t SOLIDARITY                          = 101;
    constexpr uint8_t IRAN_IRAQ_WAR                       = 102;
    constexpr uint8_t DEFECTORS                           = 103;
    constexpr uint8_t THE_CAMBRIDGE_FIVE                  = 104;
    constexpr uint8_t SPECIAL_RELATIONSHIP                = 105;
    constexpr uint8_t NORAD                               = 106;
    constexpr uint8_t CHE                                 = 107;
    constexpr uint8_t OUR_MAN_IN_TEHRAN                   = 108;
    constexpr uint8_t YURI_AND_SAMANTHA                   = 109;
    constexpr uint8_t AWACS_SALE                          = 110;
    constexpr uint8_t AWACS_SALE_TO_SAUDIS                = 110;
    constexpr uint8_t SPACE_WALK_DISCARD                  = 250;
}

// Cards whose event may legitimately find nothing to do.
//
// Most events are mandatory: the board always affords a legal choice, and a decision that
// offers none is a defect worth hearing about immediately, so the mask reports the whole
// position (ts::report_anomaly) before letting the player decline. That report is only useful
// while it stays rare, and a handful of cards can genuinely come up empty -- Truman Doctrine
// wants an uncontrolled European country holding USSR Influence, and some boards have none;
// Muslim Revolution removes from two Middle Eastern countries and may find one, or none.
// Those fizzle quietly.
//
// The list is deliberately short and explicit rather than inferred. Anything not on it that
// runs out of targets is reported, which is how we learn about the next one instead of
// discovering it as a hung training run.
// Junta and Tear Down This Wall grant free Ops confined to a region -- Central/South America
// and Europe -- and usable only for a coup or a realignment. That confinement belongs to the
// Ops their event granted, not to the card: the same card's own Ops, and its Ops borrowed by UN
// Intervention (which uses the value without triggering the event at all), are ordinary Ops
// and go anywhere. Asking only "is this that card?" conflated the two, and at turn 9 AR1 of
// ts-replayer game 141 the USSR named Tear Down This Wall through UN Intervention to coup
// Libya -- which the engine confined to Europe, where DEFCON 3 forbids couping, leaving no
// legal operation at all.
namespace free_action {
    inline constexpr bool region_locked(uint8_t card_id, uint8_t event_granted_ops) noexcept {
        return event_granted_ops != 0 &&
               (card_id == card_ids::JUNTA || card_id == card_ids::TEAR_DOWN_THIS_WALL);
    }
}

namespace may_fizzle {
    inline constexpr bool allowed(uint8_t card_id) noexcept {
        return card_id == card_ids::TRUMAN_DOCTRINE ||
               card_id == card_ids::MUSLIM_REVOLUTION;
    }
}

// Persistent Effects Bitfield Bit Allocations
namespace effect_bits {
    constexpr uint64_t NATO_ACTIVE                 = 1ULL << 0;  // NATO in effect
    constexpr uint64_t NATO_CANCELED_FRANCE        = 1ULL << 1;  // De Gaulle canceled NATO for France
    constexpr uint64_t NATO_CANCELED_WEST_GERMANY  = 1ULL << 2;  // Willy Brandt canceled NATO for W.Germany
    constexpr uint64_t MARSHALL_PLAN_PLAYED        = 1ULL << 3;  // Enables NATO
    constexpr uint64_t WARSAW_PACT_PLAYED          = 1ULL << 4;  // Enables NATO
    constexpr uint64_t US_JAPAN_PACT_ACTIVE        = 1ULL << 5;  // USSR cannot coup/realign Japan
    constexpr uint64_t CONTAINMENT_ACTIVE          = 1ULL << 6;  // US +1 Ops this turn
    constexpr uint64_t PURGE_US_ACTIVE             = 1ULL << 7;  // US -1 Ops this turn
    constexpr uint64_t PURGE_USSR_ACTIVE           = 1ULL << 8;  // USSR -1 Ops this turn
    constexpr uint64_t VIETNAM_REVOLTS_ACTIVE      = 1ULL << 9;  // USSR +1 Ops in SE Asia this turn
    constexpr uint64_t FORMOSAN_RESOLUTION_ACTIVE  = 1ULL << 10; // Taiwan is BG until US plays China Card
    constexpr uint64_t CMC_ACTIVE_US               = 1ULL << 11; // CMC played by US (USSR coups cause loss)
    constexpr uint64_t CMC_ACTIVE_USSR             = 1ULL << 12; // CMC played by USSR (US coups cause loss)
    constexpr uint64_t NUCLEAR_SUBS_ACTIVE         = 1ULL << 13; // US BG coups don't degrade DEFCON this turn
    constexpr uint64_t QUAGMIRE_ACTIVE             = 1ULL << 14; // US trapped in Quagmire
    constexpr uint64_t BEAR_TRAP_ACTIVE            = 1ULL << 15; // USSR trapped in Bear Trap
    constexpr uint64_t SALT_ACTIVE                 = 1ULL << 16; // -1 to all coup rolls this turn
    // USSR +3 VP unless the US plays UN Intervention as its Event on their next action round.
    // Deliberately absent from TURN_CLEANUP_MASK: "next action round" is the next one there is,
    // and the card is usually played late in a turn, so the debt is most often collected in the
    // turn after. Clearing it at the turn boundary cancelled it outright -- at turn 4 AR7 of
    // ts-replayer game 131 the US triggers it and the 3 VP the USSR collects at turn 5 AR1
    // never moved. A game that ends with the debt outstanding simply never pays it: the flag
    // is read only when the US picks a card in an action round, and final scoring does not
    // look at it.
    constexpr uint64_t WE_WILL_BURY_YOU_PENDING    = 1ULL << 17;
    constexpr uint64_t BREZHNEV_DOCTRINE_ACTIVE    = 1ULL << 18; // USSR +1 Ops this turn
    constexpr uint64_t FLOWER_POWER_ACTIVE         = 1ULL << 19; // USSR +2 VP per US war card
    constexpr uint64_t U2_INCIDENT_ACTIVE          = 1ULL << 20; // USSR +1 VP if UN Intervention played this turn
    constexpr uint64_t SHUTTLE_DIPLOMACY_ACTIVE    = 1ULL << 21; // -1 USSR BG on next ME/Asia scoring
    constexpr uint64_t DEATH_SQUADS_US             = 1ULL << 22; // US +1, USSR -1 CA/SA coups this turn
    constexpr uint64_t DEATH_SQUADS_USSR           = 1ULL << 23; // USSR +1, US -1 CA/SA coups this turn
    constexpr uint64_t CAMP_DAVID_PLAYED           = 1ULL << 24; // Blocks Arab-Israeli War
    constexpr uint64_t IRON_LADY_PLAYED            = 1ULL << 25; // Blocks Socialist Governments
    constexpr uint64_t NORTH_SEA_OIL_PLAYED        = 1ULL << 26; // Blocks OPEC
    constexpr uint64_t NORTH_SEA_OIL_ACTIVE        = 1ULL << 27; // US gets 8th AR this turn
    constexpr uint64_t THE_REFORMER_PLAYED         = 1ULL << 28; // USSR no coups in Europe; enables Glasnost Ops
    constexpr uint64_t IRAN_CONTRA_ACTIVE          = 1ULL << 29; // US -1 Realignment rolls this turn
    constexpr uint64_t EVIL_EMPIRE_PLAYED          = 1ULL << 30; // Cancels Flower Power
    constexpr uint64_t ALDRICH_AMES_ACTIVE         = 1ULL << 31; // US hand revealed this turn
    constexpr uint64_t JOHN_PAUL_II_PLAYED         = 1ULL << 32; // Enables Solidarity
    constexpr uint64_t NORAD_ACTIVE                = 1ULL << 33; // US adds 1 inf if DEFCON dropped to 2 in AR
    constexpr uint64_t YURI_AND_SAMANTHA_ACTIVE    = 1ULL << 34; // USSR +1 VP per US coup this turn
    constexpr uint64_t AWACS_PLAYED                = 1ULL << 35; // Blocks Muslim Revolution
    constexpr uint64_t IRANIAN_HOSTAGE_CRISIS_PLAY = 1ULL << 36; // US discards 2 cards on Terrorism
    constexpr uint64_t WILLY_BRANDT_PLAYED         = 1ULL << 37; // Willy Brandt in effect
    constexpr uint64_t TEAR_DOWN_THIS_WALL_PLAYED  = 1ULL << 38; // Cancels Willy Brandt
    constexpr uint64_t CHERNOBYL_ACTIVE            = 1ULL << 39; // USSR cannot place Ops in region

    // Bits 40..42 encode Chernobyl forbidden region index (0..5)
    constexpr uint64_t CHERNOBYL_REGION_SHIFT      = 40;
    constexpr uint64_t CHERNOBYL_REGION_MASK       = 0x7ULL << CHERNOBYL_REGION_SHIFT;

    // Space Race Attempt Flags (Bits 43..46)
    constexpr uint64_t SPACE_US_ATTEMPT_1          = 1ULL << 43; // US 1st space race attempt used this turn
    constexpr uint64_t SPACE_US_ATTEMPT_2          = 1ULL << 44; // US 2nd space race attempt used this turn
    constexpr uint64_t SPACE_USSR_ATTEMPT_1        = 1ULL << 45; // USSR 1st space race attempt used this turn
    constexpr uint64_t SPACE_USSR_ATTEMPT_2        = 1ULL << 46; // USSR 2nd space race attempt used this turn
    constexpr uint64_t DEFCON_SUICIDE_PROVOKED     = 1ULL << 47; // DEFCON degraded to 1 by opponent action (event trap)
    constexpr uint64_t CMC_SUICIDE_LOSS            = 1ULL << 48; // Lost by couping under Cuban Missile Crisis without the influence to cancel it

    // Mask for flags that are cleared automatically at end of turn
    constexpr uint64_t TURN_CLEANUP_MASK           = CONTAINMENT_ACTIVE |
                                                     PURGE_US_ACTIVE |
                                                     PURGE_USSR_ACTIVE |
                                                     VIETNAM_REVOLTS_ACTIVE |
                                                     CMC_ACTIVE_US |
                                                     CMC_ACTIVE_USSR |
                                                     NUCLEAR_SUBS_ACTIVE |
                                                     SALT_ACTIVE |
                                                     BREZHNEV_DOCTRINE_ACTIVE |
                                                     U2_INCIDENT_ACTIVE |
                                                     DEATH_SQUADS_US |
                                                     DEATH_SQUADS_USSR |
                                                     NORTH_SEA_OIL_ACTIVE |
                                                     IRAN_CONTRA_ACTIVE |
                                                     ALDRICH_AMES_ACTIVE |
                                                     YURI_AND_SAMANTHA_ACTIVE |
                                                     CHERNOBYL_ACTIVE |
                                                     CHERNOBYL_REGION_MASK |
                                                     SPACE_US_ATTEMPT_1 |
                                                     SPACE_US_ATTEMPT_2 |
                                                     SPACE_USSR_ATTEMPT_1 |
                                                     SPACE_USSR_ATTEMPT_2;
}

// Special flags for MicroAction
namespace action_flags {
    constexpr uint8_t CONFIRM_DONE = 0x80; // Bit 7 set indicates early stop / pass sub-decision
}

} // namespace ts
