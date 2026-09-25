/**
 * The map and the card list, bundled into the page from rules/ -- the formal spec the engine
 * implements -- so the workbench needs no server to draw a board. (Names in the action log come
 * from the engine itself; these are for layout, tooltips and rules text.)
 */
import mapJson from "../../../rules/map.json";
import cardsJson from "../../../rules/cards.json";
import { CardMetadata, MapMetadata } from "./types";

export const MAP_METADATA = mapJson as unknown as MapMetadata;
export const CARDS_METADATA = cardsJson as unknown as CardMetadata[];
