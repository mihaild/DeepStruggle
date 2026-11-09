// The English catalog types `t`: keys and namespaces are checked at compile time.
import "i18next";
import type en from "./locales/en";

declare module "i18next" {
  interface CustomTypeOptions {
    defaultNS: "common";
    resources: typeof en;
  }
}
