import { ui } from "./state.js";
export const $ = (id) => document.getElementById(id);
export const el = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
export function notice(message, error = false) {
  $("notice").hidden = !message;
  $("notice").textContent = message;
  $("notice").className = error ? "error" : ui.busy ? "busy" : "";
}
export function action(fn) {
  return async (...args) => {
    const sid = ui.sid, generation = ui.openGeneration;
    try {
      await fn(...args);
    } catch (e) {
      if (ui.sid === sid && ui.openGeneration === generation) notice(e.message, true);
    }
  };
}
