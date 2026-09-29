import "./icons.js";
// A single preference shared by every surface on this origin.
const key = "guitarocr-theme";
const system = matchMedia("(prefers-color-scheme: dark)");
export function themePreference() {
  try {
    return localStorage.getItem(key) || "system";
  } catch {
    return "system";
  }
}
function apply() {
  const preference = themePreference();
  document.documentElement.dataset.theme =
    preference === "system" ? (system.matches ? "dark" : "light") : preference;
  document.querySelectorAll("[data-theme-picker]").forEach((node) => {
    node.value = preference;
  });
}
export function setTheme(preference) {
  if (!["light", "dark", "system"].includes(preference)) return;
  try {
    localStorage.setItem(key, preference);
  } catch {
    /* Private browser storage. */
  }
  apply();
}
system.addEventListener("change", apply);
window.addEventListener("storage", apply);
document.addEventListener("change", (event) => {
  if (event.target.matches("[data-theme-picker]")) setTheme(event.target.value);
});
apply();
document.addEventListener("DOMContentLoaded", apply);
