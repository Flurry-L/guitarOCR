// Tauri packages a self-contained frontend; theme sources stay in the workbench.
import { copyFile } from "node:fs/promises";
for (const name of ["theme.css", "theme.js", "icons.js"]) {
  await copyFile(
    new URL(`../ui/workbench/${name}`, import.meta.url),
    new URL(`../ui/launcher/${name}`, import.meta.url),
  );
}
