// Tauri packages a self-contained frontend; theme sources stay in the workbench.
import { copyFile } from "node:fs/promises";
for (const name of ["theme.css", "theme.js", "shell.css", "icons.js"]) {
  await copyFile(
    new URL(`../webapp/static/${name}`, import.meta.url),
    new URL(`./ui/${name}`, import.meta.url),
  );
}
