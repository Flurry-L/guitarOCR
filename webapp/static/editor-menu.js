import { el } from "./dom.js";

// Toolbar buttons own the commands. The menu and shortcuts use the same actions
// and disabled state, so saving or a locked score cannot bypass their guards.
export function editorMenu(menu, canvas) {
  const entries = [
    ["undoNote", "撤销", "Ctrl Z"],
    ["redoNote", "重做", "Ctrl ⇧ Z"],
    null,
    ["copyBeat", "复制当前拍", "Ctrl C"],
    ["pasteBeat", "粘贴到当前拍", "Ctrl V"],
    ["duplicateBeat", "复制到下一拍", "Ctrl D"],
    null,
    ["addChordNote", "加入和弦音", ""],
    ["makeRest", "改为休止符", "R"],
    ["addEvent", "插入空拍", "Insert"],
    ["deleteNote", "删除音符", "Delete"],
    ["deleteEvent", "删除当前拍", "⇧ Delete"],
    null,
    ["saveMeasure", "保存并确认小节", "Ctrl S"],
    ["sourceZoom", "查看原谱", ""],
  ];
  const run = (id) => {
    const button = document.getElementById(id);
    if (button && !button.disabled) button.click();
  };
  function close(focus = false) {
    const open = !menu.hidden;
    menu.hidden = true;
    if (open && focus) canvas.focus({ preventScroll: true });
  }
  function open(x, y) {
    menu.replaceChildren(
      ...entries.map((entry) => {
        if (!entry) return el("hr");
        const [id, label, shortcut] = entry;
        const item = el("button");
        item.type = "button";
        item.setAttribute("role", "menuitem");
        item.append(el("span", label), el("kbd", shortcut));
        item.disabled = document.getElementById(id)?.disabled ?? true;
        item.onclick = () => {
          close();
          run(id);
          canvas.focus({ preventScroll: true });
        };
        return item;
      }),
    );
    menu.hidden = false;
    menu.style.left = `${Math.max(8, Math.min(x, innerWidth - menu.offsetWidth - 8))}px`;
    menu.style.top = `${Math.max(8, Math.min(y, innerHeight - menu.offsetHeight - 8))}px`;
    menu.querySelector("button:not(:disabled)")?.focus({ preventScroll: true });
  }
  menu.addEventListener("keydown", (event) => {
    if (event.key === "Escape" || event.key === "Tab") {
      event.preventDefault();
      event.stopPropagation();
      close(true);
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const items = [...menu.querySelectorAll("button:not(:disabled)")];
    const index = items.indexOf(document.activeElement);
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? items.length - 1
          : (index + (event.key === "ArrowDown" ? 1 : -1) + items.length) %
            items.length;
    items[next]?.focus();
  });
  document.addEventListener("pointerdown", (event) => {
    if (!menu.contains(event.target)) close();
  });
  window.addEventListener("resize", () => close());
  window.addEventListener(
    "scroll",
    (event) => {
      if (!menu.contains(event.target)) close();
    },
    true,
  );
  return { open, close, run };
}
