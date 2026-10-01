import { ui } from "./state.js";
import { $, el } from "./dom.js";
export function initPages(loadPage) {
  function renderPages() {
    const list = $("pageList");
    list.replaceChildren();
    $("pageSelect").replaceChildren();
    $("pageListTitle").textContent = `全部 ${ui.state.pages.length} 页`;
    ui.state.pages.forEach((p, i) => {
      const b = el(
        "button",
        undefined,
        "page-thumb" + (i === ui.pageIndex ? " active" : ""),
      );
      const option = el("option", `第 ${i + 1} / ${ui.state.pages.length} 页`);
      option.value = i;
      $("pageSelect").append(option);
      const img = el("img");
      img.src = p.url;
      img.loading = "lazy";
      img.alt = `第 ${i + 1} 页`;
      b.append(img, el("span", `第 ${i + 1} 页`));
      b.onclick = () => selectPage(i);
      list.append(b);
    });
    updatePageNavigation();
  }
  function updatePageNavigation() {
    $("prevPage").disabled = ui.busy || !ui.state?.pages || ui.pageIndex <= 0;
    $("nextPage").disabled =
      ui.busy || !ui.state?.pages || ui.pageIndex >= ui.state.pages.length - 1;
    $("pageSelect").disabled = ui.busy || !ui.state?.pages;
    $("pageSelect").value = ui.pageIndex;
  }
  function selectPage(index) {
    if (ui.busy || !ui.state?.pages) return;
    ui.pageIndex = Math.max(0, Math.min(ui.state.pages.length - 1, index));
    ui.selected = -1;
    $("canvasScroll").scrollTop = 0;
    renderPages();
    loadPage();
    $("pageList").children[ui.pageIndex]?.scrollIntoView({ block: "nearest" });
  }
  $("prevPage").onclick = () => selectPage(ui.pageIndex - 1);
  $("nextPage").onclick = () => selectPage(ui.pageIndex + 1);
  $("pageSelect").onchange = () => selectPage(+$("pageSelect").value);

  return { renderPages, updatePageNavigation };
}
