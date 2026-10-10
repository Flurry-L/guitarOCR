import { $, el, action, notice } from "./dom.js";
import { ui, endpoint, receiveProject } from "./state.js";
import { api } from "./api.js";
const chordKey = name => String(name || "").replaceAll("♭", "b").replaceAll("♯", "#").replace(/\s/g, "");

export function chordFromEffects(effects = []) {
  const symbol = effects.find((e) => e.startsWith("chord:"));
  let name = symbol?.slice(6) || "";
  try {
    name = decodeURIComponent(name);
  } catch {
    /* Preserve malformed text for editing. */
  }
  const encoded = effects.find((e) => e.startsWith("diagram:"));
  if (!encoded) return { name, diagram: null };
  const [, base, frets, fingers, barres] = encoded.split(":");
  if (!base || !frets || !fingers || !barres) return { name, diagram: null };
  return {
    name,
    diagram: {
      base_fret: +base,
      frets: frets.split("/").map((f) => (f === "x" ? "x" : +f)),
      fingers: fingers.split("/").map((f) => (f === "-" ? null : +f)),
      barres:
        barres === "-"
          ? []
          : barres.split(";").map((b) => b.split("/").map(Number)),
    },
  };
}

export function unresolvedChordChoices(record, voice, event) {
  const {name, diagram} = chordFromEffects(event?.effects);
  if (!name || !event || (record?.reviewed && !record?.needs_review)) return [];
  return (record?.chord_diagram_choices || []).filter(choice =>
    (!diagram || choice.source === "chord_evidence") &&
    choice.voice === voice && choice.start === event.start && chordKey(choice.name) === chordKey(name));
}

export function hasPendingChordCandidate(annotation) {
  if (["confirmed", "rejected"].includes(chordReviewDecision(annotation))) return false;
  const evidence = annotation?.chord_evidence;
  const candidate = [annotation?.parsed?.diagram,
    ...["diagram", "proposed_diagram", "secondary_diagram", "original_diagram", "rejected_diagram_candidate"].map(key => evidence?.[key])]
    .some(diagram => diagram !== null && typeof diagram === "object" && !Array.isArray(diagram));
  const isChord = ["chord", "chord_diagram"].includes(annotation?.parsed?.kind);
  return !!(evidence?.review_required || evidence?.name_needs_review || evidence?.accepted?.full === false
    || evidence?.conflict_fields?.length || (candidate && annotation?.parsed?.diagram == null))
    && (candidate || (isChord && !!evidence?.name_needs_review));
}

export function chordReviewDecision(annotation) {
  return annotation?.chord_review?.source === "manual" ? annotation.chord_review.status : undefined;
}

export function chordReviewReason(annotation) {
  const evidence = annotation?.chord_evidence;
  if (evidence?.name_needs_review) return "和弦名称尚未识别清楚，请对照原谱修改名称。";
  return evidence?.conflict_fields?.length ? "识别结果有冲突，请对照原谱选择。" : "指法尚待确认，请对照原谱选择。";
}

export function chordLibraryEntries(state, partId = "part-1") {
  const sources = state.metadata?.parts?.length
    ? state.metadata.parts.filter(part => part.id === partId) : [state.metadata];
  const values = new Map();
  for (const source of sources)
    for (const annotation of source?.document_metadata?.score_annotations || []) {
      if (chordReviewDecision(annotation) === "rejected") continue;
      const parsed = annotation.parsed;
      const id = JSON.stringify([parsed?.text, parsed?.diagram || null]);
      if (["chord", "chord_diagram"].includes(parsed?.kind) && parsed.text && !values.has(id))
        values.set(id, parsed);
      const evidence = annotation.chord_evidence;
      if (parsed?.text && evidence?.name_needs_review && chordReviewDecision(annotation) !== "confirmed") {
        const previous = values.get(id);
        values.set(id, {...parsed, ...previous, review: true,
          labels: [...new Set([...(previous?.labels || []), "名称待确认"])],
          reason: chordReviewReason(annotation)});
      }
      if (parsed?.text && hasPendingChordCandidate(annotation)) {
        for (const [key, label] of [["original_diagram", "原始候选"], ["rejected_diagram_candidate", "待确认的原始候选"], ["diagram", "图示模型候选"], ["proposed_diagram", "图示模型候选"], ["secondary_diagram", "另一图示模型候选"]]) {
          const diagram = evidence[key];
          if (!diagram?.frets?.length) continue;
          const candidateId = JSON.stringify([parsed.text, diagram]);
          const previous = values.get(candidateId);
          values.set(candidateId, {text: parsed.text, diagram, review: true,
            labels: [...new Set([...(previous?.labels || []), label])],
            reason: evidence.name_needs_review || !previous?.reason ? chordReviewReason(annotation) : previous.reason});
        }
      }
    }
  const illustrated = new Set([...values.values()].filter(value => value.diagram).map(value => chordKey(value.text)));
  return [...values.values()].filter(value => value.diagram || !illustrated.has(chordKey(value.text)));
}

function effectsFor(name, d) {
  // Parentheses delimit music fields; match Python's quote(..., safe="").
  const encoded = encodeURIComponent(name).replace(/[!'()*]/g, (c) =>
    "%" + c.charCodeAt(0).toString(16).toUpperCase(),
  );
  const effects = name ? ["chord:" + encoded] : [];
  if (d)
    effects.push(
      `diagram:${d.base_fret}:${d.frets.join("/")}:${d.fingers.map((f) => f ?? "-").join("/")}:${d.barres.map((b) => b.join("/")).join(";") || "-"}`,
    );
  return effects;
}

export function diagramSvg(name, d) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 150 180");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${name} 和弦指法`);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
  };
  add(
    "text",
    {
      x: 75,
      y: 20,
      "text-anchor": "middle",
      fill: "currentColor",
      "font-size": 16,
    },
    name,
  );
  if (!d) return svg;
  const count = d.frets.length,
    gap = 100 / Math.max(1, count - 1),
    top = 46;
  const rows = Math.max(
    5,
    ...d.frets.filter(Number.isInteger).map((f) => f - d.base_fret + 1),
  );
  const dy = 100 / rows;
  const line = (x1, y1, x2, y2, width = 1) =>
    add("line", {
      x1,
      y1,
      x2,
      y2,
      stroke: "currentColor",
      "stroke-width": width,
      "stroke-linecap": "round",
    });
  for (let i = 0; i < count; i++)
    line(28 + i * gap, top, 28 + i * gap, top + 100);
  for (let f = 0; f <= rows; f++)
    line(
      28,
      top + f * dy,
      128,
      top + f * dy,
      f === 0 && d.base_fret === 1 ? 3 : 1,
    );
  if (d.base_fret > 1)
    add(
      "text",
      {
        x: 23,
        y: top + dy * 0.8,
        "text-anchor": "end",
        fill: "currentColor",
        "font-size": 11,
      },
      d.base_fret,
    );
  for (const [f, low, high] of d.barres || [])
    line(
      28 + low * gap,
      top + (f - d.base_fret + 0.5) * dy,
      28 + high * gap,
      top + (f - d.base_fret + 0.5) * dy,
      7,
    );
  d.frets.forEach((f, i) => {
    const x = 28 + i * gap;
    if (f === "x" || f === 0)
      add(
        "text",
        {
          x,
          y: top - 7,
          "text-anchor": "middle",
          fill: "currentColor",
          "font-size": 12,
        },
        f === "x" ? "×" : "○",
      );
    else
      add("circle", {
        cx: x,
        cy: top + (f - d.base_fret + 0.5) * dy,
        r: 4,
        fill: "currentColor",
      });
    if (d.fingers?.[i] != null)
      add(
        "text",
        {
          x,
          y: 163,
          "text-anchor": "middle",
          fill: "currentColor",
          "font-size": 11,
        },
        d.fingers[i],
      );
  });
  return svg;
}

export function readChordFields(prefix = "chord") {
  const field = name => $(prefix + name);
  const name = field("Name").value.trim();
  const fretsText = field("Frets").value.trim();
  let d = null;
  if (fretsText) {
    const frets = fretsText
      .toLowerCase()
      .split(/[\s,]+/)
      .map((f) => (f === "x" ? "x" : Number(f)));
    const base = Number(field("Base").value || 1);
    if (
      frets.length < 3 ||
      frets.length > 12 ||
      frets.some(
        (f) => f !== "x" && (!Number.isInteger(f) || f < 0 || f > 36),
      ) ||
      !Number.isInteger(base) ||
      base < 1 ||
      base > 36
    )
      throw new Error(
        "指法请按低音弦到高音弦填写 3–12 个品位，使用空格分隔，x 表示不弹。",
      );
    const fingers = field("Fingers").value.trim()
      ? field("Fingers")
          .value.trim()
          .split(/[\s,]+/)
          .map((f) => (f === "-" ? null : Number(f)))
      : frets.map(() => null);
    if (
      fingers.length !== frets.length ||
      fingers.some(
        (f) => f !== null && (!Number.isInteger(f) || f < 0 || f > 4),
      )
    )
      throw new Error(
        "指法数字每弦一个，0 为拇指，1–4 为手指，- 为无标注。",
      );
    const barres = field("Barres")
      .value.trim()
      .split(/[\s,;]+/)
      .filter(Boolean)
      .map((text) => {
        const match = /^(\d+):(\d+)-(\d+)$/.exec(text);
        if (!match)
          throw new Error(
            "横按格式为 品位:弦号-弦号，例如 1:6-1；多个横按用空格分隔。",
          );
        const [, fret, first, last] = match.map(Number);
        const low = frets.length - Math.max(first, last),
          high = frets.length - Math.min(first, last);
        if (
          fret < base ||
          fret > 36 ||
          low < 0 ||
          high >= frets.length ||
          low >= high ||
          frets[low] !== fret ||
          frets[high] !== fret
        )
          throw new Error("横按两端须与所填弦的品位一致；1 弦为最高音弦。");
        return [fret, low, high];
      });
    if (frets.some((f) => Number.isInteger(f) && f > 0 && f < base))
      throw new Error("按弦品位不能低于图示起始品。");
    d = { base_fret: base, frets, fingers, barres };
  }
  return {name, diagram: d};
}

export function writeChordFields({name, diagram}, prefix = "chord") {
  const field = name => $(prefix + name);
  field("Name").value = name;
  field("Frets").value = diagram?.frets.join(" ") || "";
  field("Base").value = diagram?.base_fret || 1;
  field("Fingers").value =
    diagram?.fingers?.map((f) => f ?? "-").join(" ") || "";
  field("Barres").value =
    diagram?.barres?.map(
        ([f, low, high]) =>
          `${f}:${diagram.frets.length - low}-${diagram.frets.length - high}`,
      )
      .join(" ") || "";
  field("Preview").replaceChildren(
    ...(diagram ? [diagramSvg(name, diagram)] : []),
  );
}

export function chordEditor(change, event) {
  const assign = (name, diagram) =>
    change(() => {
      if (!event()) return;
      event().effects = [
        ...(event().effects || []).filter(
          (e) => !e.startsWith("chord:") && !e.startsWith("diagram:"),
        ),
        ...effectsFor(name, diagram),
      ];
    });
  $("applyChord").onclick = () =>
    change(() => {
      const {name, diagram: d} = readChordFields();
      event().effects = [
        ...(event().effects || []).filter(
          (e) => !e.startsWith("chord:") && !e.startsWith("diagram:"),
        ),
        ...effectsFor(name, d),
      ];
    });
  $("removeChord").onclick = () => assign("", null);
  return {
    render(e) {
      const { name, diagram } = chordFromEffects(e?.effects);
      writeChordFields({name, diagram});
      $("applyChord").disabled = $("removeChord").disabled = !e;
    },
    library(state, partId = "part-1") {
      const entries = chordLibraryEntries(state, partId);
      const library = $("chordLibrary");
      library.hidden = !entries.length;
      library.replaceChildren(
        ...entries.map((p) => {
          const button = el(
            "button",
            undefined,
            p.diagram ? "chord-card" : "chord-symbol",
          );
          button.title = `将 ${p.text || "指法"} 放入当前拍`;
          if (p.diagram) button.append(diagramSvg(p.text, p.diagram));
          else button.textContent = p.text;
          if (p.review) {
            button.dataset.chordCandidate = "true";
            button.append(el("span", p.labels.join(" / "), "chord-candidate-status"));
            button.title += `。${p.reason}`;
            button.setAttribute("aria-label", `${p.text} ${p.labels.join(" / ")}：${p.reason}`);
          }
          button.onclick = () => assign(p.text, p.diagram || null);
          return button;
        }),
      );
    },
  };
}

function libraryReviewEntries(metadata, partId) {
  const part = metadata?.parts?.length
    ? metadata.parts.find(part => part.id === partId) : metadata;
  return (part?.document_metadata?.score_annotations || []).flatMap((annotation, index) => {
    const choices = new Map();
    for (const [diagram, label] of [[annotation.parsed?.diagram, "当前指法"],
      [annotation.chord_review?.original_parsed?.diagram, "原始候选"],
      ...[["original_diagram", "原始候选"], ["diagram", "图示模型候选"],
        ["proposed_diagram", "图示模型候选"], ["secondary_diagram", "另一图示模型候选"],
        ["rejected_diagram_candidate", "待确认的原始候选"]]
        .map(([key, label]) => [annotation.chord_evidence?.[key], label])]) {
      if (diagram?.frets?.length && !choices.has(JSON.stringify(diagram)))
        choices.set(JSON.stringify(diagram), {diagram, label});
    }
    if (!choices.size && !annotation.chord_evidence && !annotation.chord_review
      && !["chord", "chord_diagram"].includes(annotation.parsed?.kind)) return [];
    return [{index, annotation, choices: [...choices.values()]}];
  });
}

export function initChordLibraryReview({render, setBusy}) {
  let entries = [], partId = "", selected = "", sourceRequest = 0;
  const current = () => entries.find(entry => String(entry.index) === selected);
  const status = annotation => chordReviewDecision(annotation) === "confirmed" ? "已确认"
    : chordReviewDecision(annotation) === "rejected" ? "已排除"
    : hasPendingChordCandidate(annotation) ? "待确认" : "已识别";
  function controls() {
    const entry = current();
    $("libraryChordReset").disabled = ui.busy || !entry?.annotation.chord_review;
    $("libraryChordDiscard").hidden = !ui.libraryDirty;
    $("libraryChordStatus").textContent = ui.libraryDirty ? "指法库有未保存的修改。"
      : entry ? status(entry.annotation) : "";
  }
  function showEntry() {
    const request = ++sourceRequest, canvas = $("libraryChordSource");
    canvas.hidden = true;
    const entry = current();
    if (!entry) return;
    $("libraryChordCandidate").replaceChildren(...entry.choices.map((choice, index) => new Option(choice.label, index)));
    $("libraryChordCandidate").closest("label").hidden = entry.choices.length < 2;
    const name = entry.annotation.parsed?.text || entry.annotation.chord_review?.original_parsed?.text || "";
    writeChordFields({name, diagram: entry.annotation.parsed?.diagram || entry.choices[0]?.diagram}, "libraryChord");
    controls();
    const annotation = entry.annotation, page = ui.state.pages[(annotation.page || 1) - 1];
    $("libraryChordSourceLabel").textContent = page ? `原谱第 ${annotation.page || 1} 页` : "原谱裁图不可用";
    if (!page?.url || !Array.isArray(annotation.bbox)) return;
    const image = new Image();
    image.onload = () => {
      if (request !== sourceRequest) return;
      const [x, y, width, height] = annotation.bbox;
      const left = Math.max(0, x - width * 0.15), top = Math.max(0, y - height * 0.15);
      const w = Math.min(image.naturalWidth, x + width * 1.15) - left;
      const h = Math.min(image.naturalHeight, y + height * 1.15) - top;
      if (!(w > 0 && h > 0)) return;
      const scale = Math.min(420 / w, 240 / h);
      canvas.width = Math.max(1, Math.round(w * scale));
      canvas.height = Math.max(1, Math.round(h * scale));
      canvas.getContext("2d").drawImage(image, left, top, w, h, 0, 0, canvas.width, canvas.height);
      canvas.hidden = false;
    };
    image.onerror = () => {
      if (request === sourceRequest) $("libraryChordSourceLabel").textContent = "原谱裁图加载失败，请在页面预览中核对。";
    };
    image.src = page.url;
  }
  $("libraryChordEntry").onchange = () => {
    if (ui.libraryDirty) {
      $("libraryChordEntry").value = selected;
      return notice("请先保存或放弃当前指法修改。", true);
    }
    selected = $("libraryChordEntry").value;
    showEntry();
  };
  $("libraryChordCandidate").onchange = () => {
    const diagram = current()?.choices[+$("libraryChordCandidate").value]?.diagram;
    writeChordFields({name: $("libraryChordName").value, diagram}, "libraryChord");
    ui.libraryDirty = true;
    controls();
  };
  for (const field of ["Name", "Frets", "Base", "Fingers", "Barres"])
    $("libraryChord" + field).oninput = () => {
      ui.libraryDirty = true;
      try {
        const {name, diagram} = readChordFields("libraryChord");
        $("libraryChordPreview").replaceChildren(...(diagram ? [diagramSvg(name, diagram)] : []));
      } catch {
        // Partial input must not keep displaying a different, previously valid diagram.
        $("libraryChordPreview").replaceChildren();
      }
      controls();
    };
  $("libraryChordDiscard").onclick = () => {
    ui.libraryDirty = false;
    showEntry();
  };
  for (const [id, decision] of [["Confirm", "confirm"], ["Reject", "reject"], ["Reset", "reset"]])
    $("libraryChord" + id).onclick = action(async () => {
      const entry = current();
      if (!entry || ui.busy) return;
      if (ui.metadataDirty || ui.measureDirty || ui.boxDirty)
        return notice("请先保存或放弃其他未保存的修改，再保存指法库。", true);
      const body = {part_id: partId || "part-1", decision};
      if (decision === "confirm") {
        Object.assign(body, readChordFields("libraryChord"));
        if (!body.name || !body.diagram) throw new Error("请填写和弦名称与完整指法，再确认保存。");
      }
      const sid = ui.sid, generation = ui.openGeneration;
      setBusy(true);
      try {
        const saved = await api(endpoint(`/chord-library/${entry.index}`), "PUT", body, ui.state.revision);
        if (ui.sid !== sid || ui.openGeneration !== generation) return;
        receiveProject(saved);
        render();
        notice(decision === "confirm" ? "指法库已保存。关联到具体拍点后，指法图会随该拍导出到 GP。"
          : decision === "reject" ? "已从指法库排除该候选，可恢复原识别结果。" : "已恢复原识别结果。");
      } finally {
        if (ui.sid === sid && ui.openGeneration === generation) setBusy(false);
      }
    });
  function renderLibrary(nextPart = partId) {
    if (ui.libraryDirty) return;
    if (partId !== nextPart) selected = "";
    partId = nextPart;
    entries = libraryReviewEntries(ui.state?.metadata, partId);
    $("libraryChordReview").hidden = !entries.length;
    $("libraryChordEntry").replaceChildren(...entries.map(({index, annotation}) =>
      new Option(`${index + 1}. ${annotation.parsed?.text || annotation.chord_review?.original_parsed?.text || "未命名"} · ${status(annotation)}`, index)));
    if (!entries.some(entry => String(entry.index) === selected)) selected = String(entries[0]?.index ?? "");
    $("libraryChordEntry").value = selected;
    showEntry();
  }
  return {renderLibrary, controls};
}
