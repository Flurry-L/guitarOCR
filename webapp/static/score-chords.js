import { $, el } from "./dom.js";

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

function diagramSvg(name, d) {
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
  for (const [f, low, high] of d.barres)
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
    if (d.fingers[i] !== null)
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
      const name = $("chordName").value.trim();
      const fretsText = $("chordFrets").value.trim();
      let d = null;
      if (fretsText) {
        const frets = fretsText
          .toLowerCase()
          .split(/[\s,]+/)
          .map((f) => (f === "x" ? "x" : Number(f)));
        const base = Number($("chordBase").value || 1);
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
        const fingers = $("chordFingers").value.trim()
          ? $("chordFingers")
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
        const barres = $("chordBarres")
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
      $("chordName").value = name;
      $("chordFrets").value = diagram?.frets.join(" ") || "";
      $("chordBase").value = diagram?.base_fret || 1;
      $("chordFingers").value =
        diagram?.fingers.map((f) => f ?? "-").join(" ") || "";
      $("chordBarres").value =
        diagram?.barres
          .map(
            ([f, low, high]) =>
              `${f}:${diagram.frets.length - low}-${diagram.frets.length - high}`,
          )
          .join(" ") || "";
      $("chordPreview").replaceChildren(
        ...(diagram ? [diagramSvg(name, diagram)] : []),
      );
      $("applyChord").disabled = $("removeChord").disabled = !e;
    },
    library(state, partId = "part-1") {
      const sources = state.metadata?.parts?.length
        ? state.metadata.parts.filter((p) => p.id === partId)
        : [state.metadata];
      const values = new Map();
      for (const source of sources)
        for (const a of source?.document_metadata?.score_annotations || []) {
          const p = a.parsed;
          if (["chord", "chord_diagram"].includes(p?.kind) && p.text)
            values.set(JSON.stringify([p.text, p.diagram || null]), p);
        }
      const key = (name) =>
        name.replaceAll("♭", "b").replaceAll("♯", "#").replace(/\s/g, "");
      const illustrated = new Set(
        [...values.values()].filter((p) => p.diagram).map((p) => key(p.text)),
      );
      const entries = [...values.values()].filter(
        (p) => p.diagram || !illustrated.has(key(p.text)),
      );
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
          button.onclick = () => assign(p.text, p.diagram || null);
          return button;
        }),
      );
    },
  };
}
