import { ui } from "./state.js";
import { $, el, action, notice } from "./dom.js";
import { api, endpoint } from "./api.js";
export function initMeasures({ start, go, renderExport }) {
  const currentMode = () => ui.state.measures?.[ui.measureIndex]?.mode || ui.state.mode;
  $("recognize").onclick = action(async () => {
    if (
      ui.state.recognition &&
      !confirm("重新识别会替换当前结果和手工校对内容，继续？")
    )
      return;
    await start("/recognize", undefined, () => {
      ui.measureDirty = false;
    });
  });
  $("resumeOcr").onclick = action(() => start("/recognize", { resume: true }));
  $("retryIssues").onclick = action(() =>
    start("/recognize", { measures: ui.state.review_measures }),
  );
  $("retryMeasure").onclick = action(async () => {
    if (!confirm("重新识别会替换当前小节的编辑，其余小节会保留，继续？"))
      return;
    await start("/recognize", { measures: [ui.measureIndex + 1] }, () => {
      ui.measureDirty = false;
    });
  });
  function renderMeasures() {
    $("resumeOcr").hidden = !ui.state.ocr_task;
    $("retryIssues").hidden = !ui.state.review_measures?.length;
    const measures = ui.state.measures || [];
    $("ocrEmpty").hidden = !!measures.length;
    $("reviewEditor").hidden = !measures.length;
    $("recognize").textContent = measures.length
      ? "重新识别小节"
      : "开始识别小节";
    $("recognitionSummary").textContent = measures.length
      ? `共 ${measures.length} 小节 · ${ui.state.review_measures.length} 个待检查`
      : "";
    if (!measures.length) return;
    ui.measureIndex = Math.min(ui.measureIndex, measures.length - 1);
    $("measureSelect").replaceChildren();
    measures.forEach((m, i) => {
      const option = el(
        "option",
        `第 ${i + 1} 小节${m.needs_review ? " · 待检查" : ""}`,
      );
      option.value = i;
      $("measureSelect").append(option);
    });
    renderMeasure();
  }
  function selectMeasure(i) {
    if (ui.measureDirty && !confirm("当前小节尚未保存，放弃修改并切换？"))
      return;
    $("advanced").open = false;
    ui.measureDirty = false;
    ui.measureIndex = Math.max(0, Math.min(ui.state.measures.length - 1, i));
    renderMeasure();
  }
  $("measureSelect").onchange = () => selectMeasure(+$("measureSelect").value);
  $("prevMeasure").onclick = () => selectMeasure(ui.measureIndex - 1);
  $("nextMeasure").onclick = () => selectMeasure(ui.measureIndex + 1);
  $("nextIssue").onclick = () => {
    const indexes = ui.state.measures
      .map((m, i) => (m.needs_review ? i : -1))
      .filter((i) => i >= 0);
    if (!indexes.length) return notice("没有必须检查的小节。");
    selectMeasure(indexes.find((i) => i > ui.measureIndex) ?? indexes[0]);
  };
  function input(value, type = "text") {
    const n = el("input");
    n.type = type;
    n.value = value;
    return n;
  }
  function select(options, value) {
    const n = el("select");
    for (const [v, t] of options) {
      const o = el("option", t);
      o.value = v;
      n.append(o);
    }
    n.value = value;
    return n;
  }
  function noteText(notes) {
    return (notes || [])
      .map((n) =>
        currentMode() === "notation"
          ? String(n.pitch)
          : `${n.string}:${n.fret}${currentMode() === "both" ? `:${n.pitch}` : ""}`,
      )
      .join(",");
  }
  function renderMeasure() {
    const m = ui.state.measures[ui.measureIndex];
    $("measureSelect").value = ui.measureIndex;
    $("measureImage").src = m.url;
    $("measureText").value = m.score_text;
    $("timeSignature").value = m.parsed.time_signature || "";
    $("measureTempo").value = m.parsed.tempo_quarter || "";
    $("reviewStatus").textContent = m.needs_review
      ? "待检查"
      : m.manually_edited
        ? "已人工确认"
        : "识别完成";
    $("fallback").hidden = !m.needs_review;
    $("fallback").textContent =
      "这个小节未能通过识别校验，暂以整小节休止符占位。请对照原图修改，或确认这里确实应为休止。\n" +
      (m.fallback_reason || []).join("; ");
    $("measureSaved").textContent = m.manually_edited
      ? "已保存人工校对结果"
      : "";
    $("notesHeading").textContent =
      currentMode() === "notation"
        ? "音符（MIDI 音高，逗号分隔）"
        : currentMode() === "both"
          ? "音符（弦:品:MIDI 音高）"
          : "音符（弦:品，逗号分隔）";
    ui.eventData = m.parsed.voices.flatMap((v) =>
      v.events.map((e) => ({ voice: v.voice, event: structuredClone(e) })),
    );
    renderEvents();
  }
  function renderEvents() {
    const tbody = $("eventRows");
    tbody.replaceChildren();
    ui.eventData.forEach((row, i) => {
      const tr = el("tr");
      const e = row.event;
      const voice = select(
        [
          ["0", "声部 1"],
          ["1", "声部 2"],
        ],
        row.voice,
      );
      voice.onchange = () => {
        row.voice = +voice.value;
        ui.measureDirty = true;
      };
      const start = input(e.start / 960, "number");
      start.step = "0.0625";
      start.min = "0";
      start.oninput = () => {
        e.start = Math.round(+start.value * 960);
        ui.measureDirty = true;
      };
      const duration = select(
        [
          [1, "全音符"],
          [2, "二分"],
          [4, "四分"],
          [8, "八分"],
          [16, "十六分"],
          [32, "三十二分"],
          [64, "六十四分"],
        ],
        e.duration.value,
      );
      duration.onchange = () => {
        e.duration.value = +duration.value;
        ui.measureDirty = true;
      };
      const dots = select(
        [
          ["0", "无"],
          ["1", "附点"],
          ["2", "双附点"],
        ],
        e.duration.double_dotted ? 2 : e.duration.dotted ? 1 : 0,
      );
      dots.onchange = () => {
        e.duration.dotted = dots.value === "1";
        e.duration.double_dotted = dots.value === "2";
        ui.measureDirty = true;
      };
      const status = select(
        [
          ["normal", "音符"],
          ["rest", "休止"],
          ["empty", "空拍"],
        ],
        e.status,
      );
      status.onchange = () => {
        e.status = status.value;
        ui.measureDirty = true;
      };
      const notes = input(row.noteInput ?? noteText(e.notes));
      notes.className = "notes";
      notes.oninput = () => {
        row.noteInput = notes.value;
        ui.measureDirty = true;
      };
      const remove = el("button", "×", "quiet");
      remove.setAttribute("aria-label", `删除第 ${i + 1} 个事件`);
      remove.onclick = () => {
        ui.eventData.splice(i, 1);
        ui.measureDirty = true;
        renderEvents();
      };
      for (const n of [voice, start, duration, dots, status, notes, remove]) {
        const td = el("td");
        td.append(n);
        tr.append(td);
      }
      tbody.append(tr);
    });
  }
  $("addEvent").onclick = () => {
    const last = ui.eventData.at(-1)?.event;
    const start = last
      ? last.start +
        ((3840 / last.duration.value) *
          (last.duration.double_dotted
            ? 1.75
            : last.duration.dotted
              ? 1.5
              : 1) *
          (last.duration.tuplet_times || 1)) /
          (last.duration.tuplet_enters || 1)
      : 0;
    ui.eventData.push({
      voice: ui.eventData.at(-1)?.voice || 0,
      event: {
        start: Math.round(start),
        duration: { value: 4 },
        status: "rest",
        notes: [],
        effects: [],
      },
    });
    ui.measureDirty = true;
    renderEvents();
  };
  for (const id of ["timeSignature", "measureTempo", "measureText"])
    $(id).oninput = () => {
      ui.measureDirty = true;
    };
  function editedMeasure() {
    const m = structuredClone(ui.state.measures[ui.measureIndex].parsed);
    m.time_signature = $("timeSignature").value.trim() || null;
    m.print_time_signature = !!m.time_signature;
    m.tempo_quarter = $("measureTempo").value ? +$("measureTempo").value : null;
    const voices = new Map();
    for (const row of ui.eventData) {
      const e = structuredClone(row.event);
      if (row.noteInput !== undefined && e.status === "normal") {
        e.notes = row.noteInput
          .split(",")
          .filter((s) => s.trim())
          .map((s) => {
            const match = s
              .trim()
              .match(
                currentMode() === "notation"
                  ? /^(\d+)$/
                  : currentMode() === "both"
                    ? /^(\d+):(x|\d+):(\d+)$/
                    : /^(\d+):(x|\d+)$/,
              );
            if (!match)
              throw new Error("音符格式不正确，请按表头中的格式填写。");
            if (currentMode() === "notation") {
              const old = e.notes.find((n) => n.pitch === +match[1]) || {};
              return { ...old, pitch: +match[1] };
            }
            const old = e.notes.find((n) => n.string === +match[1]) || {};
            return {
              ...old,
              string: +match[1],
              fret: match[2] === "x" ? "x" : +match[2],
              ...(currentMode() === "both" ? { pitch: +match[3] } : {}),
            };
          });
      }
      if (e.status !== "normal") e.notes = [];
      if (!voices.has(row.voice)) voices.set(row.voice, []);
      voices.get(row.voice).push(e);
    }
    m.voices = Array.from(voices, ([voice, events]) => ({
      voice,
      events: events.sort((a, b) => a.start - b.start),
    })).sort((a, b) => a.voice - b.voice);
    return m;
  }
  async function saveMeasure(raw) {
    const body = raw
      ? { target: $("measureText").value.trim(), reviewed: true }
      : { measure: editedMeasure(), reviewed: true };
    ui.state = await api(
      endpoint(`/measures/${ui.measureIndex + 1}`),
      "PUT",
      body,
    );
    ui.measureDirty = false;
    renderMeasures();
    renderExport();
    notice(`第 ${ui.measureIndex + 1} 小节已保存并确认。`);
  }
  $("saveMeasure").onclick = action(() => saveMeasure(false));
  $("saveRaw").onclick = action(() => saveMeasure(true));
  $("toExport").onclick = () => go(4);

  return { renderMeasures };
}
