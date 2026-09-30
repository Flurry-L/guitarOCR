import { ui, endpoint, receiveProject } from "./state.js";
import { $, el, action, notice } from "./dom.js";
import { api } from "./api.js";
import { editorMenu } from "./editor-menu.js";
import { scoreView } from "./score-view.js";
import { measureProfile, reviewKind } from "./score-engraving.js";
import { chordEditor } from "./score-chords.js";
import {
  clone,
  ticks,
  pitchName,
  parsePitch,
  pitchShift,
  setFret,
  setPitch,
  removeNote,
  changeDuration,
  insertEvent,
  removeEvent,
  drumNames,
} from "./score-model.js";

export function initMeasures({ start, go, renderExport, setBusy }) {
  let draft,
    selected,
    profile,
    undo = [],
    redo = [],
    saving = false,
    digits = "",
    digitTime = 0,
    digitUndoRecorded = false;
  let projectId,
    projectRevision,
    track = "all";
  const histories = new Map();
  let clipboard = null;
  const menu = editorMenu($("scoreContextMenu"), $("scoreCanvas"));
  const studio = $("reviewEditor");
  function inspector(visible) {
    studio.classList.toggle("hide-inspector", !visible);
    $("toggleInspector").setAttribute("aria-pressed", String(visible));
  }
  inspector(!matchMedia("(max-width: 760px)").matches);
  $("toggleInspector").onclick = () =>
    inspector(studio.classList.contains("hide-inspector"));
  function focusEditor(focused) {
    studio.classList.toggle("is-focused", focused);
    $("focusEditor").setAttribute("aria-pressed", String(focused));
    $("focusEditor").title = focused
      ? "退出专注编辑（Esc）"
      : "专注编辑（Esc 退出）";
    $("scoreCanvas").focus({ preventScroll: true });
  }
  $("focusEditor").onclick = () =>
    focusEditor(!studio.classList.contains("is-focused"));
  document.addEventListener("keydown", (e) => {
    if (
      e.key === "Escape" &&
      studio.classList.contains("is-focused") &&
      !document.querySelector("dialog[open]")
    ) {
      focusEditor(false);
    }
  });
  studio.addEventListener("click", (e) => {
    if (
      e.target.closest(
        ".command-toolbar button,.effect-tools button,#fretButtons button,#pitchButtons button",
      )
    )
      $("scoreCanvas").focus({ preventScroll: true });
  });
  studio.addEventListener("change", (e) => {
    if (e.target.closest(".command-toolbar"))
      $("scoreCanvas").focus({ preventScroll: true });
  });
  const view = scoreView($("scoreCanvas"), {
    onSelect: (hit) => selectMeasure(hit.mi, hit),
    onContext: async (hit, x, y) => {
      if (await selectMeasure(hit.mi, hit)) menu.open(x, y);
    },
    onPlace: async (hit, pitch, add) => {
      if (!(await selectMeasure(hit.mi, hit))) return;
      change(() => {
        selected.ni = setPitch(
          event(),
          add ? -1 : selected.ni,
          pitch +
            pitchShift(
              profile.pitch_context,
              event().effects,
              profile.instrument,
            ),
          profile.mode,
          profile.tuning,
        );
        selected.string = event().notes[selected.ni].string || 1;
      });
    },
    onError: (message) => {
      $("scoreWarning").hidden = false;
      $("scoreWarning").textContent =
        "谱面暂时无法绘制，可在右侧修改小节文本。";
      console.warn("Score rendering failed", message);
    },
  });
  const current = () => ui.state?.measures?.[ui.measureIndex];
  const measureLabel = (m, i) =>
    `${m.part_name ? m.part_name + " · " : ""}${m.staff_id && m.staff_id !== "staff-1" ? "谱表 " + m.staff_id.replace("staff-", "") + " · " : ""}第 ${(m.bar_index ?? i) + 1} 小节`;
  function neighbour(delta) {
    const row = current(),
      indexes = ui.state.measures.flatMap((m, i) =>
        m.part_id === row?.part_id && m.staff_id === row?.staff_id ? [i] : [],
      );
    return (
      indexes[
        Math.max(
          0,
          Math.min(
            indexes.length - 1,
            indexes.indexOf(ui.measureIndex) + delta,
          ),
        )
      ] ?? ui.measureIndex
    );
  }
  const event = () => draft?.voices[selected?.vi]?.events[selected?.ei];
  const chords = chordEditor(change, event);
  function feedback(message = "", error = false) {
    $("measureFeedback").textContent = message;
    $("measureFeedback").hidden = !message;
    $("measureFeedback").classList.toggle("error", error);
  }
  function snapshot() {
    return { draft: clone(draft), selected: clone(selected) };
  }
  function change(fn, { merge = false, typed = false } = {}) {
    if (ui.busy || saving || !draft || ui.editorMode !== "score") return;
    if (!typed) {
      digits = "";
      digitTime = 0;
    }
    const before = snapshot();
    try {
      fn();
    } catch (error) {
      draft = before.draft;
      selected = before.selected;
      feedback(error.message, true);
      return;
    }
    if (JSON.stringify(before.draft) === JSON.stringify(draft)) return;
    if (!merge || !undo.length) undo.push(before);
    if (undo.length > 100) undo.shift();
    redo = [];
    updateDirty();
    feedback();
    draw();
    return true;
  }
  function updateDirty() {
    ui.measureDirty =
      ui.editorMode === "text"
        ? $("measureText").value.trim() !== current()?.score_text?.trim()
        : JSON.stringify(draft) !== JSON.stringify(current()?.parsed);
  }
  function restore(from, to) {
    if (!from.length || ui.busy || saving || ui.editorMode !== "score") return;
    digits = "";
    digitTime = 0;
    to.push(snapshot());
    const before = from.pop();
    draft = before.draft;
    selected = before.selected;
    updateDirty();
    feedback();
    draw();
  }
  $("undoNote").onclick = () => restore(undo, redo);
  $("redoNote").onclick = () => restore(redo, undo);
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
  $("retryIssues").onclick = action(async () => {
    if (!confirm("重新识别会替换标记小节中的内容，继续？")) return;
    await start("/recognize", { measures: ui.state.review_measures });
  });
  $("retryMeasure").onclick = action(async () => {
    if (!confirm("重新识别会替换当前小节的编辑，继续？")) return;
    ui.measureDirty = false;
    await start("/recognize", { measures: [ui.measureIndex + 1] });
  });
  function renderNavigator() {
    const measures = ui.state.measures || [],
      parts = new Map();
    for (const m of measures)
      parts.set(m.part_id || "part-1", m.part_name || "音轨 1");
    if (track !== "all" && !parts.has(track)) track = "all";
    $("trackSelect").replaceChildren(
      new Option("全部音轨", "all"),
      ...[...parts].map(([id, name]) => new Option(name, id)),
    );
    $("trackSelect").value = track;
    const inTrack = measures.flatMap((m, i) =>
      track === "all" || (m.part_id || "part-1") === track ? [i] : [],
    );
    const filter = $("reviewFilter").value;
    const visible = inTrack.filter(
      (i) =>
        filter === "all" ||
        (filter === "review"
          ? measures[i].needs_review
          : measures[i].reviewed && !measures[i].needs_review),
    );
    $("trackSummary").textContent =
      `${inTrack.length} 个谱表小节 · ${inTrack.filter((i) => measures[i].needs_review).length} 个待检查`;
    const groups = new Map();
    for (const i of visible) {
      const m = measures[i],
        key = `${m.part_id || "part-1"}/${m.staff_id || "staff-1"}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(i);
    }
    $("measureMap").replaceChildren(
      ...[...groups.values()].map((indexes) => {
        const row = el("div", undefined, "measure-map-group");
        {
          const m = measures[indexes[0]];
          row.append(
            el(
              "span",
              `${m.part_name || "音轨"}${m.staff_id && m.staff_id !== "staff-1" ? " · 谱表 " + m.staff_id.replace("staff-", "") : ""}`,
              "map-track-label",
            ),
          );
        }
        row.append(
          ...indexes.map((i) => {
            const m = measures[i],
              button = el(
                "button",
                String((m.bar_index ?? i) + 1),
                `${reviewKind(m)} ${m.reviewed ? "confirmed" : ""} ${i === ui.measureIndex ? "active" : ""}`,
              );
            button.title =
              measureLabel(m, i) +
              (m.needs_review ? " · 待检查" : m.reviewed ? " · 已确认" : "");
            button.setAttribute("aria-label", button.title);
            button.setAttribute("aria-pressed", String(i === ui.measureIndex));
            button.disabled = ui.busy;
            button.onclick = () => selectMeasure(i, undefined, true);
            return button;
          }),
        );
        return row;
      }),
    );
    if (!visible.length)
      $("measureMap").append(el("p", "当前音轨没有符合条件的小节。"));
  }
  $("trackSelect").onchange = async () => {
    const next = $("trackSelect").value;
    if (ui.measureDirty && !(await save(false))) {
      $("trackSelect").value = track;
      return;
    }
    const bar = current()?.bar_index;
    track = next;
    let index = ui.state.measures.findIndex(
      (m) =>
        m.bar_index === bar &&
        (track === "all" || (m.part_id || "part-1") === track),
    );
    if (index < 0)
      index = ui.state.measures.findIndex(
        (m) => track === "all" || (m.part_id || "part-1") === track,
      );
    if (index >= 0) await selectMeasure(index, undefined, true);
    renderNavigator();
    draw();
  };
  $("reviewFilter").onchange = renderNavigator;
  $("confirmNext").onclick = async () => {
    if (!(await save())) return;
    const indexes = ui.state.measures.flatMap((m, i) =>
      m.needs_review && (track === "all" || (m.part_id || "part-1") === track)
        ? [i]
        : [],
    );
    if (indexes.length)
      await selectMeasure(
        indexes.find((i) => i > ui.measureIndex) ?? indexes[0],
        undefined,
        true,
      );
    else feedback("当前音轨的待检查小节已全部确认。");
  };
  function renderSummary() {
    const measures = ui.state.measures || [];
    $("resumeOcr").hidden = !ui.state.ocr_task;
    $("retryIssues").hidden = !ui.state.review_measures?.length;
    $("ocrEmpty").hidden = !!measures.length;
    $("reviewEditor").hidden = !measures.length;
    $("recognize").textContent = measures.length ? "重新识别整谱" : "识别乐谱";
    $("recognize").classList.toggle("primary", !measures.length);
    const bars = new Set(measures.map((m, i) => m.bar_index ?? i)).size,
      parts = new Set(measures.map((m) => m.part_id || "part-1")).size;
    $("recognitionSummary").textContent =
      `${bars} 小节${parts > 1 ? ` · ${parts} 音轨` : ""} · ${ui.state.review_measures?.length || 0} 个待检查`;
    $("scoreTitle").textContent =
      ui.state.metadata?.title || ui.state.input_names?.[0] || "未命名乐谱";
    $("scoreCredits").textContent = ui.state.metadata?.artist || "";
    $("measureSelect").replaceChildren(
      ...measures.map((m, i) => {
        const o = el(
          "option",
          `${measureLabel(m, i)}${m.needs_review ? " · 待检查" : m.reviewed ? " · 已确认" : ""}`,
        );
        o.value = i;
        return o;
      }),
    );
    $("measureSelect").value = ui.measureIndex;
    renderNavigator();
  }
  function renderMeasures() {
    if (projectId !== ui.sid || projectRevision !== ui.state.revision) {
      histories.clear();
      draft = null;
      if (projectId !== ui.sid) {
        track = "all";
        $("reviewFilter").value = "all";
      }
      projectId = ui.sid;
      projectRevision = ui.state.revision;
    }
    renderSummary();
    if (!ui.state.measures?.length) {
      draft = null;
      updateMeasureControls();
      return;
    }
    ui.measureIndex = Math.min(ui.measureIndex, ui.state.measures.length - 1);
    if (!draft) loadMeasure();
    draw();
  }
  function firstHit() {
    const vi = 0,
      ei = 0,
      note = draft.voices[vi]?.events[ei]?.notes?.[0];
    return {
      mi: ui.measureIndex,
      vi,
      ei,
      ni: note ? 0 : -1,
      string: note?.string || 1,
      kind: profile.mode === "notation" ? "notation" : "tab",
    };
  }
  async function selectMeasure(index, hit, scroll = false) {
    if (ui.busy || saving || !draft) return false;
    index = Math.max(0, Math.min(ui.state.measures.length - 1, index));
    if (index !== ui.measureIndex) {
      if (ui.measureDirty && !(await save(false))) {
        $("measureSelect").value = ui.measureIndex;
        return false;
      }
      histories.set(ui.measureIndex, { undo, redo, selected });
      ui.measureIndex = index;
      if (track !== "all" && (current().part_id || "part-1") !== track)
        track = current().part_id || "part-1";
      loadMeasure();
      renderNavigator();
    }
    if (hit) selected = { ...hit };
    selected.string =
      event()?.notes[selected.ni]?.string || selected.string || 1;
    digits = "";
    digitTime = 0;
    menu.close();
    draw();
    view.select(selected, scroll);
    $("scoreCanvas").focus({ preventScroll: true });
    return true;
  }
  $("measureSelect").onchange = () =>
    selectMeasure(+$("measureSelect").value, undefined, true);
  $("prevMeasure").onclick = () =>
    selectMeasure(neighbour(-1), undefined, true);
  $("nextMeasure").onclick = () => selectMeasure(neighbour(1), undefined, true);
  function openIssue() {
    const indexes = ui.state.measures.flatMap((m, i) =>
      m.needs_review ? [i] : [],
    );
    if (indexes.length)
      selectMeasure(
        indexes.find((i) => i > ui.measureIndex) ?? indexes[0],
        undefined,
        true,
      );
  }
  $("nextIssue").onclick = openIssue;
  function loadMeasure() {
    profile = measureProfile(current(), ui.state);
    draft = clone(current().parsed);
    const history = histories.get(ui.measureIndex);
    undo = history?.undo || [];
    redo = history?.redo || [];
    selected = history?.selected || firstHit();
    $("measureText").value = current().score_text;
    renderReference();
    feedback();
    editorMode();
  }
  function renderReference() {
    const m = current();
    $("currentMeasureTitle").textContent =
      `原谱 · ${measureLabel(m, ui.measureIndex)}`;
    $("measureSelect").value = ui.measureIndex;
    $("measureImage").src = m.url;
    $("reviewStatus").textContent = m.needs_review
      ? "待检查"
      : m.reviewed
        ? "已确认"
        : "识别完成";
    $("reviewStatus").className = `pill ${reviewKind(m)}`;
    $("fallback").hidden = !m.needs_review;
    $("fallback").textContent = m.timing_errors?.length
      ? `${m.timing_errors.join("；")}。请调整起点或时值。`
      : m.export_errors?.length
        ? m.export_errors.join("；")
        : m.fingering_errors?.length
          ? "音高或和弦与当前调弦不匹配，请核对音符和调弦。"
          : m.annotation_review
            ? "和弦标注未能确认，请对照原图核对名称与拍点。未确认的内容保留在小节文字中。"
            : m.fallback_reason?.length
              ? "该小节未通过识别检查，请对照原图核对音符、奏法和音高。"
              : "请对照原谱核对谱号、移调和音高，确认后即可导出。";
  }
  function updateMeasureControls() {
    const count = ui.state?.measures?.length || 0,
      busy = ui.busy || saving;
    $("prevMeasure").disabled =
      busy || !count || neighbour(-1) === ui.measureIndex;
    $("nextMeasure").disabled =
      busy || !count || neighbour(1) === ui.measureIndex;
    $("nextIssue").hidden = !ui.state?.review_measures?.length;
    $("toExport").disabled = busy || !count;
    $("confirmNext").disabled = busy || !count;
    $("trackSelect").disabled = busy || !count;
    $("reviewFilter").disabled = busy || !count;
    $("saveMeasure").disabled =
      busy ||
      (!ui.measureDirty && !current()?.needs_review && !!current()?.reviewed);
    $("saveMeasure").textContent = ui.measureDirty
      ? "保存并确认"
      : current()?.reviewed && !current()?.needs_review
        ? "已确认"
        : "确认小节";
    $("saveStatus").textContent = saving
      ? "正在保存…"
      : ui.measureDirty
        ? "未保存"
        : "已保存";
    $("discardMeasure").hidden = !ui.measureDirty;
    $("copyBeat").disabled = busy || !event() || ui.editorMode !== "score";
    $("pasteBeat").disabled =
      busy || !clipboard || !event() || ui.editorMode !== "score";
    $("duplicateBeat").disabled = busy || !event() || ui.editorMode !== "score";
    $("undoNote").disabled = busy || !undo.length || ui.editorMode === "text";
    $("redoNote").disabled = busy || !redo.length || ui.editorMode === "text";
    document
      .querySelectorAll(
        ".note-tools button,.note-tools select,.effect-tools button,.effect-tools select,#fretTools button,#fretTools select,#pitchTools button,#pitchTools select,#drumTools button,#drumTools select,#timeSignature,#measureTempo,#eventStart",
      )
      .forEach((node) => {
        if (
          ![
            "undoNote",
            "redoNote",
            "copyBeat",
            "pasteBeat",
            "duplicateBeat",
          ].includes(node.id)
        )
          node.disabled = busy || !draft || ui.editorMode === "text";
      });
    document.querySelectorAll("[data-effect],#bendValue").forEach((node) => {
      node.disabled ||= !event()?.notes[selected?.ni];
    });
    document
      .querySelectorAll(
        ".chord-settings input,.chord-settings button,#chordLibrary button",
      )
      .forEach((node) => {
        node.disabled = busy || !event() || ui.editorMode !== "score";
      });
  }
  function editorMode() {
    $("editMode").value = ui.editorMode;
    $("textEditor").hidden = ui.editorMode !== "text";
  }
  $("editMode").onchange = async () => {
    const next = $("editMode").value;
    if (ui.measureDirty && !(await save(false))) {
      $("editMode").value = ui.editorMode;
      return;
    }
    ui.editorMode = next;
    editorMode();
    draw();
  };
  function draw() {
    if (!draft) return;
    selected.mi = ui.measureIndex;
    $("timeSignature").value = draft.time_signature || "";
    $("measureTempo").value = draft.tempo_quarter || "";
    $("scoreWarning").hidden = true;
    view.render(ui.state, { index: ui.measureIndex, measure: draft }, track);
    view.select(selected);
    const e = event(),
      n = e?.notes[selected?.ni];
    chords.render(e);
    chords.library(ui.state, current()?.part_id || "part-1");
    if (e) {
      const voices = Math.min(
        16,
        Math.max(2, ...draft.voices.map((v) => v.voice + 2)),
      );
      $("editorVoice").replaceChildren(
        ...Array.from(
          { length: voices },
          (_, i) => new Option(`声部 ${i + 1}`, i),
        ),
      );
      $("editorVoice").value = draft.voices[selected.vi].voice;
      for (const b of $("durationTools").children)
        b.setAttribute(
          "aria-pressed",
          String(+b.dataset.duration === e.duration.value),
        );
      $("durationDots").value = e.duration.double_dotted
        ? 2
        : e.duration.dotted
          ? 1
          : 0;
      const tuplet = `${e.duration.tuplet_enters || 1}:${e.duration.tuplet_times || 1}`;
      $("eventTuplet").value = tuplet;
      if (!$("eventTuplet").value) {
        const o = el("option", tuplet);
        o.value = tuplet;
        $("eventTuplet").append(o);
        $("eventTuplet").value = tuplet;
      }
      $("eventStart").value = e.start / 960;
      $("selectedNote").textContent =
        `${measureLabel(current(), ui.measureIndex)} · 声部 ${draft.voices[selected.vi].voice + 1} · 第 ${e.start / 960 + 1} 拍${selected.kind === "tab" ? ` · 第 ${selected.string} 弦` : ""}${n ? "" : " · 休止或空位"}`;
    }
    const drums = profile.instrument === "drums",
      tab = selected.kind === "tab";
    $("noteValue").textContent = n
      ? drums
        ? drumNames[n.pitch] || `鼓音 ${n.pitch}`
        : tab
          ? `${selected.string} 弦 · ${n.fret === "x" ? "X" : n.fret + " 品"}`
          : pitchName(
              n.pitch -
                pitchShift(
                  profile.pitch_context,
                  e.effects,
                  profile.instrument,
                ),
            )
      : e?.status === "rest"
        ? "休止符"
        : "空弦位";
    $("noteDescription").textContent = e
      ? `${{ 1: "全", 2: "二分", 4: "四分", 8: "八分", 16: "十六分", 32: "三十二分", 64: "六十四分" }[e.duration.value] || e.duration.value}音符${e.duration.double_dotted ? " · 双附点" : e.duration.dotted ? " · 附点" : ""}${e.notes.length > 1 ? ` · ${e.notes.length} 音和弦` : ""}`
      : "";
    const signature =
      draft.time_signature ||
      current().score_state?.time ||
      ui.state.metadata?.time_signature ||
      "4/4";
    const [beats, unit] = String(signature).split("/").map(Number);
    const expected = (beats * 3840) / unit;
    const voiceEvents = draft.voices[selected.vi]?.events || [];
    const used = Math.max(0, ...voiceEvents.map((e) => e.start + ticks(e)));
    const over = used > expected;
    $("rhythmStatus").textContent = Number.isFinite(expected)
      ? `${signature} · 本声部 ${Number((used / 960).toFixed(3))} / ${expected / 960} 拍${over ? " · 超出小节" : used < expected ? " · 尚有空拍" : " · 时值完整"}`
      : "";
    $("rhythmStatus").classList.toggle("warning", over);
    $("fretTools").hidden = !tab;
    $("pitchTools").hidden = tab || drums;
    $("drumTools").hidden = !drums;
    $("raiseNote").hidden = $("lowerNote").hidden = drums;
    if (drums && n) $("drumValue").value = n.pitch;
    if (n && !tab && !drums) {
      const written =
        n.pitch -
        pitchShift(profile.pitch_context, e.effects, profile.instrument);
      const octave = String(Math.floor(written / 12) - 1);
      if (
        ![...$("pitchOctave").options].some((option) => option.value === octave)
      )
        $("pitchOctave").append(new Option(octave, octave));
      $("pitchOctave").value = octave;
    }
    $("editHint").textContent = tab
      ? "点选弦位，输入数字或 X。方向键换拍、换弦，Delete 删音。"
      : drums
        ? "点选鼓音，在鼓件栏更换。左右键换拍，Delete 删音。"
        : "点选或拖动音符，上下键升降半音。双击谱表空位加音，Delete 删音。";
    for (const b of $("fretButtons").children)
      b.setAttribute(
        "aria-pressed",
        String(
          n?.fret !== undefined &&
            String(n.fret).toLowerCase() === b.dataset.fret.toLowerCase(),
        ),
      );
    for (const b of document.querySelectorAll("[data-effect]")) {
      b.hidden =
        drums &&
        ["tie", "hammer", "sl", "pm", "vib"].includes(b.dataset.effect);
      b.setAttribute(
        "aria-pressed",
        String(!!n?.effects?.includes(b.dataset.effect)),
      );
    }
    $("bendValue").hidden = drums;
    if (drums) document.querySelector(".note-input-panel").open = true;
    const bend = n?.effects?.find((e) => e.startsWith("bend:")) || "";
    $("bendValue").value = bend;
    if (bend && !$("bendValue").value) {
      const o = el("option", "原谱推弦");
      o.value = bend;
      $("bendValue").append(o);
      $("bendValue").value = bend;
    }
    updateMeasureControls();
  }
  function applyValue(value, merge = false, typed = false) {
    return change(
      () => {
        const e = event();
        if (selected.kind === "tab")
          selected.ni = setFret(
            e,
            selected.string,
            value,
            profile.mode,
            profile.tuning,
          );
        else {
          const pitch =
            profile.instrument === "drums"
              ? Number(value)
              : parsePitch(value) +
                pitchShift(
                  profile.pitch_context,
                  e.effects,
                  profile.instrument,
                );
          selected.ni = setPitch(
            e,
            selected.ni,
            pitch,
            profile.mode,
            profile.tuning,
          );
          selected.string = e.notes[selected.ni].string || selected.string;
        }
      },
      { merge, typed },
    );
  }
  $("copyBeat").onclick = () => {
    if (!event() || ui.busy || saving) return;
    digits = "";
    digitTime = 0;
    clipboard = { beat: clone(event()), profile: clone(profile) };
    feedback("已复制当前拍，可在目标拍粘贴。");
    updateMeasureControls();
  };
  function pasteBeat() {
    if (!clipboard) return;
    change(() => {
      const source = clipboard.profile,
        copy = clone(clipboard.beat);
      if ((source.instrument === "drums") !== (profile.instrument === "drums"))
        throw new Error(
          "鼓轨与旋律音轨的音符类型不同，请在相同类型音轨间粘贴。",
        );
      const same =
        source.mode === profile.mode &&
        JSON.stringify(source.tuning) === JSON.stringify(profile.tuning);
      if (!same) {
        const notes = copy.notes;
        copy.notes = [];
        for (const note of notes) {
          const pitch =
            note.pitch ??
            source.tuning[note.string - 1] +
              (note.fret === "x" ? 0 : note.fret);
          const index = setPitch(
            copy,
            -1,
            pitch,
            profile.mode === "tab" ? "both" : profile.mode,
            profile.tuning,
          );
          copy.notes[index].effects = clone(note.effects || []);
          if (profile.mode === "tab") delete copy.notes[index].pitch;
          if (note.fret === "x" && profile.mode !== "notation")
            copy.notes[index].fret = "x";
        }
      }
      const voice = draft.voices[selected.vi],
        start = event().start;
      changeDuration(voice, selected.ei, copy.duration);
      copy.start = start;
      voice.events[selected.ei] = copy;
      selected.ni =
        selected.kind === "tab"
          ? copy.notes.findIndex((n) => n.string === selected.string)
          : copy.notes.length
            ? 0
            : -1;
    });
  }
  $("pasteBeat").onclick = pasteBeat;
  $("duplicateBeat").onclick = () =>
    change(() => {
      const copy = clone(event()),
        voice = draft.voices[selected.vi];
      const index = insertEvent(voice, selected.ei);
      copy.start = voice.events[index].start;
      voice.events[index] = copy;
      selected.ei = index;
    });
  function fretButtons() {
    const begin = +$("fretRange").value;
    $("fretButtons").replaceChildren(
      ...[...Array(Math.min(12, 30 - begin) + 1)]
        .map((_, i) => String(i + begin))
        .concat("X")
        .map((value) => {
          const b = el("button", value);
          b.dataset.fret = value;
          b.title = value === "X" ? "闷音" : `第 ${value} 品`;
          b.onclick = () => applyValue(value);
          return b;
        }),
    );
  }
  $("fretRange").onchange = () => {
    fretButtons();
    draw();
  };
  fretButtons();
  $("pitchButtons").replaceChildren(
    ...["C", "D", "E", "F", "G", "A", "B"].map((name) => {
      const b = el("button", name);
      b.onclick = () => applyValue(name + $("pitchOctave").value);
      return b;
    }),
  );
  $("drumValue").replaceChildren(
    ...Object.entries(drumNames).map(([v, text]) => {
      const o = el("option", text);
      o.value = v;
      return o;
    }),
  );
  $("drumValue").onchange = () => applyValue($("drumValue").value);
  $("placeDrum").onclick = () => applyValue($("drumValue").value);
  function shiftNote(delta) {
    change(() => {
      const e = event(),
        n = e.notes[selected.ni];
      if (!n) throw new Error("请先选择一个音符。");
      if (selected.kind === "tab")
        selected.ni = setFret(
          e,
          selected.string,
          (n.fret === "x" ? 0 : n.fret) + delta,
          profile.mode,
          profile.tuning,
        );
      else
        selected.ni = setPitch(
          e,
          selected.ni,
          n.pitch + delta,
          profile.mode,
          profile.tuning,
        );
      selected.string = e.notes[selected.ni]?.string || selected.string;
    });
  }
  $("lowerNote").onclick = () => shiftNote(-1);
  $("raiseNote").onclick = () => shiftNote(1);
  function removeSelected() {
    change(() => {
      if (selected.ni >= 0) removeNote(event(), selected.ni);
      selected.ni = -1;
    });
  }
  $("deleteNote").onclick = removeSelected;
  $("deleteEvent").onclick = () =>
    change(() => {
      selected.ei = removeEvent(draft.voices[selected.vi], selected.ei);
      selected.ni = -1;
    });
  $("makeRest").onclick = () =>
    change(() => {
      event().notes = [];
      event().status = "rest";
      selected.ni = -1;
    });
  $("addEvent").onclick = () =>
    change(() => {
      selected.ei = insertEvent(draft.voices[selected.vi], selected.ei);
      selected.ni = -1;
    });
  $("addChordNote").onclick = () =>
    change(() => {
      const e = event();
      if (profile.mode !== "notation") {
        const string =
          profile.tuning.findIndex(
            (_, i) => !e.notes.some((n) => n.string === i + 1),
          ) + 1;
        if (!string) throw new Error("这个和弦的每根弦都已有音符。");
        selected.string = string;
        selected.kind = "tab";
        selected.ni = setFret(e, string, 0, profile.mode, profile.tuning);
      } else {
        const pitch =
          profile.instrument === "drums"
            ? +$("drumValue").value
            : parsePitch("C" + $("pitchOctave").value) +
              pitchShift(profile.pitch_context, e.effects, profile.instrument);
        selected.ni = setPitch(e, -1, pitch, profile.mode, profile.tuning);
      }
    });
  function duration(value = event()?.duration.value) {
    change(() => {
      const [enters, times] = $("eventTuplet").value.split(":").map(Number);
      changeDuration(draft.voices[selected.vi], selected.ei, {
        ...event().duration,
        value,
        dotted: $("durationDots").value === "1",
        double_dotted: $("durationDots").value === "2",
        tuplet_enters: enters,
        tuplet_times: times,
      });
    });
  }
  $("durationTools").onclick = (e) => {
    const b = e.target.closest("[data-duration]");
    if (b) duration(+b.dataset.duration);
  };
  for (const id of ["durationDots", "eventTuplet"])
    $(id).onchange = () => duration();
  $("eventStart").onchange = () =>
    change(() => {
      const value = Number($("eventStart").value);
      if (!Number.isFinite(value) || value < 0)
        throw new Error("起点不能小于零。");
      const chosen = event();
      chosen.start = Math.round(value * 960);
      draft.voices[selected.vi].events.sort((a, b) => a.start - b.start);
      selected.ei = draft.voices[selected.vi].events.indexOf(chosen);
    });
  $("editorVoice").onchange = () => {
    if (!draft) return;
    const id = +$("editorVoice").value,
      index = draft.voices.findIndex((v) => v.voice === id);
    if (index >= 0) {
      selected = {
        vi: index,
        ei: 0,
        ni: -1,
        string: 1,
        kind: profile.mode === "notation" ? "notation" : "tab",
      };
      draw();
    } else
      change(() => {
        draft.voices.push({
          voice: id,
          events: [
            {
              start: 0,
              duration: { value: 4 },
              status: "rest",
              notes: [],
              effects: [],
            },
          ],
        });
        selected.vi = draft.voices.length - 1;
        selected.ei = 0;
        selected.ni = -1;
      });
  };
  $("timeSignature").onchange = () =>
    change(() => {
      const value = $("timeSignature").value.trim();
      if (value && !/^[1-9]\d?\/(1|2|4|8|16|32|64)$/.test(value))
        throw new Error(
          "拍号格式为 4/4、6/8 等，分母须为 1、2、4、8、16、32 或 64。",
        );
      draft.time_signature = value || null;
      draft.print_time_signature = !!value;
    });
  $("measureTempo").onchange = () =>
    change(() => {
      draft.tempo_quarter = $("measureTempo").value
        ? +$("measureTempo").value
        : null;
    });
  $("measureText").oninput = () => {
    updateDirty();
    updateMeasureControls();
  };
  document.querySelectorAll("[data-effect]").forEach(
    (b) =>
      (b.onclick = () =>
        change(() => {
          const n = event().notes[selected.ni];
          if (!n) return;
          n.effects ||= [];
          n.effects = n.effects.includes(b.dataset.effect)
            ? n.effects.filter((e) => e !== b.dataset.effect)
            : [...n.effects, b.dataset.effect];
        })),
  );
  $("bendValue").onchange = () =>
    change(() => {
      const n = event().notes[selected.ni];
      if (!n) return;
      n.effects = (n.effects || []).filter((e) => !e.startsWith("bend:"));
      if ($("bendValue").value) n.effects.push($("bendValue").value);
    });
  $("scoreZoom").onchange = () => view.zoom(+$("scoreZoom").value);
  $("sourceZoom").onclick = () => {
    $("largeSource").src = current().url;
    $("sourceDialog").showModal();
  };
  $("showSource").onclick = () => $("sourceZoom").click();
  $("closeSource").onclick = () => $("sourceDialog").close();
  $("reviewEditor").addEventListener("keydown", (e) => {
    if (ui.busy || saving) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      save();
      return;
    }
    if (ui.editorMode !== "score" || e.target.matches("input,textarea,select"))
      return;
    const modifier = e.ctrlKey || e.metaKey;
    const key = e.key.toLowerCase();
    const command = modifier
      ? { c: "copyBeat", v: "pasteBeat", d: "duplicateBeat", y: "redoNote" }[
          key
        ]
      : null;
    if (command) {
      e.preventDefault();
      menu.run(command);
      return;
    }
    if (modifier && key === "z") {
      e.preventDefault();
      restore(e.shiftKey ? redo : undo, e.shiftKey ? undo : redo);
    }
  });
  $("scoreCanvas").onkeydown = (e) => {
    if (
      ui.busy ||
      saving ||
      ui.editorMode !== "score" ||
      !event() ||
      e.ctrlKey ||
      e.metaKey ||
      e.altKey
    )
      return;
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      if (e.shiftKey) $("deleteEvent").click();
      else removeSelected();
      return;
    }
    if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) {
      e.preventDefault();
      const rect =
        $("scoreCanvas")
          .querySelector(".note-selection,.score-selection")
          ?.getBoundingClientRect() || $("scoreCanvas").getBoundingClientRect();
      menu.open(rect.left + 10, rect.top + 24);
      return;
    }
    if (["+", "=", "-", ".", "t", "T", "Insert"].includes(e.key)) {
      e.preventDefault();
      if (e.key === "Insert") $("addEvent").click();
      else if (e.key === ".") {
        $("durationDots").value = String(
          (Number($("durationDots").value) + 1) % 3,
        );
        duration();
      } else if (e.key.toLowerCase() === "t") {
        $("eventTuplet").value =
          $("eventTuplet").value === "3:2" ? "1:1" : "3:2";
        duration();
      } else
        duration(
          Math.max(
            1,
            Math.min(64, event().duration.value * (e.key === "-" ? 0.5 : 2)),
          ),
        );
      return;
    }
    if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      selected.ei =
        e.key === "Home" ? 0 : draft.voices[selected.vi].events.length - 1;
      selected.ni =
        selected.kind === "tab"
          ? event().notes.findIndex((n) => n.string === selected.string)
          : event().notes.length
            ? 0
            : -1;
      digits = "";
      digitTime = 0;
      draw();
      view.select(selected, true);
      return;
    }
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const delta = e.key === "ArrowLeft" ? -1 : 1,
        next = selected.ei + delta;
      if (next < 0 || next >= draft.voices[selected.vi].events.length) {
        const mi = neighbour(delta),
          m = mi !== ui.measureIndex ? ui.state.measures[mi] : null;
        if (m) {
          const vi = Math.min(selected.vi, m.parsed.voices.length - 1),
            events = m.parsed.voices[vi].events;
          const target = measureProfile(m, ui.state),
            ei = delta < 0 ? events.length - 1 : 0;
          const kind =
            target.mode === "both"
              ? selected.kind
              : target.mode === "notation"
                ? "notation"
                : "tab";
          const string = Math.min(selected.string || 1, target.tuning.length),
            notes = events[ei]?.notes || [];
          const ni =
            kind === "tab"
              ? notes.findIndex((n) => n.string === string)
              : notes.length
                ? 0
                : -1;
          selectMeasure(mi, { mi, vi, ei, ni, string, kind }, true);
        }
        return;
      }
      selected.ei = next;
      selected.ni =
        selected.kind === "tab"
          ? event().notes.findIndex((n) => n.string === selected.string)
          : event().notes.length
            ? 0
            : -1;
      digits = "";
      digitTime = 0;
      draw();
      view.select(selected, true);
      return;
    }
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      digits = "";
      digitTime = 0;
      if (e.shiftKey && profile.instrument !== "drums") {
        shiftNote(e.key === "ArrowUp" ? 1 : -1);
        return;
      }
      if (selected.kind === "tab") {
        selected.string = Math.max(
          1,
          Math.min(
            profile.tuning.length,
            selected.string + (e.key === "ArrowUp" ? -1 : 1),
          ),
        );
        selected.ni = event().notes.findIndex(
          (n) => n.string === selected.string,
        );
        draw();
      } else if (profile.instrument !== "drums")
        shiftNote(e.key === "ArrowUp" ? 1 : -1);
      return;
    }
    if (selected.kind === "tab" && /^[0-9xX]$/.test(e.key)) {
      e.preventDefault();
      let merge = Date.now() - digitTime < 800 && !!digits;
      digits = merge ? digits + e.key : e.key;
      digitTime = Date.now();
      if (digits.length > 2 || Number(digits) > 30 || /x/i.test(digits)) {
        digits = e.key;
        merge = false;
      }
      if (!merge) digitUndoRecorded = false;
      const changed = applyValue(digits, merge && digitUndoRecorded, true);
      digitUndoRecorded ||= !!changed;
    } else if (e.key.toLowerCase() === "r") {
      $("makeRest").click();
      e.preventDefault();
    } else if (
      selected.kind === "notation" &&
      profile.instrument !== "drums" &&
      /^[a-g]$/i.test(e.key)
    ) {
      applyValue(e.key.toUpperCase() + $("pitchOctave").value);
      e.preventDefault();
    }
  };
  async function save(reviewed = true) {
    if (saving || ui.busy || !draft) return false;
    digits = "";
    digitTime = 0;
    saving = true;
    setBusy(true);
    feedback();
    try {
      const chosen = event()?.notes[selected?.ni];
      const body = !ui.measureDirty
        ? { reviewed }
        : ui.editorMode === "text"
          ? { target: $("measureText").value.trim(), reviewed }
          : { measure: draft, reviewed };
      receiveProject(
        await api(
          endpoint(`/measures/${ui.measureIndex + 1}`),
          "PUT",
          body,
          ui.state.revision,
        ),
      );
      projectRevision = ui.state.revision;
      draft = clone(current().parsed);
      if (ui.editorMode === "text") {
        undo = [];
        redo = [];
        selected = firstHit();
      } else if (chosen)
        selected.ni =
          event()?.notes.findIndex((n) =>
            profile.mode === "notation"
              ? n.pitch === chosen.pitch
              : n.string === chosen.string,
          ) ?? -1;
      $("measureText").value = current().score_text;
      renderSummary();
      renderReference();
      draw();
      renderExport();
      feedback(reviewed ? "已保存并确认。" : "已保存修改。");
      return true;
    } catch (error) {
      feedback(error.message, true);
      notice(error.message, true);
      return false;
    } finally {
      saving = false;
      setBusy(false);
      updateMeasureControls();
      if (ui.editorMode === "score")
        $("scoreCanvas").focus({ preventScroll: true });
    }
  }
  $("saveMeasure").onclick = () => save();
  $("discardMeasure").onclick = action(async () => {
    if (!confirm("放弃当前小节未保存的修改，载入已保存的内容？")) return;
    setBusy(true);
    try {
      receiveProject(await api(endpoint("")));
      draft = null;
      histories.clear();
      renderMeasures();
      renderExport();
    } finally {
      setBusy(false);
    }
  });
  $("toExport").onclick = async () => {
    if (!ui.measureDirty || (await save(false))) go(4);
  };
  return { renderMeasures, updateMeasureControls, openIssue };
}
