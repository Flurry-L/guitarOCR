import { ui, endpoint, receiveProject, commitMeasureDraft, projectTitle } from "./state.js";
import { $, el, action, notice } from "./dom.js";
import { api } from "./api.js";
import { editorMenu } from "./editor-menu.js";
import { scoreView } from "./score-view.js";
import { measureProfile, reviewKind } from "./score-engraving.js";
import { durations, durationName } from "./score-theory.js";
import { dynamicValue } from "./score-effects.js";
import { measureContext, musicalContext } from "./score-structure.js";
import { measureProperties, effectProperties } from "./properties.js";
import { chordEditor, unresolvedChordChoices } from "./score-chords.js";
import {
  clone,
  visibleEvent,
  setVisibleEvent,
  retainUnchangedVisualEvents,
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
  staffPositionChoices,
  staffPositionLabel,
  beatStartChoices,
  beatStartLabel,
  summarizeMText,
} from "./score-model.js";
import { playback } from "./audio-preview.js";
import { initPlaybackControls } from "./playback-controls.js";

export function initMeasures({ start, go, renderExport, exportScore, setBusy }) {
  let draft,
    observationDraft,
    visualTechniques,
    selected,
    profile,
    undo = [],
    redo = [],
    saving = false,
    digits = "",
    digitTime = 0,
    digitUndoRecorded = false;
  let projectGeneration,
    projectRevision,
    draftRevision,
    track = "all";
  let selectionScope = "measure";
  let sourceUrl, sourceRequest = 0, sourceTimer;
  const properties = measureProperties({ getDraft: () => draft, getProfile: () => profile, getEvent: () => event(), change });
  const effects = effectProperties({ getNote: () => event()?.notes[selected?.ni], getEvent: () => event(), getProfile: () => profile, change });
  function loadSource(url) {
    sourceUrl = url;
    const request = ++sourceRequest;
    clearTimeout(sourceTimer);
    $("measureImage").hidden = true;
    $("sourceImageStatus").hidden = false;
    $("sourceImageStatus").textContent = "正在加载原谱…";
    const image = new Image();
    const failed = () => {
      if (request !== sourceRequest) return;
      clearTimeout(sourceTimer);
      image.onload = image.onerror = null;
      $("sourceImageStatus").textContent = "原谱未能加载，点击重试。";
      $("sourceZoom").setAttribute("aria-label", "重新加载原谱");
    };
    image.onload = () => {
      if (request !== sourceRequest) return;
      clearTimeout(sourceTimer);
      $("measureImage").src = url;
      $("measureImage").hidden = false;
      $("sourceImageStatus").hidden = true;
      $("sourceZoom").setAttribute("aria-label", "放大原谱");
    };
    image.onerror = failed;
    sourceTimer = setTimeout(failed, 20000);
    image.src = url;
  }
  const histories = new Map();
  const draftKey = () => `guitarocr-draft:${ui.sid}`;
  function persistDraft() {
    if (!draft || !ui.sid) return;
    try {
      if (ui.measureDirty) sessionStorage.setItem(draftKey(), JSON.stringify({
        index: ui.measureIndex, revision: draftRevision, draft, observationDraft, visualTechniques, selected,
        mode: ui.editorMode, text: $("measureText").value, track,
      }));
      else sessionStorage.removeItem(draftKey());
    } catch { /* Saving to the server remains available in restricted browsers. */ }
  }

  function rememberContext() {
    if (!ui.sid || !draft) return;
    const viewport = document.querySelector(".score-viewport");
    try { sessionStorage.setItem(`guitarocr-editor:${ui.sid}`, JSON.stringify({
      index: ui.measureIndex, selected, track, mode: ui.editorMode,
      zoom: $("scoreZoom").value, inspector: !studio.classList.contains("hide-inspector"),
      map: $("measureMapPanel")?.open || false,
      scroll: { top: viewport.scrollTop, left: viewport.scrollLeft },
    })); } catch {}
  }
  window.addEventListener("beforeunload", rememberContext);
  window.addEventListener("guitarocr:leave-project", rememberContext);
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
        ".command-toolbar button,.playback-toolbar button,.effect-tools button,#fretButtons button,#pitchButtons button",
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
      if (await selectMeasure(hit.mi, hit) && !hit.preview) menu.open(x, y);
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
    onWarning: (message) => {
      $("scoreWarning").hidden = !message;
      $("scoreWarning").textContent = message;
    },
    onError: (message) => {
      $("scoreWarning").hidden = false;
      $("scoreWarning").textContent =
        "谱面暂时无法绘制，可在右侧修改小节文本。";
      console.warn("Score rendering failed", message);
    },
  });
  const current = () => ui.state?.measures?.[ui.measureIndex];
  const transport = initPlaybackControls({
    prepare: async () => {
      if (!draft || ui.busy || saving) return null;
      const navigation = ui.navigation;
      if ((ui.editorMode === "text" && ui.measureDirty || observationChanged()) && !(await save())) return null;
      if (ui.navigation !== navigation || ui.step !== 3) return null;
      return { state: ui.state, index: ui.measureIndex, measure: draft, visualTechniques };
    },
    onPosition: position => view.playback(position),
    onError: message => feedback(message, true),
  });
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
  const timeSignature = () => measureContext(ui.state, ui.measureIndex, draft).time;
  const chords = chordEditor(change, event, () => profile?.tuning.length);
  function feedback(message = "", error = false) {
    $("measureFeedback").textContent = message;
    $("measureFeedback").hidden = !message;
    $("measureFeedback").classList.toggle("error", error);
  }
  function snapshot() {
    return { draft: clone(draft), visualTechniques: clone(visualTechniques), selected: clone(selected) };
  }
  const observationChanged = () => JSON.stringify(observationDraft) !== JSON.stringify(current()?.observation);
  const invalidObservationInput = () => [...$("observationFields").querySelectorAll("input,select")].find(input => !input.validity.valid);
  function change(fn, { merge = false, typed = false } = {}) {
    if (ui.busy || saving || !draft || ui.editorMode !== "score") return;
    if (current()?.parsed.incomplete) {
      inspector(true);
      $("incompleteObservation").open = true;
      feedback("这是已确认内容的预览，请在识别观测中纠正未决内容。");
      return;
    }
    if (observationChanged() || invalidObservationInput()) {
      feedback("请先保存补全的识别内容，再编辑谱面。");
      return;
    }
    if (!typed) {
      digits = "";
      digitTime = 0;
    }
    selectionScope = "note";
    const before = snapshot();
    try {
      fn();
    } catch (error) {
      draft = before.draft;
      visualTechniques = before.visualTechniques;
      selected = before.selected;
      feedback(error.message, true);
      return;
    }
    visualTechniques = retainUnchangedVisualEvents(visualTechniques, before.draft, draft);
    if (JSON.stringify(before.draft) === JSON.stringify(draft) &&
        JSON.stringify(before.visualTechniques) === JSON.stringify(visualTechniques)) return;
    if (!merge || !undo.length) undo.push(before);
    if (undo.length > 100) undo.shift();
    redo = [];
    updateDirty();
    feedback();
    draw();
    return true;
  }
  function updateDirty() {
    playback.stop();
    ui.measureDirty =
      (ui.editorMode === "text"
        ? $("measureText").value.trim() !== current()?.score_text?.trim()
        : JSON.stringify(draft) !== JSON.stringify(current()?.parsed)) ||
      JSON.stringify(visualTechniques ?? null) !== JSON.stringify(current()?.visual_techniques ?? null) ||
      observationChanged() || !!invalidObservationInput();
    persistDraft();
  }
  function restore(from, to) {
    if (!from.length || ui.busy || saving || ui.editorMode !== "score" || observationChanged() || invalidObservationInput()) return;
    digits = "";
    digitTime = 0;
    to.push(snapshot());
    const before = from.pop();
    draft = before.draft;
    visualTechniques = before.visualTechniques;
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
      !confirm(`重新识别整份乐谱（${ui.state.measures.length} 个谱表小节）将替换所有人工编辑内容${ui.measureDirty ? "和未保存草稿" : ""}。失败或停止会保留原结果。继续？`)
    )
      return;
    await start("/recognize", undefined, () => {
      ui.measureDirty = false;
    });
  });
  $("resumeOcr").onclick = action(async () => {
    if (ui.measureDirty && !confirm("继续会从中断记录恢复，成功会替换当前未保存草稿；失败或取消会保留草稿。继续？")) return;
    await start("/recognize", { resume: true });
  });
  $("retryIssues").onclick = action(async () => {
    const targets = [...ui.state.review_measures];
    if (!confirm(`重新识别 ${targets.length} 个有问题的谱表小节，将替换这些小节的人工修改，并重新连接相邻小节。失败或停止会保留原结果。继续？`)) return;
    if (ui.measureDirty && !targets.includes(ui.measureIndex + 1) && !(await save())) return;
    await start("/recognize", { measures: targets });
  });
  $("retryMeasure").onclick = action(async () => {
    if (!confirm(`重新识别「${measureLabel(current(), ui.measureIndex)}」将替换该小节人工修改${ui.measureDirty ? "和未保存草稿" : ""}，并重新连接相邻小节。失败或停止会保留原结果。继续？`)) return;
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
    const visible = inTrack.filter(i => filter === "all" || measures[i].needs_review);
    const issues = inTrack.filter(i => measures[i].needs_review).length;
    $("trackSummary").textContent =
      `${inTrack.length} 个谱表小节${issues ? ` · ${issues} 个有问题` : ""}`;
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
                `${reviewKind(m)} ${i === ui.measureIndex ? "active" : ""}`,
              );
            button.title =
              measureLabel(m, i) +
              (m.needs_review ? " · 有问题" : "");
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
    if (ui.measureDirty && !(await save())) {
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
    selectionScope = "track";
    renderNavigator();
    draw();
  };
  $("reviewFilter").onchange = renderNavigator;
  $("trackProperties")?.querySelector("button").addEventListener("click", () => {
    if (ui.measureDirty || ui.metadataDirty || ui.libraryDirty) return;
    const part = $("metadataPart");
    if ([...part.options].some((option) => option.value === current()?.part_id)) {
      part.value = current().part_id;
      part.dispatchEvent(new Event("change"));
    }
  });
  function renderSummary() {
    const measures = ui.state.measures || [];
    $("resumeOcr").hidden = !ui.state.ocr_task;
    $("retryIssues").hidden = !ui.state.review_measures?.length;
    $("ocrEmpty").hidden = !!measures.length;
    $("reviewEditor").hidden = !measures.length;
    $("recognize").textContent = measures.length ? "重新识别整谱" : "识别乐谱";
    $("recognize").classList.toggle("primary", !measures.length);
    if (!measures.length) $("recognize").closest("details").open = true;
    const bars = new Set(measures.map((m, i) => m.bar_index ?? i)).size,
      parts = new Set(measures.map((m) => m.part_id || "part-1")).size;
    const issues = new Set(measures.flatMap((m, i) => m.needs_review ? [m.bar_index ?? i] : [])).size;
    $("recognitionSummary").textContent =
      `${bars} 小节${parts > 1 ? ` · ${parts} 音轨` : ""}${issues ? ` · ${issues} 个小节有问题` : ""}`;
    $("scoreTitle").textContent =
      projectTitle(ui.state);
    $("scoreCredits").textContent = ui.state.metadata?.artist || "";
    $("measureSelect").replaceChildren(
      ...measures.map((m, i) => {
        const o = el(
          "option",
          `${measureLabel(m, i)}${m.needs_review ? " · 有问题" : ""}`,
        );
        o.value = i;
        return o;
      }),
    );
    $("measureSelect").value = ui.measureIndex;
    renderNavigator();
  }
  function renderMeasures() {
    const opening = projectGeneration !== ui.openGeneration;
    let context;
    if (opening) {
      try { context = JSON.parse(sessionStorage.getItem(`guitarocr-editor:${ui.sid}`) || "null"); } catch {}
    }
    if (opening || (projectRevision !== ui.state.revision && !ui.measureDirty)) {
      // Assembly can update neighbouring bars, so a new server snapshot
      // invalidates undo histories based on the previous musical content.
      histories.clear();
      draft = null;
      if (!opening) {
        try { sessionStorage.removeItem(draftKey()); } catch {}
      }
      if (opening) {
        track = "all";
        $("reviewFilter").value = "all";
      }
      projectGeneration = ui.openGeneration;
      projectRevision = ui.state.revision;
    }
    if (context) {
      ui.measureIndex = context.index;
      track = context.track;
      ui.editorMode = context.mode;
      $("scoreZoom").value = context.zoom;
      view.zoom(+context.zoom);
      inspector(context.inspector);
      if ($("measureMapPanel")) $("measureMapPanel").open = context.map;
    }
    renderSummary();
    if (!ui.state.measures?.length) {
      draft = null;
      updateMeasureControls();
      return;
    }
    ui.measureIndex = Math.min(ui.measureIndex, ui.state.measures.length - 1);
    if (!draft) {
      loadMeasure();
      if (context?.selected) selected = context.selected;
      try {
        const cached = JSON.parse(sessionStorage.getItem(draftKey()) || "null");
        if (opening && cached && ui.state.measures[cached.index]) {
          ui.measureIndex = cached.index;
          loadMeasure();
          draft = cached.draft;
          observationDraft = cached.observationDraft;
          renderObservation();
          visualTechniques = cached.visualTechniques;
          selected = cached.selected;
          draftRevision = cached.revision;
          ui.editorMode = cached.mode;
          track = cached.track;
          $("measureText").value = cached.text;
          ui.measureDirty = true;
          editorMode();
          renderSummary();
          renderReference();
          feedback(cached.revision === ui.state.revision
            ? "已恢复未保存草稿。"
            : "已恢复草稿，但项目已有新版本。请先复制需要的内容，再放弃草稿载入新版本；不会覆盖新版本。", cached.revision !== ui.state.revision);
        }
      } catch { /* Ignore incomplete browser storage. */ }
    }
    draw();
    if (context?.scroll) view.restoreViewport(context.scroll);
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
      if (ui.measureDirty && !(await save())) {
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
    if (hit?.preview) {
      selected = firstHit();
      inspector(true);
      ui.editorMode = "score";
      editorMode();
    } else if (hit) selected = { ...hit };
    selectionScope = !hit?.preview && hit?.ni >= 0 ? "note" : "measure";
    renderReference();
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
      m.needs_review && (track === "all" || (m.part_id || "part-1") === track) ? [i] : [],
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
    playback.stop();
    profile = measureProfile(current());
    draft = clone(current().parsed);
    observationDraft = clone(current().observation);
    renderObservation();
    visualTechniques = clone(current().visual_techniques ?? null);
    draftRevision = ui.state.revision;
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
    if (sourceUrl !== m.url) loadSource(m.url);
    $("reviewStatus").textContent = m.needs_review ? "检测到问题" : "识别完成";
    $("reviewStatus").className = `pill ${reviewKind(m)}`;
    $("reviewHint").hidden = !m.needs_review;
    $("reviewHint").textContent = (m.issues || []).map(issue => issue.message).join("\n");
  }
  function renderObservation() {
    const panel = $("incompleteObservation"), fields = $("observationFields");
    panel.hidden = !current().parsed.incomplete || !observationDraft;
    fields.replaceChildren();
    if (panel.hidden) return;
    panel.open = false;
    const meter = observationDraft.state?.time?.value || current()?.score_state?.time || ui.state.metadata?.time_signature || "4/4";
    const field = (parent, label, object, key, { choices, text = false, numeric = !choices,
      custom = false, formatValue = String, placeholder = "" } = {}) => {
      const wrapper = el("label", label);
      const input = el(choices ? "select" : "input");
      const customInput = custom ? el("input") : null;
      const customHint = custom ? el("small", "", "muted") : null;
      if (choices) {
        input.replaceChildren(new Option("未知", ""), ...choices.map(([value, name]) => new Option(name, value)));
        if (object[key] != null && !choices.some(([value]) => String(value) === String(object[key]))) {
          input.append(new Option(formatValue(object[key]), object[key]));
        }
        if (custom) input.append(new Option("自定义…", "custom"));
      } else {
        input.type = text ? "text" : "number";
        input.step = "1";
      }
      input.value = object[key] == null ? "" : String(object[key]);
      input.setAttribute("aria-label", label);
      const update = (control) => {
        const raw = control.value.trim();
        let value = raw || null;
        if (raw && numeric && raw !== "unknown" && raw.toLowerCase() !== "x") value = Number(raw);
        if (text && raw.toLowerCase() === "x") value = "x";
        control.setCustomValidity("");
        if (typeof value === "number" && (!Number.isSafeInteger(value) || ["start", "fret"].includes(key) && value < 0))
          control.setCustomValidity("请输入有效整数；起点和品位从 0 开始。");
        if (control.validity.valid) object[key] = value;
        if (customHint) customHint.textContent = control.validity.valid && value !== null ? formatValue(value) : "";
        updateDirty();
        updateMeasureControls();
      };
      input.oninput = () => {
        if (custom && input.value === "custom") {
          customInput.value = object[key] ?? "";
          customInput.setCustomValidity("");
          customHint.textContent = object[key] == null ? "" : formatValue(object[key]);
          wrapper.append(customInput, customHint);
          customInput.focus();
          return;
        }
        customInput?.remove();
        customHint?.remove();
        update(input);
      };
      if (customInput) {
        customInput.type = "number";
        customInput.step = "1";
        customInput.placeholder = placeholder;
        customInput.setAttribute("aria-label", key === "start" ? "自定义起点（tick）" : "自定义谱位");
        customInput.oninput = () => update(customInput);
      }
      wrapper.append(input);
      parent.append(wrapper);
    };
    const clef = observationDraft.state?.clef?.value || current()?.score_state?.clef || profile.pitch_context?.clef || (profile.instrument === "bass" ? "F4" : "G2");
    const startChoices = beatStartChoices(meter);
    const posChoices = staffPositionChoices(clef, profile.instrument);
    const choices = durations.flatMap(([v, , name]) => [[v, name], [v + ".", name + "附点"], [v + "[3:2]", name + "三连音"]]);
    for (const voice of observationDraft.voices || []) for (const [ei, event] of voice.events.entries()) {
      const group = el("fieldset", undefined, "measure-property-fields"),
        title = el("legend", `声部 ${voice.voice + 1} · 第 ${ei + 1} 个${event.status === "rest" ? "休止" : "音符"}`);
      group.append(title);
      field(group, "节拍位置（起点）", event, "start", {choices: startChoices, numeric: true, custom: true,
        formatValue: value => beatStartLabel(value, meter), placeholder: "输入 tick（四分音符 = 960）"});
      field(group, "时值", event, "duration", {choices});
      if (event.status == null) field(group, "类型", event, "status", {choices: [["normal", "音符"], ["rest", "休止"]]});
      const representations = profile.mode === "both"
        ? [["TAB · ", event.notes, false], ["五线谱 · ", event.notation_notes, true]]
        : [["", event.notes, profile.mode === "notation"]];
      for (const [label, notes, notation] of representations) for (const [ni, note] of (notes || []).entries()) {
        const prefix = label + (notes.length > 1 ? `音 ${ni + 1} · ` : "");
        if (!notation) {
          field(group, prefix + "弦", note, "string");
          field(group, prefix + "品位（闷音填 X）", note, "fret", {text: true});
        } else {
          field(group, prefix + (profile.instrument === "drums" ? "五线谱鼓位" : "五线谱音高 / 线间位置"), note, "staff_position", {choices: posChoices, numeric: true, custom: true,
            formatValue: value => staffPositionLabel(value, clef, profile.instrument), placeholder: "0 为第一线，每线或间加 1"});
          field(group, prefix + "临时升降号（留空表示未打印）", note, "accidental", {numeric: true, choices: [["unknown", "看不清"], ["0", "还原"], ["1", "升"], ["-1", "降"], ["2", "重升"], ["-2", "重降"]]});
          if (profile.instrument === "drums") field(group, prefix + "音头形状", note, "notehead", {
            choices: [["normal", "普通"], ["x", "叉形"], ["circle-x", "圆圈叉形"], ["circle-slash", "圆圈斜线"],
              ["diamond", "菱形"], ["heavy-x", "粗叉形"], ["heavy-x-hat", "带帽粗叉形"], ["triangle-up", "三角形"]],
          });
        }
      }
      fields.append(group);
    }
  }
  function updateMeasureControls() {
    const count = ui.state?.measures?.length || 0,
      busy = ui.busy || saving,
      editingObservation = observationChanged() || !!invalidObservationInput();
    if (draft) {
      properties.render(busy || ui.editorMode !== "score");
      effects.render(busy || ui.editorMode !== "score");
    }
    for (const id of ["visiblePalmMute", "visibleLetRing"])
      $(id).disabled = busy || !event()?.notes?.length || ui.editorMode !== "score";
    $("prevMeasure").disabled =
      busy || !count || neighbour(-1) === ui.measureIndex;
    $("nextMeasure").disabled =
      busy || !count || neighbour(1) === ui.measureIndex;
    $("nextIssue").hidden = !ui.state?.review_measures?.length;
    $("nextIssue").disabled = busy || !(ui.state?.measures || []).some((m) =>
      m.needs_review && (track === "all" || (m.part_id || "part-1") === track));
    $("toExport").disabled = busy || !count;
    $("finishEditing").disabled = busy || !count;
    transport.update(!busy && !!count && !!draft);
    $("finishEditing").textContent = $("toExport").textContent = ui.measureDirty ? "保存并导出 GP5" : "直接导出 GP5";
    $("trackSelect").disabled = busy || !count;
    $("reviewFilter").disabled = busy || !count;
    $("saveMeasure").disabled = busy || !ui.measureDirty;
    $("saveMeasure").textContent = "保存";
    $("observationFields").querySelectorAll("input,select").forEach(input => {
      input.disabled = busy || ui.editorMode === "text" || JSON.stringify(draft) !== JSON.stringify(current()?.parsed)
        || $("measureText").value.trim() !== (current()?.score_text || "").trim();
    });
    $("measureText").disabled = busy || editingObservation;
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
    $("undoNote").disabled = busy || !undo.length || ui.editorMode === "text" || editingObservation;
    $("redoNote").disabled = busy || !redo.length || ui.editorMode === "text" || editingObservation;
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
    $("eventStart").disabled ||= !event();
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
  function updateMTextSummary() {
    $("mTextSummary").textContent = summarizeMText($("measureText").value, {
      instrument: profile?.instrument,
      timeSignature: current()?.score_state?.time || ui.state.metadata?.time_signature || "4/4",
    });
  }
  function editorMode() {
    $("editMode").value = ui.editorMode;
    $("textEditor").hidden = ui.editorMode !== "text";
    if (ui.editorMode === "text") updateMTextSummary();
  }
  $("editMode").onchange = async () => {
    const next = $("editMode").value;
    if (ui.measureDirty && !(await save())) {
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
    selected.vi = Math.max(0, Math.min(selected.vi, draft.voices.length - 1));
    selected.ei = Math.max(0, Math.min(selected.ei, draft.voices[selected.vi]?.events.length - 1));
    if (selected.ni >= (event()?.notes.length || 0)) selected.ni = -1;
    $("timeSignature").value = draft.time_signature || "";
    $("measureTempo").value = draft.tempo_quarter || "";
    const headerNotes = ui.state.metadata?.document_metadata?.header_notes || "";
    $("scoreHeaderNotes").textContent = headerNotes;
    $("scoreHeaderNotes").hidden = !headerNotes;
    $("scoreWarning").hidden = true;
    view.render(ui.state, { index: ui.measureIndex, measure: draft, visualTechniques }, track);
    view.select(selected);
    const e = event(),
      n = e?.notes[selected?.ni];
    const visible = visibleEvent(visualTechniques, draft.voices[selected.vi]?.voice, e?.start);
    const visibleMarks = visible?.status === "classified" ? visible.effects :
      [...(e?.effects || []), ...(e?.notes || []).flatMap(note => note.effects || [])];
    $("visiblePalmMute").checked = visibleMarks?.includes("pm") || false;
    $("visibleLetRing").checked = visibleMarks?.includes("let") || false;
    $("visibleTechniqueStatus").textContent = visible?.status === "unresolved"
      ? "印刷范围待核对，当前显示原有奏法。"
      : visible?.status === "classified"
        ? "按当前拍的印刷范围显示。保留已知音符归属；未确定归属的和弦技巧以拍文字导出。"
        : "当前按音符奏法显示；可在此校对整拍的印刷范围。";
    const scopeTitle = $("inspectorScopeTitle");
    if (scopeTitle) scopeTitle.textContent = selectionScope === "note" && n ? "音符属性" : "当前拍 / 小节";
    document.querySelector(".note-properties").hidden = selectionScope === "track";
    document.querySelector(".effect-tools").hidden = selectionScope !== "note" || !n;
    const trackProperties = $("trackProperties");
    if (trackProperties) {
      trackProperties.hidden = selectionScope !== "track";
      $("trackPropertiesText").textContent = `${current().part_name || "音轨"} · ${({guitar:"吉他",bass:"贝斯",pitched:"旋律乐器",drums:"打击乐器"})[profile.instrument] || profile.instrument} · ${({tab:"TAB",notation:"五线谱",both:"五线谱 + TAB"})[profile.mode] || profile.mode}${profile.tuning?.length ? ` · ${profile.tuning.length} 弦` : ""}`;
    }
    const context = musicalContext(ui.state, ui.measureIndex, draft, selected, profile);
    for (const [key, value] of Object.entries(context)) $("context" + key[0].toUpperCase() + key.slice(1)).textContent = value;
    const dynamic = e?.effects?.find(effect => effect.startsWith("dyn:"));
    $("beatDynamic").value = dynamic ? String((dynamicValue(Number(dynamic.slice(4))) + 1) * 16 - 1) : "";
    $("beatDynamic").disabled = ui.busy || saving || !e || ui.editorMode !== "score";
    const choices = unresolvedChordChoices(current(), draft.voices[selected.vi]?.voice, e);
    $("chordChoiceHint").hidden = !choices.length;
    $("chordChoiceHint").textContent = choices.map(choice => `${choice.name || "和弦"}：${choice.reason || "有多种指法，请从指法库选择当前拍使用的图示。"}`).join(" ");
    chords.render(e);
    chords.library(ui.state, current()?.part_id || "part-1");
    const signature = timeSignature();
    $("eventStart").value = e ? e.start / 960 : "";
    updateEventStartHint();
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
      $("selectedNote").textContent =
        `${measureLabel(current(), ui.measureIndex)} · 声部 ${draft.voices[selected.vi].voice + 1} · ${beatStartLabel(e.start, signature)}${selected.kind === "tab" ? ` · 第 ${selected.string} 弦` : ""}${n ? "" : " · 休止或空位"}`;
    } else {
      $("selectedNote").textContent = `${measureLabel(current(), ui.measureIndex)} · 声部 ${draft.voices[selected.vi].voice + 1}`;
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
      ? `${durationName(e.duration.value)}${e.duration.double_dotted ? " · 双附点" : e.duration.dotted ? " · 附点" : ""}${e.notes.length > 1 ? ` · ${e.notes.length} ${drums ? "个鼓音" : "音和弦"}` : ""}`
      : "";
    const [beats, unit] = String(signature).split("/").map(Number);
    const beatTicks = 3840 / unit, expected = beats * beatTicks;
    const voiceEvents = draft.voices[selected.vi]?.events || [];
    const used = Math.max(0, ...voiceEvents.map((e) => e.start + ticks(e)));
    const over = used > expected;
    $("rhythmStatus").textContent = Number.isFinite(expected)
      ? `${signature} · 本声部 ${Number((used / beatTicks).toFixed(3))} / ${beats} 拍${over ? " · 超出小节" : used < expected ? " · 尚有空拍" : " · 时值完整"}`
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
  function updateEventStartHint() {
    const value = $("eventStart").valueAsNumber;
    const tick = Math.round(value * 960);
    $("eventStartHint").textContent = Number.isFinite(tick) && tick >= 0
      ? `${beatStartLabel(tick, timeSignature())} · ${tick} tick（四分音符 = 960 tick）`
      : "0 为小节开头；1 表示经过一个四分音符。";
  }
  $("eventStart").oninput = updateEventStartHint;
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
      if (!$("measureTempo").validity.valid) throw new Error("速度须为 1 至 999 的整数。");
      draft.tempo_quarter = $("measureTempo").value
        ? +$("measureTempo").value
        : null;
    });
  $("measureText").oninput = () => {
    updateMTextSummary();
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
  $("beatDynamic").onchange = () => change(() => {
    const e = event();
    if (!e) return;
    e.effects = (e.effects || []).filter(effect => !effect.startsWith("dyn:"));
    if ($("beatDynamic").value) e.effects.push(`dyn:${$("beatDynamic").value}`);
  });
  for (const id of ["visiblePalmMute", "visibleLetRing"]) $(id).onchange = () => change(() => {
    const e = event();
    if (!e?.notes?.length) return;
    const effects = [$("visiblePalmMute").checked ? "pm" : null, $("visibleLetRing").checked ? "let" : null].filter(Boolean);
    visualTechniques = setVisibleEvent(visualTechniques, draft.voices[selected.vi].voice, e.start, effects);
  });
  $("contextMeter").onclick = () => {
    inspector(true);
    $("measureSettingsSummary").parentElement.open = true;
    $("timeSignature").focus();
  };
  $("contextTrack").onclick = () => {
    selectionScope = "track";
    inspector(true);
    draw();
  };
  $("scoreZoom").onchange = () => view.zoom(+$("scoreZoom").value);
  $("sourceZoom").onclick = () => {
    if ($("measureImage").hidden) { loadSource(current().url); return; }
    $("largeSource").src = current().url;
    $("sourceDialog").showModal();
  };
  $("showSource").onclick = () => $("sourceZoom").click();
  $("closeSource").onclick = () => $("sourceDialog").close();
  document.addEventListener("keydown", (e) => {
    if (ui.step !== 3 || document.querySelector('[data-screen="workbench"]').hidden) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      if (!ui.busy && !saving) save();
    }
  });
  $("reviewEditor").addEventListener("keydown", (e) => {
    if (ui.busy || saving) return;
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
          const target = measureProfile(m),
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
  async function save() {
    if (saving || ui.busy || !draft) return false;
    const invalid = invalidObservationInput();
    if (invalid) { invalid.reportValidity(); invalid.focus(); return false; }
    if (!ui.measureDirty) {
      feedback("修改已保存。");
      return true;
    }
    const sid = ui.sid;
    digits = "";
    digitTime = 0;
    saving = true;
    setBusy(true);
    feedback();
    try {
      const chosen = event()?.notes[selected?.ni];
      const changedObservation = observationChanged(), wasIncomplete = !!current().parsed.incomplete;
      if (wasIncomplete && !changedObservation && ui.editorMode !== "text")
        throw new Error("请在识别观测中纠正未决内容；预览不会替代完整小节。");
      const music = clone(draft);
      for (const key of ["incomplete", "observation", "resolved", "unresolved"]) delete music[key];
      const body = changedObservation
        ? { observation: observationDraft }
        : ui.editorMode === "text"
          ? { target: $("measureText").value.trim() }
          : { measure: music, visual_techniques: visualTechniques };
      if (!(await commitMeasureDraft({
        sid, index: ui.measureIndex, revision: draftRevision, body,
      }, api))) return false;
      draftRevision = ui.state.revision;
      persistDraft();
      projectRevision = ui.state.revision;
      histories.clear();
      profile = measureProfile(current());
      draft = clone(current().parsed);
      observationDraft = clone(current().observation);
      renderObservation();
      visualTechniques = clone(current().visual_techniques ?? null);
      if (ui.editorMode === "text" || changedObservation || wasIncomplete) {
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
      updateMTextSummary();
      renderSummary();
      renderReference();
      draw();
      renderExport();
      feedback(current().parsed.incomplete ? "已保存；还有内容需要补全。" : changedObservation ? "已保存并重新组谱。" : "已保存修改。");
      return true;
    } catch (error) {
      if (ui.sid === sid) {
        feedback(error.message, true);
        notice(error.message, true);
      }
      return false;
    } finally {
      saving = false;
      if (ui.sid === sid) {
        setBusy(false);
        updateMeasureControls();
        if (ui.editorMode === "score")
          $("scoreCanvas").focus({ preventScroll: true });
      }
    }
  }
  $("saveMeasure").onclick = save;
  $("discardMeasure").onclick = action(async () => {
    if (!confirm("放弃当前小节未保存的修改，载入已保存的内容？")) return;
    setBusy(true);
    try {
      receiveProject(await api(endpoint("")));
      sessionStorage.removeItem(draftKey());
      draft = null;
      histories.clear();
      renderMeasures();
      renderExport();
    } finally {
      setBusy(false);
    }
  });
  $("finishEditing").onclick = $("toExport").onclick = action(async () => {
    const navigation = ui.navigation;
    if (ui.measureDirty && !(await save())) return;
    if (ui.navigation !== navigation) return;
    go(4);
    if (ui.step === 4) await exportScore();
  });
  return { renderMeasures, updateMeasureControls, openIssue };
}
