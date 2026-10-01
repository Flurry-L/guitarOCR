import { ui, endpoint, receiveProject } from "./state.js";
import { $, action, notice } from "./dom.js";
import { api } from "./api.js";
import { programOptions } from "./instruments.js";

export function initMetadata({ start, go, render, setBusy }) {
  let activePart = '';
  $("midiProgram").replaceChildren(...programOptions());
  function renderMetadata() {
    const root = ui.state.metadata, parts = root?.parts || [];
    const selected = $("metadataPart").value;
    $("metadataPart").replaceChildren(...parts.map(p=>new Option(p.name,p.id)));
    if (parts.some(p=>p.id===selected)) $("metadataPart").value=selected;
    activePart=$("metadataPart").value;
    $("metadataPart").closest('label').hidden=parts.length<2;
    const m = parts.find(p=>p.id===$("metadataPart").value) || root;
    $("partNameField").hidden = !parts.length;
    $("partName").required = !!parts.length;
    $("partName").value = m?.name || '';
    $("title").value = root?.title || "";
    $("artist").value = root?.artist || "";
    $("instrument").value = m?.instrument || "guitar";
    $("midiProgram").value = String(m?.midi_program ?? ({guitar:25,bass:33,pitched:0,drums:0}[$("instrument").value]));
    $("tempo").value = m?.document_metadata?.tempo_quarter || 120;
    $("capo").value = m?.capo || 0;
    $("transpose").value = m?.transpose ?? "";
    const shifts = [...new Set((m?.document_metadata?.pitch_instructions || [])
      .filter((p) => p.parsed?.kind === "instrument" && Number.isInteger(p.parsed.semitones))
      .map((p) => p.parsed.semitones))];
    $("transpose").placeholder = shifts.length === 1 ? `自动读取：${shifts[0]}` : "自动读取";
    $("tuning").value = (m?.tuning_used || [64, 59, 55, 50, 45, 40]).join(",");
    const preset = Array.from($("tuningPreset").options).some(
      (o) => o.value === $("tuning").value,
    );
    $("tuningPreset").value = preset ? $("tuning").value : "custom";
    $("customTuning").hidden = preset;
    instrumentFields();
    updateMetadataControls();
    const warnings = [...(m?.document_metadata?.warnings || []), m?.document_metadata?.tuning_issue].filter(Boolean);
    $("infoWarnings").hidden = !warnings.length;
    $("infoWarnings").textContent = warnings.join("\n");
  }
  $("metadataPart").onchange = () => {
    if (ui.metadataDirty) {
      $("metadataPart").value=activePart;
      notice('请先保存当前音轨的修改，再切换音轨。');
      return;
    }
    renderMetadata();
  };
  function instrumentFields() {
    const fretted = ["guitar", "bass"].includes($("instrument").value);
    $("tuningField").hidden = !fretted;
    $("capoField").hidden = !fretted;
    $("transposeField").hidden = $("instrument").value === "drums";
    $("midiProgramField").hidden = $("instrument").value === "drums";
    $("customTuning").hidden = !fretted || $("tuningPreset").value !== "custom";
  }
  $("instrument").onchange = () => {
    const tuning =
      $("instrument").value === "bass" ? "43,38,33,28" : "64,59,55,50,45,40";
    $("tuning").value = tuning;
    $("tuningPreset").value = tuning;
    $("midiProgram").value = String({guitar:25,bass:33,pitched:0,drums:0}[$("instrument").value]);
    instrumentFields();
    ui.metadataDirty = true;
    updateMetadataControls();
  };
  $("tuningPreset").onchange = () => {
    $("customTuning").hidden = $("tuningPreset").value !== "custom";
    if ($("tuningPreset").value !== "custom")
      $("tuning").value = $("tuningPreset").value;
    ui.metadataDirty = true;
    updateMetadataControls();
  };
  for (const id of ["title", "artist", "partName", "midiProgram", "tempo", "capo", "tuning", "transpose"])
    $(id).oninput = () => {
      ui.metadataDirty = true;
    updateMetadataControls();
    };
  function updateMetadataControls() {
    $("discardInfo").hidden = !ui.metadataDirty;
    $("discardInfo").disabled = ui.busy;
    $("infoSaveStatus").textContent = ui.metadataDirty
      ? "有未保存的修改。保存后生效，或放弃修改返回已保存内容。"
      : "更改乐器、调弦或移调后需重新识别小节。";
  }
  $("discardInfo").onclick = () => {
    if (ui.busy || !confirm("放弃当前音轨未保存的信息修改？")) return;
    ui.metadataDirty = false;
    renderMetadata();
    notice("已恢复保存的谱面信息。");
  };
  $("readInfo").onclick = action(async () => {
    if (
      (ui.metadataDirty || ui.state.info) &&
      !confirm("自动读取会替换当前信息；调弦改变时需重新识别小节，继续？")
    )
      return;
    await start("/information", undefined, () => {
      ui.metadataDirty = ui.measureDirty = false;
    });
  });
  $("saveInfo").onclick = action(async () => {
    if (!ui.metadataDirty && ui.state.info) {
      if (showPendingTuning()) return;
      go(3);
      return;
    }
    for (const id of ["partName", "tempo", "capo", "transpose"]) {
      if (!$(id).reportValidity()) return;
    }
    const fretted = ["guitar", "bass"].includes($("instrument").value);
    const tuning = fretted
      ? $("tuning")
          .value.split(",")
          .map((v) => Number(v.trim()))
      : [];
    if (
      fretted &&
      (!$("tuning").value.trim() ||
        tuning.length > 12 ||
        tuning.some((v) => !Number.isInteger(v) || v < 0 || v > 127) ||
        $("tuning")
          .value.split(",")
          .some((v) => !v.trim()))
    )
      throw new Error("请填写 1 至 12 个弦的 MIDI 音高（0 至 127），用逗号分隔。");
    const root = ui.state.metadata;
    const previous = root?.parts?.find(p => p.id === $("metadataPart").value) || root;
    const transpose = $("transpose").value.trim() ? +$("transpose").value : null;
    if (ui.state.recognition && (
      previous?.instrument !== $("instrument").value ||
      JSON.stringify(previous?.tuning_used || []) !== JSON.stringify(tuning) ||
      (previous?.transpose ?? null) !== transpose
    ) && !confirm("更改乐器、调弦或记谱移调后，现有识别和人工校对结果会失效，需要重新识别。仍要保存？")) return;
    const navigation = ui.navigation;
    setBusy(true);
    try {
      const saved = await api(endpoint("/metadata"), "PUT", {
        part_id: $("metadataPart").value || null,
        part_name: $("partNameField").hidden ? null : $("partName").value.trim(),
        title: $("title").value || "未命名乐谱",
        artist: $("artist").value,
        instrument: $("instrument").value,
        midi_program: $("instrument").value === 'drums' ? 0 : +$("midiProgram").value,
        tempo_quarter: +$("tempo").value,
        capo: fretted ? +$("capo").value : 0,
        tuning_used: tuning,
        transpose: $("transpose").value.trim() ? +$("transpose").value : null,
      }, ui.state.revision);
      receiveProject(saved);
      render();
    } finally {
      setBusy(false);
    }
    if (ui.navigation === navigation && showPendingTuning()) return;
    go(3, ui.navigation === navigation);
    if (ui.navigation === navigation)
      notice(ui.state.recognition ? "谱面信息已保存，音符修改已保留。" : "谱面信息已保存，下一步识别音符与节奏。");
  });
  function showPendingTuning() {
    const root = ui.state.metadata;
    const pending = (root?.parts?.length ? root.parts : [root]).find(p => p?.document_metadata?.tuning_issue);
    if (!pending) return false;
    if (pending.id) $("metadataPart").value = pending.id;
    renderMetadata();
    go(2);
    $("tuningPreset").focus();
    notice(`请先确认${pending.name ? `“${pending.name}”的` : ""}调弦，再继续识别。`);
    return true;
  }
  return { renderMetadata, updateMetadataControls };
}
