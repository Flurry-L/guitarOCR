import { ui, endpoint, receiveProject } from "./state.js";
import { $, action, notice } from "./dom.js";
import { api } from "./api.js";

export function initMetadata({ start, go, render, setBusy }) {
  function renderMetadata() {
    const m = ui.state.metadata;
    $("title").value = m?.title || "";
    $("artist").value = m?.artist || "";
    $("instrument").value = m?.instrument || "guitar";
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
    const warnings = m?.document_metadata?.warnings || [];
    $("infoWarnings").hidden = !warnings.length;
    $("infoWarnings").textContent = warnings.join("\n");
  }
  function instrumentFields() {
    const fretted = ["guitar", "bass"].includes($("instrument").value);
    $("tuningField").hidden = !fretted;
    $("capoField").hidden = !fretted;
    $("transposeField").hidden = $("instrument").value === "drums";
    $("customTuning").hidden = !fretted || $("tuningPreset").value !== "custom";
  }
  $("instrument").onchange = () => {
    const tuning =
      $("instrument").value === "bass" ? "43,38,33,28" : "64,59,55,50,45,40";
    $("tuning").value = tuning;
    $("tuningPreset").value = tuning;
    instrumentFields();
    ui.metadataDirty = true;
  };
  $("tuningPreset").onchange = () => {
    $("customTuning").hidden = $("tuningPreset").value !== "custom";
    if ($("tuningPreset").value !== "custom")
      $("tuning").value = $("tuningPreset").value;
    ui.metadataDirty = true;
  };
  for (const id of ["title", "artist", "tempo", "capo", "tuning", "transpose"])
    $(id).oninput = () => {
      ui.metadataDirty = true;
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
      go(3);
      return;
    }
    for (const id of ["tempo", "capo", "transpose"]) {
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
        tuning.length > 7 ||
        tuning.some((v) => !Number.isInteger(v) || v < 0 || v > 127) ||
        $("tuning")
          .value.split(",")
          .some((v) => !v.trim()))
    )
      throw new Error("请填写 1 至 7 个弦的 MIDI 音高（0 至 127），用逗号分隔。");
    setBusy(true);
    try {
      const saved = await api(endpoint("/metadata"), "PUT", {
        title: $("title").value || "未命名乐谱",
        artist: $("artist").value,
        instrument: $("instrument").value,
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
    notice("谱面信息已保存。");
    go(3);
  });
  return { renderMetadata };
}
