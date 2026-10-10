import { ui, endpoint, receiveProject } from "./state.js";
import { $, action, notice } from "./dom.js";
import { api } from "./api.js";
import { hasPendingChordCandidate, initChordLibraryReview } from "./score-chords.js";

const instrumentPrograms = [
  ['钢琴', ['大钢琴','明亮钢琴','电声三角钢琴','酒吧钢琴','电钢琴 1','电钢琴 2','羽管键琴','击弦古钢琴']],
  ['有调打击乐', ['钢片琴','钟琴','音乐盒','颤音琴','马林巴','木琴','管钟','杜尔西默琴']],
  ['风琴', ['拉杆风琴','打击风琴','摇滚风琴','教堂管风琴','簧风琴','手风琴','口琴','探戈手风琴']],
  ['吉他', ['尼龙弦吉他','钢弦吉他','爵士电吉他','清音电吉他','闷音电吉他','过载电吉他','失真电吉他','吉他泛音']],
  ['贝斯', ['原声贝斯','指弹贝斯','拨片贝斯','无品贝斯','击弦贝斯 1','击弦贝斯 2','合成贝斯 1','合成贝斯 2']],
  ['弦乐', ['小提琴','中提琴','大提琴','低音提琴','弦乐震音','弦乐拨奏','竖琴','定音鼓']],
  ['合奏与人声', ['弦乐合奏 1','弦乐合奏 2','合成弦乐 1','合成弦乐 2','合唱啊声','人声喔声','合成人声','管弦乐齐奏']],
  ['铜管', ['小号','长号','大号','弱音小号','圆号','铜管合奏','合成铜管 1','合成铜管 2']],
  ['木管', ['高音萨克斯','中音萨克斯','次中音萨克斯','上低音萨克斯','双簧管','英国管','巴松','单簧管']],
  ['笛类', ['短笛','长笛','竖笛','排箫','吹瓶','尺八','口哨','陶笛']],
  ['合成主音', ['方波主音','锯齿波主音','汽笛主音','吹管主音','吉他主音','人声主音','五度主音','贝斯与主音']],
  ['合成铺底', ['新时代铺底','温暖铺底','复音铺底','合唱铺底','弓弦铺底','金属铺底','光环铺底','扫弦铺底']],
  ['合成效果', ['雨声效果','音轨效果','水晶效果','气氛效果','明亮效果','精灵效果','回声效果','科幻效果']],
  ['民族乐器', ['西塔琴','班卓琴','三味线','筝','拇指琴','风笛','民族提琴','唢呐']],
  ['打击乐', ['铃铛','阿哥哥鼓','钢鼓','木鱼','太鼓','旋律鼓','合成鼓','反向镲']],
  ['音效', ['吉他换把声','呼吸声','海浪','鸟鸣','电话铃','直升机','掌声','枪声']],
];

function programOptions() {
  return instrumentPrograms.map(([name, programs], group) => {
    const node = document.createElement('optgroup');
    node.label = name;
    node.append(...programs.map((label, index) => new Option(label, group * 8 + index)));
    return node;
  });
}


export function initMetadata({ start, go, render, setBusy }) {
  let activePart = '';
  const library = initChordLibraryReview({render, setBusy});
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
    $("subtitle").value = root?.document_metadata?.subtitle || "";
    $("headerNotes").value = root?.document_metadata?.header_notes || "";
    $("instrument").value = m?.instrument || "";
    $("midiProgram").value = String(m?.midi_program ?? ({guitar:25,bass:33,pitched:0,drums:0}[$("instrument").value]));
    $("tempo").value = m?.document_metadata?.tempo_quarter ?? "";
    $("capo").value = m?.capo || 0;
    $("transpose").value = m?.transpose ?? "";
    const shifts = [...new Set((m?.document_metadata?.pitch_instructions || [])
      .filter((p) => p.parsed?.kind === "instrument" && Number.isInteger(p.parsed.semitones))
      .map((p) => p.parsed.semitones))];
    $("transpose").placeholder = shifts.length === 1 ? `自动读取：${shifts[0]}` : "自动读取";
    $("tuning").value = (m?.tuning_used || []).join(",");
    const preset = Array.from($("tuningPreset").options).some(
      (o) => o.value === $("tuning").value,
    );
    $("tuningPreset").value = preset ? $("tuning").value : "custom";
    $("customTuning").hidden = preset;
    $("confirmInstrument").checked = $("confirmTuning").checked = false;
    $("confirmInstrumentField").hidden = !(m?.document_metadata?.instrument_issue || m?.document_metadata?.instrument_needs_review);
    $("confirmTuningField").hidden = !(m?.document_metadata?.tuning_issue || m?.document_metadata?.tuning_needs_review || m?.document_metadata?.string_count_needs_review);
    instrumentFields();
    updateMetadataControls();
    const warnings = [...new Set([...(root?.document_metadata?.warnings || []), ...(m?.document_metadata?.warnings || []), m?.document_metadata?.instrument_issue, m?.document_metadata?.tuning_issue].filter(Boolean))];
    const pendingChords = (m?.document_metadata?.score_annotations || []).filter(hasPendingChordCandidate);
    const pendingNames = pendingChords.filter(annotation => annotation.chord_evidence?.name_needs_review);
    if (pendingNames.length) warnings.push(`${pendingNames.length} 处和弦名称尚未识别清楚，请在和弦指法库对照原谱修改名称。`);
    const pendingDiagrams = pendingChords.filter(annotation => !annotation.chord_evidence?.name_needs_review);
    if (pendingDiagrams.length) warnings.push(`${pendingDiagrams.length} 处和弦图示仍有未决或冲突的指法。请在下面的谱面指法库对照原谱确认。`);
    $("infoWarnings").hidden = !warnings.length;
    $("infoWarnings").textContent = warnings.join("\n");
    library.renderLibrary(activePart);
  }
  $("metadataPart").onchange = () => {
    if (ui.metadataDirty || ui.libraryDirty) {
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
    $("midiProgramField").hidden = !$("instrument").value || $("instrument").value === "drums";
    $("customTuning").hidden = !fretted || $("tuningPreset").value !== "custom";
  }
  $("instrument").onchange = () => {
    $("confirmInstrument").checked = $("confirmTuning").checked = false;
    const tuning = {guitar: "64,59,55,50,45,40", bass: "43,38,33,28"}[$("instrument").value] || "";
    $("tuning").value = tuning;
    $("tuningPreset").value = tuning;
    $("midiProgram").value = String({guitar:25,bass:33,pitched:0,drums:0}[$("instrument").value]);
    instrumentFields();
    ui.metadataDirty = true;
    updateMetadataControls();
  };
  $("tuningPreset").onchange = () => {
    $("confirmTuning").checked = false;
    $("customTuning").hidden = $("tuningPreset").value !== "custom";
    if ($("tuningPreset").value !== "custom")
      $("tuning").value = $("tuningPreset").value;
    ui.metadataDirty = true;
    updateMetadataControls();
  };
  for (const id of ["title", "artist", "subtitle", "headerNotes", "partName", "midiProgram", "tempo", "capo", "tuning", "transpose"])
    $(id).oninput = () => {
      if (id === "tuning") $("confirmTuning").checked = false;
      ui.metadataDirty = true;
    updateMetadataControls();
    };
  for (const id of ["confirmInstrument", "confirmTuning"])
    $(id).onchange = () => {
      ui.metadataDirty = true;
      updateMetadataControls();
    };
  function updateMetadataControls() {
    library.controls();
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
      (ui.metadataDirty || ui.libraryDirty || ui.state.info) &&
      !confirm("自动读取会替换当前信息；调弦改变时需重新识别小节，继续？")
    )
      return;
    await start("/information", undefined, () => {
      ui.metadataDirty = ui.libraryDirty = ui.measureDirty = false;
    });
  });
  $("saveInfo").onclick = action(async () => {
    if (ui.libraryDirty) return notice("请先确认保存或放弃指法库的修改。", true);
    if (!ui.metadataDirty && ui.state.info) {
      if (showPendingIdentity()) return;
      go(3);
      return;
    }
    for (const id of ["partName", "instrument", "tempo", "capo", "transpose"]) {
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
        title: $("title").value.trim(),
        artist: $("artist").value,
        subtitle: $("subtitle").value,
        header_notes: $("headerNotes").value,
        instrument: $("instrument").value,
        midi_program: $("instrument").value === 'drums' ? 0 : +$("midiProgram").value,
        tempo_quarter: $("tempo").value === "" ? null : +$("tempo").value,
        capo: fretted ? +$("capo").value : 0,
        tuning_used: tuning,
        confirm_instrument: $("confirmInstrument").checked,
        confirm_tuning: $("confirmTuning").checked,
        transpose: $("transpose").value.trim() ? +$("transpose").value : null,
      }, ui.state.revision);
      receiveProject(saved);
      render();
    } finally {
      setBusy(false);
    }
    if (ui.navigation === navigation && showPendingIdentity()) return;
    go(3, ui.navigation === navigation);
    if (ui.navigation === navigation)
      notice(ui.state.recognition ? "谱面信息已保存，音符修改已保留。" : "谱面信息已保存，下一步识别音符与节奏。");
  });
  function showPendingIdentity() {
    if (ui.state.recognition) return false;
    const root = ui.state.metadata;
    const pending = (root?.parts?.length ? root.parts : [root]).find(p => {
      const m = p?.document_metadata;
      return m?.instrument_issue || m?.instrument_needs_review || m?.tuning_issue || m?.tuning_needs_review || m?.string_count_needs_review;
    });
    if (!pending) return false;
    if (pending.id) $("metadataPart").value = pending.id;
    renderMetadata();
    go(2);
    const instrument = pending.document_metadata.instrument_issue || pending.document_metadata.instrument_needs_review;
    $(instrument ? "instrument" : "tuningPreset").focus();
    notice(`请先对照原谱确认${pending.name ? `“${pending.name}”的` : ""}${instrument ? "乐器" : "调弦"}，再继续识别。`);
    return true;
  }
  return { renderMetadata, updateMetadataControls };
}
