import { $ } from "./dom.js";
import { writtenKey } from "./score-structure.js";
import { parsePosition } from "./score-effects.js";
import { parsePitch, pitchName, pitchShift } from "./score-model.js";

const tonics = ["CFlat", "GFlat", "DFlat", "AFlat", "EFlat", "BFlat", "F", "C", "G", "D", "A", "E", "B", "FSharp", "CSharp"];
const minors = ["AFlat", "EFlat", "BFlat", "F", "C", "G", "D", "A", "E", "B", "FSharp", "CSharp", "GSharp", "DSharp", "ASharp"];
const keyCode = (fifths, minor) => (minor ? minors : tonics)[fifths + 7] + (minor ? "Minor" : "Major");
const keyChoices = [false, true].flatMap(minor =>
  (minor ? minors : tonics).map((tonic, index) => ({
    value: keyCode(index - 7, minor),
    label: `${tonic.replace("Flat", "♭").replace("Sharp", "♯")} ${minor ? "小调" : "大调"}`,
  })));

// The inspector edits the printed key; the existing ScoreIR keeps concert key.
function keyForDisplay(key, profile) {
  return key ? keyCode(writtenKey(key, profile.pitch_context?.instrument_transpose), String(key).includes("Minor")) : "";
}
function keyFromDisplay(key, profile) {
  if (!key) return null;
  if (!keyChoices.some(choice => choice.value === key)) throw new Error("请选择有效调号。");
  return keyCode(writtenKey(key, -(profile.pitch_context?.instrument_transpose || 0)), key.includes("Minor"));
}
function endingsText(mask) {
  return Array.from({length: 8}, (_, i) => i + 1).filter(n => (mask || 0) & (1 << (n - 1))).join(", ");
}
function endingsMask(value) {
  if (!value.trim()) return null;
  const values = value.trim().split(/[\s,，]+/);
  if (values.some(n => !/^[1-8]$/.test(n))) throw new Error("跳房子填写 1 至 8，用逗号分隔，例如 1, 2。");
  return values.reduce((mask, n) => mask | (1 << (Number(n) - 1)), 0);
}
function flag(draft, name, enabled) {
  draft.bars = (draft.bars || []).filter(value => value !== name);
  if (enabled) draft.bars.push(name);
}
function setMeasureProperty(draft, field, value, profile) {
  if (field === "key") {
    draft.key_signature = keyFromDisplay(value, profile);
    draft.print_key_signature = !!value;
  } else if (["repeat_open", "repeat_close", "double", "final"].includes(field)) {
    flag(draft, field, value);
    if (value && field === "final") flag(draft, "double", false);
    if (value && field === "double") flag(draft, "final", false);
    if (field === "repeat_close") draft.repeat_count = value ? draft.repeat_count || 2 : null;
  } else if (field === "repeat_count") {
    const count = Number(value);
    if (!Number.isInteger(count) || count < 2 || count > 99) throw new Error("反复次数须为 2 至 99 的整数。");
    draft.repeat_count = count;
    flag(draft, "repeat_close", true);
  } else if (field === "alternate_endings") draft.alternate_endings = endingsMask(value);
  else if (["section", "direction", "from_direction", "triplet_feel"].includes(field)) draft[field] = value.trim() || null;
  else throw new Error("未知小节属性。");
}

const targets = ["Coda", "Double Coda", "Segno", "Segno Segno", "Fine"];
const jumps = ["Da Capo", "Da Capo al Fine", "Da Capo al Coda", "Da Capo al Double Coda",
  "Da Segno", "Da Segno al Fine", "Da Segno al Coda", "Da Segno al Double Coda",
  "Da Segno Segno", "Da Segno Segno al Fine", "Da Segno Segno al Coda", "Da Segno Segno al Double Coda",
  "Da Coda", "Da Double Coda"];
const decode = value => { try { return decodeURIComponent(value); } catch { return value; } };
export function measureProperties({ getDraft, getProfile, getEvent, change }) {
  $("measureKey").replaceChildren(new Option("沿用上一小节", ""), ...keyChoices.map(c => new Option(c.label, c.value)));
  for (const [id, values] of [["measureDirection", targets], ["measureJump", jumps]])
    $(id).replaceChildren(new Option("无", ""), ...values.map(value => new Option(value, value)));
  const fields = { measureKey: "key", measureSection: "section", repeatStart: "repeat_open",
    repeatEnd: "repeat_close", repeatCount: "repeat_count", alternateEndings: "alternate_endings",
    doubleBar: "double", finalBar: "final", measureDirection: "direction", measureJump: "from_direction", tripletFeel: "triplet_feel" };
  for (const [id, field] of Object.entries(fields)) $(id).onchange = () => change(() =>
    setMeasureProperty(getDraft(), field, $(id).type === "checkbox" ? $(id).checked : $(id).value, getProfile()));
  $("beatText").onchange = () => change(() => {
    const event = getEvent();
    if (!event) return;
    const text = $("beatText").value.trim();
    event.effects = (event.effects || []).filter(effect => !effect.startsWith("text:"));
    if (text) event.effects.push(`text:${encodeURIComponent(text)}`);
  });
  return {
    render(disabled) {
      const draft = getDraft(), profile = getProfile();
      $("measureKeyField").hidden = profile.mode === "tab" || profile.instrument === "drums";
      for (const [id, field] of Object.entries(fields)) {
        const node = $(id);
        node.disabled = disabled || (id === "repeatCount" && !draft.bars?.includes("repeat_close"));
        if (node.type === "checkbox") node.checked = !!draft.bars?.includes(field);
        else {
          const value = field === "key" ? keyForDisplay(draft.key_signature, profile) :
            field === "alternate_endings" ? endingsText(draft.alternate_endings) : draft[field] ?? "";
          if (node.tagName === "SELECT" && value && ![...node.options].some(o => o.value === value))
            node.append(new Option(value, value));
          node.value = value;
        }
      }
      const event = getEvent(), text = event?.effects?.find(effect => effect.startsWith("text:"));
      $("beatText").value = text ? decode(text.slice(5)) : "";
      $("beatText").disabled = disabled || !event;
    },
  };
}

const family = (effect, name) => effect === name || effect.startsWith(name + ":");
function replaceEffect(owner, name, value) {
  owner.effects = (owner.effects || []).filter(effect => !family(effect, name));
  if (value) owner.effects.push(value);
}
function integer(value, min, max, label) {
  if (String(value).trim() === "" || !Number.isInteger(Number(value)) || +value < min || +value > max)
    throw new Error(`${label}须为 ${min} 至 ${max} 的整数。`);
  return Number(value);
}
function setBend(note, kind, semitones) {
  if (!kind) return replaceEffect(note, "bend", "");
  if (!["bend", "bendRelease", "bendReleaseBend", "prebend", "prebendRelease"].includes(kind))
    throw new Error("请选择推弦轮廓。");
  const value = Number(semitones);
  if (!(value > 0 && value <= 6) || !Number.isInteger(value * 2))
    throw new Error("推弦幅度为 0.5 至 6 个半音，步长为 0.5。");
  // GP stores 50 units per semitone; alphaTab uses quarter-tone steps.
  replaceEffect(note, "bend", `bend:${kind}:${value * 50}`);
}
function setHarmonic(note, kind, values = {}) {
  if (!kind) return replaceEffect(note, "harm", "");
  if (!["natural", "artificial", "pinch", "tapped", "semi"].includes(kind)) throw new Error("请选择泛音类型。");
  let effect = `harm:${kind}`;
  if (kind === "tapped") effect += `:${integer(values.fret, 0, 30, "触弦品位")}`;
  if (kind === "artificial") {
    const pitch = integer(values.pitch, 0, 11, "泛音音级"), octave = integer(values.octave, 0, 4, "泛音八度");
    const accidental = [1, 3, 6, 8, 10].includes(pitch) ? 1 : 0;
    effect += `:${pitch - accidental}:${accidental}:${octave}`;
  }
  replaceEffect(note, "harm", effect);
}
const slideGroups = { slideIn: ["sib", "sia"], slideOut: ["sl", "ss", "sod", "sou"] };
const beatGroups = { pick: ["pick_up", "pick_down"], stroke: ["stroke_up", "stroke_down"],
  slap: ["slap:tapping", "slap:slapping", "slap:popping", "tap"] };
function setChoice(owner, choices, value) {
  if (value && !choices.includes(value)) throw new Error("请选择有效奏法。");
  owner.effects = (owner.effects || []).filter(effect => !choices.includes(effect));
  if (value) owner.effects.push(value);
}
function setSlide(note, field, value) { setChoice(note, slideGroups[field], value); }
function setBeatProperty(event, field, value) {
  if (beatGroups[field]) return setChoice(event, beatGroups[field], value);
  if (["rasg", "fade"].includes(field)) return replaceEffect(event, field, value ? field : "");
  if (field === "tempo") return replaceEffect(event, field, value === "" ? "" : `tempo:${integer(value, 1, 999, "速度")}`);
  if (field === "ottava" && ["", "-24", "-12", "12", "24"].includes(String(value)))
    return replaceEffect(event, field, value === "" ? "" : `ottava:${value}`);
  throw new Error("未知拍属性。");
}
function ornamentPosition(value, note, event, profile) {
  if (String(value).trim().toLowerCase() === "x") return "x";
  if (profile.mode === "tab" || profile.mode === "both") {
    const fret = integer(value, 0, 30, "辅助音品位"), open = profile.tuning?.[note.string - 1];
    if (!Number.isFinite(open)) throw new Error("请先确认当前弦的调弦。");
    return profile.mode === "both" ? `f${fret}p${open + fret}` : `f${fret}`;
  }
  const pitch = parsePitch(value) + pitchShift(profile.pitch_context, event.effects, profile.instrument);
  if (pitch < 0 || pitch > 127) throw new Error("辅助音超出 MIDI 范围。");
  return `p${pitch}`;
}
function setOrnament(note, event, profile, kind, values) {
  const position = ornamentPosition(values.position, note, event, profile);
  const duration = Number(values.duration);
  if (!(kind === "grace" ? [8, 16, 32, 64, 128] : [16, 32, 64]).includes(duration))
    throw new Error("请选择有效的装饰音时值。");
  if (kind === "trill" && position === "x") throw new Error("颤音辅助音须有音高。");
  let effect = `${kind}:${position}:${duration}`;
  if (kind === "grace") {
    if (!["none", "slide", "bend", "hammer"].includes(values.transition)) throw new Error("请选择装饰音过渡方式。");
    effect += `:${values.transition}:${values.onBeat ? 1 : 0}:${position === "x" || values.dead ? 1 : 0}`;
  }
  replaceEffect(note, kind, effect);
}
function selectedEffect(owner, name) { return owner?.effects?.find(effect => family(effect, name)) || ""; }
function selectValue(id, value) {
  const node = $(id);
  node.querySelectorAll("option[data-source]").forEach(option => option.remove());
  if (value && ![...node.options].some(option => option.value === value)) {
    const option = new Option(`原谱：${value}`, value);
    option.dataset.source = "true";
    node.add(option);
  }
  node.value = value;
}
export function effectProperties({ getNote, getEvent, getProfile, change }) {
  const noteChange = fn => change(() => { if (getNote()) fn(getNote()); });
  $("applyBend").onclick = () => noteChange(note => setBend(note, $("bendShape").value, $("bendSemitones").value));
  $("applyHarmonic").onclick = () => noteChange(note => setHarmonic(note, $("harmonicKind").value,
    { fret: $("harmonicFret").value, pitch: $("harmonicPitch").value, octave: $("harmonicOctave").value }));
  const harmonicFields = () => {
    $("harmonicFretField").hidden = $("harmonicKind").value !== "tapped";
    $("artificialHarmonicFields").hidden = $("harmonicKind").value !== "artificial";
  };
  $("harmonicKind").onchange = harmonicFields;
  for (const name of Object.keys(slideGroups)) $(name).onchange = () => noteChange(note => setSlide(note, name, $(name).value));
  for (const kind of ["grace", "trill"]) {
    $(kind + "Apply").onclick = () => noteChange(note => setOrnament(note, getEvent(), getProfile(), kind, {
      position: $(kind + "Position").value, duration: $(kind + "Duration").value,
      transition: $("graceTransition").value, onBeat: $("graceOnBeat").checked, dead: $("graceDead").checked,
    }));
    $(kind + "Remove").onclick = () => noteChange(note => replaceEffect(note, kind, ""));
  }
  const beatFields = { beatTempo: "tempo", beatOttava: "ottava", beatPick: "pick", beatStroke: "stroke",
    beatSlap: "slap", beatRasgueado: "rasg", beatFade: "fade" };
  for (const [id, field] of Object.entries(beatFields)) $(id).onchange = () => change(() => {
    if (getEvent()) setBeatProperty(getEvent(), field, $(id).type === "checkbox" ? $(id).checked : $(id).value);
  });
  return { render(disabled) {
    const note = getNote(), event = getEvent(), profile = getProfile();
    $("noteEffectProperties").hidden = !note || profile.instrument === "drums";
    $("noteEffectProperties").querySelectorAll("input,select,button").forEach(node => { node.disabled = disabled || !note; });
    for (const [id, field] of Object.entries(beatFields)) {
      const node = $(id), value = beatGroups[field] ? event?.effects?.find(e => beatGroups[field].includes(e)) : selectedEffect(event, field);
      node.disabled = disabled || !event;
      if (node.type === "checkbox") node.checked = !!value;
      else if (node.tagName === "SELECT") selectValue(id, beatGroups[field] ? value || "" : value?.split(":")[1] || "");
      else node.value = value?.split(":")[1] || "";
    }
    const [, bend, height] = selectedEffect(note, "bend").split(":");
    selectValue("bendShape", bend || "");
    $("bendSemitones").value = height === undefined ? 2 : Number(height) / 50;
    const [, harmonic, fret, accidental, octave] = selectedEffect(note, "harm").split(":");
    selectValue("harmonicKind", harmonic || "");
    $("harmonicFret").value = harmonic === "tapped" ? fret ?? Number(note?.fret || 0) + 12 : 12;
    $("harmonicPitch").value = harmonic === "artificial" && octave !== undefined ? ((Number(fret) + Number(accidental)) % 12 + 12) % 12 : ((note?.pitch ?? (profile.tuning?.[(note?.string || 1) - 1] || 0) + Number(note?.fret || 0)) % 12);
    $("harmonicOctave").value = octave ?? 1;
    harmonicFields();
    for (const [id, choices] of Object.entries(slideGroups)) selectValue(id, note?.effects?.find(e => choices.includes(e)) || "");
    for (const kind of ["grace", "trill"]) {
      const effect = selectedEffect(note, kind), [, position, ...fields] = effect.split(":");
      const duration = /^\d+$/.test(fields[0] || "") ? fields.shift() : kind === "grace" ? "32" : "16";
      const auxiliary = parsePosition(position), fretted = profile.mode !== "notation";
      $(kind + "PositionLabel").textContent = fretted ? "辅助音品位" : "辅助音（谱面音名）";
      $(kind + "Position").placeholder = fretted ? "例如 5" : "例如 F#4";
      $(kind + "Position").value = position === "x" ? "X" : fretted ? auxiliary.fret ?? "" :
        auxiliary.pitch === undefined ? "" : pitchName(auxiliary.pitch - pitchShift(profile.pitch_context, event?.effects, profile.instrument));
      selectValue(kind + "Duration", duration);
      $(kind + "Status").textContent = effect ? position ? "已设置" : "辅助音待确认" : "未设置";
      if (kind === "grace") {
        selectValue("graceTransition", fields[0] || "none");
        $("graceOnBeat").checked = fields[1] === "1";
        $("graceDead").checked = fields[2] === "1" || position === "x";
      }
    }
  } };
}
