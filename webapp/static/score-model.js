// Edit the existing score structure; untouched notes retain all their fields.
export const clone = (value) => structuredClone(value);
export function ticks(event) {
  const d = event.duration;
  return Math.round(3840 / d.value * (d.double_dotted ? 1.75 : d.dotted ? 1.5 : 1)
    * (d.tuplet_times || 1) / (d.tuplet_enters || 1));
}
export function pitchShift(context = {}, effects = [], instrument = "guitar") {
  if (instrument === "drums") return 0;
  const octave = effects.find((e) => e.startsWith("ottava:"));
  return (context.instrument_transpose ?? (["guitar", "bass"].includes(instrument) ? -12 : 0))
    + (context.clef_octave || 0) - (context.capo || 0) + (octave ? +octave.split(":")[1] : 0);
}
const sharpNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const flatNames = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
export function pitchName(pitch, flat = false) {
  return (flat ? flatNames : sharpNames)[((pitch % 12) + 12) % 12] + (Math.floor(pitch / 12) - 1);
}
export function parsePitch(value) {
  const m = String(value).trim().match(/^([A-Ga-g])([#b♯♭]?)(-?\d+)$/);
  if (!m) throw new Error("音名格式为 C4、F#4 或 Bb3。");
  const pitch = ({ C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 })[m[1].toUpperCase()]
    + (["#", "♯"].includes(m[2]) ? 1 : ["b", "♭"].includes(m[2]) ? -1 : 0) + (+m[3] + 1) * 12;
  if (pitch < 0 || pitch > 127) throw new Error("音高超出 MIDI 范围。");
  return pitch;
}
export function setFret(event, string, value, mode, tuning) {
  const fret = String(value).toLowerCase() === "x" ? "x" : Number(value);
  if (String(value).trim() === "" || (fret !== "x" && (!Number.isInteger(fret) || fret < 0 || fret > 30)))
    throw new Error("品位使用 0 至 30 的整数，闷音使用 X。");
  let note = event.notes.find((n) => n.string === string);
  if (!note) { note = { string, effects: [] }; event.notes.push(note); }
  note.fret = fret;
  note.effects = (note.effects || []).filter((e) => e !== "dead");
  if (fret === "x") note.effects.push("dead");
  if (mode === "both") note.pitch = tuning[string - 1] + (fret === "x" ? 0 : fret);
  event.status = "normal";
  return event.notes.indexOf(note);
}
export function removeNote(event, index) {
  event.notes.splice(index, 1);
  if (!event.notes.length) event.status = "rest";
}
export function setPitch(event, index, pitch, mode, tuning = []) {
  if (!Number.isInteger(pitch) || pitch < 0 || pitch > 127)
    throw new Error("音高超出 MIDI 范围。");
  let note = event.notes[index];
  if (mode === "notation" && note?.pitch !== pitch && event.notes.some((n, i) => i !== index && n.pitch === pitch))
    throw new Error("这个和弦已经包含该音高。");
  if (mode === "both") {
    const available = tuning.map((open, i) => ({ string: i + 1, fret: pitch - open }))
      .filter(n => n.fret >= 0 && n.fret <= 30
        && !event.notes.some((other, i) => i !== index && other.string === n.string));
    const position = available.find(n => n.string === note?.string)
      || available.sort((a, b) => a.fret - b.fret)[0];
    if (!position) throw new Error("当前调弦下没有可用弦位，请调整和弦或调弦。");
    if (!note) { note = { effects: [] }; event.notes.push(note); }
    Object.assign(note, position);
    note.effects = (note.effects || []).filter(e => e !== "dead");
  } else if (!note) { note = { effects: [] }; event.notes.push(note); }
  note.pitch = pitch;
  event.status = "normal";
  return event.notes.indexOf(note);
}
export function changeDuration(voice, index, duration) {
  const event = voice.events[index], delta = ticks({ duration }) - ticks(event), end = event.start + ticks(event);
  event.duration = clone(duration);
  for (const next of voice.events.slice(index + 1)) if (next.start >= end) next.start += delta;
}
export function insertEvent(voice, index) {
  const previous = voice.events[index];
  const event = { start: previous ? previous.start + ticks(previous) : 0,
    duration: clone(previous?.duration || { value: 4 }), status: "rest", notes: [], effects: [] };
  for (const next of voice.events.slice(index + 1)) next.start += ticks(event);
  voice.events.splice(index + 1, 0, event);
  return index + 1;
}
export function removeEvent(voice, index) {
  if (voice.events.length === 1) {
    voice.events[0].notes = []; voice.events[0].status = "rest"; return 0;
  }
  const [event] = voice.events.splice(index, 1), end = event.start + ticks(event);
  for (const next of voice.events.slice(index)) if (next.start >= end) next.start -= ticks(event);
  return Math.min(index, voice.events.length - 1);
}
export const drumNames = {
  35: "原声底鼓", 36: "底鼓", 37: "边击", 38: "军鼓", 42: "闭镲", 44: "踩镲",
  46: "开镲", 43: "低嗵鼓", 45: "中低嗵鼓", 47: "中嗵鼓", 48: "高嗵鼓", 50: "高音嗵鼓",
  49: "吊镲", 51: "叮叮镲", 52: "中国镲", 53: "镲帽", 55: "水镲", 57: "吊镲 2", 59: "叮叮镲 2",
};
