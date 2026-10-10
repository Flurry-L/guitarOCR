// Edit the existing score structure; untouched notes retain all their fields.
export const clone = (value) => structuredClone(value);
// A display projection only. Incomplete evidence never becomes an editable
// measure or an exportable score merely because some events can be drawn.
export function observationPreview(record) {
  const conflicts = new Set((record.unresolved || [])
    .filter(issue => issue.reason === "dual_note_pairing_unresolved")
    .map(issue => `${issue.voice}:${issue.event}`));
  const duration = token => {
    const match = /^(w|h|q|e|s|t|f|d[0-9]+)(\.\.|\.)?(?:\[([0-9]+):([0-9]+)\])?$/.exec(token);
    if (!match) return null;
    const value = { w: 1, h: 2, q: 4, e: 8, s: 16, t: 32, f: 64 }[match[1]] ?? Number(match[1].slice(1));
    const enters = Number(match[3] || 1), times = Number(match[4] || 1);
    return value > 0 && enters > 0 && times > 0
      ? { value, dotted: match[2] === ".", double_dotted: match[2] === "..", tuplet_enters: enters, tuplet_times: times }
      : null;
  };
  const state = record.resolved?.state || {};
  const needs = record.mode === "tab" ? ["string", "fret"]
    : record.mode === "notation" ? ["pitch"] : ["string", "fret", "pitch"];
  const voices = (record.resolved?.voices || []).map(voice => ({
    voice: voice.voice,
    events: (voice.events || []).flatMap((event, index) => {
      const rhythm = duration(event.duration);
      if (!Number.isSafeInteger(event.start) || event.start < 0 || !rhythm
        || !["normal", "rest", "empty"].includes(event.status)
        || conflicts.has(`${voice.voice}:${index}`)) return [];
      // Conflicting rhythm has no faithful single-voice engraving. Do not
      // move a later note to the end of an overlapping observation.
      if (voice.events.some((other, j) => {
        const d = duration(other.duration);
        return j !== index && Number.isSafeInteger(other.start) && d
          && other.start < event.start + ticks({duration: rhythm})
          && event.start < other.start + ticks({duration: d});
      })) return [];
      const notes = event.notes || [];
      if (event.status === "normal" && (!notes.length
        || notes.some(note => needs.some(key => note[key] == null)))) return [];
      return [{ ...clone(event), duration: rhythm, source_event: index }];
    }).sort((a, b) => a.start - b.start),
  }));
  return { preview: true, voices, time_signature: state.time?.value, key_signature: state.key?.value };
}
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

const isMark = effect => effect === "pm" || effect === "let";
const valid = overlay => overlay?.protocol === "visible_events_v1" && overlay.scope === "visible_event" && Array.isArray(overlay.events);
export const visibleEvent = (overlay, voice, onset) => valid(overlay)
  ? overlay.events.find(event => event.voice === voice && event.onset === onset) : undefined;

// Event coverage cannot assign a technique to every tone of a chord. Keep
// known owners, and display new marks without owners as beat text.
export function projectVisibleTechniques(measure, overlay) {
  if (!valid(overlay)) return measure;
  const copy = structuredClone(measure);
  for (const voice of copy.voices || []) for (const event of voice.events || []) {
    const visible = visibleEvent(overlay, voice.voice, event.start);
    if (visible?.status !== "classified" || !Array.isArray(visible.effects) || visible.effects.some(effect => !isMark(effect))) continue;
    const desired = new Set(visible.effects), notes = event.notes || [];
    event.effects = (event.effects || []).filter(effect => !isMark(effect));
    for (const note of notes) {
      note.effects = (note.effects || []).filter(effect => !isMark(effect) || desired.has(effect));
      if (notes.length === 1) note.effects = [...new Set([...note.effects, ...desired])];
    }
    const unowned = [...desired].filter(effect => !notes.some(note => note.effects.includes(effect)));
    if (unowned.length) {
      const text = "text:" + encodeURIComponent(unowned.map(effect => effect === "pm" ? "P.M." : "let ring").join(" / "));
      if (!event.effects.includes(text)) event.effects.push(text);
    }
  }
  return copy;
}

export function setVisibleEvent(overlay, voice, onset, effects) {
  const events = valid(overlay) ? overlay.events.filter(event => event.voice !== voice || event.onset !== onset) : [];
  return {protocol: "visible_events_v1", scope: "visible_event",
    events: [...events, {voice, onset, effects: [...new Set(effects.filter(isMark))], status: "classified"}]
      .sort((a, b) => a.voice - b.voice || a.onset - b.onset)};
}

// A moved/edited beat must not inherit an obsolete image interpretation.
// Text/chord annotations can change without changing the musical anchor.
export function retainUnchangedVisualEvents(overlay, before, after) {
  if (!valid(overlay)) return overlay;
  const signature = (measure, mark) => {
    const event = measure.voices?.find(voice => voice.voice === mark.voice)?.events?.find(event => event.start === mark.onset);
    return event && JSON.stringify({duration: event.duration, status: event.status, notes: event.notes,
      marks: event.effects?.filter(isMark) || []});
  };
  return {...overlay, events: overlay.events.filter(mark => {
    const first = signature(before, mark), second = signature(after, mark);
    return first && first === second;
  })};
}
