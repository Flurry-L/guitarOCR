import * as alphaTab from "./vendor/alphaTab.mjs";
import { pitchName } from "./score-model.js";
const M = alphaTab.model;
const directions = {
  Coda: "TargetCoda", "Double Coda": "TargetDoubleCoda", Segno: "TargetSegno",
  "Segno Segno": "TargetSegnoSegno", Fine: "TargetFine", "Da Capo": "JumpDaCapo",
  "Da Capo al Coda": "JumpDaCapoAlCoda", "Da Capo al Double Coda": "JumpDaCapoAlDoubleCoda",
  "Da Capo al Fine": "JumpDaCapoAlFine", "Da Segno": "JumpDalSegno",
  "Da Segno al Coda": "JumpDalSegnoAlCoda", "Da Segno al Double Coda": "JumpDalSegnoAlDoubleCoda",
  "Da Segno al Fine": "JumpDalSegnoAlFine", "Da Segno Segno": "JumpDalSegnoSegno",
  "Da Segno Segno al Coda": "JumpDalSegnoSegnoAlCoda",
  "Da Segno Segno al Double Coda": "JumpDalSegnoSegnoAlDoubleCoda",
  "Da Segno Segno al Fine": "JumpDalSegnoSegnoAlFine", "Da Coda": "JumpDaCoda",
  "Da Double Coda": "JumpDaDoubleCoda",
};
const decode = (value) => { try { return decodeURIComponent(value); } catch { return value; } };

// A master bar belongs to the whole score. Gather all synchronized staves
// before creating it; a sparse first staff must not erase another staff's signs.
export function buildMasterBars(score, records, state) {
  const grouped = new Map(), warnings = [];
  for (const entry of records) {
    const index = entry.record.bar_index ?? entry.mi;
    if (!grouped.has(index)) grouped.set(index, []);
    grouped.get(index).push(entry);
  }
  let time = state.metadata?.time_signature || "4/4";
  const last = Math.max(-1, ...grouped.keys());
  for (let index = 0; index <= last; index++) {
    const entries = (grouped.get(index) || []).sort((a, b) =>
      `${a.record.part_id || ""}/${a.record.staff_id || ""}`.localeCompare(`${b.record.part_id || ""}/${b.record.staff_id || ""}`));
    const choose = (field, label) => {
      const values = [...new Set(entries.map(({data}) => data[field]).filter(value => value !== undefined && value !== null && value !== ""))];
      if (values.length > 1) warnings.push(`第 ${index + 1} 小节的${label}在不同谱表间不一致，请检查。`);
      return values[0];
    };
    const master = new M.MasterBar();
    time = choose("time_signature", "拍号") || (index === 0 ? entries.find(({record}) => record.score_state?.time)?.record.score_state.time : null) || time;
    [master.timeSignatureNumerator, master.timeSignatureDenominator] = time.split("/").map(Number);
    master.isRepeatStart = entries.some(({data}) => data.bars?.includes("repeat_open"));
    master.repeatCount = Math.max(0, ...entries.map(({data}) => data.bars?.includes("repeat_close") ? data.repeat_count || 2 : 0));
    master.isDoubleBar = entries.some(({data}) => data.bars?.includes("double"));
    master.alternateEndings = entries.reduce((bits, {data}) => bits | (data.alternate_endings || 0), 0);
    const section = [...new Set(entries.map(({data}) => data.section).filter(Boolean))];
    if (section.length) {
      master.section = new M.Section();
      master.section.text = section.map(decode).join(" / ");
    }
    master.tripletFeel = { eighth: M.TripletFeel.Triplet8th, sixteenth: M.TripletFeel.Triplet16th }[choose("triplet_feel", "摇摆节奏")] ?? M.TripletFeel.NoTripletFeel;
    const tempo = choose("tempo_quarter", "速度") || (index === 0 ? state.metadata?.document_metadata?.tempo_quarter : null);
    const tempos = new Map(tempo ? [[0, tempo]] : []), observed = new Map();
    // alphaTab removes legacy beat tempo automations in finishTuplet(). Tempo
    // is global: collect all voices/staves into the master bar before finish.
    for (const {data} of entries)
      for (const voice of [...(data.voices || [])].sort((a, b) => a.voice - b.voice))
        for (const event of voice.events || []) {
          const value = Number(event.effects?.find(effect => effect.startsWith("tempo:"))?.slice(6));
          if (!(value > 0)) continue;
          if (observed.has(event.start)) {
            if (observed.get(event.start) !== value) warnings.push(`第 ${index + 1} 小节同一拍的速度不一致，请检查。`);
          } else {
            observed.set(event.start, value);
            tempos.set(event.start, value);
          }
        }
    for (const [start, value] of [...tempos].sort((a, b) => a[0] - b[0]))
      master.tempoAutomations.push(M.Automation.buildTempoAutomation(false, start / master.calculateDuration(false), value, 2));
    for (const {data} of entries)
      for (const field of ["direction", "from_direction"])
        if (directions[data[field]]) master.addDirection(M.Direction[directions[data[field]]]);
    score.addMasterBar(master);
  }
  return warnings;
}

const keys = {
  C: 0,
  G: 1,
  D: 2,
  A: 3,
  E: 4,
  B: 5,
  FSharp: 6,
  CSharp: 7,
  F: -1,
  BFlat: -2,
  EFlat: -3,
  AFlat: -4,
  DFlat: -5,
  GFlat: -6,
  CFlat: -7,
  AMinor: 0,
  EMinor: 1,
  BMinor: 2,
  FSharpMinor: 3,
  CSharpMinor: 4,
  GSharpMinor: 5,
  DSharpMinor: 6,
  ASharpMinor: 7,
  DMinor: -1,
  GMinor: -2,
  CMinor: -3,
  FMinor: -4,
  BFlatMinor: -5,
  EFlatMinor: -6,
  AFlatMinor: -7,
};
export function writtenKey(key, shift) {
  if (typeof key === "number") return key; // score_state keys are already written.
  const normalized = String(key)
    .replace(/[♯#]/g, "Sharp")
    .replace(/[♭b]/g, "Flat")
    .replace(/Major(?=Sharp|Flat|$)/, "")
    .replace(/Minor(Sharp|Flat)/, "$1Minor");
  const fifths = keys[normalized] ?? 0;
  if (!shift || shift % 12 === 0) return fifths;
  const tonic = (((7 * fifths - shift) % 12) + 12) % 12;
  return Array.from({ length: 15 }, (_, i) => i - 7)
    .filter((k) => (((7 * k) % 12) + 12) % 12 === tonic)
    .sort(
      (a, b) =>
        Math.abs(a) - Math.abs(b) ||
        Math.abs(a - fifths) - Math.abs(b - fifths),
    )[0];
}

export function musicalContext(state, index, draft, selected, profile) {
  const current = state.measures[index], bar = current.bar_index ?? index;
  let time = state.metadata?.time_signature || "4/4", key = current.score_state?.key;
  let tempo = state.metadata?.document_metadata?.tempo_quarter;
  const groups = new Map();
  state.measures.forEach((record, mi) => {
    const position = record.bar_index ?? mi;
    if (position > bar) return;
    if (!groups.has(position)) groups.set(position, []);
    groups.get(position).push({record, data: mi === index ? draft : record.parsed});
  });
  for (const [position, group] of [...groups.entries()].sort((a,b) => a[0]-b[0])) {
    group.sort((a,b) => `${a.record.part_id || ""}/${a.record.staff_id || ""}`.localeCompare(`${b.record.part_id || ""}/${b.record.staff_id || ""}`));
    time = group.find(({data}) => data.time_signature)?.data.time_signature ||
      (position === 0 ? group.find(({record}) => record.score_state?.time)?.record.score_state.time : null) || time;
    tempo = group.find(({data}) => data.tempo_quarter)?.data.tempo_quarter || tempo;
    const tempoEvents = [];
    for (const {record,data} of group) {
      if (record.part_id === current.part_id && record.staff_id === current.staff_id)
        key = data.key_signature ?? key ?? record.score_state?.key;
      for (const voice of data.voices || []) for (const [eventIndex, event] of (voice.events || []).entries()) {
        if (position === bar && event.start > (draft.voices[selected.vi]?.events[selected.ei]?.start ?? 0)) continue;
        const mark = event.effects?.find(effect => effect.startsWith("tempo:"));
        if (mark) tempoEvents.push({start: event.start, value: Number(mark.slice(6)),
          lane: `${record.part_id || ""}/${record.staff_id || ""}`, voice: voice.voice, eventIndex});
      }
    }
    // Tempo applies to the whole score. A later event wins across all tracks
    // and voices. At the same start, use the first lane/voice/event in this
    // fixed ordering, matching the stable track priority for master-bar marks.
    tempoEvents.sort((a,b) => a.start-b.start || a.lane.localeCompare(b.lane) || a.voice-b.voice || a.eventIndex-b.eventIndex);
    let previousStart;
    for (const change of tempoEvents) {
      if (change.start !== previousStart) tempo = change.value;
      previousStart = change.start;
    }
  }
  const minor = String(key).includes("Minor");
  const keyNames = minor ? ["A♭", "E♭", "B♭", "F", "C", "G", "D", "A", "E", "B", "F♯", "C♯", "G♯", "D♯", "A♯"] :
    ["C♭", "G♭", "D♭", "A♭", "E♭", "B♭", "F", "C", "G", "D", "A", "E", "B", "F♯", "C♯"];
  const keyName = key === undefined || key === null ? null :
    `${keyNames[writtenKey(key, profile.pitch_context?.instrument_transpose) + 7]}${minor ? " 小调" : ""}`;
  const event = draft.voices[selected.vi]?.events[selected.ei];
  return {
    track: current.part_name || "音轨",
    meter: `${time} · ${keyName || "调号待确认"}${tempo ? ` · ♩ = ${tempo}` : " · 速度待确认"}`,
    tuning: [profile.tuning?.length && ["guitar", "bass"].includes(profile.instrument) ? profile.tuning.map(pitchName).join(" ") : "",
      profile.pitch_context?.capo ? `Capo ${profile.pitch_context.capo}` : ""].filter(Boolean).join(" · "),
    position: `第 ${bar + 1} 小节 · 声部 ${(draft.voices[selected.vi]?.voice ?? 0) + 1}${event ? ` · 第 ${event.start / 960 + 1} 拍` : ""}`,
  };
}
