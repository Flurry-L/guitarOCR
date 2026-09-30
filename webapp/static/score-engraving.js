import * as alphaTab from "./vendor/alphaTab.mjs";
import { pitchShift, ticks } from "./score-model.js";

export { alphaTab };
const M = alphaTab.model;
export function measureProfile(measure, state) {
  const metadata = state.metadata || {};
  return {
    mode: measure.mode || state.mode,
    instrument: measure.instrument || metadata.instrument || "guitar",
    tuning: measure.tuning || metadata.tuning_used || [64, 59, 55, 50, 45, 40],
    pitch_context: {
      ...(measure.pitch_context || {}),
      capo: measure.capo ?? metadata.capo ?? 0,
    },
  };
}
export function reviewKind(measure) {
  if (!measure.needs_review) return "";
  if (measure.annotation_review) return "review";
  return measure.fallback_reason?.length || measure.timing_errors?.length
    ? "failed"
    : "review";
}
// Serialized Ottavia values; upstream's minifier renames members starting with '_'.
const ottava = (value) =>
  ({ 24: 0, 12: 1, "-12": 3, "-24": 4 })[value] ?? M.Ottavia.Regular;
const decode = (value) => {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

// OCR can disagree between the written note and its TAB fret. Display both
// readings faithfully until the user corrects them. Everything else, including
// spacing, beams, accidentals and ties, is engraved by alphaTab.
class EngravedNote extends M.Note {
  writtenPitch;
  get displayValueWithoutBend() {
    return this.writtenPitch ?? super.displayValueWithoutBend;
  }
}
function noteEffects(note, effects) {
  const flags = {
    tie: "isTieDestination",
    dead: "isDead",
    ghost: "isGhost",
    pm: "isPalmMute",
    stacc: "isStaccato",
    let: "isLetRing",
    hammer: "isHammerPullOrigin",
  };
  for (const effect of effects) {
    if (flags[effect]) note[flags[effect]] = true;
    else if (effect === "vib") note.vibrato = M.VibratoType.Slight;
    else if (effect === "tap") note.beat.tap = true;
    else if (effect === "accent") note.accentuated = M.AccentuationType.Normal;
    else if (effect === "heavy") note.accentuated = M.AccentuationType.Heavy;
    else if (effect === "sl") note.slideOutType = M.SlideOutType.Legato;
    else if (effect === "ss") note.slideOutType = M.SlideOutType.Shift;
    else if (effect === "sib") note.slideInType = M.SlideInType.IntoFromBelow;
    else if (effect === "sia") note.slideInType = M.SlideInType.IntoFromAbove;
    else if (effect === "sod") note.slideOutType = M.SlideOutType.OutDown;
    else if (effect === "sou") note.slideOutType = M.SlideOutType.OutUp;
    else if (effect.startsWith("harm:")) {
      const [, kind, fret] = effect.split(":");
      note.harmonicType =
        {
          natural: M.HarmonicType.Natural,
          artificial: M.HarmonicType.Artificial,
          pinch: M.HarmonicType.Pinch,
          tapped: M.HarmonicType.Tap,
          semi: M.HarmonicType.Semi,
        }[kind] || M.HarmonicType.None;
      note.harmonicValue = kind === "natural" ? note.fret : Number(fret) || 12;
    } else if (effect.startsWith("bend:")) {
      const [, kind, value] = effect.split(":"),
        height = Number(value || 100) / 25;
      const curves = {
        bend: [
          [0, 0],
          [30, height],
          [60, height],
        ],
        bendRelease: [
          [0, 0],
          [30, height],
          [60, 0],
        ],
        bendReleaseBend: [
          [0, 0],
          [20, height],
          [40, 0],
          [60, height],
        ],
        prebend: [
          [0, height],
          [60, height],
        ],
        prebendRelease: [
          [0, height],
          [60, 0],
        ],
      };
      for (const [x, y] of curves[kind] || curves.bend)
        note.addBendPoint(new M.BendPoint(x, y));
    } else if (effect.startsWith("trill")) {
      const [, position, duration] = effect.split(":");
      note.trillValue = position?.startsWith("p")
        ? +position.slice(1)
        : note.stringTuning +
          (position?.startsWith("f") ? +position.slice(1) : note.fret + 1);
      note.trillSpeed = Number(duration) || 16;
    } else if (effect.startsWith("trem"))
      note.beat.tremoloSpeed = Number(effect.split(":")[1]) || 16;
  }
}
function beatEffects(beat, effects) {
  for (const effect of effects) {
    if (effect === "pm") beat.isPalmMute = true;
    else if (effect === "let") beat.isLetRing = true;
    else if (effect === "vib") beat.vibrato = M.VibratoType.Slight;
    else if (effect === "fade") beat.fadeIn = true;
    else if (effect === "pick_up") beat.pickStroke = M.PickStroke.Up;
    else if (effect === "pick_down") beat.pickStroke = M.PickStroke.Down;
    else if (effect === "stroke_up") beat.brushType = M.BrushType.BrushUp;
    else if (effect === "stroke_down") beat.brushType = M.BrushType.BrushDown;
    else if (effect === "tap" || effect === "slap:tapping") beat.tap = true;
    else if (effect === "slap:slapping") beat.slap = true;
    else if (effect === "slap:popping") beat.pop = true;
    else if (effect.startsWith("ottava:"))
      beat.ottava = ottava(+effect.split(":")[1]);
    else if (effect.startsWith("text:"))
      beat.text = decode(effect.slice(effect.indexOf(":") + 1));
    else if (effect.startsWith("tempo:"))
      beat.automations.push(
        M.Automation.buildTempoAutomation(false, 0, +effect.split(":")[1], 2),
      );
  }
  const name = effects.find((e) => e.startsWith("chord:"));
  const diagram = effects.find((e) => e.startsWith("diagram:"));
  if (name || diagram) {
    const chord = new M.Chord();
    chord.name = name ? decode(name.slice(6)) : "";
    chord.showDiagram = !!diagram;
    chord.showFingering = false;
    if (diagram) {
      const [, base, frets, , barres] = diagram.split(":");
      chord.firstFret = Number(base);
      // alphaTab uses the same high-to-low order as the staff tuning.
      chord.strings = frets
        .split("/")
        .reverse()
        .map((f) => (f === "x" ? -1 : Number(f)));
      chord.barreFrets =
        barres === "-"
          ? []
          : [...new Set(barres.split(";").map((b) => Number(b.split("/")[0])))];
    }
    const staff = beat.voice.bar.staff;
    const id = `${chord.name}:${diagram || "name"}`;
    staff.addChord(id, chord);
    beat.chordId = id;
  }
}
function colorStyle(Style, Elements, color) {
  const style = new Style();
  for (const key of Object.values(Elements))
    if (typeof key === "number") style.colors.set(key, color);
  return style;
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
function writtenKey(key, shift) {
  const normalized = key
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

export function engrave(state, settings, edited = null, mode = null) {
  const score = new M.Score();
  const sourceOf = new Map(),
    beats = new Map(),
    tracks = new Map(),
    staves = new Map(),
    keysByStaff = new Map();
  M.Score.resetIds();
  let time = "4/4";
  state.measures.forEach((record, mi) => {
    const data = edited?.index === mi ? edited.measure : record.parsed;
    const profile = measureProfile(record, state);
    const partId = record.part_id || "part-1",
      staffId = `${partId}/${record.staff_id || "staff-1"}`;
    let track = tracks.get(partId),
      staff = staves.get(staffId);
    if (!track) {
      track = new M.Track();
      score.addTrack(track);
      tracks.set(partId, track);
      track.name = record.part_name || "";
      track.playbackInfo.program =
        record.midi_program ?? state.metadata?.midi_program ?? 0;
    }
    const drums = profile.instrument === "drums";
    if (!staff) {
      staff = new M.Staff();
      track.addStaff(staff);
      staves.set(staffId, staff);
      staff.isPercussion = drums;
      staff.stringTuning.tunings = [...profile.tuning];
      const displayMode = mode && mode !== "mixed" ? mode : profile.mode;
      staff.showTablature = displayMode !== "notation";
      staff.showStandardNotation = displayMode !== "tab";
      staff.capo = profile.pitch_context.capo;
      staff.displayTranspositionPitch =
        profile.pitch_context.instrument_transpose ??
        (["guitar", "bass"].includes(profile.instrument) ? -12 : 0);
    }
    const barIndex = record.bar_index ?? mi;
    const master = new M.MasterBar();
    time = data.time_signature || record.score_state?.time || time;
    [master.timeSignatureNumerator, master.timeSignatureDenominator] = time
      .split("/")
      .map(Number);
    master.isRepeatStart = data.bars?.includes("repeat_open") || false;
    master.repeatCount = data.bars?.includes("repeat_close")
      ? data.repeat_count || 2
      : 0;
    master.isDoubleBar = data.bars?.includes("double") || false;
    master.alternateEndings = data.alternate_endings || 0;
    if (data.section) {
      master.section = new M.Section();
      master.section.text = decode(data.section);
    }
    if (data.triplet_feel)
      master.tripletFeel =
        {
          eighth: M.TripletFeel.Triplet8th,
          sixteenth: M.TripletFeel.Triplet16th,
        }[data.triplet_feel] ?? M.TripletFeel.NoTripletFeel;
    if (data.tempo_quarter)
      master.tempoAutomations.push(
        M.Automation.buildTempoAutomation(false, 0, data.tempo_quarter, 2),
      );
    while (score.masterBars.length < barIndex) {
      const missing = new M.MasterBar();
      [missing.timeSignatureNumerator, missing.timeSignatureDenominator] = time
        .split("/")
        .map(Number);
      score.addMasterBar(missing);
    }
    if (!score.masterBars[barIndex]) score.addMasterBar(master);
    while (staff.bars.length < barIndex) {
      const missing = new M.Bar(),
        voice = new M.Voice(),
        rest = new M.Beat();
      staff.addBar(missing);
      missing.addVoice(voice);
      voice.addBeat(rest);
      rest.isEmpty = true;
    }
    const bar = new M.Bar();
    staff.addBar(bar);
    sourceOf.set(bar, { mi });
    bar.clef = drums
      ? M.Clef.Neutral
      : (M.Clef[profile.pitch_context.clef] ??
        (profile.instrument === "bass" ? M.Clef.F4 : M.Clef.G2));
    bar.clefOttava = ottava(profile.pitch_context.clef_octave || 0);
    const key = data.key_signature || keysByStaff.get(staffId);
    if (key) keysByStaff.set(staffId, key);
    if (key) {
      bar.keySignature = writtenKey(
        key,
        profile.pitch_context.instrument_transpose,
      );
      bar.keySignatureType = key.includes("Minor") ? 1 : 0;
    }
    const kind = reviewKind(record),
      color = kind
        ? kind === "failed"
          ? new M.Color(181, 48, 58)
          : new M.Color(153, 102, 18)
        : null;
    if (color) bar.style = colorStyle(M.BarStyle, M.BarSubElement, color);
    // Preserve voice numbers, even when only the second voice has content.
    const count = Math.max(1, ...data.voices.map((v) => v.voice + 1));
    for (let v = 0; v < count; v++) {
      const voice = new M.Voice();
      bar.addVoice(voice);
      const vi = data.voices.findIndex((item) => item.voice === v);
      const events = data.voices[vi]?.events || [];
      let cursor = 0;
      events.forEach((event, ei) => {
        if (event.start > cursor) {
          const gap = new M.Beat();
          gap.isEmpty = true;
          gap.overrideDisplayDuration = event.start - cursor;
          voice.addBeat(gap);
        }
        const beat = new M.Beat();
        voice.addBeat(beat);
        sourceOf.set(beat, { mi, vi, ei });
        beats.set(`${mi}:${vi}:${ei}`, beat);
        beat.duration = event.duration.value;
        beat.dots = event.duration.double_dotted
          ? 2
          : event.duration.dotted
            ? 1
            : 0;
        beat.tupletNumerator = event.duration.tuplet_enters || 1;
        beat.tupletDenominator = event.duration.tuplet_times || 1;
        beat.isEmpty = event.status === "empty";
        beatEffects(beat, event.effects || []);
        if (color)
          beat.style = colorStyle(M.BeatStyle, M.BeatSubElement, color);
        if (event.status === "normal")
          event.notes.forEach((source, ni) => {
            const note = new EngravedNote();
            beat.addNote(note);
            sourceOf.set(note, { mi, vi, ei, ni, string: source.string });
            if (drums) note.percussionArticulation = source.pitch;
            else {
              if (source.string) {
                note.string = profile.tuning.length + 1 - source.string;
                note.fret = source.fret === "x" ? 0 : source.fret;
              }
              if (Number.isFinite(source.pitch)) {
                note.octave = Math.floor(source.pitch / 12);
                note.tone = source.pitch % 12;
                note.writtenPitch =
                  source.pitch -
                  pitchShift(
                    profile.pitch_context,
                    event.effects,
                    profile.instrument,
                  );
              }
              if (source.swap_accidentals)
                note.accidentalMode = M.NoteAccidentalMode.ForceFlat;
            }
            note.isDead = source.fret === "x";
            noteEffects(note, source.effects || []);
            if (color)
              note.style = colorStyle(M.NoteStyle, M.NoteSubElement, color);
          });
        cursor = event.start + ticks(event);
      });
      if (!voice.beats.length) {
        const empty = new M.Beat();
        empty.isEmpty = true;
        voice.addBeat(empty);
      }
    }
  });
  for (const staff of staves.values())
    while (staff.bars.length < score.masterBars.length) {
      const bar = new M.Bar(),
        voice = new M.Voice(),
        rest = new M.Beat();
      staff.addBar(bar);
      bar.addVoice(voice);
      voice.addBeat(rest);
      rest.isEmpty = true;
    }
  score.finish(settings);
  return { score, sourceOf, beats, tracks };
}
