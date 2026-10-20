import * as alphaTab from "./vendor/alphaTab.mjs";
const M = alphaTab.model;
import { decodeScoreText } from "./score-theory.js";
const ottava = (value) => ({24: 0, 12: 1, "-12": 3, "-24": 4})[value] ?? M.Ottavia.Regular;
export const dynamicValue = (velocity) => Math.max(0, Math.min(7, Math.floor((velocity + 1) / 16) - 1));
export function addBeatText(beat, text) {
  beat.text = [beat.text, text].filter(Boolean).join(" · ");
}
export function parsePosition(position = "") {
  const match = /^(?:f(\d+))?(?:p(\d+))?$/.exec(position);
  return match ? {
    fret: match[1] === undefined ? undefined : Number(match[1]),
    pitch: match[2] === undefined ? undefined : Number(match[2]),
  } : {};
}

// GP5 integer touch positions and alphaTab fractional harmonic nodes use
// different representations (the same conversion as alphaTab's GP importer).
const harmonicValue = fret => ({2: 2.4, 3: 3.2, 8: 8.2, 10: 9.6, 14: 14.7, 15: 14.7, 21: 21.7, 22: 21.7})[fret]
  ?? ([4, 5, 7, 9, 12, 16, 17, 19, 24].includes(fret) ? fret : 12);

export function harmonicSettings(effect, fret, stringTuning) {
  const [, kind, position, accidental, octave] = effect.split(":");
  const harmonicType = {
    natural: M.HarmonicType.Natural, artificial: M.HarmonicType.Artificial,
    pinch: M.HarmonicType.Pinch, tapped: M.HarmonicType.Tap, semi: M.HarmonicType.Semi,
  }[kind] ?? M.HarmonicType.None;
  let value = kind === "natural" ? fret : kind === "tapped" ? Number(position ?? fret + 12) : 12;
  if (kind === "artificial" && octave !== undefined) {
    const base = (fret + stringTuning) % 12;
    value = Number(position) + Number(accidental) + 12 * Number(octave) - base;
    if (value < 0) value += 12;
  }
  return { harmonicType, harmonicValue: harmonicValue(value) };
}

// Convert M bend units to alphaTab's normalized engraving points.
export function bendPoints(effect) {
  const [, kind, value] = effect.split(":"), height = Number(value || 100) / 50;
  const curves = {
    bend: [[0, 0], [0.5, height], [1, height]],
    bendRelease: [[0, 0], [0.5, height], [1, 0]],
    bendReleaseBend: [[0, 0], [1 / 3, height], [2 / 3, 0], [1, height]],
    prebend: [[0, height], [1, height]],
    prebendRelease: [[0, height], [0.5, height], [1, 0]],
  };
  return curves[kind] || curves.bend;
}

// Change enharmonic spelling in either direction, respecting the current key.
export function swappedAccidental(pitch, key) {
  const chroma = ((pitch % 12) + 12) % 12;
  const naturals = [0, 2, 4, 5, 7, 9, 11];
  const keyDegrees = (key < 0 ? [6, 2, 5, 1, 4, 0, 3] : [3, 0, 4, 1, 5, 2, 6]).slice(0, Math.abs(key));
  let offset;
  for (let degree = 0; degree < 7; degree++) {
    const accidental = keyDegrees.includes(degree) ? Math.sign(key) : 0;
    if ((naturals[degree] + accidental + 12) % 12 === chroma) { offset = accidental; break; }
  }
  offset ??= naturals.includes(chroma) ? 0 : key < 0 ? -1 : 1;
  return offset < 0 ? M.NoteAccidentalMode.ForceSharp :
    offset > 0 ? M.NoteAccidentalMode.ForceFlat : M.NoteAccidentalMode.Default;
}

export function addGraceNotes(note, source, profile, sourceOf, location, NoteClass, writtenShift) {
  for (const effect of source.effects || []) {
    if (effect !== "grace" && !effect.startsWith("grace:")) continue;
    const [, position, ...fields] = effect.split(":");
    const auxiliary = parsePosition(position);
    if (auxiliary.fret === undefined && auxiliary.pitch === undefined && position !== "x") {
      addBeatText(note.beat, "装饰音（音高待确认）");
      continue;
    }
    const duration = /^\d+$/.test(fields[0] || "") ? Number(fields.shift()) : 32;
    const [transition, onBeat, dead] = fields;
    const beat = new M.Beat(), grace = new NoteClass();
    beat.duration = [8, 16, 32, 64, 128].includes(duration) ? duration : 32;
    beat.graceType = onBeat === "1" ? M.GraceType.OnBeat : M.GraceType.BeforeBeat;
    beat.dynamics = note.beat.dynamics;
    note.beat.voice.addGraceBeat(beat);
    grace.beat = beat;
    grace.string = note.string;
    const stringPitch = profile.tuning[(source.string || 1) - 1];
    grace.fret = auxiliary.fret ?? (auxiliary.pitch !== undefined && note.string > 0 ? auxiliary.pitch - stringPitch : note.fret);
    const pitch = auxiliary.pitch ?? (auxiliary.fret !== undefined && note.string > 0 ? stringPitch + auxiliary.fret : source.pitch);
    if (Number.isFinite(pitch)) {
      grace.tone = pitch % 12;
      grace.octave = Math.floor(pitch / 12);
      grace.writtenPitch = pitch - writtenShift;
    }
    grace.isDead = position === "x" || dead === "1";
    beat.addNote(grace);
    grace.dynamics = note.dynamics;
    if (transition === "slide") { grace.slideOutType = M.SlideOutType.Legato; grace.slideTarget = note; }
    if (transition === "hammer") grace.isHammerPullOrigin = true;
    const mainPitch = source.pitch ?? (Number.isFinite(stringPitch) && Number.isFinite(source.fret) ? stringPitch + source.fret : undefined);
    if (transition === "bend" && Number.isFinite(pitch) && Number.isFinite(mainPitch)) {
      grace.addBendPoint(new M.BendPoint(0, 0));
      grace.addBendPoint(new M.BendPoint(60, (mainPitch - pitch) * 2));
    }
    // Grace is edited through the source note's effect inspector.
    sourceOf.set(beat, location);
    sourceOf.set(grace, {...location, string: source.string});
  }
}

export function noteEffects(note, effects) {
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
      Object.assign(note, harmonicSettings(effect, note.fret, note.stringTuning));
    } else if (effect.startsWith("bend:")) {
      for (const [position, semitones] of bendPoints(effect))
        note.addBendPoint(new M.BendPoint(position * 60, semitones * 2));
    } else if (effect.startsWith("trill")) {
      const [, position, duration] = effect.split(":");
      const auxiliary = parsePosition(position);
      if (auxiliary.pitch === undefined && auxiliary.fret === undefined) {
        addBeatText(note.beat, "tr（辅助音待确认）");
        continue;
      }
      note.trillValue = auxiliary.pitch ??
        (note.stringTuning + (auxiliary.fret ?? note.fret + 1));
      note.trillSpeed = Number(duration) || 16;
    } else if (effect.startsWith("trem"))
      note.beat.tremoloSpeed = Number(effect.split(":")[1]) || 16;
  }
}
export function beatEffects(beat, effects) {
  for (const effect of effects) {
    if (effect === "pm") beat.isPalmMute = true;
    else if (effect === "let") beat.isLetRing = true;
    else if (effect === "vib") beat.vibrato = M.VibratoType.Slight;
    else if (effect === "rasg") beat.rasgueado = M.Rasgueado.Ii;
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
      addBeatText(beat, decodeScoreText(effect.slice(effect.indexOf(":") + 1)));
  }
  const dynamic = effects.find((effect) => effect.startsWith("dyn:"));
  if (dynamic) {
    beat.dynamics = dynamicValue(Number(dynamic.slice(4)));
    beat.engravedDynamic = true;
  }
  const name = effects.find((e) => e.startsWith("chord:"));
  const diagram = effects.find((e) => e.startsWith("diagram:"));
  if (name || diagram) {
    const chord = new M.Chord();
    chord.name = name ? decodeScoreText(name.slice(6)) : "";
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
