import * as alphaTab from "./vendor/alphaTab.mjs";
import { addGraceNotes, noteEffects, beatEffects, swappedAccidental, dynamicValue } from "./score-effects.js";
import { buildMasterBars, writtenKey } from "./score-structure.js";
import { pitchShift, ticks, projectVisibleTechniques } from "./score-model.js";

export const scoreDisplaySettings = Object.freeze({effectBandPaddingBottom: 8});

// alphaTab 1.8.4 exposes the renderer factories through Environment. Reuse
// its native glyphs and layout bands; keep these narrow adaptations together
// so an alphaTab upgrade can be checked against the browser regression.
const factories = alphaTab.Environment.defaultRenderers;
const notation = factories.find(factory => factory.staffId === 'score');
const tab = factories.find(factory => factory.staffId === 'tab');
const elements = alphaTab.NotationElement;
const palmMute = tab?.effectBands.find(band => band.effect.notationElement === elements.EffectPalmMute);
if (notation && palmMute && !notation.effectBands.some(band => band.effect.notationElement === elements.EffectPalmMute)) {
  const effect = Object.create(palmMute.effect);
  // PM continuation lines need their own band beside chord names.
  Object.defineProperty(effect, 'canShareBand', {value: false});
  notation.effectBands.push({...palmMute, effect, shouldCreate: staff => !staff.showTablature});
}
for (const factory of factories) for (const band of factory.effectBands || []) {
  if (band.effect.notationElement === elements.EffectMarker) {
    const effect = Object.create(band.effect);
    // A section name and the first chord can share the same x coordinate.
    Object.defineProperty(effect, 'canShareBand', {value: false});
    band.effect = effect;
  }
  if (band.effect.notationElement === elements.EffectPalmMute) {
    const effect = Object.create(band.effect);
    // A shared TAB band can put P.M. directly on a chord name. Padding
    // cannot separate two glyphs that were assigned the same band.
    Object.defineProperty(effect, 'canShareBand', {value: false});
    band.effect = effect;
  }
  if (band.effect.notationElement !== elements.EffectDynamics) continue;
  const effect = Object.create(band.effect);
  // Unmarked notes use F internally, but that playback default is not a
  // printed source marking. Repeated explicit markings must remain visible.
  effect.shouldCreateGlyph = (_settings, beat) => beat.engravedDynamic === true && !beat.isEmpty;
  band.effect = effect;
}

export { alphaTab };
const M = alphaTab.model;
export function measureProfile(measure) {
  return {
    mode: measure.mode,
    instrument: measure.instrument,
    tuning: measure.tuning,
    pitch_context: {
      ...(measure.pitch_context || {}),
      capo: measure.capo,
    },
  };
}
export function reviewKind(measure) {
  if (!measure.needs_review) return "";
  return measure.timing_errors?.length
    ? "failed"
    : "review";
}
// Serialized Ottavia values; upstream's minifier renames members starting with '_'.
const ottava = (value) =>
  ({ 24: 0, 12: 1, "-12": 3, "-24": 4 })[value] ?? M.Ottavia.Regular;
// OCR can disagree between the written note and its TAB fret. Display both
// readings faithfully until the user corrects them. Everything else, including
// spacing, beams, accidentals and ties, is engraved by alphaTab.
class EngravedNote extends M.Note {
  writtenPitch;
  get displayValueWithoutBend() {
    return this.writtenPitch ?? super.displayValueWithoutBend;
  }
}

export function engrave(state, settings, edited = null, mode = null) {
  const score = new M.Score();
  const sourceOf = new Map(),
    beats = new Map(),
    tracks = new Map(),
    staves = new Map(),
    keysByStaff = new Map(),
    dynamicsByVoice = new Map();
  M.Score.resetIds();
  const metadata = state.metadata || {}, document = metadata.document_metadata || {};
  score.title = metadata.title || "";
  score.artist = metadata.artist || "";
  score.subTitle = document.subtitle || "";
  score.notices = document.header_notes || "";
  const records = state.measures.map((record, mi) => ({record, mi, data: projectVisibleTechniques(
    edited?.index === mi ? edited.measure : record.parsed,
    edited?.index === mi && Object.hasOwn(edited, "visualTechniques") ? edited.visualTechniques : record.visual_techniques)}));
  const warnings = buildMasterBars(score, records, state);
  const incomplete = records.filter(({record}) => record.parsed.incomplete).map(({record, mi}) => (record.bar_index ?? mi) + 1);
  if (incomplete.length) warnings.push(`第 ${[...new Set(incomplete)].join("、")} 小节尚未补全，空白谱格不表示休止。请选中小节，对照原谱补全音符。`);
  records.sort((a, b) => (a.record.bar_index ?? a.mi) - (b.record.bar_index ?? b.mi));
  records.forEach(({record, mi, data}) => {
    const profile = measureProfile(record);
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
    if (data.bars?.includes("final")) bar.barLineRight = M.BarLineStyle.LightHeavy;
    sourceOf.set(bar, { mi });
    bar.clef = drums
      ? M.Clef.Neutral
      : (M.Clef[profile.pitch_context.clef] ??
        (profile.instrument === "bass" ? M.Clef.F4 : M.Clef.G2));
    bar.clefOttava = ottava(profile.pitch_context.clef_octave || 0);
    const key = data.key_signature ?? keysByStaff.get(staffId) ?? record.score_state?.key;
    if (key !== undefined && key !== null) {
      keysByStaff.set(staffId, key);
      bar.keySignature = writtenKey(
        key,
        profile.pitch_context.instrument_transpose,
      );
      bar.keySignatureType = String(key).includes("Minor") ? 1 : 0;
    }
    // Preserve voice numbers, even when only the second voice has content.
    const count = Math.max(1, ...data.voices.map((v) => v.voice + 1));
    for (let v = 0; v < count; v++) {
      const voice = new M.Voice();
      bar.addVoice(voice);
      const vi = data.voices.findIndex((item) => item.voice === v);
      const events = data.voices[vi]?.events || [];
      let cursor = 0;
      const dynamicKey = `${staffId}/${v}`;
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
        beat.dynamics = dynamicsByVoice.get(dynamicKey) ?? M.DynamicValue.F;
        beatEffects(beat, event.effects || []);
        dynamicsByVoice.set(dynamicKey, beat.dynamics);
        if (event.status === "normal")
          event.notes.forEach((source, ni) => {
            const note = new EngravedNote();
            note.beat = beat;
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
                note.accidentalMode = swappedAccidental(note.displayValueWithoutBend, bar.keySignature);
            }
            note.isDead = source.fret === "x";
            // addNote indexes the already assigned string for ties and hammer-ons.
            beat.addNote(note);
            note.dynamics = source.velocity !== undefined && source.velocity !== 95
              ? dynamicValue(source.velocity) : beat.dynamics;
            noteEffects(note, source.effects || []);
            addGraceNotes(note, source, profile, sourceOf, {mi, vi, ei, ni, string: source.string},
              EngravedNote, pitchShift(profile.pitch_context, event.effects, profile.instrument));
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
  return { score, sourceOf, beats, tracks, warnings };
}
