import { alphaTab, engrave } from "./score-engraving.js";

const { MidiFile, MidiFileGenerator, AlphaSynthMidiFileHandler, MidiEventType: E, EndOfTrackEvent } = alphaTab.midi;

// Trim generated MIDI, including notes held across the start/end of the preview.
// alphaTab owns all musical interpretation; this only selects the time window.
export function sliceMidi(source, start, end) {
  const result = new MidiFile(), pending = new Map();
  result.division = source.division;
  const add = (event, tick) => result.addEvent(Object.assign(
    Object.create(Object.getPrototypeOf(event)), event, { tick },
  ));
  for (const event of source.events) {
    const key = `${event.channel}:${event.noteKey}`;
    if (event.type === E.NoteOn) {
      if (!pending.has(key)) pending.set(key, []);
      pending.get(key).push(event);
    } else if (event.type === E.NoteOff) {
      const onset = pending.get(key)?.shift();
      if (onset && onset.tick < end && event.tick > start) {
        add(onset, Math.max(0, onset.tick - start));
        add(event, Math.min(end, event.tick) - start);
      }
    } else if (event.type !== E.EndOfTrack && event.tick < end) {
      // Replay earlier tempo, program, controllers and per-note bends at zero.
      add(event, Math.max(0, event.tick - start));
    }
  }
  result.addEvent(new EndOfTrackEvent(0, end - start));
  return result;
}

export function measurePlayback(state, index, edited = {}) {
  const settings = new alphaTab.Settings();
  const { score, sourceOf } = engrave(state, settings, {
    index, measure: state.measures[index].parsed, ...edited,
  });
  const bar = score.tracks.flatMap(track => track.staves.flatMap(staff => staff.bars))
    .find(bar => sourceOf.get(bar)?.mi === index);
  if (!bar) throw new Error("当前小节尚未准备好。");

  // Retain neighbouring bars for ties/slides and global tempo from every track,
  // but generate sound only for the selected staff.
  const staff = bar.staff, track = staff.track;
  score.tracks.splice(0, score.tracks.length, track);
  track.index = 0;
  track.staves.splice(0, track.staves.length, staff);
  staff.index = 0;
  track.playbackInfo.primaryChannel = staff.isPercussion ? 9 : 0;
  track.playbackInfo.secondaryChannel = staff.isPercussion ? 9 : 1;

  // The next bar's grace notes can start before its barline. Keep their timing
  // and links for alphaTab, but omit their sound from this bar's preview.
  for (const voice of staff.bars[bar.index + 1]?.voices || [])
    for (const beat of voice.beats)
      if (beat.graceType !== alphaTab.model.GraceType.None) beat.notes.length = 0;

  // A single-bar preview visits every bar once, including bars skipped by jumps.
  for (const master of score.masterBars) {
    master.isRepeatStart = false;
    master.repeatCount = 0;
    master.alternateEndings = 0;
    master.directions = null;
  }
  score.rebuildRepeatGroups();

  const midi = new MidiFile();
  const generator = new MidiFileGenerator(score, settings, new AlphaSynthMidiFileHandler(midi));
  generator.generate();
  const range = generator.tickLookup.masterBars.find(range => range.masterBar === bar.masterBar);
  const graceStart = Math.min(0, ...bar.voices.flatMap(voice => voice.beats.map(beat => beat.playbackStart)));
  const start = range.start + graceStart + midi.tickShift, end = range.end + midi.tickShift;
  const preview = sliceMidi(midi, start, end);
  if (!preview.events.some(event => event.type === E.NoteOn))
    throw new Error("当前小节没有可播放的音符。");
  return preview;
}
