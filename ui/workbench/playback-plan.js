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

export function scorePlayback(state, index, { scope = "score", partId = "all", tempo, ...edited } = {}) {
  const settings = new alphaTab.Settings();
  const { score, sourceOf, tracks } = engrave(state, settings, {
    index, measure: state.measures[index].parsed, ...edited,
  });
  const bar = score.tracks.flatMap(track => track.staves.flatMap(staff => staff.bars))
    .find(bar => sourceOf.get(bar)?.mi === index);
  if (!bar) throw new Error("当前小节尚未准备好。");

  if (tempo !== undefined) {
    for (const master of score.masterBars) master.tempoAutomations.length = 0;
    score.masterBars[0].tempoAutomations.push(alphaTab.model.Automation.buildTempoAutomation(false, 0, tempo, 2));
  }

  // Tempo belongs to the score, including markings on a track we are not playing.
  if (partId !== "all") {
    const track = tracks.get(partId);
    if (!track) throw new Error("请选择要播放的音轨。");
    score.tracks.splice(0, score.tracks.length, track);
  }
  let channel = 0;
  const nextChannel = () => {
    if (channel === 9) channel++;
    return channel++;
  };
  score.tracks.forEach((track, i) => {
    track.index = i;
    const drums = track.staves.every(staff => staff.isPercussion);
    // AlphaSynth accepts virtual channels beyond the 16-channel MIDI wire format.
    track.playbackInfo.primaryChannel = drums ? 9 : nextChannel();
    track.playbackInfo.secondaryChannel = drums ? 9 : nextChannel();
  });

  if (scope === "measure") {
    // A following bar's pickup belongs to that bar, even when it sounds early.
    for (const track of score.tracks)
      for (const staff of track.staves)
        for (const voice of staff.bars[bar.index + 1]?.voices || [])
          for (const beat of voice.beats)
            if (beat.graceType !== alphaTab.model.GraceType.None) beat.notes.length = 0;

    // Only a single-bar preview ignores repeats and jumps. Song playback follows them.
    for (const master of score.masterBars) {
      master.isRepeatStart = false;
      master.repeatCount = 0;
      master.alternateEndings = 0;
      master.directions = null;
    }
    score.rebuildRepeatGroups();
  }

  const midi = new MidiFile();
  const generator = new MidiFileGenerator(score, settings, new AlphaSynthMidiFileHandler(midi));
  generator.generate();
  let ranges = generator.tickLookup.masterBars, start = 0, end = midi.events.at(-1).tick;
  if (scope !== "score") {
    const occurrence = ranges.findIndex(range => range.masterBar === bar.masterBar);
    if (occurrence < 0) throw new Error("当前小节不在反复与跳转的播放顺序中，可选择「仅本节」试听。");
    ranges = scope === "measure" ? [ranges[occurrence]] : ranges.slice(occurrence);
    const graceStart = Math.min(0, ...score.tracks.flatMap(track => track.staves.flatMap(staff =>
      staff.bars[bar.index].voices.flatMap(voice => voice.beats.map(beat => beat.playbackStart)))));
    start = ranges[0].start + graceStart + midi.tickShift;
    if (scope === "measure") end = ranges[0].end + midi.tickShift;
  }
  return {
    midi: scope === "score" ? midi : sliceMidi(midi, start, end),
    positionAt(tick) {
      const absolute = tick + start - midi.tickShift;
      const range = ranges.findLast(range => range.start <= absolute) || ranges[0];
      return { barIndex: range.masterBar.index, barTick: Math.max(0, absolute - range.start) };
    },
  };
}
