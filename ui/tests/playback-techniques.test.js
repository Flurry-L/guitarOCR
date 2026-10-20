import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { scorePlayback, sliceMidi } from "../workbench/playback-plan.js";
import { midi, synth } from "../workbench/vendor/alphaTab.mjs";
import { note, beat, measure, record, state } from "./playback-fixtures.js";

const E = midi.MidiEventType;
const events = (file, type) => file.events.filter(event => event.type === type);
const measurePlayback = (project, index, edited = {}) => scorePlayback(project, index, {
  scope: "measure", partId: project.measures[index].part_id, ...edited,
}).midi;
const play = (music, fields) => measurePlayback(state(record(music, 0, fields)), 0);

test("alphaTab generates tied notes, capo and the selected instrument without changing the draft", () => {
  const music = measure([beat(0, [note(5)]), beat(960, [note(5, ["tie"])]), beat(1920, [note(7)])]);
  const before = structuredClone(music), file = play(music, { midi_program: 25 });
  assert.deepEqual(events(file, E.NoteOn).map(event => [event.tick, event.noteKey]), [[0, 71], [1920, 73]]);
  assert.deepEqual(events(file, E.NoteOff).map(event => event.tick), [1920, 2880]);
  assert.ok(events(file, E.ProgramChange).every(event => event.program === 25));
  assert.deepEqual(music, before);
});

test("a preview sounds a cross-bar tie and clips it at the selected barline", () => {
  const first = record(measure([beat(0, [note(7, ["harm:natural", "bend:bend:100"])], [], 1)]));
  const middle = record(measure([beat(0, [note(7, ["tie"])], [], 1)]), 1);
  const last = record(measure([beat(0, [note(7, ["tie"])], [], 1)]), 2);
  const file = measurePlayback(state(first, middle, last), 1);
  assert.deepEqual(events(file, E.NoteOn).map(event => [event.tick, event.noteKey]), [[0, 85]]);
  assert.equal(events(file, E.NoteOff)[0].tick, 3840);
  assert.ok(events(file, E.PerNotePitchBend).some(event => event.value > 0x80000000));
  assert.ok(file.events.every(event => event.tick >= 0 && event.tick <= 3840));
});

test("only the selected track sounds while tempo follows the full score", () => {
  const first = record({ ...measure([beat(0, [note(0)], ["tempo:80"])]), tempo_quarter: 120 });
  const current = record(measure([beat(0, [note(5)]), beat(960, [note(7)])]), 1);
  const other = record(measure([beat(960, [{ pitch: 48 }], ["tempo:60"])]), 1,
    { part_id: "bass", staff_id: "bass", instrument: "pitched", mode: "notation", tuning: [], capo: 0 });
  const file = measurePlayback(state(first, current, other), 1);
  assert.deepEqual(events(file, E.NoteOn).map(event => event.noteKey), [71, 73]);
  const tempos = events(file, E.TempoChange);
  assert.equal(tempos.filter(event => event.tick === 0).at(-1).beatsPerMinute, 80);
  assert.equal(tempos.find(event => event.tick === 960).beatsPerMinute, 60);
});

test("preview uses unsaved notes and articulations", () => {
  const saved = measure([beat(0, [note(0)])]);
  const draft = measure([beat(0, [note(5, ["stacc"])])]);
  const project = state(record(saved));
  const file = measurePlayback(project, 0, { measure: draft });
  assert.equal(events(file, E.NoteOn)[0].noteKey, 71);
  assert.ok(events(file, E.NoteOff)[0].tick < 960);
  assert.deepEqual(project.measures[0].parsed, saved);
});

test("single-bar preview remains available after Fine and inside alternate endings", () => {
  const project = state(
    record({ ...measure([beat(0, [note(0)])]), bars: ["repeat_open"], direction: "Fine" }),
    record({ ...measure([beat(0, [note(2)])]), bars: ["repeat_close"], repeat_count: 2,
      alternate_endings: 4, from_direction: "Da Capo al Fine" }, 1),
    record(measure([beat(0, [note(5)])]), 2),
  );
  const before = structuredClone(project);
  for (const [index, pitch] of [[1, 68], [2, 71]]) {
    const file = measurePlayback(project, index);
    assert.deepEqual(events(file, E.NoteOn).map(event => [event.tick, event.noteKey]), [[0, pitch]]);
    assert.equal(events(file, E.EndOfTrack)[0].tick, 3840);
  }
  assert.deepEqual(project, before);
});

test("alphaTab handles grace notes, tremolo picking and trills", () => {
  for (const effect of ["grace:f2:32:hammer:0:0", "trem:16", "trill:f7:16"]) {
    const file = play(measure([beat(0, [note(5, [effect])])]));
    assert.ok(events(file, E.NoteOn).length > 1, effect);
    assert.ok(file.events.every(event => Number.isFinite(event.tick)), effect);
  }
});

test("a following bar's pickup grace note stays outside the selected preview", () => {
  const project = state(
    record(measure([beat(0, [note(0)], [], 1)])),
    record(measure([beat(0, [note(7, ["grace:f5:32:hammer:0:0"])], [], 1)]), 1),
  );
  const before = structuredClone(project);
  assert.deepEqual(events(measurePlayback(project, 0), E.NoteOn).map(event => event.noteKey), [66]);
  assert.deepEqual(events(measurePlayback(project, 1), E.NoteOn).map(event => event.noteKey), [71, 73]);
  assert.deepEqual(project, before);
});

test("legato slides keep one attack while shift slides reattack", () => {
  for (const [effect, attacks] of [["sl", 1], ["ss", 2]]) {
    const file = play(measure([beat(0, [note(2, [effect])]), beat(960, [note(7)])]));
    assert.equal(events(file, E.NoteOn).length, attacks, effect);
    assert.ok(events(file, E.PerNotePitchBend).some(event => event.value !== 0x80000000));
  }
});

test("simultaneous bends retain separate note pitch events", () => {
  const file = play(measure([beat(0, [note(0, ["bend:bend:100"]), note(3, ["bend:bend:50"], 2)])]));
  const bentKeys = new Set(events(file, E.PerNotePitchBend).filter(event => event.value > 0x80000000).map(event => event.noteKey));
  assert.deepEqual(bentKeys, new Set([66, 64]));
});

test("drums use percussion channel and notation retains sounding pitch", () => {
  const drums = play(measure([beat(0, [{ pitch: 36 }, { pitch: 42 }])]),
    { instrument: "drums", mode: "notation", tuning: [], capo: 0 });
  assert.deepEqual(events(drums, E.NoteOn).map(event => event.noteKey), [36, 42]);
  assert.ok(events(drums, E.NoteOn).every(event => event.channel === 9));
  const clarinet = play(measure([beat(0, [{ pitch: 58 }])]),
    { instrument: "pitched", mode: "notation", tuning: [], capo: 0, pitch_context: { instrument_transpose: -2 } });
  assert.equal(events(clarinet, E.NoteOn)[0].noteKey, 58);
});

test("MIDI trimming preserves attacks at the start and excludes attacks at the end", () => {
  const file = new midi.MidiFile(), handler = new midi.AlphaSynthMidiFileHandler(file);
  handler.addTempo(0, 100);
  handler.addNote(0, 0, 960, 60, 95, 0);
  handler.addNote(0, 960, 960, 60, 95, 0);
  handler.addNote(0, 1920, 960, 64, 95, 0);
  const cut = sliceMidi(file, 960, 1920);
  assert.deepEqual(events(cut, E.NoteOn).map(event => [event.tick, event.noteKey]), [[0, 60]]);
  assert.deepEqual(events(cut, E.NoteOff).map(event => event.tick), [960]);
  assert.equal(events(cut, E.TempoChange)[0].beatsPerMinute, 100);
  assert.doesNotThrow(() => cut.toBinary());
});

test("song playback mixes independent instrument channels and reserves channel 9 for drums", () => {
  const project = state(
    record(measure([beat(0, [note(0)])]), 0, { midi_program: 24 }),
    record(measure([beat(0, [note(2)])]), 1, { midi_program: 24 }),
    record(measure([beat(0, [{ pitch: 40 }])]), 0,
      { part_id: "bass", instrument: "bass", mode: "notation", tuning: [], capo: 0, midi_program: 33 }),
    record(measure([beat(0, [{ pitch: 36 }])]), 0,
      { part_id: "drums", instrument: "drums", mode: "notation", tuning: [], capo: 0 }),
  );
  const before = structuredClone(project), { midi: file, positionAt } = scorePlayback(project, 1);
  const notes = events(file, E.NoteOn), programs = new Map(events(file, E.ProgramChange).map(e => [e.channel, e.program]));
  assert.equal(notes.length, 4);
  assert.equal(programs.get(notes.find(n => n.noteKey === 66).channel), 24);
  assert.equal(programs.get(notes.find(n => n.noteKey === 40).channel), 33);
  assert.equal(notes.find(n => n.noteKey === 36).channel, 9);
  assert.equal(notes.find(n => n.noteKey === 68).tick, 3840);
  assert.deepEqual(positionAt(4800), { barIndex: 1, barTick: 960 });
  assert.deepEqual(project, before);
});

test("whole-song and from-current playback follow repeats and map repeated bars", () => {
  const project = state(
    record({ ...measure([beat(0, [note(0)])]), bars: ["repeat_open"] }),
    record({ ...measure([beat(0, [note(2)])]), bars: ["repeat_close"], repeat_count: 2 }, 1),
    record(measure([beat(0, [note(5)])]), 2),
  );
  const full = scorePlayback(project, 1);
  assert.deepEqual(events(full.midi, E.NoteOn).map(e => e.noteKey), [66, 68, 66, 68, 71]);
  assert.deepEqual([0, 3840, 7680, 11520, 15360].map(t => full.positionAt(t).barIndex), [0, 1, 0, 1, 2]);
  const from = scorePlayback(project, 1, { scope: "from" });
  assert.deepEqual(events(from.midi, E.NoteOn).map(e => e.noteKey), [68, 66, 68, 71]);
  assert.equal(from.positionAt(0).barIndex, 1);
  assert.equal(from.positionAt(3840).barIndex, 0);
});

test("starting on a held note keeps its remaining cross-bar duration and inherited tempo", () => {
  const project = state(
    record({ ...measure([beat(0, [note(5)], [], 1)]), tempo_quarter: 80 }),
    record(measure([beat(0, [note(5, ["tie"])], [], 1)]), 1),
    record(measure([beat(0, [note(5, ["tie"])], [], 1)]), 2),
  );
  const file = scorePlayback(project, 1, { scope: "from" }).midi;
  assert.deepEqual(events(file, E.NoteOn).map(e => [e.tick, e.noteKey]), [[0, 71]]);
  assert.equal(events(file, E.NoteOff)[0].tick, 7680);
  assert.equal(events(file, E.TempoChange).at(-1).beatsPerMinute, 80);
});

test("song playback follows alternate endings and Da Capo al Fine", () => {
  const project = (...fields) => state(...fields.map((fields, i) =>
    record({ ...measure([beat(0, [note(i * 2)])]), ...fields }, i)));
  const endings = scorePlayback(project(
    { bars: ["repeat_open"] },
    { bars: ["repeat_close"], repeat_count: 2, alternate_endings: 1 },
    { alternate_endings: 2 },
  ), 0);
  assert.deepEqual(events(endings.midi, E.NoteOn).map(e => e.noteKey), [66, 68, 66, 70]);
  assert.deepEqual([0, 3840, 7680, 11520].map(t => endings.positionAt(t).barIndex), [0, 1, 0, 2]);
  const jump = scorePlayback(project({}, { direction: "Fine" }, { from_direction: "Da Capo al Fine" }), 0);
  assert.deepEqual(events(jump.midi, E.NoteOn).map(e => e.noteKey), [66, 68, 70, 66, 68]);
  assert.equal(jump.positionAt(15360).barIndex, 1);
});

test("the selected track includes both staves, and a rest-only bar keeps its duration", () => {
  const project = state(
    record(measure([beat(0, [{ pitch: 72 }])]), 0, { part_id: "piano", instrument: "pitched", tuning: [], capo: 0, mode: "notation" }),
    record(measure([beat(0, [{ pitch: 48 }])]), 0, { part_id: "piano", staff_id: "lower", instrument: "pitched", tuning: [], capo: 0, mode: "notation" }),
    record(measure([beat(0, [note(5)])])),
  );
  const file = scorePlayback(project, 0, { partId: "piano" }).midi;
  assert.deepEqual(events(file, E.NoteOn).map(e => e.noteKey).sort(), [48, 72]);
  const rest = { ...beat(0, [], [], 1), status: "rest" };
  const silent = scorePlayback(state(record(measure([rest]))), 0, { scope: "measure" }).midi;
  assert.equal(events(silent, E.NoteOn).length, 0);
  assert.equal(silent.events.at(-1).tick, 3840);
});

// A sample sink lets the real alphaTab synthesizer run without browser audio.
function sampleSink() {
  const emitter = () => ({ handlers: [], on(fn) { this.handlers.push(fn); }, trigger(value) { this.handlers.forEach(fn => fn(value)); } });
  return { sampleRate: 44100, ready: emitter(), sampleRequest: emitter(), samplesPlayed: emitter(),
    open() { this.ready.trigger(); }, pause() {}, destroy() {}, resetSamples() {}, activate() {}, play() {} };
}

test("bundled soundfont produces real guitar and drum samples through alphaTab", async () => {
  const font = new Uint8Array(await readFile(new URL("../workbench/vendor/soundfont/sonivox.sf2", import.meta.url)));
  const player = new synth.AlphaSynth(sampleSink(), 500);
  for (const file of [
    play(measure([beat(0, [note(5)])])),
    play(measure([beat(0, [{ pitch: 36 }])]), { instrument: "drums", mode: "notation", tuning: [], capo: 0 }),
  ]) {
    const options = new synth.AudioExportOptions();
    options.soundFonts = [font];
    const exporter = player.exportAudio(options, file, [], new Map());
    const chunk = exporter.render(500);
    assert.ok(chunk.samples.some(value => Math.abs(value) > 0.001));
    assert.ok(chunk.samples.every(Number.isFinite));
  }
  player.destroy();
});

test("AlphaSynth scales the full tempo map and changes speed without moving the musical position", () => {
  const project = state(
    record({ ...measure([beat(0, [note(0)], [], 2), beat(1920, [note(2)], ["tempo:60"], 2)]), tempo_quarter: 120 }),
    record({ ...measure([beat(0, [note(2)], [], 1)]), tempo_quarter: 80 }, 1),
  );
  const player = new synth.AlphaSynth(sampleSink(), 500);
  player.loadMidiFile(scorePlayback(project, 0).midi);
  assert.ok(Math.abs(player.currentPosition.endTime - 6000) < 0.01);
  player.tickPosition = 4800;
  const position = player.tickPosition;
  player.playbackSpeed = 0.5;
  assert.ok(Math.abs(player.currentPosition.endTime - 12000) < 0.01);
  assert.equal(player.tickPosition, position);
  assert.equal(player.currentPosition.originalTempo, 80);
  assert.equal(player.currentPosition.modifiedTempo, 40);
  player.playbackSpeed = 2;
  assert.ok(Math.abs(player.currentPosition.endTime - 3000) < 0.01);
  assert.equal(player.tickPosition, position);
  assert.equal(player.currentPosition.modifiedTempo, 160);
  player.destroy();
});

test("AlphaSynth renders more than eight pitched tracks on separate virtual channels", async () => {
  const project = state(...Array.from({ length: 10 }, (_, i) => record(
    measure([beat(0, [note(i)])]), 0, { part_id: `part-${i}`, midi_program: 24 + i },
  )));
  const file = scorePlayback(project, 0).midi;
  const channels = events(file, E.NoteOn).map(e => e.channel);
  assert.equal(new Set(channels).size, 10);
  assert.ok(channels.some(c => c > 15));
  assert.ok(channels.every(c => c !== 9));
  const player = new synth.AlphaSynth(sampleSink(), 500), options = new synth.AudioExportOptions();
  options.soundFonts = [new Uint8Array(await readFile(new URL("../workbench/vendor/soundfont/sonivox.sf2", import.meta.url)))];
  const chunk = player.exportAudio(options, file, [], new Map()).render(500);
  assert.ok(chunk.samples.some(value => Math.abs(value) > 0.001));
  assert.ok(chunk.samples.every(Number.isFinite));
  player.destroy();
});
