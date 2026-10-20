import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { measurePlayback, sliceMidi } from "../workbench/playback-plan.js";
import { midi, synth } from "../workbench/vendor/alphaTab.mjs";
import { note, beat, measure, record, state } from "./playback-fixtures.js";

const E = midi.MidiEventType;
const events = (file, type) => file.events.filter(event => event.type === type);
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

test("only the selected staff sounds while tempo follows the full score", () => {
  const first = record({ ...measure([beat(0, [note(0)], ["tempo:80"])]), tempo_quarter: 120 });
  const current = record(measure([beat(0, [note(5)]), beat(960, [note(7)])]), 1);
  const other = record(measure([beat(960, [{ pitch: 48 }], ["tempo:60"])]), 1,
    { staff_id: "bass", instrument: "pitched", mode: "notation", tuning: [], capo: 0 });
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
