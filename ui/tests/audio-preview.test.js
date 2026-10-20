import test from "node:test";
import assert from "node:assert/strict";
import { createAudioPreview } from "../workbench/audio-preview.js";
import { note, beat, measure, record, state } from "./playback-fixtures.js";

const project = state(record(measure([beat(0, [note(5)])])));
function fakePlayer() {
  return {
    plays: 0, stops: 0, pauses: 0, loads: 0,
    isSoundFontLoaded: false, isReadyForPlayback: false,
    finished: { on(fn) { this.emit = fn; } },
    positionChanged: { on(fn) { this.emit = fn; } },
    currentPosition: { endTick: 3840 }, tickPosition: 0,
    output: { activate() {} },
    loadSoundFont() { this.isSoundFontLoaded = true; },
    loadMidiFile(file) { this.loads++; this.midi = file; this.isReadyForPlayback = this.isSoundFontLoaded; },
    play() { this.plays++; return true; },
    stop() { this.stops++; this.tickPosition = 0; },
    pause() { this.pauses++; },
  };
}

test("stopping while soundfont loads prevents deferred playback", async () => {
  const player = fakePlayer(), statuses = [];
  let resolve;
  const preview = createAudioPreview({ makePlayer: () => player, loadFont: () => new Promise(done => { resolve = done; }) });
  const pending = preview.play(project, 0, { onStatus: status => statuses.push(status) });
  assert.equal(preview.status(), "loading");
  assert.deepEqual(statuses, ["loading"]);
  assert.equal(player.plays, 0);
  preview.stop();
  resolve(new Uint8Array());
  await pending;
  assert.equal(player.plays, 0);
  assert.deepEqual(statuses, ["loading", "idle"]);
  await preview.play(project, 0, { onStatus: status => statuses.push(status) });
  assert.equal(player.plays, 1);
  player.finished.emit();
  assert.equal(preview.status(), "idle");
  assert.deepEqual(statuses, ["loading", "idle", "loading", "playing", "idle"]);
  statuses.length = 0;
  await preview.play(project, 0, { onStatus: status => statuses.push(status) });
  preview.stop();
  assert.deepEqual(statuses, ["playing", "idle"]);
});

test("soundfont errors reset playback and a subsequent request retries", async () => {
  const player = fakePlayer(), statuses = [];
  let attempts = 0;
  const preview = createAudioPreview({ makePlayer: () => player, loadFont: async () => {
    if (++attempts === 1) throw new Error("Load failed");
    return new Uint8Array();
  } });
  const options = { onStatus: status => statuses.push(status) };
  await assert.rejects(preview.play(project, 0, options), /Load failed/);
  assert.equal(preview.status(), "idle");
  assert.deepEqual(statuses, ["loading", "idle"]);
  await preview.play(project, 0, options);
  assert.equal(attempts, 2);
  assert.equal(player.plays, 1);
  assert.deepEqual(statuses, ["loading", "idle", "loading", "playing"]);
  preview.stop();
});

test("pause/resume keeps the loaded song and position; stop returns to the start", async () => {
  const player = fakePlayer(), statuses = [];
  const preview = createAudioPreview({ makePlayer: () => player, loadFont: async () => new Uint8Array() });
  await preview.play(project, 0, { onStatus: status => statuses.push(status) });
  preview.seek(0.5);
  preview.pause();
  assert.equal(preview.status(), "paused");
  assert.equal(player.tickPosition, 1920);
  preview.setSpeed(0.75);
  preview.resume();
  assert.equal(player.tickPosition, 1920);
  assert.equal(player.playbackSpeed, 0.75);
  assert.equal(player.loads, 1);
  assert.equal(player.plays, 2);
  preview.stop();
  assert.equal(player.tickPosition, 0);
  assert.deepEqual(statuses, ["loading", "playing", "paused", "playing", "idle"]);
});

test("loop completion keeps playing and reports the score position until the final finish", async () => {
  const player = fakePlayer(), positions = [];
  const preview = createAudioPreview({ makePlayer: () => player, loadFont: async () => new Uint8Array() });
  preview.setLooping(true);
  preview.setSpeed(0.5);
  await preview.play(state(...project.measures, record(measure([beat(0, [note(7)])]), 1)), 0, {
    onPosition: value => positions.push(value),
  });
  assert.equal(player.isLooping, true);
  assert.equal(player.playbackSpeed, 0.5);
  player.finished.emit();
  assert.equal(preview.status(), "playing");
  player.positionChanged.emit({ currentTick: 4800 });
  assert.equal(positions.at(-1).barIndex, 1);
  assert.equal(positions.at(-1).barTick, 960);
  preview.setLooping(false);
  player.finished.emit();
  assert.equal(preview.status(), "idle");
  assert.equal(positions.at(-1), null);
});

test("only the newest preview starts after shared soundfont loading", async () => {
  const player = fakePlayer();
  let resolve;
  const preview = createAudioPreview({ makePlayer: () => player, loadFont: () => new Promise(done => { resolve = done; }) });
  const first = preview.play(project, 0);
  const second = preview.play(project, 0, { measure: measure([beat(0, [note(7)])]) });
  resolve(new Uint8Array());
  await Promise.all([first, second]);
  assert.equal(player.plays, 1);
  assert.ok(player.midi.events.some(event => event.noteKey === 73));
  preview.stop();
});
