import test from "node:test";
import assert from "node:assert/strict";
import { createAudioPreview } from "../workbench/audio-preview.js";
import { note, beat, measure, record, state } from "./playback-fixtures.js";

const project = state(record(measure([beat(0, [note(5)])])));
function fakePlayer() {
  return {
    plays: 0, stops: 0, activations: 0,
    isSoundFontLoaded: false, isReadyForPlayback: false,
    finished: { on(fn) { this.emit = fn; } },
    output: { activate() {} },
    loadSoundFont() { this.isSoundFontLoaded = true; },
    loadMidiFile(file) { this.midi = file; this.isReadyForPlayback = this.isSoundFontLoaded; },
    play() { this.plays++; return true; },
    stop() { this.stops++; },
  };
}

test("stopping while soundfont loads prevents deferred playback", async () => {
  const player = fakePlayer(), statuses = [];
  let resolve;
  const preview = createAudioPreview({ makePlayer: () => player, loadFont: () => new Promise(done => { resolve = done; }) });
  const pending = preview.play(project, 0, { onStatus: status => statuses.push(status) });
  assert.equal(preview.isActive(), true);
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
  assert.equal(preview.isActive(), false);
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
  assert.equal(preview.isActive(), false);
  assert.deepEqual(statuses, ["loading", "idle"]);
  await preview.play(project, 0, options);
  assert.equal(attempts, 2);
  assert.equal(player.plays, 1);
  assert.deepEqual(statuses, ["loading", "idle", "loading", "playing"]);
  preview.stop();
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
