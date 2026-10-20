import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createAudioPreview } from "../workbench/audio-preview.js";
import { beat, measure, record, state } from "./playback-fixtures.js";

const font = new Uint8Array(await readFile(new URL("../workbench/vendor/soundfont/sonivox.sf2", import.meta.url)));
const bufferFrames = 4096;

// Only the audio device is simulated: alphaTab generates and buffers the real PCM.
function audioDevice(t, sampleRate) {
  const contexts = [];
  class AudioContext {
    sampleRate = sampleRate;
    state = "running";
    destination = {};
    frames = 0;
    constructor() { contexts.push(this); }
    createBuffer() { return {}; }
    createBufferSource() { return { connect() {}, disconnect() {}, start() {}, stop() {} }; }
    createScriptProcessor() {
      this.node = { connect() {}, disconnect() {} };
      return this.node;
    }
    close() {}
    render() {
      const channels = [new Float32Array(bufferFrames), new Float32Array(bufferFrames)];
      this.node.onaudioprocess({ outputBuffer: { getChannelData: i => channels[i] } });
      this.frames += bufferFrames;
      return channels[0];
    }
  }
  const previous = Object.getOwnPropertyDescriptor(globalThis, "AudioContext");
  Object.defineProperty(globalThis, "AudioContext", { configurable: true, value: AudioContext });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "AudioContext", previous);
    else delete globalThis.AudioContext;
  });
  return contexts;
}

function guitarPitch(chunks, sampleRate) {
  const sample = index => chunks[Math.floor(index / bufferFrames)][index % bufferFrames];
  const start = Math.round(sampleRate * 0.05), count = Math.round(sampleRate * 0.03);
  let bestLag, bestDifference = Infinity;
  for (let lag = Math.floor(sampleRate / 500); lag <= Math.ceil(sampleRate / 380); lag++) {
    let difference = 0, energy = 0;
    for (let i = start; i < start + count; i++) {
      const a = sample(i), b = sample(i + lag);
      difference += (a - b) ** 2;
      energy += a * a + b * b;
    }
    if (difference / energy < bestDifference) {
      bestDifference = difference / energy;
      bestLag = lag;
    }
  }
  return sampleRate / bestLag;
}

for (const sampleRate of [44100, 48000, 96000, 192000]) {
  for (const bpm of [60, 120, 240]) {
    test(`real audio stays at ${bpm} BPM on a ${sampleRate} Hz device`, async t => {
      const contexts = audioDevice(t, sampleRate);
      const project = state(record({
        ...measure([0, 960, 1920, 2880].map(start => beat(start, [{ pitch: 69 }]))),
        tempo_quarter: bpm,
      }, 0, { mode: "notation", instrument: "pitched", tuning: [], capo: 0, midi_program: 24 }));
      const preview = createAudioPreview({ loadFont: async () => font });
      t.after(() => preview.stop());
      let position;
      await preview.play(project, 0, { onPosition: value => { if (value) position = value; } });
      assert.equal(contexts.length, 1, "playback opens one audio device");
      const device = contexts[0], duration = 240 / bpm;
      const samples = [];
      while (device.frames / sampleRate < duration + 0.5 && preview.status() === "playing") {
        samples.push(device.render());
        const seconds = device.frames / sampleRate;
        if (seconds < duration - 0.1) {
          assert.ok(Math.abs(position.currentTime / 1000 - seconds) < 0.002,
            `audio device played ${seconds}s but score advanced ${position.currentTime / 1000}s`);
        }
      }
      assert.equal(preview.status(), "idle", "playback finishes at the score duration");
      assert.ok(device.frames / sampleRate >= duration);
      assert.ok(samples.some(chunk => chunk.some(value => Math.abs(value) > 0.001)));
      assert.ok(samples.every(chunk => chunk.every(Number.isFinite)));
      const pitch = guitarPitch(samples, sampleRate);
      assert.ok(Math.abs(pitch - 440) < 5, `A4 sounds at ${pitch} Hz, expected 440 Hz`);
    });
  }
}
