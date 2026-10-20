import { synth } from "./vendor/alphaTab.mjs";
import { measurePlayback } from "./playback-plan.js";

async function loadSoundFont() {
  const response = await fetch(new URL("./vendor/soundfont/sonivox.sf2", import.meta.url));
  if (!response.ok) throw new Error("试听音色未能加载，请重试。");
  return new Uint8Array(await response.arrayBuffer());
}

function createPlayer() {
  const player = new synth.AlphaSynth(new synth.AlphaSynthScriptProcessorOutput(), 500);
  player.masterVolume = 0.7;
  return player;
}

// The application owns selection/cancellation; alphaTab owns MIDI synthesis.
export function createAudioPreview({ makePlayer = createPlayer, loadFont = loadSoundFont } = {}) {
  let player, font, fontLoaded = false, playbackStatus = "idle", onStatus, generation = 0;

  function setStatus(value) {
    if (playbackStatus === value) return;
    playbackStatus = value;
    onStatus?.(value);
  }
  function stop() {
    generation++;
    player?.stop();
    setStatus("idle");
  }
  async function play(state, index, { onStatus: callback, ...edited } = {}) {
    stop();
    const request = generation;
    const midi = measurePlayback(state, index, edited);
    onStatus = callback;
    try {
      if (!fontLoaded) setStatus("loading");
      if (!player) {
        player = makePlayer();
        player.finished.on(() => setStatus("idle"));
      }
      // Activate WebAudio during the click, before fetching the bundled font.
      player.output.activate();
      if (!fontLoaded) {
        font ||= loadFont().catch(error => { font = null; throw error; });
        const bytes = await font;
        if (request !== generation) return;
        player.loadSoundFont(bytes, false);
        fontLoaded = player.isSoundFontLoaded;
      }
      player.loadMidiFile(midi);
      if (!player.isReadyForPlayback || !player.play())
        throw new Error("试听播放器未能就绪，请重试。");
      setStatus("playing");
    } catch (error) {
      if (request !== generation) return;
      stop();
      throw error;
    }
  }
  return { play, stop, isActive: () => playbackStatus !== "idle" };
}

const preview = createAudioPreview();
export const playMeasure = preview.play;
export const stopPlayback = preview.stop;
export const isActive = preview.isActive;
