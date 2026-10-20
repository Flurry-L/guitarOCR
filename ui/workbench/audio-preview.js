import { synth } from "./vendor/alphaTab.mjs";
import { scorePlayback } from "./playback-plan.js";

async function loadSoundFont() {
  const response = await fetch(new URL("./vendor/soundfont/sonivox.sf2", import.meta.url));
  if (!response.ok) throw new Error("试听音色未能加载，请重试。");
  return new Uint8Array(await response.arrayBuffer());
}

class PreviewAudioOutput extends synth.AlphaSynthScriptProcessorOutput {
  open(bufferTime) {
    // AlphaSynth calls open again after reading sampleRate in its constructor.
    if (this.context) this.onReady();
    else super.open(bufferTime);
  }
}

function createPlayer() {
  const output = new PreviewAudioOutput();
  // Open the device first so synthesis uses its actual sample rate.
  output.open(500);
  const player = new synth.AlphaSynth(output, 500);
  player.masterVolume = 0.7;
  return player;
}

// The application owns selection/cancellation; alphaTab owns MIDI synthesis.
export function createAudioPreview({ makePlayer = createPlayer, loadFont = loadSoundFont } = {}) {
  let player, font, plan, fontLoaded = false, playbackStatus = "idle", onStatus, onPosition, generation = 0;
  let looping = false;

  function setStatus(value) {
    if (playbackStatus === value) return;
    playbackStatus = value;
    onStatus?.(value);
  }
  function activate() {
    if (!player) {
      player = makePlayer();
      player.finished.on(() => {
        // AlphaSynth also emits finished on each loop before returning to its start.
        if (!player.isLooping) {
          plan = null;
          setStatus("idle");
          onPosition?.(null);
        }
      });
      player.positionChanged.on(position => {
        if (plan) onPosition?.({ ...position, ...plan.positionAt(position.currentTick) });
      });
    }
    player.output.activate();
  }
  function stop() {
    generation++;
    plan = null;
    player?.stop();
    setStatus("idle");
    onPosition?.(null);
  }
  function pause() {
    if (playbackStatus === "loading") return stop();
    if (playbackStatus !== "playing") return;
    player.pause();
    setStatus("paused");
  }
  function resume() {
    if (playbackStatus !== "paused") return;
    activate();
    if (!player.play()) throw new Error("播放未能继续，请重新播放。");
    setStatus("playing");
  }
  function setLooping(value) {
    looping = value;
    if (player) player.isLooping = value;
  }
  function seek(fraction) {
    if (plan && (playbackStatus === "playing" || playbackStatus === "paused"))
      player.tickPosition = Math.max(0, Math.min(1, fraction)) * player.currentPosition.endTick;
  }
  async function play(state, index, { onStatus: callback, onPosition: positionCallback, ...options } = {}) {
    stop();
    const request = generation;
    onStatus = callback;
    onPosition = positionCallback;
    try {
      plan = scorePlayback(state, index, options);
      if (!fontLoaded) setStatus("loading");
      activate();
      if (!fontLoaded) {
        font ||= loadFont().catch(error => { font = null; throw error; });
        const bytes = await font;
        if (request !== generation) return;
        player.loadSoundFont(bytes, false);
        fontLoaded = player.isSoundFontLoaded;
      }
      player.loadMidiFile(plan.midi);
      player.isLooping = looping;
      if (!player.isReadyForPlayback || !player.play())
        throw new Error("试听播放器未能就绪，请重试。");
      setStatus("playing");
    } catch (error) {
      if (request !== generation) return;
      stop();
      throw error;
    }
  }
  return { activate, play, stop, pause, resume, setLooping, seek, status: () => playbackStatus };
}

export const playback = createAudioPreview();
