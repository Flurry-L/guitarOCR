import { $ } from "./dom.js";
import { playback } from "./audio-preview.js";

const time = milliseconds => {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};

export function initPlaybackControls({ prepare, onPosition, onError }) {
  let enabled = false, request = 0;
  function render() {
    const status = playback.status(), playing = status === "playing", loading = status === "loading";
    $("playScore").textContent = {
      idle: "▶ 播放", loading: "⏳ 加载音色…", playing: "Ⅱ 暂停", paused: "▶ 继续",
    }[status];
    $("playScore").title = loading ? "点击取消加载" : "播放 / 暂停（空格键）";
    $("playScore").setAttribute("aria-pressed", String(playing));
    $("playScore").setAttribute("aria-busy", String(loading));
    $("playScore").disabled = !enabled;
    $("stopScore").disabled = status === "idle";
    $("playbackProgress").disabled = !["playing", "paused"].includes(status);
    for (const id of ["playbackScope", "playbackTracks", "playbackSpeed", "playbackLoop"])
      $(id).disabled = !enabled;
  }
  function position(value) {
    $("playbackProgress").value = value?.endTick ? Math.round(1000 * value.currentTick / value.endTick) : 0;
    $("playbackPosition").textContent = value
      ? `第 ${value.barIndex + 1} 小节 · ${time(value.currentTime)} / ${time(value.endTime)} · ♩ = ${Math.round(value.modifiedTempo)}`
      : "按谱面速度播放";
    onPosition(value);
  }
  function stop() {
    request++;
    playback.stop();
  }
  async function toggle() {
    try {
      switch (playback.status()) {
        case "loading": stop(); return;
        case "playing": playback.pause(); return;
        case "paused": playback.resume(); return;
      }
      if (!$("playbackSpeed").reportValidity()) return;
      const pending = ++request;
      // Keep WebAudio activation in the click gesture, before a text edit is saved.
      playback.activate();
      const context = await prepare();
      if (!context || pending !== request) return;
      const { state, index, measure, visualTechniques } = context;
      playback.setSpeed(Number($("playbackSpeed").value) / 100);
      await playback.play(state, index, {
        measure, visualTechniques,
        scope: $("playbackScope").value,
        partId: $("playbackTracks").value === "all" ? "all" : state.measures[index].part_id || "part-1",
        onStatus: render,
        onPosition: position,
      });
    } catch (error) {
      onError(error.message);
    }
  }
  $("playScore").onclick = toggle;
  $("stopScore").onclick = stop;
  $("playbackScope").onchange = $("playbackTracks").onchange = stop;
  $("playbackSpeed").oninput = () => {
    if ($("playbackSpeed").validity.valid) playback.setSpeed(Number($("playbackSpeed").value) / 100);
  };
  $("playbackSpeed").onchange = () => $("playbackSpeed").reportValidity();
  $("playbackLoop").onchange = () => playback.setLooping($("playbackLoop").checked);
  $("playbackProgress").oninput = () => playback.seek(Number($("playbackProgress").value) / 1000);
  $("scoreCanvas").addEventListener("keydown", event => {
    if (event.key !== " " || event.ctrlKey || event.metaKey || event.altKey) return;
    event.preventDefault();
    if (!event.repeat) {
      if (event.shiftKey) stop();
      else $("playScore").click();
    }
  });
  window.addEventListener("guitarocr:leave-project", stop);
  render();
  return {
    update(available) {
      enabled = available;
      if (!enabled) playback.stop();
      render();
    },
  };
}
