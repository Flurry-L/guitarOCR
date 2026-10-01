import {
  alphaTab,
  engrave,
  measureProfile,
  reviewKind,
} from "./score-engraving.js";
import { el } from "./dom.js";

const contains = (box, x, y, padding = 0) =>
  x >= box.x - padding &&
  x <= box.x + box.w + padding &&
  y >= box.y - padding &&
  y <= box.y + box.h + padding;
function rectangle(node, bounds) {
  Object.assign(node.style, {
    left: `${bounds.x}px`,
    top: `${bounds.y}px`,
    width: `${bounds.w}px`,
    height: `${bounds.h}px`,
  });
}

// The renderer owns engraving and hit bounds; this view
// translates those bounds back to the original measures, voices and notes.
function engravedSection(
  host,
  { onSelect, onPlace, onError, onContext },
  range,
) {
  const surface = el("div", undefined, "engraved-score");
  const overlay = el("div", undefined, "score-overlay");
  host.replaceChildren(surface, overlay);
  const palette = getComputedStyle(document.documentElement);
  const api = new alphaTab.AlphaTabApi(surface, {
    core: {
      useWorkers: false,
      includeNoteBounds: true,
      engine: "svg",
      enableLazyLoading: false,
      smuflFontSources: {
        [alphaTab.FontFileFormat.Woff2]: new URL(
          "./vendor/font/Bravura.woff2",
          import.meta.url,
        ).href,
      },
    },
    display: {
      startBar: range.start + 1,
      barCount: range.end - range.start,
      scale: 1.1,
      layoutMode: alphaTab.LayoutMode.Page,
      padding: [28, 20, 28, 20],
      resources: {
        mainGlyphColor: palette.getPropertyValue("--score-ink").trim(),
        secondaryGlyphColor: palette.getPropertyValue("--score-ink").trim(),
        barNumberColor: palette.getPropertyValue("--score-muted").trim(),
        staffLineColor: palette.getPropertyValue("--score-line").trim(),
      },
    },
    notation: {
      elements: {
        scoreTitle: false,
        scoreSubTitle: false,
        scoreArtist: false,
        scoreAlbum: false,
        scoreWords: false,
        scoreMusic: false,
        scoreCopyright: false,
        guitarTuning: false,
        trackNames: true,
        effectDynamics: false,
      },
    },
    player: {
      enablePlayer: false,
      enableCursor: false,
      enableUserInteraction: false,
      enableElementHighlighting: false,
      scrollMode: alphaTab.ScrollMode.Off,
    },
  });
  let state,
    rendered,
    selected,
    signature,
    pointer,
    scrollToSelection = false,
    focusedPart = "all";
  const visibleTracks = () =>
    focusedPart === "all"
      ? rendered.score.tracks.map((_, i) => i)
      : [rendered.tracks.get(focusedPart)?.index].filter(
          (i) => i !== undefined,
        );
  api.error.on((error) => onError(error.message || String(error)));
  function kinds(bounds) {
    const source = rendered?.sourceOf.get(bounds.beat);
    const mode = source
      ? measureProfile(state.measures[source.mi], state).mode
      : range.mode;
    if (mode !== "both") return mode === "tab" ? "tab" : "notation";
    const bars =
      api.boundsLookup.findMasterBar(bounds.barBounds.bar.masterBar)?.bars ||
      [];
    const last = bars
      .filter((b) => b.bar === bounds.barBounds.bar)
      .sort((a, b) => a.visualBounds.y - b.visualBounds.y)
      .at(-1);
    return last === bounds.barBounds ? "tab" : "notation";
  }
  function writtenPitchAt(bounds, y) {
    const beat = bounds.beat,
      source = rendered.sourceOf.get(beat),
      profile = measureProfile(
        state.measures[source?.mi ?? beat.voice.bar.index],
        state,
      );
    if (profile.instrument === "drums") return null;
    const staff = bounds.barBounds.visualBounds;
    const bottom =
      { G2: 30, F4: 18, C3: 24, C4: 22 }[profile.pitch_context.clef] ??
      (profile.instrument === "bass" ? 18 : 30);
    const step = bottom + Math.round((staff.y + staff.h - y) / (staff.h / 8));
    const degree = ((step % 7) + 7) % 7,
      key = beat.voice.bar.keySignature;
    const altered = (key > 0 ? [3, 0, 4, 1, 5, 2, 6] : [6, 2, 5, 1, 4, 0, 3])
      .slice(0, Math.abs(key))
      .includes(degree);
    return (
      (Math.floor(step / 7) + 1) * 12 +
      [0, 2, 4, 5, 7, 9, 11][degree] +
      (altered ? Math.sign(key) : 0)
    );
  }
  function hitAt(x, y) {
    const lookup = api.boundsLookup;
    let beat = lookup?.getBeatAtPos(x, y);
    if (!beat) return null;
    const master = lookup.findMasterBar(beat.voice.bar.masterBar);
    const candidates = (master?.bars || []).flatMap((bar) => bar.beats);
    // Chords and a second voice may share an x coordinate. Prefer the actual
    // note head before falling back to the surrounding beat.
    let noteHit = null,
      distance = Infinity;
    for (const bounds of candidates)
      for (const note of bounds.notes || []) {
        const b = note.noteHeadBounds;
        if (contains(b, x, y, 5)) {
          const d = Math.hypot(x - b.x - b.w / 2, y - b.y - b.h / 2);
          if (d < distance) {
            distance = d;
            noteHit = { bounds, note: note.note };
          }
        }
      }
    let bounds = noteHit?.bounds;
    if (noteHit) beat = noteHit.note.beat;
    else {
      const matching = candidates.filter(
        (b) => contains(b.realBounds, x, y, 4) && rendered.sourceOf.has(b.beat),
      );
      bounds =
        matching.find(
          (b) => rendered.sourceOf.get(b.beat).vi === selected?.vi,
        ) ||
        matching[0] ||
        (lookup.findBeats(beat) || []).reduce((nearest, b) => {
          const distance = (box) =>
            Math.abs(y - box.visualBounds.y - box.visualBounds.h / 2);
          return !nearest || distance(b) < distance(nearest) ? b : nearest;
        }, null);
      if (bounds) beat = bounds.beat;
    }
    const source = rendered.sourceOf.get(beat);
    if (!bounds || !source) return null;
    const profile = measureProfile(state.measures[source.mi], state),
      kind = kinds(bounds);
    const hit = {
      ...source,
      ni: noteHit ? (rendered.sourceOf.get(noteHit.note)?.ni ?? -1) : -1,
      kind,
    };
    const staff = bounds.barBounds.visualBounds;
    if (kind === "tab") {
      const gap = staff.h / Math.max(1, profile.tuning.length - 1);
      hit.string = Math.max(
        1,
        Math.min(profile.tuning.length, Math.round((y - staff.y) / gap) + 1),
      );
      if (noteHit) hit.string = rendered.sourceOf.get(noteHit.note).string;
    } else if (profile.instrument !== "drums")
      hit.pitch = writtenPitchAt(bounds, y);
    return hit;
  }
  function markSelection() {
    overlay
      .querySelectorAll(".score-selection,.note-selection")
      .forEach((n) => n.remove());
    if (!selected || !rendered || !api.boundsLookup) return;
    const beat = rendered.beats.get(
      `${selected.mi}:${selected.vi}:${selected.ei}`,
    );
    if (!beat) return;
    const boxes = api.boundsLookup.findBeats(beat) || [];
    for (const bounds of boxes) {
      const cursor = el("div", undefined, "score-selection");
      const b = bounds.realBounds;
      rectangle(cursor, {
        x: bounds.onNotesX - 8,
        y: b.y - 5,
        w: Math.max(20, Math.min(b.w, 32)),
        h: b.h + 10,
      });
      overlay.append(cursor);
      if (kinds(bounds) === "tab") {
        if (!Number.isInteger(selected.string)) continue;
        const count = measureProfile(state.measures[selected.mi], state).tuning
          .length;
        const staff = bounds.barBounds.visualBounds;
        const node = el("div", undefined, "note-selection");
        rectangle(node, {
          x: bounds.onNotesX - 6,
          y:
            staff.y +
            ((selected.string - 1) * staff.h) / Math.max(1, count - 1) -
            10,
          w: 25,
          h: 20,
        });
        overlay.append(node);
      } else {
        const note = (bounds.notes || []).find(
          (n) => rendered.sourceOf.get(n.note)?.ni === selected.ni,
        );
        if (note) {
          const n = el("div", undefined, "note-selection");
          const b = note.noteHeadBounds;
          rectangle(n, { x: b.x - 5, y: b.y - 5, w: b.w + 10, h: b.h + 10 });
          overlay.append(n);
        }
      }
    }
    if (scrollToSelection && boxes.length) {
      scrollToSelection = false;
      const cursor = overlay.querySelector(".score-selection");
      const viewport = host.closest(".score-viewport");
      if (cursor && viewport) {
        const rect = cursor.getBoundingClientRect(),
          frame = viewport.getBoundingClientRect();
        const dy =
          rect.top < frame.top + 24
            ? rect.top - frame.top - 24
            : rect.bottom > frame.bottom - 24
              ? rect.bottom - frame.bottom + 24
              : 0;
        const dx =
          rect.left < frame.left + 12
            ? rect.left - frame.left - 12
            : rect.right > frame.right - 12
              ? rect.right - frame.right + 12
              : 0;
        viewport.scrollBy({ top: dy, left: dx, behavior: "instant" });
      }
    }
  }
  function decorate() {
    overlay.replaceChildren();
    if (!api.boundsLookup || !state) return;
    for (let i = 0; i < state.measures.length; i++) {
      const index = state.measures[i].bar_index ?? i;
      if (index < range.start || index >= range.end) continue;
      const kind = reviewKind(state.measures[i]);
      if (!kind) continue;
      const bounds = api.boundsLookup.findMasterBarByIndex(index);
      if (!bounds) continue;
      const local = (bounds.bars || [])
        .filter((b) => rendered.sourceOf.get(b.bar)?.mi === i)
        .map((b) => b.realBounds);
      if (!local.length) continue;
      const x = Math.min(...local.map((b) => b.x)),
        y = Math.min(...local.map((b) => b.y));
      const box = {
        x,
        y,
        w: Math.max(...local.map((b) => b.x + b.w)) - x,
        h: Math.max(...local.map((b) => b.y + b.h)) - y,
      };
      const marker = el("div", undefined, `score-mark ${kind}`);
      rectangle(marker, box);
      const flag = el(
        "button",
        kind === "failed" ? "识别失败" : "待检查",
        "measure-flag",
      );
      flag.type = "button";
      flag.setAttribute("aria-label", `第 ${i + 1} 小节，${flag.textContent}`);
      flag.onclick = () =>
        onSelect({
          mi: i,
          vi: 0,
          ei: 0,
          ni: -1,
          string: 1,
          kind:
            measureProfile(state.measures[i], state).mode === "notation"
              ? "notation"
              : "tab",
        });
      marker.append(flag);
      overlay.append(marker);
    }
    markSelection();
  }
  let viewportPosition = null;
  const viewport = host.closest(".score-viewport");
  function rememberViewport() {
    if (viewport && !viewportPosition)
      viewportPosition = { top: viewport.scrollTop, left: viewport.scrollLeft };
  }
  api.postRenderFinished.on(() => {
    if (viewportPosition && viewport) {
      viewport.scrollTop = viewportPosition.top;
      viewport.scrollLeft = viewportPosition.left;
      viewportPosition = null;
    }
    decorate();
  });
  let width = 0;
  const resize = new ResizeObserver(([entry]) => {
    const next = Math.round(entry.contentRect.width);
    if (next !== width) {
      width = next;
      if (next > 0 && rendered)
        api.renderScore(rendered.score, visibleTracks());
    }
  });
  resize.observe(host);
  function coordinates(event) {
    const b = surface.getBoundingClientRect();
    return { x: event.clientX - b.left, y: event.clientY - b.top };
  }
  host.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest("button")) return;
    const p = coordinates(event),
      hit = hitAt(p.x, p.y);
    if (!hit) return;
    pointer = {
      hit,
      ...p,
      id: event.pointerId,
      drag: false,
      canDrag:
        event.pointerType !== "touch" && hit.kind === "notation" && hit.ni >= 0,
    };
    host.closest("[tabindex]")?.focus({ preventScroll: true });
  });
  host.addEventListener("pointermove", (event) => {
    if (!pointer?.canDrag) return;
    const p = coordinates(event);
    if (Math.abs(p.y - pointer.y) > 6) pointer.drag = true;
    host.classList.toggle("dragging-note", pointer.drag);
  });
  host.addEventListener("pointerup", (event) => {
    if (!pointer || pointer.id !== event.pointerId) return;
    const start = pointer;
    pointer = null;
    host.classList.remove("dragging-note");
    const p = coordinates(event);
    const beat = rendered.beats.get(
      `${start.hit.mi}:${start.hit.vi}:${start.hit.ei}`,
    );
    const bounds = (api.boundsLookup.findBeats(beat) || []).find(
      (b) => kinds(b) === "notation",
    );
    const pitch = bounds ? writtenPitchAt(bounds, p.y) : null;
    if (start.drag && Number.isFinite(pitch)) onPlace(start.hit, pitch, false);
    else onSelect(start.hit);
  });
  const releasePointer = () => {
    pointer = null;
    host.classList.remove("dragging-note");
  };
  host.addEventListener("pointercancel", releasePointer);
  window.addEventListener("pointerup", releasePointer);
  host.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    const p = coordinates(event),
      hit = hitAt(p.x, p.y);
    if (hit) onContext?.(hit, event.clientX, event.clientY);
  });
  host.addEventListener("dblclick", (event) => {
    const p = coordinates(event),
      hit = hitAt(p.x, p.y);
    if (hit?.ni < 0 && Number.isFinite(hit.pitch))
      onPlace(hit, hit.pitch, true);
  });
  return {
    render(next, edited, part = "all") {
      state = next;
      focusedPart = part;
      const changed =
        edited &&
        JSON.stringify(edited.measure) !==
          JSON.stringify(next.measures[edited.index]?.parsed)
          ? edited
          : null;
      const key = `${state.id}:${state.revision}:${part}:${JSON.stringify(changed)}`;
      if (key === signature) return;
      signature = key;
      try {
        rememberViewport();
        rendered = engrave(state, api.settings, changed, range.mode);
        if (host.clientWidth > 0)
          api.renderScore(rendered.score, visibleTracks());
      } catch (error) {
        signature = null;
        onError(error.message);
      }
    },
    restoreViewport(position) { viewportPosition = position; },
    select(hit, scroll = false) {
      selected = hit;
      scrollToSelection = scroll;
      if (scroll && viewportPosition) return;
      markSelection();
    },
    zoom(scale) {
      rememberViewport();
      api.settings.display.scale = scale;
      api.updateSettings();
      api.render();
    },
    destroy() {
      resize.disconnect();
      window.removeEventListener("pointerup", releasePointer);
      api.destroy();
    },
    hitAt,
  };
}

// A change between TAB and notation starts a new staff system. Each section
// engraves a range of the complete score so inherited signatures and ties keep
// their musical context; measures within that range share one layout.
export function scoreView(host, callbacks) {
  let sections = [],
    rangeKey,
    scale = 1.1;
  return {
    render(state, edited, part = "all") {
      const ranges = [];
      const grouped =
        new Set(
          state.measures.map((m) => `${m.part_id || ""}/${m.staff_id || ""}`),
        ).size > 1;
      if (grouped)
        ranges.push({
          mode: "mixed",
          start: 0,
          end: Math.max(...state.measures.map((m, i) => m.bar_index ?? i)) + 1,
        });
      else
        state.measures.forEach((measure, i) => {
          const mode = measureProfile(measure, state).mode,
            last = ranges.at(-1);
          if (last?.mode === mode) last.end = i + 1;
          else ranges.push({ mode, start: i, end: i + 1 });
        });
      const key = JSON.stringify(ranges);
      if (rangeKey !== key) {
        sections.forEach((s) => s.view.destroy());
        host.replaceChildren();
        rangeKey = key;
        sections = ranges.map((range) => {
          const node = el("div", undefined, "engraved-section");
          host.append(node);
          const view = engravedSection(node, callbacks, range);
          if (scale !== 1.1) view.zoom(scale);
          return { node, view, range };
        });
      }
      sections.forEach((s) => s.view.render(state, edited, part));
    },
    restoreViewport(position) {
      sections.forEach((s) => s.view.restoreViewport(position));
    },
    select(hit, scroll = false) {
      sections.forEach((s) =>
        s.view.select(
          hit &&
            (s.range.mode === "mixed" ||
              (hit.mi >= s.range.start && hit.mi < s.range.end))
            ? hit
            : null,
          scroll,
        ),
      );
    },
    zoom(value) {
      scale = value;
      sections.forEach((s) => s.view.zoom(value));
    },
    hitAt(x, y) {
      const origin = host.getBoundingClientRect();
      for (const s of sections) {
        const b = s.node.getBoundingClientRect(),
          offset = b.top - origin.top;
        if (y >= offset && y <= offset + b.height)
          return s.view.hitAt(x - b.left + origin.left, y - offset);
      }
      return null;
    },
  };
}
