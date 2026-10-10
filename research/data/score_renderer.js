// Use the application's established engraving adapter, with no editor overlays.
import { alphaTab, engrave, scoreDisplaySettings } from "../../ui/workbench/score-engraving.js";

// Observe native effect painting for research supervision. The recorder calls
// every original method unchanged and is removed after rendering. Its linked
// beats establish a visible group; source note ownership is never inferred.
function observeTechniques(factories, sourceOf, records) {
  const originals = new Map();
  for (const factory of factories) for (const band of factory.effectBands || []) {
    const effect = band.effect;
    const kind = effect.notationElement === alphaTab.NotationElement.EffectPalmMute ? "pm"
      : effect.notationElement === alphaTab.NotationElement.EffectLetRing ? "let" : null;
    if (!kind || originals.has(effect)) continue;
    const create = effect.createNewGlyph;
    originals.set(effect, create);
    effect.createNewGlyph = function (...args) {
      const glyph = create.apply(this, args), paint = glyph.paint;
      glyph.paint = function (x, y, canvas) {
        const text = canvas.fillText, rect = canvas.fillRect, commands = [];
        canvas.fillText = function (value, tx, ty, ...tail) {
          const size = this.measureText(value);
          commands.push({ type: "text", value,
            bbox: [tx, ty - size.height / 2, size.width, size.height] });
          return text.call(this, value, tx, ty, ...tail);
        };
        canvas.fillRect = function (rx, ry, rw, rh, ...tail) {
          commands.push({ type: "rect", bbox: [rx, ry, rw, rh] });
          return rect.call(this, rx, ry, rw, rh, ...tail);
        };
        try { return paint.call(this, x, y, canvas); }
        finally {
          canvas.fillText = text; canvas.fillRect = rect;
          if (commands.length) {
            const events = [], seen = new Set();
            let current = this;
            do {
              if (seen.has(current)) throw new Error("Cyclic native effect chain");
              seen.add(current);
              const source = sourceOf.get(current.beat);
              if (source) events.push({ ...source });
              current = current.isLinkedWithNext ? current.nextGlyph : null;
            } while (current);
            const left = Math.min(...commands.map(c => c.bbox[0]));
            const top = Math.min(...commands.map(c => c.bbox[1]));
            const right = Math.max(...commands.map(c => c.bbox[0] + c.bbox[2]));
            const bottom = Math.max(...commands.map(c => c.bbox[1] + c.bbox[3]));
            records.push({ kind, events, commands, scale: canvas.scale,
              bbox: [left, top, right - left, bottom - top] });
          }
        }
      };
      return glyph;
    };
  }
  return () => { for (const [effect, create] of originals) effect.createNewGlyph = create; };
}

export async function render(state) {
  // alphaTab 1.8.4 supplies a PM band only on its TAB factory. Reuse that native
  // band for notation-only studies; otherwise the image would contradict M.
  const factories = alphaTab.Environment.defaultRenderers;
  const standard = factories.find(f => f.staffId === "score");
  const palmMute = factories.find(f => f.staffId === "tab")?.effectBands.find(
    b => b.effect.notationElement === alphaTab.NotationElement.EffectPalmMute);
  if (palmMute && !standard.effectBands.some(b => b.effect.notationElement === alphaTab.NotationElement.EffectPalmMute)) {
    const effect = Object.create(palmMute.effect);
    Object.defineProperty(effect, "canShareBand", { value: false });
    standard.effectBands.push({ ...palmMute, effect, shouldCreate: staff => !staff.showTablature });
  }
  document.body.style.cssText = "margin:0;background:white";
  const host = document.createElement("div");
  host.id = "score";
  const options = state.render_options || {};
  host.style.width = `${options.width || 1500}px`;
  document.body.replaceChildren(host);
  const api = new alphaTab.AlphaTabApi(host, {
    core: { useWorkers: false, engine: "svg", includeNoteBounds: true,
      // Research screenshots and geometry must describe the same complete
      // page, including systems below the browser viewport.
      enableLazyLoading: false,
      smuflFontSources: { [alphaTab.FontFileFormat.Woff2]: new URL(
        "../../ui/workbench/vendor/font/Bravura.woff2", import.meta.url).href } },
    display: { ...scoreDisplaySettings, scale: options.scale || 1.5,
      barsPerRow: options.bars_per_row || 2, padding: [45, 30, 35, 30] },
    notation: { elements: { scoreTitle: !options.hide_headers, scoreArtist: false,
      guitarTuning: !options.hide_headers && !options.hide_tuning,
      trackNames: !options.hide_headers, effectDynamics: false,
      [alphaTab.NotationElement.BarNumber]: !options.hide_bar_numbers } },
    player: { enablePlayer: false },
  });
  // Training studies may vary ordinary engraving typography independently
  // of the notes. The native renderer still owns label placement and bounds.
  if (options.label_font_size) {
    api.settings.display.resources.elementFonts.set(alphaTab.NotationElement.TrackNames,
      new alphaTab.model.Font(options.label_font_family || "Arial", options.label_font_size));
  }
  // The editor adapter consumes sounding M, while training labels are written
  // pitches. Convert only this private rendering copy, retaining the label data.
  const editorState = structuredClone(state);
  for (const row of editorState.measures) {
    if (row.score_state.key === -1) row.score_state.key = "DMinor";
    for (const voice of row.parsed.voices) for (const event of voice.events) {
      for (const note of event.notes) if (Number.isFinite(note.pitch))
        note.pitch += row.pitch_context.instrument_transpose || 0;
    }
  }
  const rendered = engrave(editorState, api.settings);
  rendered.score.title = state.title;
  const techniqueSpans = [];
  const restore = options.capture_techniques
    ? observeTechniques(factories, rendered.sourceOf, techniqueSpans) : () => {};
  try {
    await new Promise((resolve, reject) => {
      api.error.on(reject);
      api.postRenderFinished.on(resolve);
      api.renderScore(rendered.score, rendered.score.tracks.map((_, i) => i));
    });
  } finally { restore(); }
  await document.fonts.ready;
  // Canvas commands are in unscaled coordinates local to a partial SVG. Bind
  // them to that exact DOM text node, then use its transform into screenshot
  // pixels. This avoids guessing a title offset or treating SVG units as px.
  const textNodes = [...host.querySelectorAll("svg text")], used = new Set();
  const hostBounds = host.getBoundingClientRect();
  for (const span of techniqueSpans) {
    const text = span.commands.find(command => command.type === "text");
    if (!text) throw new Error("Effect group has no visible text anchor");
    const scale = span.scale ?? api.settings.display.scale;
    const tx = text.bbox[0] * scale, ty = (text.bbox[1] + text.bbox[3] / 2) * scale;
    const node = textNodes.find(node => !used.has(node) && node.textContent === text.value
      && Math.abs(Number(node.getAttribute("x")) - tx) < 0.001
      && Math.abs(Number(node.getAttribute("y")) - ty) < 0.001);
    if (!node) throw new Error("Could not bind effect commands to their rendered SVG");
    used.add(node);
    const matrix = node.getScreenCTM();
    span.commands = span.commands.map(command => {
      const [x, y, w, h] = command.bbox;
      const a = new DOMPoint(x * scale, y * scale).matrixTransform(matrix);
      const b = new DOMPoint((x + w) * scale, (y + h) * scale).matrixTransform(matrix);
      return { ...command, bbox: [a.x - hostBounds.x, a.y - hostBounds.y, b.x - a.x, b.y - a.y] };
    });
    const left = Math.min(...span.commands.map(c => c.bbox[0]));
    const top = Math.min(...span.commands.map(c => c.bbox[1]));
    const right = Math.max(...span.commands.map(c => c.bbox[0] + c.bbox[2]));
    const bottom = Math.max(...span.commands.map(c => c.bbox[1] + c.bbox[3]));
    span.bbox = [left, top, right - left, bottom - top];
    span.coordinate_space = "host_screenshot_pixels";
    delete span.scale;
  }
  const boxes = state.measures.map((row, i) => {
    const master = api.boundsLookup.findMasterBarByIndex(row.bar_index);
    const bars = master.bars.filter(b => rendered.sourceOf.get(b.bar)?.mi === i).map(b => b.realBounds);
    const x = Math.min(...bars.map(b => b.x));
    const top = Math.min(...bars.map(b => b.y));
    const bottom = Math.max(...bars.map(b => b.y + b.h));
    const other = master.bars.filter(b => rendered.sourceOf.get(b.bar)?.mi !== i).map(b => b.realBounds);
    const above = other.filter(b => b.y + b.h <= top);
    const below = other.filter(b => b.y >= bottom);
    // Staff bounds omit ledger lines, chord names and technique labels. Use the
    // full system extent, dividing inter-part whitespace for ensemble crops.
    const y = above.length ? (top + Math.max(...above.map(b => b.y + b.h))) / 2 : master.realBounds.y;
    const end = below.length ? (bottom + Math.min(...below.map(b => b.y))) / 2 : master.realBounds.y + master.realBounds.h;
    return [x, y, Math.max(...bars.map(b => b.x + b.w)) - x, end - y];
  });
  // Keep the renderer alive through the caller's screenshot, release on navigation.
  window.scoreRenderApi = api;
  // Training-only alignment truth comes from the engraver's actual note bounds.
  // It is never inferred by spreading onset ticks uniformly across a measure.
  const eventAnchors = [];
  for (const [key, beat] of rendered.beats) {
    const [mi, vi, ei] = key.split(":").map(Number);
    const row = state.measures[mi], event = row.parsed.voices[vi].events[ei];
    if (!event.notes.length) continue;
    const views = (api.boundsLookup.findBeats(beat) || []).map(bounds => {
      const heads = (bounds.notes || []).map(note => note.noteHeadBounds);
      if (!heads.length) return null;
      const x = Math.min(...heads.map(b => b.x));
      const y = Math.min(...heads.map(b => b.y));
      const right = Math.max(...heads.map(b => b.x + b.w));
      const bottom = Math.max(...heads.map(b => b.y + b.h));
      return { bbox: [x, y, right - x, bottom - y],
        note_heads: heads.map(b => [b.x, b.y, b.w, b.h]) };
    }).filter(Boolean).sort((a, b) => a.bbox[1] - b.bbox[1]);
    eventAnchors.push({ measure: mi, voice: row.parsed.voices[vi].voice,
      onset: event.start, event_index: ei, views });
  }
  return { boxes, eventAnchors, techniqueSpans, width: host.clientWidth, height: host.scrollHeight };
}
