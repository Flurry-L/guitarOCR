// Use the application's established engraving adapter, with no editor overlays.
import { alphaTab, engrave, scoreDisplaySettings } from "../../ui/workbench/score-engraving.js";

// Record the glyphs that actually paint. In particular, an allocated TAB note
// can have no number at all when its tie continues: source fret != printed fret.
function observeTab(factories, sourceOf, records) {
  const restores = [];
  let serial = 0;
  const wrapPaint = (glyph, value) => {
    const paint = glyph.paint;
    glyph.paint = function (x, y, canvas) {
      const id = `observation-${serial++}`, texts = [], paths = [], symbols = [], strokes = [];
      let path = [];
      const text = canvas.fillText, move = canvas.moveTo, curve = canvas.bezierCurveTo;
      const begin = canvas.beginPath, line = canvas.lineTo, stroke = canvas.stroke;
      const symbol = canvas.fillMusicFontSymbol, multiple = canvas.fillMusicFontSymbols;
      canvas.fillMusicFontSymbol = function (x, y, size, value, ...tail) {
        symbols.push(value); return symbol.call(this, x, y, size, value, ...tail);
      };
      canvas.fillMusicFontSymbols = function (x, y, size, values, ...tail) {
        symbols.push(...values); return multiple.call(this, x, y, size, values, ...tail);
      };
      canvas.fillText = function (value, ...args) {
        if (String(value).length) texts.push(String(value));
        return text.call(this, value, ...args);
      };
      canvas.beginPath = function (...args) { path = []; return begin.apply(this, args); };
      canvas.moveTo = function (...point) {
        paths.push(point); path.push(["move", ...point]); return move.apply(this, point);
      };
      canvas.lineTo = function (...point) {
        paths.push(point); path.push(["line", ...point]); return line.apply(this, point);
      };
      canvas.bezierCurveTo = function (...points) {
        paths.push(points); path.push(["curve", ...points]); return curve.apply(this, points);
      };
      canvas.stroke = function (...args) { strokes.push([...path]); return stroke.apply(this, args); };
      canvas.beginGroup(id);
      try { return paint.call(this, x, y, canvas); }
      finally {
        canvas.endGroup();
        canvas.fillText = text; canvas.moveTo = move; canvas.bezierCurveTo = curve;
        canvas.beginPath = begin; canvas.lineTo = line; canvas.stroke = stroke;
        canvas.fillMusicFontSymbol = symbol; canvas.fillMusicFontSymbols = multiple;
        records.push({ ...(typeof value === "function" ? value() : value),
          id, texts, paths, symbols, strokes, scale: canvas.scale });
      }
    };
  };
  for (const factory of factories.filter(f => f.staffId === "tab")) {
    const create = factory.create;
    restores.push(() => { factory.create = create; });
    factory.create = function (...args) {
      const renderer = create.apply(this, args), addBeat = renderer.addBeatGlyph;
      const addPre = renderer.addPreBeatGlyph;
      renderer.addPreBeatGlyph = function (glyph) {
        wrapPaint(glyph, { kind: "signature", ...sourceOf.get(this.bar) });
        return addPre.call(this, glyph);
      };
      const paintContent = renderer.paintContent;
      renderer.paintContent = function (x, y, canvas) {
        const id = `observation-${serial++}`;
        canvas.beginGroup(id);
        try { return paintContent.call(this, x, y, canvas); }
        finally {
          canvas.endGroup();
          records.push({ kind: "staff", id, ...sourceOf.get(this.bar),
            lines: Array.from({ length: this.drawnLineCount }, (_, i) => y + this.y + this.getLineY(i)),
            scale: canvas.scale });
        }
      };
      renderer.addBeatGlyph = function (glyph) {
        const addTie = glyph.addTie, createTies = glyph.createTies;
        let owner;
        glyph.createTies = function (note) {
          owner = note;
          try { return createTies.call(this, note); }
          finally { owner = null; }
        };
        glyph.addTie = function (tie) {
          if (/^tab\.tie\.\d+$/.test(tie.slurEffectId || ""))
            wrapPaint(tie, () => ({ kind: "tie", start: sourceOf.get(tie.startNote), end: sourceOf.get(tie.endNote) }));
          else if (/^tab\.slur\.effect\./.test(tie.slurEffectId || ""))
            wrapPaint(tie, () => ({ kind: "technique", start: sourceOf.get(tie.startNote), end: sourceOf.get(tie.endNote) }));
          else if (owner && !tie.slurEffectId) {
            // Native slides have no public type/owner fields. Capture their
            // creation owner and actual strokes; classify the visible line
            // relative to its note bounds after rendering, without source flags.
            wrapPaint(tie, { kind: "local-path", start: sourceOf.get(owner),
              end: sourceOf.get(owner.slideTarget || owner), note: owner });
          }
          return addTie.call(this, tie);
        };
        const paintNotes = glyph.onNotes.paint;
        glyph.onNotes.paint = function (...args) {
          if (this.restGlyph && !this.restGlyph.observationWrapped) {
            this.restGlyph.observationWrapped = true;
            wrapPaint(this.restGlyph, { kind: "rest", ...sourceOf.get(glyph.beat) });
          }
          for (const [string, number] of this.noteNumbers?.notesPerString || []) {
            if (number.observationWrapped) continue;
            number.observationWrapped = true;
            const note = glyph.beat.notes.find(note => note.string === string);
            wrapPaint(number, { kind: "fret", ...sourceOf.get(note) });
          }
          return paintNotes.apply(this, args);
        };
        return addBeat.call(this, glyph);
      };
      return renderer;
    };
  }
  const observedEffects = new Set();
  for (const factory of factories) for (const band of factory.effectBands || []) {
    const effect = band.effect;
    const kind = effect.notationElement === alphaTab.NotationElement.EffectDynamics ? "dynamic"
      : effect.notationElement === alphaTab.NotationElement.EffectChordNames ? "chord" : null;
    if (!kind || observedEffects.has(effect)) continue;
    observedEffects.add(effect);
    const create = effect.createNewGlyph;
    restores.push(() => { effect.createNewGlyph = create; });
    effect.createNewGlyph = function (...args) {
      const glyph = create.apply(this, args);
      const owner = args.map(value => sourceOf.get(value)).find(value => value?.ei !== undefined);
      if (!owner) throw new Error(`Printed ${kind} has no source event identity`);
      const beat = args.find(value => sourceOf.get(value)?.ei !== undefined);
      wrapPaint(glyph, { kind, ...owner,
        ...(kind === "chord" ? { sourceName: beat.chord.name } : {}) });
      return glyph;
    };
  }
  return () => restores.forEach(restore => restore());
}

// Observe native effect painting for research supervision. The recorder calls
// every original method unchanged and is removed after rendering. Its linked
// beats establish a visible group; source note ownership is never inferred.
function observeTechniques(factories, sourceOf, records, options = {}) {
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
      const layout = glyph.doLayout;
      const label = value => kind === "let" && value === "LetRing" ? options.let_label || value : value;
      glyph.doLayout = function (...args) {
        const canvas = this.renderer.scoreRenderer.canvas, measure = canvas.measureText;
        canvas.measureText = function (value) { return measure.call(this, label(value)); };
        try { return layout.apply(this, args); }
        finally { canvas.measureText = measure; }
      };
      glyph.paint = function (x, y, canvas) {
        const text = canvas.fillText, rect = canvas.fillRect, commands = [];
        const metrics = this.renderer.smuflMetrics;
        const dashSize = metrics.lineRangedGlyphDashSize, dashGap = metrics.lineRangedGlyphDashGap;
        metrics.lineRangedGlyphDashSize = dashSize * (options.effect_line_scale || 1);
        metrics.lineRangedGlyphDashGap = dashGap * (options.effect_line_scale || 1);
        canvas.fillText = function (value, tx, ty, ...tail) {
          value = label(value);
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
          metrics.lineRangedGlyphDashSize = dashSize;
          metrics.lineRangedGlyphDashGap = dashGap;
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
  const factories = alphaTab.Environment.defaultRenderers;
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
      effectBandPaddingBottom: options.effect_padding ?? scoreDisplaySettings.effectBandPaddingBottom,
      justifyLastSystem: options.justify_last_system ?? false,
      barsPerRow: options.bars_per_row || 2, padding: [45, 30, 35, 30] },
    notation: { notationMode: options.notation_mode ?? alphaTab.NotationMode.GuitarPro,
      extendLineEffectsToBeatEnd: options.extend_line_effects || false,
      elements: { scoreTitle: !options.hide_headers, scoreArtist: false,
      guitarTuning: !options.hide_headers && !options.hide_tuning,
      trackNames: !options.hide_headers, effectDynamics: true,
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
    for (const voice of row.parsed.voices) for (const event of voice.events) {
      for (const note of event.notes) if (Number.isFinite(note.pitch))
        note.pitch += row.pitch_context.instrument_transpose;
    }
  }
  const rendered = engrave(editorState, api.settings);
  rendered.score.title = state.title;
  if (options.chord_diagrams_in_score !== undefined)
    rendered.score.stylesheet.globalDisplayChordDiagramsInScore = options.chord_diagrams_in_score;
  if (options.chord_diagrams_on_top !== undefined || options.chord_diagrams_in_score)
    rendered.score.stylesheet.globalDisplayChordDiagramsOnTop = options.chord_diagrams_on_top ?? false;
  if (!options.hide_headers)
    rendered.score.stylesheet.firstSystemTrackNameMode = alphaTab.model.TrackNameMode.FullName;
  const techniqueSpans = [];
  const painted = [];
  const restore = options.capture_techniques
    ? observeTechniques(factories, rendered.sourceOf, techniqueSpans, options) : () => {};
  const restoreTab = options.capture_observations ? observeTab(factories, rendered.sourceOf, painted) : () => {};
  try {
    await new Promise((resolve, reject) => {
      api.error.on(reject);
      api.postRenderFinished.on(resolve);
      api.renderScore(rendered.score, rendered.score.tracks.map((_, i) => i));
    });
  } finally { restore(); restoreTab(); }
  await document.fonts.ready;
  // Canvas commands are in unscaled coordinates local to a partial SVG. Bind
  // them to that exact DOM text node, then use its transform into screenshot
  // pixels. This avoids guessing a title offset or treating SVG units as px.
  const textNodes = [...host.querySelectorAll("svg text")], used = new Set();
  for (const node of textNodes) if (node.textContent === "rendered by alphaTab") node.remove();
  const hostBounds = host.getBoundingClientRect();
  const glyphMetrics = document.createElement("canvas").getContext("2d");
  const paintedRect = node => {
    if (node.tagName !== "text" || !/[\uE000-\uF8FF]/u.test(node.textContent))
      return node.getBoundingClientRect();
    // SVG text boxes use the music font's full line height, which is much
    // taller than an x, open-string circle or fingering dot. Measure the
    // actual painted glyph with the same loaded font and SVG transform.
    const style = getComputedStyle(node);
    glyphMetrics.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    const text = glyphMetrics.measureText(node.textContent);
    const anchor = node.getAttribute("text-anchor") || style.textAnchor;
    const x = Number(node.getAttribute("x") || 0)
      - (anchor === "middle" ? text.width / 2 : anchor === "end" ? text.width : 0);
    const y = Number(node.getAttribute("y") || 0), matrix = node.getScreenCTM();
    const a = new DOMPoint(x - text.actualBoundingBoxLeft, y - text.actualBoundingBoxAscent).matrixTransform(matrix);
    const b = new DOMPoint(x + text.actualBoundingBoxRight, y + text.actualBoundingBoxDescent).matrixTransform(matrix);
    return { x: a.x, y: a.y, right: b.x, bottom: b.y, width: b.x - a.x, height: b.y - a.y };
  };
  const nodeBounds = nodes => {
    const rects = nodes.map(paintedRect).filter(rect => rect.width > 0 && rect.height > 0);
    if (!rects.length) return null;
    const x = Math.min(...rects.map(rect => rect.x));
    const y = Math.min(...rects.map(rect => rect.y));
    return [x - hostBounds.x, y - hostBounds.y,
      Math.max(...rects.map(rect => rect.right)) - x,
      Math.max(...rects.map(rect => rect.bottom)) - y];
  };
  for (const item of painted) {
    const node = host.querySelector(`g.${item.id}`);
    if (!node) throw new Error(`Missing painted observation group ${item.id}`);
    const bbox = node.getBoundingClientRect();
    item.bbox = [bbox.x - hostBounds.x, bbox.y - hostBounds.y, bbox.width, bbox.height];
    if (item.kind === "chord") {
      // Bind the source identity to actual painted glyphs. Names and grid
      // digits have different jobs even when the engraver puts them together.
      const nameNodes = [...node.querySelectorAll("text")]
        .filter(text => text.textContent === item.sourceName && item.sourceName);
      item.name_bbox = nodeBounds(nameNodes);
      item.name = item.name_bbox ? nameNodes.map(text => text.textContent).join("") : null;
      item.diagram_bbox = nodeBounds([...node.querySelectorAll("path,rect,circle,ellipse,line,polyline,polygon,text")]
        .filter(child => !nameNodes.includes(child)));
      item.bbox = nodeBounds([...node.querySelectorAll("path,rect,circle,ellipse,line,polyline,polygon,text")]);
      for (const text of node.querySelectorAll("text")) used.add(text);
      delete item.sourceName;
    }
    if (item.kind === "signature")
      item.clef_bbox = nodeBounds([...node.querySelectorAll("text")]
        .filter(text => /[\uE06D\uE06E]/u.test(text.textContent)));
    const matrix = node.getScreenCTM(), scale = item.scale ?? api.settings.display.scale;
    if (item.lines) item.lines = item.lines.map(y => new DOMPoint(0, y * scale).matrixTransform(matrix).y - hostBounds.y);
    const transformPoints = points => {
      const out = [];
      for (let i = 0; i < points.length; i += 2) {
        const point = new DOMPoint(points[i] * scale, points[i + 1] * scale).matrixTransform(matrix);
        out.push(point.x - hostBounds.x, point.y - hostBounds.y);
      }
      return out;
    };
    if (item.paths) item.paths = item.paths.map(transformPoints);
    if (item.kind === "local-path") {
      item.strokes = item.strokes.filter(path => path.length === 2
        && path[0][0] === "move" && path[1][0] === "line")
        .map(path => transformPoints([...path[0].slice(1), ...path[1].slice(1)]));
      item.noteBounds = (api.boundsLookup.findBeats(item.note.beat) || [])
        .flatMap(beat => (beat.notes || []).filter(note => note.note === item.note)
          .map(note => note.noteHeadBounds));
      delete item.note;
    } else delete item.strokes;
    if (item.kind === "technique" && item.paths.length) {
      if (item.texts.some(text => /^[HP]$/.test(text))) item.effect = "hammer";
      else if (item.texts.includes("sl.")) item.effect = "sl";
    }
    delete item.scale;
  }
  const ownerKey = owner => [owner.mi, owner.vi, owner.ei, owner.ni].join(":");
  const legato = new Set(painted.filter(item => item.effect === "sl")
    .map(item => ownerKey(item.start)));
  const localPaths = painted.filter(item => item.kind === "local-path");
  painted.splice(0, painted.length, ...painted.filter(item => item.kind !== "local-path"));
  for (const item of localPaths) {
    if (item.texts.length || item.symbols.length) continue;
    for (const [x1, y1, x2, y2] of item.strokes) {
      if (y1 === y2) continue;
      const cy = (y1 + y2) / 2;
      const note = item.noteBounds.toSorted((a, b) => Math.abs(a.y + a.h / 2 - cy)
        - Math.abs(b.y + b.h / 2 - cy))[0];
      if (!note) throw new Error("Painted slide has no native note bounds");
      let effect;
      if (x1 < note.x && x2 <= note.x) effect = y2 < y1 ? "sib" : "sia";
      else if (x1 >= note.x + note.w) {
        effect = ownerKey(item.start) !== ownerKey(item.end)
          ? (legato.has(ownerKey(item.start)) ? "sl" : "ss") : y2 < y1 ? "sou" : "sod";
      }
      if (!effect) throw new Error("Painted slide does not lie beside its native note");
      painted.push({ kind: "technique", effect, start: item.start,
        end: effect === "ss" || effect === "sl" ? item.end : item.start,
        texts: [], symbols: [], paths: [[x1, y1], [x2, y2]],
        bbox: [Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1)] });
    }
  }
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
      if (command.type === "text") {
        const box = node.getBoundingClientRect();
        return { ...command, bbox: [box.x - hostBounds.x, box.y - hostBounds.y, box.width, box.height] };
      }
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
  const systems = [];
  const annotations = painted.filter(item => item.kind === "chord" && item.bbox[2] > 0 && item.bbox[3] > 0)
    .map(item => ({ label: "annotation_region", kind: item.diagram_bbox ? "chord_diagram" : "chord",
      mi: item.mi, vi: item.vi, ei: item.ei, bbox: item.bbox, name: item.name,
      name_bbox: item.name_bbox, diagram_bbox: item.diagram_bbox }));
  const boxes = state.measures.map((row, i) => {
    const master = api.boundsLookup.findMasterBarByIndex(row.bar_index);
    systems.push({ mi: i, system_index: master.staffSystemBounds.index, page_index: 0 });
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
  const music_boxes = boxes.map(box => [...box]);
  for (const item of annotations) {
    const box = boxes[item.mi], top = Math.min(box[1], item.bbox[1]);
    const bottom = Math.max(box[1] + box[3], item.bbox[1] + item.bbox[3]);
    // Preserve every local annotation vertically. Its grid can overhang a
    // barline; the separate annotation retains that complete object, while
    // the measure's horizontal boundary still excludes neighbouring notes.
    box[1] = top;
    box[3] = bottom - top;
  }
  const firstSystemTop = Math.min(...boxes.map(box => box[1]));
  const names = new Set(rendered.score.tracks.map(track => track.name));
  const context = options.hide_headers ? [] : textNodes.flatMap(node => {
    const box = node.getBoundingClientRect();
    if (!node.isConnected || !box.width || !box.height || !node.textContent.trim()
        || used.has(node) || /[\uE000-\uF8FF]/u.test(node.textContent)) return [];
    const bbox = [box.x - hostBounds.x, box.y - hostBounds.y, box.width, box.height];
    if (bbox[1] + bbox[3] > firstSystemTop && !names.has(node.textContent)) return [];
    return [{ mi: 0, text: node.textContent, bbox,
      label: names.has(node.textContent) ? "annotation_region"
        : node.textContent === state.title ? "title_region" : "tuning_region" }];
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
  // This renderer enables only title, instrument name and tuning in its
  // header. Keep the tuning caption and individual string labels together.
  const tuning = context.filter(item => item.label === "tuning_region");
  const tuningBounds = tuning.length ? [Math.min(...tuning.map(item => item.bbox[0])),
    Math.min(...tuning.map(item => item.bbox[1])),
    Math.max(...tuning.map(item => item.bbox[0] + item.bbox[2])),
    Math.max(...tuning.map(item => item.bbox[1] + item.bbox[3]))] : null;
  const layout_annotations = [...annotations, ...context.filter(item => item.label !== "tuning_region"),
    ...(tuningBounds ? [{ label: "tuning_region", bbox: [tuningBounds[0], tuningBounds[1],
      tuningBounds[2] - tuningBounds[0], tuningBounds[3] - tuningBounds[1]] }] : []),
    ...painted.filter(item => item.clef_bbox).map(item => ({ label: "clef_region", bbox: item.clef_bbox })),
    ...techniqueSpans.map(span => ({ label: "annotation_region", bbox: span.bbox, kind: "technique" }))];
  return { renderer: "alphatab", boxes, music_boxes, annotations, layout_annotations, systems, eventAnchors, techniqueSpans, painted, context,
    width: host.clientWidth, height: host.scrollHeight };
}
