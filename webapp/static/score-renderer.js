import "./vendor/vexflow.js";
import { pitchName, pitchShift } from "./score-model.js";
const VF = globalThis.Vex.Flow;
const NS = "http://www.w3.org/2000/svg";
function node(tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) n.setAttribute(key, value);
  if (text !== undefined) n.textContent = text;
  return n;
}
const drumKeys = { 35:"f/4",36:"f/4",37:"c/5/x2",38:"c/5",42:"g/5/x2",44:"d/4/x2",46:"g/5/x2",
  43:"g/4",45:"a/4",47:"b/4",48:"d/5",50:"e/5",49:"a/5/x2",51:"f/5/x2",52:"b/5/x2",53:"f/5/d2",55:"b/5/x2",57:"a/5/x2",59:"f/5/x2" };
const effects = { vib:"vib.", pm:"P.M.", hammer:"H/P", let:"let ring", tap:"T", sl:"slide", ss:"slide",
  sia:"slide in", sib:"slide in", sou:"slide out", sod:"slide out", fade:"fade", pick_up:"↑", pick_down:"↓",
  stroke_up:"↑", stroke_down:"↓", rasg:"rasg.", ghost:"( )", dead:"X" };
function effectText(event) {
  return [...new Set([...(event.effects || []), ...event.notes.flatMap(n => n.effects || [])].flatMap(e => {
    if (effects[e]) return [effects[e]];
    const [kind, ...args] = e.split(":");
    if (kind === "ottava") return [{12:"8va",'-12':"8vb",24:"15ma",'-24':"15mb"}[args[0]]];
    if (kind === "text" || kind === "chord") { try { return [decodeURIComponent(args.join(":"))]; } catch { return [args.join(":")]; } }
    if (kind === "bend") return ["bend " + args.at(-1)];
    if (kind === "harm") return ["harm."];
    if (kind === "trill") return ["tr"];
    if (kind === "grace") return ["grace"];
    if (kind === "trem") return ["trem."];
    return [];
  }))].join(" ");
}
function notationKey(pitch, flat) {
  const m = pitchName(pitch, flat).match(/^([A-G])([#b]?)(-?\d+)$/);
  return { key: `${m[1].toLowerCase()}${m[2]}/${m[3]}`, accidental: m[2] || "n" };
}
export function renderScore(host, measure, profile, selection, onSelect, options = {}) {
  host.replaceChildren();
  const { mode, instrument, tuning, pitch_context: context = {} } = profile;
  const tab = mode !== "notation", notation = mode !== "tab", drums = instrument === "drums";
  const rows = measure.voices.flatMap((v, vi) => v.events.map((event, ei) => ({event, vi, ei, voice:v.voice})));
  const starts = [...new Set(rows.map(r => r.event.start))].sort((a,b)=>a-b);
  const width = options.width || Math.max(560, host.closest(".score-scroll")?.clientWidth || 0, 140 + starts.length * 100);
  const tabY = notation ? 205 : 42, inset = options.overview ? 0 : 12;
  const height = tab ? tabY + 130 + Math.max(0, tuning.length - 5) * 13 : 225;
  const renderer = new VF.Renderer(host, VF.Renderer.Backends.SVG);
  renderer.resize(width, height);
  const ctx = renderer.getContext();
  const svg = host.querySelector("svg");
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", options.overview ? "小节谱面" : "识别结果，可点选音符编辑");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const clef = drums ? "percussion" : ({G2:"treble",F4:"bass",C3:"alto",C4:"tenor"}[context.clef] || (instrument === "bass" ? "bass" : "treble"));
  let stave, tabStave;
  if (notation) {
    stave = new VF.Stave(inset, 48, width - 2 * inset);
    if (options.showClef !== false) stave.addClef(clef);
    // Explicit accidentals show each written pitch without guessing inherited spelling.
    if (measure.time_signature && options.showTime !== false) stave.addTimeSignature(measure.time_signature);
    stave.setContext(ctx).draw();
  }
  if (tab) {
    tabStave = new VF.TabStave(inset, tabY, width - 2 * inset, { num_lines: tuning.length || 6 });
    if (options.showClef !== false) tabStave.addClef("tab");
    if (!notation && measure.time_signature && options.showTime !== false) tabStave.addTimeSignature(measure.time_signature);
    tabStave.setContext(ctx).draw();
  }
  const left = Math.max(stave?.getNoteStartX() || 0, tabStave?.getNoteStartX() || 0) + 24;
  const end = width - (options.overview ? 30 : 55), stride = (end-left)/Math.max(1,starts.length-1);
  const positions = new Map(starts.map((s,i)=>[s, left+i*stride]));
  const hits = [], previous = new Map();
  for (const row of rows) {
    const {event, vi, ei, voice} = row, x = positions.get(event.start) + (voice ? 13 : 0);
    const rest = event.status !== "normal" || !event.notes.length;
    const dur = String(event.duration.value), stem = voice ? -1 : 1;
    let drawn;
    if (notation) {
      const written = event.notes.map(n => n.pitch - pitchShift(context,event.effects,instrument));
      const keys = rest ? [{key:clef === "bass" ? "d/3" : "b/4"}] : event.notes.map((n,i) => drums
        ? {key:drumKeys[n.pitch] || "c/5"} : notationKey(written[i], !!n.swap_accidentals));
      const note = new VF.StaveNote({ keys: keys.map(k=>k.key), duration:dur+(rest?"r":""), clef, stem_direction:stem });
      if (!rest && !drums) keys.forEach((k,i)=>note.addModifier(new VF.Accidental(k.accidental),i));
      const dots = event.duration.double_dotted ? 2 : event.duration.dotted ? 1 : 0;
      for(let i=0;i<dots;i++) VF.Dot.buildAndAttach([note], { all:true });
      if (!rest) event.notes.forEach((n,i)=>{
        for (const [effect, code] of [["stacc","a."],["accent","a>"],["heavy","a^"]])
          if (n.effects?.includes(effect)) note.addModifier(new VF.Articulation(code).setPosition(3),i);
      });
      note.setStave(stave).setContext(ctx);
      new VF.TickContext().addTickable(note).preFormat().setX(x-stave.getNoteStartX());
      note.draw(); drawn=note;
      note.getYs().forEach((y, ni)=>hits.push({ vi,ei,ni:rest?-1:ni,x:note.getAbsoluteX()+5,y, kind:"notation" }));
      if (!rest) event.notes.forEach((n,i)=>{
        if (n.effects?.includes("tie")) {
          const prior = previous.get(`${vi}:${n.pitch}`);
          new VF.StaveTie({first_note:prior?.note,last_note:note,first_indices:[prior?.i||0],last_indices:[i]}).setContext(ctx).draw();
        }
        previous.set(`${vi}:${n.pitch}`,{note,i});
      });
    }
    if (tab) {
      if (!rest) {
        const note = new VF.TabNote({positions:event.notes.map(n=>({str:n.string,fret:n.fret})),duration:dur,stem_direction:stem}, !notation);
        note.setStave(tabStave).setContext(ctx);
        new VF.TickContext().addTickable(note).preFormat().setX(x-tabStave.getNoteStartX());
        const dots = event.duration.double_dotted ? 2 : event.duration.dotted ? 1 : 0;
        for(let i=0;i<dots;i++) VF.Dot.buildAndAttach([note],{all:true});
        note.draw();
      } else {
        const note = new VF.StaveNote({keys:["b/4"],duration:dur+"r"}).setStave(tabStave).setContext(ctx);
        new VF.TickContext().addTickable(note).preFormat().setX(x-tabStave.getNoteStartX());
        note.draw();
      }
      for (let string=1;string<=(tuning.length||6);string++)
        hits.push({vi,ei,ni:event.notes.findIndex(n=>n.string===string),string,x:x+4,y:tabStave.getYForLine(string-1),kind:"tab"});
    }
    const label = effectText(event);
    if (label) svg.append(node("text",{x,y:notation?32:24,class:"score-effect","text-anchor":"middle"},label));
    const d=event.duration;
    if ((d.tuplet_enters||1)!==(d.tuplet_times||1)) svg.append(node("text",{x,y:notation?185:tabY+100,class:"score-tuplet","text-anchor":"middle"},`${d.tuplet_enters}:${d.tuplet_times}`));
    if (event.status==="empty") svg.append(node("text",{x,y:height-10,class:"score-effect"},"空拍"));
    if (drawn) drawn.setAttribute("data-event",`${vi}:${ei}`);
  }
  if (options.overview) return { hits, width, height };
  if (notation && !drums && options.onPlace) {
    // Clicking the staff places a written pitch at the nearest existing beat.
    // Keep this behind note hit targets so selecting a note never changes it.
    const surface = node("rect", {x:left-18, y:50, width:Math.max(1,width-left-2), height:125,
      fill:"transparent", stroke:"none", class:"staff-placement"});
    surface.addEventListener("click", e => {
      const point = svg.createSVGPoint(); point.x=e.clientX; point.y=e.clientY;
      const local = point.matrixTransform(svg.getScreenCTM().inverse());
      const start = starts.reduce((best,s) => Math.abs(positions.get(s)-local.x)<Math.abs(positions.get(best)-local.x)?s:best, starts[0]);
      const voice = measure.voices[selection?.vi||0], ei=voice.events.findIndex(e=>e.start===start);
      if(ei<0)return;
      const bottom = {treble:30,bass:18,alto:24,tenor:22}[clef] ?? 30;
      const diatonic = bottom + Math.round((stave.getYForLine(4)-local.y)/5);
      const pitch = (Math.floor(diatonic/7)+1)*12 + [0,2,4,5,7,9,11][((diatonic%7)+7)%7];
      options.onPlace({vi:selection?.vi||0, ei, pitch});
    });
    svg.append(surface);
  }
  for (const hit of hits) {
    const active = selection && hit.vi===selection.vi && hit.ei===selection.ei
      && (hit.kind==="tab" ? hit.string===selection.string : hit.ni===selection.ni);
    const box = node("rect",{x:hit.x-13,y:hit.y-11,width:30,height:22,rx:4,
      class:`score-hit${active?" selected":""}${hit.ni<0?" empty-note":""}`,
      "data-voice":hit.vi,"data-event":hit.ei,"data-note":hit.ni,
      "data-kind":hit.kind,...(hit.string?{"data-string":hit.string}:{}),tabindex:0,role:"button",
      "aria-label":`声部 ${measure.voices[hit.vi].voice+1}，第 ${hit.ei+1} 个音符${hit.string?`，第 ${hit.string} 弦`:""}`});
    box.addEventListener("click",()=>onSelect(hit));
    box.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();onSelect(hit);}});
    svg.append(box);
  }
  return { hits, width, height };
}
