import test from 'node:test';
import assert from 'node:assert/strict';
import {setFret,setPitch,removeNote,changeDuration,insertEvent,removeEvent,pitchShift,parsePitch,ticks} from '../webapp/static/score-model.js';
const tuning=[64,59,55,50,45,40];
test('mixed score editing keeps pitch, frets and effects consistent',()=>{
  const event={status:'normal',notes:[{string:1,fret:3,pitch:67,effects:['vib']},{string:2,fret:1,pitch:60,effects:[]}]};
  setFret(event,1,8,'both',tuning);
  assert.equal(event.notes[0].pitch,72);
  setPitch(event,0,69,'both',tuning);
  assert.equal(event.notes[0].fret,5);
  assert.deepEqual(event.notes[0].effects,['vib']);
  setPitch(event,0,55,'both',tuning);
  assert.equal(event.notes[0].string,3);
  assert.equal(event.notes[0].fret,0);
  const before=structuredClone(event);
  assert.throws(()=>setPitch(event,0,20,'both',tuning));
  assert.deepEqual(event,before);
  setPitch(event,0,60,'both',tuning); // Unisons on separate guitar strings are playable.
  assert.throws(()=>setPitch(event,-1,60,'notation',[]));
});
test('duration edits preserve tuplets, subsequent timing and note effects',()=>{
  const voice={events:[{start:0,duration:{value:4},status:'normal',notes:[{string:1,fret:3,effects:['tie','vib']}]},
    {start:960,duration:{value:2},status:'rest',notes:[]}]};
  changeDuration(voice,0,{value:8,tuplet_enters:3,tuplet_times:2});
  assert.equal(ticks(voice.events[0]),320);
  assert.equal(voice.events[1].start,320);
  assert.deepEqual(voice.events[0].notes[0].effects,['tie','vib']);
  const original=structuredClone(voice);
  const at=insertEvent(voice,0);assert.equal(voice.events[2].start,640);
  removeEvent(voice,at);assert.deepEqual(voice,original);
  removeNote(voice.events[0],0);assert.equal(voice.events[0].status,'rest');
});
test('written pitch incorporates instrument, clef, capo and local ottava',()=>{
  assert.equal(parsePitch('Bb3'),58);
  assert.equal(parsePitch('C4')+pitchShift({instrument_transpose:-2},[],'pitched'),58);
  assert.equal(pitchShift({instrument_transpose:-12,capo:2,clef_octave:-12},['ottava:12'],'guitar'),-14);
  assert.equal(pitchShift({instrument_transpose:-12},['ottava:12'],'drums'),0);
});

// Opening another project must not reuse its predecessor's revision or drafts.
import { ui, openProject, receiveProject, hasDrafts } from '../webapp/static/state.js';
test('project snapshots and editing drafts have separate lifetimes', () => {
  const saved = {id:'first',revision:4,pages:[{}],boxes:[{page:1,bbox:[0,0,50,50]}]};
  receiveProject(saved);
  ui.boxes[0].bbox[0]=20;
  ui.boxDirty=true;
  assert.equal(saved.boxes[0].bbox[0],0);
  assert.equal(hasDrafts(),true);
  openProject('second');
  assert.equal(ui.state,null);
  assert.equal(hasDrafts(),false);
  assert.deepEqual(ui.boxes,[]);
  receiveProject({id:'second',revision:9,pages:[{}],boxes:[]});
  assert.equal(ui.state.revision,9);
});

import {alphaTab,engrave} from '../webapp/static/score-engraving.js';
const eventWith = (notes,effects=[]) => ({start:0,duration:{value:4},status:'normal',notes,effects});
function engravedState(mode,instrument,events,context={},stringTuning=tuning) {
  return {mode,metadata:{instrument,tuning_used:stringTuning},measures:[{mode,pitch_context:context,
    parsed:{time_signature:'4/4',voices:[{voice:0,events}]}}]};
}
test('engraving maps TAB string order and preserves conflicting OCR pitches without rewriting source',()=>{
  const state=engravedState('both','guitar',[eventWith([{string:1,fret:3,pitch:66,effects:['bend:bend:100']}])]);
  const original=structuredClone(state);
  const result=engrave(state,new alphaTab.Settings()),beat=result.beats.get('0:0:0'),note=beat.notes[0];
  assert.equal(note.string,6);
  assert.equal(note.realValue,67);
  assert.equal(note.displayValueWithoutBend,78); // Faithful to explicit OCR F#5, despite TAB G4.
  assert.deepEqual(result.sourceOf.get(note),{mi:0,vi:0,ei:0,ni:0,string:1});
  assert.equal(note.bendPoints.at(-1).value,4); // alphaTab measures bends in quarter tones.
  assert.deepEqual(state,original);
});
test('engraving retains transposed written notes, clef octave, tuplets and cross-bar ties',()=>{
  const first=eventWith([{pitch:70,effects:[]}],['ottava:12']);
  first.duration={value:8,tuplet_enters:3,tuplet_times:2};
  const state=engravedState('notation','pitched',[first],{instrument_transpose:-2,clef:'G2',clef_octave:0});
  state.measures.push({...structuredClone(state.measures[0]),parsed:{voices:[{voice:0,events:[
    eventWith([{pitch:70,effects:['tie']}],['ottava:12'])]}]}});
  const result=engrave(state,new alphaTab.Settings()),beat=result.beats.get('0:0:0');
  assert.equal(beat.notes[0].displayValueWithoutBend,60);
  assert.equal(beat.ottava,1);
  assert.equal(beat.displayDuration,320);
  assert.equal(result.beats.get('1:0:0').notes[0].tieOrigin,beat.notes[0]);
  state.measures[0].pitch_context.clef_octave=-12;
  assert.equal(engrave(state,new alphaTab.Settings()).beats.get('0:0:0').notes[0].displayValueWithoutBend,72);
});
test('engraving handles percussion, non-six-string TAB and sparse voice numbers',()=>{
  const drums=engravedState('notation','drums',[eventWith([{pitch:38,effects:[]}])]);
  drums.measures[0].parsed.voices[0].voice=1;
  const percussion=engrave(drums,new alphaTab.Settings()).beats.get('0:0:0');
  assert.equal(percussion.voice.index,1);
  assert.equal(percussion.notes[0].percussionArticulation,38);
  assert.equal(percussion.voice.bar.staff.showTablature,false);
  const bass=engravedState('tab','bass',[eventWith([{string:5,fret:2,effects:[]}])],{},[43,38,33,28,23]);
  const note=engrave(bass,new alphaTab.Settings()).beats.get('0:0:0').notes[0];
  assert.equal(note.string,1);
  assert.equal(note.realValue,25);
});
