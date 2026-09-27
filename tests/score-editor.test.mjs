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
