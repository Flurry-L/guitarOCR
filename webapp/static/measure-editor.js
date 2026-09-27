import { ui, endpoint, receiveProject } from './state.js';
import { $, el, action, notice } from './dom.js';
import { api } from './api.js';
import { renderScore } from './score-renderer.js';
import { scoreOverview, measureProfile, reviewKind } from './score-overview.js';
import { clone, parsePitch, pitchShift, setFret, setPitch, removeNote, changeDuration,
  insertEvent, removeEvent, drumNames } from './score-model.js';

export function initMeasures({ start, go, renderExport, setBusy }) {
  let draft, selected, profile, undo = [], redo = [], saving = false, digits = '', digitTime = 0;
  const overview = scoreOverview($('scoreOverview'), selectMeasure);
  const current = () => ui.state?.measures?.[ui.measureIndex];
  const event = () => draft?.voices[selected?.vi]?.events[selected?.ei];
  const dialog = $('measureDialog');
  function feedback(message = '', error = false) {
    $('measureFeedback').textContent = message;
    $('measureFeedback').hidden = !message;
    $('measureFeedback').classList.toggle('error',error);
  }
  function snapshot() { return {draft:clone(draft),selected:clone(selected)}; }
  function change(fn) {
    if (ui.busy || saving) return;
    const before = snapshot();
    try { fn(); } catch(error) {draft=before.draft;selected=before.selected;feedback(error.message,true);return;}
    if (JSON.stringify(before.draft)===JSON.stringify(draft)) return;
    undo.push(before);if(undo.length>100)undo.shift();redo=[];
    ui.measureDirty=true;feedback();draw();
  }
  function restore(from,to) {
    if (!from.length || ui.busy || saving) return;
    to.push(snapshot());const before=from.pop();draft=before.draft;selected=before.selected;
    ui.measureDirty=true;feedback();draw();
  }
  $('undoNote').onclick=()=>restore(undo,redo);
  $('redoNote').onclick=()=>restore(redo,undo);
  $('recognize').onclick=action(async()=>{
    if(ui.state.recognition && !confirm('重新识别会替换当前结果和手工校对内容，继续？'))return;
    await start('/recognize',undefined,()=>{ui.measureDirty=false;});
  });
  $('resumeOcr').onclick=action(()=>start('/recognize',{resume:true}));
  $('retryIssues').onclick=action(async()=>{
    if(!confirm('重新识别会替换标记小节中的内容，继续？'))return;
    await start('/recognize',{measures:ui.state.review_measures});
  });
  $('retryMeasure').onclick=action(async()=>{
    if(!confirm('重新识别会替换当前小节的编辑，继续？'))return;
    ui.measureDirty=false;dialog.close();
    await start('/recognize',{measures:[ui.measureIndex+1]});
  });
  function renderMeasures() {
    const measures=ui.state.measures||[];
    $('resumeOcr').hidden=!ui.state.ocr_task;
    $('retryIssues').hidden=!ui.state.review_measures?.length;
    $('ocrEmpty').hidden=!!measures.length;$('reviewEditor').hidden=!measures.length;
    $('recognize').textContent=measures.length?'重新识别整谱':'识别乐谱';
    $('recognize').classList.toggle('primary',!measures.length);
    $('recognitionSummary').textContent=`${measures.length} 小节 · ${ui.state.review_measures?.length||0} 个待检查`;
    $('scoreTitle').textContent=ui.state.metadata?.title||ui.state.input_names?.[0]||'未命名乐谱';
    $('scoreCredits').textContent=ui.state.metadata?.artist||'';
    overview.render(ui.state);
    updateMeasureControls();
    if(!measures.length){dialog.close();draft=null;return;}
    ui.measureIndex=Math.min(ui.measureIndex,measures.length-1);
    $('measureSelect').replaceChildren(...measures.map((m,i)=>{
      const o=el('option',`第 ${i+1} 小节${m.needs_review?' · 待检查':m.reviewed?' · 已确认':''}`);o.value=i;return o;
    }));
    if(dialog.open)renderMeasure();
  }
  async function selectMeasure(index) {
    if(ui.busy||saving)return;
    if(ui.measureDirty && !await save()){$('measureSelect').value=ui.measureIndex;return;}
    ui.measureIndex=Math.max(0,Math.min(ui.state.measures.length-1,index));
    if(!dialog.open)dialog.showModal();
    renderMeasure();overview.highlight(ui.measureIndex);
    $('scoreCanvas').focus({preventScroll:true});
  }
  async function closeEditor() {
    if(ui.busy||saving)return;
    if(ui.measureDirty && !await save())return;
    dialog.close();
    $('scoreOverview').querySelector(`[data-measure="${ui.measureIndex}"]`)?.focus({preventScroll:true});
  }
  $('closeMeasure').onclick=closeEditor;
  dialog.addEventListener('cancel',e=>{e.preventDefault();closeEditor();});
  $('measureSelect').onchange=()=>selectMeasure(+$('measureSelect').value);
  $('prevMeasure').onclick=()=>selectMeasure(ui.measureIndex-1);
  $('nextMeasure').onclick=()=>selectMeasure(ui.measureIndex+1);
  function openIssue() {
    const indexes=ui.state.measures.flatMap((m,i)=>m.needs_review?[i]:[]);
    if(indexes.length)selectMeasure(dialog.open?(indexes.find(i=>i>ui.measureIndex)??indexes[0]):indexes[0]);
  }
  $('nextIssue').onclick=openIssue;
  function renderMeasure() {
    const m=current();profile=measureProfile(m,ui.state);
    draft=clone(m.parsed);undo=[];redo=[];
    selected={vi:0,ei:0,ni:draft.voices[0]?.events[0]?.notes?.length?0:-1,
      string:draft.voices[0]?.events[0]?.notes?.[0]?.string||1,kind:profile.mode==='notation'?'notation':'tab'};
    $('measureDialogTitle').textContent=`第 ${ui.measureIndex+1} 小节`;
    $('measureSelect').value=ui.measureIndex;$('measureImage').src=m.url;$('measureText').value=m.score_text;
    $('reviewStatus').textContent=m.needs_review?'待检查':m.reviewed?'已确认':'识别完成';
    $('reviewStatus').className=`pill ${reviewKind(m)}`;
    $('fallback').hidden=!m.needs_review;
    $('fallback').textContent=m.timing_errors?.length?`${m.timing_errors.join('；')}。请调整起点或时值。`
      :m.fallback_reason?.length?'识别失败，当前为休止占位。请对照原图修改；原谱确为休止时可直接确认。'
      :'请对照原谱核对谱号、移调和音高，确认后即可导出。';
    $('editMode').value=ui.editorMode;editorMode();feedback();draw();
  }
  function updateMeasureControls() {
    const count=ui.state?.measures?.length||0,busy=ui.busy||saving;
    $('prevMeasure').disabled=busy||ui.measureIndex===0;
    $('nextMeasure').disabled=busy||ui.measureIndex>=count-1;
    $('nextIssue').hidden=!ui.state?.review_measures?.length;
    $('toExport').disabled=busy||!count;
    $('saveMeasure').disabled=busy||(!ui.measureDirty && !current()?.needs_review && !!current()?.reviewed);
    $('saveMeasure').textContent=ui.measureDirty?'保存修改':current()?.reviewed?'已保存并确认':'确认此小节';
    $('saveStatus').textContent=saving?'正在保存…':ui.measureDirty?'有未保存的修改，切换小节时会保存。':'修改后保存即可更新整谱。';
    $('discardMeasure').hidden=!ui.measureDirty;
    $('undoNote').disabled=busy||!undo.length;$('redoNote').disabled=busy||!redo.length;
  }
  function editorMode() {
    $('scoreEditor').hidden=ui.editorMode!=='score';$('textEditor').hidden=ui.editorMode!=='text';
  }
  $('editMode').onchange=async()=>{
    const next=$('editMode').value;
    if(ui.measureDirty && !await save()){$('editMode').value=ui.editorMode;return;}
    ui.editorMode=next;$('editMode').value=next;editorMode();if(next==='score')draw();
  };
  function draw() {
    if(!draft)return;
    $('timeSignature').value=draft.time_signature||'';$('measureTempo').value=draft.tempo_quarter||'';
    try {
      renderScore($('scoreCanvas'),draft,profile,selected,hit=>{
        selected={...hit};digits='';draw();$('scoreCanvas').focus({preventScroll:true});
      }, {onPlace:({vi,ei,pitch})=>change(()=>{
        selected={vi,ei,ni:-1,string:1,kind:'notation'};
        selected.ni=setPitch(event(),-1,pitch+pitchShift(profile.pitch_context,event().effects,profile.instrument),profile.mode,profile.tuning);
      })});
      $('scoreWarning').hidden=true;
    } catch(error) {
      $('scoreWarning').hidden=false;$('scoreWarning').textContent='本小节暂时无法绘制，可切换到小节文本修改。';
      console.warn('Score rendering failed',error);
    }
    const e=event(),n=e?.notes[selected?.ni];
    if(e){
      $('editorVoice').value=draft.voices[selected.vi].voice;
      for(const b of $('durationTools').children)b.setAttribute('aria-pressed',String(+b.dataset.duration===e.duration.value));
      $('durationDots').value=e.duration.double_dotted?2:e.duration.dotted?1:0;
      const tuplet=`${e.duration.tuplet_enters||1}:${e.duration.tuplet_times||1}`;
      $('eventTuplet').value=tuplet;
      if(!$('eventTuplet').value){const o=el('option',tuplet);o.value=tuplet;$('eventTuplet').append(o);$('eventTuplet').value=tuplet;}
      $('eventStart').value=e.start/960;
      $('selectedNote').textContent=`声部 ${draft.voices[selected.vi].voice+1} · 起点 ${e.start/960} 拍${selected.kind==='tab'?` · 第 ${selected.string} 弦`:''}${n?'':' · 休止或空位'}`;
    }
    const drums=profile.instrument==='drums',tab=selected.kind==='tab';
    $('fretTools').hidden=!tab;$('pitchTools').hidden=tab||drums;$('drumTools').hidden=!drums;
    $('raiseNote').hidden=$('lowerNote').hidden=drums;
    if(drums && n)$('drumValue').value=n.pitch;
    $('editHint').textContent=tab?'点击弦位，再用品位栏或数字键输入。左右键换拍，上下键换弦，Delete 删音。'
      :drums?'点选鼓音，再从鼓件栏选择鼓件。左右键换拍，Delete 删音。'
      :'点击五线谱空位可添加音符。左右键换拍，上下键升降半音，Delete 删音。';
    for(const b of $('fretButtons').children)b.setAttribute('aria-pressed',String(n?.fret!==undefined && String(n.fret)===b.dataset.fret));
    updateMeasureControls();
  }
  function applyValue(value) {
    change(()=>{
      const e=event();
      if(selected.kind==='tab')selected.ni=setFret(e,selected.string,value,profile.mode,profile.tuning);
      else {
        const pitch=profile.instrument==='drums'?Number(value):parsePitch(value)+pitchShift(profile.pitch_context,e.effects,profile.instrument);
        selected.ni=setPitch(e,selected.ni,pitch,profile.mode,profile.tuning);
        selected.string=e.notes[selected.ni].string||selected.string;
      }
    });
  }
  function fretButtons() {
    const begin=+$('fretRange').value;
    $('fretButtons').replaceChildren(...[...Array(Math.min(12,30-begin)+1)].map((_,i)=>String(i+begin)).concat('X').map(value=>{
      const b=el('button',value);b.dataset.fret=value;b.title=value==='X'?'闷音':`第 ${value} 品`;
      b.onclick=()=>applyValue(value);return b;
    }));
  }
  $('fretRange').onchange=()=>{fretButtons();draw();};fretButtons();
  $('pitchButtons').replaceChildren(...['C','D','E','F','G','A','B'].map(name=>{
    const b=el('button',name);b.onclick=()=>applyValue(name+$('pitchOctave').value);return b;
  }));
  $('drumValue').replaceChildren(...Object.entries(drumNames).map(([v,text])=>{const o=el('option',text);o.value=v;return o;}));
  $('drumValue').onchange=()=>applyValue($('drumValue').value);
  $('placeDrum').onclick=()=>applyValue($('drumValue').value);
  function shiftNote(delta) {
    change(()=>{
      const e=event(),n=e.notes[selected.ni];if(!n)throw new Error('请先选择一个音符。');
      if(selected.kind==='tab')selected.ni=setFret(e,selected.string,(n.fret==='x'?0:n.fret)+delta,profile.mode,profile.tuning);
      else selected.ni=setPitch(e,selected.ni,n.pitch+delta,profile.mode,profile.tuning);
    });
  }
  $('lowerNote').onclick=()=>shiftNote(-1);$('raiseNote').onclick=()=>shiftNote(1);
  function removeSelected() {change(()=>{if(selected.ni>=0)removeNote(event(),selected.ni);selected.ni=-1;});}
  $('deleteNote').onclick=removeSelected;
  $('deleteEvent').onclick=()=>change(()=>{selected.ei=removeEvent(draft.voices[selected.vi],selected.ei);selected.ni=-1;});
  $('makeRest').onclick=()=>change(()=>{event().notes=[];event().status='rest';selected.ni=-1;});
  $('addEvent').onclick=()=>change(()=>{selected.ei=insertEvent(draft.voices[selected.vi],selected.ei);selected.ni=-1;});
  $('addChordNote').onclick=()=>change(()=>{
    const e=event();
    if(profile.mode!=='notation'){
      const string=profile.tuning.findIndex((_,i)=>!e.notes.some(n=>n.string===i+1))+1;
      if(!string)throw new Error('这个和弦的每根弦都已有音符。');
      selected.string=string;selected.kind='tab';selected.ni=setFret(e,string,0,profile.mode,profile.tuning);
    }else{
      const pitch=profile.instrument==='drums'?+$('drumValue').value:parsePitch('C'+$('pitchOctave').value)+pitchShift(profile.pitch_context,e.effects,profile.instrument);
      selected.ni=setPitch(e,-1,pitch,profile.mode,profile.tuning);
    }
  });
  function duration(value=event()?.duration.value) {
    change(()=>{
      const [enters,times]=$('eventTuplet').value.split(':').map(Number);
      changeDuration(draft.voices[selected.vi],selected.ei,{...event().duration,value,
        dotted:$('durationDots').value==='1',double_dotted:$('durationDots').value==='2',tuplet_enters:enters,tuplet_times:times});
    });
  }
  $('durationTools').onclick=e=>{const b=e.target.closest('[data-duration]');if(b)duration(+b.dataset.duration);};
  for(const id of ['durationDots','eventTuplet'])$(id).onchange=()=>duration();
  $('eventStart').onchange=()=>change(()=>{
    const value=Number($('eventStart').value);if(!Number.isFinite(value)||value<0)throw new Error('起点不能小于零。');
    const chosen=event();chosen.start=Math.round(value*960);
    draft.voices[selected.vi].events.sort((a,b)=>a.start-b.start);selected.ei=draft.voices[selected.vi].events.indexOf(chosen);
  });
  $('editorVoice').onchange=()=>{
    const id=+$('editorVoice').value,index=draft.voices.findIndex(v=>v.voice===id);
    if(index>=0){selected={vi:index,ei:0,ni:-1,string:1,kind:profile.mode==='notation'?'notation':'tab'};draw();}
    else change(()=>{draft.voices.push({voice:id,events:[{start:0,duration:{value:4},status:'rest',notes:[],effects:[]}]});selected.vi=draft.voices.length-1;selected.ei=0;selected.ni=-1;});
  };
  $('timeSignature').onchange=()=>change(()=>{draft.time_signature=$('timeSignature').value.trim()||null;draft.print_time_signature=!!draft.time_signature;});
  $('measureTempo').onchange=()=>change(()=>{draft.tempo_quarter=$('measureTempo').value?+$('measureTempo').value:null;});
  $('measureText').oninput=()=>{ui.measureDirty=true;updateMeasureControls();};
  dialog.addEventListener('keydown',e=>{
    if(ui.busy||saving||ui.editorMode!=='score'||e.target.matches('input,textarea,select'))return;
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();restore(e.shiftKey?redo:undo,e.shiftKey?undo:redo);}
  });
  $('scoreCanvas').onkeydown=e=>{
    if(ui.busy||saving||!event()||e.ctrlKey||e.metaKey)return;
    if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();removeSelected();return;}
    if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
      e.preventDefault();selected.ei=Math.max(0,Math.min(draft.voices[selected.vi].events.length-1,selected.ei+(e.key==='ArrowLeft'?-1:1)));
      selected.ni=selected.kind==='tab'?event().notes.findIndex(n=>n.string===selected.string):(event().notes.length?0:-1);digits='';draw();return;
    }
    if(e.key==='ArrowUp'||e.key==='ArrowDown'){
      e.preventDefault();if(selected.kind==='tab'){
        selected.string=Math.max(1,Math.min(profile.tuning.length,selected.string+(e.key==='ArrowUp'?-1:1)));
        selected.ni=event().notes.findIndex(n=>n.string===selected.string);draw();
      }else if(profile.instrument!=='drums')shiftNote(e.key==='ArrowUp'?1:-1);
      return;
    }
    if(selected.kind==='tab'&&/^[0-9xX]$/.test(e.key)){
      e.preventDefault();digits=Date.now()-digitTime<800?digits+e.key:e.key;digitTime=Date.now();
      if(digits.length>2||Number(digits)>30||/x/i.test(digits))digits=e.key;
      applyValue(digits);
    }else if(e.key.toLowerCase()==='r'){$('makeRest').click();e.preventDefault();}
    else if(selected.kind==='notation'&&profile.instrument!=='drums'&&/^[a-g]$/i.test(e.key)){applyValue(e.key.toUpperCase()+$('pitchOctave').value);e.preventDefault();}
  };
  async function save() {
    if(saving||ui.busy)return false;
    saving=true;setBusy(true);feedback();
    try {
      const body=!ui.measureDirty?{reviewed:true}:ui.editorMode==='text'?{target:$('measureText').value.trim(),reviewed:true}:{measure:draft,reviewed:true};
      receiveProject(await api(endpoint(`/measures/${ui.measureIndex+1}`),'PUT',body,ui.state.revision));
      ui.measureDirty=false;renderMeasures();renderExport();
      feedback('已保存并确认。');return true;
    }catch(error){feedback(error.message,true);notice(error.message,true);return false;}
    finally{saving=false;setBusy(false);updateMeasureControls();}
  }
  $('saveMeasure').onclick=save;
  $('discardMeasure').onclick=action(async()=>{
    if(!confirm('放弃当前小节未保存的修改，载入已保存的内容？'))return;
    setBusy(true);
    try {receiveProject(await api(endpoint('')));renderMeasures();renderExport();}
    finally {setBusy(false);}
  });
  $('toExport').onclick=()=>go(4);
  let scoreWidth=0;
  new ResizeObserver(([entry])=>{
    const width=Math.round(entry.contentRect.width);
    if(width>0&&width!==scoreWidth){scoreWidth=width;if(draft&&dialog.open&&!$('scoreEditor').hidden)draw();}
  }).observe($('scoreScroll'));
  return {renderMeasures,updateMeasureControls,openIssue};
}
