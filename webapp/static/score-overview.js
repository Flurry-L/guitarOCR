import { el } from './dom.js';
import { renderScore } from './score-renderer.js';

export function measureProfile(measure, state) {
  const metadata = state.metadata || {};
  return { mode:measure.mode || state.mode, instrument:measure.instrument || metadata.instrument || 'guitar',
    tuning:measure.tuning || metadata.tuning_used || [64,59,55,50,45,40],
    pitch_context:{...(measure.pitch_context || {}),capo:metadata.capo || 0} };
}
export function reviewKind(measure) {
  if (!measure.needs_review) return '';
  return measure.fallback_reason?.length || measure.timing_errors?.length ? 'failed' : 'review';
}

// Lay out complete systems. Off-screen measures are drawn only when approached.
export function scoreOverview(host, onOpen) {
  let state, observer, width = 0;
  function render(next = state) {
    state = next;
    if (!state) return;
    observer?.disconnect(); host.replaceChildren();
    const available = host.clientWidth || 1000;
    let system, used = 0, previousProfile = '', time = '4/4';
    observer = new IntersectionObserver(entries => {
      for (const {target,isIntersecting} of entries) if(isIntersecting) {
        target.draw(); observer.unobserve(target);
      }
    }, {rootMargin:'500px'});
    state.measures.forEach((measure,index) => {
      const profile = measureProfile(measure,state);
      const signature=JSON.stringify([profile.mode,profile.instrument,profile.tuning,profile.pitch_context.clef]);
      const beats = new Set(measure.parsed.voices.flatMap(v=>v.events.map(e=>e.start))).size;
      const natural = Math.max(190,100 + beats*27);
      if (!system || used+natural>available || signature!==previousProfile) {
        system=el('div',undefined,'score-system');host.append(system);used=0;
      }
      const first=used===0;
      used+=natural; previousProfile=signature;
      time=measure.parsed.time_signature || time;
      const kind=reviewKind(measure), status=kind==='failed'?'识别失败':kind==='review'?'待检查':measure.reviewed?'已确认':'';
      const bar=el('button',undefined,`score-bar ${kind}`);
      bar.type='button';bar.dataset.measure=index;bar.style.flexGrow=natural;
      bar.style.flexBasis=`${natural}px`;
      bar.setAttribute('aria-label',`第 ${index+1} 小节${status?'，'+status:''}，点击编辑`);
      const caption=el('span',undefined,'bar-caption');
      caption.append(el('span',String(index+1)),el('span',status,'bar-status'));
      const notation=el('div',undefined,'bar-notation');
      notation.style.height=`${profile.mode==='notation'?225:(profile.mode==='both'?205:42)+130+Math.max(0,profile.tuning.length-5)*13}px`;
      bar.append(caption,notation);system.append(bar);
      bar.onclick=()=>onOpen(index);
      const effective={...measure.parsed,time_signature:time};
      bar.draw=()=>{
        try {
          renderScore(notation,effective,profile,null,null,{overview:true,
            width:Math.max(natural,bar.clientWidth),showClef:first,
            showTime:first || !!measure.parsed.print_time_signature});
        } catch(error) {
          notation.replaceChildren(el('span','谱面无法绘制，点击校对','render-error'));
          bar.classList.add('failed');
          console.warn('Measure rendering failed',index+1,error);
        }
      };
      observer.observe(bar);
    });
  }
  new ResizeObserver(([entry])=>{
    const next=Math.round(entry.contentRect.width);
    if(next>0 && next!==width){width=next;render();}
  }).observe(host);
  return {render, highlight(index) {
    for(const bar of host.querySelectorAll('.score-bar'))bar.classList.toggle('active',+bar.dataset.measure===index);
  }};
}
