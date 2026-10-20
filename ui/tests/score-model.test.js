import test from "node:test";
import assert from "node:assert/strict";
import { parseTuning, formatTuning, staffPositionChoices, staffPositionLabel, beatStartChoices, summarizeMText } from "../workbench/score-model.js";

test("tuning accepts note names, MIDI, and common separators", () => {
  const standard = [64, 59, 55, 50, 45, 40];
  assert.deepEqual(parseTuning("E4, B3 G3，D3、A2/E2"), standard);
  assert.deepEqual(parseTuning("64,59,55,50,45,40"), standard);
  assert.deepEqual(parseTuning(formatTuning(standard)), standard);
  assert.deepEqual(parseTuning("F♯3 Bb2 40"), [54, 46, 40]);
  for (const value of ["", "128", "E", "-1", Array(13).fill("E4").join(",")])
    assert.throws(() => parseTuning(value));
});

test("tuning does not shift string numbers when an entry is missing", () => {
  for (const value of ["E4, , G3, D3, A2, E2", "64,,55,50,45,40", ",E4", "E4,", "E4//B3"])
    assert.throws(() => parseTuning(value));
});

test("staff labels match the bottom line of each clef", () => {
  for (const [clef, note] of [["G2", "E4"], ["F4", "G2"], ["C3", "F3"], ["C4", "D3"]]) {
    const choices = new Map(staffPositionChoices(clef));
    assert.equal(choices.get("0"), `${note} · 第一线`);
  }
  const treble = new Map(staffPositionChoices("G2"));
  assert.equal(treble.get("-2"), "C4 · 下加1线");
  assert.equal(treble.get("8"), "F5 · 第五线");
  const drums = new Map(staffPositionChoices("percussion", "drums"));
  assert.match(drums.get("4"), /中嗵鼓/);
  assert.doesNotMatch(drums.get("4"), /军鼓/);
  assert.equal(staffPositionLabel(18, "G2"), "B6 · 上加5线");
  assert.equal(staffPositionLabel(-10, "G2"), "B2 · 下加5线");
  assert.equal(staffPositionLabel(18, "F4"), "D5 · 上加5线");
  assert.equal(staffPositionLabel(18, "percussion", "drums"), "上加5线");
});

test("beat choices follow the meter denominator and end inside the bar", () => {
  const sixEight = new Map(beatStartChoices("6/8"));
  assert.equal(sixEight.get("480"), "第 2 拍起点");
  assert.equal(sixEight.get("2400"), "第 6 拍起点");
  assert.ok([...sixEight.keys()].every(tick => +tick < 2880));
  assert.ok(beatStartChoices("3/16").every(([tick]) => +tick < 720));
  assert.equal(new Map(beatStartChoices("32/4")).get(String(31 * 960)), "第 32 拍起点");
});

test("M summaries retain every voice, chords, tuplets, and dotted durations", () => {
  const text = "M time=4/4 | V0{@0:q.:s1f5,s2f5 @1440:e[3:2]:r @1760:e[3:2]:^} || V1{@0:h..:p60}";
  const summary = summarizeMText(text);
  assert.match(summary, /声部 1（3 个事件）/);
  assert.match(summary, /声部 2（1 个事件）/);
  assert.match(summary, /1弦5品 \+ 2弦5品/);
  assert.match(summary, /附点四分音符/);
  assert.match(summary, /八分音符（3:2 连音） · 休止/);
  assert.match(summary, /同前事件/);
  assert.match(summary, /双附点二分音符 · C4/);
});

test("M summaries use drum keys and inherited meter", () => {
  const summary = summarizeMText("M | V0{@480:e:p36,p42<dyn:80>}", { instrument: "drums", timeSignature: "6/8" });
  assert.match(summary, /第 2 拍/);
  assert.match(summary, /底鼓 \+ 闭镲/);
  assert.doesNotMatch(summary, /dyn:/);
  assert.doesNotThrow(() => summarizeMText("M | V0{@0:e[3:"));
});
