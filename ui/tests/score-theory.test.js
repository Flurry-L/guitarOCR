import test from "node:test";
import assert from "node:assert/strict";
import { writtenKey, keyCode, keyChoices, parseDuration } from "../workbench/score-theory.js";

test("printed keys preserve minor mode and normalize the supported accidental spellings", () => {
  for (const key of ["BMajorFlat", "BFlatMajor", "B♭Major", "BbMajor"])
    assert.equal(writtenKey(key, -2), 0); // Concert B♭ is written C for B♭ instruments.
  assert.equal(keyCode(writtenKey("CMinor", -2), true), "DMinor");
  assert.equal(writtenKey("CMajor", 2), -2);
  assert.equal(writtenKey(-3, -2), -3); // Observation keys are already written.
  for (const choice of keyChoices)
    assert.equal(keyCode(writtenKey(choice.value, 0), choice.value.includes("Minor")), choice.value);
});

test("theoretical keys use an equivalent display key and unknown keys report the source", () => {
  assert.equal(writtenKey("FMajorFlat", 0), 4);
  assert.equal(writtenKey("GMajorSharp", 0), -4);
  assert.equal(writtenKey("EMinorSharp", 12), -4);
  assert.throws(() => writtenKey("unreadable", 0), /unreadable/);
});

test("custom durations retain dots and tuplet ratios while incomplete input stays incomplete", () => {
  assert.deepEqual(parseDuration("d128..[7:4]"), {
    value: 128, dotted: false, double_dotted: true, tuplet_enters: 7, tuplet_times: 4,
  });
  for (const token of ["e[3:", "q[0:1]", "d0", "q..."]) assert.equal(parseDuration(token), null);
});
