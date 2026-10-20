import test from "node:test";
import assert from "node:assert/strict";
globalThis.location = new URL("http://localhost/");
const { chordLibraryEntries } = await import("../workbench/score-chords.js");

test("a part's chord library does not borrow another part's metadata", () => {
  const root = {document_metadata: {score_annotations: [{parsed: {kind: "chord", text: "C"}}]}};
  const bass = {id: "bass", document_metadata: {score_annotations: [{parsed: {kind: "chord", text: "Em"}}]}};
  assert.deepEqual(chordLibraryEntries({metadata: root}).map(c => c.text), ["C"]);
  const metadata = {...root, parts: [{...root, id: "guitar"}, bass]};
  assert.deepEqual(chordLibraryEntries({metadata}, "bass").map(c => c.text), ["Em"]);
  assert.deepEqual(chordLibraryEntries({metadata}, "missing"), []);
});
