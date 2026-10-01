# Native music export: compatibility contracts

Runtime code is Rust-only (`src/music_exports.rs` and its submodules), using
`encoding_rs` for text. Python is retained only for development reference tests
and the separate research/training pipeline. No exporter launches an interpreter.

Public APIs accept either a recognition manifest or `guitarocr.score/2`:
- `score_musicxml(&Value) -> Result<Vec<u8>, String>`
- `write_musicxml(&Value, &Path) -> Result<Value, String>`
- `write_gp5(&Value, &Path) -> Result<Value, String>`

## Preserve existing policy; do not infer bugs from invisible fields

The score IR remains authoritative. File-format projection is distinct from IR
validity and OCR soft warnings. The following defaults are intentional contracts
or observed legacy-file behavior, not claims about a real Guitar Pro application:

- Overfull bars remain legal and GP5-exportable. They are neither trimmed nor
  refused. Source: `tests/test_m2_semantics.py::test_overfull_legacy_measure_remains_legal_and_exportable`.
- TAB omits accidental-swap spelling and key metadata. These belong to notation
  display, not TAB supervision. Source: comments in `shared/m2.py::_note_token`
  and `format_measure_target`.
- MusicXML stays at 960 divisions and truncates fractional duration ticks with
  `int`, while IR arithmetic remains rational. Source:
  `tests/test_m2_semantics.py::test_duration_math_stays_exact_until_export_projection`.
- GP5's legacy codec does not serialize `isDoubleDotted`. A source quarter with
  two dots remains 1680 ticks in IR and MusicXML but projects to a 960-tick GP5
  note; if another note begins at 1680, the existing writer inserts a 720-tick
  rest. Native output matches those real legacy files. The proposed tied-duration
  replacement was removed; it is not a default or hidden option.
- GP5 dynamics retain the codec's coarse velocity mapping, including values that
  PyGuitarPro decodes as -1 for the lowest step. This is a codec observation, not
  a claim about real-device playback. Quantization is recorded in the report.
- Existing partial voices are not padded to full bars, and missing V1 remains
  empty. Missing whole voice-groups receive the existing V0 rest placeholder.
- GP5 takes initial tempo from the first track; explicit beat `tempo:` is encoded.
  It does not invent additional beat tempo events from later measure metadata.
- GP5 uses already-sounding pitches without applying written-pitch context again;
  MusicXML retains the existing transposition/clef/ottava projection.
- MusicXML preserves the prior projection: chord symbols/frames, ottava, clefs,
  transposing-instrument pitch/key spelling, ties, beams, tuplets, note-technique
  text, section and repeat barlines. It does not newly render navigation signs,
  alternate-ending brackets, swing, dynamics or beat tempo that the old function
  omitted. Those fields remain in the source IR.

The double-dot observation was checked at three levels: M2 arithmetic,
PyGuitarPro's in-memory model, and actual legacy GP5 files read by standard and
identity-aware PyGuitarPro readers. No real Guitar Pro application, screenshot,
engraving, or audio playback was used; application behavior is not established.
History traces the extra `isDoubleDotted` assignment to commit `6748134`; no
specific comment proving intentional double-dot duration loss was found. It is
therefore described as a projection difference, not conclusively as a product bug.

## Implemented native GP5 mapping

Physical and pitch-only notes, tie-chain string reservation, virtual pitched and
percussion slots, seven-string/two-voice synchronized splitting, explicit gap
rests, dots/rational tuplets, excerpt tie reports, CP936, display modes, shared part MIDI
channels, key/meter/repeats/endings/sections/navigation/swing, chord names and
frames/barres, velocity/accidental flags, bends, all six slides, grace/trill/tremolo,
natural/artificial/tapped/pinch/semi harmonics, vibrato/hammer/ghost/palm mute/
let-ring/articulations, pick/strum/slap/fade/rasgueado, text and ottava.

A bounded native reader checks emitted note values, ties and voice timing before
publication. This reader supports the encoder's subset, not arbitrary GP import.
Unknown techniques and actual unrepresentable format limits return explicit
errors; score JSON is not rewritten. Advanced fingering search/resource limits
and some dense virtual-slot partition choices
still require additional parity coverage; passing the fixed corpus is not proof
of universal byte-for-byte GP application equivalence.

CP936 compatibility includes an exhaustive BMP check: WHATWG GBK's extra mappings
are excluded so existing CP936 readers do not decode different characters. All
CP936-supported BMP characters were accounted for. Replacement and legacy byte
length truncation are reported; a split multibyte character is rejected, matching
the legacy reader's inability to read that output.

## Writes and validation

Validation/encoding finish before publishing a file. Each file is atomically
replaced. GP5's `.encoding.json` report is written before the GP5; if the second
replacement fails the report may remain. This is not a two-file transaction.

Run native tests with `cargo test --test music_exports`. The optional development
oracle `tests/verify_exports_oracle.py <native_export_binary>` creates both real
legacy and native GP5s and compares decoded notes, effects, timing, voice/track
mapping, display settings and navigation metadata. It also compares MusicXML
against the existing function's element tree. No model inference or packaging
is involved. `examples/native_export.rs` is the small test driver.

Wire-format reference: https://pyguitarpro.readthedocs.io/en/stable/pyguitarpro/format.html
The encoder is independently authored around published fields; PyGuitarPro 0.11
(LGPL-3.0-only) is only a development oracle, not vendored runtime code.
