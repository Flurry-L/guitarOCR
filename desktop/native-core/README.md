# GuitarOCR native core

Rust library for the desktop migration. It does not embed, launch, download, or
require Python. The Python training/research pipeline remains a separate project.
This library is an incremental replacement, not a claim that all desktop routes,
OCR behavior, exporters, or platform packaging have already been migrated.

Requires Rust 1.88 or newer (including resolved image/ICU dependencies).

## Portable project persistence

`project` implements the version-1 ZIP format used by `pipeline/archive.py`:

- `import_project(source, workspace, &ImportOptions) -> Result<ImportedProject, ImportError>`
- `export_project(workspace, session_id, destination) -> Result<PathBuf, ImportError>`
- `load_session(workspace, session_id) -> Result<serde_json::Value, ImportError>`
- `write_session_atomic(workspace, session_id, &state) -> Result<(), ImportError>`

`ImportOptions::default()` generates a fresh 32-character lowercase hexadecimal
ID, allows at most 100 pages, and caps uncompressed content at 2 GiB. Callers can
set `session_id`, `max_bytes`, and `max_pages`; caller limits cannot enlarge the
format limits. `ImportedProject` returns the ID, absolute directory, persisted
session JSON, and imported file count.

Import verifies ZIP paths, physical file counts, case-folded duplicates,
symlinks/special files, declared and actual file sizes, manifest membership,
SHA-256 checksums, JSON/JSONL syntax, page references, stage/schema identity,
recognition M2 syntax, and referenced files. `project://` strings are rewritten
to the selected workspace/new session directory. It preserves unknown JSON
fields and arbitrary-precision JSON numbers. The imported session has the new
ID, revision zero, and no originating-machine `ocr_task` checkpoint.

Only a newly created directory is cleaned up if import fails. Existing projects,
including destination symlinks, are never overwritten or deleted. `session.json`
is the final commit marker; import is not a single atomic directory rename.
The workspace must remain private to the caller during a write: this is not a
sandbox against a concurrent local process replacing directories.

Export preserves the Python archive shape and hashes. It converts in-project
references to portable paths and omits `session.json`, `job.json`, symlinks,
`tmp` directories, `.tmp` and `.zip` files. It can recover an interrupted final
JSONL record without changing the source. Other malformed JSON fails. Export
and session saving use a sibling temporary file followed by atomic replacement;
failed export keeps an existing destination unchanged. Revision comparison and
concurrent-edit locking belong to the caller.

### Deliberate stricter boundaries

- Import accepts ordinary single-disk ZIPs, including stored/deflated files;
  ZIP64 central directories, split ZIPs, self-extracting prefixes, and encryption
  are rejected
- Windows device names, forbidden filename characters, trailing dots/spaces,
  case-aliased ancestor directories, and file/directory collisions are rejected
- Every `project://` reference must match a listed file, including in unknown
  JSON fields; page references are checked in every JSON/JSONL file
- JSON must be UTF-8; numeric M2 tokens must fit the native score parser's integer
  bounds, while unrelated JSON numbers are preserved without float conversion
- Export destinations must be outside the source session directory

These boundaries accept the repository's real Python-exported demonstration
project. Recognition targets are validated by the shared native `score` module,
not a second importer-specific grammar. Saving retains JSON semantics, not the
original whitespace or object-key ordering.

## Quick checks

```sh
cargo check --locked --manifest-path desktop/native-core/Cargo.toml
cargo test --locked --manifest-path desktop/native-core/Cargo.toml
```

`tests/project_import.rs` exercises the real bundled demo, native
import/export/reimport, unknown fields, precision, SHA/CRC corruption, zip-slip,
case and exact duplicate entries, symlinks, invalid page references, byte/page
limits, invalid stage/M2 data, journal recovery, and failed-write preservation.
No model execution or Python is needed for these tests.
