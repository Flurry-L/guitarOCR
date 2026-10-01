# Native desktop packaging (unreleased)

The Tauri application launches only `guitarocr-native-service[.exe]` with
`--projects DIR --assets DIR --port 0`. The service reports
`GUITAROCR_READY http://127.0.0.1:PORT`. The launcher owns its process group/job,
keeps project files under application data, and times out failed startup.
Remote connections and protected download destinations remain available.

The default local launcher uses mode `native` and starts the packaged Rust
service, including editing and the local recognition workflow. It never falls
back to Python. Research/training Python remains in the checkout and can be run
separately. No Python, uv, backend code or model weights are bundled by this route.
Model preparation has a separate confirmation in the workbench; packaged native
libraries do not download on startup. Artifact metadata remains `preview: true`
until platform acceptance is complete; that is a release-status boundary, not a
second implementation or a read-only launcher mode.

## Build and resource layout

From `desktop`, run:

```
npm ci
cargo fetch --locked --manifest-path src-tauri/Cargo.toml
cargo fetch --locked --manifest-path native-service/Cargo.toml
npm run native:build
npm run resources
npm run build
```

The Node preparer has no network/download step. It stages only:

- `webapp/static/`: existing browser workbench assets
- `native/guitarocr-native-service[.exe]`: host-target Cargo-built service
- `native/manifest.json`: service digest and explicit component inventory
- `native/<component>/<relative file>`: optional validated native libraries,
  executables, licenses and notices
- `licenses/`: dependency license files and SPDX/source inventory for both
  Tauri and native-service Cargo dependency graphs, plus third-party notices

`--binary PATH` allows an already-built native service (e.g. a debug smoke-test
build). Binary format and architecture must match the packaging host. Thin
macOS target binaries are expected; universal binaries are not accepted yet.
`--destination DIR` supports isolated packaging checks. Neither option is a
substitute for service integration tests.

The retained `prepare_desktop.py` command is only a developer compatibility
wrapper for Node. It does not package Python. The Linux rootless build helper
also calls the Node preparer and builds native-service first.

## Supplying native inference dependencies

Use `node scripts/prepare_native_desktop.mjs --manifest /path/manifest.json`.
The manifest and every listed file are local build inputs. Obtain artifacts
from the official upstream build/release process or build reviewed pinned
source; an upstream source label alone is not artifact authenticity proof.
No guessed precompiled release URLs are used.

The manifest schema is:

```
{
  "schema": 1,
  "target": "linux-x64",
  "components": [
    {
      "name": "onnxruntime",
      "version": "<verified upstream version>",
      "source": "https://github.com/microsoft/onnxruntime",
      "files": [
        {"path": "lib/libonnxruntime.so", "sha256": "<actual 64-character SHA-256>", "role": "library"},
        {"path": "LICENSE", "sha256": "<actual 64-character SHA-256>", "role": "license"}
      ]
    }
  ]
}
```

`target` uses Node names: `linux-x64`, `win32-x64`, `darwin-arm64`,
`darwin-x64`. Additional supported component names/sources:

- `pdfium`: `https://pdfium.googlesource.com/pdfium`
- `llama.cpp`: `https://github.com/ggml-org/llama.cpp`; version must equal
  the commit pinned in `scripts/llamacpp-runtime.json`

Every component needs a license and at least one library (ORT/PDFium) or
executable (llama.cpp). Include all transitive dynamic libraries and notices,
not just the main binary. Relative paths are preserved within each component
so adjacent-library layouts remain possible. Symlinks, traversal, target
mismatches, duplicate file paths and incorrect SHA-256 are rejected. Copies of
versioned shared libraries must be regular files rather than symlinks.

`--require-inference` fails unless all three components are present. It is an
inventory gate, not a promise that recognition is complete: service wiring,
model installation/checksums, accelerator checks and end-to-end validation
remain separate requirements. The service receives `GUITAROCR_NATIVE_RESOURCES`
pointing at the staged `native/` directory; it must explicitly resolve component
paths from the manifest before enabling inference.

## Validation status and release gate

The workflow produces preview CI artifacts only, without release publication.
Its OS matrix is build configuration, not evidence of a successful run on those
platforms. Do not claim macOS/Windows support until those builds and runtime
checks pass. Linux resource-contract tests and cargo checks do not replace
full native recognition, packaging or signing validation.

Some upstream Cargo archives omit standalone license texts. The preview
manifest records those packages in `licenseReviewRequired`; their SPDX labels
and source URLs are still inventoried. Review and supply their actual upstream
license/notice files under `--license-overrides DIR/<package-version>/` before
a release. `--require-inference` rejects an unresolved license-text inventory
as well as missing native runtime components.

## Acquiring native build inputs

The manual desktop workflow now defaults to `include_inference: true`: it obtains
fixed official ORT 1.23.2 and PDFium 5.13.0 wheels, builds the pinned llama source
(CPU on Linux/Windows/Intel Mac, Metal on Apple Silicon), then stages a complete
inventory with `--require-inference`. Python is a build tool only. No Python files
or model weights enter the application. Turning that input off explicitly makes
an editor-only preview. These workflow paths still need native runner acceptance;
configuration alone is not a successful build or a release.

For a local build, obtain the matching platform's files from the official
[ORT release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2) or
[ORT PyPI release](https://pypi.org/project/onnxruntime/1.23.2/#files), and
[PDFium wheel release](https://pypi.org/project/pypdfium2/5.13.0/#files).
Do not install the wheels. Build/package llama with the existing pinned-source
helper first. Then run from the repository root, replacing the example paths:

```
python scripts/collect_native_assets.py --target macos-arm64-metal \
  --ort-archive /path/onnxruntime-osx-arm64-1.23.2.tgz \
  --pdfium-wheel /path/pypdfium2-5.13.0-py3-none-macosx_13_0_arm64.whl \
  --llama-archive /path/llamacpp-local-build.zip \
  --llama-manifest /path/macos-arm64-metal.json --output output/native-components
node scripts/prepare_native_desktop.mjs \
  --manifest output/native-components/manifest.json --require-inference
```

Run on the packaging host. Other targets are `linux-x64-cpu`, `windows-x64-cpu`
and `macos-x64-cpu`. ORT accepts its native tar/ZIP or wheel; PDFium uses its wheel.
The collector reads official upstream metadata and rejects a missing/mismatched
full archive digest. Wheels also require matching per-file RECORD hashes. Native
ORT archives require GitHub's actual release asset digest; a URL or local hash
alone is insufficient. Only regular native binaries and license/notice files are
copied, preserving directories; Python modules and SONAME symlinks are excluded.
The Node preparer subsequently checks every selected binary's platform/architecture.

The fixed PDFium Mac wheel requires macOS 13.0. The official ORT 1.23.2
Apple Silicon native library's `LC_BUILD_VERSION` records macOS 13.4 (SDK 14.4),
so the native application's minimum version is 13.4. Inspect all Mach-O dependencies before release and raise that baseline
if the actual binaries require more. Windows requires the documented VC++ runtime;
Linux records actual ELF requirements rather than assuming the host's baseline.

The original Linux development-cache route is retained:

```
python scripts/collect_native_assets.py --cache /path/to/existing/uv-cache \
  --llama-archive /path/llamacpp-local-build.zip \
  --llama-manifest /path/linux-x64-cpu.json --output output/native-components
```

It checks the cached wheel receipt against official PyPI metadata and each selected
file against RECORD, without downloading wheels. This is explicitly wheel-extracted
native code, not a claim of standalone upstream binary distribution. PDFium's full
build notices and llama's local-build provenance are retained in either route.

Reviewed supplementary Cargo license texts live in `desktop/native-licenses`
and are used by default. Their provenance identifies the published crate's
upstream commit, same-commit sibling, or the standard license explicitly named
in its published Cargo metadata. `selectors` also includes corresponding crate
source under MPL-2.0. `unicode-casefold` selects the declared Apache-2.0 option,
retains original crate author metadata, and includes Unicode's data license.
This is a reproducible notice inventory, not legal certification.

The macOS objc2-family supplements retain the original repository `LICENSE.md`
from each published crate's exact VCS commit, with archive SHA-256, Git blob ID,
file SHA-256 and crate metadata. The upstream file is a license declaration with
linked terms and an Apple SDK caveat; it is not replaced by invented MIT attribution
or presented as legal clearance. Source modules such as `src/copying.rs` are never
counted as license notices. The same collector can audit a target's Cargo graph
without compiling that target via `stageCargoLicenses(destination, overrides, target)`.

Even with a complete resource inventory, `manifest.preview` remains true until
application integration and platform acceptance are complete.

Three small read-only model metadata JSON files are also staged at
`native/models/`: `distribution.json`, `score_image_policy.json`, and
`capabilities.json`. Their digests and source paths are recorded under
`modelMetadata`. No model weights, tokenizer, training configuration or Python
source accompany them. Actual model acquisition remains a separate explicit
application operation. The current reused Linux llama build requires glibc
2.38 (recorded by the collector); rebuild on the intended baseline before
claiming compatibility with older Linux distributions.

## Native image, layout and score-structure core

The client core is Rust-only. `auxiliary_onnx` explicitly initializes a verified
absolute ONNX Runtime library path and loads the exported layout/signature
models. It does not invoke Python, download a runtime, or silently fall back to
research code. `ort 2.0.0-rc.10` requests the ORT 1.22 C API; the local native
smoke check also passed with the existing 1.23.2 shared library.

The model contracts are preserved:

- Layout: RGB float32 `[1,3,800,800]`, divided by 255; `im_shape=[800,800]`,
  `scale_factor=[800/original_height,800/original_width]`; `fetch_name_0` is
  `[N,7+]`. The exported graph already performs NMS and original-pixel mapping.
  The client clips rounded boxes and sorts column 7; it never rescales or NMSes
  those measure boxes again
- Signature: `[B,2,3,192,512]`, full measure plus staff-prefix view, left-aligned
  and vertically centered on white, mapped to `[-1,1]`. Named output heads are
  `key:[B,16]`, `numerator:[B,33]`, `denominator:[B,8]`. First-argmax tie behavior,
  absent key/time classes and denominator lookup match the research pipeline
- `image_transforms` implements Pillow's 22-bit-weight, two-pass Lanczos RGB8
  arithmetic and OpenCV's A=-0.75 cubic arithmetic. Nine small Pillow goldens
  are byte-exact; nine OpenCV default-backend goldens differ by at most one
  intensity level. OpenCV itself changes rounding between optimized backends;
  this is a bounded compatibility check, not a universal bit-parity claim

`layout_postprocess` ports row grouping, weighted non-overlapping selection,
notation/TAB duplicate resolution, row reconciliation, faint-fragment rejection,
and barline-supported gap recovery. Large-page overlapping detail crops add
only document instruction regions. `staff_geometry` supplies the real raster
staff/barline fallback and missing-bar evidence; `staff_classifier` supplies
layout classification and conservative TAB string-count voting. Manual rows
retain the editor's reading order without automatic deduplication.

`score_structure` validates/repairs row assignments, keeps grand-staff hands
and simultaneous fretted lanes distinct, reconciles page profiles by strict
majority, and maintains stable global part IDs across page turns. `score_grid`
resolves common simultaneous bar columns, merges false splits, splits spanning
boxes, recovers only visibly bounded missing bars, and fuses only complementary
notation/TAB rows assigned to the same part and staff. It regenerates changed
crops and preserves provenance/source measure numbers. Pure logic tests do not
substitute for a full recognition run.

### Image and PDF boundary

`image_boundary::for_each_image_page` decodes TIFF/BigTIFF frames in original
IFD order through a bounded callback, with per-frame EXIF orientation applied.
JPEG/WebP orientation is also applied. The single-image helper rejects multi-
page TIFF rather than dropping later pages. Limits are 40 megapixels per page,
200 megapixels per document and the caller's page count (100 in the service).
Initial measure crops use the original Pillow grayscale coefficients and
180-DPI context padding: 1.5 mm horizontally, 4 mm vertically.

PDF rendering uses PDFium's public C ABI through `libloading`, loaded only from
an explicit verified path. A global mutex protects initialization, rendering,
and all native handles; RAII destroys handles on errors. Documents are bounded
to 200 MiB and rendered at 180 DPI to grayscale PNGs with visible page rotation.
The output directory must be new; the application import transaction owns
cleanup on failure. PDFium is BSD-style, but the selected binary's complete
third-party notices, build version/source and real checksums must accompany it.
ONNX Runtime is MIT, ort/image are MIT OR Apache-2.0, and libloading is ISC.
These native binaries may be extracted from a verified wheel as explicitly
recorded build inputs; that does not introduce a Python client dependency.

The default automatic/image route detects every PDF page from pixels and reads
metadata through image regions. It therefore does not require PDF text or
vector-path extraction. The legacy explicit vector-geometry route uses
`layout/pdf_geometry.py` with `line_paths` and `words`; the PDF-only metadata
route uses `document_info/pdf_metadata.py`. Those text/vector extractors have
not been ported. A native request for that route must be visibly rejected or
explicitly switch to image processing with a warning; it must never pretend
that PDF vector/text evidence was read. Manual layouts without document regions
also need an explicit image-region recovery/warning before metadata recognition.

Focused checks include eight layout geometry reference cases, six raster staff
fixtures, four common-grid/fusion reference cases, twelve structure parse/repair
fixtures, cross-page identity/majority/recovery cases, TIFF/JPEG orientation,
and two opt-in actual-library smoke tests (one synthetic signature inference
and one tiny PDF render). No layout-model performance run or complete-package
validation is implied by these checks.

The resampling code is a source-informed adaptation, not a clean-room claim.
Retain `desktop/native-core/licenses/` in the native distribution: the fixed
OpenCV 4.13.0 source-file notice and project Apache-2.0 license, Pillow 12.3.0
MIT-CMU license, and `image-transforms-SOURCES.txt` provenance/digests. These
source-adaptation notices supplement the Cargo dependency license inventory.

### Conservative annotation pixel geometry

`pixel_refinement` ports chord-grid morphology, spacing fit, visible dot/open/mute
markers and barre bridges. OpenCV's even-kernel anchor/border behavior is preserved
by a native binary morphology implementation; no OpenCV/Python runtime is added.
A skewed/incomplete/ambiguous grid or missing visible marker returns no correction,
so the original model diagram is retained. Printed base fret and fingers remain
unchanged; chord names never supply positions. The recognition caller performs
the existing diagram normalization after this optional correction.

`ottava_geometry` links only visible bare dashed continuations to an explicit
printed ±12/±24 octave anchor. It preserves same-staff identity, above/below side,
original gap/margin tests, and stopping at hooks, new text, unmarked intervening
rows or replacement labels. TAB-only records are excluded. It returns copied
predictions with `model_parsed` and `octave_continuation` provenance; it does not
shift notes, modify event timing, or alter export/display projections. The native
image-resolver adapter supports stored crops and equivalent in-memory region
crops without temporary files. Small Python-reference goldens cover chord/open/
mute/barre evidence, malformed grids and dashed-line continuation boundaries;
these checks are not full model accuracy or visual playback validation.
