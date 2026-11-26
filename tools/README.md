# Tools

The scripts in this directory fall into three groups: the analysis checks that also run under
`ctest`, the drivers for static analysis and coverage, and the qualification scripts that answer
one question each with sequential Theia and Hyperion renders.

| Script | Group | Needs | Question it answers |
|--------|-------|-------|---------------------|
| `check_arch.py` | analysis check | Python | Does the include graph of `src/` follow `ARCHITECTURE.md`? |
| `check_dead.py` | analysis check | a build directory | Is there dead code? |
| `check_tidy.py` | static analysis | a configured build | Does clang-tidy report correctness findings? |
| `coverage.py` | coverage | Clang | Which lines of `src/` do the tests cover? |
| `generate_gallery.ps1` | render | the GPU | The 30 README screenshots |
| `qualify_sms.py` | qualification | the GPU, the family checkouts | Is manifold sampling unbiased and stable? |
| `test_qualify_sms.py` | CPU test | Python | Does `qualify_sms.py` compute what it should? |
| `qualify_gap_c.py` | qualification | the GPU, the family checkouts | Does the third-vertex reconnection shift run? |
| `qualify_k4.py` | qualification | the GPU, the family checkouts | Is the fourth-vertex reconnection shift correct? |
| `exercise_bias_mode.py` | qualification | the GPU, the family checkouts | Do the bias-correction levels do what they say? |
| `check_dead.exceptions` | — | — | The accepted findings of `check_dead.py` |

All scripts are run from the repository root. Exit code 77 means "skipped": a toolchain or an input
is missing, and `ctest` treats it as a skip.

## Analysis checks

### `check_arch.py`

The vendored engine (version 1.5.1) of the check-arch skill. It reads the `#include` lines of every
`.hpp` and `.cpp` file under `src/` and compares the resulting graph with the contract in the `arch`
block of `ARCHITECTURE.md`. The rules of the block are explained in
[ARCHITECTURE.md](../ARCHITECTURE.md).

```powershell
python tools/check_arch.py                      # check
python tools/check_arch.py --report             # also print the layer stack and the measured graph
python tools/check_arch.py --json              # the summary as JSON
python tools/check_arch.py --diagram            # rewrite the generated diagram section
python tools/check_arch.py --derive --out draft.md   # draft a contract from the measured graph
python tools/check_arch.py --selftest           # the engine's offline self-test
```

`--repo DIR` selects the repository (default: the directory above the script) and `--doc FILE` the
contract (default: `ARCHITECTURE.md` at the root).

Output: one line per violation — an include along an undeclared edge, an edge that points up the
layer stack, an include cycle, a file that belongs to no module, a file above the line limit, a
forbidden include, or a diagram section that differs from the generated one. With `--report`, the
smell proxies follow (large module, hub, mutual pair, orphan); they are information and never fail
the check. `ctest` registers the script as `test_arch` where the check-arch engine is installed in
the user's opencode skills directory.

Exit codes: 0 clean, 1 violations, 2 usage or contract error, 77 skipped.

### `check_dead.py`

The vendored engine (version 1.1.2) of the check-dead skill. It reports three classes of findings:

- orphan headers: headers that no file includes;
- never-compiled sources: `.cpp` files that are neither in the newest `compile_commands.json` under
  a `build*` directory nor named in any CMake file;
- FetchContent declarations that are never made available.

```powershell
python tools/check_dead.py --family .. --report
```

`--family DIR` names the directory that holds the sibling repositories. A header with no consumer in
this repository is then checked against the siblings' include graphs; a header that a sibling
includes is listed as rescued and is not a finding. `ctest` registers the script as `test_dead` with
`--family` set to the parent of the repository, where the check-dead engine is installed.

A finding that is justified goes into `check_dead.exceptions`, one line per finding:
`<orphan-headers|never-compiled|fetchcontent> <glob-or-name> <reason>`. The file is empty.

Output: one line per finding, with `--report` the derived scan roots and file counts first.

Exit codes: 0 clean, 1 findings, 77 nothing scanned.

## Static analysis and coverage

### `check_tidy.py`

Runs clang-tidy over every project translation unit in `build/compile_commands.json`, in parallel.
clang-tidy is taken from the LLVM component of the newest Visual Studio installation (found through
vswhere) or from a fixed fallback path. Translation units under `build/`, `_deps/` and
`third_party/` are skipped.

```powershell
python tools/check_tidy.py [--db build\compile_commands.json] [--jobs N]
```

Output: findings are classified by the `WarningsAsErrors` list of `.clang-tidy`. `clang-diagnostic-*`,
`clang-analyzer-*` and `bugprone-*` findings are printed and fail the run; `modernize-*`,
`performance-*` and `portability-*` findings are summarised by count and do not fail. The complete
output per file goes to `build/clang_tidy.log`. `ctest` registers the script as `test_tidy`.

Exit codes: 0 clean, 1 failing findings, 77 clang-tidy or the database missing.

### `coverage.py`

Builds an instrumented tree, runs the tests with profiling enabled and prints the coverage report
for `src/`.

```powershell
python tools/coverage.py                    # configure, build, test, report
python tools/coverage.py --skip-build       # reuse the existing build/coverage
python tools/coverage.py --show             # annotated sources instead of the summary
```

The build directory is `build/coverage` (`--build DIR` overrides it). The script configures it with
`-DTHEIA_COVERAGE=ON`, forwarding generator, compilers and toolchain from the cache of the main
build directory when that exists, and builds it. It runs `ctest -LE analysis` with
`LLVM_PROFILE_FILE` set to `coverage-profiles/%p.profraw`, merges the profiles with `llvm-profdata`
and reports with `llvm-cov` for every test binary inside the build directory. The LLVM tools are
looked up next to the compiler first, then on `PATH`; without `llvm-profdata` or `llvm-cov` the
script stops with exit code 77.

Output: the commands it runs, then the `llvm-cov` report (lines covered per file of `src/`).

Exit codes: 0 report produced, 1 test or coverage failure, 77 `llvm-profdata` or `llvm-cov` missing,
2 usage or configuration problem.

## Rendering

### `generate_gallery.ps1`

Renders the 30 scenes of the README gallery one after another.

```powershell
.\tools\generate_gallery.ps1 [-Renderer build\theia.exe] [-OutputDir screenshots] `
                             [-Frames 256] [-Width 1280] [-Height 720]
```

Each scene runs with `--scene <name> --offscreen-frames <Frames> --width <Width> --height <Height>
--output <OutputDir>\<name>.png`. The script prints the `Saved` or `ERROR` line of each render and
warns when a scene exits with a non-zero code. The gallery takes a long time: the renders saturate
the GPU and nothing else should run on it meanwhile.

## Qualification scripts

The four qualification scripts share a setup:

- The family is expected under `C:\Development\GitHub`, with `Theia/build/theia.exe`,
  `Hyperion/build/hyperion.exe` and `Harmonia/tools/compare_renders.py` built and present.
- The fixtures are expected under `Theia/build/sms-qualification/scenes/`: `qualify_sms.py` stages
  them there itself from `Harmonia/tests/assets/sms`; the other three read
  `build/sms-qualification/scenes/HYBRID/hybrid_k3.scene.toml` and
  `build/sms-qualification/scenes/HYBRID_K4/hybrid_k4.scene.toml` and expect them staged already.
- All renders run one after another in a single process, with validation enabled and
  `HARMONIA_FRAME_TIMING=1` set. A render whose log contains a validation message aborts the script.
- Results (EXRs, logs, `run_manifest.json`) go under `Theia/build/sms-qualification/`.
- "Different" always means a pixel difference (`max|difference| > 0`), never a file hash: EXR headers
  carry metadata, so pixel-identical files hash differently.
- Every image must be non-degenerate (maximum above 0 and a standard deviation above 0) before it is
  compared. The parity thresholds pass for two black images, so an all-black frame would otherwise
  read as a pass.

They need Python with NumPy and the modules that `Harmonia/tools/compare_renders.py` imports.

### `qualify_sms.py`

Validates specular manifold sampling against Hyperion on small fixtures.

```powershell
python tools/qualify_sms.py --stage smoke       --case C1
python tools/qualify_sms.py --stage reference    --case C1
python tools/qualify_sms.py --stage correctness  --case C1
python tools/qualify_sms.py --stage performance  --case C1
python tools/qualify_sms.py --stage stability    --case C1
```

`--stage` is required. `--case` selects a fixture and defaults to `C1`; the cases are `C1`, `C2`,
`C3`, `COMPLEX_CASE`, `DRAGON` and `HYBRID`. Each case names a scene and a region of interest in
normalised coordinates (the `CASES` table at the top of the script). The stages run in the order
above; every stage after `reference` needs the reference that `reference` stored:

| Stage | What it does | What passes |
|-------|--------------|-------------|
| `smoke` | Theia in `off`, `local` and `reuse` mode and Hyperion, 2 frames or samples each | Every EXR and PNG exists and is finite |
| `reference` | Hyperion with two seeds at 4096, 16384 and 65536 samples per pixel, until the seeds agree | Split-seed uncertainty of the region's energy ≤ 0.5 %; the final reference is then rendered and recorded |
| `correctness` | Theia in `local` and `reuse` mode at 512, 2048 and 8192 frames, four seeds each | At 8192 frames the pooled energy of the region is within ±2 % of the reference (512 and 2048 are recorded only), and the error falls from 512 to 2048 and from 2048 to 8192 frames |
| `performance` | Equal-time comparison of `off`, `local` and `reuse` at 2 s and 8 s, calibrated by pilot renders | `reuse` has a lower MSE than `local` at both targets, and every render stayed within 5 % of its time target |
| `stability` | Two `reuse` renders with the same seed, then one per seed | The two same-seed renders are pixel-identical; the per-seed energy spread is recorded |

Output: one log line per render with the pooled ratios, spreads and MSE, EXRs and logs under
`build/sms-qualification/`, and `run_manifest.json`, which records every accepted stage with its
provenance (repository heads, tracked diffs, untracked files, executable and shader hashes, and the
hashes of all fixture inputs). A later stage refuses a reference whose provenance no longer matches.
A failed check ends the script with a message and exit code 1.

### `test_qualify_sms.py`

CPU-only unit tests of `qualify_sms.py`; no renders. They cover the rejection of non-finite pixels and
mismatched shapes, the region-of-interest and reference validation, the frame-timing evidence, the
stale-provenance rejection, the removal of stale outputs, the pooling of the correctness and
performance stages, and the outward winding of the shared sphere casters.

```powershell
python -m unittest tools.test_qualify_sms
```

Output: the unittest summary. Exit code 0 when all tests pass.

### `qualify_gap_c.py`

Shows that the third-vertex reconnection shift runs on the `hybrid_k3` fixture, by rendering Theia
with several `THEIA_SHIFT_MASK` values, and compares the shifted result with Hyperion.

```powershell
python tools/qualify_gap_c.py [--out DIR] [--frames 256] [--spp 256] [--width 320] [--height 240]
                              [--seed 17] [--sms-mode local] [--skip-render]
```

The arms are `replay` (mask 0), `reconnection_k2` (mask 1), `hybrid_k3` (mask 3) and `hybrid_bit0`
(mask 2), plus one Hyperion reference. Checks:

| Check | Meaning |
|-------|---------|
| Content | Every frame is non-degenerate |
| A | `reconnection_k2` and `hybrid_k3` differ: the third-vertex shift body runs |
| B | `replay` and `reconnection_k2` differ: the mask reaches the estimator |
| C | `hybrid_k3` against Hyperion passes the family thresholds (`compare_renders.py --gate all`) |
| D | `hybrid_bit0` and `hybrid_k3` are identical: bit 0 is not read on its own |

Output: the pixel difference, the content statistics and the verdict per check. Default output
directory: `build/sms-qualification/gapc`. `--skip-render` re-evaluates the EXRs already on disk.

Check C is the mechanical family comparison. On this fixture its thresholds are out of reach for
every arm, Hyperion against itself included, so the script ends with exit code 1 on check C while
checks A, B and D hold. The recorded acceptance for the shift is the noise-envelope criterion that
`qualify_k4.py` and `exercise_bias_mode.py` apply.

Exit codes: 0 pass, 1 Hyperion comparison fails, 2 an input or render is missing, 3 the third-vertex
shift never runs, 4 the mask does not reach the estimator, 5 a degenerate frame.

### `qualify_k4.py`

Shows that the fourth-vertex reconnection shift runs and sits inside Hyperion's own noise on the
`hybrid_k4` fixture.

```powershell
python tools/qualify_k4.py [--out DIR] [--frames 256] [--spp 256] [--width 320] [--height 240]
                            [--seed 17] [--ref2-seed 18] [--sms-mode local] [--skip-render]
```

The arms are `replay` (mask 0), `reconnection_k2` (mask 1), `k4_fallback` (mask 3, where a
fourth-vertex record falls back to replay) and `k4` (mask 7), a Hyperion reference with `--seed` and
a second reference with `--ref2-seed`. The second reference measures the noise envelope in the same
run. Checks: content, B (mask reaches the estimator), A (`k4` and `k4_fallback` differ: the
fourth-vertex body runs), and C (`k4` and `k4_fallback` each stay within the envelope on all four
metrics: mean difference, relative MSE, tone-mapped mean and SSIM, with SSIM inverted).

Output: the four metrics for the envelope and every arm, the pixel differences and the verdict. The
mechanical family thresholds are printed as context; they fail for every arm on this fixture,
Hyperion against itself included. Default output directory: `build/sms-qualification/gapk4gate`.

Exit codes: 0 pass, 1 an arm is outside the envelope, 2 an input or render is missing, 3 the
fourth-vertex shift never runs, 4 the mask does not reach the estimator, 5 a degenerate frame.

### `exercise_bias_mode.py`

Checks the three levels of `THEIA_BIAS_CORRECTION` on the `hybrid_k3` fixture.

```powershell
python tools/exercise_bias_mode.py [--out DIR] [--frames 256] [--spp 256] [--width 320] [--height 240]
                                   [--seed 17] [--ref2-seed 18] [--sms-mode local] [--skip-render]
```

Renders: a Hyperion reference with `--seed` and a second with `--ref2-seed` (the noise envelope),
and Theia with the variable unset and set to 1, 2 and 0. Checks:

| Check | Meaning |
|-------|---------|
| A | Every Theia log contains its `THEIA_BIAS_CORRECTION = <n>` line: the variable was parsed |
| B | The default and level 1 are bit-identical |
| C | Level 2 differs from level 1: the visibility test rejects donors on this fixture |
| D | Level 2 stays within Hyperion's self-noise on all four metrics |

Output: the parse lines, the pixel differences and the metrics. Default output directory:
`build/sms-qualification/biasmode`.

Exit codes: 0 complete, 1 level 2 is outside the envelope, 2 an input or render is missing, 3 level 2
never differs from level 1, 4 the parse line is missing, 5 a degenerate frame, 6 the default and
level 1 are not identical.
