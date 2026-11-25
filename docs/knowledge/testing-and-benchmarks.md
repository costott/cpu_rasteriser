---
sources:
  - cpu_rasteriser/Cargo.toml
  - engine/Cargo.toml
  - Cargo.toml
  - .cargo/config.toml
  - cpu_rasteriser/benches/micro/main.rs
  - cpu_rasteriser/benches/micro/fixtures.rs
  - cpu_rasteriser/benches/micro/binning.rs
  - cpu_rasteriser/benches/frame/main.rs
  - cpu_rasteriser/benches/frame/workloads.rs
  - cpu_rasteriser/benches/frame/check.rs
  - engine/benches/texture_sampling.rs
  - cpu_rasteriser/src/lib.rs
  - engine/src/lib.rs
  - rasteriser_macros/src/lib.rs
verified: b3b02d8
---

# Testing, benchmarks and build flags

## Build flags

`target-cpu=native` is set once, in the workspace-root `.cargo/config.toml`. Cargo searches for config files from the current working directory upwards, so the flag applies to every build in the workspace, whichever directory cargo runs from.

There is no custom `[profile.release]`. The only formatting configuration is rustfmt's defaults, and the code is currently `cargo fmt --check` clean.

## Tests

- **Location:** unit tests sit in `#[cfg(test)] mod tests` beside the code. Names are descriptive behaviour phrases, e.g. `rasterise_respects_bounds`, `process_triangle_culls_back_facing_triangles`.
- **Coverage:** concentrated on maths, geometry processing, triangle rasterisation, blending and SIMD/scalar equivalence. There are no integration tests that render a whole pass and compare images.
- **Fakes:** the `graphics/triangle.rs` unit tests write `SimdInterpolate` implementations by hand; the `renderer.rs` tests and the benches use the derive. See `simd.md`.
- **Doctests:** every public item's examples run as doctests; examples that open a window are `no_run`, and none are `ignore`d. The `cpu_rasteriser` and `engine` READMEs are their crate docs (`#![doc = include_str!("../README.md")]`), so their Rust examples are doctests too. Doctests run from the package directory, so `engine` examples reach the tracked assets as `../assets/...`.
- **Documentation lints:** each crate root warns on `missing_docs`, `clippy::missing_errors_doc` and `clippy::missing_panics_doc`, so a new public item without docs, or without the `# Errors` or `# Panics` section it needs, shows up in the build and in clippy.
- **Baseline:** the current pass and fail baseline is in the project `CLAUDE.md`.

## Benchmarks (criterion 0.7, `harness = false`)

There are two tiers in `cpu_rasteriser`, because they answer different questions. Microbenchmarks time one piece of the render path in isolation, so a regression points at its cause. Frame benchmarks time whole render passes, which includes the tiling and threading that no microbenchmark times, as well as the per-triangle index fetch, vertex copies and raster-command allocation, and per-draw costs, as the difference between `many_draws_opaque`, drawn with a draw call per cell, and `small_triangles_opaque`, the same frame in one draw call. They leave out reading the frame back, which the `output` microbenchmarks time. Both tiers pin components on synthetic workloads: the frame benchmarks pin the whole-pass machinery, not the cost of a realistic scene, and finding where a realistic scene spends its time is profiling's job.

| Target | Measures |
|---|---|
| `cpu_rasteriser/benches/micro/` (needs `internals`) | One module per area: `maths` (every operation of `Vec2`, `Vec3`, `Vec4`, `Mat4` and `Vec3Simd`), `colour` (`Colour` and `ColourSimd`), `interpolate`, `blend`, `geometry` (`process_triangle` and `clip_triangle`), `binning` (bounding boxes, `TileBinner`, `TilePool`, tile set-up and merge), `rasterise` (`rasterise_segment` and its SIMD form), `buffers` (every `FrameBuffer` and `DepthBuffer` operation the renderer calls, including the unchecked 8-wide accesses) and `output` (render-target read-back) |
| `cpu_rasteriser/benches/frame/` (public API only) | Whole passes on both `Pipeline` and `SimdPipeline`: a dense grid of small triangles and a few full-target layers, with and without depth and opaque and blended, at three resolutions, the grid drawn with one draw call per cell, and a two-pass chain with `LoadOp::Load` |
| `engine/benches/texture_sampling.rs` | Texture sampler variants, scalar and SIMD |

Run one with `cargo bench -p cpu_rasteriser --features internals --bench micro` or `cargo bench -p cpu_rasteriser --bench frame`, and select cases with a Criterion filter after `--`. Test mode (`cargo test -p cpu_rasteriser --features internals --bench micro`, or `--bench frame`) runs each case once; the frame output checks run there as well as before timing under `cargo bench`. `cargo test -p cpu_rasteriser --benches` also runs the library's unit tests, so the baseline failures stop it before the benches.

**Conventions of the microbenchmarks.**
- **Names:** the group is the area and the id is `<case>`, or `<case>/scalar` and `<case>/simd` where both implementations do the same work on the same data.
- **Batches:** each Criterion iteration processes a batch (`fixtures::BATCH` items, or a whole tile, triangle or buffer), because single operations of 1–10 ns varied by up to 2× between identical runs. Inputs are built outside `iter`, deterministic and representative, and results are consumed with `black_box`. `BATCH` and the other sizes are provisional; see [#35](https://github.com/costott/cpu_rasteriser/issues/35).
- **Production code only:** they call the real types and functions. Stand-ins replace only user code and private types: pass-through vertex and fragment shaders, and `BoundsCommand` in `binning.rs`, a `RasterCommand` that supplies a bounding box, because the real commands are private and binning never calls `rasterise`. The raster commands themselves have no microbenchmark: their parts do, and the frame benchmarks time the whole command.
- **Coverage:** every operation that computes is covered, scalar and SIMD alike. Pure field copies, operations that run exactly the same code as a covered one, and crate-internal operations that nothing in the crate calls are skipped, and each module's doc comment says which. Operators that look alike (`Add` and `Sub`, `Mul<f32>` and `Div<f32>`) are separate cases, because each compiles to its own code and guards its own code generation. Variants that differ only in which operand the same code reads (blend factors that select the source or the destination colour) share one case.

**Frame benchmarks check their output.** Each case renders once before timing and asserts the covered-pixel count and the mean colour (`check.rs`), so a workload that culls everything can't pass as fast. The expected values are derived from how each workload is built (layers of flat-coloured cells that tile the target) and never recorded from a run; correctness itself belongs to the unit tests. The `large_triangles` and `multipass` layers are drawn out of depth order, so depth testing changes the result there; `small_triangles` is a single layer, so its checks can't tell whether depth testing ran. The checks guard the workload (coverage and overall colour), not every pixel. The mean-colour tolerance is provisional. Only the colour channels are compared, because the scalar and SIMD blends compute alpha differently ([#18](https://github.com/costott/cpu_rasteriser/issues/18)), and the SIMD colours sit slightly below the exact mean (see `simd.md`).

**The microbenchmarks need the `internals` feature.** They use internal types from the `#[doc(hidden)]` `internals` module (see `lib.rs` for the list) and declare `required-features = ["internals"]`, so cargo skips them, clippy included, unless the feature is on. Beyond the re-exports, the feature compiles in three constructors and accessors the benches use (`Triangle2D::new`, which the unit tests also use, and `Renderer::tile_pool_mut` and `Renderer::merge_tile_result`, which are `#[doc(hidden)]`) and lifts the `expect(dead_code)` on items the re-exports make reachable, so it doesn't change the measured code. The frame benchmarks need no feature.

## Measuring performance

1. Before a timing run, ask the user to plug in the charger and close other programs, because battery power and background load change the timings.
2. Compare before and after on the same machine, in the same session, with the same build configuration. A comparison across a change to the build flags measures that change too.
3. Release builds only.
4. For whole-frame work, the `frame` benchmarks are the established workload. Their output check catches a workload that stops drawing or changes its overall colour, not a change confined to individual pixels, so compare the output by other means when a change could move pixels. Take a microbenchmark comparison for the piece you changed as well.
