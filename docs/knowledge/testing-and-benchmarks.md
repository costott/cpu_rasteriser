---
sources:
  - cpu_rasteriser/Cargo.toml
  - engine/Cargo.toml
  - Cargo.toml
  - .cargo/config.toml
  - cpu_rasteriser/benches/rasterise_segment.rs
  - cpu_rasteriser/benches/raster_command.rs
  - engine/benches/texture_sampling.rs
  - engine/examples/bloom.rs
verified: ffed0e7
---

# Testing, benchmarks and build flags

## Build flags

`target-cpu=native` is set once, in the workspace-root `.cargo/config.toml`. Cargo searches for config files from the current working directory upwards, so the flag applies to every build in the workspace, whichever directory cargo runs from.

There is no custom `[profile.release]`. The only formatting configuration is rustfmt's defaults, and the code is currently `cargo fmt --check` clean.

## Tests

- **Location:** unit tests sit in `#[cfg(test)] mod tests` beside the code. Names are descriptive behaviour phrases, e.g. `rasterise_respects_bounds`, `process_triangle_culls_back_facing_triangles`.
- **Coverage:** concentrated on maths, geometry processing, triangle rasterisation, blending and SIMD/scalar equivalence. There are no integration tests that render a whole pass and compare images.
- **Fakes:** the `graphics/triangle.rs` unit tests and the benches write `SimdInterpolate` implementations by hand; the `renderer.rs` tests use the derive. See `simd.md`.
- **Doctests:** most doc examples are marked `ignore`. The one on `Renderer` isn't, and it doesn't compile ([#25](https://github.com/costott/cpu_rasteriser/issues/25)).
- **Baseline:** the current pass and fail baseline is in the project `CLAUDE.md`.

## Benchmarks (criterion 0.7, `harness = false`)

| Bench | Measures |
|---|---|
| `cpu_rasteriser/benches/rasterise_segment.rs` | Scanline rasterisation, scalar and SIMD, across triangle sizes, shapes and bounds: bare overhead and compute, fragment callbacks with depth writes and full shading, throughput, a scalar against SIMD shader comparison, and 8-pixel framebuffer stores |
| `cpu_rasteriser/benches/raster_command.rs` | Whole raster commands in groups `pre_simd`, `simd_scalar_shader` and `simd_shader` |
| `engine/benches/texture_sampling.rs` | Texture sampler variants. Some benchmark functions are defined but unused (clippy `dead_code`). |

**The benches contain their own copies of the raster command logic.** The renderer's `RasterCommand` types are private, so the benches re-implement them. Changing `TriangleRasterCommand(Simd)` in `renderer.rs` doesn't change what the benches measure. Keep the copies in sync deliberately, or say that they diverge. `pre_simd` is a frozen historical baseline. The copies' depth-buffer `expect` messages describe a missing buffer as a failure, because the benches build raster commands directly and skip the check that `draw` and `draw_simd` make.

## Measuring performance

1. Compare before and after on the same machine, in the same session, with the same build configuration. A comparison across a change to the build flags measures that change too.
2. Release builds only.
3. For whole-frame work, the bloom example (`engine/examples/bloom.rs`) is the established workload. Check that output is unchanged.
