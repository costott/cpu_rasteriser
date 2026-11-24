---
sources:
  - cpu_rasteriser/src/renderer.rs
  - cpu_rasteriser/src/lib.rs
  - cpu_rasteriser/src/colour.rs
  - cpu_rasteriser/src/framebuffer.rs
  - cpu_rasteriser/src/maths/mat4.rs
  - engine/src/components/camera.rs
  - engine/src/components/texture.rs
  - engine/src/engine.rs
  - engine/src/loaders/obj.rs
  - engine/src/loaders/mtl.rs
  - README.md
verified: b3b02d8
---

# Design decisions

Decisions whose reasoning isn't obvious from the current code. Each entry gives its evidence. **Inferred** marks reasoning that isn't stated anywhere.

- **Tiles own their buffers.** This avoids all locking during fragment processing, at the cost of load and merge copies. *Evidence:* README "Parallel Tile-Based Rendering".
  - Their buffers come from a pool that persists across passes, so a warm pass allocates none.
- **`RenderTarget` holds no sampling state.** A render target is its framebuffer and depth buffer; it offers allocation-free reads (`write_pixels`, `write_pixels_u32`) and nothing shaped around one use. Reusing texture storage across frames is a texture concern, so it is part of `engine`'s `Texture` (`from_render_target`, `update_from`): a texture whose pixels come from a render target, not a separate type. *Evidence:* the commit that introduced `RenderTarget::write_pixels`.
- **Raster commands are `Arc`-shared across tiles.** This avoids cloning triangles, shaders and uniforms into every overlapping tile. *Evidence:* `d1a643a` "Optimize tile command storage with Arc".
- **Draw commands are type-erased (`Box<dyn RenderPassCommand>`)** so one pass can contain draws from pipelines with different shader types, scalar and SIMD. *Evidence:* the rustdoc on `RenderPassCommand`; README "multiple pipelines per render pass".
- **Shaders are generic traits with associated types,** not runtime-configurable shading. This gives compile-time type safety between vertex output and fragment input. *Evidence:* README "Generic Shader Pipeline Architecture", which credits the `black` project as inspiration.
- **Rendering is organised as explicit render targets and passes.** Rendering commands don't share a frame's lifetime, which enables multipass and render-to-texture. *Evidence:* `5e069d7`; README "Pipeline and Render Pass Architecture".
- **Each draw owns its vertex uniforms,** and a queued draw call's lifetime is decoupled from the renderer (`cad6fe3`, `a8a78bb`). **Inferred:** this lets callers build uniforms per draw without keeping them borrowed for the whole pass.
- **Rasteriser and engine are separate crates.** This mirrors graphics API versus engine separation; the rasteriser has no scene or asset concepts. *Evidence:* `5c64d3f`; README "Architecture".
- **SIMD is a separate pipeline type, selected at compile time,** and `FragmentShaderSimd` is independent of `FragmentShader`. SIMD shaders don't have to provide a scalar version, and the API is explicit about which path runs. *Evidence:* README "SIMD Rendering"; trait docs.
- **The SIMD depth variants are const-generic,** which removes per-group depth branches. *Evidence:* `rasterise_impl<TEST_DEPTH, WRITE_DEPTH>`. **Inferred:** the motivation was performance; no record of a measurement exists.
- **The crate root is the public API.** All of `cpu_rasteriser`'s modules are private, and each public item is re-exported explicitly at the root, so the public surface is a deliberate list rather than a side effect of module visibility. The prelude repeats the list, minus `wide`, for glob import. The benches' internal types sit behind a default-off `internals` feature in a `#[doc(hidden)]` module, which keeps them out of the documented API. A feature was chosen over a `--cfg` flag because setting `RUSTFLAGS` overrides the workspace's `target-cpu=native`. *Evidence:* `cpu_rasteriser/src/lib.rs`; the `internals` feature in `cpu_rasteriser/Cargo.toml`.
- **Benchmarks come in two tiers.** Microbenchmarks isolate one piece of the render path, so a regression points at its cause. Frame benchmarks time whole passes through the public API, because tiling and threading show only there, and they check their output so that a workload can't silently stop drawing. The raster commands have no microbenchmark: it would have to time a copy of them, which drifts from production, or call the real ones, and exposing those would make an unsound call reachable, since `RasterCommand::rasterise` is a safe function whose SIMD implementation needs buffers exactly `bounds` in size ([#69](https://github.com/costott/cpu_rasteriser/issues/69)). The frame benchmarks time the real commands instead. *Evidence:* `cpu_rasteriser/benches/`.
- **Errors are for the environment, panics for bugs.** Failures a correct program can still meet (I/O, malformed files, the windowing system) return per-domain `thiserror` enums, `#[non_exhaustive]` and with causes chained through `source()`. A broken documented precondition panics where it is detected, on the calling thread, and is documented under `# Panics`. *Evidence:* `EngineError` (`engine/src/engine.rs`), `ObjError` and `MtlError` (`engine/src/loaders/`).
- **`wide` is re-exported from `cpu_rasteriser`** so that code generated by `#[derive(SimdInterpolate)]` doesn't force users to depend on `wide`. *Evidence:* `6fbf4d4`.
- **Colours are unclamped `f32` (HDR).** They are clamped only on conversion to `u32`, which allows bloom and similar effects. *Evidence:* `55bfb21` "add HDR floating-point colour support".
- **The camera is right-handed and the projection OpenGL-style.** *Evidence:* `dda13c9` "fix camera handedness"; `Mat4::perspective`.
