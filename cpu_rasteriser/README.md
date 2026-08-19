# Rust Software Rasteriser

> A reusable CPU implementation of a modern programmable graphics pipeline written in Rust.

The renderer provides a low-level rendering API inspired by modern graphics APIs such as Vulkan, Direct3D, and Metal. It exposes concepts including render targets, render passes, pipelines, shaders, draw calls, and load/store operations, while executing entirely on the CPU.

## Features

- Generic vertex and fragment shader pipelines
- Perspective-correct interpolation
- Homogeneous clipping
- Triangle rasterisation
- Depth buffering with configurable test/write state (`DepthState`)
- Configurable colour blending — 5 blend operations and 10 blend factors (`BlendState`, `BlendOp`, `BlendFactor`), with `ALPHA_BLEND` and `ADDITIVE` presets
- Backface culling
- Explicit render targets (`RenderTarget`), decoupled from the screen
- Render passes (`RenderPass`) with per-attachment `LoadOp::Load` / `LoadOp::Clear` behaviour for colour and depth
- Multipass rendering — chain passes onto the same target, or render into an offscreen target and sample it in a later pass
- Tile-based multithreaded rendering
- Strongly typed shader interfaces
- `#[derive(Interpolate)]` macro (via the `rasteriser_macros` crate) for perspective-correct interpolation of custom vertex/varying types

## Examples

```bash
cargo run --release -p cpu_rasteriser --example triangle
cargo run --release -p cpu_rasteriser --example viewport
cargo run --release -p cpu_rasteriser --example pipelines
cargo run --release -p cpu_rasteriser --example blending
cargo run --release -p cpu_rasteriser --example render_passes
cargo run --release -p cpu_rasteriser --example mandelbrot_shader
cargo run --release -p cpu_rasteriser --example mandelbulb
cargo run --release -p cpu_rasteriser --example winit
```

- `triangle` — minimal single-triangle draw call.
- `viewport` — rendering into a sub-region of the render target via `Viewport`.
- `pipelines` — multiple shader pipelines drawn within a single render pass.
- `blending` — depth testing alongside alpha and additive blending, configured per pipeline via `DepthState` and `BlendState`.
- `render_passes` — multipass rendering: a procedural cloud background pass followed by a foreground geometry pass loaded onto the same target with `LoadOp::Load`.
- `mandelbrot_shader` — a fullscreen Mandelbrot set rendered entirely in a fragment shader.
- `mandelbulb` — a ray-marched Mandelbulb fractal, shaded per-fragment.
- `winit` — running the renderer on the `winit` windowing backend directly (without the `engine` crate).

The higher-level engine built on top of this renderer can be found in the `engine` crate.
