# Rust Software Graphics Engine

> A lightweight 3D engine built on top of the `cpu_rasteriser` crate.

The engine provides higher-level abstractions for building 3D applications while delegating all rendering to the underlying software renderer.

It introduces concepts that intentionally do not exist in the rasteriser itself, such as cameras, models, materials, lighting, asset loading, and scene management.

## Features

- Cross-backend application framework (`Application` trait) with two interchangeable windowing backends — `WinitEngine` and `MinifbEngine` — behind a common `EngineBackend` trait
- Unified input handling (keyboard and mouse) normalised across both backends
- Perspective and orthographic cameras
- Built-in `OrbitControls` and `FirstPersonControls` camera controllers
- Transforms
- Meshes and models, with OBJ / MTL loading
- Materials (ambient/diffuse/specular/shininess, with texture slots)
- Textures, with configurable wrap mode (repeat/clamp) and filter mode (nearest/bilinear)
- Directional lighting
- Render-to-texture: any `RenderTarget` can be sampled as a texture in a later pass via `render_target_sampler`
- Multipass rendering and post-processing pipelines (chromatic aberration, vignette, greyscale, invert, Gaussian blur, Sobel edge detection are demonstrated as examples)
- Automatic window-resize handling

## Architecture

```text
Application
      │
      ▼
 Engine
 (Scene, Camera, Models, Materials, Lights)
      │
      ▼
 CPU Rasteriser
 (Pipelines, Render Passes, Draw Calls)
      │
      ▼
 Render Targets
```

An `Application` implementation receives a `RenderContext` each frame, which can begin a render pass directly onto the presentation target (`begin_presentation_pass`) or onto an offscreen `RenderTarget` (`begin_render_pass`) — the same mechanism used to build multipass and post-processing effects.

## Examples

```bash
cargo run --release -p engine --example teapot
cargo run --release -p engine --example resize
cargo run --release -p engine --example pipelines
cargo run --release -p engine --example textured_cube
cargo run --release -p engine --example render_passes
```

- `teapot` — minimal scene: load an OBJ model and render it with a basic shader.
- `resize` — window resizing and render target reconfiguration.
- `pipelines` — several shader pipelines (Gouraud, Phong, alpha-blended, additive-blended) drawn together with orbit camera controls.
- `textured_cube` — OBJ/MTL loading with textured materials.
- `render_passes` — multipass rendering: clouds and a lit teapot composited into an offscreen render target, then run through a swappable post-processing pass (press SPACE to cycle effects).

The engine is designed as a thin layer over the renderer, keeping the underlying `cpu_rasteriser` reusable as a standalone software graphics library.
