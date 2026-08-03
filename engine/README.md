# Rust Software Graphics Engine

> A lightweight 3D engine built on top of the `cpu_rasteriser` crate.

The engine provides higher-level abstractions for building 3D applications while delegating all rendering to the underlying software renderer.

It includes concepts that intentionally do not exist in the renderer itself, such as cameras, models, materials, lighting, asset loading, and scene management.

## Features

- Cameras
- Transforms
- Meshes and models
- Materials
- Directional lighting
- OBJ loading
- Scene rendering utilities

## Architecture

```
Application
      │
      ▼
 Engine (Scene, Camera, Models, Lights)
      │
      ▼
 Renderer (Pipelines, Draw Calls, Rasterisation)
      │
      ▼
 Framebuffer
```

## Examples

```bash
cargo run -p engine --example teapot
cargo run -p engine --example pipelines
cargo run -p engine --examples textured_cube
```

The engine is designed as a thin layer over the renderer, allowing the renderer to remain reusable as a standalone graphics library.
