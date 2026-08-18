# Rust Software Graphics Engine

> A lightweight 3D engine built on top of the `cpu_rasteriser` crate.

The engine provides higher-level abstractions for building 3D applications while delegating all rendering to the underlying software renderer.

It introduces concepts that intentionally do not exist in the rasteriser itself, such as cameras, models, materials, lighting, asset loading, and scene management.

## Features

- Cameras and projections
- Transforms
- Meshes and models
- Materials and textures
- Directional lighting
- OBJ/MTL asset loading
- Scene management
- Scene rendering utilities
- Multiple render passes

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

The engine is deliberately kept separate from the low-level renderer. The engine provides the abstractions needed to construct a 3D scene, while `cpu_rasteriser` handles the actual graphics pipeline and rasterisation.

## Examples

```bash
cargo run --release -p engine --example teapot
cargo run --release -p engine --example resize
cargo run --release -p engine --example pipelines
cargo run --release -p engine --example textured_cube
cargo run --release -p engine --example render_passes
```

The engine is designed as a thin layer over the renderer, keeping the underlying `cpu_rasteriser` reusable as a standalone software graphics library.
