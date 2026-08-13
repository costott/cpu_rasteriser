# Rust Software Graphics Engine

> A fully custom 3D rendering framework written in Rust, implementing a modern graphics pipeline from scratch without relying on OpenGL, Vulkan, or DirectX, or existing GPU APIs. All geometry processing, rasterisation, interpolation, and shading are implemented from scratch in Rust.

> The project explores the fundamentals of real-time rendering by building the core systems behind a traditional GPU pipeline, including vertex processing, clipping, rasterisation, interpolation, lighting, and materials.

![Rust](https://img.shields.io/badge/Rust-Programming%20Language-orange)

## Example

The same Utah teapot rendered using two independent shader pipelines within a single frame.

The left model uses Phong shading (lighting calculated per fragment), while the right model uses Gouraud shading (lighting calculated per vertex). Both share the same renderer, scene, and geometry; only the pipeline changes.

<p align="center">
    <img src="./assets/phong-vs-gouraud.gif" width="700">
</p>

Rendering with multiple pipelines requires only a few API calls:

```rust
let mut frame = renderer.begin_frame(&viewport);

for draw_call in gouraud_teapot.draw_calls(|_| GouraudFragmentUniforms) {
    frame.draw(&gouraud_pipeline, draw_call, gouraud_vertex_uniforms.clone());
}

for draw_call in phong_teapot.draw_calls(|mesh| PhongFragmentUniforms {
    scene: scene_uniforms.clone(),
    material: phong_teapot
        .materials
        .get(mesh.material_index.unwrap())
        .cloned(),
}) {
    frame.draw(&phong_pipeline, draw_call, phong_vertex_uniforms.clone());
}

frame.finish();
```

The complete example, including the shader implementations, camera setup, lighting, and model loading, can be found in [`engine/examples/pipelines.rs`](./engine/examples/pipelines.rs).

---

## Architecture

The project is organised as two independent crates:

```
software-graphics/
├── cpu_rasteriser/     # Low-level CPU rendering API
├── rasteriser_macros/  # Procedural macros
└── engine/             # High-level scene and asset management
```

The `renderer` crate provides a reusable software implementation of a modern graphics pipeline, exposing concepts such as pipelines, shaders, draw calls, and frame recording.

The `engine` crate builds on top of the renderer, providing higher-level abstractions including cameras, models, materials, lights, scene management, and asset loading.

This separation mirrors the design of modern graphics ecosystems, where rendering APIs remain independent of engine-level concepts.

--

## Features

### Complete Rendering Pipeline

Implemented a full CPU-based rendering pipeline:

```
Vertex Buffers
    ↓
Frame Submission
    ↓
Pipeline Selection
    ↓
Vertex Processing
    ↓
Clipping
    ↓
Primitive Assembly
    ↓
Tile Binning
    ↓
Parallel Rasterisation
    ↓
Fragment Shading
    ↓
Framebuffer
```

Supported features:

- Model, view, and projection transformations
- Perspective and orthographic cameras
- Homogeneous coordinate clipping
- Near-plane and frustum clipping
- Backface culling
- Depth buffering
- Indexed mesh rendering
- Programmable vertex and fragment shaders

---

## Rasterisation

The rasteriser includes custom implementations of:

- Triangle scanline rasterisation
- Bresenham line drawing
- Circle rasterisation
- Barycentric interpolation
- Perspective-correct interpolation

Perspective correction ensures attributes such as normals, colours, and depth values are correctly interpolated across projected triangles.

---

## Parallel Tile-Based Rendering

To improve rendering performance, the rasteriser uses a tile-based rendering pipeline executed across multiple CPU threads.

After vertex processing and clipping, triangles are converted into rasterisation commands and binned into fixed-size screen-space tiles. Each tile is then rasterised independently by a worker thread.

The tile system is independent of the active shader pipeline, allowing different vertex and fragment shader combinations to contribute rendering work to the same frame.

Once all worker threads have completed, the tile framebuffers are merged into the final image.

This approach provides:

- Parallel triangle rasterisation across CPU cores
- No locking during fragment processing
- Improved cache locality through tile-based rendering
- Memory-safe multithreading using Rust's ownership model

---

## Generic Shader Pipeline Architecture

The renderer uses a strongly typed, generic shader pipeline inspired by modern graphics APIs and projects such as [black](https://github.com/sinclairzx81/black).

Rather than hard-coding specific rendering techniques, the pipeline is parameterised over user-defined vertex and fragment shader implementations:

```rust
trait VertexShader {
    type Vertex;
    type Uniforms;
    type Varyings;

    fn shade(
        &self,
        vertex: Self::Vertex,
        uniforms: &Self::Uniforms,
    ) -> (Vec4, Self::Varyings);
}


trait FragmentShader<Varyings> {
    type Uniforms;

    fn shade(
        &self,
        varyings: Varyings,
        uniforms: &Self::Uniforms,
    ) -> Colour;
}
```

This allows different rendering techniques to share the same underlying pipeline while maintaining compile-time type safety.

The renderer does not need to know the details of a shader implementation. It only executes the generic pipeline stages and processes the resulting geometry and fragments.

---

## Pipeline and Frame Architecture

The renderer uses an explicit pipeline and frame submission model inspired by modern graphics APIs such as Vulkan, Direct3D, and Metal.

Rendering is separated into:

- Pipeline state
- Frame recording
- Draw commands
- Rasterisation

A pipeline contains programmable stages and fixed rendering configuration:

```rust
Pipeline<VertexShader, FragmentShader>
```

including:

- Vertex shader
- Fragment shader
- Culling mode
- Future pipeline state such as depth testing and blending

Frames act as temporary command buffers. Draw calls are recorded during a frame and executed when the frame is submitted.

This separation allows multiple pipelines to be used within the same frame while keeping the renderer independent from specific shader implementations.

---

## Supported Features

### Rasteriser

| Feature                           | Status |
| --------------------------------- | ------ |
| 3D transformations                | ✅     |
| Triangle rasterisation            | ✅     |
| Depth buffering                   | ✅     |
| Backface culling                  | ✅     |
| Frustum clipping                  | ✅     |
| Perspective-correct interpolation | ✅     |
| Tile-based rendering              | ✅     |
| Multithreaded rasterisation       | ✅     |
| Generic shader pipelines          | ✅     |
| Pipeline state abstraction        | ✅     |
| Frame-based command submission    | ✅     |
| Multiple pipelines per frame      | ✅     |

### Engine

| Feature              | Status |
| -------------------- | ------ |
| Perspective camera   | ✅     |
| Orthographic camera  | ✅     |
| Indexed meshes       | ✅     |
| Multiple materials   | ✅     |
| Directional lighting | ✅     |
| Gouraud shading      | ✅     |
| Phong shading        | ✅     |
| OBJ loading          | ✅     |
| Textures             | ✅     |

---

## Technical Highlights

This project demonstrates experience with:

- Rust systems programming
- Graphics pipeline architecture
- Linear algebra and 3D mathematics
- CPU rasterisation algorithms
- Programmable shader design
- Tile-based rendering
- Concurrent rendering using thread pools
- Memory-safe multithreading with Rust
- Performance optimisation and cache-aware rendering
- Trait-based extensible rendering architecture
- Rendering API architecture inspired by modern graphics APIs
- Command-based rendering architecture

---

## Future Improvements

Potential extensions:

- Normal mapping
- Shadow mapping
- Physically based rendering
- SIMD optimisation
- Render passes and framebuffer attachments
- Pipeline state objects with configurable blending and depth states

---

## Built With

- **Language:** Rust
- **Rendering:** Custom CPU rasterisation pipeline
- **Math:** Custom vector and matrix mathematics

---

## Motivation

Graphics APIs hide much of the complexity involved in rendering. Building a rasteriser from scratch provides a deeper understanding of the algorithms and engineering decisions behind real-time graphics.

This project was built to explore the intersection of mathematics, computer graphics, and performance-focused systems programming.
