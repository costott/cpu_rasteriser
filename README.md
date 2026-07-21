# Rust Software Rasteriser

>A fully custom 3D software rasteriser written in Rust, implementing a modern graphics pipeline from scratch without relying on OpenGL, Vulkan, or DirectX.

>The project explores the fundamentals of real-time rendering by building the core systems behind a traditional GPU pipeline, including vertex processing, clipping, rasterisation, interpolation, lighting, and materials.

![Rust](https://img.shields.io/badge/Rust-Programming%20Language-orange)

## Features

### Complete Rendering Pipeline

Implemented a full CPU-based rendering pipeline:

```
Model Data
    ↓
Vertex Processing
    ↓
World Transform
    ↓
View Transform
    ↓
Projection
    ↓
Clipping
    ↓
Rasterisation
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

# Lighting & Materials

The renderer implements a configurable lighting system supporting:

- Gouraud shading
- Phong shading
- Directional lights
- Material properties:
  - Ambient colour
  - Diffuse colour
  - Specular colour
  - Shininess

Lighting is calculated per-fragment using interpolated surface attributes.

---

# Rasterisation

The rasteriser includes custom implementations of:

- Triangle scanline rasterisation
- Bresenham line drawing
- Circle rasterisation
- Barycentric interpolation
- Perspective-correct interpolation

Perspective correction ensures attributes such as normals, colours, and depth values are correctly interpolated across projected triangles.

---

# Shader System

The renderer uses a programmable shader architecture inspired by modern graphics APIs.

## Vertex Shader

Responsible for processing world-space vertices before projection and clipping.

```rust
trait VertexShader {
    fn shade(&self, vertex: Vertex3D, uniforms: &VertexUniforms) -> Vertex3D;
}
```

## Fragment Shader

Responsible for calculating the final colour of each pixel.

```rust
trait FragmentShader {
    fn shade(&self, fragment: Fragment, uniforms: &FragmentUniforms) -> Option<Fragment>;
}
```

This design allows new rendering techniques to be added without changing the underlying pipeline.

---

# Supported Features

| Feature | Status |
|---|---|
| 3D transformations | ✅ |
| Perspective camera | ✅ |
| Orthographic camera | ✅ |
| Triangle rasterisation | ✅ |
| Depth buffering | ✅ |
| Backface culling | ✅ |
| Frustum clipping | ✅ |
| Indexed meshes | ✅ |
| Multiple materials | ✅ |
| Directional lighting | ✅ |
| Gouraud shading | ✅ |
| Phong shading | ✅ |
| Perspective-correct interpolation | ✅ |

---

# Technical Highlights

This project demonstrates experience with:

- Rust systems programming
- Linear algebra and 3D mathematics
- Graphics pipeline design
- Rendering algorithms
- CPU optimisation
- Concurrent programming
- Memory-safe multithreading
- Software architecture

---

# Future Improvements

Potential extensions:

- Texture mapping
- Normal mapping
- Shadow mapping
- Physically based rendering
- SIMD optimisation

---

# Built With

- **Language:** Rust
- **Rendering:** Custom CPU rasterisation pipeline
- **Math:** Custom vector and matrix mathematics

---

# Motivation

Graphics APIs hide much of the complexity involved in rendering. Building a rasteriser from scratch provides a deeper understanding of the algorithms and engineering decisions behind real-time graphics.

This project was built to explore the intersection of mathematics, computer graphics, and performance-focused systems programming.
