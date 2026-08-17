# Rust Software Rasteriser

> A reusable CPU implementation of a modern programmable graphics pipeline written in Rust.

The renderer provides a low-level rendering API inspired by modern graphics APIs such as Vulkan, Direct3D, and Metal. It exposes concepts including pipelines, shaders, draw calls, and frame recording.

## Features

- Generic vertex and fragment shader pipelines
- Perspective-correct interpolation
- Homogeneous clipping
- Triangle rasterisation
- Depth buffering
- Backface culling
- Tile-based multithreaded rendering
- Strongly typed shader interfaces

## Examples

```bash
cargo run -p cpu_rasteriser --example triangle
cargo run -p cpu_rasteriser --example pipelines
cargo run -p cpu_rasteriser --example mandelbrot_shader
cargo run -p cpu_rasteriser --example mandelbulb
cargo run -p cpu_rasteriser --example winit
```

The higher-level engine built on top of this renderer can be found in the `engine` crate.
