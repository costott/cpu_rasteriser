# Rust Software Rasteriser

> A reusable CPU implementation of a modern programmable graphics pipeline written in Rust.

The renderer provides a low-level rendering API inspired by modern graphics APIs such as Vulkan, Direct3D, and Metal. It exposes programmable shaders, pipelines, render passes, draw calls, and strongly typed shader interfaces while executing entirely on the CPU.

## Features

- Generic vertex and fragment shader pipelines
- Multiple render passes per frame
- Load and store operations for render targets
- Perspective-correct interpolation
- Homogeneous clipping
- Triangle rasterisation
- Depth buffering
- Backface culling
- Tile-based multithreaded rendering
- Strongly typed shader interfaces

## Examples

```bash
cargo run --release -p cpu_rasteriser --example triangle
cargo run --release -p cpu_rasteriser --example pipelines
cargo run --release -p cpu_rasteriser --example mandelbrot_shader
cargo run --release -p cpu_rasteriser --example mandelbulb
cargo run --release -p cpu_rasteriser --example winit
```

The higher-level engine built on top of this renderer can be found in the `engine` crate.
