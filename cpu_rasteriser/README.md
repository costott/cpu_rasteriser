# cpu_rasteriser

> A programmable graphics pipeline that runs entirely on the CPU, in the style of a modern GPU API.

`cpu_rasteriser` is a low-level rendering library inspired by modern graphics APIs such as Vulkan, Direct3D, and Metal. You write vertex and fragment shaders as Rust types, combine them with fixed-function state into pipelines, and record draw calls into render passes that draw into explicit render targets. The renderer then clips, culls, and rasterises the geometry and shades it across all CPU cores. It doesn't open windows or load assets: read the finished pixels back and present them however you like, or use the `engine` crate, which does both.

## Features

- Strongly typed programmable shaders: implement `VertexShader` and `FragmentShader`, with your own vertex, uniform, and varying types
- Two pipelines: `Pipeline` shades one fragment at a time; `SimdPipeline` rasterises, depth tests, shades, and blends eight fragments at a time with a `FragmentShaderSimd`
- `#[derive(Interpolate)]` and `#[derive(SimdInterpolate)]` for varying structs, generating the scalar and eight-lane interpolation code
- Homogeneous clipping, back-face culling (`CullingMode`), and perspective-correct interpolation
- Depth testing and depth writes, configured per pipeline (`DepthState`)
- Colour blending with 5 operations and 10 factors (`BlendState`, `BlendOp`, `BlendFactor`), and `BlendState::ALPHA_BLEND` and `BlendState::ADDITIVE` presets
- Explicit render targets (`RenderTarget`), independent of any window, with an optional depth buffer
- Render passes (`RenderPass`) with a `LoadOp::Clear` or `LoadOp::Load` for colour and depth, so passes can be chained onto one target or render offscreen for a later pass to use
- Viewports (`Viewport`) for drawing into part of a target
- Multithreaded rendering that preserves draw order

## Getting started

Everything is exported at the crate root, and `use cpu_rasteriser::prelude::*;` imports it all at once. The crate also re-exports `wide`, whose `f32x8` type SIMD shaders use.

This renders a red triangle over the bottom-left half of a small offscreen target, then checks the pixels:

```rust
use cpu_rasteriser::prelude::*;

/// Passes clip-space positions straight through.
struct PassThrough;
impl VertexShader for PassThrough {
    type Vertex = Vec4;
    type Uniforms = ();
    type Varyings = f32;

    fn shade(&self, position: Vec4, _: &()) -> (Vec4, f32) {
        (position, 0.0)
    }
}

/// Fills every fragment with the draw call's colour.
struct FlatColour;
impl FragmentShader<f32> for FlatColour {
    type Uniforms = Colour;

    fn shade(&self, _: f32, colour: &Colour) -> Colour {
        *colour
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut renderer = Renderer::new()?;
    let extent = Extent::new(64, 64);
    let mut target = RenderTarget::new(extent);
    let pipeline = Pipeline::new(PassThrough, FlatColour);

    // Counter-clockwise in normalised device coordinates, so front-facing.
    let vertices = [
        Vec4::new(-1.0, -1.0, 0.0, 1.0),
        Vec4::new(1.0, -1.0, 0.0, 1.0),
        Vec4::new(-1.0, 1.0, 0.0, 1.0),
    ];

    let mut pass = renderer.begin_render_pass(
        &mut target,
        RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: None,
        },
    );
    pass.draw(
        &pipeline,
        DrawCall::new(&vertices, &[0, 1, 2], PrimitiveMode::TRIANGLES, Colour::RED),
        (),
    );
    pass.finish();

    let pixels = target.pixels();
    assert_eq!(pixels[63 * 64], Colour::RED); // bottom-left corner
    assert_eq!(pixels[63], Colour::BLACK); // top-right corner

    // One packed u32 per pixel, as `Colour::to_u32` describes, for presenting.
    let _presentable: Vec<u32> = target.pixels_u32();
    Ok(())
}
```

## Examples

Each example opens a window, with `minifb` unless stated otherwise:

```bash
cargo run --release -p cpu_rasteriser --example triangle
cargo run --release -p cpu_rasteriser --example viewport
cargo run --release -p cpu_rasteriser --example pipelines
cargo run --release -p cpu_rasteriser --example blending
cargo run --release -p cpu_rasteriser --example render_passes
cargo run --release -p cpu_rasteriser --example mandelbrot_shader
cargo run --release -p cpu_rasteriser --example mandelbulb
cargo run --release -p cpu_rasteriser --example winit
cargo run --release -p cpu_rasteriser --example simd
```

- `triangle`: a single triangle in one draw call.
- `viewport`: rendering into a sub-region of the target with `Viewport`.
- `pipelines`: several shader pipelines drawn in one render pass.
- `blending`: depth testing alongside alpha and additive blending, configured per pipeline with `DepthState` and `BlendState`.
- `render_passes`: multipass rendering, with a procedural cloud background pass followed by a foreground geometry pass loaded onto the same target with `LoadOp::Load`.
- `mandelbrot_shader`: the Mandelbrot set, rendered entirely in a fragment shader.
- `mandelbulb`: a ray-marched Mandelbulb fractal, shaded per fragment.
- `winit`: presenting through `winit` and `softbuffer` directly, without the `engine` crate.
- `simd`: a single triangle drawn with `SimdPipeline`, derived SIMD varyings, and a SIMD fragment shader.

## Relation to `engine`

`cpu_rasteriser` is the core of the workspace. The `engine` crate builds an application framework on top of it: windowing and input backends, cameras, models, materials, textures, and OBJ/MTL loading. Apps built with `engine` still render through this crate's pipelines and render passes, so everything here applies to them.
