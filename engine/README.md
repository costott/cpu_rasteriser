# engine

> A lightweight 3D application framework built on the `cpu_rasteriser` software renderer.

`engine` adds what a 3D app needs around the renderer: a window and frame loop, input, cameras, models, materials, textures, and asset loading. All rendering is still done by `cpu_rasteriser`'s pipelines and render passes, which apps use directly; the engine ships no shaders of its own.

## Features

- An application framework: implement the `Application` trait and run it on either of two interchangeable windowing backends, `WinitEngine` and `MinifbEngine`, behind the common `EngineBackend` trait
- Keyboard and mouse input normalised across backends (`InputEvent`, `InputState`)
- Automatic window-resize handling, and control of cursor visibility and grab (`WindowState`)
- Perspective and orthographic cameras (`Camera`, `Projection`), with built-in `OrbitControls` and `FirstPersonControls` controllers
- Meshes and models (`Mesh`, `Model`, `ModelTransform`), with cube and sphere primitives
- OBJ and MTL loading (`load_obj`, `load_mtl`)
- Materials with ambient, diffuse, and specular colours, shininess, and texture slots (`Material`)
- Textures with repeat or clamp wrapping and nearest or bilinear filtering (`Texture`, `TextureSampler`, `WrapMode`, `FilterMode`), sampled in scalar or SIMD shaders
- Directional lights (`DirectionalLight`) to pass to shaders as uniforms
- Render-to-texture: `Texture::from_render_target` copies any `RenderTarget` into a texture to sample in a later pass, and `Texture::update_from` refills it each frame, reusing its storage once earlier samplers are dropped
- Multipass rendering and post-processing, demonstrated in the examples with chromatic aberration, vignette, greyscale, invert, Gaussian blur, Sobel edge detection, and bloom
- Typed errors with chained causes: `EngineError` from the backends, `ObjError` and `MtlError` with line numbers from the loaders, and `TextureError` and `ModelError`

## Architecture

```text
Application
      │
      ▼
 Engine
 (Backends, Input, Cameras, Models, Materials, Textures, Lights)
      │
      ▼
 CPU Rasteriser
 (Pipelines, Render Passes, Draw Calls)
      │
      ▼
 Render Targets
```

Each frame, the backend calls the app's `update` with the elapsed time, then its `render` with a `RenderContext`. The context begins render passes onto the presentation target, which the window shows (`begin_presentation_pass`), or onto an offscreen `RenderTarget` (`begin_render_pass`). Rendering offscreen and sampling the result in a later pass is how multipass and post-processing effects are built. `render` must return the `PresentedFrame` that only finishing the presentation pass produces, so a frame can't be left unpresented.

## Getting started

`use engine::prelude::*;` imports the engine's items. Shaders, pipelines, and render passes come from `use cpu_rasteriser::prelude::*;`.

This app draws a cube that the arrow keys or WASD orbit around, and exits on Escape:

```rust,no_run
use cpu_rasteriser::prelude::*;
use engine::prelude::*;

#[derive(Clone)]
struct VertexUniforms {
    model_view_projection: Mat4,
}

#[derive(Interpolate)]
struct Varyings {
    colour: Vec4,
}

struct ColourVertexShader;
impl VertexShader for ColourVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: ObjVertex, uniforms: &VertexUniforms) -> (Vec4, Varyings) {
        let position = uniforms.model_view_projection * vertex.position.to_point4();
        (position, Varyings { colour: vertex.colour.into() })
    }
}

struct ColourFragmentShader;
impl FragmentShader<Varyings> for ColourFragmentShader {
    type Uniforms = ();

    fn shade(&self, varyings: Varyings, _uniforms: &()) -> Colour {
        varyings.colour.into()
    }
}

struct CubeApp {
    camera: Camera,
    controls: OrbitControls,
    cube: Model<ObjVertex>,
    pipeline: Pipeline<ColourVertexShader, ColourFragmentShader>,
}

impl Application for CubeApp {
    fn update(&mut self, dt: f32) {
        self.controls.update_from_events(&mut self.camera, dt);
    }

    fn render<'frame>(&mut self, context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        let extent = context.presentation_target().extent();
        let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: Some(LoadOp::Clear(1.0)),
        });

        let uniforms = VertexUniforms {
            model_view_projection: self.camera.projection_matrix()
                * self.camera.view_matrix()
                * self.cube.transform.model_matrix(),
        };
        self.cube
            .draw_to_render_pass(&mut pass, &self.pipeline, uniforms, |_| ());

        pass.finish()
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect_ratio(width as f32 / height as f32);
    }

    fn event(&mut self, event: AppEvent, handle: &mut AppHandle) {
        self.controls.handle_event(event);

        if let AppEvent::Input(InputEvent::Key {
            key: InputKey::Escape,
            state: ButtonState::Pressed,
        }) = event
        {
            handle.request_exit();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = Camera::new(
        Vec3::new(2.0, 1.0, 2.0),
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Projection::Perspective(PerspectiveProjection::new(
            70.0_f32.to_radians(),
            16.0 / 9.0,
            0.1,
            100.0,
        )),
    );
    let controls = OrbitControls::new(&camera);
    let cube = Model::new(
        vec![Mesh::cube(Colour::new(1.0, 0.5, 0.0, 1.0), None)],
        Vec::new(),
        ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
    )?;
    let pipeline = Pipeline::new(ColourVertexShader, ColourFragmentShader);

    MinifbEngine::new()
        .with_title("Cube")
        .with_size(640, 360)
        .run(CubeApp { camera, controls, cube, pipeline })?;
    Ok(())
}
```

## Examples

Run these from the workspace root, where the examples find their assets:

```bash
cargo run --release -p engine --example teapot
cargo run --release -p engine --example resize
cargo run --release -p engine --example pipelines
cargo run --release -p engine --example textured_cube
cargo run --release -p engine --example render_passes
cargo run --release -p engine --example bloom
cargo run --release -p engine --example silicon_01
```

- `teapot`: the minimal scene. Loads an OBJ model and renders it with a basic shader and orbit camera controls.
- `resize`: window resizing and render-target reconfiguration.
- `pipelines`: several shader pipelines (Gouraud, Phong, alpha-blended, and additive-blended) drawn in one pass, with orbit camera controls.
- `textured_cube`: OBJ and MTL loading with textured materials.
- `render_passes`: multipass rendering. Clouds and a lit teapot are composited into an offscreen render target, then run through a swappable post-processing pass (press Space to cycle effects).
- `bloom`: SIMD multipass bloom. Renders a glass teapot, extracts its bright areas, applies a Gaussian blur, and composites the blurred result back onto the scene.
- `silicon_01`: SILICON 01: FIRST LIGHT, a looping trailer-style short and the first of the `silicon` showcase series. A procedural CPU with a floorplanned die, HDR bloom and captions set from a signed-distance-field font, which render in as the screen's tiles do (traced in lines, shaded, then inked) and cut away; the Utah teapot assembling itself from triangles streaming out of the die; a warping, ray-marched mandelbulb hologram; a montage of 56 worlds, each with its own scene, shadow maps or screen-space stylisation; and the screen's tiles powering on and off, as the renderer's threads take them. Space pauses the director and lets you orbit the camera, Left and Right skip between acts, and Escape quits. Flags, after `--`:
  - `--window <width>x<height>` sets the window's size (default 1280x720).
  - `--scale <factor>` renders the scene at that fraction of the window's resolution (default 0.6, for smooth playback). Above 1 supersamples.
  - `--from <seconds>` starts at that time.
  - `--record <dir> [--fps <n>] [--to <seconds>]` renders every frame at a fixed time step to numbered PNGs, for encoding as a GIF or video, then exits. Frames are saved at the window's size. Frame time doesn't matter, so `--scale 2` gives the best quality.
  - `--shots <dir> --at <t1,t2,…>` saves stills at the listed times, then exits.
  - `--theme <name>` shows only that world whenever the montage is on screen.
  - `--no-captions` leaves out every caption.
  - `--soundtrack <file.wav> [--from <seconds>] [--to <seconds>]` renders the soundtrack to a WAV file without opening a window, then exits.

  The film has an original score and sound effects, timed by the code that times the picture. They are heard in the rendered video; the window plays silently. The recordings they use are credited in [`assets/silicon_01/audio/CREDITS.md`](../assets/silicon_01/audio/CREDITS.md).

  `engine/examples/silicon_01/media.sh` records the whole loop as a video with its soundtrack (`media.sh video <out.mp4> [fps] [size]`, 1080p by default) or the README's silent GIF (`media.sh gif <out.gif> [fps] [width]`) at full quality and encodes it with ffmpeg. `media.sh audio <video.mp4> <out.mp4>` renders the soundtrack afresh onto an existing video of the whole loop without recording it again. Run it from the workspace root.

## Relationship to `cpu_rasteriser`

The engine is a thin layer over the renderer. `cpu_rasteriser` knows nothing of windows, cameras, models, or files: it renders draw calls into render targets. Keeping those concepts in `engine` leaves `cpu_rasteriser` usable on its own as a standalone software graphics library, while apps built on the engine still write their shaders and pipelines against its API directly.
