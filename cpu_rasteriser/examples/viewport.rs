use cpu_rasteriser::prelude::*;

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[derive(Clone)]
struct Vertex {
    pub position: Vec3,
    pub colour: Vec4,
}

struct VertexUniforms {}

#[derive(Interpolate)]
struct Varyings {
    pub colour: Vec4,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = Vertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, _uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let clip_position = vertex.position.to_point4();

        let varyings = Varyings {
            colour: vertex.colour,
        };

        (clip_position, varyings)
    }
}

struct FragmentUniforms;

struct BasicFragmentShader;
impl FragmentShader<Varyings> for BasicFragmentShader {
    type Uniforms = FragmentUniforms;

    fn shade(&self, varyings: Varyings, _uniforms: &Self::Uniforms) -> Colour {
        varyings.colour.into()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Triangle Demo - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    window.set_target_fps(60);

    let extent = Extent::new(WIDTH, HEIGHT);

    let mut renderer = Renderer::new()?;

    let mut render_target = RenderTarget::new(extent);

    let simple_pipeline = Pipeline::new(BasicVertexShader, BasicFragmentShader);

    let triangle = vec![
        Vertex {
            position: Vec3::new(-1.0, -1.0, 0.0),
            colour: Colour::from_u32(0xff0000).into(),
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, 0.0),
            colour: Colour::from_u32(0x00ff00).into(),
        },
        Vertex {
            position: Vec3::new(0.0, 1.0, 0.0),
            colour: Colour::from_u32(0x0000ff).into(),
        },
    ];
    let triangle_indices = vec![0, 1, 2];

    let mut viewport = Viewport::new(0, 0, WIDTH / 5, HEIGHT / 5);

    let start_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let vertex_uniforms = VertexUniforms {};

        let elapsed = start_time.elapsed().as_secs_f32();

        viewport.width = ((WIDTH as f32 * (0.5 - 0.5 * (elapsed * 0.2).sin())) as usize).max(1);
        viewport.x =
            ((WIDTH - viewport.width) as f32 * (0.5 + 0.5 * (elapsed * 0.5).sin())) as usize;
        viewport.y =
            ((HEIGHT - viewport.height) as f32 * (0.5 + 0.5 * (elapsed * 0.5).cos())) as usize;

        let mut frame = renderer.begin_render_pass(
            &mut render_target,
            RenderPassDescriptor {
                viewport: viewport,
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        frame.draw(
            &simple_pipeline,
            DrawCall::new(
                &triangle,
                &triangle_indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms,
            ),
            vertex_uniforms,
        );

        frame.finish();

        window
            .update_with_buffer(render_target.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
