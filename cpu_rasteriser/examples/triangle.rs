use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
    renderer::{CullingMode, DrawCall, Pipeline, PrimitiveMode, Renderer},
};

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[derive(Clone)]
struct Vertex {
    pub position: Vec3,
    pub colour: Vec4,
}

struct VertexUniforms {
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}

#[derive(Interpolate)]
struct Varyings {
    pub colour: Vec4,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = Vertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let view_position = uniforms.view_matrix * vertex.position.to_point4();
        let clip_position = uniforms.projection_matrix * view_position;

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

    let viewport = Viewport::new(WIDTH, HEIGHT);

    let mut renderer = Renderer::new(&viewport)?;

    let simple_pipeline =
        Pipeline::new(BasicVertexShader, BasicFragmentShader).with_culling_mode(CullingMode::None);

    let eye = Vec3::new(0.0, 0.0, 1.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

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

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let vertex_uniforms = VertexUniforms {
            view_matrix: Mat4::look_at(eye, look_at, up),
            projection_matrix: Mat4::perspective(90.0, WIDTH as f32 / HEIGHT as f32, 0.01, 50.0),
        };

        let mut frame = renderer.begin_frame(&viewport);

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
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
