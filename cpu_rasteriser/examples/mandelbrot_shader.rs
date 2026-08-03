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
}

struct VertexUniforms {
    pub model_matrix: Mat4,
}

#[derive(Interpolate)]
struct Varyings {
    pub world_position: Vec3,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = Vertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();

        let varyings = Varyings {
            world_position: world_position.homogenize_to_vec3(),
        };

        (world_position, varyings)
    }
}

struct FragmentUniforms {
    pixel_step: f32,
    centre: Vec2,
    max_iterations: u32,
}

fn length_squared(v: Vec2) -> f32 {
    v.x * v.x + v.y * v.y
}

fn iterate_mandelbrot(z: Vec2, c: Vec2) -> Vec2 {
    Vec2::new(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y)
}

fn mandelbrot_iterations(c: Vec2, max_iterations: u32) -> u32 {
    let mut z = Vec2::new(0.0, 0.0);
    let mut iterations = 0;

    while length_squared(z) <= 4.0 && iterations < max_iterations {
        z = iterate_mandelbrot(z, c);
        iterations += 1;
    }

    iterations
}

struct BasicFragmentShader;
impl FragmentShader<Varyings> for BasicFragmentShader {
    type Uniforms = FragmentUniforms;

    fn shade(&self, varyings: Varyings, uniforms: &Self::Uniforms) -> Colour {
        let aspect = WIDTH as f32 / HEIGHT as f32;

        let pixel_coord = varyings.world_position.xy();
        let pixel_coord = Vec2::new(pixel_coord.x * aspect, pixel_coord.y);

        let dc = pixel_coord * uniforms.pixel_step;
        let c = uniforms.centre + dc;

        let iterations = mandelbrot_iterations(c, uniforms.max_iterations);
        let intensity = iterations as f32 / uniforms.max_iterations as f32;

        Colour::from_f32(intensity, intensity, intensity, 1.0)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Mandelbrot Shader Demo - ESC to exit",
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

    let triangle = vec![
        Vertex {
            position: Vec3::new(-1.0, -1.0, 0.0),
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, 0.0),
        },
        Vertex {
            position: Vec3::new(0.0, 1.0, 0.0),
        },
    ];
    let triangle_indices = vec![0, 1, 2];

    let pixel_step = 1.0;
    let centre = Vec2::new(-0.5, 0.0);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let vertex_uniforms = VertexUniforms {
            model_matrix: Mat4::scaling_vec(Vec3::ONE * 3.0),
        };

        let mut frame = renderer.begin_frame(&viewport);

        frame.draw(
            &simple_pipeline,
            DrawCall::new(
                &triangle,
                &triangle_indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    pixel_step,
                    centre,
                    max_iterations: 500,
                },
            ),
            &vertex_uniforms,
        );

        frame.finish();

        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
