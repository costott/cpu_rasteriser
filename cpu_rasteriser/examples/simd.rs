use cpu_rasteriser::prelude::*;

use minifb::{Key, Window, WindowOptions};
use wide::f32x8;

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

#[derive(Interpolate, SimdInterpolate)]
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

struct SimdFragmentShader;
impl FragmentShaderSimd<Varyings> for SimdFragmentShader {
    type Uniforms = FragmentUniforms;

    #[inline(always)]
    fn shade_simd(&self, varyings: VaryingsSimd, _uniforms: &Self::Uniforms) -> ColourSimd {
        let r = varyings.colour[0];
        let g = varyings.colour[1];
        let b = varyings.colour[2];

        let r = (r * f32x8::splat(1.37) + g * f32x8::splat(0.21)).fast_max(f32x8::splat(0.0));

        let g = (g * f32x8::splat(0.91) + b * f32x8::splat(0.34)).fast_max(f32x8::splat(0.0));

        let b = (b * f32x8::splat(1.13) + r * f32x8::splat(0.17)).fast_max(f32x8::splat(0.0));

        ColourSimd {
            r,
            g,
            b,
            a: f32x8::splat(1.0),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "SIMD Demo - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    window.set_target_fps(60);

    let mut renderer = Renderer::new()?;

    let extent = Extent::new(WIDTH, HEIGHT);
    let mut render_target = RenderTarget::new(extent);

    let simd_pipeline = SimdPipeline::new(BasicVertexShader, SimdFragmentShader);

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

        let mut frame = renderer.begin_render_pass(
            &mut render_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        frame.draw_simd(
            &simd_pipeline,
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
            .update_with_buffer(&render_target.pixels_u32(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
