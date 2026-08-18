use cpu_rasteriser::prelude::*;

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[derive(Clone)]
struct Vertex {
    position: Vec3,
    colour: Vec4,
}

struct SceneVertexUniforms {
    view_matrix: Mat4,
    projection_matrix: Mat4,
}

#[derive(Interpolate)]
struct SceneVaryings {
    colour: Vec4,
}

struct SceneVertexShader;

impl VertexShader for SceneVertexShader {
    type Vertex = Vertex;
    type Uniforms = SceneVertexUniforms;
    type Varyings = SceneVaryings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let view_position = uniforms.view_matrix * vertex.position.to_point4();

        let clip_position = uniforms.projection_matrix * view_position;

        (
            clip_position,
            SceneVaryings {
                colour: vertex.colour,
            },
        )
    }
}

struct SceneFragmentShader;

impl FragmentShader<SceneVaryings> for SceneFragmentShader {
    type Uniforms = ();

    fn shade(&self, varyings: SceneVaryings, _uniforms: &Self::Uniforms) -> Colour {
        varyings.colour.into()
    }
}

struct CloudVertexUniforms {}

struct CloudVertexShader;

impl VertexShader for CloudVertexShader {
    type Vertex = Vertex;
    type Uniforms = CloudVertexUniforms;
    type Varyings = Vec2;

    fn shade(&self, vertex: Self::Vertex, _uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        (
            vertex.position.to_point4(),
            Vec2::new(vertex.position.x * 0.5 + 0.5, vertex.position.y * 0.5 + 0.5),
        )
    }
}

struct CloudFragmentUniforms {
    time: f32,
    resolution: Vec2,
}

struct CloudFragmentShader;

impl FragmentShader<Vec2> for CloudFragmentShader {
    type Uniforms = CloudFragmentUniforms;

    fn shade(&self, uv: Vec2, uniforms: &Self::Uniforms) -> Colour {
        let mut p = uv;

        p.x *= uniforms.resolution.x / uniforms.resolution.y;

        p.x += uniforms.time * 0.02;

        let n1 = noise(p * 2.0);
        let n2 = noise(p * 4.0);
        let n3 = noise(p * 8.0);

        let cloud = (n1 * 0.65 + n2 * 0.25 + n3 * 0.10).clamp(0.0, 1.0);

        let sky = Vec3::new(0.15, 0.35, 0.75);

        let cloud_colour = Vec3::new(0.95, 0.97, 1.0);

        let colour = sky * (1.0 - cloud) + cloud_colour * cloud;

        colour.into()
    }
}

fn noise(p: Vec2) -> f32 {
    let x = p.x.floor();
    let y = p.y.floor();

    let fx = p.x - x;
    let fy = p.y - y;

    let fade_x = fx * fx * (3.0 - 2.0 * fx);
    let fade_y = fy * fy * (3.0 - 2.0 * fy);

    let a = hash(x, y);
    let b = hash(x + 1.0, y);
    let c = hash(x, y + 1.0);
    let d = hash(x + 1.0, y + 1.0);

    let ab = a + (b - a) * fade_x;
    let cd = c + (d - c) * fade_x;

    ab + (cd - ab) * fade_y
}

fn hash(x: f32, y: f32) -> f32 {
    let v = (x * 127.1 + y * 311.7).sin() * 43_758.547;
    v.fract().abs()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Render Passes Demo - ESC to exit",
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
    let mut screen_target = RenderTarget::new(extent).with_depth();

    let cloud_pipeline = Pipeline::new(CloudVertexShader, CloudFragmentShader)
        .with_culling_mode(CullingMode::None)
        .with_depth_state(DepthState::DISABLED);

    let scene_pipeline = Pipeline::new(SceneVertexShader, SceneFragmentShader)
        .with_culling_mode(CullingMode::None)
        .with_depth_state(DepthState::READ_ONLY)
        .with_blend_state(BlendState::ALPHA_BLEND);

    let cloud_vertices = vec![
        Vertex {
            position: Vec3::new(-1.0, -1.0, 0.0),
            colour: Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: Vec3::new(3.0, -1.0, 0.0),
            colour: Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: Vec3::new(-1.0, 3.0, 0.0),
            colour: Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
    ];

    let cloud_indices = vec![0, 1, 2];

    let triangle = vec![
        Vertex {
            position: Vec3::new(-0.8, -0.6, 0.0),
            colour: Colour::from_u32(0xaaff2020).into(),
        },
        Vertex {
            position: Vec3::new(0.8, -0.6, 0.0),
            colour: Colour::from_u32(0xaa20ff20).into(),
        },
        Vertex {
            position: Vec3::new(0.0, 0.8, 0.0),
            colour: Colour::from_u32(0xaa2020ff).into(),
        },
    ];

    let triangle_indices = vec![0, 1, 2];

    let start_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let time = start_time.elapsed().as_secs_f32();

        // Pass 1: procedural cloud background
        let mut render_pass_1 = renderer.begin_render_pass(
            &mut screen_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: Some(LoadOp::Clear(1.0)),
            },
        );

        render_pass_1.draw(
            &cloud_pipeline,
            DrawCall::new(
                &cloud_vertices,
                &cloud_indices,
                PrimitiveMode::TRIANGLES,
                CloudFragmentUniforms {
                    time,
                    resolution: Vec2::new(WIDTH as f32, HEIGHT as f32),
                },
            ),
            CloudVertexUniforms {},
        );

        render_pass_1.finish();

        // Pass 2: foreground geometry
        let mut render_pass_2 = renderer.begin_render_pass(
            &mut screen_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Load,
                depth_load_op: Some(LoadOp::Load),
            },
        );

        let eye = Vec3::new(0.0, 0.0, 0.5);
        let look_at = Vec3::new(0.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 1.0, 0.0);

        let scene_uniforms = SceneVertexUniforms {
            view_matrix: Mat4::look_at(eye, look_at, up),
            projection_matrix: Mat4::perspective(60.0, WIDTH as f32 / HEIGHT as f32, 0.01, 50.0),
        };

        render_pass_2.draw(
            &scene_pipeline,
            DrawCall::new(&triangle, &triangle_indices, PrimitiveMode::TRIANGLES, ()),
            scene_uniforms,
        );

        render_pass_2.finish();

        window
            .update_with_buffer(&screen_target.pixels_u32(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
