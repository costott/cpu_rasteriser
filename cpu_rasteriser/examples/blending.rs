//! Demonstrates depth testing and alpha/additive blending using per-pipeline
//! `DepthState` and `BlendState` configuration.
//!
//! The window contains three demonstrations arranged from left to right:
//!
//! - Left:   Two opaque triangles at different depths demonstrate depth testing.
//! - Centre: Two translucent triangles demonstrate alpha blending.
//! - Right:  Two additive triangles demonstrate additive blending.

use cpu_rasteriser::prelude::*;

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[derive(Clone)]
struct Vertex {
    pub position: Vec3,
}

struct VertexUniforms {
    pub model_matrix: Mat4,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}

#[derive(Interpolate)]
struct Varyings {}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = Vertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();
        let view_position = uniforms.view_matrix * world_position;
        let clip_position = uniforms.projection_matrix * view_position;

        (clip_position, Varyings {})
    }
}

struct FragmentUniforms {
    pub colour: Colour,
}

struct SolidColourFragmentShader;
impl FragmentShader<Varyings> for SolidColourFragmentShader {
    type Uniforms = FragmentUniforms;

    fn shade(&self, _varyings: Varyings, uniforms: &Self::Uniforms) -> Colour {
        uniforms.colour
    }
}

/// A small triangle mesh centred on the origin
fn triangle_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let vertices = vec![
        Vertex {
            position: Vec3::new(-0.35, -0.3, 0.0),
        },
        Vertex {
            position: Vec3::new(0.35, -0.3, 0.0),
        },
        Vertex {
            position: Vec3::new(0.0, 0.35, 0.0),
        },
    ];
    (vertices, vec![0, 1, 2])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Blending & Depth Testing Demo - ESC to exit",
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

    // Standard opaque pipeline: depth test AND write enabled, blending off.
    // This matches `Pipeline::new`'s defaults; it's spelled out explicitly
    // here for contrast with the pipelines below.
    let opaque_pipeline = Pipeline::new(BasicVertexShader, SolidColourFragmentShader)
        .with_culling_mode(CullingMode::None)
        .with_depth_state(DepthState::DEFAULT)
        .without_blend_state();

    // Alpha-blended pipeline: still depth-*tests* against existing opaque
    // geometry, but doesn't write depth, so multiple translucent triangles
    // composite correctly as long as they're submitted back-to-front.
    let alpha_pipeline = Pipeline::new(BasicVertexShader, SolidColourFragmentShader)
        .with_culling_mode(CullingMode::None)
        .with_depth_state(DepthState::READ_ONLY)
        .with_blend_state(BlendState::ALPHA_BLEND);

    // Additive pipeline: same depth behaviour as alpha blending, but colours
    // accumulate (src + dst) rather than interpolate, producing a glow where
    // triangles overlap.
    let additive_pipeline = Pipeline::new(BasicVertexShader, SolidColourFragmentShader)
        .with_culling_mode(CullingMode::None)
        .with_depth_state(DepthState::READ_ONLY)
        .with_blend_state(BlendState::ADDITIVE);

    let (vertices, indices) = triangle_mesh();

    let eye = Vec3::new(0.0, 0.0, 1.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

    let view_matrix = Mat4::look_at(eye, look_at, up);

    // Orthographic projection is preferable here because this example is
    // demonstrating depth and blending state, not perspective.
    // Objects at different depths therefore remain the same size.
    let aspect_ratio = WIDTH as f32 / HEIGHT as f32;
    let half_height = 1.8;
    let half_width = half_height * aspect_ratio;

    let projection_matrix = Mat4::orthographic(
        -half_width,
        half_width,
        -half_height,
        half_height,
        0.01,
        50.0,
    );

    // Builds per-draw vertex uniforms for a triangle centred at `(x, y, z)`
    // in world space. Smaller `z` values are farther from the camera (the
    // eye sits at z = 1.0, looking toward the origin).
    let vertex_uniforms_at = |x: f32, y: f32, z: f32| VertexUniforms {
        model_matrix: Mat4::translation(x, y, z) * Mat4::scaling(1.5, 1.5, 1.5),
        view_matrix,
        projection_matrix,
    };

    let start_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let elapsed = start_time.elapsed().as_secs_f32();

        let background_colour = Colour::from_f32(
            0.3 + 0.3 * (elapsed * 0.5).sin(),
            0.3 + 0.3 * (elapsed * 0.3).cos(),
            0.3 + 0.3 * (elapsed * 0.7).sin(),
            1.0,
        );

        let mut frame = renderer.begin_render_pass(
            &mut screen_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(background_colour),
                depth_load_op: Some(LoadOp::Clear(1.0)),
            },
        );

        // --- Left: depth testing -------------------------------------------
        // Both triangles have exactly the same size and almost completely
        // overlap. The blue triangle is closer to the camera but submitted
        // first. The red triangle is farther away and therefore fails the
        // depth test wherever the blue triangle is already present.

        frame.draw(
            &opaque_pipeline,
            DrawCall::new(
                &vertices,
                &indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    colour: Colour::from_u32(0x3399ff),
                },
            ),
            vertex_uniforms_at(-1.65, 0.0, 0.3),
        );

        frame.draw(
            &opaque_pipeline,
            DrawCall::new(
                &vertices,
                &indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    colour: Colour::from_u32(0xff3333),
                },
            ),
            vertex_uniforms_at(-1.50, 0.0, -0.6),
        );

        // --- Centre: alpha blending ---------------------------------------
        // The backdrop is opaque. The two translucent triangles are drawn
        // back-to-front. Because alpha blending reads but does not write
        // depth, both triangles can contribute to the overlapping region.

        frame.draw(
            &opaque_pipeline,
            DrawCall::new(
                &vertices,
                &indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    colour: Colour::from_u32(0x303030),
                },
            ),
            vertex_uniforms_at(0.0, 0.0, -0.6),
        );

        frame.draw(
            &alpha_pipeline,
            DrawCall::new(
                &vertices,
                &indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    colour: Colour::from_f32(1.0, 0.2, 0.2, 0.5),
                },
            ),
            vertex_uniforms_at(-0.15, 0.0, 0.2),
        );

        frame.draw(
            &alpha_pipeline,
            DrawCall::new(
                &vertices,
                &indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    colour: Colour::from_f32(0.2, 1.0, 0.2, 0.5),
                },
            ),
            vertex_uniforms_at(0.15, 0.0, 0.4),
        );

        // --- Right: additive blending -------------------------------------
        // The triangles overlap and their RGB values are added together.
        // The overlap therefore becomes substantially brighter.

        frame.draw(
            &additive_pipeline,
            DrawCall::new(
                &vertices,
                &indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    colour: Colour::from_f32(1.0, 0.25, 0.0, 0.6),
                },
            ),
            vertex_uniforms_at(1.50, 0.0, 0.2),
        );

        frame.draw(
            &additive_pipeline,
            DrawCall::new(
                &vertices,
                &indices,
                PrimitiveMode::TRIANGLES,
                FragmentUniforms {
                    colour: Colour::from_f32(0.0, 0.5, 1.0, 0.6),
                },
            ),
            vertex_uniforms_at(1.70, 0.0, 0.4),
        );

        frame.finish();

        window
            .update_with_buffer(screen_target.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
