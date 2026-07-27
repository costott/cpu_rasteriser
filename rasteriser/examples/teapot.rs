use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{
        camera::{Camera, Projection},
        fragment_shader::FragmentShader,
        lighting::DirectionalLight,
        vertex_shader::VertexShader,
    },
    loaders::obj::load_obj,
    renderer::{CullingMode, Renderer},
};

mod common;
use common::camera_controller::OrbitControls;

use minifb::{Key, Window, WindowOptions};
use std::sync::Arc;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

struct VertexUniforms {
    pub model_matrix: Mat4,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}

#[derive(Interpolate)]
struct Varyings {
    pub world_position: Vec3,
    pub colour: Colour,
    pub normal: Vec3,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_homogenous();
        let normal_matrix = uniforms.model_matrix.inverse().transpose();

        let view_position = uniforms.view_matrix * world_position;
        let clip_position = uniforms.projection_matrix * view_position;

        let varyings = Varyings {
            world_position: world_position.homogenize_to_vec3(),
            colour: vertex.colour,
            normal: (normal_matrix * vertex.normal.to_homogenous())
                .homogenize_to_vec3()
                .normalise(),
        };

        (clip_position, varyings)
    }
}

struct FragmentUniforms;

struct BasicFragmentShader;
impl FragmentShader<Varyings> for BasicFragmentShader {
    type Uniforms = FragmentUniforms;

    fn shade(&self, varyings: Varyings, _uniforms: &Self::Uniforms) -> Colour {
        varyings.colour
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Teapot Demo - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    window.set_target_fps(60);

    let viewport = Viewport::new(WIDTH, HEIGHT);

    let mut renderer = Renderer::new(&viewport, BasicVertexShader, BasicFragmentShader)?;
    renderer.set_culling_mode(CullingMode::None);

    let mut camera = Camera::new(
        Vec3::new(0.0, 0.75, 1.25),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Projection::Perspective(
            cpu_rasteriser::graphics::camera::PerspectiveProjection::new(
                90.0,
                WIDTH as f32 / HEIGHT as f32,
                0.01,
                50.0,
            ),
        ),
    );
    let mut controls = OrbitControls::new(&camera);

    let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
    teapot.transform.scale = Vec3::ONE * 0.3;
    teapot.transform.rotation.y = 90_f32.to_radians();

    let mut scene = Scene::new(camera);
    scene.add_light(DirectionalLight::new(
        Vec3::new(0.0, -1.0, -1.0),
        Colour::from_u32(0xfffde8),
    ));
    scene.add_model(teapot);

    let mut previous_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let dt = std::time::Instant::now()
            .duration_since(previous_time)
            .as_secs_f32();
        previous_time = std::time::Instant::now();

        controls.update(&mut scene.camera, &window, dt);

        let vertex_uniforms = VertexUniforms {
            model_matrix: scene.models()[0].transform.model_matrix(),
            view_matrix: scene.camera.view_matrix(),
            projection_matrix: scene.camera.projection_matrix(),
        };

        renderer.draw_scene(
            &scene,
            &vertex_uniforms,
            Arc::new(FragmentUniforms),
            &viewport,
        );

        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
