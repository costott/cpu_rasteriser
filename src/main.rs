use minifb::{Key, Window, WindowOptions};

pub mod colour;
pub mod depthbuffer;
pub mod framebuffer;
pub mod graphics;
pub mod maths;
pub mod renderer;
use renderer::Renderer;
pub mod viewport;
use viewport::Viewport;

pub mod prelude;
use prelude::*;

use crate::graphics::{
    camera::{Camera, Projection},
    fragment_shader::{BasicFragmentShader, PhongFragmentShader},
    lighting::DirectionalLight,
    vertex_shader::{BasicVertexShader, GouraudVertexShader},
};
use crate::renderer::CullingMode;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn main() {
    let mut window = Window::new(
        "CPU rasteriser - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    window.set_target_fps(60);

    let viewport = Viewport::new(WIDTH, HEIGHT);

    let mut renderer = Renderer::new(
        &viewport,
        Box::new(BasicVertexShader),
        Box::new(PhongFragmentShader),
    );
    renderer.set_culling_mode(CullingMode::BackFace);

    let mut camera = Camera::new(
        Vec3::new(0.0, -0.75, 1.25),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        // Projection::Orthographic(graphics::camera::OrthographicProjection::new(
        //     -1.0, 1.0, -1.0, 1.0, 0.1, 10.0,
        // )),
        Projection::Perspective(graphics::camera::PerspectiveProjection::new(
            90.0,
            WIDTH as f32 / HEIGHT as f32,
            0.1,
            10.0,
        )),
    );

    let shiny_material = Material::new(
        Colour::from_u32(0xfffde8),
        Colour::from_u32(0xfffde8),
        Colour::from_u32(0xfffde8),
        50.0,
    );

    let mut cube_model = Model::new(
        vec![Mesh::cube(Colour::WHITE, 0)],
        vec![shiny_material],
        ModelTransform::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ),
    );
    cube_model.calculate_vertex_normals();

    let mut scene = Scene::new(camera);
    scene.add_light(DirectionalLight::new(
        Vec3::new(0.0, 1.0, -1.0),
        Colour::from_u32(0xfffde8),
    ));

    let mut t: f32 = 0.0;
    let mut angle: f32 = 0.0;

    let mut previous_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let dt = std::time::Instant::now()
            .duration_since(previous_time)
            .as_secs_f32();
        previous_time = std::time::Instant::now();
        t += dt;

        renderer.clear(Colour::BLACK);

        angle += 1.0 * dt;
        // camera.eye.z = 1.0 + 1.0 * t.sin();

        cube_model.transform.rotation.y = angle;
        cube_model.transform.rotation.x = 1.1 * angle;

        renderer.draw_model(&cube_model, &scene, &viewport);

        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }
}
