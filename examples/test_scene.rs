use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{
        camera::{Camera, Projection},
        fragment_shader::PhongFragmentShader,
        lighting::DirectionalLight,
        vertex_shader::BasicVertexShader,
    },
    loaders::obj::load_obj,
    renderer::{CullingMode, Renderer},
};

mod common;
use common::camera_controller::FirstPersonControls;

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let mut controls = FirstPersonControls::new(&camera);

    let floor_material = Material::new(
        "Floor".to_string(),
        Colour::from_u32(0x808080),
        Colour::from_u32(0x404040),
        Colour::from_u32(0xffffff),
        1.0,
    );

    let red_plastic = Material::new(
        "Red Plastic".to_string(),
        Colour::from_u32(0xff0000),
        Colour::from_u32(0x990000),
        Colour::from_u32(0xffffff),
        64.0,
    );

    let polished_brass = Material::new(
        "Polished Brass".to_string(),
        Colour::from_u32(0x543808),
        Colour::from_u32(0x8b7500),
        Colour::from_u32(0xffffff),
        21.8,
    );

    let mut floor_model = Model::new(
        vec![Mesh::cube(Colour::from_u32(0x808080), Some(0))],
        vec![floor_material],
        ModelTransform::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(50.0, 0.1, 50.0),
        ),
    )
    .unwrap();
    floor_model.calculate_vertex_normals();

    let mut cube1 = Model::new(
        vec![Mesh::cube(Colour::WHITE, Some(0))],
        vec![polished_brass],
        ModelTransform::new(
            Vec3::new(-0.8, 0.5, -1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ),
    )
    .unwrap();
    cube1.calculate_vertex_normals();

    let mut cube2 = Model::new(
        vec![Mesh::cube(Colour::WHITE, Some(0))],
        vec![red_plastic],
        ModelTransform::new(
            Vec3::new(0.5, 0.25, 0.5),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
        ),
    )
    .unwrap();
    cube2.calculate_vertex_normals();

    let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
    teapot.transform.scale = Vec3::ONE * 0.1;

    let mut loaded_cube = load_obj(std::path::Path::new("assets/cube/cube.obj"))?;
    loaded_cube.transform.scale = Vec3::ONE * 0.5;

    let mut scene = Scene::new(camera);
    scene.add_light(DirectionalLight::new(
        Vec3::new(0.0, -1.0, -1.0),
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
        // scene.camera.eye.z = 1.0 + 1.0 * t.sin();

        // cube_model.transform.rotation.y = angle;
        // cube_model.transform.rotation.x = 1.1 * angle;

        controls.update(&mut scene.camera, &window, dt);

        renderer.draw_model(&floor_model, &scene, &viewport);
        renderer.draw_model(&cube1, &scene, &viewport);
        renderer.draw_model(&cube2, &scene, &viewport);
        renderer.draw_model(&teapot, &scene, &viewport);
        renderer.draw_model(&loaded_cube, &scene, &viewport);
        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
