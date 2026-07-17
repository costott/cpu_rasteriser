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
    geometry_processing::GeometryProcessor,
};

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

    let mut renderer = Renderer::new(&viewport);

    let mut camera = Camera::new(
        Vec3::new(0.0, 0.0, 1.0),
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

    let y_axis_max = Vertex3D::new(Vec3::new(0.0, 100.0, 0.0), Colour::WHITE);
    let y_axis_min = Vertex3D::new(Vec3::new(0.0, -100.0, 0.0), Colour::WHITE);
    let x_axis_max = Vertex3D::new(Vec3::new(100.0, 0.0, 0.0), Colour::WHITE);
    let x_axis_min = Vertex3D::new(Vec3::new(-100.0, 0.0, 0.0), Colour::WHITE);

    let a_orig = Vertex3D::new(Vec3::new(-0.5, -0.5, 0.25), Colour::BLUE);
    let b_orig = Vertex3D::new(Vec3::new(0.5, -0.5, 0.25), Colour::RED);
    let c_orig = Vertex3D::new(Vec3::new(0.0, 0.5, 0.25), Colour::GREEN);

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
        // camera.eye.z = 1.0 + 0.5 * (t * 2.0).sin();

        // draw axes
        draw_line(
            &mut renderer,
            GeometryProcessor::process(y_axis_min, Mat4::identity(), &camera, &viewport),
            GeometryProcessor::process(y_axis_max, Mat4::identity(), &camera, &viewport),
        );
        draw_line(
            &mut renderer,
            GeometryProcessor::process(x_axis_min, Mat4::identity(), &camera, &viewport),
            GeometryProcessor::process(x_axis_max, Mat4::identity(), &camera, &viewport),
        );

        // rotate triangle by angle
        let rotation_matrix = Mat4::rotate_y(angle);
        let a = Vertex3D::new(
            (rotation_matrix * a_orig.position.to_homogenous()).homogenize_to_vec3(),
            a_orig.colour,
        );
        let b = Vertex3D::new(
            (rotation_matrix * b_orig.position.to_homogenous()).homogenize_to_vec3(),
            b_orig.colour,
        );
        let c = Vertex3D::new(
            (rotation_matrix * c_orig.position.to_homogenous()).homogenize_to_vec3(),
            c_orig.colour,
        );

        let triangle = Triangle::new(
            GeometryProcessor::process(a, Mat4::identity(), &camera, &viewport),
            GeometryProcessor::process(b, Mat4::identity(), &camera, &viewport),
            GeometryProcessor::process(c, Mat4::identity(), &camera, &viewport),
        );

        triangle.draw_filled(&mut renderer);

        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }
}
