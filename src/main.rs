use minifb::{Key, Window, WindowOptions};

pub mod colour;
pub mod depthbuffer;
pub mod framebuffer;
pub mod graphics;
pub mod maths;
pub mod renderer;
use renderer::Renderer;

pub mod prelude;
use prelude::*;

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

    let mut renderer = Renderer::new(WIDTH, HEIGHT);

    let triangle = Triangle::new(
        (100.0, 100.0, Colour::BLUE, 0.25).into(),
        (200.0, 150.0, Colour::RED, 0.25).into(),
        (150.0, 200.0, Colour::GREEN, 0.25).into(),
    );

    let mut behind_triangle = Triangle::new(
        (120.0, 100.0, Colour::from_u32(0x338000), 0.5).into(),
        (220.0, 150.0, Colour::from_u32(0x008080), 0.5).into(),
        (170.0, 200.0, Colour::from_u32(0x800080), 0.0).into(),
    );

    let mut t: f32 = 0.0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        renderer.clear(Colour::BLACK);

        behind_triangle.draw_filled(&mut renderer);
        triangle.draw_filled(&mut renderer);

        behind_triangle.c.depth = 0.5 + 0.5 * -t.sin().powi(2);

        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
        t += 0.017;
    }
}
