use minifb::{Key, Window, WindowOptions};

pub mod framebuffer;
use framebuffer::FrameBuffer;
pub mod colour;
pub mod graphics;
pub mod maths;

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

    let mut framebuffer = FrameBuffer::new(WIDTH, HEIGHT);

    let triangle = Triangle::new(
        (100.0, 100.0, Colour::BLUE).into(),
        (200.0, 150.0, Colour::RED).into(),
        (150.0, 200.0, Colour::GREEN).into(),
    );

    while window.is_open() && !window.is_key_down(Key::Escape) {
        framebuffer.clear(Colour::BLACK);

        triangle.draw_filled(&mut framebuffer);
        // triangle.draw_filled(&mut framebuffer, Colour::BLUE);

        window
            .update_with_buffer(framebuffer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }
}
