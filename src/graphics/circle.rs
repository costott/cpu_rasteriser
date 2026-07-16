use crate::framebuffer::FrameBuffer;
use crate::prelude::*;

/// Plots the 8 symmetrical points of a circle given a center point and an offset (x, y) from the center.
fn plot8(framebuffer: &mut FrameBuffer, center: Vec2, x: i32, y: i32, colour: Colour) {
    framebuffer.set_pixel((center.x as i32 + x, center.y as i32 + y).into(), colour);
    framebuffer.set_pixel((center.x as i32 - x, center.y as i32 + y).into(), colour);
    framebuffer.set_pixel((center.x as i32 + x, center.y as i32 - y).into(), colour);
    framebuffer.set_pixel((center.x as i32 - x, center.y as i32 - y).into(), colour);
    framebuffer.set_pixel((center.x as i32 + y, center.y as i32 + x).into(), colour);
    framebuffer.set_pixel((center.x as i32 - y, center.y as i32 + x).into(), colour);
    framebuffer.set_pixel((center.x as i32 + y, center.y as i32 - x).into(), colour);
    framebuffer.set_pixel((center.x as i32 - y, center.y as i32 - x).into(), colour);
}

/// Draws a circle using the basic circle drawing algorithm y = +-sqrt(r^2 - x^2) from the center point `center` with the specified `radius` and `colour`. Not filled.
pub fn draw_circle_basic(framebuffer: &mut FrameBuffer, center: Vec2, radius: f32, colour: Colour) {
    let radius_squared = radius * radius;
    let mut x = 0.0;
    let y = radius;

    while x <= y {
        let y_offset = (radius_squared - x * x).sqrt();
        let y_int = y_offset.round() as i32;

        plot8(framebuffer, center, x as i32, y_int, colour);

        x += 1.0;
    }
}

/// Draws a circle using the Midpoint Circle Algorithm from the center point `center` with the specified `radius` and `colour`. Not filled.
pub fn draw_circle(framebuffer: &mut FrameBuffer, center: Vec2, radius: f32, colour: Colour) {
    let mut x = radius as i32;
    let mut y = 0;
    let mut p = 1 - x;

    while x > y {
        plot8(framebuffer, center, x, y, colour);

        y += 1;
        if p <= 0 {
            p = p + 2 * y + 1;
        } else {
            x -= 1;
            p = p + 2 * y - 2 * x + 1;
        }
    }
}
