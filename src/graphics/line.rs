use crate::prelude::*;
use crate::renderer::Renderer;

pub fn draw_horizontal(renderer: &mut Renderer, p1: RasterVertex, p2: RasterVertex) {
    let y = p1.position.y as i32;
    let x_start = p1.position.x.min(p2.position.x) as i32;
    let x_end = p1.position.x.max(p2.position.x) as i32;

    for x in x_start..=x_end {
        let t = x as f32 / (x_end - x_start) as f32;
        renderer.write_fragment(
            (x, y).into(),
            p1.varyings.colour.lerp(&p2.varyings.colour, t).into(),
            p1.varyings.depth * (1.0 - t) + p2.varyings.depth * t,
        );
    }
}

pub fn draw_vertical(renderer: &mut Renderer, p1: RasterVertex, p2: RasterVertex) {
    let x = p1.position.x as i32;
    let y_start = p1.position.y.min(p2.position.y) as i32;
    let y_end = p1.position.y.max(p2.position.y) as i32;

    for y in y_start..=y_end {
        let t = y as f32 / (y_end - y_start) as f32;
        renderer.write_fragment(
            (x, y).into(),
            p1.varyings.colour.lerp(&p2.varyings.colour, t).into(),
            p1.varyings.depth * (1.0 - t) + p2.varyings.depth * t,
        );
    }
}

/// Draws a line using Bresenham's line algorithm from point `p1` to point `p2`.
pub fn draw_line(renderer: &mut Renderer, p1: RasterVertex, p2: RasterVertex) {
    let mut x0 = p1.position.x as i32;
    let mut y0 = p1.position.y as i32;
    let x1 = p2.position.x as i32;
    let y1 = p2.position.y as i32;

    if y0 == y1 {
        draw_horizontal(renderer, p1, p2);
        return;
    }

    if x0 == x1 {
        draw_vertical(renderer, p1, p2);
        return;
    }

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();

    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut err = dx - dy;

    loop {
        let t = ((x0 - p1.position.x as i32).abs() as f32) / (dx as f32);
        renderer.write_fragment(
            (x0, y0).into(),
            p1.varyings.colour.lerp(&p2.varyings.colour, t).into(),
            p1.varyings.depth * (1.0 - t) + p2.varyings.depth * t,
        );

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = err * 2;

        if e2 > -dy {
            err -= dy;
            x0 += sx;
        }

        if e2 < dx {
            err += dx;
            y0 += sy;
        }
    }
}

pub fn draw_line_dda(renderer: &mut Renderer, p1: Vec2, p2: Vec2, colour: Colour) {
    let x0 = p1.x;
    let y0 = p1.y;
    let x1 = p2.x;
    let y1 = p2.y;

    let dx = x1 - x0;
    let dy = y1 - y0;

    let steps = dx.abs().max(dy.abs());

    let x_inc = dx / steps;
    let y_inc = dy / steps;

    let mut x = x0;
    let mut y = y0;

    for _ in 0..=steps as usize {
        renderer.write_fragment((x, y).into(), colour, 0.0);
        x += x_inc;
        y += y_inc;
    }
}
