use crate::prelude::*;
use crate::renderer::Renderer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle2D {
    pub a: Vertex2D,
    pub b: Vertex2D,
    pub c: Vertex2D,
}
impl Triangle2D {
    pub fn new(a: Vertex2D, b: Vertex2D, c: Vertex2D) -> Self {
        Self { a, b, c }
    }

    pub fn draw(&self, renderer: &mut Renderer) {
        draw_line(renderer, self.a, self.b);
        draw_line(renderer, self.b, self.c);
        draw_line(renderer, self.c, self.a);
    }

    /// Uses a scanline algorithm to fill the triangle
    pub fn draw_filled(&self, renderer: &mut Renderer) {
        let mut vertices = [self.a, self.b, self.c];

        vertices.sort_by(|a, b| a.position.y.partial_cmp(&b.position.y).unwrap());

        let v0 = vertices[0];
        let v1 = vertices[1];
        let v2 = vertices[2];

        // Degenerate triangle
        if (v0.position.y - v2.position.y).abs() < f32::EPSILON {
            return;
        }

        let y_start = v0.position.y.ceil() as i32;
        let y_end = v2.position.y.ceil() as i32;

        for y in y_start..y_end {
            let fy = y as f32 + 0.5;

            let second_half =
                fy > v1.position.y || (v1.position.y - v0.position.y).abs() < f32::EPSILON;

            let (left, right) = if second_half {
                let t1 = (fy - v0.position.y) / (v2.position.y - v0.position.y);
                let t2 = (fy - v1.position.y) / (v2.position.y - v1.position.y);

                (v0.lerp(&v2, t1), v1.lerp(&v2, t2))
            } else {
                let t1 = (fy - v0.position.y) / (v1.position.y - v0.position.y);
                let t2 = (fy - v0.position.y) / (v2.position.y - v0.position.y);

                (v0.lerp(&v1, t1), v0.lerp(&v2, t2))
            };

            // Ensure left to right ordering
            let (left, right) = if left.position.x > right.position.x {
                (right, left)
            } else {
                (left, right)
            };

            let x_start = left.position.x.ceil() as i32;
            let x_end = right.position.x.ceil() as i32;

            for x in x_start..x_end {
                let fx = x as f32 + 0.5;

                let t = if (right.position.x - left.position.x).abs() < f32::EPSILON {
                    0.0
                } else {
                    (fx - left.position.x) / (right.position.x - left.position.x)
                };

                let pixel = left.lerp(&right, t);

                renderer.write_fragment((x, y).into(), pixel.colour, pixel.depth);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleClip {
    pub a: ClipVertex,
    pub b: ClipVertex,
    pub c: ClipVertex,
}
impl TriangleClip {
    pub fn new(a: ClipVertex, b: ClipVertex, c: ClipVertex) -> Self {
        Self { a, b, c }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle3D {
    pub a: Vertex3D,
    pub b: Vertex3D,
    pub c: Vertex3D,
}
impl Triangle3D {
    pub fn new(a: Vertex3D, b: Vertex3D, c: Vertex3D) -> Self {
        Self { a, b, c }
    }
}
