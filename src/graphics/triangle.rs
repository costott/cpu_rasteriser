use crate::prelude::*;

use crate::graphics::fragment::Fragment;
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

    // TODO: multithreaded rasterisation
    /// Uses a scanline algorithm to fill the triangle and interpolate vertex attributes.
    pub fn rasterise<F>(&self, mut callback: F)
    where
        F: FnMut(Fragment),
    {
        let mut vertices = [self.a, self.b, self.c];

        vertices.sort_by(|a, b| a.position.y.partial_cmp(&b.position.y).unwrap());

        let v0 = vertices[0];
        let v1 = vertices[1];
        let v2 = vertices[2];

        // Skip drawing if the triangle has zero height
        if (v0.position.y - v2.position.y).abs() < f32::EPSILON {
            return;
        }

        let y_start = (v0.position.y - 0.5).ceil() as i32;
        let y_end = (v2.position.y - 0.5).ceil() as i32;

        let mut left;
        let mut right;

        let mut edge_long = Edge::new(v0, v2, y_start as f32 + 0.5);

        let mut edge_short = Edge::new(v0, v1, y_start as f32 + 0.5);

        for y in y_start..y_end {
            // // If the scanline is in the second half of the triangle, use the edge from v1 to v2 instead of v0 to v1.
            let second_half = y as f32 + 0.5 > v1.position.y;

            if second_half {
                edge_short = Edge::new(v1, v2, y as f32 + 0.5);
            }

            // Determine which edge is on the left and which is on the right for the current scanline.
            if edge_long.x < edge_short.x {
                left = edge_long;
                right = edge_short;
            } else {
                left = edge_short;
                right = edge_long;
            }

            let x_start = (left.x - 0.5).ceil() as i32;
            let x_end = (right.x - 0.5).ceil() as i32;

            let width = right.x - left.x;

            let x_step = if width.abs() < f32::EPSILON {
                0.0
            } else {
                1.0 / width
            };

            let t = (x_start as f32 + 0.5 - left.x) * x_step;

            let mut colour = left.colour + (right.colour - left.colour) * t;
            let colour_step = (right.colour - left.colour) * x_step;

            let mut normal = left.normal + (right.normal - left.normal) * t;
            let normal_step = (right.normal - left.normal) * x_step;

            let mut world_position =
                left.world_position + (right.world_position - left.world_position) * t;
            let world_position_step = (right.world_position - left.world_position) * x_step;

            let mut depth = left.depth + (right.depth - left.depth) * t;
            let depth_step = (right.depth - left.depth) * x_step;

            for x in x_start..x_end {
                callback(Fragment::new(
                    (x, y).into(),
                    world_position,
                    colour.into(),
                    normal,
                    depth,
                ));

                colour = colour + colour_step;
                normal = normal + normal_step;
                world_position = world_position + world_position_step;
                depth += depth_step;
            }

            edge_long.step();
            edge_short.step();
        }
    }
}

/// Represents an edge of a triangle in 2D space, used for scanline rasterization.
#[derive(Clone, Copy)]
struct Edge {
    x: f32,
    x_step: f32,

    colour: Vec3,
    colour_step: Vec3,

    normal: Vec3,
    normal_step: Vec3,

    world_position: Vec3,
    world_position_step: Vec3,

    depth: f32,
    depth_step: f32,
}

impl Edge {
    fn new(a: Vertex2D, b: Vertex2D, y_start: f32) -> Self {
        let dy = b.position.y - a.position.y;

        let t = if dy.abs() < f32::EPSILON {
            0.0
        } else {
            (y_start - a.position.y) / dy
        };

        let height = if dy.abs() < f32::EPSILON {
            0.0
        } else {
            1.0 / dy
        };

        Self {
            x: a.position.x + (b.position.x - a.position.x) * t,
            x_step: (b.position.x - a.position.x) * height,

            colour: Vec3::new(a.colour.r as f32, a.colour.g as f32, a.colour.b as f32)
                + (Vec3::new(b.colour.r as f32, b.colour.g as f32, b.colour.b as f32)
                    - Vec3::new(a.colour.r as f32, a.colour.g as f32, a.colour.b as f32))
                    * t,
            colour_step: (Vec3::new(b.colour.r as f32, b.colour.g as f32, b.colour.b as f32)
                - Vec3::new(a.colour.r as f32, a.colour.g as f32, a.colour.b as f32))
                * height,

            normal: a.normal + (b.normal - a.normal) * t,
            normal_step: (b.normal - a.normal) * height,

            world_position: a.world_position + (b.world_position - a.world_position) * t,
            world_position_step: (b.world_position - a.world_position) * height,

            depth: a.depth + (b.depth - a.depth) * t,
            depth_step: (b.depth - a.depth) * height,
        }
    }

    fn step(&mut self) {
        self.x += self.x_step;
        self.colour = self.colour + self.colour_step;
        self.normal = self.normal + self.normal_step;
        self.world_position = self.world_position + self.world_position_step;
        self.depth += self.depth_step;
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

    pub fn normal(&self) -> Vec3 {
        let ab = self.b.position - self.a.position;
        let ac = self.c.position - self.a.position;

        ab.cross(&ac).normalise()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::{fragment_shader::BasicFragmentShader, vertex_shader::BasicVertexShader};

    #[test]
    fn draw_filled_uses_pixel_centres_for_coverage() {
        let viewport = Viewport::new(4, 4);
        let mut renderer = Renderer::new(
            &viewport,
            Box::new(BasicVertexShader),
            Box::new(BasicFragmentShader),
        );
        renderer.clear(Colour::BLACK);

        let triangle = Triangle2D::new(
            Vertex2D::new(
                Vec2::new(0.1, 0.1),
                Vec3::ZERO,
                Colour::RED,
                Vec3::ZERO,
                0.5,
            ),
            Vertex2D::new(
                Vec2::new(2.9, 0.1),
                Vec3::ZERO,
                Colour::RED,
                Vec3::ZERO,
                0.5,
            ),
            Vertex2D::new(
                Vec2::new(0.1, 2.9),
                Vec3::ZERO,
                Colour::RED,
                Vec3::ZERO,
                0.5,
            ),
        );

        triangle.rasterise(|fragment| {
            renderer.write_fragment(fragment.position, fragment.colour, fragment.depth);
        });

        let black = Colour::BLACK.to_u32();
        let covered_pixels = [0, 1, 4, 5];

        for &index in &covered_pixels {
            assert_ne!(
                renderer.pixels()[index],
                black,
                "expected coverage at index {index}"
            );
        }

        for (index, pixel) in renderer.pixels().iter().enumerate() {
            if covered_pixels.contains(&index) {
                continue;
            }

            assert_eq!(*pixel, black, "unexpected filled pixel at index {index}");
        }
    }

    #[test]
    fn draw_filled_interpolates_colour_across_scanlines() {
        let viewport = Viewport::new(4, 4);
        let mut renderer = Renderer::new(
            &viewport,
            Box::new(BasicVertexShader),
            Box::new(BasicFragmentShader),
        );
        renderer.clear(Colour::BLACK);

        let triangle = Triangle2D::new(
            Vertex2D::new(
                Vec2::new(0.5, 0.5),
                Vec3::ZERO,
                Colour::RED,
                Vec3::ZERO,
                0.5,
            ),
            Vertex2D::new(
                Vec2::new(3.5, 0.5),
                Vec3::ZERO,
                Colour::GREEN,
                Vec3::ZERO,
                0.5,
            ),
            Vertex2D::new(
                Vec2::new(0.5, 3.5),
                Vec3::ZERO,
                Colour::RED,
                Vec3::ZERO,
                0.5,
            ),
        );

        triangle.rasterise(|fragment| {
            renderer.write_fragment(fragment.position, fragment.colour, fragment.depth);
        });

        let pixel = Colour::from_u32(renderer.pixels()[5]);

        assert!(
            pixel.r < 255,
            "expected red channel to decrease across the scanline"
        );
        assert!(
            pixel.g > 0,
            "expected green channel to increase across the scanline"
        );
    }

    #[test]
    fn draw_filled_interpolates_normals_across_scanlines() {
        let left = Vertex2D::new(
            Vec2::new(0.0, 0.0),
            Vec3::ZERO,
            Colour::WHITE,
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
        );
        let right = Vertex2D::new(
            Vec2::new(0.0, 4.0),
            Vec3::ZERO,
            Colour::WHITE,
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
        );

        let edge = Edge::new(left, right, 0.0);

        assert_eq!(edge.normal, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(edge.normal_step, Vec3::new(-0.25, 0.25, 0.0));

        let mut stepped = edge;
        stepped.step();

        assert_eq!(stepped.normal, Vec3::new(0.75, 0.25, 0.0));
    }
}
