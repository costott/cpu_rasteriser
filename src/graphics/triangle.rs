use crate::prelude::*;

use crate::graphics::fragment::Fragment;
use crate::renderer::Renderer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle2D {
    pub a: RasterVertex,
    pub b: RasterVertex,
    pub c: RasterVertex,
}
impl Triangle2D {
    pub fn new(a: RasterVertex, b: RasterVertex, c: RasterVertex) -> Self {
        Self { a, b, c }
    }

    pub fn bounding_box(&self) -> (Vec2, Vec2) {
        let min_x = self
            .a
            .position
            .x
            .min(self.b.position.x)
            .min(self.c.position.x);
        let max_x = self
            .a
            .position
            .x
            .max(self.b.position.x)
            .max(self.c.position.x);
        let min_y = self
            .a
            .position
            .y
            .min(self.b.position.y)
            .min(self.c.position.y);
        let max_y = self
            .a
            .position
            .y
            .max(self.b.position.y)
            .max(self.c.position.y);
        (Vec2::new(min_x, min_y), Vec2::new(max_x, max_y))
    }

    pub fn intersects_rect(&self, rect: crate::renderer::Rect) -> bool {
        let (min, max) = self.bounding_box();

        let triangle_min_x = min.x.floor() as i32;
        let triangle_max_x = max.x.ceil() as i32;
        let triangle_min_y = min.y.floor() as i32;
        let triangle_max_y = max.y.ceil() as i32;

        !(triangle_max_x < rect.min_x
            || triangle_min_x > rect.max_x
            || triangle_max_y < rect.min_y
            || triangle_min_y > rect.max_y)
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

            let mut varyings = left.varyings + (right.varyings - left.varyings) * t;
            let varyings_step = (right.varyings - left.varyings) * x_step;

            for x in x_start..x_end {
                let perspective = if varyings.inv_w.abs() < f32::EPSILON {
                    0.0
                } else {
                    1.0 / varyings.inv_w
                };

                callback(Fragment::new(
                    (x, y).into(),
                    varyings.world_position * perspective,
                    (varyings.colour * perspective).into(),
                    (varyings.normal * perspective).normalise(),
                    varyings.depth,
                ));

                varyings = varyings + varyings_step;
            }

            edge_long.step();
            edge_short.step();
        }
    }

    pub fn rasterise_segment<F>(&self, bounds: crate::renderer::Rect, mut callback: F)
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

        let y_start = ((v0.position.y - 0.5).ceil() as i32).max(bounds.min_y);
        let y_end = ((v2.position.y - 0.5).ceil() as i32).min(bounds.max_y);

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

            let x_start = ((left.x - 0.5).ceil() as i32).max(bounds.min_x);
            let x_end = ((right.x - 0.5).ceil() as i32).min(bounds.max_x);

            let width = right.x - left.x;

            let x_step = if width.abs() < f32::EPSILON {
                0.0
            } else {
                1.0 / width
            };

            let t = (x_start as f32 + 0.5 - left.x) * x_step;

            let mut varyings = left.varyings + (right.varyings - left.varyings) * t;
            let varyings_step = (right.varyings - left.varyings) * x_step;

            for x in x_start..x_end {
                let perspective = if varyings.inv_w.abs() < f32::EPSILON {
                    0.0
                } else {
                    1.0 / varyings.inv_w
                };

                callback(Fragment::new(
                    (x, y).into(),
                    varyings.world_position * perspective,
                    (varyings.colour * perspective).into(),
                    (varyings.normal * perspective).normalise(),
                    varyings.depth,
                ));

                varyings = varyings + varyings_step;
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

    varyings: RasterVaryings,
    varyings_step: RasterVaryings,
}

impl Edge {
    fn new(a: RasterVertex, b: RasterVertex, y_start: f32) -> Self {
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

            varyings: a.varyings + (b.varyings - a.varyings) * t,
            varyings_step: (b.varyings - a.varyings) * height,
        }
    }

    fn step(&mut self) {
        self.x += self.x_step;
        self.varyings = self.varyings + self.varyings_step;
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
    use std::sync::Arc;

    use super::*;
    use crate::graphics::{fragment_shader::BasicFragmentShader, vertex_shader::BasicVertexShader};

    #[test]
    fn draw_filled_uses_pixel_centres_for_coverage() {
        let viewport = Viewport::new(4, 4);
        let mut renderer = Renderer::new(
            &viewport,
            Box::new(BasicVertexShader),
            Arc::new(BasicFragmentShader),
        )
        .unwrap();
        renderer.clear(Colour::BLACK);

        let triangle = Triangle2D::new(
            RasterVertex::new(
                Vec2::new(0.1, 0.1),
                RasterVaryings::new(Vec3::ZERO, Colour::RED.into(), Vec3::ZERO, 0.5, 1.0),
            ),
            RasterVertex::new(
                Vec2::new(2.9, 0.1),
                RasterVaryings::new(Vec3::ZERO, Colour::RED.into(), Vec3::ZERO, 0.5, 1.0),
            ),
            RasterVertex::new(
                Vec2::new(0.1, 2.9),
                RasterVaryings::new(Vec3::ZERO, Colour::RED.into(), Vec3::ZERO, 0.5, 1.0),
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
            Arc::new(BasicFragmentShader),
        )
        .unwrap();
        renderer.clear(Colour::BLACK);

        let triangle = Triangle2D::new(
            RasterVertex::new(
                Vec2::new(0.5, 0.5),
                RasterVaryings::new(Vec3::ZERO, Colour::RED.into(), Vec3::ZERO, 0.5, 1.0),
            ),
            RasterVertex::new(
                Vec2::new(3.5, 0.5),
                RasterVaryings::new(Vec3::ZERO, Colour::GREEN.into(), Vec3::ZERO, 0.5, 1.0),
            ),
            RasterVertex::new(
                Vec2::new(0.5, 3.5),
                RasterVaryings::new(Vec3::ZERO, Colour::RED.into(), Vec3::ZERO, 0.5, 1.0),
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
        let left = RasterVertex::new(
            Vec2::new(0.0, 0.0),
            RasterVaryings::new(
                Vec3::ZERO,
                Colour::WHITE.into(),
                Vec3::new(1.0, 0.0, 0.0),
                0.0,
                1.0,
            ),
        );
        let right = RasterVertex::new(
            Vec2::new(0.0, 4.0),
            RasterVaryings::new(
                Vec3::ZERO,
                Colour::WHITE.into(),
                Vec3::new(0.0, 1.0, 0.0),
                0.0,
                1.0,
            ),
        );

        let edge = Edge::new(left, right, 0.0);

        assert_eq!(edge.varyings.normal, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(edge.varyings_step.normal, Vec3::new(-0.25, 0.25, 0.0));

        let mut stepped = edge;
        stepped.step();

        assert_eq!(stepped.varyings.normal, Vec3::new(0.75, 0.25, 0.0));
    }
}
