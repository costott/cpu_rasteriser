use crate::prelude::*;

use crate::graphics::fragment::{Fragment, FragmentSimd};

use wide::f32x8;

#[derive(Clone)]
pub struct Triangle2D<V>
where
    V: Interpolate,
{
    pub a: RasterVertex<V>,
    pub b: RasterVertex<V>,
    pub c: RasterVertex<V>,
}
impl<V: Interpolate> Triangle2D<V> {
    pub fn new(a: RasterVertex<V>, b: RasterVertex<V>, c: RasterVertex<V>) -> Self {
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

    // pub fn wireframe(&self, renderer: &mut Renderer) {
    //     draw_line(renderer, self.a, self.b);
    //     draw_line(renderer, self.b, self.c);
    //     draw_line(renderer, self.c, self.a);
    // }

    /// Uses a scanline algorithm to fill the triangle and interpolate vertex attributes.
    pub fn rasterise_segment<F>(&self, bounds: crate::renderer::Rect, mut callback: F)
    where
        F: FnMut(Fragment<V>),
    {
        let mut vertices = [self.a.clone(), self.b.clone(), self.c.clone()];

        vertices.sort_by(|a, b| a.position.y.partial_cmp(&b.position.y).unwrap());

        let v0 = vertices[0].clone();
        let v1 = vertices[1].clone();
        let v2 = vertices[2].clone();

        // Skip drawing if the triangle has zero height
        if (v0.position.y - v2.position.y).abs() < f32::EPSILON {
            return;
        }

        let y_start = ((v0.position.y - 0.5).ceil() as i32).max(bounds.min_y);
        let y_end = ((v2.position.y - 0.5).ceil() as i32).min(bounds.max_y);

        let mut edge_long = Edge::new(v0.clone(), v2.clone(), y_start as f32 + 0.5);

        let mut using_second_half = y_start as f32 + 0.5 > v1.position.y;

        let mut edge_short = if using_second_half {
            Edge::new(v1.clone(), v2.clone(), y_start as f32 + 0.5)
        } else {
            Edge::new(v0.clone(), v1.clone(), y_start as f32 + 0.5)
        };

        for y in y_start..y_end {
            // If the scanline is in the second half of the triangle, use the edge from v1 to v2 instead of v0 to v1.
            if !using_second_half && y as f32 + 0.5 > v1.position.y {
                using_second_half = true;
                edge_short = Edge::new(v1.clone(), v2.clone(), y as f32 + 0.5);
            }

            // Determine which edge is on the left and which is on the right for the current scanline.
            let (left, right) = if edge_long.x < edge_short.x {
                (&edge_long, &edge_short)
            } else {
                (&edge_short, &edge_long)
            };

            let x_start = ((left.x - 0.5).ceil() as i32).max(bounds.min_x);
            let x_end = ((right.x - 0.5).ceil() as i32).min(bounds.max_x);

            let width = right.x - left.x;

            let x_step = if width.abs() < f32::EPSILON {
                0.0
            } else {
                1.0 / width
            };

            let t = (x_start as f32 + 0.5 - left.x) * x_step;

            let mut depth = left.depth + (right.depth - left.depth) * t;
            let depth_step = (right.depth - left.depth) * x_step;

            let mut inv_w = left.inv_w + (right.inv_w - left.inv_w) * t;
            let inv_w_step = (right.inv_w - left.inv_w) * x_step;

            let mut varyings = left.varyings.interpolate(&right.varyings, t);
            let varyings_step = right.varyings.difference(&left.varyings).scale(x_step);

            for x in x_start..x_end {
                let perspective = if inv_w.abs() < f32::EPSILON {
                    0.0
                } else {
                    1.0 / inv_w
                };

                callback(Fragment::new(
                    (x, y).into(),
                    depth,
                    varyings.scale(perspective),
                ));

                depth += depth_step;
                inv_w += inv_w_step;
                varyings = varyings.add_scaled(&varyings_step, 1.0);
            }

            edge_long.step();
            edge_short.step();
        }
    }
}

impl<V> Triangle2D<V>
where
    V: Interpolate + SimdInterpolate,
{
    pub fn rasterise_segment_simd<F>(&self, bounds: crate::renderer::Rect, mut simd_callback: F)
    where
        F: FnMut(FragmentSimd<V>),
    {
        let mut vertices = [self.a.clone(), self.b.clone(), self.c.clone()];

        vertices.sort_by(|a, b| a.position.y.partial_cmp(&b.position.y).unwrap());

        let v0 = vertices[0].clone();
        let v1 = vertices[1].clone();
        let v2 = vertices[2].clone();

        if (v0.position.y - v2.position.y).abs() < f32::EPSILON {
            return;
        }

        let y_start = ((v0.position.y - 0.5).ceil() as i32).max(bounds.min_y);

        let y_end = ((v2.position.y - 0.5).ceil() as i32).min(bounds.max_y);

        let mut edge_long = Edge::new(v0.clone(), v2.clone(), y_start as f32 + 0.5);

        let mut using_second_half = y_start as f32 + 0.5 > v1.position.y;

        let mut edge_short = if using_second_half {
            Edge::new(v1.clone(), v2.clone(), y_start as f32 + 0.5)
        } else {
            Edge::new(v0.clone(), v1.clone(), y_start as f32 + 0.5)
        };

        let lanes = f32x8::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        for y in y_start..y_end {
            if !using_second_half && y as f32 + 0.5 > v1.position.y {
                using_second_half = true;

                edge_short = Edge::new(v1.clone(), v2.clone(), y as f32 + 0.5);
            }

            let (left, right) = if edge_long.x < edge_short.x {
                (&edge_long, &edge_short)
            } else {
                (&edge_short, &edge_long)
            };

            let x_start = ((left.x - 0.5).ceil() as i32).max(bounds.min_x);

            let x_end = ((right.x - 0.5).ceil() as i32).min(bounds.max_x);

            if x_start >= x_end {
                edge_long.step();
                edge_short.step();
                continue;
            }

            let width = right.x - left.x;

            let x_step = if width.abs() < f32::EPSILON {
                0.0
            } else {
                1.0 / width
            };

            let t = (x_start as f32 + 0.5 - left.x) * x_step;

            let depth = left.depth + (right.depth - left.depth) * t;
            let depth_step = (right.depth - left.depth) * x_step;

            let inv_w = left.inv_w + (right.inv_w - left.inv_w) * t;
            let inv_w_step = (right.inv_w - left.inv_w) * x_step;

            let varyings = left.varyings.interpolate(&right.varyings, t);
            let varyings_step = right.varyings.difference(&left.varyings).scale(x_step);

            let step8 = f32x8::splat(8.0);

            let depth_step8 = f32x8::splat(depth_step);
            let inv_w_step8 = f32x8::splat(inv_w_step);

            let mut depth8 = f32x8::splat(depth) + lanes * depth_step8;

            let mut inv_w8 = f32x8::splat(inv_w) + lanes * inv_w_step8;

            let mut varyings8 = V::simd_step(&varyings, &varyings_step, lanes);

            let mut x = x_start;

            while x + 8 <= x_end {
                let perspective8 = inv_w8.recip();
                let corrected_varyings8 = V::simd_perspective(varyings8, perspective8);

                simd_callback(FragmentSimd::new(
                    x,
                    y,
                    depth8,
                    corrected_varyings8,
                    f32x8::splat(f32::from_bits(u32::MAX)),
                ));

                depth8 += depth_step8 * step8;
                inv_w8 += inv_w_step8 * step8;

                varyings8 = V::simd_add_scaled(&varyings8, &varyings_step, step8);

                x += 8;
            }

            let remaining = (x_end - x) as usize;

            if remaining != 0 {
                let perspective8 = inv_w8.recip();
                let corrected_varyings8 = V::simd_perspective(varyings8, perspective8);

                simd_callback(FragmentSimd::new(
                    x,
                    y,
                    depth8,
                    corrected_varyings8,
                    mask_for_lanes(remaining),
                ));
            }

            edge_long.step();
            edge_short.step();
        }
    }
}

#[inline(always)]
fn mask_for_lanes(lanes: usize) -> f32x8 {
    debug_assert!(lanes <= 8);

    f32x8::new([
        if lanes > 0 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
        if lanes > 1 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
        if lanes > 2 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
        if lanes > 3 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
        if lanes > 4 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
        if lanes > 5 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
        if lanes > 6 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
        if lanes > 7 {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        },
    ])
}

/// Represents an edge of a triangle in 2D space, used for scanline rasterization.
#[derive(Clone)]
struct Edge<V>
where
    V: Interpolate,
{
    x: f32,
    x_step: f32,

    depth: f32,
    depth_step: f32,

    inv_w: f32,
    inv_w_step: f32,

    varyings: V,
    varyings_step: V,
}

impl<V: Interpolate> Edge<V> {
    fn new(a: RasterVertex<V>, b: RasterVertex<V>, y_start: f32) -> Self {
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

            depth: a.depth + (b.depth - a.depth) * t,
            depth_step: (b.depth - a.depth) * height,

            inv_w: a.inv_w + (b.inv_w - a.inv_w) * t,
            inv_w_step: (b.inv_w - a.inv_w) * height,

            varyings: a.varyings.interpolate(&b.varyings, t),
            varyings_step: b.varyings.difference(&a.varyings).scale(height),
        }
    }

    fn step(&mut self) {
        self.x += self.x_step;
        self.depth += self.depth_step;
        self.inv_w += self.inv_w_step;
        self.varyings = self.varyings.add_scaled(&self.varyings_step, 1.0);
    }
}

#[derive(Clone)]
pub struct TriangleClip<V>
where
    V: Interpolate,
{
    pub a: ClipVertex<V>,
    pub b: ClipVertex<V>,
    pub c: ClipVertex<V>,
}
impl<V: Interpolate> TriangleClip<V> {
    pub fn new(a: ClipVertex<V>, b: ClipVertex<V>, c: ClipVertex<V>) -> Self {
        Self { a, b, c }
    }
}

#[derive(Clone)]
pub struct Triangle3D<V: Clone> {
    pub a: V,
    pub b: V,
    pub c: V,
}
impl<V: Clone> Triangle3D<V> {
    pub fn new(a: V, b: V, c: V) -> Self {
        Self { a, b, c }
    }
}
impl Triangle3D<ObjVertex> {
    pub fn normal(&self) -> Vec3 {
        let ab = self.b.position - self.a.position;
        let ac = self.c.position - self.a.position;

        ab.cross(&ac).normalise()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Interpolate)]
    struct TestVaryings {
        colour: Vec3,
    }

    impl SimdInterpolate for TestVaryings {
        type Simd = TestVaryingsSimd;

        fn simd_step(value: &Self, step: &Self, lanes: f32x8) -> Self::Simd {
            TestVaryingsSimd {
                colour: [
                    f32x8::splat(value.colour.x) + f32x8::splat(step.colour.x) * lanes,
                    f32x8::splat(value.colour.y) + f32x8::splat(step.colour.y) * lanes,
                    f32x8::splat(value.colour.z) + f32x8::splat(step.colour.z) * lanes,
                ],
            }
        }

        fn simd_add_scaled(value: &Self::Simd, step: &Self, scale: f32x8) -> Self::Simd {
            TestVaryingsSimd {
                colour: [
                    value.colour[0] + f32x8::splat(step.colour.x) * scale,
                    value.colour[1] + f32x8::splat(step.colour.y) * scale,
                    value.colour[2] + f32x8::splat(step.colour.z) * scale,
                ],
            }
        }

        fn simd_perspective(value: Self::Simd, perspective: f32x8) -> Self::Simd {
            TestVaryingsSimd {
                colour: [
                    value.colour[0] * perspective,
                    value.colour[1] * perspective,
                    value.colour[2] * perspective,
                ],
            }
        }

        fn simd_extract_all(value: &Self::Simd) -> [Self; 8] {
            let r = value.colour[0].to_array();
            let g = value.colour[1].to_array();
            let b = value.colour[2].to_array();

            [
                Self {
                    colour: Vec3::new(r[0], g[0], b[0]),
                },
                Self {
                    colour: Vec3::new(r[1], g[1], b[1]),
                },
                Self {
                    colour: Vec3::new(r[2], g[2], b[2]),
                },
                Self {
                    colour: Vec3::new(r[3], g[3], b[3]),
                },
                Self {
                    colour: Vec3::new(r[4], g[4], b[4]),
                },
                Self {
                    colour: Vec3::new(r[5], g[5], b[5]),
                },
                Self {
                    colour: Vec3::new(r[6], g[6], b[6]),
                },
                Self {
                    colour: Vec3::new(r[7], g[7], b[7]),
                },
            ]
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct TestVaryingsSimd {
        colour: [f32x8; 3],
    }

    fn vertex(x: f32, y: f32, colour: Vec3) -> RasterVertex<TestVaryings> {
        RasterVertex {
            position: Vec2::new(x, y),
            depth: 0.5,
            inv_w: 1.0,
            varyings: TestVaryings { colour },
        }
    }

    fn bounds() -> crate::renderer::Rect {
        crate::renderer::Rect {
            min_x: 0,
            min_y: 0,
            max_x: 4,
            max_y: 4,
        }
    }

    #[test]
    fn bounding_box_returns_expected_values() {
        let triangle = Triangle2D::new(
            vertex(1.0, 2.0, Vec3::ZERO),
            vertex(4.0, 0.0, Vec3::ZERO),
            vertex(2.0, 3.0, Vec3::ZERO),
        );

        let (min, max) = triangle.bounding_box();

        assert_eq!(min, Vec2::new(1.0, 0.0));
        assert_eq!(max, Vec2::new(4.0, 3.0));
    }

    #[test]
    fn intersects_rect_returns_true_when_overlapping() {
        let triangle = Triangle2D::new(
            vertex(1.0, 1.0, Vec3::ZERO),
            vertex(3.0, 1.0, Vec3::ZERO),
            vertex(2.0, 3.0, Vec3::ZERO),
        );

        let rect = crate::renderer::Rect {
            min_x: 2,
            min_y: 2,
            max_x: 4,
            max_y: 4,
        };

        assert!(triangle.intersects_rect(rect));
    }

    #[test]
    fn intersects_rect_returns_false_when_disjoint() {
        let triangle = Triangle2D::new(
            vertex(0.0, 0.0, Vec3::ZERO),
            vertex(1.0, 0.0, Vec3::ZERO),
            vertex(0.0, 1.0, Vec3::ZERO),
        );

        let rect = crate::renderer::Rect {
            min_x: 5,
            min_y: 5,
            max_x: 6,
            max_y: 6,
        };

        assert!(!triangle.intersects_rect(rect));
    }

    #[test]
    fn rasterise_uses_pixel_centres_for_coverage() {
        let triangle = Triangle2D::new(
            vertex(0.1, 0.1, Vec3::ZERO),
            vertex(2.9, 0.1, Vec3::ZERO),
            vertex(0.1, 2.9, Vec3::ZERO),
        );

        let mut pixels = Vec::new();

        triangle.rasterise_segment(bounds(), |fragment| {
            pixels.push(fragment.position);
        });

        assert!(pixels.contains(&Vec2::new(0.0, 0.0)));
        assert!(pixels.contains(&Vec2::new(1.0, 0.0)));
        assert!(pixels.contains(&Vec2::new(0.0, 1.0)));

        assert!(!pixels.contains(&Vec2::new(3.0, 3.0)));
    }

    #[test]
    fn rasterise_interpolates_varyings_across_scanlines() {
        let triangle = Triangle2D::new(
            vertex(0.5, 0.5, Vec3::new(1.0, 0.0, 0.0)),
            vertex(3.5, 0.5, Vec3::new(0.0, 1.0, 0.0)),
            vertex(0.5, 3.5, Vec3::new(1.0, 0.0, 0.0)),
        );

        let mut fragments = Vec::new();

        triangle.rasterise_segment(bounds(), |fragment| {
            fragments.push(fragment);
        });

        let fragment = fragments
            .iter()
            .find(|f| f.position == Vec2::new(1.0, 1.0))
            .unwrap();

        assert!(fragment.varyings.colour.x < 1.0);
        assert!(fragment.varyings.colour.y > 0.0);
        assert_eq!(fragment.varyings.colour.z, 0.0);
    }

    #[test]
    fn rasterise_respects_bounds() {
        let triangle = Triangle2D::new(
            vertex(0.0, 0.0, Vec3::ZERO),
            vertex(4.0, 0.0, Vec3::ZERO),
            vertex(0.0, 4.0, Vec3::ZERO),
        );

        let rect = crate::renderer::Rect {
            min_x: 1,
            min_y: 1,
            max_x: 2,
            max_y: 2,
        };

        triangle.rasterise_segment(rect, |fragment| {
            assert!(fragment.position.x >= 1.0);
            assert!(fragment.position.x < 2.0);
            assert!(fragment.position.y >= 1.0);
            assert!(fragment.position.y < 2.0);
        });
    }

    #[test]
    fn rasterise_zero_height_triangle_emits_no_fragments() {
        let triangle = Triangle2D::new(
            vertex(0.0, 1.0, Vec3::ZERO),
            vertex(2.0, 1.0, Vec3::ZERO),
            vertex(4.0, 1.0, Vec3::ZERO),
        );

        let mut fragments = Vec::new();

        triangle.rasterise_segment(bounds(), |fragment| {
            fragments.push(fragment);
        });

        assert!(fragments.is_empty());
    }

    #[test]
    fn edge_new_interpolates_initial_values() {
        let edge = Edge::new(
            vertex(0.0, 0.0, Vec3::new(1.0, 0.0, 0.0)),
            vertex(4.0, 4.0, Vec3::new(0.0, 1.0, 0.0)),
            2.0,
        );

        assert_eq!(edge.x, 2.0);
        assert_eq!(edge.depth, 0.5);
        assert_eq!(edge.inv_w, 1.0);
        assert_eq!(edge.varyings.colour, Vec3::new(0.5, 0.5, 0.0));
    }

    #[test]
    fn edge_step_interpolates_attributes() {
        let mut edge = Edge::new(
            vertex(0.0, 0.0, Vec3::new(1.0, 0.0, 0.0)),
            vertex(0.0, 4.0, Vec3::new(0.0, 1.0, 0.0)),
            0.0,
        );

        edge.step();

        assert_eq!(edge.varyings.colour, Vec3::new(0.75, 0.25, 0.0));
    }

    #[test]
    fn rasterise_simd_matches_scalar() {
        let triangle = Triangle2D::new(
            vertex(0.5, 0.5, Vec3::new(1.0, 0.0, 0.0)),
            vertex(3.5, 0.5, Vec3::new(0.0, 1.0, 0.0)),
            vertex(0.5, 3.5, Vec3::new(1.0, 0.0, 0.0)),
        );

        let bounds = crate::renderer::Rect {
            min_x: 0,
            min_y: 0,
            max_x: 4,
            max_y: 4,
        };

        let mut scalar_fragments = Vec::new();

        triangle.rasterise_segment(bounds, |fragment| {
            scalar_fragments.push(fragment);
        });

        let mut simd_fragments = Vec::new();

        triangle.rasterise_segment_simd(bounds, |fragment_simd| {
            let depth = fragment_simd.depth.to_array();
            let r = fragment_simd.varyings.colour[0].to_array();
            let g = fragment_simd.varyings.colour[1].to_array();
            let b = fragment_simd.varyings.colour[2].to_array();

            let mask_bits = fragment_simd.mask.to_bitmask();

            for lane in 0..8 {
                if mask_bits & (1 << lane) == 0 {
                    continue;
                }

                simd_fragments.push((
                    fragment_simd.x_start + lane as i32,
                    fragment_simd.y,
                    depth[lane],
                    r[lane],
                    g[lane],
                    b[lane],
                ));
            }
        });

        assert_eq!(scalar_fragments.len(), simd_fragments.len(),);

        for (scalar, simd) in scalar_fragments.iter().zip(&simd_fragments) {
            assert_eq!(scalar.position.x as i32, simd.0,);

            assert_eq!(scalar.position.y as i32, simd.1,);

            assert!((scalar.depth - simd.2).abs() < 1e-5);

            assert!((scalar.varyings.colour.x - simd.3).abs() < 1e-5);

            assert!((scalar.varyings.colour.y - simd.4).abs() < 1e-5);

            assert!((scalar.varyings.colour.z - simd.5).abs() < 1e-5);
        }
    }
}
