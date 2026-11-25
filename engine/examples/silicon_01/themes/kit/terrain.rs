//! Ground: a polar heightfield whose rings widen with distance, so facets stay small near the
//! chip and the world still reaches the horizon for few triangles.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use super::shape::Shape;

/// The radii of `count` rings from `inner` to `outer`, spaced geometrically.
pub fn ring_radii(inner: f32, outer: f32, count: usize) -> Vec<f32> {
    let k = (outer / inner).powf(1.0 / (count - 1) as f32);
    (0..count).map(|i| inner * k.powi(i as i32)).collect()
}

/// Builds terrain centred on the origin as flat-shaded triangles, facing up.
///
/// `radii` are the ring radii from the centre outwards (see [`ring_radii`]) and `segments` the
/// number of facets round each ring; alternate rings are offset half a facet so the triangulation
/// doesn't show radial seams. `height(x, z)` is the surface and `colour(centroid, normal)` colours
/// each facet. Keep the same `height` to place props with, so they stand on the surface.
pub fn terrain(
    radii: &[f32],
    segments: usize,
    height: impl Fn(f32, f32) -> f32,
    colour: impl Fn(Vec3, Vec3) -> Vec3,
) -> Shape {
    let point = |ring: usize, i: usize| {
        let offset = if ring.is_multiple_of(2) { 0.0 } else { 0.5 };
        let angle = TAU * (i as f32 + offset) / segments as f32;
        let r = radii[ring];
        let (x, z) = (r * angle.cos(), -r * angle.sin());
        Vec3::new(x, height(x, z), z)
    };
    let centre = Vec3::new(0.0, height(0.0, 0.0), 0.0);
    let mut shape = Shape::new();
    let add = |shape: &mut Shape, a: Vec3, b: Vec3, c: Vec3| {
        let normal = (b - a).cross(c - a).normalise();
        shape.triangle(a, b, c, colour((a + b + c) / 3.0, normal));
    };
    for i in 0..segments {
        add(&mut shape, centre, point(0, i), point(0, i + 1));
    }
    for ring in 0..radii.len() - 1 {
        for i in 0..segments {
            let (a0, a1) = (point(ring, i), point(ring, i + 1));
            let (b0, b1) = (point(ring + 1, i), point(ring + 1, i + 1));
            if ring.is_multiple_of(2) {
                add(&mut shape, a0, b0, a1);
                add(&mut shape, a1, b0, b1);
            } else {
                add(&mut shape, a0, b0, b1);
                add(&mut shape, a0, b1, a1);
            }
        }
    }
    shape
}
