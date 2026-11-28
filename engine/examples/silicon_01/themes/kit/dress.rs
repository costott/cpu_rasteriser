//! Dressing the chip for a world.
//!
//! A stage restyles the chip in two ways. Its [`Look`](crate::chip::Look) recolours the chip's own
//! materials and light. Extras add things to it: build a [`Shape`](super::shape::Shape) in the
//! chip's own space (lying flat, die facing +y, the package top at y = 0, 4 units wide) and draw
//! it with [`Motion::OnChip`](super::draw::Motion::OnChip), so it moves with the chip's pose.
//! [`rim_point`] picks spots for moss, barnacles or pins, and [`chip_shadow`] lets the chip shade
//! the ground.

use cpu_rasteriser::prelude::*;

use super::random::Rng;
use crate::chip::HALF;
use crate::shaders::{View, transform_point};
use crate::simd::{F, clamp01, s, smoothstep};

/// How far in from the package's edge [`rim_point`] ranges: clear of the capacitors and the die.
const RIM_WIDTH: f32 = 0.45;

/// A random spot on the package's top near its edge, and the horizontal direction pointing out
/// of the package there, in the chip's own space.
pub fn rim_point(rng: &mut Rng) -> (Vec3, Vec3) {
    let along = rng.range(-HALF, HALF);
    let inset = rng.range(0.0, RIM_WIDTH);
    let edge = HALF - inset;
    match rng.index(4) {
        0 => (Vec3::new(along, 0.0, -edge), Vec3::new(0.0, 0.0, -1.0)),
        1 => (Vec3::new(along, 0.0, edge), Vec3::new(0.0, 0.0, 1.0)),
        2 => (Vec3::new(-edge, 0.0, along), Vec3::new(-1.0, 0.0, 0.0)),
        _ => (Vec3::new(edge, 0.0, along), Vec3::new(1.0, 0.0, 0.0)),
    }
}

/// How much of the sun is blocked from `world` by the chip's package: 0 in the open to 1 in its
/// shadow, with a penumbra that widens with distance. `sun` points towards the sun.
#[inline(always)]
pub fn chip_shadow(view: &View, sun: Vec3, world: Vec3Simd) -> F {
    let inverse = &view.frame.chip_inverse;
    let [px, py, pz, _] = transform_point(inverse, world);
    let direction = |row: usize| {
        let m = &inverse.data[row];
        s(m[0] * sun.x + m[1] * sun.y + m[2] * sun.z)
    };
    let (dx, dy, dz) = (direction(0), direction(1), direction(2));
    let t = (-py / dy).fast_max(s(0.0));
    let hit_x = (px + dx * t).abs();
    let hit_z = (pz + dz * t).abs();
    let edge = s(HALF) - hit_x.fast_max(hit_z);
    let penumbra = s(0.06) + t * s(0.05);
    smoothstep(0.0, 1.0, clamp01(edge / penumbra * s(0.5) + s(0.5))) * clamp01(t * s(8.0))
}
