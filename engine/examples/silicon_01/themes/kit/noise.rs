//! Procedural noise for fragment shaders, eight pixels at a time. All of it is analytic: no
//! textures. Each function costs a handful of hashes, so use few octaves on large surfaces.

use crate::simd::{F, hash2, hash2_scalar, mix, s};

#[inline(always)]
fn smooth(t: F) -> F {
    t * t * (s(3.0) - s(2.0) * t)
}

/// Smooth value noise in [0, 1), with a period of one lattice cell.
#[inline(always)]
pub fn value(x: F, y: F) -> F {
    let (xi, yi) = (x.floor(), y.floor());
    let (u, v) = (smooth(x - xi), smooth(y - yi));
    let corner = |dx: f32, dy: f32| hash2(xi + s(dx), yi + s(dy));
    mix(
        mix(corner(0.0, 0.0), corner(1.0, 0.0), u),
        mix(corner(0.0, 1.0), corner(1.0, 1.0), u),
        v,
    )
}

/// Smooth value noise in [0, 1) over three dimensions.
#[inline(always)]
pub fn value3(x: F, y: F, z: F) -> F {
    let (zi, zf) = (z.floor(), z - z.floor());
    let layer = |dz: f32| {
        let offset = (zi + s(dz)) * s(37.0);
        value(x + offset, y + offset * s(0.61))
    };
    mix(layer(0.0), layer(1.0), smooth(zf))
}

/// Fractal noise in [0, 1): `octaves` layers of [`value`], each twice the frequency and half the
/// weight of the last.
#[inline(always)]
pub fn fbm(mut x: F, mut y: F, octaves: u32) -> F {
    let (mut sum, mut weight, mut total) = (s(0.0), 0.5, 0.0);
    for _ in 0..octaves {
        sum += value(x, y) * s(weight);
        total += weight;
        weight *= 0.5;
        (x, y) = (x * s(2.03) + s(17.1), y * s(2.03) + s(9.7));
    }
    sum / s(total)
}

/// Fractal noise in [0, 1) over three dimensions.
#[inline(always)]
pub fn fbm3(mut x: F, mut y: F, mut z: F, octaves: u32) -> F {
    let (mut sum, mut weight, mut total) = (s(0.0), 0.5, 0.0);
    for _ in 0..octaves {
        sum += value3(x, y, z) * s(weight);
        total += weight;
        weight *= 0.5;
        (x, y, z) = (
            x * s(2.03) + s(17.1),
            y * s(2.03) + s(9.7),
            z * s(2.03) + s(3.3),
        );
    }
    sum / s(total)
}

/// Cellular (Worley) noise: the distance to the nearest of a jittered point per lattice cell,
/// from 0 on a point to about 1 between points. `animate` drifts the points in circles, for
/// caustics and bubbling surfaces.
#[inline(always)]
pub fn cellular(x: F, y: F, animate: f32) -> F {
    let (xi, yi) = (x.floor(), y.floor());
    let (xf, yf) = (x - xi, y - yi);
    let mut nearest = s(8.0);
    for dy in [-1.0, 0.0, 1.0] {
        for dx in [-1.0, 0.0, 1.0] {
            let (cx, cy) = (xi + s(dx), yi + s(dy));
            let angle = hash2(cx + s(31.0), cy + s(7.0)) * s(std::f32::consts::TAU) + s(animate);
            let px = s(dx) + s(0.5) + angle.cos() * s(0.4) - xf;
            let py = s(dy) + s(0.5) + angle.sin() * s(0.4) - yf;
            nearest = nearest.fast_min(px * px + py * py);
        }
    }
    nearest.sqrt()
}

/// [`value`] for one point, for building geometry and colours on the CPU.
pub fn value_scalar(x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let (u, v) = (smooth(x - xi), smooth(y - yi));
    let corner = |dx: f32, dy: f32| hash2_scalar(xi + dx, yi + dy);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    lerp(
        lerp(corner(0.0, 0.0), corner(1.0, 0.0), u),
        lerp(corner(0.0, 1.0), corner(1.0, 1.0), u),
        v,
    )
}

/// [`fbm`] for one point, for building geometry and colours on the CPU.
pub fn fbm_scalar(mut x: f32, mut y: f32, octaves: u32) -> f32 {
    let (mut sum, mut weight, mut total) = (0.0, 0.5, 0.0);
    for _ in 0..octaves {
        sum += value_scalar(x, y) * weight;
        total += weight;
        weight *= 0.5;
        (x, y) = (x * 2.03 + 17.1, y * 2.03 + 9.7);
    }
    sum / total
}
