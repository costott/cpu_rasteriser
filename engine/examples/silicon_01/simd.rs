//! Small SIMD maths helpers shared by the shaders. Vectors of colour use `Vec3Simd` with x, y and
//! z as red, green and blue.

use cpu_rasteriser::prelude::*;
use cpu_rasteriser::wide::f32x8;

pub type F = f32x8;

#[inline(always)]
pub fn s(value: f32) -> F {
    F::splat(value)
}

#[inline(always)]
pub fn v3(components: [F; 3]) -> Vec3Simd {
    Vec3Simd::new(components[0], components[1], components[2])
}

#[inline(always)]
pub fn splat3(v: Vec3) -> Vec3Simd {
    Vec3Simd::new(s(v.x), s(v.y), s(v.z))
}

#[inline(always)]
pub fn rgb(r: f32, g: f32, b: f32) -> Vec3Simd {
    Vec3Simd::new(s(r), s(g), s(b))
}

#[inline(always)]
pub fn mul3(a: Vec3Simd, b: Vec3Simd) -> Vec3Simd {
    Vec3Simd::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

#[inline(always)]
pub fn cross(a: Vec3Simd, b: Vec3Simd) -> Vec3Simd {
    Vec3Simd::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

#[inline(always)]
pub fn reflect(incident: Vec3Simd, normal: Vec3Simd) -> Vec3Simd {
    incident - normal * (s(2.0) * incident.dot(normal))
}

#[inline(always)]
pub fn length(v: Vec3Simd) -> F {
    v.dot(v).sqrt()
}

/// `if_true` in the lanes where `mask` is set, else `if_false`.
#[inline(always)]
pub fn select3(mask: F, if_true: Vec3Simd, if_false: Vec3Simd) -> Vec3Simd {
    Vec3Simd::new(
        mask.select(if_true.x, if_false.x),
        mask.select(if_true.y, if_false.y),
        mask.select(if_true.z, if_false.z),
    )
}

#[inline(always)]
pub fn mix(a: F, b: F, t: F) -> F {
    a + (b - a) * t
}

#[inline(always)]
pub fn mix3(a: Vec3Simd, b: Vec3Simd, t: F) -> Vec3Simd {
    a + (b - a) * t
}

#[inline(always)]
pub fn clamp01(x: F) -> F {
    x.fast_max(s(0.0)).fast_min(s(1.0))
}

#[inline(always)]
pub fn smoothstep(edge0: f32, edge1: f32, x: F) -> F {
    let t = clamp01((x - s(edge0)) / s(edge1 - edge0));
    t * t * (s(3.0) - s(2.0) * t)
}

#[inline(always)]
pub fn fract(x: F) -> F {
    x - x.floor()
}

#[inline(always)]
pub fn select(mask: F, if_true: F, if_false: F) -> F {
    mask.select(if_true, if_false)
}

#[inline(always)]
pub fn lt(a: F, b: F) -> F {
    a.simd_lt(b)
}

#[inline(always)]
pub fn gt(a: F, b: F) -> F {
    a.simd_gt(b)
}

#[inline(always)]
pub fn ge(a: F, b: F) -> F {
    a.simd_ge(b)
}

/// 1 inside a shape whose signed distance is `d`, falling to 0 across one pixel (`aa`).
#[inline(always)]
pub fn cover(d: F, aa: F) -> F {
    clamp01(s(0.5) - d / aa)
}

/// Coverage of a hex digit in seven-segment strokes at `(x, y)`, with the digit centred on the
/// origin, about 1.2 wide and 1.7 tall, and `y` up. `seed` in [0, 1) picks the digit; `aa` is the
/// edge softness as in [`cover`].
#[inline(always)]
pub fn hex_digit(x: F, y: F, aa: F, seed: F) -> F {
    const SEGMENTS: [u32; 16] = [
        0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F, 0x77, 0x7C, 0x39, 0x5E, 0x79,
        0x71,
    ];
    let digits = (seed * s(16.0)).floor().to_array();
    let lit = F::from(digits.map(|d| SEGMENTS[(d as usize).min(15)] as f32));
    let on = |bit: u32| {
        let shifted = (lit * s(1.0 / (1u32 << bit) as f32)).floor();
        (fract(shifted * s(0.5)) * s(2.0)).fast_min(s(1.0))
    };
    let bar = |cx: f32, cy: f32, hx: f32, hy: f32, bit: u32| {
        let dx = (x - s(cx)).abs() - s(hx);
        let dy = (y - s(cy)).abs() - s(hy);
        cover(dx.fast_max(dy), aa) * on(bit)
    };
    bar(0.0, 0.78, 0.26, 0.07, 0)
        .fast_max(bar(0.34, 0.39, 0.07, 0.3, 1))
        .fast_max(bar(0.34, -0.39, 0.07, 0.3, 2))
        .fast_max(bar(0.0, -0.78, 0.26, 0.07, 3))
        .fast_max(bar(-0.34, -0.39, 0.07, 0.3, 4))
        .fast_max(bar(-0.34, 0.39, 0.07, 0.3, 5))
        .fast_max(bar(0.0, 0.0, 0.26, 0.07, 6))
}

/// 1.0 where `a < b`, else 0.0.
#[inline(always)]
pub fn step_lt(a: F, b: F) -> F {
    select(lt(a, b), s(1.0), s(0.0))
}

/// `x.powf(n)` on [0, 1] for a highlight's exponent `n`, rounded to the nearest power of two so
/// it costs a few squarings.
#[inline(always)]
pub fn pow_highlight(x: F, n: f32) -> F {
    pow2n(x, n.max(1.0).log2().round() as u32)
}

/// `x` raised to `2^squarings`.
#[inline(always)]
pub fn pow2n(mut x: F, squarings: u32) -> F {
    for _ in 0..squarings {
        x = x * x;
    }
    x
}

/// A sine-free hash of two coordinates to [0, 1) (Dave Hoskins' `hash12`).
#[inline(always)]
pub fn hash2(x: F, y: F) -> F {
    let mut px = fract(x * s(0.1031));
    let mut py = fract(y * s(0.1031));
    let mut pz = fract(x * s(0.1031));
    let d = px * (py + s(33.33)) + py * (pz + s(33.33)) + pz * (px + s(33.33));
    px += d;
    py += d;
    pz += d;
    fract((px + py) * pz)
}

/// The scalar version of [`hash2`], for building geometry.
pub fn hash2_scalar(x: f32, y: f32) -> f32 {
    let fract = |v: f32| v - v.floor();
    let mut px = fract(x * 0.1031);
    let mut py = fract(y * 0.1031);
    let mut pz = fract(x * 0.1031);
    let d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33);
    px += d;
    py += d;
    pz += d;
    fract((px + py) * pz)
}

/// Inigo Quilez's cosine palette, `a + b * cos(2π(c t + d))`, with fixed `a`, `b` and `c`.
#[inline(always)]
pub fn palette(t: F, phase: Vec3) -> Vec3Simd {
    let tau = s(std::f32::consts::TAU);
    let channel = |d: f32| s(0.5) + s(0.5) * (tau * (t + s(d))).cos();
    Vec3Simd::new(channel(phase.x), channel(phase.y), channel(phase.z))
}

/// Krzysztof Narkowicz's fit of the ACES filmic tone curve.
#[inline(always)]
pub fn aces(x: F) -> F {
    let x = x.fast_max(s(0.0));
    clamp01((x * (s(2.51) * x + s(0.03))) / (x * (s(2.43) * x + s(0.59)) + s(0.14)))
}

#[inline(always)]
pub fn to_colour(c: Vec3Simd, alpha: F) -> ColourSimd {
    ColourSimd {
        r: c.x,
        g: c.y,
        b: c.z,
        a: alpha,
    }
}

#[inline(always)]
pub fn from_colour(c: ColourSimd) -> Vec3Simd {
    Vec3Simd::new(c.r, c.g, c.b)
}
