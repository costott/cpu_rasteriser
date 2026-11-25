//! Streaks of light: camera-facing quads whose cores stay a couple of pixels wide at 720p however
//! far away they are, however the effect that places them moves them.

use cpu_rasteriser::prelude::*;

use crate::shaders::View;
use crate::simd::{F, clamp01, mix3, rgb, s, smoothstep, to_colour, v3};

/// The standard deviation of a streak's core, in pixels of a 720p frame: a full width at half
/// maximum of about 2 px.
const CORE_SIGMA: f32 = 0.9;
/// How many core deviations a streak's quad reaches either side of its axis.
const REACH: f32 = 7.0;
/// How many core deviations the quad extends beyond each end, for the glow around the head.
const CAP: f32 = 6.0;

/// One quad corner of a streak: `corner.x` runs from the tail (0) to the head (1) and `corner.y`
/// across the streak (-1 to 1). `data` is the effect's own per-streak data.
#[derive(Clone, Copy)]
pub struct StreakVertex<D> {
    pub corner: Vec2,
    pub data: D,
}

/// The four corners of a streak's quad, with `data` on each.
pub fn streak_corners<D: Copy>(data: D) -> [StreakVertex<D>; 4] {
    [(0.0, -1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 1.0)].map(|(along, across)| StreakVertex {
        corner: Vec2::new(along, across),
        data,
    })
}

/// A streak of light from `tail` to `head`, in world space.
#[derive(Clone, Copy)]
pub struct Streak {
    pub tail: Vec3,
    pub head: Vec3,
    /// The core's width, relative to the standard 2 px.
    pub width: f32,
    /// The colour of the body; the head is hotter.
    pub colour: Vec3,
    /// How quickly brightness falls from the head to the tail: 0 for an even beam.
    pub falloff: f32,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct StreakVaryings {
    /// Distance from the axis, in core deviations.
    across: f32,
    /// 0 at the tail, 1 at the head, beyond them over the end caps.
    pub along: f32,
    colour: Vec3,
    /// Falloff exponent, the head's length as a fraction of the streak, and the length of an end
    /// cap as a fraction of the streak.
    shape: Vec3,
    /// The height above the centre of a rolling tear (see [`torn`]), or [`NO_TEAR`] for a streak
    /// nothing tears. A streak a tear crosses sets it after [`expand`].
    pub above_tear: f32,
}

/// An [`above_tear`](StreakVaryings::above_tear) far enough from any tear for [`torn`] to leave
/// the light exactly as it is.
pub const NO_TEAR: f32 = 10.0;

/// How a rolling tear scales light `above` its centre, in heights of the projection's frustum: it
/// dims a broad band and brightens a thin line through its middle (widths and strengths set by
/// eye, provisional).
#[inline(always)]
pub fn torn(above: F) -> F {
    let squared = above * above;
    s(1.0) - (-squared * s(250.0)).exp() * s(0.4) + (-squared * s(5600.0)).exp() * s(0.5)
}

/// Expands `streak` into the screen-facing quad's `corner`. A streak that is dark or has no
/// length goes off screen.
pub fn expand(view: &View, streak: &Streak, corner: Vec2) -> (Vec4, StreakVaryings) {
    let axis = streak.head - streak.tail;
    let length = axis.length();
    if length < 1e-4 || streak.colour.dot(Vec3::ONE) < 1e-4 {
        return hidden();
    }
    let direction = axis / length;
    let pixel = view.tan_half_fov * 2.0 / 720.0 * CORE_SIGMA * streak.width;
    let middle = (streak.tail + streak.head) * 0.5;
    let sigma = pixel * (middle - view.eye).length();
    let cap = CAP * sigma / length;
    let along = -cap + (1.0 + 2.0 * cap) * corner.x;
    let centre = streak.tail + axis * along;
    let to_centre = centre - view.eye;
    let sideways = direction.cross(to_centre).normalise();
    let half = REACH * pixel * to_centre.length();
    let world = centre + sideways * (corner.y * half);
    (
        view.view_projection * world.to_point4(),
        StreakVaryings {
            across: corner.y * REACH,
            along,
            colour: streak.colour,
            shape: Vec3::new(streak.falloff, (8.0 * sigma / length).min(0.4), cap),
            above_tear: NO_TEAR,
        },
    )
}

/// A streak that isn't there: a point outside the view.
pub fn hidden() -> (Vec4, StreakVaryings) {
    (
        Vec4::new(3.0, 3.0, 0.5, 1.0),
        StreakVaryings {
            across: 0.0,
            along: 0.0,
            colour: Vec3::ZERO,
            shape: Vec3::ONE,
            above_tear: NO_TEAR,
        },
    )
}

pub struct StreakShader;

impl FragmentShaderSimd<StreakVaryings> for StreakShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: StreakVaryingsSimd, _: &()) -> ColourSimd {
        let colour = v3(v.colour);
        let [falloff, head, cap] = v.shape;
        let across = v.across;
        let core = (-(across * across) * s(0.5)).exp();
        let glow =
            (-across.abs() * s(0.45)).exp() * (s(1.0) - smoothstep(4.0, REACH, across.abs()));
        let before = clamp01(-v.along / cap);
        let beyond = clamp01((v.along - s(1.0)) / cap);
        let ends = (s(1.0) - before) * (s(1.0) - beyond);
        let ends = ends * ends;
        let body = (falloff * clamp01(v.along).fast_max(s(1e-3)).ln()).exp();
        let nearness = (s(1.0) - clamp01(v.along)) / head;
        let hot = (-(nearness * nearness)).exp() * core;
        let brightness = colour
            .x
            .fast_max(colour.y)
            .fast_max(colour.z)
            .fast_min(s(2.0));
        let tint = mix3(colour, rgb(1.0, 1.0, 1.0), s(0.7));
        let light = colour * ((core + glow * s(0.3)) * body * ends)
            + tint * (hot * ends * brightness * s(1.3));
        to_colour(light * torn(v.above_tear), s(0.0))
    }
}
