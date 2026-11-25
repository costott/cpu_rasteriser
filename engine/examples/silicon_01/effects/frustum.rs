//! The projection frustum: thin rays shoot up from the die's edges, a bright rim draws itself
//! across the top, and a faint fill and scan rings give the volume between them. The whole
//! projection, the hologram in it included, flickers as one (see [`Flicker`]).

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::director::{Phase, ramp, window};
use crate::effects::streak::{Streak, StreakVaryings, StreakVertex, expand, streak_corners, torn};
use crate::effects::{DIE_HALF, DIE_TOP, GREEN, GREEN_BRIGHT, quad_model, random};
use crate::shaders::{View, ViewRef};
use crate::simd::{fract, hash2, hash2_scalar, s, smoothstep, to_colour, v3};

/// The frustum's height above the die, and half the width of its top.
pub const HEIGHT: f32 = 5.2;
const TOP_HALF: f32 = 2.5;

/// When the rays leave the die, how long they take to reach the top, when the rim starts to
/// draw, and when the hologram is formed.
const HOLOGRAM: f32 = Phase::Hologram.start();
const RAYS: f32 = HOLOGRAM + 0.4;
const RISE: f32 = 1.1;
const RIM: f32 = HOLOGRAM + 1.4;

/// The chance that a 1/24 s frame of the projection dips in brightness.
const DIP_CHANCE: f32 = 0.1;

/// The projection's flicker at one moment: a steady 29 Hz shimmer, broken by brief dips to half or
/// two thirds that also jolt the image, with a tear that rolls up it.
#[derive(Clone, Copy)]
pub struct Flicker {
    /// What the projection's light is scaled by.
    pub brightness: f32,
    /// How far a dip jolts the image, in clip space per unit of `w`.
    jolt: Vec2,
    /// The height of the tear's centre, 0 at the die and 1 at the frustum's top.
    pub tear: f32,
}

impl Flicker {
    /// The flicker in `view`'s frame.
    pub fn new(view: &View) -> Self {
        let time = view.frame.time;
        let tear = tear(time);
        let brightness = flicker_level(time);
        if !dips(time) {
            return Self {
                brightness,
                jolt: Vec2::new(0.0, 0.0),
                tear,
            };
        }
        let frame = flicker_frame(time);
        // One to two pixels either way.
        let jolt = |seed: f32| {
            let size = 1.0 + hash2_scalar(frame, seed);
            if hash2_scalar(frame, seed + 1.0) < 0.5 {
                -size
            } else {
                size
            }
        };
        let pixel = view.pixel_angle / view.tan_half_fov;
        Self {
            brightness,
            jolt: Vec2::new(jolt(3.1) * pixel / view.aspect, jolt(8.9) * pixel),
            tear,
        }
    }

    /// `clip` moved by the jolt.
    pub fn jolted(&self, mut clip: Vec4) -> Vec4 {
        clip.x += self.jolt.x * clip.w;
        clip.y += self.jolt.y * clip.w;
        clip
    }
}

/// The height of the tear that rolls up the projection at `time`, 0 at the die and 1 at the
/// frustum's top; it passes a little beyond both.
pub fn tear(time: f32) -> f32 {
    -0.1 + 1.2 * (time * 0.37).fract()
}

/// The 1/24 s frame of the projection that `time` is in.
fn flicker_frame(time: f32) -> f32 {
    (time * 24.0).floor()
}

/// Whether the projection dips in brightness at `time`.
pub fn dips(time: f32) -> bool {
    hash2_scalar(flicker_frame(time), 11.7) < DIP_CHANCE
}

/// What the projection's light is scaled by at `time`: its shimmer, and a dip's depth in one.
pub fn flicker_level(time: f32) -> f32 {
    let shimmer = 0.9 + 0.1 * (time * 29.0).sin();
    if dips(time) {
        shimmer * (0.5 + 0.2 * hash2_scalar(flicker_frame(time), 5.3))
    } else {
        shimmer
    }
}

/// [`expand`] for one of the projection's streaks, so it flickers with the rest: `levels` are the
/// heights of its tail and head, 0 at the die and 1 at the frustum's top.
fn expand_projected(
    view: &View,
    mut streak: Streak,
    levels: [f32; 2],
    corner: Vec2,
) -> (Vec4, StreakVaryings) {
    let flicker = Flicker::new(view);
    streak.colour = streak.colour * flicker.brightness;
    let (clip, mut varyings) = expand(view, &streak, corner);
    varyings.above_tear = levels[0] + (levels[1] - levels[0]) * varyings.along - flicker.tear;
    (flicker.jolted(clip), varyings)
}

/// The corner `index` (0 to 3, going round) of the frustum's cross-section at `level`, 0 at the
/// die and 1 at the top, in the chip's space.
pub fn corner(index: usize, level: f32) -> Vec3 {
    let (x, z) = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)][index % 4];
    let half = DIE_HALF + (TOP_HALF - DIE_HALF) * level;
    Vec3::new(x * half, DIE_TOP + HEIGHT * level, z * half)
}

/// How far up the rays' heads have shot, 0 to 1: they retreat into the die as the hologram closes.
pub fn rise(time: f32) -> f32 {
    ramp(time, RAYS, RAYS + RISE) * (1.0 - ramp(time, HOLOGRAM + 10.9, HOLOGRAM + 12.3))
}

/// How much of the top rim has been drawn, 0 to 1.
pub fn rim(time: f32) -> f32 {
    ramp(time, RIM, RIM + 0.8) * (1.0 - ramp(time, HOLOGRAM + 10.3, HOLOGRAM + 11.3))
}

/// How present the frustum's volume is, 0 to 1.
pub fn volume(time: f32) -> f32 {
    window(time, RIM + 0.3, HOLOGRAM + 11.8, 1.2)
}

/// A ray along `from` to `to`, shot from its start after `delay` seconds.
#[derive(Clone, Copy)]
pub struct Ray {
    from: Vec3,
    to: Vec3,
    delay: f32,
    brightness: f32,
    width: f32,
    /// Whether it draws along the rim rather than shooting up.
    rim: bool,
}

pub struct RayVertexShader;

impl VertexShader for RayVertexShader {
    type Vertex = StreakVertex<Ray>;
    type Uniforms = ViewRef;
    type Varyings = StreakVaryings;

    fn shade(&self, vertex: StreakVertex<Ray>, view: &ViewRef) -> (Vec4, StreakVaryings) {
        let frame = &view.frame;
        let ray = &vertex.data;
        let shot = if ray.rim {
            rim(frame.time)
        } else {
            rise(frame.time - ray.delay)
        };
        let settle = ramp(frame.time - ray.delay, RAYS + 0.7, RAYS + 1.6);
        let comet = (shot - 0.3).max(0.0);
        let tail = if ray.rim { 0.0 } else { comet * (1.0 - settle) };
        let at = |t: f32| (frame.chip * (ray.from + (ray.to - ray.from) * t).to_point4()).xyz();
        let shimmer = 0.92 + 0.08 * (frame.time * 31.0 + ray.delay * 40.0).sin();
        let streak = Streak {
            tail: at(tail),
            head: at(shot),
            width: ray.width,
            colour: GREEN_BRIGHT * (ray.brightness * shimmer),
            falloff: 0.25,
        };
        let level = |t: f32| (ray.from.y + (ray.to.y - ray.from.y) * t - DIE_TOP) / HEIGHT;
        expand_projected(view, streak, [level(tail), level(shot)], vertex.corner)
    }
}

/// The rays: the four edges, a few thin ones across each face, and the rim.
pub fn rays() -> Vec<[StreakVertex<Ray>; 4]> {
    let mut rays = Vec::new();
    for i in 0..4 {
        rays.push(Ray {
            from: corner(i, 0.0),
            to: corner(i, 1.0),
            delay: 0.0,
            brightness: 1.1,
            width: 1.0,
            rim: false,
        });
        rays.push(Ray {
            from: corner(i, 1.0),
            to: corner(i + 1, 1.0),
            delay: 0.0,
            brightness: 1.1,
            width: 1.0,
            rim: true,
        });
        for k in 1..5 {
            let u = k as f32 / 5.0 + (random(i as f32, k as f32) - 0.5) * 0.06;
            let along =
                |level: f32| corner(i, level) + (corner(i + 1, level) - corner(i, level)) * u;
            rays.push(Ray {
                from: along(0.0),
                to: along(1.0),
                delay: random(i as f32 + 3.0, k as f32) * 0.5,
                brightness: 0.2,
                width: 0.8,
                rim: false,
            });
        }
    }
    rays.into_iter().map(streak_corners).collect()
}

/// A scan ring: a cross-section of the frustum that climbs it, drawn as one streak per side.
#[derive(Clone, Copy)]
pub struct Scan {
    side: usize,
    phase: f32,
}

pub struct ScanVertexShader;

impl VertexShader for ScanVertexShader {
    type Vertex = StreakVertex<Scan>;
    type Uniforms = ViewRef;
    type Varyings = StreakVaryings;

    fn shade(&self, vertex: StreakVertex<Scan>, view: &ViewRef) -> (Vec4, StreakVaryings) {
        let frame = &view.frame;
        let scan = &vertex.data;
        let level = (frame.time * 0.11 + scan.phase).fract();
        let presence = volume(frame.time) * (level * std::f32::consts::PI).sin();
        let at = |i: usize| (frame.chip * corner(i, level).to_point4()).xyz();
        let streak = Streak {
            tail: at(scan.side),
            head: at(scan.side + 1),
            width: 0.8,
            colour: GREEN * (0.3 * presence),
            falloff: 0.0,
        };
        expand_projected(view, streak, [level, level], vertex.corner)
    }
}

pub fn scans() -> Vec<[StreakVertex<Scan>; 4]> {
    (0..2)
        .flat_map(|ring| {
            (0..4).map(move |side| Scan {
                side,
                phase: ring as f32 / 2.0,
            })
        })
        .map(streak_corners)
        .collect()
}

/// One corner of a face of the frustum: `uv.x` across the face, `uv.y` up it.
#[derive(Clone, Copy)]
pub struct FaceVertex {
    side: usize,
    uv: Vec2,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct FaceVaryings {
    uv: Vec2,
    /// Which face, for varying the pattern between them.
    side: f32,
}

pub struct FaceVertexShader;

impl VertexShader for FaceVertexShader {
    type Vertex = FaceVertex;
    type Uniforms = ViewRef;
    type Varyings = FaceVaryings;

    fn shade(&self, vertex: FaceVertex, view: &ViewRef) -> (Vec4, FaceVaryings) {
        let frame = &view.frame;
        let level = vertex.uv.y * rise(frame.time);
        let (a, b) = (corner(vertex.side, level), corner(vertex.side + 1, level));
        let world = frame.chip * (a + (b - a) * vertex.uv.x).to_point4();
        (
            Flicker::new(view).jolted(view.view_projection * world),
            FaceVaryings {
                uv: Vec2::new(vertex.uv.x, level),
                side: vertex.side as f32,
            },
        )
    }
}

pub struct FaceShader;

impl FragmentShaderSimd<FaceVaryings> for FaceShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: FaceVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let time = view.frame.time;
        let [u, level] = v.uv;
        let presence = s(volume(time));
        let fade = (s(1.0) - level).fast_max(s(0.0));
        let glow = fade * fade * s(0.08);
        let edge = smoothstep(0.88, 1.0, (u * s(2.0) - s(1.0)).abs()) * s(0.08) * fade;
        let column = (u * s(36.0) + v.side * s(7.0)).floor();
        let rate = hash2(column, v.side) * s(0.5) + s(0.2);
        let drift = fract(level * s(1.5) - s(time) * rate + hash2(column, s(3.0)));
        let streaks = smoothstep(0.9, 1.0, drift) * s(0.05) * fade;
        let bands = smoothstep(0.94, 1.0, fract(level * s(22.0) - s(time * 0.9))) * s(0.035);
        let flicker = Flicker::new(view);
        let light = (glow + edge + streaks + bands)
            * presence
            * s(flicker.brightness)
            * torn(level - s(flicker.tear));
        to_colour(
            v3([
                light * s(GREEN.x * 1.2),
                light * s(GREEN.y * 1.2),
                light * s(GREEN.z * 1.2),
            ]),
            s(0.0),
        )
    }
}

pub fn faces() -> Model<FaceVertex> {
    quad_model((0..4).map(|side| {
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(x, y)| FaceVertex {
            side,
            uv: Vec2::new(x, y),
        })
    }))
}

/// The glow where the die throws the frustum: a soft light on the die while the rays leave it.
pub fn die_glow(time: f32) -> f32 {
    window(time, RAYS - 0.2, HOLOGRAM + 12.3, 0.8) * (0.5 + 0.5 * ramp(time, RAYS, RAYS + RISE))
}
