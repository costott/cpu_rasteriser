//! pastel-vapour: a vaporwave afternoon. A giant sliced sun sinking behind soft clouds, a glossy
//! checkerboard plain running to the horizon that reflects it, palm silhouettes either side and
//! drifting sparkle.
//!
//! The backdrop is one full-screen shader. The sky is evaluated on the plane at unit distance
//! ahead of the camera (`sx`, `sy` below), so the floor's reflection is the same function of the
//! mirrored ray.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, fract, mix3, rgb, s, select, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVertexShader, Motion, Sway, ToonShader, ToonUniforms, pipeline,
};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::{expand, hdr_of_displayed};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::{Scene, Stage};

/// The height of the checkerboard plain.
const FLOOR: f32 = -3.0;
/// The side of a checker square, and how fast the plain slides towards the viewer.
const SQUARE: f32 = 2.2;
const SLIDE: f32 = 1.6;
/// The sun's centre on the sky plane and its radius.
const SUN: (f32, f32, f32) = (0.0, 0.12, 0.21);
/// The direction towards the light that casts the chip's shadow on the plain.
const LIGHT: Vec3 = Vec3 {
    x: 0.0,
    y: 0.35,
    z: -0.94,
};

const HORIZON: (f32, f32, f32) = (1.0, 0.74, 0.82);
const LILAC: (f32, f32, f32) = (0.78, 0.65, 0.95);
const ZENITH: (f32, f32, f32) = (0.36, 0.78, 0.92);
const PINK: (f32, f32, f32) = (1.0, 0.56, 0.78);
const AQUA: (f32, f32, f32) = (0.46, 0.88, 0.93);
const SUN_TOP: (f32, f32, f32) = (1.0, 0.93, 0.66);
const SUN_BOTTOM: (f32, f32, f32) = (1.0, 0.4, 0.62);
const PLUM: Vec3 = Vec3 {
    x: 0.12,
    y: 0.05,
    z: 0.24,
};

#[inline(always)]
fn ink(colour: (f32, f32, f32)) -> Vec3Simd {
    rgb(colour.0, colour.1, colour.2)
}

#[inline(always)]
fn coverage(d: F, pixel: F) -> F {
    clamp01(s(0.5) - d / pixel)
}

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Vapour;

impl FragmentShaderSimd<ScreenVaryings> for Vapour {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let pixel = s(view.pixel_angle);
        let ahead = (-ray.z).fast_max(s(0.05));
        let (sx, sy) = (ray.x / ahead, ray.y / ahead);

        let below = ray.y.simd_lt(s(-0.001));
        let t = (s(FLOOR - view.eye.y) / ray.y.fast_min(s(-0.001))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);

        let sky_colour = sky(sx, sy, pixel, u.time);
        let across = pixel * t;
        let along = across / ray.y.abs().fast_max(s(0.02));
        let board = checker(wx, wz - s(SLIDE * u.time), across, along);
        let mut floor = mix3(ink(AQUA), ink(PINK), board);
        let mirrored = sky(sx, -sy, pixel, u.time);
        let grazing = (s(1.0) - ray.y.abs() * s(3.0)).fast_max(s(0.0));
        floor = mix3(floor, mirrored, s(0.18) + grazing * grazing * s(0.4));
        let shadow = chip_shadow(view, LIGHT, Vec3Simd::new(wx, s(FLOOR), wz));
        floor = mix3(floor, rgb(0.42, 0.28, 0.62), shadow * s(0.45));
        floor = mix3(floor, ink(HORIZON), clamp01(s(1.0) - (-t * s(0.028)).exp()));

        let colour = mix3(sky_colour, floor, select(below, s(1.0), s(0.0)));
        let distance = select(below, t, s(SKY_DISTANCE));
        to_colour(expand(colour), distance)
    }
}

/// A checkerboard, 0 or 1, box-filtered over the pixel's footprint (`width` across, `depth` along)
/// so it settles to an even grey towards the horizon.
#[inline(always)]
fn checker(x: F, z: F, width: F, depth: F) -> F {
    let wave = |p: F, w: F| {
        let w = w / s(SQUARE) + s(0.001);
        let q = p / s(SQUARE);
        ((fract((q - w * s(0.5)) * s(0.5)) - s(0.5)).abs()
            - (fract((q + w * s(0.5)) * s(0.5)) - s(0.5)).abs())
            * s(2.0)
            / w
    };
    s(0.5) - s(0.5) * wave(x, width) * wave(z, depth)
}

/// The sky at (`sx`, `sy`) on the plane one unit ahead: a gradient, the sliced sun and clouds.
#[inline(always)]
fn sky(sx: F, sy: F, pixel: F, time: f32) -> Vec3Simd {
    let low = mix3(ink(HORIZON), ink(LILAC), smoothstep(0.0, 0.4, sy));
    let mut colour = mix3(low, ink(ZENITH), smoothstep(0.25, 0.8, sy));

    let (cx, cy, radius) = SUN;
    let (dx, dy) = (sx - s(cx), sy - s(cy));
    let r = (dx * dx + dy * dy).sqrt();
    let halo = (-(r - s(radius)).fast_max(s(0.0)) * s(6.0)).exp() * s(0.4);
    colour = colour + rgb(1.0, 0.5, 0.7) * (halo * s(0.35));

    let height = dy / s(radius);
    let body = mix3(ink(SUN_BOTTOM), ink(SUN_TOP), smoothstep(-0.8, 0.7, height));
    let period = s(0.17);
    let phase = fract(height / period - s(time * 0.06));
    let gap = ((s(0.25) - height) * s(0.7))
        .fast_max(s(0.0))
        .fast_min(s(0.6));
    let edge = pixel / (s(radius) * period);
    let slice = smoothstep(0.0, 1.0, clamp01((phase - gap) / edge + s(0.5)));
    let sun = coverage(r - s(radius), pixel) * select(height.simd_gt(s(0.25)), s(1.0), slice);
    colour = mix3(colour, body, sun);

    for (n, (x, y, width, speed)) in [
        (-0.62, 0.2, 0.32, 0.012),
        (0.55, 0.3, 0.38, -0.009),
        (0.05, 0.07, 0.3, 0.007),
        (-0.3, 0.42, 0.26, 0.01),
    ]
    .into_iter()
    .enumerate()
    {
        let drifted =
            fract((s(x + time * speed + n as f32 * 0.31) + s(1.5)) / s(3.0)) * s(3.0) - s(1.5);
        colour = cloud(colour, sx - drifted, sy - s(y), width, pixel);
    }
    colour
}

/// A flat-bottomed cloud of three overlapping puffs centred at the origin of (`x`, `y`).
#[inline(always)]
fn cloud(colour: Vec3Simd, x: F, y: F, width: f32, pixel: F) -> Vec3Simd {
    let puff = |cx: f32, cy: f32, rx: f32, ry: f32| {
        let (px, py) = (
            (x - s(cx * width)) / s(rx * width),
            (y - s(cy * width)) / s(ry * width),
        );
        ((px * px + py * py).sqrt() - s(1.0)) * s(rx.min(ry) * width)
    };
    let d = puff(-0.45, 0.1, 0.5, 0.45)
        .fast_min(puff(0.0, 0.3, 0.55, 0.6))
        .fast_min(puff(0.5, 0.1, 0.5, 0.42))
        .fast_max(-y - s(0.0));
    let shade = clamp01(y / s(width * 0.8));
    let body = mix3(rgb(0.86, 0.66, 0.96), rgb(1.0, 0.88, 0.94), shade.sqrt());
    mix3(colour, body, coverage(d, pixel) * s(0.92))
}

/// A palm: a leaning trunk crowned with drooping fronds, standing at `base`.
fn palm(base: Vec3, height: f32, lean: f32, turn: f32) -> Shape {
    let spine = |u: f32| base + Vec3::new(lean * u * u * height * 0.3, u * height, 0.0);
    let mut shape = Shape::new();
    for i in 0..8 {
        let (a, b) = (i as f32 / 8.0, (i + 1) as f32 / 8.0);
        let radius = |u: f32| 0.2 - 0.09 * u;
        shape.append(&Shape::limb(
            spine(a),
            spine(b),
            radius(a),
            radius(b),
            6,
            PLUM,
        ));
    }
    let crown = spine(1.0);
    for k in 0..9 {
        let angle = turn + k as f32 * std::f32::consts::TAU / 9.0;
        let reach = 2.6 + 0.4 * (k as f32 * 2.3).sin();
        let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
        let points: Vec<Vec3> = (0..7)
            .map(|i| {
                let t = i as f32 / 6.0;
                crown
                    + direction * (reach * t)
                    + Vec3::new(0.0, reach * (0.55 * t - 1.0 * t * t), 0.0)
            })
            .collect();
        let widths: Vec<f32> = (0..7)
            .map(|i| 0.7 * (1.0 - i as f32 / 6.0).powf(0.7) + 0.02)
            .collect();
        let sideways = Vec3::new(-direction.z, 0.0, direction.x);
        for across in [
            sideways,
            (sideways * 0.5 + Vec3::new(0.0, 0.85, 0.0)).normalise(),
        ] {
            shape.append(&Shape::ribbon(&points, &widths, across, PLUM * 0.8));
        }
    }
    shape.sway_by_height(base.y, base.y + height, 1.4)
}

fn palms() -> Shape {
    let mut rng = Rng::new(33);
    let mut all = Shape::new();
    for (x, z, height, lean) in [
        (-6.0, -4.0, 8.0, -0.9),
        (-9.0, -9.0, 7.0, 0.7),
        (6.3, -3.0, 7.4, 0.8),
        (9.5, -8.0, 6.4, -0.6),
    ] {
        let turn = rng.range(0.0, 1.0);
        all.append(&palm(Vec3::new(x, FLOOR, z), height, lean, turn));
    }
    all
}

pub struct PastelVapour {
    look: Look,
    palms: Shape,
    palm_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Vapour>,
    sparkle: Particles,
}

impl PastelVapour {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.86, 0.7, 0.92),
                die: Vec3::new(0.14, 0.2, 0.45),
                gold: Vec3::new(0.92, 0.96, 1.0),
                streams: Vec3::new(0.35, 0.95, 1.0),
                key: Vec3::new(-0.4, 0.5, 0.75).normalise(),
                key_colour: Vec3::new(2.6, 2.1, 2.4),
                sky: Vec3::new(0.3, 0.7, 0.85),
                ground: Vec3::new(0.9, 0.35, 0.6),
                softbox: Vec3::new(1.0, 0.85, 0.95),
            },
            palms: palms(),
            palm_pipeline: pipeline(ToonShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, Vapour)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            sparkle: Particles::new(70, 33),
        }
    }
}

impl Stage for PastelVapour {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::SawLead],
            treatment: Treatment::Vast,
            ambience: vec![Ambience::once(
                Source::Synth(Accent::SoftSweep),
                0.0,
                0.45,
                0.12,
                -0.3,
            )],
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        scene.chip.draw(pass, view, &self.look);
        let lighting = Lighting {
            sun: Vec3::new(-0.4, 0.55, 0.7).normalise(),
            sun_colour: Vec3::new(1.5, 0.9, 1.2),
            sky: Vec3::new(0.5, 0.4, 0.75),
            ground: Vec3::new(0.5, 0.2, 0.4),
            bands: 3.0,
            rim: Vec3::new(0.9, 0.45, 0.7),
            fog: Fog {
                colour: hdr_of_displayed(Vec3::new(HORIZON.0, HORIZON.1, HORIZON.2)),
                density: 0.007,
                falloff: 0.0,
                base: FLOOR,
            },
        };
        self.palms.draw(
            pass,
            &self.palm_pipeline,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Sway(Sway {
                    amplitude: 0.2,
                    speed: 1.3,
                    wind: Vec2::new(1.0, 0.2),
                }),
            },
            ToonUniforms {
                view: view.clone(),
                lighting,
            },
        );
        let uniforms = Uniforms {
            view: view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
        self.sparkle.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.0, 0.0),
                half_extent: Vec3::new(9.0, 5.0, 4.0),
                velocity: Vec3::new(0.15, 0.35, 0.0),
                wander: 0.4,
                size: (0.03, 0.09),
                colour: Vec3::new(1.2, 0.95, 1.2),
                twinkle: 2.5,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.3,
            vignette: 0.18,
            grain: 0.02,
            aberration: 0.004,
            desaturate: 0.0,
        }
    }
}
