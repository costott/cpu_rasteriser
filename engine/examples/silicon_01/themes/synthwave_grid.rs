//! synthwave-grid: a neon night drive. A striped sun sinking behind two ranges of faceted
//! mountains outlined in neon, stars overhead, and a magenta grid racing towards the horizon,
//! with the chip in black glass edged in magenta, under bloom and faint scanlines.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, fract, hash2, mix3, rgb, s, select, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::post::{Chain, Crt, Edges, Effect, Noise, expand};
use crate::themes::kit::sky::stars;
use crate::themes::{Scene, Stage};

/// The height of the grid below the chip, the distance between its lines and how fast it slides
/// towards the viewer.
const FLOOR: f32 = -3.0;
const GRID: f32 = 2.0;
const SLIDE: f32 = 5.0;
/// The distances over which the lines running across the view, then those running away from
/// it, give way to a steady glow. Beyond the far distance neighbouring lines are under about
/// eight pixels apart and would shimmer.
const ACROSS_FADE: (f32, f32) = (14.0, 30.0);
const AWAY_FADE: (f32, f32) = (30.0, 90.0);
/// The glow the far grid settles to: a line's mean coverage.
const AVERAGE: f32 = 0.02;
/// The sun's centre on the sky plane one unit ahead, and its radius.
const SUN: (f32, f32, f32) = (0.0, 0.14, 0.22);

const MAGENTA: Vec3 = Vec3 {
    x: 1.0,
    y: 0.1,
    z: 0.62,
};
const CYAN: Vec3 = Vec3 {
    x: 0.1,
    y: 0.85,
    z: 1.0,
};

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Night;

impl FragmentShaderSimd<ScreenVaryings> for Night {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let pixel = s(view.line_angle);
        let ahead = (-ray.z).fast_max(s(0.05));
        let (sx, sy) = (ray.x / ahead, ray.y / ahead);
        let time = u.time;

        let mut colour = sky(sy, ray.x, ray.y, ray.z, time);
        colour = sun(colour, sx, sy, pixel, time);
        colour = mountains(colour, sx, sy, pixel);

        let below = ray.y.simd_lt(s(-0.001));
        let t = (s(FLOOR - view.eye.y) / ray.y.fast_min(s(-0.001))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let ground = floor(wx, wz - s(SLIDE * time), t, pixel, ray.y.abs(), sx);
        let colour = mix3(colour, ground, select(below, s(1.0), s(0.0)));
        to_colour(colour, s(SKY_DISTANCE))
    }
}

/// HDR colour from a displayed one scaled by `gain`, for neon that blooms.
#[inline(always)]
fn glow(colour: Vec3, gain: f32) -> Vec3Simd {
    expand(splat3(colour)) * s(gain)
}

#[inline(always)]
fn coverage(d: F, pixel: F) -> F {
    clamp01(s(0.5) - d / pixel)
}

/// A night sky running from indigo overhead to a hot violet horizon, with twinkling stars.
#[inline(always)]
fn sky(sy: F, x: F, y: F, z: F, time: f32) -> Vec3Simd {
    let low = smoothstep(0.0, 0.55, sy);
    let displayed = mix3(rgb(0.5, 0.06, 0.42), rgb(0.07, 0.03, 0.2), low.sqrt());
    let colour = expand(displayed * s(0.7));
    let field = stars(x, y, z, 0.8, s(time)) * smoothstep(0.08, 0.4, y);
    colour + rgb(0.9, 0.85, 1.0) * field
}

/// The sun: a disc from yellow to hot pink, its lower half cut by slits that widen towards the
/// bottom and drift down.
#[inline(always)]
fn sun(colour: Vec3Simd, sx: F, sy: F, pixel: F, time: f32) -> Vec3Simd {
    let (cx, cy, radius) = SUN;
    let (dx, dy) = (sx - s(cx), sy - s(cy));
    let r = (dx * dx + dy * dy).sqrt();
    let halo = (-(r - s(radius)).fast_max(s(0.0)) * s(7.0)).exp();
    let mut result = colour + rgb(1.0, 0.12, 0.45) * (halo * s(0.5));

    let height = dy / s(radius);
    let body = mix3(
        rgb(1.0, 0.1, 0.45),
        rgb(1.0, 0.85, 0.25),
        smoothstep(-0.9, 0.6, height),
    );
    let period = s(0.2);
    let phase = fract(height / period + s(time * 0.12));
    let gap = ((s(0.35) - height) * s(0.5))
        .fast_max(s(0.0))
        .fast_min(s(0.55));
    let edge = pixel / (s(radius) * period);
    let slice = clamp01((phase - gap) / edge + s(0.5));
    let cut = select(height.simd_gt(s(0.35)), s(1.0), slice);
    let disc = coverage(r - s(radius), pixel) * cut;
    result = mix3(result, expand(body) * s(1.5), disc);
    result
}

/// Two ranges of faceted peaks, dark glass with a neon ridge line, the far one magenta and the
/// near one cyan.
#[inline(always)]
fn mountains(colour: Vec3Simd, sx: F, sy: F, pixel: F) -> Vec3Simd {
    let mut result = colour;
    for (layer, (frequency, height, line)) in [(3.4f32, 0.1f32, MAGENTA), (5.8, 0.07, CYAN)]
        .into_iter()
        .enumerate()
    {
        let along = sx * s(frequency) + s(layer as f32 * 5.3);
        let cell = along.floor();
        let peak = s(height) * (s(0.35) + hash2(cell, s(layer as f32 + 2.0)));
        let ridge = peak * (s(1.0) - (fract(along) * s(2.0) - s(1.0)).abs());
        let slope = peak * s(2.0 * frequency);
        let distance = (sy - ridge) / (s(1.0) + slope * slope).sqrt();
        let inside = coverage(distance, pixel);
        let depth = clamp01((ridge - sy) / s(height * 1.4));
        let fill = mix3(rgb(0.02, 0.0, 0.05), splat3(line * 0.12), depth.sqrt());
        let rim = coverage(distance.abs() - pixel * s(0.9), pixel);
        let neon = glow(line, 1.6) * rim
            + glow(line, 0.3) * (-distance.abs() * s(40.0)).exp() * clamp01(s(1.0) - inside);
        let front = mix3(result, fill, inside);
        result = front + neon * select(sy.simd_gt(s(-0.002)), s(1.0), s(0.0));
    }
    result
}

/// The grid: glowing magenta lines on dark glass that fade into the horizon haze, with the sun's
/// reflection pooled beneath it.
#[inline(always)]
fn floor(wx: F, wz: F, t: F, pixel: F, slant: F, sx: F) -> Vec3Simd {
    let sharp = |(from, to): (f32, f32)| s(1.0) - smoothstep(from, to, t);
    let (rows, columns) = (sharp(ACROSS_FADE), sharp(AWAY_FADE));
    let across = pixel * t;
    let along = across / slant.fast_max(s(0.02));
    let line = |p: F, size: F, fade: F| {
        let d = (fract(p * s(1.0 / GRID) + s(0.5)) - s(0.5)).abs() * s(GRID);
        let core = clamp01(s(1.5) - d / size);
        let halo = (-d / (size * s(3.0))).exp() * s(0.25);
        (core + halo) * fade + (s(1.0) - fade) * s(AVERAGE)
    };
    let lines = line(wx, across, columns).fast_max(line(wz, along, rows));
    let haze = (-t * s(0.03)).exp();
    let base = mix3(rgb(0.1, 0.01, 0.1), rgb(0.01, 0.0, 0.04), haze);
    let horizon = glow(MAGENTA, 0.12) * ((s(1.0) - rows) * (-slant * s(60.0)).exp());
    let pool = (-(sx * sx) * s(30.0)).exp() * haze * s(0.5);
    let ground =
        expand(base * s(0.6)) + glow(MAGENTA, 1.8) * (lines * (haze * s(0.8) + s(0.2))) + horizon;
    ground + rgb(1.0, 0.25, 0.4) * pool * rows * smoothstep(0.0, 1.0, fract(wz * s(0.35)))
}

pub struct SynthwaveGrid {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Night>,
    chain: Chain,
}

impl SynthwaveGrid {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.03, 0.02, 0.07),
                die: Vec3::new(0.02, 0.01, 0.06),
                gold: Vec3::new(0.1, 0.8, 1.0) * 0.7,
                streams: Vec3::new(1.0, 0.15, 0.75),
                key: Vec3::new(-0.5, 0.45, 0.7).normalise(),
                key_colour: Vec3::new(1.6, 0.3, 1.3),
                sky: Vec3::new(0.35, 0.08, 0.5),
                ground: Vec3::new(0.7, 0.05, 0.4),
                softbox: Vec3::ZERO,
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Night)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            chain: Chain::new(vec![
                Effect::Edges(Edges {
                    ink: MAGENTA * 3.0,
                    width: 1.2,
                    ..Edges::default()
                }),
                Effect::Crt(Crt {
                    curvature: 0.0,
                    scanlines: 0.28,
                    pitch: 3.0,
                    mask: 0.0,
                    convergence: 0.7,
                    flicker: 0.02,
                }),
                Effect::Noise(Noise {
                    amount: 0.05,
                    floor: 0.0,
                    cell: 1.0,
                    chroma: 0.0,
                }),
            ]),
        }
    }
}

impl Stage for SynthwaveGrid {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::SawLead],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::EngineSweep),
                0.0,
                1.3,
                0.1,
                0.0,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        let uniforms = Uniforms {
            view: scene.view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        Some(self.chain.run(context, scene, image))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 1.0,
            vignette: 0.4,
            grain: 0.012,
            aberration: 0.004,
            desaturate: 0.0,
        }
    }
}
