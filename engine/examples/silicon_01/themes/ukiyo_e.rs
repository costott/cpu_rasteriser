//! ukiyo-e: a woodblock print in the manner of Hokusai. A cream paper sky with a vermilion sun and
//! ribbon clouds, a snow-capped cone mountain, a sea of indigo scallops that rocks, a great wave
//! curling over with clawed foam, and small boats, all in a few flat inks with a dark outline.
//!
//! One full-screen shader in screen units (the screen is 2 high, so a pixel is `2 / height`).
//! Every shape is a signed distance function drawn as a flat fill with the outline cut into its
//! edge, the way a keyblock prints it, over paper with visible fibres.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::ViewRef;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::simd::{F, clamp01, fract, hash2, mix3, rgb, s, select, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::noise::{fbm, value};
use crate::themes::kit::post::expand;
use crate::themes::{Scene, Stage};

/// Where the sea meets the mountain's foot: the world's horizon line.
const HORIZON: f32 = -0.5;
/// The distance the whole scene reports as its view distance.
const SCENE_DISTANCE: f32 = 30.0;
/// Where the mountain stands and how high its peak is.
const MOUNTAIN: (f32, f32) = (0.55, 0.3);
/// The great wave's curl: the outer circle's centre and radius, and the inner circle's.
const WAVE_OUTER: (f32, f32, f32) = (-0.5, -0.35, 0.7);
const WAVE_INNER: (f32, f32, f32) = (-0.3, -0.47, 0.52);

const INK: (f32, f32, f32) = (0.04, 0.07, 0.15);
const INDIGO: (f32, f32, f32) = (0.07, 0.17, 0.38);
const BLUE: (f32, f32, f32) = (0.18, 0.4, 0.58);
const PALE: (f32, f32, f32) = (0.72, 0.82, 0.84);
const CREAM: (f32, f32, f32) = (0.93, 0.88, 0.74);
const VERMILION: (f32, f32, f32) = (0.82, 0.22, 0.1);

#[inline(always)]
fn ink(colour: (f32, f32, f32)) -> Vec3Simd {
    rgb(colour.0, colour.1, colour.2)
}

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Print;

impl FragmentShaderSimd<ScreenVaryings> for Print {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let (a, b) = (nx * s(view.aspect), ny);
        let pixel = s(view.pixel_angle / view.tan_half_fov);
        let time = u.time;

        let mut colour = sky(a, b, pixel, time);
        colour = mountain(colour, a, b, pixel);
        colour = sea(colour, a, b, pixel, time);
        colour = great_wave(colour, a, b, pixel, time);
        colour = boats(colour, a, b, pixel, time);
        colour = paper(colour, a, b);
        to_colour(expand(colour), s(SCENE_DISTANCE))
    }
}

/// 1 inside a shape whose signed distance is `d`, fading across one pixel.
#[inline(always)]
fn coverage(d: F, pixel: F) -> F {
    clamp01(s(0.5) - d / pixel)
}

/// Fills a shape with `fill` and prints its outline just inside the edge.
#[inline(always)]
fn print(colour: Vec3Simd, d: F, fill: Vec3Simd, line: f32, pixel: F) -> Vec3Simd {
    let outline = coverage(d + s(line), pixel);
    let inside = mix3(ink(INK), fill, outline);
    mix3(colour, inside, coverage(d, pixel))
}

/// The sky: pale indigo grading to cream at the horizon, a vermilion sun and drifting ribbons of
/// cloud.
#[inline(always)]
fn sky(a: F, b: F, pixel: F, time: f32) -> Vec3Simd {
    let height = smoothstep(HORIZON, 0.9, b);
    let banded = (height * s(5.0)).floor() / s(5.0);
    let mut colour = mix3(ink(CREAM), ink(PALE), banded.sqrt());

    for (n, (y, thickness, length, speed)) in [
        (0.12, 0.018, 0.5, 0.012),
        (0.3, 0.014, 0.4, -0.009),
        (-0.16, 0.012, 0.6, 0.007),
    ]
    .into_iter()
    .enumerate()
    {
        let wrapped = fract((a + s(time * speed + n as f32 * 1.3)) / s(2.6)) * s(2.6) - s(1.3);
        let (dx, dy) = (wrapped.abs() - s(length * 0.5), (b - s(y)).abs());
        let (ox, oy) = (dx.fast_max(s(0.0)), dy - s(thickness));
        let d = (ox * ox + oy.fast_max(s(0.0)) * oy.fast_max(s(0.0))).sqrt() + oy.fast_min(s(0.0));
        let ribbon = coverage(d, pixel);
        colour = mix3(colour, ink(CREAM) * s(1.04), ribbon * s(0.9));
    }

    let (sx, sy) = (a + s(0.12), b - s(0.52));
    let sun = (sx * sx + sy * sy).sqrt() - s(0.11);
    mix3(colour, ink(VERMILION), coverage(sun, pixel))
}

/// The snow-capped cone mountain, its flanks streaked downward and a band of mist at its foot.
#[inline(always)]
fn mountain(colour: Vec3Simd, a: F, b: F, pixel: F) -> Vec3Simd {
    let (centre, top) = MOUNTAIN;
    let across = a - s(centre);
    let reach = clamp01((s(top) - b) / s(0.8));
    let half = s(0.05) + s(0.62) * reach * reach.sqrt();
    let d = (across.abs() - half)
        .fast_max(b - s(top))
        .fast_max(s(HORIZON - 0.1) - b);

    let streak = fract(across * s(12.0) / (s(0.3) + reach)) - s(0.5);
    let tone = smoothstep(0.0, 0.8, reach);
    let flank = mix3(ink(BLUE), ink(INDIGO), tone)
        * (s(1.0) - (streak.abs() * s(0.5)).fast_min(s(0.2)) * s(0.4) * (s(1.0) - reach));
    let snow_line = s(top - 0.2)
        + (fract(across * s(7.0)) - s(0.5)).abs() * s(0.1)
        + (across * s(5.0)).sin() * s(0.015);
    let snowy = coverage(snow_line - b, pixel);
    let flank = mix3(flank, ink(CREAM) * s(1.03), snowy);
    let printed = print(colour, d, flank, 0.004, pixel);

    let mist = smoothstep(HORIZON + 0.2, HORIZON, b) * smoothstep(HORIZON - 0.1, HORIZON - 0.02, b);
    mix3(printed, ink(CREAM), mist * s(0.55))
}

/// The sea: rows of overlapping scallops, wider and taller towards the viewer, rocking.
#[inline(always)]
fn sea(colour: Vec3Simd, a: F, b: F, pixel: F, time: f32) -> Vec3Simd {
    let v = (s(HORIZON) - b).fast_max(s(0.0));
    let rows = (s(1.0) + v * s(7.0)).ln() * s(3.2);
    let row_height = (s(1.0) + v * s(7.0)) / s(22.4);
    let width = row_height * s(2.6);
    let row = rows.floor();
    let rock = (row * s(1.7) + s(time * 0.8)).sin() * s(0.05);
    let drift = select(fract(row * s(0.5)).simd_lt(s(0.5)), s(1.0), s(-1.0));
    let along = a / width + row * s(0.5) + s(time * 0.15) * drift + rock;
    let local = fract(along) * s(2.0) - s(1.0);
    let arch = (s(1.0) - local * local).fast_max(s(0.0)).sqrt() * s(0.42);
    let level = rows - arch;
    let band = level.floor();
    let inner = fract(level);

    let shade = hash2(band, along.floor()) * s(0.35);
    let alternate = select(fract(band * s(0.5)).simd_lt(s(0.5)), s(0.0), s(1.0));
    let base = mix3(ink(INDIGO), ink(BLUE), alternate * s(0.6) + shade * s(0.4));
    let body = base * (s(1.25) - inner * s(0.5));
    let crest_line = s(1.0) - smoothstep(0.0, 0.1, inner.fast_min(s(1.0) - inner) - s(0.04));
    let foam = (s(1.0) - smoothstep(0.0, 0.2, inner))
        * smoothstep(0.55, 0.9, hash2(band, along.floor() + s(3.0)));
    let mut water = mix3(body, ink(INK), crest_line * s(0.85));
    water = mix3(water, ink(CREAM), foam * s(0.7));
    water = mix3(water, ink(PALE), smoothstep(0.1, 0.0, v) * s(0.5));
    mix3(colour, water, coverage(b - s(HORIZON), pixel))
}

/// The great wave: a dark crescent of water curling over, its crest tearing into claws of foam
/// with spray flying off.
#[inline(always)]
fn great_wave(colour: Vec3Simd, a: F, b: F, pixel: F, time: f32) -> Vec3Simd {
    let (ox, oy, outer_radius) = WAVE_OUTER;
    let (ix, iy, inner_radius) = WAVE_INNER;
    let (px, py) = (a - s(ox), b - s(oy));
    let distance = (px * px + py * py).sqrt();
    let (qx, qy) = (a - s(ix), b - s(iy));
    let hollow = (qx * qx + qy * qy).sqrt() - s(inner_radius);
    let surge = (s(time * 0.9)).sin() * s(0.012);
    let d = (distance - s(outer_radius) - surge)
        .fast_max(-hollow)
        .fast_max(s(-0.66) - b);

    let angle = py.atan2(px);
    let rise = smoothstep(0.1, 1.3, angle) * (s(1.0) - smoothstep(2.6, 3.0, angle));
    let claws = (angle * s(11.0) + s(time * 0.6)).sin().abs() * s(0.075) * rise;
    let foam_edge = s(outer_radius - 0.06) - claws + surge;
    let foam = smoothstep(0.0, 0.01, distance - foam_edge) * rise;
    let lines = fract((distance - s(outer_radius)) * s(26.0)) - s(0.5);
    let water = mix3(
        ink(INDIGO),
        ink(BLUE),
        smoothstep(0.35, 0.5, lines.abs()) * s(0.6),
    );
    let body = mix3(water, ink(CREAM), foam);
    let printed = print(colour, d, body, 0.005, pixel);

    let (u, w) = (a * s(34.0) + s(time * 2.0), b * s(34.0) - s(time * 1.4));
    let (cu, cv) = (u.floor(), w.floor());
    let (dx, dy) = (fract(u) - s(0.5), fract(w) - s(0.5));
    let drop = select(hash2(cu, cv).simd_gt(s(0.92)), s(1.0), s(0.0));
    let spray_zone = smoothstep(0.0, 0.1, distance - s(outer_radius + 0.02))
        * (s(1.0) - smoothstep(0.1, 0.3, distance - s(outer_radius)))
        * rise;
    let spray = (s(1.0) - smoothstep(0.1, 0.2, (dx * dx + dy * dy).sqrt())) * drop * spray_zone;
    mix3(printed, ink(CREAM), spray)
}

/// Two boats rowing in the swell, each a dark upturned hull with a few oarsmen.
#[inline(always)]
fn boats(colour: Vec3Simd, a: F, b: F, pixel: F, time: f32) -> Vec3Simd {
    let mut result = colour;
    for (n, (x, y, length)) in [(0.95, -0.7, 0.2), (0.3, -0.62, 0.13)]
        .into_iter()
        .enumerate()
    {
        let bob = (s(time * 0.9 + n as f32 * 2.0)).sin() * s(0.008);
        let (dx, dy) = (a - s(x), b - s(y) - bob);
        let prow = (dx / s(length)) * (dx / s(length));
        let hull = (dy - prow * s(0.03) + s(0.006)).abs()
            - s(0.014) * (s(1.0) - prow).fast_max(s(0.0)).sqrt();
        let d = hull.fast_max(dx.abs() - s(length));
        result = print(result, d, ink(CREAM) * s(0.92), 0.003, pixel);
        let oars = fract(dx * s(10.0 / length) + s(0.5)) - s(0.5);
        let rowers =
            (oars.abs() * s(0.05) + (dy - s(0.014)).abs() * s(0.3)).fast_max(s(0.0)) - s(0.003);
        let on_deck = coverage(dx.abs() - s(length * 0.6), pixel);
        result = mix3(
            result,
            ink(VERMILION),
            coverage(rowers, pixel) * on_deck * select(dy.simd_gt(s(0.0)), s(1.0), s(0.0)),
        );
    }
    result
}

/// The paper: fibres, soft stains and a darker, aged border, multiplied over every ink.
#[inline(always)]
fn paper(colour: Vec3Simd, a: F, b: F) -> Vec3Simd {
    let fibre = value(a * s(420.0), b * s(16.0)) * s(0.5)
        + value(a * s(18.0), b * s(380.0)) * s(0.5)
        - s(0.5);
    let stain = fbm(a * s(2.4), b * s(2.4) + s(5.0), 4) - s(0.5);
    let edge = smoothstep(1.0, 1.9, (a * s(0.55)).abs().fast_max(b.abs()));
    colour * (s(1.0) + fibre * s(0.06) + stain * s(0.14) - edge * s(0.2))
}

pub struct UkiyoE {
    look: Look,
    pipeline: SimdPipeline<ScreenVertexShader, Print>,
}

impl UkiyoE {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.1, 0.2, 0.42),
                die: Vec3::new(0.04, 0.09, 0.22),
                gold: Vec3::new(0.85, 0.22, 0.1),
                streams: Vec3::new(1.0, 0.3, 0.12),
                key: Vec3::new(-0.35, 0.5, 0.8).normalise(),
                key_colour: Vec3::new(1.1, 1.0, 0.85),
                sky: Vec3::new(0.7, 0.75, 0.7),
                ground: Vec3::new(0.1, 0.2, 0.35),
                softbox: Vec3::ZERO,
            },
            pipeline: SimdPipeline::new(ScreenVertexShader, Print)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for UkiyoE {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Harp],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::WoodClap),
                0.0,
                0.45,
                0.17,
                0.3,
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
            .draw_to_render_pass_simd(pass, &self.pipeline, (), |_| uniforms.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.05,
            bloom: 0.1,
            vignette: 0.15,
            grain: 0.006,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
