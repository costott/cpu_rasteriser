//! deep-nebula: a stellar nursery. Domain-warped gas in violet and teal glows from within, dark
//! dust pillars rise along the lower edge rimmed by the light of young stars, and bright stars
//! throw diffraction spikes across the dust.
//!
//! The backdrop is one full-screen shader on the plane at unit distance ahead of the camera
//! (`sx`, `sy`).

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, hash2, mix3, rgb, s, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::noise::fbm;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::sky::stars;
use crate::themes::{Scene, Stage};

/// The height of the pillars' base on the sky plane; their tops rise to about `BASE + REACH`.
const BASE: f32 = -0.36;
const REACH: f32 = 0.3;

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Nebula;

/// The skyline of the dust pillars at `x`: a few broad columns with ragged, slowly boiling tops.
#[inline(always)]
fn skyline(x: F, time: f32) -> F {
    let column = |centre: f32, width: f32, height: f32| {
        let d = (x - s(centre)) / s(width);
        s(height) * (-d * d).exp()
    };
    let columns = column(-0.62, 0.13, 1.0)
        + column(-0.2, 0.09, 0.5)
        + column(0.46, 0.11, 0.75)
        + column(0.8, 0.12, 0.9);
    let ragged = fbm(x * s(16.0) + s(time * 0.05), s(2.0), 4);
    s(BASE) + (columns * s(REACH) + ragged * s(0.12) + s(0.03)).fast_max(s(0.0))
}

/// Gas at (`x`, `y`): returns its density and a blend factor between its two hues.
#[inline(always)]
fn gas(x: F, y: F, time: f32) -> (F, F, F) {
    let (px, py) = (x * s(2.4) + s(time * 0.012), y * s(2.4) - s(time * 0.008));
    let qx = fbm(px, py, 3);
    let qy = fbm(px + s(5.2), py + s(1.3), 3);
    let flow = s(2.2) + s(0.5) * (s(time * 0.15)).sin();
    let body = fbm(px + qx * flow, py + qy * flow, 4);
    let filaments = fbm(
        px * s(3.0) + qy * s(2.0) + s(time * 0.02),
        py * s(3.0) + qx * s(2.0),
        3,
    );
    (body, filaments, qx)
}

/// A star's diffraction cross at offset (`dx`, `dy`) from its centre.
#[inline(always)]
fn spikes(dx: F, dy: F, pixel: F) -> F {
    let (ax, ay) = (dx.abs(), dy.abs());
    let width = pixel * s(0.8);
    let horizontal = (-(ax / s(0.05)) - (ay / width)).exp();
    let vertical = (-(ay / s(0.05)) - (ax / width)).exp();
    let core = (-(ax * ax + ay * ay) / (pixel * pixel * s(2.5))).exp();
    horizontal * s(0.5) + vertical * s(0.5) + core * s(1.6)
}

/// Bright stars on a coarse grid: a third of the cells hold one, in white or amber, drifting with
/// a slow twinkle.
#[inline(always)]
fn beacons(sx: F, sy: F, pixel: F, time: f32) -> Vec3Simd {
    const CELL: f32 = 0.42;
    let (gx, gy) = (sx / s(CELL), sy / s(CELL));
    let (cx, cy) = (gx.floor(), gy.floor());
    let mut sum = rgb(0.0, 0.0, 0.0);
    for oy in [0.0, 1.0] {
        for ox in [-1.0, 0.0] {
            let (ix, iy) = (cx + s(ox), cy + s(oy - 1.0));
            let h = hash2(ix + s(11.0), iy + s(3.0));
            let present = clamp01((h - s(0.62)) * s(40.0));
            let jx = hash2(ix + s(2.0), iy + s(8.0)) * s(0.7) + s(0.15);
            let jy = hash2(ix + s(5.0), iy + s(1.0)) * s(0.7) + s(0.15);
            let (dx, dy) = ((ix + jx) * s(CELL) - sx, (iy + jy) * s(CELL) - sy);
            let twinkle = s(0.8) + s(0.2) * (s(time * 1.7) + h * s(60.0)).sin();
            let tint = mix3(
                rgb(0.8, 0.9, 1.3),
                rgb(1.4, 1.0, 0.7),
                smoothstep(0.7, 0.95, h),
            );
            sum = sum + tint * (spikes(dx, dy, pixel) * present * twinkle * (s(0.5) + h));
        }
    }
    sum
}

impl FragmentShaderSimd<ScreenVaryings> for Nebula {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let pixel = s(view.pixel_angle);
        let ahead = (-ray.z).fast_max(s(0.05));
        let (sx, sy) = (ray.x / ahead, ray.y / ahead);

        let (body, filaments, blend) = gas(sx, sy, u.time);
        let density = smoothstep(0.32, 0.8, body);
        let hue = mix3(
            rgb(0.03, 0.42, 0.55),
            rgb(0.36, 0.1, 0.58),
            smoothstep(0.3, 0.7, blend),
        );
        let core = smoothstep(0.62, 0.95, body);
        let glow = (s(0.35) + filaments * s(0.9)) * density;
        let mut colour = rgb(0.004, 0.005, 0.016)
            + hue * (glow * s(0.6))
            + rgb(1.2, 0.5, 0.32) * (core * core * filaments * s(0.55));

        let starfield = stars(ray.x, ray.y, ray.z, 1.0, s(u.time)) * (s(1.0) - density * s(0.7));
        colour = colour + rgb(0.9, 0.95, 1.1) * (starfield * s(1.2));
        colour = colour + beacons(sx, sy, pixel, u.time) * (s(1.0) - density * s(0.35));

        let top = skyline(sx, u.time);
        let depth = top - sy;
        let inside = clamp01(depth / pixel + s(0.5));
        let grain = fbm(sx * s(14.0), sy * s(14.0) - s(u.time * 0.01), 4);
        let rim = (-depth.abs() * s(32.0)).exp() * (s(0.4) + grain * s(0.9));
        let dust = rgb(0.006, 0.005, 0.012) + rgb(0.02, 0.012, 0.02) * grain;
        let lit = rgb(1.6, 0.62, 0.3) * (rim * s(0.28)) * (s(1.0) - inside * s(0.25));
        colour = mix3(
            colour + lit * (s(1.0) - inside),
            dust + lit * inside,
            inside,
        );

        to_colour(colour, s(SKY_DISTANCE))
    }
}

pub struct DeepNebula {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Nebula>,
    motes: Particles,
}

impl DeepNebula {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.09, 0.08, 0.13),
                die: Vec3::new(0.03, 0.02, 0.08),
                gold: Vec3::new(0.5, 0.35, 0.9),
                streams: Vec3::new(0.55, 0.45, 1.0),
                key: Vec3::new(0.4, 0.5, -0.75).normalise(),
                key_colour: Vec3::new(1.5, 0.8, 2.6),
                sky: Vec3::new(0.12, 0.08, 0.3),
                ground: Vec3::new(0.05, 0.2, 0.28),
                softbox: Vec3::ZERO,
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Nebula)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            motes: Particles::new(80, 46),
        }
    }
}

impl Stage for DeepNebula {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Horn],
            treatment: Treatment::Vast,
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Shimmer),
                0.0,
                0.45,
                0.1,
                -0.3,
            )],
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        scene.chip.draw(pass, view, &self.look);
        let uniforms = Uniforms {
            view: view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
        self.motes.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.0, 0.0),
                half_extent: Vec3::new(9.0, 5.0, 4.0),
                velocity: Vec3::new(0.12, 0.06, 0.0),
                wander: 0.6,
                size: (0.012, 0.04),
                colour: Vec3::new(0.7, 0.55, 1.2),
                twinkle: 1.6,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.7,
            vignette: 0.4,
            grain: 0.018,
            aberration: 0.003,
            desaturate: 0.0,
        }
    }
}
