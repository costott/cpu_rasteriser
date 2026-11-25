//! black-hole: a lensed accretion disc over a dead planet's limb. The disc is a tilted ring of
//! turbulent hot gas whose far side is bent over the hole's shadow, a thin photon ring hugs the
//! shadow, and the stars behind are pulled round it.
//!
//! The backdrop is one full-screen shader on the plane at unit distance ahead of the camera
//! (`sx`, `sy`). It is analytic: the lensing is the weak-field deflection of a point mass applied
//! to the star field, and the disc is shaded on its own plane and projected, so nothing is marched.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, mix3, rgb, s, select, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::noise::fbm;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::sky::stars;
use crate::themes::{Scene, Stage};

/// The hole's centre on the sky plane, and the radius of its shadow.
const CENTRE: (f32, f32) = (-0.27, 0.04);
const SHADOW: f32 = 0.075;
/// The disc's inner and outer radius in its own plane, and how much the tilt squashes its depth.
const INNER: f32 = 0.14;
const OUTER: f32 = 0.55;
const SQUASH: f32 = 0.2;
/// The deflection constant: stars at distance `r` from the hole appear where the unlensed sky
/// has `r - EINSTEIN / r`.
const EINSTEIN: f32 = 0.0128;
/// How long a turbulence layer winds up before it fades out and a fresh one fades in.
const WIND_PERIOD: f32 = 5.0;

/// The limb of the planet: the sky-plane height of its top and its radius.
const LIMB_TOP: f32 = -0.16;
const LIMB_RADIUS: f32 = 2.6;

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Hole;

/// The disc's heat colour for `t`: 0 is a dull ember, about 1 is white-hot.
#[inline(always)]
fn heat(t: F) -> Vec3Simd {
    let ember = rgb(0.5, 0.07, 0.02);
    let orange = rgb(2.4, 0.78, 0.14);
    let white = rgb(6.0, 4.4, 2.4);
    let c = mix3(ember, orange, smoothstep(0.05, 0.5, t));
    mix3(c, white, smoothstep(0.5, 1.2, t))
}

/// Turbulent gas at (`x`, `z`) in the disc's plane, `rho` from the hole: noise that winds up at
/// the Keplerian rate, so the inner gas shears ahead of the outer. Two layers a half period apart
/// crossfade so the shear stays bounded.
#[inline(always)]
fn gas(x: F, z: F, rho: F, time: f32) -> F {
    let layer = |phase: f32| {
        let angle = s(phase * 0.05) / (rho * rho.sqrt());
        let (sin, cos) = (angle.sin(), angle.cos());
        let (qx, qz) = (x * cos - z * sin, x * sin + z * cos);
        let ridge =
            s(0.5) + s(0.5) * (rho * s(120.0) + fbm(qx * s(9.0), qz * s(9.0), 2) * s(6.0)).sin();
        let cloud = fbm(qx * s(16.0) + s(3.0), qz * s(16.0), 4);
        cloud * s(0.85) + ridge * s(0.25)
    };
    let cycle = time / WIND_PERIOD;
    let fade = (std::f32::consts::PI * cycle.fract()).sin().powi(2);
    layer(cycle.fract() * WIND_PERIOD) * s(fade)
        + layer((cycle + 0.5).fract() * WIND_PERIOD) * s(1.0 - fade)
}

/// The disc's light at (`x`, `z`) in its plane: heat falling with radius, mottled by gas and
/// brightened on the side moving towards the viewer.
#[inline(always)]
fn disc(x: F, z: F, time: f32) -> (Vec3Simd, F) {
    let rho = (x * x + z * z).sqrt().fast_max(s(0.001));
    let towards = s(1.0) - s(0.6) * x / rho;
    let core = s(INNER) / rho;
    let t = core * core.sqrt() * (s(0.35) + gas(x, z, rho, time) * s(1.1)) * towards;
    let edge = smoothstep(INNER * 0.95, INNER * 1.12, rho)
        * (s(1.0) - smoothstep(OUTER * 0.55, OUTER, rho));
    (heat(t) * (s(0.35) + t * s(1.6)), edge)
}

/// Deep space seen through the lens: dust and stars at the position the light came from.
#[inline(always)]
fn space(dx: F, dy: F, r2: F, time: f32) -> Vec3Simd {
    let k = s(1.0) - s(EINSTEIN) / r2;
    let (lx, ly) = (s(CENTRE.0) + dx * k, s(CENTRE.1) + dy * k);
    let norm = (lx * lx + ly * ly + s(1.0)).sqrt();
    let point = stars(lx / norm, ly / norm, -s(1.0) / norm, 1.0, s(time));
    let tint = mix3(
        rgb(1.0, 0.8, 0.62),
        rgb(0.7, 0.82, 1.1),
        smoothstep(0.35, 0.6, fbm(lx * s(3.0), ly * s(3.0) + s(9.0), 2)),
    );
    let lane = ((ly - lx * s(0.3) - s(0.1)) / s(0.32)).abs();
    let band =
        (-lane * lane).exp() * smoothstep(0.35, 0.85, fbm(lx * s(4.0) + s(5.0), ly * s(4.0), 4));
    let dust = rgb(0.07, 0.05, 0.12) * band + rgb(0.012, 0.014, 0.03);
    dust + tint * (point * s(1.4))
}

/// The planet's surface below its limb, lit by the hole above: banded dark crust with a warm
/// glow that fades with depth and distance from the hole's axis.
#[inline(always)]
fn planet(sx: F, depth: F, time: f32) -> Vec3Simd {
    let reach = s(1.0) / (depth + s(0.08));
    let (u, v) = (sx * reach * s(2.0) + s(time * 0.02), reach * s(1.4));
    let crust = fbm(u, v, 4);
    let bands = s(0.5) + s(0.5) * (v * s(5.0) + crust * s(3.0)).sin();
    let lit = (-depth * s(5.0)).exp() * (s(1.0) - smoothstep(0.0, 1.4, sx.abs()) * s(0.6));
    let base = rgb(0.012, 0.012, 0.02) + rgb(0.02, 0.016, 0.02) * (crust * s(0.7) + bands * s(0.3));
    base + rgb(0.9, 0.34, 0.1) * (lit * (s(0.15) + crust * s(0.5) + bands * s(0.25)) * s(0.35))
}

impl FragmentShaderSimd<ScreenVaryings> for Hole {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let pixel = s(view.pixel_angle);
        let ahead = (-ray.z).fast_max(s(0.05));
        let (sx, sy) = (ray.x / ahead, ray.y / ahead);

        let (dx, dy) = (sx - s(CENTRE.0), sy - s(CENTRE.1));
        let r2 = (dx * dx + dy * dy).fast_max(s(1e-5));
        let r = r2.sqrt();
        let outside = clamp01((r - s(SHADOW)) / pixel + s(0.5));

        let mut colour = space(dx, dy, r2, u.time) * outside;
        let glow = (-(r - s(SHADOW)).fast_max(s(0.0)) * s(7.0)).exp();
        colour = colour + rgb(0.5, 0.2, 0.06) * (glow * glow * s(0.22) * outside);

        let ring = (r - s(SHADOW * 1.03)) / (pixel * s(1.6) + s(0.003));
        let towards = s(1.0) - s(0.45) * dx / r;
        colour = colour + rgb(3.2, 2.5, 1.7) * ((-ring * ring).exp() * towards * s(0.9));

        let lens = (r / s(SHADOW) - s(1.0)).fast_max(s(0.0));
        let arc = smoothstep(-0.5, 0.9, dy / r) * s(0.8) + s(0.2);
        let tilt = (s(OUTER) - s(INNER)) / s(2.4);
        let rho = s(INNER) + lens * tilt;
        let (ax, az) = (dx * rho / r, dy * rho / r);
        let (bent, _) = disc(ax, az, u.time);
        let bend_edge = smoothstep(0.0, 0.12, lens) * (s(1.0) - smoothstep(0.6, 2.4, lens));
        colour = colour + bent * (bend_edge * arc * s(0.9) * outside);

        let (px, pz) = (dx, dy / s(SQUASH));
        let (light, edge) = disc(px, pz, u.time);
        let behind = pz.simd_gt(s(0.0));
        let reveal = select(behind, outside, s(1.0));
        colour = mix3(colour, light, edge * reveal);

        let top = s(LIMB_TOP) - s(LIMB_RADIUS)
            + (s(LIMB_RADIUS * LIMB_RADIUS) - sx * sx)
                .fast_max(s(0.0))
                .sqrt();
        let depth = top - sy;
        let below = clamp01(depth / pixel + s(0.5));
        let atmosphere = (-depth.abs() * s(45.0)).exp();
        let rim = rgb(1.1, 0.5, 0.22)
            * (atmosphere * (s(1.0) - smoothstep(0.0, 1.6, sx.abs()) * s(0.7)) * s(0.45));
        let surface = planet(sx, depth.fast_max(s(0.0)), u.time);
        colour = mix3(colour + rim * (s(1.0) - below), surface, below);

        to_colour(colour, s(SKY_DISTANCE))
    }
}

pub struct BlackHole {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Hole>,
    embers: Particles,
}

impl BlackHole {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.06, 0.06, 0.08),
                die: Vec3::new(0.02, 0.016, 0.03),
                gold: Vec3::new(1.0, 0.5, 0.16),
                streams: Vec3::new(1.0, 0.62, 0.25),
                key: Vec3::new(0.2, 0.55, -0.8).normalise(),
                key_colour: Vec3::new(3.2, 1.7, 0.7),
                sky: Vec3::new(0.05, 0.07, 0.16),
                ground: Vec3::new(0.7, 0.26, 0.08),
                softbox: Vec3::ZERO,
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Hole)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            embers: Particles::new(90, 36),
        }
    }
}

impl Stage for BlackHole {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Sagging],
            treatment: Treatment::Sagging,
            ambience: vec![Ambience::once(
                Source::Synth(Accent::SubFall),
                0.0,
                1.1,
                0.35,
                0.0,
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
        self.embers.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 0.0),
                half_extent: Vec3::new(9.0, 4.0, 4.0),
                velocity: Vec3::new(-0.35, 0.05, 0.0),
                wander: 0.5,
                size: (0.015, 0.05),
                colour: Vec3::new(1.5, 0.7, 0.28),
                twinkle: 3.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.8,
            vignette: 0.45,
            grain: 0.02,
            aberration: 0.006,
            desaturate: 0.0,
        }
    }
}
