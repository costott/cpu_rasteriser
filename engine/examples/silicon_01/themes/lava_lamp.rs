//! lava-lamp: a lamp seen from the inside. Blobs of molten wax rise out of a glowing pool, swell,
//! merge and sink again through wine-dark liquid lit by the bulb below.
//!
//! The backdrop is one full-screen shader on the plane at unit distance ahead of the camera
//! (`sx`, `sy`): a metaball field of the pool and the rising blobs, shaded from its gradient.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, mix3, rgb, s, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::noise::fbm;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::{Scene, Stage};

/// A rising blob's lane (x), its sway, the height range it travels (`bottom` to `top`), how fast
/// it makes a round trip in radians per second, its phase and its radius.
struct Blob {
    lane: f32,
    sway: f32,
    bottom: f32,
    top: f32,
    speed: f32,
    phase: f32,
    radius: f32,
}

const BLOBS: [Blob; 7] = [
    Blob {
        lane: -0.36,
        sway: 0.03,
        bottom: -0.2,
        top: 0.2,
        speed: 0.31,
        phase: 1.6,
        radius: 0.055,
    },
    Blob {
        lane: -0.2,
        sway: 0.04,
        bottom: -0.2,
        top: 0.34,
        speed: 0.23,
        phase: 2.1,
        radius: 0.075,
    },
    Blob {
        lane: -0.04,
        sway: 0.03,
        bottom: -0.2,
        top: 0.28,
        speed: 0.37,
        phase: 1.0,
        radius: 0.05,
    },
    Blob {
        lane: 0.15,
        sway: 0.04,
        bottom: -0.2,
        top: 0.36,
        speed: 0.27,
        phase: 1.1,
        radius: 0.07,
    },
    Blob {
        lane: 0.3,
        sway: 0.03,
        bottom: -0.2,
        top: 0.22,
        speed: 0.34,
        phase: 2.4,
        radius: 0.055,
    },
    Blob {
        lane: 0.42,
        sway: 0.03,
        bottom: -0.2,
        top: 0.3,
        speed: 0.4,
        phase: 0.9,
        radius: 0.045,
    },
    Blob {
        lane: 0.02,
        sway: 0.05,
        bottom: -0.2,
        top: 0.45,
        speed: 0.17,
        phase: 1.9,
        radius: 0.09,
    },
];
/// The pool the blobs rise from and sink back into: its centre and radius on the sky plane.
const POOL: (f32, f32, f32) = (0.0, -1.46, 1.3);

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Lamp;

/// The field's value on a ball's surface.
const ISO: f32 = 0.5;
/// How far a ball's influence reaches, in radii.
const REACH: f32 = 2.2;

/// The metaball field and its gradient at (`x`, `y`). Each ball adds a smooth bump that dies at
/// `REACH` radii; where the sum passes `ISO` is the wax's surface.
#[inline(always)]
fn field(x: F, y: F, time: f32) -> (F, F, F) {
    let (mut f, mut gx, mut gy) = (s(0.0), s(0.0), s(0.0));
    let mut add = |cx: F, cy: F, r: f32| {
        let (dx, dy) = (x - cx, y - cy);
        let reach2 = r * r * REACH * REACH;
        let falloff = (s(1.0) - (dx * dx + dy * dy) / s(reach2)).fast_max(s(0.0));
        let squared = falloff * falloff;
        f += squared * falloff;
        let pull = s(-6.0 / reach2) * squared;
        gx += pull * dx;
        gy += pull * dy;
    };
    for b in &BLOBS {
        let rise = s(0.5) - s(0.5) * (s(time * b.speed + b.phase)).cos();
        let height = s(b.bottom) + s(b.top - b.bottom) * rise;
        let sway = s(b.sway) * (s(time * 0.4 + b.phase * 3.0)).sin();
        add(s(b.lane) + sway, height, b.radius);
    }
    add(
        s(POOL.0),
        s(POOL.1) + s(0.015) * (s(time * 0.6)).sin(),
        POOL.2,
    );
    (f, gx, gy)
}

impl FragmentShaderSimd<ScreenVaryings> for Lamp {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let pixel = s(view.pixel_angle);
        let ahead = (-ray.z).fast_max(s(0.05));
        let (sx, sy) = (ray.x / ahead, ray.y / ahead);

        let swirl = fbm(
            sx * s(2.2) + s(u.time * 0.03),
            sy * s(2.2) - s(u.time * 0.05),
            4,
        );
        let low = smoothstep(0.8, -0.5, sy);
        let liquid = mix3(rgb(0.1, 0.012, 0.11), rgb(0.55, 0.09, 0.2), low * low);
        let mut colour =
            liquid * (s(0.6) + swirl * s(0.9)) + rgb(0.9, 0.3, 0.08) * (low * low * low * s(0.35));

        let (f, gx, gy) = field(sx, sy, u.time);
        let slope = (gx * gx + gy * gy).sqrt().fast_max(s(1e-3));
        let cover = clamp01((f - s(ISO)) / (slope * pixel) + s(0.5));
        let depth = clamp01((f - s(ISO)) * s(1.4));
        let nz = depth.sqrt();
        let (nxy_x, nxy_y) = (
            -gx / slope * (s(1.0) - depth * depth).sqrt(),
            -gy / slope * (s(1.0) - depth * depth).sqrt(),
        );

        let warm = mix3(rgb(0.7, 0.05, 0.03), rgb(1.7, 0.36, 0.05), nz * nz);
        let towards_bulb = clamp01(-nxy_y * s(0.5) + s(0.5));
        let mottle = s(0.75)
            + fbm(
                sx * s(7.0) + s(u.time * 0.05),
                sy * s(7.0) - s(u.time * 0.12),
                3,
            ) * s(0.6);
        let wax = warm
            * mottle
            * (s(0.5) + towards_bulb * s(0.9))
            * (s(1.4) - clamp01((sy + s(0.4)) * s(0.9)) * s(0.6));
        let half = (s(-0.35) * (nxy_x - s(0.0)) + s(0.5) * nxy_y + s(0.7) * nz).fast_max(s(0.0));
        let sheen = half * half * half * half * half * half * half * half;
        let edge = s(1.0) - nz;
        let rim = edge * edge * edge;
        let body =
            wax + rgb(1.0, 0.7, 0.5) * (sheen * s(0.4)) + rgb(2.2, 0.6, 0.1) * (rim * s(0.5));

        colour =
            colour + rgb(0.5, 0.12, 0.05) * (smoothstep(0.1, ISO, f) * s(0.5) * (s(1.0) - cover));
        colour = mix3(colour, body, cover);

        to_colour(colour, s(SKY_DISTANCE))
    }
}

pub struct LavaLamp {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Lamp>,
    bubbles: Particles,
}

impl LavaLamp {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.3, 0.12, 0.1),
                die: Vec3::new(0.08, 0.02, 0.05),
                gold: Vec3::new(1.0, 0.62, 0.2),
                streams: Vec3::new(1.0, 0.7, 0.3),
                key: Vec3::new(-0.3, 0.2, 0.9).normalise(),
                key_colour: Vec3::new(3.0, 1.2, 0.5),
                sky: Vec3::new(0.4, 0.08, 0.22),
                ground: Vec3::new(1.0, 0.35, 0.08),
                softbox: Vec3::new(1.0, 0.6, 0.35),
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Lamp)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            bubbles: Particles::new(60, 52),
        }
    }
}

impl Stage for LavaLamp {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Marimba],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Bloop),
                0.0,
                0.45,
                0.17,
                0.3,
            )],
            ..WorldSound::default()
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
        self.bubbles.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 0.0),
                half_extent: Vec3::new(9.0, 5.0, 4.0),
                velocity: Vec3::new(0.0, 0.3, 0.0),
                wander: 0.3,
                size: (0.02, 0.06),
                colour: Vec3::new(1.4, 0.55, 0.3),
                twinkle: 1.2,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.6,
            vignette: 0.4,
            grain: 0.015,
            aberration: 0.002,
            desaturate: 0.0,
        }
    }
}
