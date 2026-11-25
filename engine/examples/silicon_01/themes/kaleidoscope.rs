//! kaleidoscope: the inside of the instrument. Jewel-toned glass shards tumble in a mirrored
//! wedge that repeats six times round the chip, over a polished floor that reflects the pattern.
//!
//! The backdrop is one full-screen shader on the plane at unit distance ahead of the camera
//! (`sx`, `sy`): the plane is folded about the chip into one wedge, and the wedge is filled with a
//! slowly turning field of cells and rings.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, hash2, mix3, pow_highlight, rgb, s, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::noise::{cellular, fbm};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::{Scene, Stage};

/// The number of mirrored wedges round the centre.
const WEDGES: f32 = 6.0;
/// The fold's centre on the sky plane, and the height of the floor's edge.
const CENTRE: (f32, f32) = (0.0, 0.1);
const HORIZON: f32 = -0.16;

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Mirrors;

/// Folds (`x`, `y`) about the centre into the first wedge, turning the whole pattern with time.
#[inline(always)]
fn fold(x: F, y: F, time: f32) -> (F, F, F) {
    let (dx, dy) = (x - s(CENTRE.0), y - s(CENTRE.1));
    let radius = (dx * dx + dy * dy).sqrt();
    let wedge = TAU / WEDGES;
    let turn = dy.atan2(dx);
    let local = turn - (turn / s(wedge)).floor() * s(wedge);
    let angle = (local - s(wedge * 0.5)).abs();
    let (u, v) = (radius * angle.cos(), radius * angle.sin());
    let spin = s(time * 0.06);
    (
        u * spin.cos() - v * spin.sin(),
        u * spin.sin() + v * spin.cos(),
        radius,
    )
}

/// The glass at folded position (`u`, `v`), `radius` from the centre: coloured cells with bright
/// facet lines, with rings and a soft inner glow.
#[inline(always)]
fn glass(u: F, v: F, radius: F, time: f32) -> Vec3Simd {
    let (wx, wy) = (u * s(5.5) + s(time * 0.15), v * s(5.5) - s(time * 0.1));
    let warp = fbm(wx * s(0.5), wy * s(0.5), 3);
    let cells = cellular(wx + warp * s(1.2), wy + warp * s(1.2), time * 0.1);
    let id = hash2((wx * s(0.7)).floor(), (wy * s(0.7)).floor());
    let phase = radius * s(0.9) + warp * s(0.35) + id * s(0.12) - s(time * 0.04);
    let h = s(0.5) + s(0.5) * (s(TAU) * phase).sin();
    let glassy = mix3(
        rgb(0.04, 0.03, 0.22),
        rgb(0.55, 0.08, 0.5),
        smoothstep(0.0, 0.5, h),
    );
    let hue = mix3(glassy, rgb(1.1, 0.62, 0.22), smoothstep(0.6, 1.0, h));
    let hue = mix3(hue, rgb(0.05, 0.5, 0.6), smoothstep(0.82, 0.95, id));
    let jewel = mix3(
        rgb(0.03, 0.02, 0.1),
        hue,
        smoothstep(0.05, 0.6, cells) * s(0.9) + s(0.1),
    );
    let facets = pow_highlight(s(0.5) + s(0.5) * (cells * s(22.0)).sin(), 6.0);
    let rings = pow_highlight(
        s(0.5) + s(0.5) * (radius * s(34.0) - s(time * 0.5)).sin(),
        8.0,
    );
    jewel * (s(0.55) + warp * s(0.7))
        + rgb(1.5, 1.2, 0.8) * (facets * s(0.4))
        + rgb(0.9, 0.6, 1.6) * (rings * s(0.18))
}

impl FragmentShaderSimd<ScreenVaryings> for Mirrors {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let pixel = s(view.pixel_angle);
        let ahead = (-ray.z).fast_max(s(0.05));
        let (sx, sy) = (ray.x / ahead, ray.y / ahead);

        let above = clamp01((sy - s(HORIZON)) / pixel + s(0.5));
        let (u0, v0, r0) = fold(sx, sy.fast_max(s(HORIZON)), u.time);
        let wall = glass(u0, v0, r0, u.time) * (s(1.0) - smoothstep(0.0, 1.8, r0) * s(0.45));

        let depth = (s(HORIZON) - sy).fast_max(s(0.0));
        let (ru, rv, rr) = fold(sx, s(HORIZON) + depth * s(1.0), u.time);
        let floor = glass(ru, rv, rr, u.time) * (s(0.55) * (-depth * s(1.6)).exp() + s(0.04));
        let sheen = rgb(0.7, 0.45, 1.0) * ((-depth * s(14.0)).exp() * s(0.35));

        let colour = mix3(floor + sheen, wall, above);
        to_colour(colour, s(SKY_DISTANCE))
    }
}

pub struct Kaleidoscope {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Mirrors>,
    glints: Particles,
}

impl Kaleidoscope {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.1, 0.06, 0.16),
                die: Vec3::new(0.04, 0.02, 0.1),
                gold: Vec3::new(1.0, 0.7, 0.3),
                streams: Vec3::new(0.9, 0.5, 1.0),
                key: Vec3::new(0.3, 0.6, -0.7).normalise(),
                key_colour: Vec3::new(2.4, 1.6, 2.8),
                sky: Vec3::new(0.3, 0.15, 0.5),
                ground: Vec3::new(0.2, 0.3, 0.6),
                softbox: Vec3::new(0.6, 0.4, 0.8),
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Mirrors)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            glints: Particles::new(70, 61),
        }
    }
}

impl Stage for Kaleidoscope {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Glockenspiel, Instrument::Harp],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Tinkle),
                0.0,
                0.45,
                0.09,
                -0.3,
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
        self.glints.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 0.0),
                half_extent: Vec3::new(9.0, 5.0, 4.0),
                velocity: Vec3::new(0.0, -0.1, 0.0),
                wander: 0.8,
                size: (0.012, 0.045),
                colour: Vec3::new(1.2, 0.9, 1.5),
                twinkle: 2.5,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.6,
            vignette: 0.35,
            grain: 0.012,
            aberration: 0.004,
            desaturate: 0.0,
        }
    }
}
