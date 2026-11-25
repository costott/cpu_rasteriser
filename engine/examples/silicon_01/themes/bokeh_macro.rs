//! bokeh-macro: a macro shot at night. Out-of-focus lights glow as soft discs in several depth
//! layers behind the crisp chip, and blur further on a dark glossy desk below the horizon.
//!
//! The backdrop is one full-screen shader on the plane at unit distance ahead of the camera
//! (`sx`, `sy`). Each depth layer is a grid with one disc per cell; nearer layers have larger,
//! fainter discs that drift faster.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, hash2, mix3, rgb, s, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::{Scene, Stage};

/// The height of the desk's far edge on the sky plane.
const HORIZON: f32 = -0.16;

/// A layer of discs: the grid cell size (which sets the disc size), drift velocity on the sky
/// plane, brightness and the layer's hash seed.
struct Layer {
    cell: f32,
    drift: (f32, f32),
    brightness: f32,
    seed: f32,
}

const LAYERS: [Layer; 4] = [
    Layer {
        cell: 0.16,
        drift: (0.012, 0.004),
        brightness: 0.9,
        seed: 1.0,
    },
    Layer {
        cell: 0.3,
        drift: (-0.02, 0.008),
        brightness: 0.7,
        seed: 7.0,
    },
    Layer {
        cell: 0.5,
        drift: (0.03, -0.006),
        brightness: 0.45,
        seed: 13.0,
    },
    Layer {
        cell: 0.8,
        drift: (-0.045, 0.01),
        brightness: 0.3,
        seed: 29.0,
    },
];

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Lights;

/// The light of one layer's disc in the cell holding (`x`, `y`): a soft-edged disc with a brighter
/// rim, tinted by a per-cell choice from a warm palette and sometimes absent.
#[inline(always)]
fn discs(layer: &Layer, x: F, y: F, pixel: F, time: f32) -> Vec3Simd {
    let (px, py) = (x + s(time * layer.drift.0), y + s(time * layer.drift.1));
    let (gx, gy) = (px / s(layer.cell), py / s(layer.cell));
    let (ix, iy) = (gx.floor(), gy.floor());
    let id = hash2(ix + s(layer.seed), iy + s(layer.seed * 3.0));
    let jitter = |salt: f32| hash2(ix + s(salt), iy + s(layer.seed + salt)) * s(0.36) + s(0.32);
    let (dx, dy) = (
        (gx - ix - jitter(4.0)) * s(layer.cell),
        (gy - iy - jitter(9.0)) * s(layer.cell),
    );
    let radius =
        s(layer.cell * 0.3) * (s(0.55) + s(0.45) * hash2(ix + s(17.0), iy + s(layer.seed)));
    let d = (dx * dx + dy * dy).sqrt();
    let edge = pixel * s(1.5) + radius * s(0.12);
    let body = clamp01((radius - d) / edge + s(0.5));
    let rim = (-((d - radius) / (radius * s(0.12))).abs()).exp() * s(0.5);
    let present = smoothstep(0.35, 0.5, id);
    let flicker = s(0.85) + s(0.15) * (s(time * 0.9) + id * s(40.0)).sin();
    let tint = mix3(
        mix3(
            rgb(1.6, 0.85, 0.3),
            rgb(1.5, 0.35, 0.3),
            smoothstep(0.5, 0.75, id),
        ),
        rgb(0.4, 0.9, 1.4),
        smoothstep(0.85, 0.95, id),
    );
    tint * ((body * s(0.55) + rim * body) * present * flicker * s(layer.brightness))
}

impl FragmentShaderSimd<ScreenVaryings> for Lights {
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
        let height = (sy - s(HORIZON)).abs();
        let background = mix3(
            rgb(0.05, 0.03, 0.04),
            rgb(0.012, 0.014, 0.03),
            smoothstep(0.0, 0.9, sy),
        );
        let mut lights = rgb(0.0, 0.0, 0.0);
        let mirrored_y = s(HORIZON) + height;
        for layer in &LAYERS {
            lights = lights + discs(layer, sx, mirrored_y, pixel, u.time);
        }
        let desk = (-height * s(2.2)).exp();
        let desk_colour = rgb(0.03, 0.02, 0.02) + lights * (desk * s(0.35));
        let wall = background + lights;
        let colour = mix3(desk_colour, wall, above);
        to_colour(colour, s(SKY_DISTANCE))
    }
}

pub struct BokehMacro {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Lights>,
    dust: Particles,
}

impl BokehMacro {
    pub fn new() -> Self {
        Self {
            look: Look::default(),
            backdrop: SimdPipeline::new(ScreenVertexShader, Lights)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            dust: Particles::new(40, 83),
        }
    }
}

impl Stage for BokehMacro {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Piano],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::SoftChime),
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
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 1.5),
                half_extent: Vec3::new(8.0, 4.0, 2.0),
                velocity: Vec3::new(0.2, 0.05, 0.0),
                wander: 0.4,
                size: (0.05, 0.14),
                colour: Vec3::new(1.2, 0.8, 0.5),
                twinkle: 0.6,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.55,
            vignette: 0.45,
            grain: 0.016,
            aberration: 0.003,
            desaturate: 0.0,
        }
    }
}
