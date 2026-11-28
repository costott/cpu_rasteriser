//! A topographic survey: a pale hypsometric landscape of hills and a lake, drawn only with contour
//! lines (a heavy index line every fifth), with a survey ping that widens from the chip. The
//! chip is contoured by its own depth, so it reads as a relief map of itself.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, from_colour, length, mix3, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::noise::fbm_scalar;
use crate::themes::kit::post::{ScreenPass, compress, expand};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The height of the plain where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
/// The height between contour lines; every fifth line is an index line.
const INTERVAL: f32 = 0.4;
const LINE: Vec3 = Vec3 {
    x: 0.5,
    y: 0.3,
    z: 0.18,
};
const PING: Vec3 = Vec3 {
    x: 0.85,
    y: 0.25,
    z: 0.12,
};

/// A lake of flat water round the chip, hills that rise away from it, and ranges far off.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let rise = ((r - 4.0) / 22.0).clamp(0.0, 1.0);
    let behind = 0.12 + 0.88 * ((8.0 - z) / 18.0).clamp(0.0, 1.0);
    let hills = fbm_scalar(x * 0.045 + 7.0, z * 0.05, 4);
    let far = ((-z - 15.0) / 50.0).clamp(0.0, 1.0);
    FLOOR
        + rise * rise * behind * (hills * 14.0 + 3.0)
        + far * far * 18.0 * fbm_scalar(x * 0.02 + 2.0, z * 0.03 + 5.0, 3)
}

fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(-0.4, 0.8, 0.45).normalise(),
        sun_colour: Vec3::new(1.2, 1.2, 1.2),
        sky: Vec3::new(0.6, 0.6, 0.6),
        ground: Vec3::new(0.5, 0.5, 0.5),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: Vec3::ZERO,
            density: 0.0,
            falloff: 0.0,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.8, 0.78, 0.72),
        die: Vec3::new(0.35, 0.42, 0.46),
        gold: Vec3::new(0.55, 0.38, 0.22),
        streams: Vec3::ZERO,
        key: light.sun,
        key_colour: Vec3::new(1.6, 1.6, 1.6),
        sky: Vec3::new(0.6, 0.62, 0.64),
        ground: Vec3::new(0.5, 0.5, 0.5),
        softbox: Vec3::new(0.6, 0.6, 0.6),
    }
}

/// The land: a pale hypsometric tint from lake blue through sage to sand, with contour lines at
/// fixed heights that creep upwards like a rising tide, and a ping that widens from the chip.
/// Writes the distance to alpha as a negative number so the finishing pass can tell land from chip.
struct SurveyShader;

impl FragmentShaderSimd<KitVaryings> for SurveyShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let time = view.frame.time;
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let normal = v.normal().normalise();
        let facing = clamp01(normal.dot(to_eye * (s(1.0) / distance)).abs());
        let slope = ((s(1.0) - normal.y * normal.y).fast_max(s(0.0)).sqrt()
            / normal.y.abs().fast_max(s(0.05)))
        .fast_min(s(6.0));

        let height = (world.y - s(FLOOR)) * s(1.0 / 16.0);
        let valley = rgb(0.78, 0.88, 0.88);
        let meadow = rgb(0.86, 0.91, 0.8);
        let upland = rgb(0.94, 0.92, 0.78);
        let peak = rgb(0.92, 0.84, 0.76);
        let mut tint = mix3(valley, meadow, smoothstep(0.0, 0.2, height));
        tint = mix3(tint, upland, smoothstep(0.2, 0.55, height));
        tint = mix3(tint, peak, smoothstep(0.55, 1.0, height));
        let sun = splat3(lighting().sun);
        tint = tint * (s(0.9) + normal.dot(sun).fast_max(s(0.0)) * s(0.1));

        let footprint = distance * s(view.pixel_angle) / facing.fast_max(s(0.08));
        let per_pixel =
            footprint * (s(0.2) + slope * s(0.8) * facing.fast_min(s(1.0))).fast_min(s(8.0));
        let level = (world.y - s(FLOOR) - s(time * 0.12)) * s(1.0 / INTERVAL);
        let nearest = (level - level.floor() - s(0.5)).abs();
        let from_line = (s(0.5) - nearest) * s(INTERVAL);
        let index = (level * s(0.2)).floor();
        let is_index = ((level.floor() * s(0.2)) - index)
            .abs()
            .simd_lt(s(0.01))
            .select(s(1.0), s(0.0));
        let half_width = per_pixel * (s(0.8) + is_index * s(0.8));
        let aliasing = smoothstep(0.5, 1.0, per_pixel * s(1.0 / INTERVAL));
        let line = (s(1.0) - smoothstep(0.0, 1.0, from_line / half_width.fast_max(s(0.001))))
            * (s(1.0) - aliasing);
        let defined = smoothstep(0.02, 0.08, world.y - s(FLOOR)) * smoothstep(0.04, 0.2, slope);
        let mut colour = mix3(
            tint,
            splat3(LINE),
            line * defined * (s(0.65) + is_index * s(0.35)),
        );

        let radius = (world.x * world.x + world.z * world.z).sqrt();
        let outward = (radius - s(time * 2.5)) * s(1.0 / 5.0);
        let across = (outward - outward.floor() - s(0.5)).abs() * s(5.0);
        let ping = (s(1.0) - smoothstep(0.0, 1.0, across / (footprint * s(1.6) + s(0.03))))
            * smoothstep(34.0, 6.0, radius)
            * smoothstep(3.0, 5.0, radius);
        colour = mix3(colour, splat3(PING), ping * s(0.7));

        to_colour(expand(colour), -distance)
    }
}

/// A pale paper sky.
struct PaperSky;

impl FragmentShaderSimd<ScreenVaryings> for PaperSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let up = ray.y.fast_max(s(0.0));
        let colour = mix3(rgb(0.96, 0.95, 0.9), rgb(0.88, 0.92, 0.94), up.sqrt());
        to_colour(expand(colour), s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
struct ContourUniforms {
    source: Arc<TextureSampler>,
    width: f32,
    height: f32,
    time: f32,
}

/// Contours the chip by its distance from the eye, and outlines it. The land already carries its
/// own contours and keeps its colour; its alpha is made positive again.
struct ContourShader;

impl FragmentShaderSimd<ScreenVaryings> for ContourShader {
    type Uniforms = ContourUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &ContourUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (ox, oy) = (1.0 / u.width, 1.0 / u.height);
        let sample = u.source.sample_linear_clamp_simd([x, y]);
        let distance = sample.a;
        let chip = step_chip(distance);

        let neighbour = |dx: f32, dy: f32| {
            u.source
                .sample_linear_clamp_simd([x + s(dx * ox), y + s(dy * oy)])
                .a
        };
        let (a, b, c, d) = (
            neighbour(1.0, 0.0),
            neighbour(-1.0, 0.0),
            neighbour(0.0, 1.0),
            neighbour(0.0, -1.0),
        );
        let per_pixel = ((a - b).abs() + (c - d).abs())
            .fast_min(s(0.5))
            .fast_max(s(0.004));

        let level = distance.abs() * s(1.0 / 0.11) + s(u.time * 0.2);
        let nearest = (level - level.floor() - s(0.5)).abs();
        let from_line = (s(0.5) - nearest) * s(0.11);
        let index = ((level.floor() * s(0.2)) - (level.floor() * s(0.2)).floor())
            .simd_lt(s(0.01))
            .select(s(1.0), s(0.0));
        let width = per_pixel * (s(0.6) + index * s(0.5));
        let line = (s(1.0) - smoothstep(0.0, 1.0, from_line / width)) * chip;

        let shown = compress(from_colour(sample));
        let paper = rgb(0.96, 0.95, 0.9);
        let relief = mix3(
            shown * s(0.55) + paper * s(0.45),
            splat3(LINE),
            line * (s(0.6) + index * s(0.4)),
        );

        let jump = (a - distance).abs()
            + (b - distance).abs()
            + (c - distance).abs()
            + (d - distance).abs();
        let outline = smoothstep(0.4, 1.2, jump) * chip;
        let inked = mix3(relief, splat3(LINE) * s(0.6), outline);

        let out = mix3(shown, inked, chip);
        to_colour(expand(out), distance.abs())
    }
}

/// 1 where `distance` belongs to the chip: a positive (non-land), non-sky distance.
#[inline(always)]
fn step_chip(distance: F) -> F {
    let positive = distance.simd_gt(s(0.0)).select(s(1.0), s(0.0));
    positive
        * distance
            .simd_lt(s(SKY_DISTANCE * 0.5))
            .select(s(1.0), s(0.0))
}

pub struct Topographic {
    land: Shape,
    survey: SimdPipeline<KitVertexShader, SurveyShader>,
    sky: SimdPipeline<ScreenVertexShader, PaperSky>,
    contour: ScreenPass<ContourShader>,
}

impl Topographic {
    pub fn new() -> Self {
        Self {
            land: terrain(&ring_radii(1.5, 260.0, 60), 120, ground, |_, _| Vec3::ONE),
            survey: pipeline(SurveyShader, CullingMode::BackFace),
            sky: SimdPipeline::new(ScreenVertexShader, PaperSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            contour: ScreenPass::new(ContourShader),
        }
    }
}

impl Stage for Topographic {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Glockenspiel],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Ping),
                0.0,
                0.45,
                0.1,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        self.land.draw(
            pass,
            &self.survey,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Still,
            },
            toon,
        );
        scene.chip.draw(pass, view, &look());
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| view.clone());
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let uniforms = ContourUniforms {
            source: image,
            width: scene.extent.width as f32,
            height: scene.extent.height as f32,
            time: scene.local_time,
        };
        Some(self.contour.run(context, scene, uniforms))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.0,
            vignette: 0.1,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
