//! A watercolour landscape: soft washes of sage, ochre and slate on cold-pressed paper, a
//! winding river, a few blotted trees and a sky left largely unpainted. The finished frame is
//! smoothed into pools of pigment that darken at their wet edges, settle into the paper's tooth
//! and fade to bare white at the highlights and the margins.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::scene_colour;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::simd::{
    F, clamp01, from_colour, hash2, mix3, mul3, rgb, s, smoothstep, splat3, step_lt, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonUniforms,
    VariedToonShader, VariedToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::{Chain, Effect, ScreenPass, compress, expand, hdr_of_displayed};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::sky::{Clouds, Sky, SkyShader, SkyUniforms};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The height of the meadow where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.5,
    y: 0.55,
    z: -0.4,
};
const WIND: Vec2 = Vec2 { x: 1.0, y: 0.3 };
const PAPER: Vec3 = Vec3 {
    x: 0.97,
    y: 0.95,
    z: 0.9,
};

/// Where the river runs at depth `z`: it winds away from the foreground to the right of the chip.
fn river_centre(z: f32) -> f32 {
    9.0 + 0.1 * (-z) + 5.0 * (z * 0.06 + 1.0).sin()
}

/// How much of (`x`, `z`) is river, 1 in the channel and 0 on the banks.
fn river(x: f32, z: f32) -> f32 {
    let width = 2.4 + 0.5 * (z * 0.11).sin();
    let t = ((x - river_centre(z)).abs() - width).clamp(-1.0, 1.0);
    0.5 - 0.5 * t
}

/// Meadow rolling gently round the chip, falling to the river and rising into blue hills far off.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let clearing = (r / 8.0).clamp(0.0, 1.0);
    let swell = (fbm_scalar(x * 0.04 + 2.0, z * 0.05, 3) - 0.5) * 4.0;
    let far = ((-z - 40.0) / 90.0).clamp(0.0, 1.0);
    let hills = far * far * (10.0 + 40.0 * fbm_scalar(x * 0.012 + 5.0, z * 0.02, 3));
    FLOOR + clearing * clearing * swell + hills - river(x, z) * 0.5 * clearing
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.9, 1.7, 1.4),
        sky: Vec3::new(0.62, 0.7, 0.85),
        ground: Vec3::new(0.46, 0.42, 0.36),
        bands: 3.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: hdr_of_displayed(Vec3::new(0.78, 0.84, 0.9)),
            density: 0.0035,
            falloff: 0.03,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.34, 0.3, 0.27),
        die: Vec3::new(0.08, 0.14, 0.34),
        gold: Vec3::new(0.85, 0.6, 0.22),
        streams: Vec3::new(0.35, 0.65, 0.85),
        key: light.sun,
        key_colour: light.sun_colour * 0.6,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.45, 0.45, 0.45),
    }
}

fn sky() -> Sky {
    Sky {
        zenith: hdr_of_displayed(Vec3::new(0.62, 0.74, 0.88)),
        horizon: hdr_of_displayed(Vec3::new(0.96, 0.93, 0.86)),
        sun: Vec3::new(0.35, 0.3, -0.89).normalise(),
        sun_colour: Vec3::new(1.4, 1.2, 0.9),
        sun_size: 0.0,
        glow: 0.4,
        clouds: Some(Clouds {
            height: 60.0,
            coverage: 0.5,
            scale: 0.02,
            wind: Vec2::new(1.6, 0.4),
            lit: Vec3::new(2.4, 2.3, 2.1),
            shade: Vec3::new(0.6, 0.66, 0.8),
        }),
        stars: 0.0,
    }
}

/// A round-crowned tree: a leaning trunk and a clump of leaf masses in three greens.
fn tree(rng: &mut Rng) -> Shape {
    let bark = Vec3::new(0.3, 0.22, 0.17);
    let mut tree = Shape::limb(Vec3::ZERO, Vec3::new(0.1, 2.3, 0.0), 0.16, 0.07, 5, bark);
    let greens = [
        Vec3::new(0.3, 0.46, 0.2),
        Vec3::new(0.38, 0.52, 0.24),
        Vec3::new(0.22, 0.36, 0.2),
    ];
    for (i, &(x, y, z, r)) in [
        (0.0, 3.1, 0.0, 1.3),
        (-0.95, 2.6, 0.2, 0.9),
        (0.95, 2.7, -0.2, 1.0),
        (0.2, 3.9, 0.1, 0.8),
    ]
    .iter()
    .enumerate()
    {
        let lump = Shape::icosphere(r, 1, 0.22, rng.unit() * 10.0, greens[i % 3])
            .gradient_by_height(-r, r, greens[i % 3] * 0.7, greens[i % 3] * 1.25)
            .facet_variation(rng, 0.12)
            .transformed(&Mat4::translation(x, y, z));
        tree.append(&lump);
    }
    tree.sway_by_height(0.3, 4.5, 1.4)
}

/// The meadow and river: washes of sage, olive and ochre blotched by noise, and a cool river
/// whose ripples drift.
struct MeadowShader;

impl FragmentShaderSimd<KitVaryings> for MeadowShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let time = view.frame.time;
        let (x, z) = (world.x, world.z);
        let blot = fbm(x * s(0.07) + s(3.0), z * s(0.07), 4) * s(3.2);
        let glaze = blot.floor() + smoothstep(0.42, 0.58, blot - blot.floor());
        let fine = fbm(x * s(0.8), z * s(0.8), 2);
        let sage = rgb(0.55, 0.66, 0.36);
        let olive = rgb(0.36, 0.5, 0.2);
        let deep = rgb(0.14, 0.32, 0.2);
        let ochre = rgb(0.86, 0.68, 0.3);
        let mut albedo = mix3(ochre, sage, smoothstep(0.0, 1.0, glaze));
        albedo = mix3(albedo, olive, smoothstep(1.0, 2.0, glaze));
        albedo = mix3(albedo, deep, smoothstep(2.0, 3.0, glaze));
        albedo = albedo * (s(0.88) + fine * s(0.24));
        let (cx, cz) = ((x * s(0.9)).floor(), (z * s(0.9)).floor());
        let (jx, jz) = (hash2(cx, cz), hash2(cx + s(7.1), cz + s(3.3)));
        let (dx, dz) = (x * s(0.9) - cx - jx, z * s(0.9) - cz - jz);
        let dab = (s(0.22) - (dx * dx + dz * dz).sqrt()) * s(6.0);
        let dab = clamp01(dab) * step_lt(s(0.9), hash2(cx + s(1.7), cz + s(9.2)));
        let bloom = mix3(
            rgb(0.82, 0.36, 0.3),
            rgb(0.9, 0.78, 0.3),
            step_lt(hash2(cx, cz + s(4.4)), s(0.5)),
        );
        albedo = mix3(albedo, bloom, dab * s(0.85));

        let centre = s(9.0) + s(0.1) * (-z) + s(5.0) * (z * s(0.06) + s(1.0)).sin();
        let width = s(2.4) + s(0.5) * (z * s(0.11)).sin();
        let across = (x - centre).abs();
        let water = s(1.0) - smoothstep(0.0, 1.0, clamp01(across - width + s(0.5)));
        let ripple = ((z * s(1.3) + (x * s(0.6)).sin() * s(0.8) + s(time * 0.5)).sin() * s(0.5)
            + s(0.5))
            * smoothstep(0.0, 1.0, s(1.0) - across * s(1.0 / 3.0));
        let river = mix3(
            rgb(0.34, 0.52, 0.62),
            rgb(0.7, 0.82, 0.88),
            ripple * s(0.55),
        );
        let bank =
            (s(1.0) - smoothstep(0.0, 1.0, (across - width - s(0.4)).abs() * s(0.9))) * s(0.35);
        albedo = mix3(albedo, rgb(0.84, 0.75, 0.55), bank);
        albedo = mix3(albedo, river, water);

        let to_eye = splat3(view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        scene_colour(
            view,
            u.lighting.shade(view, world, normal, albedo, s(1.0)),
            world,
        )
    }
}

#[derive(Clone)]
struct FinishUniforms {
    source: Arc<TextureSampler>,
    time: f32,
    aspect: f32,
}

/// What water does to pigment on paper: the image bleeds slowly, darkens where one wash meets
/// another, granulates in the darks, vanishes to bare paper at highlights and in a ragged margin.
struct FinishShader;

#[inline(always)]
fn luminance(c: Vec3Simd) -> F {
    c.x * s(0.2126) + c.y * s(0.7152) + c.z * s(0.0722)
}

impl FragmentShaderSimd<ScreenVaryings> for FinishShader {
    type Uniforms = FinishUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &FinishUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (px, py) = (x * s(u.aspect), y);
        let drift = s(u.time * 0.04);

        let bleed_x = fbm(px * s(4.0) + drift, py * s(4.0), 3) - s(0.5);
        let bleed_y = fbm(px * s(4.0) + s(17.0), py * s(4.0) - drift, 3) - s(0.5);
        let (ux, uy) = (x + bleed_x * s(0.012), y + bleed_y * s(0.012));
        let tap = |dx: f32, dy: f32| {
            let sample = u.source.sample_linear_clamp_simd([ux + s(dx), uy + s(dy)]);
            compress(from_colour(sample))
        };
        let centre = u.source.sample_linear_clamp_simd([ux, uy]);
        let pigment = compress(from_colour(centre));
        let (rx, ry) = (0.0035, 0.0035 * u.aspect);
        let around = (tap(rx, 0.0) + tap(-rx, 0.0) + tap(0.0, ry) + tap(0.0, -ry)) * s(0.25);
        let pool = (luminance(around) - luminance(pigment)).abs();
        let wet_edge = smoothstep(0.003, 0.03, pool);
        let grey = splat3(Vec3::ONE) * luminance(pigment);
        let rich = mix3(grey, pigment, s(1.4));
        let darker = rich * (s(1.0) - wet_edge * s(0.4));

        let tone = luminance(darker);
        let tooth = fbm(px * s(260.0), py * s(260.0), 2) - s(0.5);
        let blotch = fbm(px * s(14.0) + s(5.0), py * s(14.0), 3) - s(0.5);
        let settle = s(1.0) + tooth * s(0.5) * (s(1.1) - tone) + blotch * s(0.22);
        let painted = darker * settle;

        let sheet = fbm(px * s(520.0), py * s(520.0), 2);
        let fibre = fbm(px * s(40.0), py * s(900.0), 2);
        let paper = splat3(PAPER) * (s(0.97) + sheet * s(0.04) + fibre * s(0.03));

        let bare = smoothstep(0.78, 0.96, tone + blotch * s(0.3));
        let edge = (x.fast_min(s(1.0) - x)).fast_min(y.fast_min(s(1.0) - y));
        let ragged = fbm(px * s(18.0), py * s(18.0), 3) - s(0.5);
        let margin = smoothstep(0.012, 0.05, edge + ragged * s(0.05));
        let coverage = margin * (s(1.0) - bare);
        let shown = mix3(paper, mul3(painted, paper * s(1.04)), coverage);
        to_colour(expand(shown), centre.a)
    }
}

pub struct Watercolour {
    land: Shape,
    trees: Shape,
    dust: Particles,
    land_pipeline: SimdPipeline<KitVertexShader, MeadowShader>,
    tree_pipeline: SimdPipeline<KitVertexShader, VariedToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyShader>,
    chain: Chain,
    finish: ScreenPass<FinishShader>,
}

impl Watercolour {
    pub fn new() -> Self {
        let mut rng = Rng::new(1805);
        let land = terrain(&ring_radii(1.5, 260.0, 52), 96, ground, |_, _| Vec3::ONE);
        let oaks: Vec<Shape> = (0..3).map(|_| tree(&mut rng)).collect();
        let places: Vec<Placement> = [
            (-9.0, -6.0, 1.15),
            (-17.0, -22.0, 1.3),
            (14.0, -30.0, 1.0),
            (-4.0, -46.0, 1.4),
            (-30.0, -44.0, 1.6),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(x, z, scale))| Placement {
            position: Vec3::new(x, ground(x, z), z),
            yaw: i as f32 * 2.1,
            scale,
            seed: (i as f32 * 0.29 + 0.13).fract(),
        })
        .collect();
        let mut batch = Batch::new();
        batch.add(&oaks, &places, |p| (p.seed * 7.0) as usize % oaks.len());
        Self {
            land,
            trees: batch.finish(),
            dust: Particles::new(90, 31),
            land_pipeline: pipeline(MeadowShader, CullingMode::BackFace),
            tree_pipeline: pipeline(VariedToonShader, CullingMode::None),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            chain: Chain::new(vec![Effect::Kuwahara { radius: 3.0 }]),
            finish: ScreenPass::new(FinishShader),
        }
    }
}

impl Stage for Watercolour {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Flute],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::WaterDrop),
                0.0,
                0.45,
                0.13,
                -0.3,
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
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let breeze = Motion::Sway(Sway {
            amplitude: 0.25,
            speed: 0.9,
            wind: WIND,
        });

        self.land.draw(
            pass,
            &self.land_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.trees.draw(
            pass,
            &self.tree_pipeline,
            motion(breeze),
            VariedToonUniforms {
                toon: toon.clone(),
                variation: 0.12,
            },
        );
        scene.chip.draw(pass, view, &look());
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| SkyUniforms {
                view: view.clone(),
                sky: sky(),
            });
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 0.0),
                half_extent: Vec3::new(14.0, 5.0, 12.0),
                velocity: Vec3::new(0.3, 0.1, 0.0),
                wander: 0.8,
                size: (0.02, 0.05),
                colour: Vec3::new(1.4, 1.3, 1.1),
                twinkle: 0.8,
            },
        );
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let smoothed = self.chain.run(context, scene, image);
        let aspect = scene.extent.width as f32 / scene.extent.height as f32;
        Some(self.finish.run(
            context,
            scene,
            FinishUniforms {
                source: smoothed,
                time: scene.local_time,
                aspect,
            },
        ))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.05,
            bloom: 0.15,
            vignette: 0.0,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.1,
        }
    }
}
