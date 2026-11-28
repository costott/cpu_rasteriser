//! night-vision: the view through an image-intensifier scope. A moonless field under stars:
//! pines whose foliage glows in the infrared, grass, boulders, a fence and a watchtower, all
//! lit by the scope's own illuminator and fading into the dark. The scope's electronics turn
//! it into green phosphor with gain noise, the round lens mask, a rolling brightness band and
//! bursts of scan jitter.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::simd::{from_colour, hash2, s, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVertexShader, Motion, Sway, ToonShader, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::noise::fbm_scalar;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::{Chain, Effect, Noise, Ramp, ScreenPass};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, scatter};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::sky::{Sky, SkyShader, SkyUniforms};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::overlay::{Align, Line, Overlay, Text};
use crate::themes::{Scene, Stage};

/// The height of the ground where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -3.4;
/// The lens's radius, in half frame heights.
const LENS: f32 = 1.15;

fn reflectance(r: f32) -> Vec3 {
    Vec3::new(r, r, r)
}

fn smooth(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Flat round the chip, low rolling hills beyond.
fn ground(x: f32, z: f32) -> f32 {
    let distance = (x * x + z * z).sqrt();
    FLOOR + smooth(7.0, 30.0, distance) * (fbm_scalar(x * 0.035, z * 0.035, 3) - 0.35) * 9.0
}

pub struct NightVision {
    look: Look,
    land: Shape,
    pines: Shape,
    grass: Shape,
    works: Shape,
    solid: SimdPipeline<KitVertexShader, ToonShader>,
    sky: SimdPipeline<ScreenVertexShader, SkyShader>,
    insects: Particles,
    overlay: Overlay,
    optics: ScreenPass<OpticsShader>,
    chain: Chain,
}

impl NightVision {
    pub fn new() -> Self {
        let mut rng = Rng::new(51);
        let land = terrain(&ring_radii(1.5, 180.0, 44), 64, ground, |centroid, _| {
            let grain = fbm_scalar(centroid.x * 0.3, centroid.z * 0.3, 2);
            reflectance(0.2 + 0.22 * grain)
        });

        let clear = |x: f32, z: f32| (x * x + z * z).sqrt() > 10.0;
        let trees = scatter(
            &mut rng,
            80.0,
            7.5,
            3500,
            ground,
            |x, z| (clear(x, z) && (x.abs() > 12.0 && z < 6.0 || z < -30.0)).then_some(1.0),
            (0.8, 1.6),
        );
        let mut batch = Batch::new();
        batch.add(&[pine()], &trees, |_| 0);
        let pines = batch.finish();

        let tufts = scatter(
            &mut rng,
            45.0,
            1.4,
            5000,
            ground,
            |x, z| (clear(x, z) && z < 10.0).then_some(1.0),
            (0.7, 1.4),
        );
        let mut batch = Batch::new();
        batch.add(&[tuft()], &tufts, |_| 0);
        let grass = batch.finish();

        let mut works = Shape::new();
        for _ in 0..7 {
            let (x, z) = (rng.range(-30.0, 30.0), rng.range(-40.0, -9.0));
            if (x * x + z * z).sqrt() < 11.0 {
                continue;
            }
            let boulder =
                Shape::icosphere(rng.range(0.8, 1.8), 1, 0.3, rng.unit(), reflectance(0.3));
            works.append(&boulder.transformed(&Mat4::translation(x, ground(x, z), z)));
        }
        works.append(&fence());
        works.append(&tower());

        Self {
            look: Look {
                substrate: reflectance(0.3),
                die: reflectance(0.18),
                gold: reflectance(0.7),
                streams: reflectance(1.1),
                key: Vec3::new(0.0, 0.3, 1.0).normalise(),
                key_colour: reflectance(0.8),
                sky: reflectance(0.1),
                ground: reflectance(0.1),
                softbox: Vec3::ZERO,
            },
            land,
            pines,
            grass,
            works,
            solid: pipeline(ToonShader, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            insects: Particles::new(70, 52),
            overlay: Overlay::new(),
            optics: ScreenPass::new(OpticsShader),
            chain: Chain::new(vec![
                Effect::Ramp(Ramp::night_vision(1.2)),
                Effect::Noise(Noise {
                    amount: 0.32,
                    floor: 0.012,
                    cell: 1.6,
                    chroma: 0.0,
                }),
            ]),
        }
    }
}

/// A pine: a dark trunk under three tiers of foliage, which the infrared lights brightly.
fn pine() -> Shape {
    let mut pine = Shape::cylinder(0.3, 0.18, 2.6, 6, reflectance(0.12));
    for (i, (radius, height)) in [(2.1, 2.6), (1.6, 2.4), (1.05, 2.1)]
        .into_iter()
        .enumerate()
    {
        let tier = Shape::cone(radius, height, 7, reflectance(0.3 + 0.04 * i as f32));
        pine.append(&tier.transformed(&Mat4::translation(0.0, 1.5 + i as f32 * 1.7, 0.0)));
    }
    pine.sway_by_height(0.5, 8.0, 1.5)
}

/// A small clump of grass blades.
fn tuft() -> Shape {
    let mut tuft = Shape::new();
    for (dx, dz, height) in [
        (0.0, 0.0, 1.0),
        (0.25, 0.1, 0.75),
        (-0.2, 0.15, 0.85),
        (0.05, -0.25, 0.7),
    ] {
        let blade = Shape::cone(0.1, height, 4, reflectance(0.42)).sway_by_height(0.0, height, 1.0);
        tuft.append(&blade.transformed(&Mat4::translation(dx, 0.0, dz)));
    }
    tuft
}

/// A post-and-rail fence across the field.
fn fence() -> Shape {
    let (z, wood) = (-16.0, reflectance(0.22));
    let mut fence = Shape::new();
    for i in -9..=9 {
        let x = i as f32 * 3.0;
        let base = ground(x, z);
        fence.append(&Shape::cuboid(
            Vec3::new(x - 0.12, base, z - 0.12),
            Vec3::new(x + 0.12, base + 2.0, z + 0.12),
            wood,
        ));
        if i < 9 {
            for lift in [0.8, 1.55] {
                let (y0, y1) = (ground(x, z) + lift, ground(x + 3.0, z) + lift);
                fence.quad(
                    [
                        Vec3::new(x, y0, z),
                        Vec3::new(x + 3.0, y1, z),
                        Vec3::new(x + 3.0, y1 + 0.16, z),
                        Vec3::new(x, y0 + 0.16, z),
                    ],
                    wood,
                );
            }
        }
    }
    fence
}

/// A lookout on four legs: a platform, a hut and a roof.
fn tower() -> Shape {
    let (x, z, wood) = (-24.0, -52.0, reflectance(0.2));
    let base = ground(x, z);
    let mut tower = Shape::new();
    for (dx, dz) in [(-1.6, -1.6), (1.6, -1.6), (-1.6, 1.6), (1.6, 1.6)] {
        tower.append(&Shape::limb(
            Vec3::new(x + dx * 1.3, base, z + dz * 1.3),
            Vec3::new(x + dx, base + 9.0, z + dz),
            0.22,
            0.18,
            4,
            wood,
        ));
    }
    tower.append(&Shape::cuboid(
        Vec3::new(x - 2.3, base + 9.0, z - 2.3),
        Vec3::new(x + 2.3, base + 9.3, z + 2.3),
        wood,
    ));
    tower.append(&Shape::cuboid(
        Vec3::new(x - 1.8, base + 9.3, z - 1.8),
        Vec3::new(x + 1.8, base + 11.4, z + 1.8),
        reflectance(0.14),
    ));
    tower.append(
        &Shape::cone(3.0, 1.6, 4, reflectance(0.3)).transformed(&Mat4::translation(
            x,
            base + 11.4,
            z,
        )),
    );
    tower
}

/// The scope's own infrared torch, from the viewer's side, and a fog that swallows the distance.
fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(0.0, 0.35, 1.0).normalise(),
        sun_colour: reflectance(0.9),
        sky: reflectance(0.2),
        ground: reflectance(0.1),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: reflectance(0.012),
            density: 0.016,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

fn night_sky() -> Sky {
    Sky {
        zenith: reflectance(0.002),
        horizon: reflectance(0.03),
        sun: Vec3::new(0.0, 1.0, 0.0),
        sun_colour: Vec3::ZERO,
        sun_size: 0.05,
        glow: 0.0,
        clouds: None,
        stars: 0.7,
    }
}

impl Stage for NightVision {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Chip],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::ScopeWhine),
                0.0,
                0.45,
                0.06,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let window = (scene.extent.width, scene.extent.height);
        let pixel = 2.6 * window.1 as f32 / 720.0;
        let colour = reflectance(1.4);
        let range = format!("RNG {:3} M", 120 + (scene.local_time * 4.0) as i32 % 5);
        let mut text = Text::default();
        for (words, x, y, align) in [
            ("IR ILLUM ON", -0.42, 0.78, Align::Left),
            ("GAIN AUTO", -0.42, 0.7, Align::Left),
            (range.as_str(), 0.42, 0.78, Align::Right),
            ("GEN 3", 0.42, 0.7, Align::Right),
            ("N 352", 0.0, -0.8, Align::Centre),
        ] {
            text.line(
                Line {
                    text: words,
                    x,
                    y,
                    align,
                    colour,
                },
                pixel,
                window,
            );
        }
        self.overlay.set(text);
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
            amplitude: 0.12,
            speed: 1.4,
            wind: Vec2::new(1.0, 0.3),
        });

        scene.chip.draw(pass, view, &self.look);
        self.land
            .draw(pass, &self.solid, motion(Motion::Still), toon.clone());
        self.works
            .draw(pass, &self.solid, motion(Motion::Still), toon.clone());
        self.pines.draw(
            pass,
            &self.solid,
            motion(Motion::Sway(Sway {
                amplitude: 0.05,
                speed: 0.8,
                wind: Vec2::new(1.0, 0.3),
            })),
            toon.clone(),
        );
        self.grass.draw(pass, &self.solid, motion(breeze), toon);

        let sky = SkyUniforms {
            view: view.clone(),
            sky: night_sky(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| sky.clone());
        self.insects.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, FLOOR + 3.0, -8.0),
                half_extent: Vec3::new(18.0, 3.5, 14.0),
                velocity: Vec3::new(0.4, 0.1, 0.0),
                wander: 1.6,
                size: (0.05, 0.11),
                colour: reflectance(1.8),
                twinkle: 6.0,
            },
        );
        self.overlay.draw(pass);
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let uniforms = OpticsUniforms {
            source: image,
            time: scene.local_time,
            aspect: scene.view.aspect,
        };
        let optics = self.optics.run(context, scene, uniforms);
        Some(self.chain.run(context, scene, optics))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.9,
            vignette: 0.25,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[derive(Clone)]
struct OpticsUniforms {
    source: Arc<TextureSampler>,
    time: f32,
    aspect: f32,
}

/// The scope's electronics before the phosphor: the round lens with its soft edge and dimming
/// towards it, a brighter band rolling down the tube, and bursts of horizontal jitter that
/// shift whole groups of scan lines.
struct OpticsShader;

impl FragmentShaderSimd<ScreenVaryings> for OpticsShader {
    type Uniforms = OpticsUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &OpticsUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));

        let step = (u.time * 24.0).floor();
        let burst = s(step * 0.25).floor();
        let active = (hash2(burst, s(7.0)) - s(0.78)).fast_max(s(0.0)) * s(4.5);
        let group = (y * s(90.0)).floor();
        let shift = (hash2(group, s(step % 211.0)) - s(0.5)) * s(0.03) * active.fast_min(s(1.0));
        let sample = u.source.sample_linear_clamp_simd([x + shift, y]);

        let (px, py) = (nx * s(u.aspect), ny);
        let radius = (px * px + py * py).sqrt() * s(1.0 / LENS);
        let lens = smoothstep(1.0, 0.94, radius);
        let dim = s(1.0) - radius * radius * s(0.45);
        let roll = (y - s(1.0 - (u.time * 0.18) % 1.0)).abs();
        let band = s(1.0) + (s(1.0) - smoothstep(0.0, 0.07, roll)) * s(0.22);

        let colour = from_colour(sample) * (lens * dim * band);
        to_colour(colour, sample.a)
    }
}
