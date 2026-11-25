//! The chip on a basalt island in a crater lake of lava: a ring of black columns, a cracked
//! glowing lake, lava bombs lobbed across the dark, embers rising and ash falling under a
//! smoke-choked sky lit orange from below.

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{F, clamp01, length, mix3, mul3, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar, value, value_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, scatter};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The island's surface, level with the chip's resting height in the other worlds.
const FLOOR: f32 = -1.95;
/// The lava's surface.
const LAVA: f32 = FLOOR - 0.8;
/// Where the island ends and the lake begins, and where the lake meets the crater wall.
const SHORE: f32 = 5.6;
const WALL: f32 = 52.0;
const BOMBS: usize = 7;

fn fog() -> Fog {
    Fog {
        colour: Vec3::new(0.2, 0.04, 0.015),
        density: 0.006,
        falloff: 0.025,
        base: LAVA,
    }
}

fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(0.25, 0.6, -0.5).normalise(),
        sun_colour: Vec3::new(0.35, 0.14, 0.08),
        sky: Vec3::new(0.05, 0.03, 0.035),
        ground: Vec3::new(0.35, 0.08, 0.02),
        bands: 0.0,
        rim: Vec3::new(0.3, 0.08, 0.02),
        fog: fog(),
    }
}

fn look(time: f32) -> Look {
    let pulse = 0.78 + 0.22 * (time * 1.3).sin();
    Look {
        substrate: Vec3::new(0.025, 0.025, 0.03),
        die: Vec3::new(0.012, 0.004, 0.003),
        gold: Vec3::new(1.0, 0.27, 0.04) * pulse,
        streams: Vec3::new(1.0, 0.3, 0.05) * pulse,
        key: Vec3::new(0.3, 0.5, -0.8).normalise(),
        key_colour: Vec3::new(1.4, 0.45, 0.18),
        sky: Vec3::new(0.12, 0.05, 0.04),
        ground: Vec3::new(1.1, 0.26, 0.04),
        softbox: Vec3::new(0.2, 0.06, 0.02),
    }
}

fn blend(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The island, the lava lake at a constant level, the crater wall and the ridges beyond it.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let island = FLOOR + (fbm_scalar(x * 0.25 + 3.0, z * 0.25, 3) - 0.5) * 0.25;
    let floor = blend(island, LAVA, step(SHORE - 0.5, SHORE + 0.6, r));
    let wall = step(WALL, 90.0, r);
    let crags = (fbm_scalar(x * 0.09 + 5.0, z * 0.09, 3) - 0.3) * 6.0;
    let ridges =
        step(100.0, 190.0, r) * (3.0 + 22.0 * fbm_scalar(x * 0.03 + 9.0, z * 0.03 + 2.0, 3));
    floor + wall * (7.0 + crags) + ridges
}

fn rock_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let ash = ((centroid.y - LAVA) / 18.0).clamp(0.0, 1.0) * 0.5
        + 0.5 * (normal.y * 1.2 - 0.3).clamp(0.0, 1.0);
    let base = Vec3::new(0.035, 0.03, 0.034);
    base * (1.0 + 1.6 * ash)
}

/// Basalt columns: a ring of short ones round the island's shore and tall ones standing out of
/// the crater wall.
fn columns(rng: &mut Rng) -> Shape {
    let mut variants = |heights: &[f32], radius: f32, sink: f32| -> Vec<Shape> {
        heights
            .iter()
            .map(|&h| {
                Shape::cylinder(
                    radius,
                    radius * 0.94,
                    h + sink,
                    6,
                    Vec3::new(0.04, 0.036, 0.042),
                )
                .transformed(&Mat4::translation(0.0, -sink, 0.0))
                .facet_variation(rng, 0.25)
            })
            .collect()
    };
    let tall = variants(&[2.5, 4.0, 5.5, 7.0], 0.9, 4.0);
    let short = variants(&[0.5, 0.8, 1.1, 1.5], 0.4, 1.0);

    let wall = scatter(
        rng,
        92.0,
        5.0,
        7000,
        ground,
        |x, z| {
            let r = (x * x + z * z).sqrt();
            ((WALL + 1.0..88.0).contains(&r) && value_scalar(x * 0.045 + 7.0, z * 0.045) > 0.6)
                .then_some(1.0)
        },
        (0.8, 1.7),
    );
    let shore = scatter(
        rng,
        8.0,
        0.95,
        900,
        ground,
        |x, z| {
            let r = (x * x + z * z).sqrt();
            ((SHORE - 0.2..SHORE + 0.9).contains(&r) && z < 1.0).then_some(1.0)
        },
        (0.7, 1.1),
    );
    let mut batch = Batch::new();
    batch.add(&tall, &wall, |p| (p.seed * 7.0) as usize % tall.len());
    batch.add(&short, &shore, |p| (p.seed * 7.0) as usize % short.len());
    batch.finish()
}

/// The lava's colour for a heat in [0, 1]: black crust, orange, then yellow-white.
#[inline(always)]
fn lava_ramp(heat: F) -> Vec3Simd {
    let crust = mix3(
        rgb(0.025, 0.008, 0.006),
        rgb(2.0, 0.25, 0.02),
        smoothstep(0.0, 0.55, heat),
    );
    mix3(crust, rgb(4.0, 1.6, 0.3), smoothstep(0.55, 1.0, heat))
}

/// The island, lake, wall and columns: basalt lit by the sun, the sky and the lake's glow, and
/// wherever the surface lies at the lake's level, flowing cracked lava.
struct CraterShader;

impl FragmentShaderSimd<KitVaryings> for CraterShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let (x, y, z) = (world.x, world.y, world.z);
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let normal = facing(v.normal().normalise(), to_eye);
        let r = (x * x + z * z).sqrt().fast_max(s(0.1));
        let height = y - s(LAVA);
        let t = view.frame.time;

        let grain = s(0.55) + value(x * s(2.3) + y * s(1.3), z * s(2.3) + y * s(3.9)) * s(0.9);
        let albedo = v.colour() * grain;
        let inward = Vec3Simd::new(-x / r, s(0.0), -z / r);
        let flicker = s(0.85) + s(0.15) * value(s(t * 0.9) + r * s(0.15), x * s(0.1));
        let facing_lake = clamp01(normal.dot(inward) * s(0.8) + s(0.2));
        let glow = (-height * s(0.13)).exp() * (s(0.15) + facing_lake * s(0.85)) * flicker;
        let lit = u.lighting.shade(view, world, normal, albedo, s(1.0));
        let rock = lit + mul3(albedo, rgb(1.0, 0.27, 0.05) * (glow * s(2.8)));

        let warp = fbm(x * s(0.1) + s(t * 0.02), z * s(0.1) - s(t * 0.015), 2);
        let n = fbm(
            x * s(0.3) + warp * s(3.0) + s(t * 0.04),
            z * s(0.3) + warp * s(3.0) - s(t * 0.03),
            4,
        );
        let crack = s(1.0) - smoothstep(0.0, 0.08, (n - s(0.5)).abs());
        let rim_heat = (-(r - s(SHORE)).fast_max(s(0.0)) * s(0.9)).exp() * s(0.4);
        let pulse = s(0.8) + s(0.2) * (s(t * 1.7) + n * s(9.0)).sin();
        let heat =
            clamp01((crack * s(0.95) + rim_heat + value(x * s(1.1), z * s(1.1)) * s(0.07)) * pulse);
        let near = lava_ramp(heat);
        let far = lava_ramp(s(0.4));
        let lava = u.lighting.fog.apply(
            mix3(near, far, smoothstep(30.0, 90.0, distance)),
            world,
            distance,
        );

        let veins =
            crack * smoothstep(0.7, 0.95, normal.y) * (s(1.0) - smoothstep(1.0, 1.8, height));
        let rock = rock + lava_ramp(s(0.55)) * (veins * s(0.12));
        let on_lava = (s(1.0) - smoothstep(0.02, 0.3, height)) * smoothstep(0.6, 0.9, normal.y);
        scene_colour(view, mix3(rock, lava, on_lava), world)
    }
}

/// Hot, self-lit things: lava bombs and their glowing tails.
struct EmberShader;

impl FragmentShaderSimd<KitVaryings> for EmberShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let world = v.world();
        let distance = length(world - splat3(u.view.eye));
        scene_colour(
            &u.view,
            u.lighting.fog.apply(v.colour(), world, distance),
            world,
        )
    }
}

/// The night sky: smoke boiling over a horizon glow, its undersides lit by the lake.
struct SmokeSky;

impl FragmentShaderSimd<ScreenVaryings> for SmokeSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [x, y] = v.ndc();
        let ray = view_ray(view, x, y);
        let t = s(view.frame.time);
        let up = ray.y.fast_max(s(0.0));
        let glow = (-up * s(4.0)).exp();
        let base = mix3(rgb(0.008, 0.006, 0.01), rgb(0.5, 0.1, 0.03), glow);
        let reach = s(1.0) / up.fast_max(s(0.06));
        let (px, pz) = (ray.x * reach + t * s(0.25), ray.z * reach - t * s(0.1));
        let billow = fbm(px * s(0.5), pz * s(0.5), 4);
        let smoke = smoothstep(0.35, 0.75, billow);
        let underlit = rgb(0.9, 0.2, 0.04) * ((-up * s(3.2)).exp() * (s(1.0) - billow) * s(0.75));
        let pall = rgb(0.012, 0.01, 0.012) + underlit;
        let colour = mix3(base, pall, smoke * smoothstep(0.0, 0.12, up));
        to_colour(colour, s(SKY_DISTANCE))
    }
}

pub struct VolcanoCrater {
    ground: Shape,
    columns: Shape,
    bombs: Shape,
    embers: Particles,
    sparks: Particles,
    ash: Particles,
    crater: SimdPipeline<KitVertexShader, CraterShader>,
    hot: SimdPipeline<KitVertexShader, EmberShader>,
    sky: SimdPipeline<ScreenVertexShader, SmokeSky>,
}

impl VolcanoCrater {
    pub fn new() -> Self {
        let mut rng = Rng::new(404);
        Self {
            ground: terrain(&ring_radii(1.5, 230.0, 46), 56, ground, rock_colour),
            columns: columns(&mut rng),
            bombs: Shape::new(),
            embers: Particles::new(420, 41),
            sparks: Particles::new(140, 42),
            ash: Particles::new(260, 43),
            crater: pipeline(CraterShader, CullingMode::None),
            hot: pipeline(EmberShader, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, SmokeSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

/// The lava bombs at `time`: each is lobbed from a spot on the lake in a parabola that ends
/// back in the lava, trailing a few cooling clots. Clots shrink away as they pass close to `eye`.
fn bombs(time: f32, eye: Vec3) -> Shape {
    let mut rng = Rng::new(77);
    let mut shape = Shape::new();
    for i in 0..BOMBS {
        let (r0, a0) = (rng.range(9.0, 22.0), rng.range(0.0, std::f32::consts::TAU));
        let (r1, a1) = (rng.range(9.0, 22.0), a0 + rng.range(-1.0, 1.0));
        let apex = rng.range(7.0, 14.0);
        let period = 2.0 * (2.0 * apex / 9.0_f32).sqrt() + rng.range(1.5, 3.5);
        let size = rng.range(0.28, 0.5);
        let (start, end) = (
            Vec3::new(r0 * a0.cos(), 0.0, r0 * a0.sin()),
            Vec3::new(r1 * a1.cos(), 0.0, r1 * a1.sin()),
        );
        let phase = (time / period + rng.unit() + i as f32 * 0.37).fract();
        for clot in 0..8 {
            let u = phase - clot as f32 * 0.012;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let ground = start + (end - start) * u;
            let position = Vec3::new(ground.x, LAVA + 4.0 * apex * u * (1.0 - u), ground.z);
            let cool = 1.0 - clot as f32 * 0.11;
            let near = ((position - eye).length() - 5.0) / 6.0;
            let fade = near.clamp(0.0, 1.0);
            if fade == 0.0 {
                continue;
            }
            let colour = Vec3::new(5.0, 1.5, 0.25) * cool * cool;
            shape.append(
                &Shape::icosphere(
                    size * cool * fade,
                    1,
                    0.3,
                    i as f32 * 3.1 + clot as f32,
                    colour,
                )
                .transformed(&Mat4::translation_vec(position)),
            );
        }
    }
    shape
}

impl Stage for VolcanoCrater {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Trombone],
            treatment: Treatment::Driven,
            ambience: vec![
                Ambience::bed(Source::Synth(Accent::Lava), 0.2, 0.0),
                Ambience::bed(Source::Recording("fire", 0.0), 0.14, -0.35),
            ],
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.bombs = bombs(scene.local_time, scene.view.eye);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let still = KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };

        self.ground
            .draw(pass, &self.crater, still.clone(), toon.clone());
        self.columns
            .draw(pass, &self.crater, still.clone(), toon.clone());
        self.bombs.draw(pass, &self.hot, still, toon);
        scene.chip.draw(pass, view, &look(view.frame.time));

        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| view.clone());

        self.embers.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 6.0, -6.0),
                half_extent: Vec3::new(24.0, 10.0, 24.0),
                velocity: Vec3::new(0.5, 2.0, -0.3),
                wander: 1.2,
                size: (0.03, 0.08),
                colour: Vec3::new(3.0, 0.85, 0.12),
                twinkle: 5.0,
            },
        );
        self.sparks.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 3.0),
                half_extent: Vec3::new(8.0, 4.0, 8.0),
                velocity: Vec3::new(0.3, 1.2, -0.2),
                wander: 0.5,
                size: (0.012, 0.03),
                colour: Vec3::new(4.0, 1.4, 0.3),
                twinkle: 8.0,
            },
        );
        self.ash.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 4.0, 0.0),
                half_extent: Vec3::new(20.0, 8.0, 20.0),
                velocity: Vec3::new(0.9, -0.5, 0.2),
                wander: 0.8,
                size: (0.02, 0.05),
                colour: Vec3::new(0.35, 0.13, 0.07),
                twinkle: 1.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.8,
            vignette: 0.5,
            grain: 0.02,
            aberration: 0.002,
            desaturate: 0.0,
        }
    }
}
