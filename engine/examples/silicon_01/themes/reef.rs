//! An underwater reef: the chip a barnacled relic on a rippled sandy seabed, kelp and coral
//! swaying round it in the current, a shoal of fish circling, light shafts, caustics and bubbles.

use std::f32::consts::{FRAC_PI_2, TAU};

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{F, clamp01, mix3, pow2n, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Orbit, Sway, ToonUniforms,
    pipeline,
};
use crate::themes::kit::dress::{chip_shadow, rim_point};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{cellular, fbm_scalar, value};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The seabed's height where the chip stands.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: 0.22,
    y: 0.93,
    z: 0.28,
};
const CURRENT: Vec2 = Vec2 { x: 0.94, y: 0.34 };
const WATER: Vec3 = Vec3 {
    x: 0.02,
    y: 0.27,
    z: 0.38,
};

fn seabed(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let calm = (r / 6.0).clamp(0.0, 1.0);
    let dunes = (fbm_scalar(x * 0.05 + 3.0, z * 0.05, 3) - 0.5) * 2.2;
    let rise = ((r - 55.0) / 70.0).clamp(0.0, 1.0);
    FLOOR
        + calm * calm * dunes
        + rise * rise * (3.0 + 14.0 * fbm_scalar(x * 0.03, z * 0.03 + 5.0, 3))
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.1, 1.6, 1.6),
        sky: Vec3::new(0.17, 0.42, 0.55),
        ground: Vec3::new(0.05, 0.14, 0.18),
        bands: 0.0,
        rim: Vec3::new(0.02, 0.1, 0.12),
        fog: Fog {
            colour: WATER,
            density: 0.034,
            falloff: 0.0,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.22, 0.38, 0.34),
        die: Vec3::new(0.01, 0.3, 0.32),
        gold: Vec3::new(0.7, 0.5, 0.24),
        streams: Vec3::new(0.25, 1.0, 0.85),
        key: light.sun,
        key_colour: light.sun_colour,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.12, 0.34, 0.4),
    }
}

/// Bright web-like light cast through the surface: two drifting cellular layers multiplied.
#[inline(always)]
fn caustics(x: F, z: F, t: f32) -> F {
    let a = s(1.0)
        - cellular(
            x * s(0.42) + s(t * 0.07),
            z * s(0.42) - s(t * 0.05),
            t * 0.7,
        );
    let b = s(1.0)
        - cellular(
            x * s(0.61) - s(t * 0.06) + s(9.0),
            z * s(0.61) + s(t * 0.08),
            t * 0.55 + 3.0,
        );
    let web = pow2n(a * b, 2) * s(5.0);
    web.fast_min(s(1.6))
}

#[derive(Clone)]
struct ReefUniforms {
    toon: ToonUniforms,
    /// How strongly the surface takes caustics, which light upward-facing surfaces most.
    caustics: f32,
}

/// Lights every surface of the reef: vertex colour in the sun filtered through the surface, so
/// upward faces ripple with caustics, and the chip's own shadow on the ground.
struct ReefShader;

impl FragmentShaderSimd<KitVaryings> for ReefShader {
    type Uniforms = ReefUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ReefUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let normal = facing(v.normal().normalise(), splat3(view.eye) - world);
        let up = clamp01(normal.y * s(0.8) + s(0.2));
        let web = caustics(world.x, world.z + world.y * s(0.4), view.frame.time);
        let beams = s(0.4) + web * s(u.caustics * 0.6);
        let shadow = (s(1.0) - chip_shadow(view, u.toon.lighting.sun, world) * s(0.85))
            * (s(0.3) + up * beams);
        let tone = s(0.9) + value(world.x * s(0.7), world.z * s(0.7)) * s(0.2);
        let albedo = v.colour() * tone;
        let colour = u.toon.lighting.shade(view, world, normal, albedo, shadow);
        scene_colour(view, colour, world)
    }
}

/// The water around the world: deep blue at the horizon, brightening towards the surface, with
/// slow shafts of light.
struct WaterShader;

impl FragmentShaderSimd<ScreenVaryings> for WaterShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [x, y] = v.ndc();
        let ray = view_ray(view, x, y);
        let t = s(view.frame.time);
        let up = ray.y.fast_max(s(0.0));
        let mut colour = mix3(
            splat3(WATER),
            rgb(0.08, 0.55, 0.62),
            smoothstep(0.0, 0.85, up).sqrt(),
        );
        let slant = ray.x / (ray.y + s(0.55)) * s(3.2);
        let beams = smoothstep(0.35, 0.85, value(slant + t * s(0.12), t * s(0.05)))
            * smoothstep(0.02, 0.5, ray.y)
            * (s(1.0) - smoothstep(0.55, 1.0, up) * s(0.4));
        colour = colour + rgb(0.35, 0.8, 0.8) * (beams * s(0.9));
        to_colour(colour, s(SKY_DISTANCE))
    }
}

fn kelp_frond(rng: &mut Rng, height: f32) -> Shape {
    let points = 9;
    let phase = rng.range(0.0, TAU);
    let across = Vec3::new(rng.range(-1.0, 1.0), 0.0, rng.range(-1.0, 1.0)).normalise();
    let spine: Vec<Vec3> = (0..points)
        .map(|i| {
            let t = i as f32 / (points - 1) as f32;
            Vec3::new(
                (t * 3.0 + phase).sin() * 0.25 * t,
                t * height,
                (t * 2.4 + phase).cos() * 0.2 * t,
            )
        })
        .collect();
    let widths: Vec<f32> = (0..points)
        .map(|i| {
            let t = i as f32 / (points - 1) as f32;
            (0.55 * (1.0 - t * t)).max(0.04) * if i % 2 == 0 { 1.0 } else { 0.75 }
        })
        .collect();
    Shape::ribbon(&spine, &widths, across, Vec3::ONE)
        .gradient_by_height(
            0.0,
            height,
            Vec3::new(0.08, 0.16, 0.05),
            Vec3::new(0.42, 0.5, 0.12),
        )
        .sway_by_height(0.0, height, 1.5)
}

fn kelp_cluster(rng: &mut Rng) -> Shape {
    let mut cluster = Shape::new();
    for _ in 0..rng.index(3) + 3 {
        let offset = Vec3::new(rng.range(-0.4, 0.4), 0.0, rng.range(-0.4, 0.4));
        let height = rng.range(3.5, 8.0);
        let frond = kelp_frond(rng, height).transformed(&Mat4::translation_vec(offset));
        cluster.append(&frond);
    }
    cluster
}

fn sea_grass(rng: &mut Rng) -> Shape {
    let mut tuft = Shape::new();
    for _ in 0..6 {
        let angle = rng.range(0.0, TAU);
        let height = rng.range(0.5, 1.2);
        let base = Vec3::new(angle.cos() * 0.1, 0.0, angle.sin() * 0.1);
        let side = Vec3::new(-angle.sin(), 0.0, angle.cos()) * 0.05;
        let tip = base + Vec3::new(angle.cos() * 0.2, height, angle.sin() * 0.2);
        tuft.triangle(base - side, base + side, tip, Vec3::ONE);
        tuft.triangle(base + side, base - side, tip, Vec3::ONE);
    }
    tuft.gradient_by_height(
        0.0,
        1.2,
        Vec3::new(0.05, 0.2, 0.12),
        Vec3::new(0.35, 0.65, 0.3),
    )
    .sway_by_height(0.0, 1.2, 1.0)
}

fn branching_coral(rng: &mut Rng, colour: Vec3, tip: Vec3) -> Shape {
    struct Growth<'a> {
        shape: Shape,
        rng: &'a mut Rng,
        colour: Vec3,
        tip: Vec3,
    }

    impl Growth<'_> {
        fn grow(&mut self, from: Vec3, direction: Vec3, length: f32, radius: f32, depth: u32) {
            let to = from + direction * length;
            let shade = if depth == 0 { self.tip } else { self.colour };
            self.shape
                .append(&Shape::limb(from, to, radius, radius * 0.7, 4, shade));
            if depth == 0 {
                let bud = Shape::icosphere(radius * 0.9, 0, 0.0, 0.0, self.tip)
                    .transformed(&Mat4::translation_vec(to));
                self.shape.append(&bud);
                return;
            }
            for k in 0..3 {
                let azimuth = TAU * (k as f32 / 3.0) + self.rng.range(-0.5, 0.5);
                let spread = self.rng.range(0.45, 0.8);
                let child = Vec3::new(azimuth.cos() * spread, 1.0, azimuth.sin() * spread);
                let child = (direction + child).normalise();
                self.grow(to, child, length * 0.78, radius * 0.72, depth - 1);
            }
        }
    }
    let trunk = rng.range(0.5, 0.8);
    let mut growth = Growth {
        shape: Shape::new(),
        rng,
        colour,
        tip,
    };
    growth.grow(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), trunk, 0.1, 3);
    let coral = growth.shape;
    coral.sway_by_height(0.0, 2.6, 1.3)
}

fn brain_coral(rng: &mut Rng, colour: Vec3) -> Shape {
    Shape::icosphere(1.0, 2, 0.07, rng.range(0.0, 50.0), colour)
        .transformed(&Mat4::scaling(1.0, 0.65, 1.0))
        .colour_by(|p, _| colour * (0.8 + 0.4 * (p.x * 7.0).sin().abs() * (p.z * 6.0).cos().abs()))
}

fn table_coral(colour: Vec3) -> Shape {
    let mut coral = Shape::cylinder(0.3, 0.2, 0.8, 5, colour * 0.7);
    coral.append(&Shape::lathe(
        &[(0.0, 0.6), (0.9, 0.78), (1.05, 0.92), (0.0, 1.0)],
        9,
        colour,
    ));
    coral
}

fn fish(body: Vec3, belly: Vec3, fin: Vec3) -> Shape {
    let profile = [
        (0.0, -0.5),
        (0.12, -0.3),
        (0.2, -0.05),
        (0.18, 0.2),
        (0.07, 0.4),
        (0.0, 0.5),
    ];
    let spindle = Shape::lathe(&profile, 6, body)
        .transformed(&Mat4::rotate_z(-FRAC_PI_2))
        .colour_by(move |_, n| if n.y < -0.2 { belly } else { body });
    let mut shape = spindle;
    let tail = |dz: f32| Vec3::new(-0.78, dz, 0.0);
    shape.triangle(Vec3::new(-0.42, 0.0, 0.0), tail(0.26), tail(-0.26), fin);
    shape.triangle(Vec3::new(-0.42, 0.0, 0.0), tail(-0.26), tail(0.26), fin);
    shape.triangle(
        Vec3::new(0.0, 0.16, 0.0),
        Vec3::new(-0.3, 0.16, 0.0),
        Vec3::new(-0.2, 0.36, 0.0),
        fin,
    );
    shape.triangle(
        Vec3::new(-0.3, 0.16, 0.0),
        Vec3::new(0.0, 0.16, 0.0),
        Vec3::new(-0.2, 0.36, 0.0),
        fin,
    );
    shape.sway_by(|p| ((0.1 - p.x) / 0.88).clamp(0.0, 1.0).powf(1.5))
}

fn shoal(rng: &mut Rng, count: usize, sizes: (f32, f32)) -> Shape {
    let variants = [
        fish(
            Vec3::new(0.15, 0.4, 0.9),
            Vec3::new(0.85, 0.9, 0.95),
            Vec3::new(1.0, 0.8, 0.1),
        ),
        fish(
            Vec3::new(0.95, 0.75, 0.1),
            Vec3::new(1.0, 0.95, 0.6),
            Vec3::new(0.2, 0.3, 0.8),
        ),
        fish(
            Vec3::new(0.7, 0.8, 0.85),
            Vec3::new(0.95, 0.97, 1.0),
            Vec3::new(0.5, 0.65, 0.75),
        ),
        fish(
            Vec3::new(0.95, 0.35, 0.1),
            Vec3::new(1.0, 0.8, 0.6),
            Vec3::new(1.0, 0.5, 0.1),
        ),
    ];
    let placements: Vec<Placement> = (0..count)
        .map(|i| Placement {
            position: Vec3::ZERO,
            yaw: 0.0,
            scale: rng.range(sizes.0, sizes.1),
            seed: (i as f32 + rng.unit() * 0.8) / count as f32,
        })
        .collect();
    let mut batch = Batch::new();
    batch.add(&variants, &placements, |p| ((p.seed * 997.0) as usize) % 4);
    batch.finish()
}

/// Barnacle clusters, coral growths and weed on the chip, in its own space.
fn chip_extras(rng: &mut Rng) -> Shape {
    let mut extras = Shape::new();
    let shell = Vec3::new(0.82, 0.8, 0.72);
    for _ in 0..26 {
        let (centre, _) = rim_point(rng);
        for _ in 0..rng.index(5) + 4 {
            let spot = centre + Vec3::new(rng.range(-0.2, 0.2), 0.0, rng.range(-0.2, 0.2));
            let height = rng.range(0.05, 0.15);
            extras.append(&Shape::limb(
                spot,
                spot + Vec3::new(0.0, height, 0.0),
                rng.range(0.05, 0.1),
                0.045,
                5,
                shell,
            ));
            extras.append(&Shape::limb(
                spot + Vec3::new(0.0, height, 0.0),
                spot + Vec3::new(0.0, height + 0.003, 0.0),
                0.03,
                0.03,
                5,
                Vec3::new(0.08, 0.06, 0.06),
            ));
        }
    }
    for _ in 0..5 {
        let (spot, _) = rim_point(rng);
        extras.append(
            &branching_coral(rng, Vec3::new(0.9, 0.35, 0.45), Vec3::new(1.0, 0.7, 0.7))
                .transformed(&(Mat4::translation_vec(spot) * Mat4::scaling(0.28, 0.28, 0.28))),
        );
    }
    for _ in 0..7 {
        let (spot, _) = rim_point(rng);
        extras.append(
            &Shape::icosphere(
                1.0,
                1,
                0.3,
                rng.range(0.0, 30.0),
                Vec3::new(0.12, 0.35, 0.18),
            )
            .transformed(&(Mat4::translation_vec(spot) * Mat4::scaling(0.3, 0.12, 0.3))),
        );
    }
    extras
}

pub struct Reef {
    seabed: Shape,
    kelp: Shape,
    grass: Shape,
    coral: Shape,
    rocks: Shape,
    shoal: Shape,
    big_fish: Shape,
    extras: Shape,
    bubbles: Particles,
    plankton: Particles,
    surface: SimdPipeline<KitVertexShader, ReefShader>,
    foliage: SimdPipeline<KitVertexShader, ReefShader>,
    water: SimdPipeline<ScreenVertexShader, WaterShader>,
}

impl Reef {
    pub fn new() -> Self {
        let mut rng = Rng::new(31);
        let seabed_shape = terrain(&ring_radii(1.5, 220.0, 44), 56, seabed, |c, n| {
            let r = (c.x * c.x + c.z * c.z).sqrt();
            let sand = lerp3(
                Vec3::new(0.42, 0.37, 0.26),
                Vec3::new(0.55, 0.48, 0.34),
                fbm_scalar(c.x * 0.2, c.z * 0.2, 2),
            );
            let deep = lerp3(
                sand,
                Vec3::new(0.3, 0.38, 0.34),
                ((r - 50.0) / 50.0).clamp(0.0, 1.0),
            );
            deep * (0.95 + 0.1 * crate::simd::hash2_scalar(c.x, c.z)) * (0.8 + 0.2 * n.y)
        });

        let clear = |x: f32, z: f32| {
            let r = (x * x + z * z).sqrt();
            if r < 5.0 || (x.abs() < 3.0 && z > 0.0) {
                None
            } else {
                Some(1.0)
            }
        };
        let clusters: Vec<Shape> = (0..5).map(|_| kelp_cluster(&mut rng)).collect();
        let kelp_spots = scatter(
            &mut rng,
            40.0,
            3.4,
            4000,
            seabed,
            |x, z| clear(x, z).filter(|_| (x * x + z * z).sqrt() > 7.0),
            (0.8, 1.4),
        );
        let mut kelp = Batch::new();
        kelp.add(&clusters, &kelp_spots, |p| (p.seed * 31.0) as usize % 5);
        let kelp = kelp.finish();

        let tufts: Vec<Shape> = (0..3).map(|_| sea_grass(&mut rng)).collect();
        let grass_spots = scatter(
            &mut rng,
            22.0,
            0.9,
            6000,
            seabed,
            |x, z| {
                if (x * x + z * z).sqrt() < 3.2 || (x.abs() < 2.8 && z > 0.0) {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.9, 1.6),
        );
        let mut grass = Batch::new();
        grass.add(&tufts, &grass_spots, |p| (p.seed * 13.0) as usize % 3);
        let grass = grass.finish();

        let palette = [
            (Vec3::new(0.95, 0.4, 0.3), Vec3::new(1.0, 0.75, 0.55)),
            (Vec3::new(0.8, 0.25, 0.55), Vec3::new(1.0, 0.65, 0.85)),
            (Vec3::new(0.95, 0.6, 0.15), Vec3::new(1.0, 0.9, 0.5)),
            (Vec3::new(0.55, 0.35, 0.8), Vec3::new(0.85, 0.7, 1.0)),
        ];
        let branching: Vec<Shape> = (0..8)
            .map(|i| branching_coral(&mut rng, palette[i % 4].0, palette[i % 4].1))
            .collect();
        let coral_spots = scatter(
            &mut rng,
            24.0,
            3.6,
            3000,
            seabed,
            |x, z| clear(x, z).filter(|_| (x * x + z * z).sqrt() > 5.5),
            (0.9, 1.9),
        );
        let mut coral = Batch::new();
        let (branching_spots, other): (Vec<Placement>, Vec<Placement>) =
            coral_spots.into_iter().partition(|p| p.seed < 0.55);
        coral.add(&branching, &branching_spots, |p| {
            (p.seed * 29.0) as usize % 8
        });
        let brains: Vec<Shape> = palette
            .iter()
            .map(|c| brain_coral(&mut rng, c.0 * 0.85))
            .collect();
        let tables: Vec<Shape> = palette.iter().map(|c| table_coral(c.1 * 0.95)).collect();
        let (brain_spots, table_spots): (Vec<Placement>, Vec<Placement>) =
            other.into_iter().partition(|p| p.seed < 0.77);
        coral.add(&brains, &brain_spots, |p| (p.seed * 29.0) as usize % 4);
        coral.add(&tables, &table_spots, |p| (p.seed * 29.0) as usize % 4);
        let coral = coral.finish();

        let boulders: Vec<Shape> = (0..3)
            .map(|_| {
                Shape::icosphere(
                    1.0,
                    1,
                    0.3,
                    rng.range(0.0, 90.0),
                    Vec3::new(0.3, 0.36, 0.36),
                )
                .transformed(&Mat4::scaling(1.0, 0.7, 0.9))
                .colour_by(|_, n| {
                    lerp3(
                        Vec3::new(0.25, 0.3, 0.3),
                        Vec3::new(0.2, 0.42, 0.25),
                        (n.y - 0.2).clamp(0.0, 1.0),
                    )
                })
            })
            .collect();
        let rock_spots = scatter(&mut rng, 34.0, 6.0, 800, seabed, clear, (1.0, 3.2));
        let mut rocks = Batch::new();
        rocks.add(&boulders, &rock_spots[..rock_spots.len().min(14)], |p| {
            (p.seed * 7.0) as usize % 3
        });
        let rocks = rocks.finish();

        Self {
            seabed: seabed_shape,
            kelp,
            grass,
            coral,
            rocks,
            shoal: shoal(&mut rng, 110, (0.4, 0.6)),
            big_fish: shoal(&mut rng, 3, (1.4, 1.9)),
            extras: chip_extras(&mut rng),
            bubbles: Particles::new(120, 5),
            plankton: Particles::new(260, 9),
            surface: pipeline(ReefShader, CullingMode::BackFace),
            foliage: pipeline(ReefShader, CullingMode::None),
            water: SimdPipeline::new(ScreenVertexShader, WaterShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for Reef {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Harp],
            treatment: Treatment::Underwater,
            ambience: vec![Ambience::bed(Source::Synth(Accent::Bubbles), 0.3, 0.2)],
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let uniforms = |caustics| ReefUniforms {
            toon: toon.clone(),
            caustics,
        };
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let current = |amplitude, speed| {
            Motion::Sway(Sway {
                amplitude,
                speed,
                wind: CURRENT,
            })
        };

        self.seabed
            .draw(pass, &self.surface, motion(Motion::Still), uniforms(1.0));
        self.rocks
            .draw(pass, &self.surface, motion(Motion::Still), uniforms(0.8));
        self.coral.draw(
            pass,
            &self.surface,
            motion(current(0.1, 1.2)),
            uniforms(0.8),
        );
        self.kelp.draw(
            pass,
            &self.foliage,
            motion(current(1.1, 0.8)),
            uniforms(0.6),
        );
        self.grass.draw(
            pass,
            &self.foliage,
            motion(current(0.3, 1.5)),
            uniforms(0.8),
        );
        let swim = |speed, radius, height| {
            Motion::Orbit(Orbit {
                centre: Vec3::new(0.0, 0.0, 0.0),
                radius,
                height,
                speed,
                wiggle: 0.14,
            })
        };
        self.shoal.draw(
            pass,
            &self.surface,
            motion(swim((1.8, 2.6), (5.5, 9.5), (-1.0, 3.2))),
            uniforms(0.3),
        );
        self.big_fish.draw(
            pass,
            &self.surface,
            motion(swim((1.0, 1.4), (11.0, 14.0), (0.0, 2.5))),
            uniforms(0.3),
        );

        scene.chip.draw(pass, view, &look());
        self.extras
            .draw(pass, &self.surface, motion(Motion::OnChip), uniforms(0.5));

        let backdrop = view.clone();
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.water, (), |_| backdrop.clone());

        self.bubbles.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 3.0, 1.0),
                half_extent: Vec3::new(15.0, 7.0, 11.0),
                velocity: Vec3::new(0.05, 1.1, 0.0),
                wander: 0.35,
                size: (0.04, 0.12),
                colour: Vec3::new(0.7, 1.1, 1.2),
                twinkle: 0.0,
            },
        );
        self.plankton.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.0, 2.0),
                half_extent: Vec3::new(16.0, 5.0, 13.0),
                velocity: Vec3::new(0.25, -0.03, 0.08),
                wander: 0.5,
                size: (0.015, 0.035),
                colour: Vec3::new(0.7, 1.0, 0.9),
                twinkle: 1.3,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.6,
            vignette: 0.5,
            grain: 0.015,
            aberration: 0.002,
            desaturate: -0.05,
        }
    }
}
