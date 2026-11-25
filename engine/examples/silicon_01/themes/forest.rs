//! A low-poly forest clearing at golden hour: the chip a mossy standing stone, conifers and
//! round-leaf trees swaying round it, a fox that trots in to look at it, pollen in the light.

mod fox;

use std::f32::consts::TAU;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::scene_colour;
use crate::shaders::screen::ScreenVertexShader;
use crate::simd::{clamp01, mix, s, smoothstep, splat3};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonShader,
    ToonUniforms, VariedToonShader, VariedToonUniforms, pipeline,
};
use crate::themes::kit::dress::{chip_shadow, rim_point};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shadow::GroundShadow;
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::sky::{Clouds, Sky, SkyShader, SkyUniforms};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The height of the ground where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.62,
    y: 0.36,
    z: 0.7,
};
const WIND: Vec2 = Vec2 { x: 0.8, y: 0.6 };

/// The ground's height: a flat clearing round the chip, gentle meadow swells, and wooded ridges
/// rising beyond the trees to make the horizon.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let clearing = (r / 7.0).clamp(0.0, 1.0);
    let swell = (fbm_scalar(x * 0.06 + 4.0, z * 0.06, 3) - 0.5) * 1.6
        + (fbm_scalar(x * 0.3, z * 0.3, 2) - 0.5) * 0.35;
    let ridge_edge = ((r - 62.0) / 60.0).clamp(0.0, 1.0);
    let ridge =
        ridge_edge * ridge_edge * (4.0 + 22.0 * fbm_scalar(x * 0.02 + 9.0, z * 0.02 + 2.0, 3));
    FLOOR + clearing * clearing * swell + ridge
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(2.9, 2.05, 1.05),
        sky: Vec3::new(0.2, 0.28, 0.46),
        ground: Vec3::new(0.2, 0.15, 0.08),
        bands: 0.0,
        rim: Vec3::new(0.22, 0.14, 0.05),
        fog: Fog {
            colour: Vec3::new(0.95, 0.62, 0.38),
            density: 0.011,
            falloff: 0.05,
            base: FLOOR,
        },
    }
}

fn sky() -> Sky {
    Sky {
        zenith: Vec3::new(0.10, 0.28, 0.72),
        horizon: Vec3::new(1.0, 0.66, 0.36),
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.6, 1.15, 0.7),
        sun_size: 0.045,
        glow: 1.0,
        clouds: Some(Clouds {
            height: 380.0,
            coverage: 0.5,
            scale: 0.0042,
            wind: Vec2::new(7.0, 2.5),
            lit: Vec3::new(1.5, 1.05, 0.7),
            shade: Vec3::new(0.62, 0.52, 0.62),
        }),
        stars: 0.0,
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.2, 0.22, 0.18),
        die: Vec3::new(0.05, 0.09, 0.06),
        gold: Vec3::new(0.7, 0.5, 0.2),
        streams: Vec3::new(0.5, 0.9, 0.5),
        key: light.sun,
        key_colour: light.sun_colour * 0.55,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.25, 0.2, 0.12),
    }
}

struct GroundUniforms {
    toon: ToonUniforms,
    shadow: Arc<GroundShadow>,
}

/// Meadow and forest floor: facet colour lit by the sun, shaded by the baked tree shadows, the
/// chip and drifting clouds.
struct GroundShader;

impl FragmentShaderSimd<KitVaryings> for GroundShader {
    type Uniforms = GroundUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GroundUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let normal = facing(v.normal().normalise(), splat3(view.eye) - world);
        let tone = fbm(world.x * s(0.45), world.z * s(0.45), 2);
        let albedo = v.colour() * (s(0.8) + tone * s(0.4));
        let t = view.frame.time;
        let drift = fbm(
            world.x * s(0.035) + s(t * 0.5),
            world.z * s(0.035) + s(t * 0.2),
            2,
        );
        let cloud = mix(s(0.55), s(1.0), smoothstep(0.35, 0.65, drift));
        let shadow = u.shadow.lit(world.x, world.z)
            * (s(1.0) - chip_shadow(view, u.toon.lighting.sun, world))
            * cloud;
        let colour = u
            .toon
            .lighting
            .shade(view, world, normal, albedo, clamp01(shadow));
        scene_colour(view, colour, world)
    }
}

fn conifer(detail: bool, rng: &mut Rng) -> Shape {
    let segments = if detail { 8 } else { 6 };
    let trunk = Vec3::new(0.2, 0.12, 0.07);
    let mut tree = Shape::cylinder(0.32, 0.16, 2.2, 5, trunk);
    let tiers = if detail { 4 } else { 3 };
    for k in 0..tiers {
        let t = k as f32 / (tiers - 1) as f32;
        let radius = 2.7 * (1.0 - 0.62 * t);
        let base = 1.4 + k as f32 * 2.0;
        let green = lerp3(Vec3::new(0.05, 0.17, 0.07), Vec3::new(0.12, 0.30, 0.09), t);
        let cone = Shape::cone(radius, 3.6 - 0.5 * t, segments, green).transformed(
            &(Mat4::translation(0.0, base, 0.0) * Mat4::rotate_y(rng.range(0.0, TAU))),
        );
        tree.append(&cone);
    }
    tree.facet_variation(rng, 0.1).sway_by_height(0.5, 9.0, 1.7)
}

fn round_tree(rng: &mut Rng) -> Shape {
    let mut tree = Shape::cylinder(0.42, 0.22, 3.4, 6, Vec3::new(0.26, 0.16, 0.08));
    for (centre, radius) in [
        (Vec3::new(0.0, 5.0, 0.0), 2.3),
        (Vec3::new(1.3, 4.2, 0.5), 1.7),
        (Vec3::new(-1.2, 4.5, -0.6), 1.8),
        (Vec3::new(0.2, 6.3, 0.2), 1.5),
    ] {
        let green = Vec3::new(0.30, 0.42, 0.07);
        let crown = Shape::icosphere(radius, 1, 0.2, rng.range(0.0, 50.0), green)
            .colour_by(|p, n| {
                lerp3(
                    Vec3::new(0.2, 0.32, 0.06),
                    Vec3::new(0.52, 0.55, 0.1),
                    (n.y * 0.5 + 0.5).powf(1.5) * (p.y / 7.0).clamp(0.4, 1.0),
                )
            })
            .transformed(&Mat4::translation_vec(centre));
        tree.append(&crown);
    }
    tree.facet_variation(rng, 0.08)
        .sway_by_height(1.5, 7.5, 1.4)
}

fn grass_tuft(rng: &mut Rng) -> Shape {
    let mut tuft = Shape::new();
    for _ in 0..5 {
        let angle = rng.range(0.0, TAU);
        let height = rng.range(0.3, 0.62);
        let lean = rng.range(0.05, 0.3);
        let base = Vec3::new(angle.cos() * 0.08, 0.0, angle.sin() * 0.08);
        let side = Vec3::new(-angle.sin(), 0.0, angle.cos()) * 0.045;
        let tip = base + Vec3::new(angle.cos() * lean, height, angle.sin() * lean);
        tuft.triangle(base - side, base + side, tip, Vec3::ONE);
        tuft.triangle(base + side, base - side, tip, Vec3::ONE);
    }
    tuft.gradient_by_height(
        0.0,
        0.6,
        Vec3::new(0.10, 0.17, 0.03),
        Vec3::new(0.62, 0.6, 0.16),
    )
    .sway_by_height(0.0, 0.6, 1.0)
}

fn fern(rng: &mut Rng) -> Shape {
    let mut fern = Shape::new();
    let fronds = 8;
    for k in 0..fronds {
        let angle = TAU * k as f32 / fronds as f32 + rng.range(-0.2, 0.2);
        let length = rng.range(0.9, 1.4);
        let out = Vec3::new(angle.cos(), 0.0, angle.sin());
        let spine: Vec<Vec3> = (0..6)
            .map(|i| {
                let t = i as f32 / 5.0;
                out * (t * length)
                    + Vec3::new(0.0, 0.25 + 0.9 * t * (1.0 - 0.75 * t) * length * 0.7, 0.0)
            })
            .collect();
        let widths = [0.08, 0.34, 0.26, 0.3, 0.16, 0.0];
        fern.append(&Shape::ribbon(
            &spine,
            &widths,
            Vec3::new(-angle.sin(), 0.0, angle.cos()),
            Vec3::ONE,
        ));
    }
    fern.gradient_by_height(
        0.0,
        1.0,
        Vec3::new(0.06, 0.18, 0.05),
        Vec3::new(0.28, 0.5, 0.12),
    )
    .sway_by_height(0.2, 1.1, 1.0)
}

fn flower(rng: &mut Rng) -> Shape {
    let colours = [
        Vec3::new(1.0, 0.82, 0.2),
        Vec3::new(0.95, 0.93, 0.85),
        Vec3::new(0.9, 0.42, 0.55),
    ];
    let colour = colours[rng.index(colours.len())];
    let height = rng.range(0.5, 0.8);
    let mut flower = Shape::new();
    let side = Vec3::new(0.02, 0.0, 0.0);
    flower.triangle(
        -side,
        side,
        Vec3::new(0.0, height, 0.0),
        Vec3::new(0.15, 0.3, 0.07),
    );
    flower.triangle(
        side,
        -side,
        Vec3::new(0.0, height, 0.0),
        Vec3::new(0.15, 0.3, 0.07),
    );
    flower.append(
        &Shape::icosphere(0.055, 0, 0.0, 0.0, colour)
            .transformed(&Mat4::translation(0.0, height, 0.0)),
    );
    flower.sway_by_height(0.0, height, 1.0)
}

fn boulder(rng: &mut Rng) -> Shape {
    let stone = Vec3::new(0.40, 0.38, 0.35);
    Shape::icosphere(1.0, 1, 0.28, rng.range(0.0, 99.0), stone)
        .transformed(&Mat4::scaling(
            rng.range(0.9, 1.3),
            rng.range(0.55, 0.8),
            rng.range(0.8, 1.2),
        ))
        .colour_by(|_, n| {
            if n.y > 0.55 {
                Vec3::new(0.2, 0.36, 0.08)
            } else if n.y > 0.3 {
                lerp3(stone, Vec3::new(0.2, 0.36, 0.08), 0.5)
            } else {
                stone
            }
        })
        .facet_variation(rng, 0.1)
}

fn mushroom() -> Shape {
    let mut shape = Shape::cylinder(0.12, 0.09, 0.45, 6, Vec3::new(0.9, 0.85, 0.7));
    let cap = Shape::lathe(
        &[
            (0.0, 0.34),
            (0.5, 0.34),
            (0.48, 0.5),
            (0.32, 0.68),
            (0.0, 0.78),
        ],
        8,
        Vec3::new(0.78, 0.14, 0.07),
    );
    shape.append(&cap);
    shape.with_sway(0.0)
}

fn on_ground(placements: &mut [Placement], lift: f32) {
    for p in placements {
        p.position.y += lift;
    }
}

fn terrain_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let r = (centroid.x * centroid.x + centroid.z * centroid.z).sqrt();
    let patch = fbm_scalar(centroid.x * 0.16 + 11.0, centroid.z * 0.16, 3);
    let meadow = lerp3(
        Vec3::new(0.30, 0.38, 0.07),
        Vec3::new(0.52, 0.52, 0.12),
        patch,
    );
    let floor = Vec3::new(0.10, 0.15, 0.05);
    let ridge = Vec3::new(0.06, 0.15, 0.08);
    let soil = Vec3::new(0.22, 0.15, 0.08);
    let mut c = lerp3(meadow, floor, ((r - 12.0) / 14.0).clamp(0.0, 1.0));
    c = lerp3(c, ridge, ((r - 55.0) / 25.0).clamp(0.0, 1.0));
    c = lerp3(
        c,
        soil,
        (1.0 - (r / 3.4)).clamp(0.0, 1.0) * 0.8 + ((patch - 0.72) * 4.0).clamp(0.0, 0.5),
    );
    c = lerp3(
        c,
        Vec3::new(0.3, 0.27, 0.22),
        ((0.93 - normal.y) * 6.0).clamp(0.0, 0.7),
    );
    c * (0.94 + 0.12 * crate::simd::hash2_scalar(centroid.x, centroid.z))
}

/// Moss clumps and twig pins on the chip, in its own space.
fn chip_extras(rng: &mut Rng) -> Shape {
    let mut extras = Shape::new();
    let moss = Vec3::new(0.2, 0.4, 0.08);
    for _ in 0..34 {
        let (spot, _) = rim_point(rng);
        let radius = rng.range(0.12, 0.34);
        let clump = Shape::icosphere(1.0, 1, 0.25, rng.range(0.0, 40.0), moss)
            .transformed(
                &(Mat4::translation_vec(spot) * Mat4::scaling(radius, radius * 0.5, radius)),
            )
            .facet_variation(rng, 0.15);
        extras.append(&clump);
    }
    let twig = Vec3::new(0.27, 0.17, 0.09);
    for _ in 0..16 {
        let (spot, outward) = rim_point(rng);
        let from = spot + outward * 0.15;
        let to = from + outward * rng.range(0.4, 0.8) + Vec3::new(0.0, rng.range(-0.1, 0.12), 0.0);
        extras.append(&Shape::limb(from, to, 0.05, 0.025, 5, twig));
    }
    extras
}

pub struct Forest {
    terrain: Shape,
    trees: Shape,
    grass: Shape,
    rocks: Shape,
    fox: Shape,
    extras: Shape,
    shadow: Arc<GroundShadow>,
    pollen: Particles,
    ground_pipeline: SimdPipeline<KitVertexShader, GroundShader>,
    solid: SimdPipeline<KitVertexShader, VariedToonShader>,
    foliage: SimdPipeline<KitVertexShader, VariedToonShader>,
    plain: SimdPipeline<KitVertexShader, ToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyShader>,
}

impl Forest {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = Rng::new(2024);
        let terrain = terrain(&ring_radii(1.5, 230.0, 46), 56, ground, terrain_colour);

        let conifers: Vec<Shape> = (0..3).map(|_| conifer(true, &mut rng)).collect();
        let far_conifers: Vec<Shape> = (0..3).map(|_| conifer(false, &mut rng)).collect();
        let rounds: Vec<Shape> = (0..3).map(|_| round_tree(&mut rng)).collect();
        let clear_of_chip = |x: f32, z: f32| {
            let r = (x * x + z * z).sqrt();
            if r < 10.0 || (x.abs() < 3.2 && z > 0.0 && z < 14.0) {
                return None;
            }
            Some(if r < 22.0 { 1.35 } else { 1.0 })
        };
        let all_trees = scatter(
            &mut rng,
            82.0,
            6.2,
            9000,
            ground,
            clear_of_chip,
            (0.75, 1.5),
        );
        let (near, far): (Vec<Placement>, Vec<Placement>) = all_trees
            .into_iter()
            .partition(|p| (p.position.x.powi(2) + p.position.z.powi(2)).sqrt() < 30.0);
        let (round_near, conifer_near): (Vec<Placement>, Vec<Placement>) =
            near.iter().partition(|p| p.seed >= 0.68);
        let mut trees = Batch::new();
        trees.add(&conifers, &conifer_near, |p| (p.seed * 13.0) as usize % 3);
        trees.add(&rounds, &round_near, |p| (p.seed * 13.0) as usize % 3);
        let far_ring: Vec<Placement> = far.into_iter().filter(|p| p.seed < 0.9).collect();
        trees.add(&far_conifers, &far_ring, |p| (p.seed * 13.0) as usize % 3);
        let trees = trees.finish();

        let tufts: Vec<Shape> = (0..4).map(|_| grass_tuft(&mut rng)).collect();
        let around_chip = |x: f32, z: f32| {
            let r = (x * x + z * z).sqrt();
            if !(3.0..=26.0).contains(&r) || z > 3.5 {
                None
            } else {
                Some(1.0 + r * 0.03)
            }
        };
        let blades = scatter(&mut rng, 26.0, 0.8, 9000, ground, around_chip, (0.8, 1.4));
        let ferns: Vec<Shape> = (0..3).map(|_| fern(&mut rng)).collect();
        let fern_spots = scatter(
            &mut rng,
            34.0,
            2.4,
            1500,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 4.5 || z > 3.0 {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.7, 1.3),
        );
        let flowers: Vec<Shape> = (0..5).map(|_| flower(&mut rng)).collect();
        let blooms = scatter(
            &mut rng,
            16.0,
            1.8,
            700,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 3.4 || z > 3.5 {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.7, 1.1),
        );
        let mut grass = Batch::new();
        grass.add(&tufts, &blades, |p| (p.seed * 17.0) as usize % 4);
        grass.add(&ferns, &fern_spots, |p| (p.seed * 17.0) as usize % 3);
        grass.add(&flowers, &blooms, |p| (p.seed * 17.0) as usize % 5);
        let grass = grass.finish();

        let boulders: Vec<Shape> = (0..4).map(|_| boulder(&mut rng)).collect();
        let mut rock_spots = scatter(
            &mut rng,
            24.0,
            4.5,
            400,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 4.0 || x.abs() < 3.0 && z > 0.0 {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.35, 1.6),
        );
        rock_spots.truncate(14);
        on_ground(&mut rock_spots, -0.1);
        let shrooms = scatter(
            &mut rng,
            8.0,
            1.4,
            300,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 4.2 || z > 0.0 && x.abs() < 3.0 {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.5, 0.9),
        );
        let mut rocks = Batch::new();
        rocks.add(&boulders, &rock_spots, |p| (p.seed * 11.0) as usize % 4);
        rocks.add(&[mushroom()], &shrooms[..shrooms.len().min(9)], |_| 0);
        let mut rocks = rocks.finish();
        rocks.append(
            &Shape::icosphere(1.0, 1, 0.3, 5.0, Vec3::new(0.38, 0.36, 0.33)).transformed(
                &(Mat4::translation(-3.3, FLOOR - 0.15, 1.0) * Mat4::scaling(1.0, 0.6, 0.9)),
            ),
        );

        let sun = SUN.normalise();
        let shadow = Arc::new(GroundShadow::bake(
            Vec2::new(-82.0, -82.0),
            164.0,
            1024,
            sun,
            ground,
            &[&trees, &rocks],
            2,
        ));

        Ok(Self {
            terrain,
            trees,
            grass,
            rocks,
            fox: Shape::new(),
            extras: chip_extras(&mut rng),
            shadow,
            pollen: Particles::new(220, 77),
            ground_pipeline: pipeline(GroundShader, CullingMode::BackFace),
            solid: pipeline(VariedToonShader, CullingMode::BackFace),
            foliage: pipeline(VariedToonShader, CullingMode::None),
            plain: pipeline(ToonShader, CullingMode::BackFace),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        })
    }
}

impl Stage for Forest {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Flute],
            ambience: vec![Ambience::bed(
                Source::Recording("birdsong", 0.0),
                0.17,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.fox = fox::build(&fox::pose(fox::ON_CUT + scene.local_time, ground));
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
        let varied = |variation| VariedToonUniforms {
            toon: toon.clone(),
            variation,
        };
        let tree_sway = Motion::Sway(Sway {
            amplitude: 0.32,
            speed: 1.1,
            wind: WIND,
        });
        let grass_sway = Motion::Sway(Sway {
            amplitude: 0.22,
            speed: 2.3,
            wind: WIND,
        });

        self.terrain.draw(
            pass,
            &self.ground_pipeline,
            motion(Motion::Still),
            GroundUniforms {
                toon: toon.clone(),
                shadow: self.shadow.clone(),
            },
        );
        self.trees
            .draw(pass, &self.solid, motion(tree_sway), varied(0.14));
        self.grass
            .draw(pass, &self.foliage, motion(grass_sway), varied(0.18));
        self.rocks
            .draw(pass, &self.solid, motion(Motion::Still), varied(0.06));
        self.fox
            .draw(pass, &self.plain, motion(Motion::Still), toon.clone());

        scene.chip.draw(pass, view, &look());
        self.extras
            .draw(pass, &self.solid, motion(Motion::OnChip), varied(0.2));

        let sky = SkyUniforms {
            view: view.clone(),
            sky: sky(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| sky.clone());

        self.pollen.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 2.0),
                half_extent: Vec3::new(15.0, 4.5, 12.0),
                velocity: Vec3::new(0.35, 0.12, -0.15),
                wander: 0.5,
                size: (0.025, 0.06),
                colour: Vec3::new(1.8, 1.35, 0.7),
                twinkle: 1.6,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.55,
            vignette: 0.4,
            grain: 0.014,
            aberration: 0.0015,
            desaturate: -0.08,
        }
    }
}
