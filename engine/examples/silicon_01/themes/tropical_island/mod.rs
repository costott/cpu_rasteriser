//! A tropical lagoon at midday: the chip a sun-bleached cargo crate at the end of a wooden pier
//! that runs out from a palm island, turquoise water over white sand on both sides, a sailboat
//! bobbing at its mooring, gulls wheeling overhead and a line of surf on the far reef.

mod props;
mod water;

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use self::props::{
    barrel, beach_grass, cargo_frame, gull, gull_flight, hibiscus, palm, pier_deck, pile, rock,
    sailboat, shrub,
};
use self::water::{Bathymetry, FOAM_SPOTS, WaterShader, WaterUniforms};
use crate::chip::{HALF, Look};
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
use crate::themes::kit::dress::chip_shadow;
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

/// The pier deck's height: the chip's lowest corner rests here.
const FLOOR: f32 = -1.95;
const WATER: f32 = -2.9;
const SUN: Vec3 = Vec3 {
    x: -0.38,
    y: 0.8,
    z: 0.3,
};
const WIND: Vec2 = Vec2 { x: 0.8, y: 0.6 };
/// The island's centre on the ground plane (x, z), behind the chip.
const ISLAND: Vec2 = Vec2 { x: 0.0, y: -31.0 };
const PIER_HALF_WIDTH: f32 = 2.6;
const PIER_NEAR: f32 = 9.5;
const PIER_FAR: f32 = -18.5;
/// The reef's radius from the island's centre: where the surf breaks.
const REEF: f32 = 52.0;
const BOAT: Vec3 = Vec3 {
    x: -6.8,
    y: WATER,
    z: -5.0,
};
const BOAT_YAW: f32 = 0.5;

fn smooth(low: f32, high: f32, x: f32) -> f32 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How far inside the island's shore a point is: 0 on it, up to about 1 at its heart.
fn island_shore(x: f32, z: f32) -> f32 {
    let (dx, dz) = ((x - ISLAND.x) / 15.0, (z - ISLAND.y) / 11.0);
    let warp = (fbm_scalar(x * 0.07 + 3.0, z * 0.07, 3) - 0.5) * 0.5;
    1.0 - ((dx * dx + dz * dz).sqrt() + warp)
}

/// The island's height: a sandy beach rising to a low jungle ridge.
fn land(x: f32, z: f32) -> f32 {
    let shore = island_shore(x, z);
    if shore < 0.0 {
        return WATER - 0.35 + shore * 5.0;
    }
    let rise = ((shore - 0.22) / 0.5).clamp(0.0, 1.0);
    let hills = rise.powf(1.3) * (3.4 + 2.0 * fbm_scalar(x * 0.12 + 5.0, z * 0.12, 3));
    WATER - 0.35 + 0.73 * smooth(0.0, 0.14, shore) + hills
}

/// The lagoon's floor below the water: shallow near the island, deepening outwards, rising to
/// the reef and falling away to the open ocean beyond it.
fn lagoon_floor(x: f32, z: f32) -> f32 {
    let r = (x - ISLAND.x).hypot(z - ISLAND.y);
    let mut depth = 0.9 + 1.3 * smooth(16.0, 40.0, r);
    let crest = (-((r - REEF) / 3.5).powi(2)).exp();
    depth += (0.12 - depth) * crest;
    depth += (10.0 - depth) * smooth(54.0, 60.0, r);
    WATER - depth * (0.55 + 0.9 * fbm_scalar(x * 0.06 + 2.0, z * 0.06, 3))
}

/// The depth of the water at a point, negative where the island stands out of it.
fn depth(x: f32, z: f32) -> f32 {
    WATER - land(x, z).max(lagoon_floor(x, z))
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(2.3, 2.1, 1.75),
        sky: Vec3::new(0.36, 0.5, 0.78),
        ground: Vec3::new(0.5, 0.45, 0.35),
        bands: 0.0,
        rim: Vec3::new(0.04, 0.08, 0.1),
        fog: Fog {
            colour: Vec3::new(0.6, 0.82, 0.97),
            density: 0.0015,
            falloff: 0.0,
            base: WATER,
        },
    }
}

fn sky() -> Sky {
    Sky {
        zenith: Vec3::new(0.12, 0.38, 0.9),
        horizon: Vec3::new(0.6, 0.82, 0.97),
        sun: SUN.normalise(),
        sun_colour: Vec3::new(2.0, 1.8, 1.5),
        sun_size: 0.04,
        glow: 0.3,
        clouds: Some(Clouds {
            height: 420.0,
            coverage: 0.42,
            scale: 0.0035,
            wind: Vec2::new(6.0, 2.0),
            lit: Vec3::new(1.7, 1.7, 1.65),
            shade: Vec3::new(0.72, 0.8, 0.95),
        }),
        stars: 0.0,
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.5, 0.42, 0.32),
        die: Vec3::new(0.0, 0.34, 0.2),
        gold: Vec3::new(0.9, 0.64, 0.22),
        streams: Vec3::new(0.4, 1.0, 0.8),
        key: light.sun,
        key_colour: light.sun_colour * 0.6,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.25, 0.35, 0.45),
    }
}

fn island_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let shore = island_shore(centroid.x, centroid.z);
    let sand = lerp3(
        Vec3::new(0.96, 0.88, 0.68),
        Vec3::new(0.88, 0.77, 0.55),
        fbm_scalar(centroid.x * 0.3, centroid.z * 0.3, 2),
    );
    let patch = fbm_scalar(centroid.x * 0.2 + 5.0, centroid.z * 0.2, 3);
    let jungle = lerp3(Vec3::new(0.1, 0.3, 0.07), Vec3::new(0.22, 0.42, 0.1), patch);
    let growth = ((shore - 0.26) / 0.1 + (patch - 0.5) * 1.5).clamp(0.0, 1.0);
    let ground = lerp3(sand, jungle, growth);
    lerp3(
        ground,
        Vec3::new(0.4, 0.36, 0.3),
        ((0.9 - normal.y) * 5.0).clamp(0.0, 0.6),
    )
}

struct GroundUniforms {
    toon: ToonUniforms,
    shadow: Arc<GroundShadow>,
    bathymetry: Arc<Bathymetry>,
}

/// The island's sand and jungle floor: facet colour lit by the sun, shaded by palms and the
/// chip, darker and glossier where the sea has wetted it.
struct GroundShader;

impl FragmentShaderSimd<KitVaryings> for GroundShader {
    type Uniforms = GroundUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GroundUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let normal = facing(v.normal().normalise(), splat3(view.eye) - world);
        let grain = fbm(world.x * s(3.0), world.z * s(3.0), 2);
        let depth = u.bathymetry.depth(world.x, world.z);
        let wet = smoothstep(0.3, -0.05, depth);
        let albedo = v.colour() * (s(0.85) + grain * s(0.3)) * (s(1.0) - wet * s(0.35));
        let shadow = u.shadow.lit(world.x, world.z)
            * (s(1.0) - chip_shadow(view, u.toon.lighting.sun, world));
        let colour = u
            .toon
            .lighting
            .shade(view, world, normal, albedo, clamp01(shadow));
        scene_colour(view, colour, world)
    }
}

/// Weathered timber: vertex colour with grain along the planks' length on top faces and up the
/// piles on the sides, shaded by the chip.
struct WoodShader;

impl FragmentShaderSimd<KitVaryings> for WoodShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let normal = facing(v.normal().normalise(), splat3(view.eye) - world);
        let seed = v.seed() * s(40.0);
        let along_x = fbm(world.x * s(0.5) + seed, world.z * s(7.0), 2);
        let along_y = fbm(
            world.x * s(6.0) + world.z * s(6.0) + seed,
            world.y * s(0.7),
            2,
        );
        let grain = mix(along_y, along_x, smoothstep(0.4, 0.8, normal.y.abs()));
        let albedo = v.colour() * (s(0.76) + grain * s(0.48));
        let shadow = s(1.0) - chip_shadow(view, u.lighting.sun, world) * s(0.85);
        let colour = u.lighting.shade(view, world, normal, albedo, shadow);
        scene_colour(view, colour, world)
    }
}

fn on_island(placements: Vec<Placement>) -> Vec<Placement> {
    placements
        .into_iter()
        .map(|mut p| {
            p.position.x += ISLAND.x;
            p.position.z += ISLAND.y;
            p
        })
        .collect()
}

fn island_spots(
    rng: &mut Rng,
    spacing: f32,
    attempts: usize,
    scale: (f32, f32),
    wanted: impl Fn(f32) -> bool,
) -> Vec<Placement> {
    let spots = scatter(
        rng,
        20.0,
        spacing,
        attempts,
        |x, z| land(x + ISLAND.x, z + ISLAND.y),
        |x, z| wanted(island_shore(x + ISLAND.x, z + ISLAND.y)).then_some(1.0),
        scale,
    );
    on_island(spots)
}

pub struct TropicalIsland {
    island: Shape,
    sea: Shape,
    palms: Shape,
    cover: Shape,
    pier: Shape,
    piles: Shape,
    props: Shape,
    boat: Shape,
    boat_posed: Shape,
    gulls: Shape,
    extras: Shape,
    shadow: Arc<GroundShadow>,
    bathymetry: Arc<Bathymetry>,
    foam: Arc<[Vec3; FOAM_SPOTS]>,
    motes: Particles,
    ground_pipeline: SimdPipeline<KitVertexShader, GroundShader>,
    water_pipeline: SimdPipeline<KitVertexShader, WaterShader>,
    wood: SimdPipeline<KitVertexShader, WoodShader>,
    solid: SimdPipeline<KitVertexShader, VariedToonShader>,
    foliage: SimdPipeline<KitVertexShader, VariedToonShader>,
    two_sided: SimdPipeline<KitVertexShader, ToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyShader>,
}

impl TropicalIsland {
    pub fn new() -> Self {
        let mut rng = Rng::new(7311);
        let offset = Mat4::translation(ISLAND.x, 0.0, ISLAND.y);
        let island = terrain(
            &ring_radii(1.0, 22.0, 30),
            80,
            |x, z| land(x + ISLAND.x, z + ISLAND.y),
            |c, n| island_colour(c + Vec3::new(ISLAND.x, 0.0, ISLAND.y), n),
        )
        .transformed(&offset);
        let sea = terrain(
            &ring_radii(3.0, 600.0, 26),
            48,
            |_, _| WATER,
            |_, _| Vec3::ONE,
        );

        let palm_variants: Vec<Shape> = (0..4).map(|_| palm(&mut rng)).collect();
        let mut palm_spots = island_spots(&mut rng, 5.0, 900, (0.9, 1.3), |shore| {
            (0.2..0.8).contains(&shore)
        });
        palm_spots.truncate(12);
        let mut palms = Batch::new();
        palms.add(&palm_variants, &palm_spots, |p| {
            (p.seed * 13.0) as usize % 4
        });
        let palms = palms.finish();

        let shrubs: Vec<Shape> = (0..4).map(|_| shrub(&mut rng)).collect();
        let shrub_spots = island_spots(&mut rng, 2.2, 1200, (0.8, 1.7), |shore| {
            (0.3..0.85).contains(&shore)
        });
        let tufts: Vec<Shape> = (0..4).map(|_| beach_grass(&mut rng)).collect();
        let tuft_spots = island_spots(&mut rng, 1.0, 1500, (0.8, 1.4), |shore| {
            (0.12..0.4).contains(&shore)
        });
        let blooms: Vec<Shape> = (0..4).map(|_| hibiscus(&mut rng)).collect();
        let bloom_spots = island_spots(&mut rng, 1.8, 600, (0.8, 1.2), |shore| {
            (0.3..0.7).contains(&shore)
        });
        let rocks: Vec<Shape> = (0..3).map(|_| rock(&mut rng)).collect();
        let rock_spots = island_spots(&mut rng, 3.5, 800, (0.5, 1.5), |shore| {
            (0.0..0.1).contains(&shore)
        });
        let mut cover = Batch::new();
        cover.add(&shrubs, &shrub_spots, |p| (p.seed * 11.0) as usize % 4);
        cover.add(&tufts, &tuft_spots, |p| (p.seed * 11.0) as usize % 4);
        cover.add(&blooms, &bloom_spots, |p| (p.seed * 11.0) as usize % 4);
        cover.add(&rocks, &rock_spots[..rock_spots.len().min(10)], |p| {
            (p.seed * 11.0) as usize % 3
        });
        let cover = cover.finish();

        let pier = pier_deck(&mut rng, FLOOR, PIER_HALF_WIDTH, PIER_NEAR, PIER_FAR);
        let mut piles = Shape::new();
        let mut foam = [Vec3::new(0.0, 0.0, 0.0); FOAM_SPOTS];
        let mut spot = 0;
        for k in 0..9 {
            let z = PIER_FAR + 1.5 + 3.0 * k as f32;
            for side in [-1.0, 1.0] {
                let x = side * (PIER_HALF_WIDTH + 0.3);
                let top = if k % 2 == 0 {
                    FLOOR + 0.6
                } else {
                    FLOOR - 0.02
                };
                let post = pile(&mut rng, WATER - 0.6, top, k % 2 == 0)
                    .transformed(&Mat4::translation(x, 0.0, z));
                piles.append(&post);
                foam[spot] = Vec3::new(x, z, 0.5);
                spot += 1;
            }
        }
        for offset in [-0.9_f32, 0.9] {
            foam[spot] = Vec3::new(
                BOAT.x + offset * BOAT_YAW.cos(),
                BOAT.z - offset * BOAT_YAW.sin(),
                1.0,
            );
            spot += 1;
        }

        let mut props = Shape::new();
        for (x, z, colour) in [
            (-2.35, -3.0, Vec3::new(0.16, 0.34, 0.5)),
            (-1.7, -4.1, Vec3::new(0.62, 0.2, 0.12)),
            (2.3, -6.0, Vec3::new(0.62, 0.2, 0.12)),
        ] {
            props.append(&barrel(colour).transformed(&Mat4::translation(x, FLOOR, z)));
        }

        let sun = SUN.normalise();
        let shadow = Arc::new(GroundShadow::bake(
            Vec2::new(ISLAND.x - 22.0, ISLAND.y - 22.0),
            44.0,
            512,
            sun,
            land,
            &[&palms, &cover],
            2,
        ));
        let bathymetry = Arc::new(Bathymetry::bake(Vec2::new(-64.0, -95.0), 128.0, 768, depth));

        Self {
            island,
            sea,
            palms,
            cover,
            pier,
            piles,
            props,
            boat: sailboat(),
            boat_posed: Shape::new(),
            gulls: Shape::new(),
            extras: cargo_frame(&mut rng, HALF),
            shadow,
            bathymetry,
            foam: Arc::new(foam),
            motes: Particles::new(140, 31),
            ground_pipeline: pipeline(GroundShader, CullingMode::BackFace),
            water_pipeline: pipeline(WaterShader, CullingMode::None),
            wood: pipeline(WoodShader, CullingMode::BackFace),
            solid: pipeline(VariedToonShader, CullingMode::BackFace),
            foliage: pipeline(VariedToonShader, CullingMode::None),
            two_sided: pipeline(ToonShader, CullingMode::None),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for TropicalIsland {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::SteelPan],
            ambience: vec![
                Ambience::bed(Source::Recording("surf", 0.0), 0.22, 0.0),
                Ambience::once(Source::Recording("gull", 0.0), 0.15, 1.7, 0.32, -0.35),
            ],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let t = scene.local_time + 2.0;
        let bob = 0.06 * (t * 1.1).sin() + 0.03 * (t * 1.9).sin();
        let pose = Mat4::translation(BOAT.x, WATER + bob, BOAT.z)
            * Mat4::rotate_y(BOAT_YAW + 0.07 * (t * 0.35).sin())
            * Mat4::rotate_x(0.045 * (t * 0.9 + 1.0).sin())
            * Mat4::rotate_z(0.03 * (t * 0.7).sin());
        self.boat_posed = self.boat.transformed(&pose);

        let mut gulls = Shape::new();
        for index in 0..7 {
            let flight =
                gull_flight(t, index, Vec3::new(1.0, 0.0, -8.0)) * Mat4::scaling(0.55, 0.55, 0.55);
            gulls.append(&gull(t, index as f32 * 0.9).transformed(&flight));
        }
        self.gulls = gulls;
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
        let palm_sway = Motion::Sway(Sway {
            amplitude: 0.3,
            speed: 1.2,
            wind: WIND,
        });
        let grass_sway = Motion::Sway(Sway {
            amplitude: 0.2,
            speed: 2.2,
            wind: WIND,
        });

        self.sea.draw(
            pass,
            &self.water_pipeline,
            motion(Motion::Still),
            WaterUniforms {
                toon: toon.clone(),
                sky: sky(),
                bathymetry: self.bathymetry.clone(),
                foam: self.foam.clone(),
            },
        );
        self.island.draw(
            pass,
            &self.ground_pipeline,
            motion(Motion::Still),
            GroundUniforms {
                toon: toon.clone(),
                shadow: self.shadow.clone(),
                bathymetry: self.bathymetry.clone(),
            },
        );
        self.palms
            .draw(pass, &self.foliage, motion(palm_sway), varied(0.12));
        self.cover
            .draw(pass, &self.foliage, motion(grass_sway), varied(0.15));
        self.pier
            .draw(pass, &self.wood, motion(Motion::Still), toon.clone());
        self.piles
            .draw(pass, &self.wood, motion(Motion::Still), toon.clone());
        self.props
            .draw(pass, &self.solid, motion(Motion::Still), varied(0.05));
        self.boat_posed
            .draw(pass, &self.two_sided, motion(Motion::Still), toon.clone());
        self.gulls
            .draw(pass, &self.two_sided, motion(Motion::Still), toon.clone());

        scene.chip.draw(pass, view, &look());
        self.extras
            .draw(pass, &self.solid, motion(Motion::OnChip), varied(0.08));

        let sky = SkyUniforms {
            view: view.clone(),
            sky: sky(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| sky.clone());

        self.motes.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.0, 1.0),
                half_extent: Vec3::new(14.0, 4.0, 12.0),
                velocity: Vec3::new(0.3, 0.1, -0.2),
                wander: 0.5,
                size: (0.02, 0.05),
                colour: Vec3::new(1.5, 1.4, 1.1),
                twinkle: 1.8,
            },
        );
    }

    fn caption_ink(&self) -> Vec3 {
        Vec3::new(0.03, 0.22, 0.3)
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.45,
            vignette: 0.3,
            grain: 0.012,
            aberration: 0.0015,
            desaturate: -0.1,
        }
    }
}
