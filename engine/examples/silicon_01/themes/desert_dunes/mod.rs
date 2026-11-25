//! Desert dunes at a low, huge sun: the chip a half-buried carved tablet on rippled ochre sand,
//! razor-crested dunes running back to a hazy horizon, sandstone monoliths, a caravan on a far
//! ridge and sand streaming off the crests.

mod camel;

use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use self::camel::{camel, walker};
use crate::chip::{HALF, Look};
use crate::shaders::composite::Grade;
use crate::shaders::scene_colour;
use crate::shaders::screen::ScreenVertexShader;
use crate::simd::{F, clamp01, hash2, length, mix3, mul3, rgb, s, smoothstep, splat3};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonShader, ToonUniforms,
    VariedToonShader, VariedToonUniforms, pipeline,
};
use crate::themes::kit::dress::{chip_shadow, rim_point};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar, value};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::sky::{Clouds, Sky, SkyShader, SkyUniforms};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The sand's height where the chip lies.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.5,
    y: 0.2,
    z: -0.85,
};

/// A standing stone: where, how wide and deep at its foot, and how tall.
struct Monolith {
    x: f32,
    z: f32,
    width: f32,
    depth: f32,
    height: f32,
    lean: f32,
}

const MONOLITHS: [Monolith; 6] = [
    Monolith {
        x: -16.0,
        z: -40.0,
        width: 2.4,
        depth: 1.4,
        height: 10.0,
        lean: 0.04,
    },
    Monolith {
        x: 13.0,
        z: -56.0,
        width: 2.8,
        depth: 1.5,
        height: 14.0,
        lean: -0.03,
    },
    Monolith {
        x: -34.0,
        z: -74.0,
        width: 3.2,
        depth: 1.7,
        height: 17.0,
        lean: 0.05,
    },
    Monolith {
        x: 30.0,
        z: -96.0,
        width: 3.6,
        depth: 1.9,
        height: 20.0,
        lean: 0.0,
    },
    Monolith {
        x: 4.0,
        z: -140.0,
        width: 4.2,
        depth: 2.2,
        height: 26.0,
        lean: 0.02,
    },
    Monolith {
        x: -56.0,
        z: -128.0,
        width: 3.2,
        depth: 1.6,
        height: 18.0,
        lean: -0.04,
    },
];

/// The row of dunes the caravan walks along, far behind the chip.
const CARAVAN_Z: f32 = -112.0;

fn smooth(low: f32, high: f32, x: f32) -> f32 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The sand's height: a flat patch round the chip, a drift piled against it, and dunes with a
/// long windward slope and a steep lee face that grow taller with distance.
fn ground(x: f32, z: f32) -> f32 {
    let r = x.hypot(z);
    let open = smooth(5.0, 24.0, r);
    let warp = (fbm_scalar(x * 0.02 + 3.0, z * 0.02, 3) - 0.5) * 1.5;
    let coord = x * 0.03 + z * 0.007 + warp;
    let p = coord - coord.floor();
    let profile = if p < 0.78 {
        (p / 0.78).powf(1.3)
    } else {
        1.0 - (p - 0.78) / 0.22
    };
    let height =
        (1.0 + 0.06 * r).min(14.0) * (0.55 + 0.9 * fbm_scalar(x * 0.015 + 7.0, z * 0.015, 2));
    let cross_cut = (fbm_scalar(x * 0.05, z * 0.05 + 4.0, 3) - 0.5) * 1.1;
    let drift =
        0.45 * (-(x / 3.6).powi(2)).exp() * smooth(-0.5, 1.2, z) * (1.0 - smooth(2.6, 5.0, z));
    FLOOR + open * (height * profile + cross_cut) + drift
}

/// How high a standing stone reaches over (`x`, `z`), or the sand there if none does.
fn occluder(x: f32, z: f32) -> f32 {
    let sand = ground(x, z);
    MONOLITHS
        .iter()
        .filter(|m| (x - m.x).abs() < m.width * 0.4 && (z - m.z).abs() < m.depth * 0.4)
        .fold(sand, |top, m| top.max(ground(m.x, m.z) + m.height))
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(3.2, 2.0, 1.0),
        sky: Vec3::new(0.35, 0.28, 0.32),
        ground: Vec3::new(0.32, 0.18, 0.1),
        bands: 0.0,
        rim: Vec3::new(0.2, 0.1, 0.04),
        fog: Fog {
            colour: Vec3::new(0.95, 0.66, 0.42),
            density: 0.0048,
            falloff: 0.03,
            base: -4.0,
        },
    }
}

fn sky() -> Sky {
    Sky {
        zenith: Vec3::new(0.14, 0.3, 0.62),
        horizon: Vec3::new(1.0, 0.68, 0.42),
        sun: SUN.normalise(),
        sun_colour: Vec3::new(3.2, 2.0, 1.0),
        sun_size: 0.09,
        glow: 0.7,
        clouds: Some(Clouds {
            height: 900.0,
            coverage: 0.3,
            scale: 0.0018,
            wind: Vec2::new(8.0, 1.0),
            lit: Vec3::new(1.9, 1.2, 0.8),
            shade: Vec3::new(0.85, 0.55, 0.55),
        }),
        stars: 0.0,
    }
}

fn look() -> Look {
    Look {
        substrate: Vec3::new(0.66, 0.46, 0.3),
        die: Vec3::new(0.2, 0.09, 0.05),
        gold: Vec3::new(0.95, 0.66, 0.26),
        streams: Vec3::new(1.0, 0.62, 0.2),
        key: Vec3::new(-0.6, 0.45, 0.65).normalise(),
        key_colour: Vec3::new(1.5, 0.95, 0.55),
        sky: Vec3::new(0.5, 0.36, 0.34),
        ground: Vec3::new(0.4, 0.22, 0.12),
        softbox: Vec3::new(0.5, 0.3, 0.18),
    }
}

/// What the sand's surface is made of, baked once: its smooth normal, and how much of the sun
/// reaches it past the crests and monoliths.
struct SandMap {
    texture: TextureSampler,
    min: Vec2,
    size: f32,
}

impl SandMap {
    fn bake(min: Vec2, size: f32, resolution: usize, sun: Vec3) -> Self {
        let step = size / resolution as f32;
        let texel = |i: usize, j: usize| -> Colour {
            let x = min.x + (i as f32 + 0.5) * step;
            let z = min.y + (j as f32 + 0.5) * step;
            let here = ground(x, z);
            let slope_x = (ground(x + 0.4, z) - here) / 0.4;
            let slope_z = (ground(x, z + 0.4) - here) / 0.4;
            let normal = Vec3::new(-slope_x, 1.0, -slope_z).normalise();
            let mut open = 1.0_f32;
            let mut travelled = 0.4;
            while travelled < 260.0 {
                let (px, pz) = (x + sun.x * travelled, z + sun.z * travelled);
                let above = here + sun.y * travelled - occluder(px, pz);
                open = open.min((above / travelled * 10.0).clamp(0.0, 1.0));
                travelled *= 1.13;
            }
            Colour::new(
                open.clamp(0.0, 1.0),
                normal.x * 0.5 + 0.5,
                normal.z * 0.5 + 0.5,
                1.0,
            )
        };
        let mut pixels = vec![Colour::new(1.0, 0.5, 0.5, 1.0); resolution * resolution];
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let rows_per_chunk = resolution.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk, rows) in pixels.chunks_mut(rows_per_chunk * resolution).enumerate() {
                let texel = &texel;
                scope.spawn(move || {
                    for (k, row) in rows.chunks_mut(resolution).enumerate() {
                        let j = resolution - 1 - (chunk * rows_per_chunk + k);
                        for (i, pixel) in row.iter_mut().enumerate() {
                            *pixel = texel(i, j);
                        }
                    }
                });
            }
        });
        let pixels: Arc<[Colour]> = pixels.into();
        let texture = TextureSampler::new(
            pixels,
            resolution as u32,
            resolution as u32,
            WrapMode::Clamp,
            FilterMode::Linear,
        );
        Self { texture, min, size }
    }

    /// The fraction of the sun reaching (`x`, `z`) and the sand's unit normal there.
    #[inline(always)]
    fn at(&self, x: F, z: F) -> (F, Vec3Simd) {
        let u = (x - s(self.min.x)) * s(1.0 / self.size);
        let v = (z - s(self.min.y)) * s(1.0 / self.size);
        let texel = self.texture.sample_linear_clamp_simd([u, v]);
        let (nx, nz) = (texel.g * s(2.0) - s(1.0), texel.b * s(2.0) - s(1.0));
        let ny = (s(1.0) - nx * nx - nz * nz).fast_max(s(0.05)).sqrt();
        (texel.r, Vec3Simd::new(nx, ny, nz).normalise())
    }
}

/// Sand and stone in a low sun: shadow side deep red-violet, the lit side blazing orange.
#[inline(always)]
fn sun_ramp(normal: Vec3Simd, albedo: Vec3Simd, open: F, sun: Vec3) -> Vec3Simd {
    let wrap = clamp01((normal.dot(splat3(sun)) + s(0.2)) / s(1.2));
    let t = wrap * (s(0.3) + s(0.7) * open);
    let mut colour = mix3(
        rgb(0.3, 0.15, 0.17),
        rgb(0.8, 0.38, 0.2),
        smoothstep(0.0, 0.5, t),
    );
    colour = mix3(colour, rgb(1.4, 0.95, 0.52), smoothstep(0.45, 0.95, t));
    colour = colour + rgb(0.12, 0.1, 0.15) * (normal.y * s(0.5) + s(0.5)) * s(0.6);
    mul3(colour, albedo)
}

struct SurfaceUniforms {
    toon: ToonUniforms,
    sand: Arc<SandMap>,
}

/// The dunes: baked smooth normals and shadows, wind ripples and grains close up, glints, and
/// the chip's shadow.
struct DuneShader;

impl FragmentShaderSimd<KitVaryings> for DuneShader {
    type Uniforms = SurfaceUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &SurfaceUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let distance = length(splat3(view.eye) - world);
        let (open, mut normal) = u.sand.at(world.x, world.z);
        let near = clamp01((s(130.0) - distance) * s(0.01));
        let ripple = world.x * s(16.0) + fbm(world.x * s(0.35), world.z * s(0.35), 2) * s(8.0);
        normal =
            Vec3Simd::new(normal.x + ripple.cos() * s(0.22) * near, normal.y, normal.z).normalise();
        let cell = (world.x * s(70.0)).floor();
        let grain = hash2(cell, (world.z * s(70.0)).floor());
        let close = clamp01((s(30.0) - distance) * s(0.05));
        let albedo = v.colour() * (s(1.0) + (grain - s(0.5)) * s(0.2) * close);
        let open = open * (s(1.0) - chip_shadow(view, u.toon.lighting.sun, world));
        let mut colour = sun_ramp(normal, albedo, open, u.toon.lighting.sun);
        let glint = (grain - s(0.996)).fast_max(s(0.0)) * s(120.0) * close * open;
        colour = colour + rgb(1.6, 1.3, 0.9) * glint;
        let colour = u.toon.lighting.fog.apply(colour, world, distance);
        scene_colour(view, colour, world)
    }
}

/// Sandstone: strata in horizontal bands over the facet's colour, lit like the sand.
struct StoneShader;

impl FragmentShaderSimd<KitVaryings> for StoneShader {
    type Uniforms = SurfaceUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &SurfaceUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let distance = length(splat3(view.eye) - world);
        let normal = facing(v.normal().normalise(), splat3(view.eye) - world);
        let wobble = value(world.x * s(0.6) + world.z * s(0.4), world.y * s(0.3)) * s(2.0);
        let bands = ((world.y * s(1.9) + wobble).sin() * s(0.5) + s(0.5)) * s(0.35) + s(0.75);
        let (open, _) = u.sand.at(world.x, world.z);
        let colour = sun_ramp(normal, v.colour() * bands, s(1.0), u.toon.lighting.sun)
            * (s(0.6) + open * s(0.4));
        let colour = u.toon.lighting.fog.apply(colour, world, distance);
        scene_colour(view, colour, world)
    }
}

fn monolith(m: &Monolith) -> Shape {
    let rock = Vec3::new(0.62, 0.34, 0.22);
    Shape::cylinder(0.5, 0.4, 1.0, 4, rock)
        .transformed(&Mat4::rotate_y(FRAC_PI_4))
        .transformed(
            &(Mat4::translation(m.x, ground(m.x, m.z) - 0.6, m.z)
                * Mat4::rotate_z(m.lean)
                * Mat4::scaling(m.width * 2.0, m.height, m.depth * 2.0)),
        )
}

fn dune_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let patch = fbm_scalar(centroid.x * 0.04, centroid.z * 0.04 + 2.0, 3);
    let sand = lerp3(
        Vec3::new(0.94, 0.74, 0.5),
        Vec3::new(0.84, 0.58, 0.36),
        patch,
    );
    sand * (0.95 + 0.1 * normal.y)
}

/// Carved glyph blocks along the package's rim and drifts of sand against it, in the chip's space.
fn tablet_extras(rng: &mut Rng) -> Shape {
    let carving = Vec3::new(0.46, 0.28, 0.18);
    let mut extras = Shape::new();
    for side in 0..4 {
        for k in 0..14 {
            let along = -HALF + 0.3 + (2.0 * HALF - 0.6) * k as f32 / 13.0;
            let inset = HALF - 0.17;
            let (x, z) = match side {
                0 => (along, -inset),
                1 => (along, inset),
                2 => (-inset, along),
                _ => (inset, along),
            };
            let (long, wide) = (rng.range(0.07, 0.17), rng.range(0.03, 0.06));
            let (half_x, half_z) = if side < 2 { (long, wide) } else { (wide, long) };
            extras.append(&Shape::cuboid(
                Vec3::new(x - half_x, 0.0, z - half_z),
                Vec3::new(x + half_x, rng.range(0.03, 0.07), z + half_z),
                carving,
            ));
        }
    }
    let sand = Vec3::new(1.5, 1.1, 0.7);
    for _ in 0..8 {
        let (spot, _) = rim_point(rng);
        let radius = rng.range(0.2, 0.42);
        extras.append(
            &Shape::icosphere(1.0, 1, 0.25, rng.range(0.0, 40.0), sand)
                .transformed(
                    &(Mat4::translation_vec(spot) * Mat4::scaling(radius, radius * 0.35, radius)),
                )
                .facet_variation(rng, 0.1),
        );
    }
    extras
}

pub struct DesertDunes {
    dunes: Shape,
    stones: Shape,
    caravan: Shape,
    caravan_start: f32,
    extras: Shape,
    sand: Arc<SandMap>,
    grains: Particles,
    dust: Particles,
    dune_pipeline: SimdPipeline<KitVertexShader, DuneShader>,
    stone_pipeline: SimdPipeline<KitVertexShader, StoneShader>,
    solid: SimdPipeline<KitVertexShader, VariedToonShader>,
    two_sided: SimdPipeline<KitVertexShader, ToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyShader>,
}

impl DesertDunes {
    pub fn new() -> Self {
        let mut rng = Rng::new(4417);
        let dunes = terrain(&ring_radii(1.5, 260.0, 56), 96, ground, dune_colour);
        let mut stones = Shape::new();
        for m in &MONOLITHS {
            stones.append(&monolith(m));
        }
        let sand = Arc::new(SandMap::bake(
            Vec2::new(-130.0, -200.0),
            260.0,
            1024,
            SUN.normalise(),
        ));
        Self {
            dunes,
            stones,
            caravan: Shape::new(),
            caravan_start: -30.0,
            extras: tablet_extras(&mut rng),
            sand,
            grains: Particles::new(320, 52),
            dust: Particles::new(70, 53),
            dune_pipeline: pipeline(DuneShader, CullingMode::BackFace),
            stone_pipeline: pipeline(StoneShader, CullingMode::BackFace),
            solid: pipeline(VariedToonShader, CullingMode::BackFace),
            two_sided: pipeline(ToonShader, CullingMode::None),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for DesertDunes {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Flute],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::SandHiss),
                0.0,
                0.45,
                0.12,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let t = scene.local_time;
        let mut caravan = Shape::new();
        for k in 0..7 {
            let x = self.caravan_start + t * 0.45 + k as f32 * 3.6;
            let at = Mat4::translation(x, ground(x, CARAVAN_Z) - 0.05, CARAVAN_Z);
            let figure = if k == 0 {
                walker(t, 0.0)
            } else {
                camel(t, k as f32 * 0.7)
            };
            caravan.append(&figure.transformed(&at));
        }
        self.caravan = caravan;
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
        let surface = || SurfaceUniforms {
            toon: toon.clone(),
            sand: self.sand.clone(),
        };

        self.dunes
            .draw(pass, &self.dune_pipeline, motion(Motion::Still), surface());
        self.stones
            .draw(pass, &self.stone_pipeline, motion(Motion::Still), surface());
        self.caravan
            .draw(pass, &self.two_sided, motion(Motion::Still), toon.clone());

        scene.chip.draw(pass, view, &look());
        self.extras.draw(
            pass,
            &self.solid,
            motion(Motion::OnChip),
            VariedToonUniforms {
                toon: toon.clone(),
                variation: 0.08,
            },
        );

        let sky = SkyUniforms {
            view: view.clone(),
            sky: sky(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| sky.clone());

        let gust = (scene.local_time * 0.6).sin() * 0.5 + 0.5;
        self.grains.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, -1.2, 0.0),
                half_extent: Vec3::new(16.0, 1.4, 14.0),
                velocity: Vec3::new(5.0 + 2.0 * gust, 0.25, 1.2),
                wander: 0.3,
                size: (0.012, 0.03),
                colour: Vec3::new(1.5, 0.95, 0.5),
                twinkle: 0.0,
            },
        );
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, -0.8, -4.0),
                half_extent: Vec3::new(22.0, 2.2, 20.0),
                velocity: Vec3::new(3.0, 0.1, 0.6),
                wander: 1.2,
                size: (0.4, 0.9),
                colour: Vec3::new(0.09, 0.055, 0.03),
                twinkle: 0.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.7,
            vignette: 0.4,
            grain: 0.015,
            aberration: 0.002,
            desaturate: -0.05,
        }
    }
}
