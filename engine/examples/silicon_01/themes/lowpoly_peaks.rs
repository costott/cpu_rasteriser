//! lowpoly-peaks: five faceted mountain ranges fading into a dusk haze behind a still lake that
//! mirrors them, birds crossing and snow-dusted rock in front. The chip carries a lit summit
//! cabin.
//!
//! The ranges and the lake's reflection of them are one analytic function of a ray, so the lake
//! mirrors exactly what the sky shows.

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{F, clamp01, hash2, mix, mix3, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonShader, ToonUniforms,
    pipeline,
};
use crate::themes::kit::dress::rim_point;
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::sky::stars;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// The lake's surface.
const WATER: f32 = FLOOR - 0.3;
/// The direction towards the sun, which has just set behind the ranges.
const SUN: Vec3 = Vec3 {
    x: -0.75,
    y: 0.04,
    z: -0.66,
};

/// One range: a sheet standing across the lake at depth `z`, its ridge a polyline with a vertex
/// every `width` units, rising `amp` above the water.
struct Range {
    z: f32,
    width: f32,
    amp: f32,
    seed: f32,
    colour: Vec3,
    /// How much of the dusk haze it is dissolved in.
    haze: f32,
}

const RANGES: [Range; 5] = [
    Range {
        z: -60.0,
        width: 9.0,
        amp: 13.0,
        seed: 1.0,
        colour: Vec3 {
            x: 0.1,
            y: 0.09,
            z: 0.22,
        },
        haze: 0.1,
    },
    Range {
        z: -95.0,
        width: 12.0,
        amp: 18.0,
        seed: 4.0,
        colour: Vec3 {
            x: 0.2,
            y: 0.13,
            z: 0.32,
        },
        haze: 0.26,
    },
    Range {
        z: -140.0,
        width: 16.0,
        amp: 23.0,
        seed: 8.0,
        colour: Vec3 {
            x: 0.36,
            y: 0.2,
            z: 0.4,
        },
        haze: 0.42,
    },
    Range {
        z: -200.0,
        width: 20.0,
        amp: 28.0,
        seed: 12.0,
        colour: Vec3 {
            x: 0.55,
            y: 0.3,
            z: 0.44,
        },
        haze: 0.56,
    },
    Range {
        z: -290.0,
        width: 26.0,
        amp: 33.0,
        seed: 17.0,
        colour: Vec3 {
            x: 0.8,
            y: 0.42,
            z: 0.45,
        },
        haze: 0.7,
    },
];

const HAZE: Vec3 = Vec3 {
    x: 1.0,
    y: 0.56,
    z: 0.42,
};

/// A range's height above the water at `x`, and the slope of its facet (positive where it faces
/// the sun to the left).
#[inline(always)]
fn ridge(range: &Range, x: F) -> (F, F) {
    let peak = |k: F, seed: f32, amp: f32| {
        let h = hash2(k, s(seed));
        s(amp) * (s(0.12) + s(0.88) * h)
    };
    let along = |scale: f32, seed: f32, amp: f32| {
        let u = x * s(1.0 / scale);
        let k = u.floor();
        let (a, b) = (peak(k, seed, amp), peak(k + s(1.0), seed, amp));
        (mix(a, b, u - k), b - a)
    };
    let (coarse, slope) = along(range.width, range.seed, range.amp);
    let (fine, _) = along(range.width * 0.37, range.seed + 17.0, range.amp * 0.35);
    let envelope = fbm(x * s(1.0 / (range.width * 5.0)), s(range.seed), 2) * s(0.6) + s(0.7);
    ((coarse + fine) * envelope, slope)
}

/// The colour of a facet of `range` at height `y` and across `x`, faceted into triangles that
/// catch the last light on their sunward sides.
#[inline(always)]
fn range_colour(range: &Range, h: F, slope: F, x: F, y: F) -> Vec3Simd {
    let cell = range.width * 0.4;
    let (u, v) = (x * s(1.0 / cell), y * s(1.0 / (cell * 0.9)));
    let warp = fbm(u * s(0.45) + s(range.seed), v * s(0.45), 2) - s(0.5);
    let (u, v) = (u + warp * s(2.4) + v * s(0.35), v + warp * s(1.6));
    let (iu, iv) = (u.floor(), v.floor());
    let upper = ((u - iu) + (v - iv)).simd_gt(s(1.0)).select(s(1.0), s(0.0));
    let id = hash2(iu * s(2.0) + upper, iv + s(range.seed * 13.0));
    let lit = clamp01(slope * s(0.5 / (range.amp * 0.5)) + s(0.5));
    let shade = s(0.4) + lit * s(0.6) + upper * s(0.25) + (id - s(0.5)) * s(0.75);
    let mut colour = splat3(range.colour) * shade;

    let depth_in = h - y;
    let rim = (-depth_in * s(10.0 / range.amp)).exp();
    colour = colour + rgb(1.0, 0.45, 0.25) * (rim * s(0.4 * (1.0 - range.haze)));
    let snow = smoothstep(0.88, 0.96, y / h) * s(1.0 - range.haze * 0.8);
    colour = mix3(
        colour,
        rgb(0.95, 0.72, 0.85) * (shade * s(0.9) + s(0.15)),
        snow * s(0.7),
    );

    let mist = (-y * s(2.0 / range.amp)).exp() * s(0.2);
    let thickness = clamp01(s(range.haze) + mist * s(1.0 - range.haze));
    mix3(colour, splat3(HAZE) * s(0.92), thickness)
}

/// The dusk sky: peach at the horizon through violet to indigo, banded cloud lit from below, a
/// glow where the sun went down and a few stars.
#[inline(always)]
fn sky(dir: Vec3Simd, time: F) -> Vec3Simd {
    let up = dir.y.fast_max(s(0.0));
    let toward = dir.dot(splat3(SUN.normalise())).fast_max(s(0.0));
    let mut colour = mix3(
        rgb(1.0, 0.52, 0.32),
        rgb(0.55, 0.22, 0.42),
        smoothstep(0.0, 0.2, up),
    );
    colour = mix3(colour, rgb(0.04, 0.05, 0.2), smoothstep(0.14, 0.65, up));
    let glow = toward * toward * toward * (s(1.0) - smoothstep(0.0, 0.35, up));
    colour = colour + rgb(1.5, 0.62, 0.25) * glow * s(0.7);

    let azimuth = dir.x.atan2(-dir.z);
    let drift = time * s(0.004);
    let band = fbm(azimuth * s(3.5) + drift, up * s(20.0), 3);
    let streak = smoothstep(0.48, 0.74, band)
        * smoothstep(0.03, 0.12, up)
        * (s(1.0) - smoothstep(0.3, 0.55, up));
    let underlit = mix3(
        rgb(0.3, 0.13, 0.32),
        rgb(1.3, 0.5, 0.3),
        clamp01(toward * s(1.3) + (s(0.3) - up) * s(1.2)),
    );
    colour = mix3(colour, underlit, streak * s(0.75));

    let dark = smoothstep(0.3, 0.7, up);
    colour + rgb(1.0, 0.92, 0.9) * (stars(dir.x, dir.y, dir.z, 0.45, time) * dark)
}

/// The colour seen along a ray from `origin`: the ranges, nearest in front, over the sky.
#[inline(always)]
fn dusk(origin: Vec3Simd, dir: Vec3Simd, time: F, pixel_angle: f32) -> Vec3Simd {
    let mut colour = sky(dir, time);
    let towards = dir.z.simd_lt(s(-0.001));
    for range in RANGES.iter().rev() {
        let t = (s(range.z) - origin.z) / dir.z;
        let x = origin.x + dir.x * t;
        let y = (origin.y + dir.y * t - s(WATER)).fast_max(s(0.0));
        let (h, slope) = ridge(range, x);
        let reach = (t * s(pixel_angle)).fast_max(s(1e-3));
        let inside = clamp01((h - y) / reach + s(0.5)) * towards.select(s(1.0), s(0.0));
        colour = mix3(
            colour,
            range_colour(range, h.fast_max(s(0.1)), slope, x, y),
            inside,
        );
    }
    colour
}

fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(-0.8, 0.2, -0.1).normalise(),
        sun_colour: Vec3::new(2.4, 1.0, 0.5),
        sky: Vec3::new(0.2, 0.17, 0.4),
        ground: Vec3::new(0.12, 0.07, 0.1),
        bands: 3.0,
        rim: Vec3::new(0.9, 0.4, 0.2),
        fog: Fog {
            colour: Vec3::new(0.7, 0.34, 0.32),
            density: 0.0015,
            falloff: 0.05,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.26, 0.24, 0.34),
        die: Vec3::new(0.06, 0.1, 0.2),
        gold: Vec3::new(0.9, 0.58, 0.3),
        streams: Vec3::new(1.0, 0.62, 0.3),
        key: Vec3::new(-0.7, 0.35, -0.5).normalise(),
        key_colour: Vec3::new(2.2, 1.0, 0.55),
        sky: light.sky * 1.5,
        ground: light.ground,
        softbox: Vec3::new(0.35, 0.2, 0.25),
    }
}

/// How far the rocky island extends from the chip towards `angle`.
fn island_radius(angle: f32) -> f32 {
    3.7 + 0.5 * (angle * 3.0 + 1.0).sin() + 0.3 * (angle * 5.0 + 2.0).sin()
}

fn island(x: f32, z: f32) -> f32 {
    let r = x.hypot(z);
    let t = r / island_radius(z.atan2(x));
    let rough = (fbm_scalar(x * 0.3, z * 0.3, 3) - 0.5) * 1.1;
    let top = FLOOR + 0.12 + 0.25 * (1.0 - t).max(0.0);
    if t < 1.0 {
        top + rough * t.min(0.6)
    } else {
        top - (t - 1.0) * 3.5 + rough * 0.1
    }
}

fn rock_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let rock = lerp3(
        Vec3::new(0.22, 0.2, 0.3),
        Vec3::new(0.34, 0.3, 0.4),
        fbm_scalar(centroid.x * 0.5, centroid.z * 0.5, 2),
    );
    let snow =
        ((normal.y - 0.8) * 8.0).clamp(0.0, 1.0) * ((centroid.y - FLOOR) * 2.0).clamp(0.0, 1.0);
    lerp3(rock, Vec3::new(0.7, 0.66, 0.84), snow)
}

/// The lake: nearly all mirror, broken into long streaks by ripples, with dark water where it
/// meets the rock.
struct LakeShader;

impl FragmentShaderSimd<KitVaryings> for LakeShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let world = v.world();
        let time = s(view.frame.time);
        let incoming = (world - splat3(view.eye)).normalise();
        let ripple_x =
            (fbm(world.x * s(0.7), world.z * s(0.12) + time * s(0.05), 2) - s(0.5)) * s(0.1);
        let ripple_z = (fbm(
            world.x * s(0.2) + s(8.0),
            world.z * s(0.4) - time * s(0.07),
            2,
        ) - s(0.5))
            * s(0.05);
        let normal = Vec3Simd::new(ripple_x, s(1.0), ripple_z).normalise();
        let mut mirrored = incoming - normal * (incoming.dot(normal) * s(2.0));
        mirrored.y = mirrored.y.fast_max(s(0.01));
        let reflection = dusk(world, mirrored.normalise(), time, view.pixel_angle);

        let grazing = s(1.0) - incoming.y.abs();
        let fresnel = s(0.3) + s(0.7) * grazing * grazing * grazing;
        let depth = rgb(0.03, 0.035, 0.1);
        let radius = (world.x * world.x + world.z * world.z).sqrt();
        let reach = radius - s(7.0);
        let shore =
            (-reach.fast_max(s(0.0)) * s(0.35)).exp() * smoothstep(-2.0, 3.0, s(10.0) - radius);
        let colour = mix3(depth, reflection, fresnel) * (s(1.0) - shore * s(0.5));
        scene_colour(view, colour, world)
    }
}

/// An emissive surface, flickering a little like firelight.
struct GlowShader;

impl FragmentShaderSimd<KitVaryings> for GlowShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let flicker = s(1.0) + s(0.1) * (s(view.frame.time * 6.0) + v.world().x * s(3.0)).sin();
        scene_colour(view, v.colour() * flicker, v.world())
    }
}

/// A pine: three stacked cones on a short trunk.
fn pine(rng: &mut Rng) -> Shape {
    let mut tree = Shape::cylinder(0.12, 0.1, 0.5, 5, Vec3::new(0.12, 0.08, 0.08));
    for (k, (y, radius)) in [(0.3, 0.8), (1.0, 0.62), (1.7, 0.42)]
        .into_iter()
        .enumerate()
    {
        let green = lerp3(
            Vec3::new(0.05, 0.14, 0.16),
            Vec3::new(0.08, 0.2, 0.2),
            k as f32 / 2.0,
        );
        tree.append(
            &Shape::cone(radius, 1.1, 7, green).transformed(&Mat4::translation(0.0, y, 0.0)),
        );
    }
    tree.facet_variation(rng, 0.12)
}

/// A cabin in the chip's own space: a log box under a snowy gable with a chimney and two lit
/// windows. Returns the solid parts and the glowing parts.
fn cabin() -> (Shape, Shape) {
    let (x, z) = (-1.15, 1.2);
    let wood = Vec3::new(0.3, 0.16, 0.1);
    let mut solid = Shape::cuboid(
        Vec3::new(x - 0.4, 0.0, z - 0.3),
        Vec3::new(x + 0.4, 0.34, z + 0.3),
        wood,
    );
    let (eave, ridge) = (0.34, 0.62);
    let roof = Vec3::new(0.22, 0.1, 0.1);
    let snow = Vec3::new(0.85, 0.8, 0.95);
    let corner = |dx: f32, y: f32, dz: f32| Vec3::new(x + dx, y, z + dz);
    solid.quad(
        [
            corner(-0.48, eave, 0.38),
            corner(0.48, eave, 0.38),
            corner(0.48, ridge, 0.0),
            corner(-0.48, ridge, 0.0),
        ],
        roof,
    );
    solid.quad(
        [
            corner(-0.48, eave, -0.38),
            corner(-0.48, ridge, 0.0),
            corner(0.48, ridge, 0.0),
            corner(0.48, eave, -0.38),
        ],
        roof,
    );
    solid.quad(
        [
            corner(-0.5, eave + 0.03, 0.4),
            corner(0.5, eave + 0.03, 0.4),
            corner(0.5, ridge + 0.03, 0.0),
            corner(-0.5, ridge + 0.03, 0.0),
        ],
        snow,
    );
    solid.quad(
        [
            corner(-0.5, eave + 0.03, -0.4),
            corner(-0.5, ridge + 0.03, 0.0),
            corner(0.5, ridge + 0.03, 0.0),
            corner(0.5, eave + 0.03, -0.4),
        ],
        snow,
    );
    solid.append(&Shape::cuboid(
        corner(0.22, 0.4, -0.12),
        corner(0.34, 0.78, 0.0),
        Vec3::new(0.25, 0.2, 0.22),
    ));
    solid.triangle(
        corner(-0.4, eave, 0.3),
        corner(0.4, eave, 0.3),
        corner(0.0, ridge, 0.3),
        wood,
    );

    let warm = Vec3::new(3.2, 1.7, 0.5);
    let mut glow = Shape::new();
    for dx in [-0.2, 0.18] {
        glow.quad(
            [
                corner(dx - 0.07, 0.12, 0.305),
                corner(dx + 0.07, 0.12, 0.305),
                corner(dx + 0.07, 0.26, 0.305),
                corner(dx - 0.07, 0.26, 0.305),
            ],
            warm,
        );
    }
    (solid, glow)
}

pub struct LowpolyPeaks {
    island: Shape,
    rocks: Shape,
    pines: Shape,
    water: Shape,
    cabin: Shape,
    windows: Shape,
    birds: Shape,
    rock_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    lake: SimdPipeline<KitVertexShader, LakeShader>,
    glow: SimdPipeline<KitVertexShader, GlowShader>,
    sky: SimdPipeline<ScreenVertexShader, RangesSky>,
}

impl LowpolyPeaks {
    pub fn new() -> Self {
        let mut rng = Rng::new(3125);
        let island_shape = terrain(&ring_radii(1.2, 20.0, 22), 48, island, rock_colour);
        let water = terrain(
            &ring_radii(3.0, 700.0, 40),
            72,
            |_, _| WATER,
            |_, _| Vec3::ONE,
        );

        let boulders: Vec<Shape> = (0..4)
            .map(|k| {
                let size = Vec3::new(
                    rng.range(0.8, 1.5),
                    rng.range(0.6, 1.1),
                    rng.range(0.8, 1.4),
                );
                Shape::icosphere(1.0, 1, 0.3, k as f32 * 3.0, Vec3::new(0.3, 0.27, 0.38))
                    .transformed(&Mat4::scaling(size.x, size.y, size.z))
                    .colour_by(|p, n| rock_colour(p + Vec3::new(0.0, 0.5, 0.0), n))
            })
            .collect();
        let spots = scatter(
            &mut rng,
            14.0,
            2.6,
            600,
            island,
            |x, z| {
                let r = x.hypot(z);
                (r > 2.4 && r < island_radius(z.atan2(x)) - 0.2 && !(x.abs() < 2.6 && z > 0.0))
                    .then_some(1.0)
            },
            (0.7, 1.6),
        );
        let mut rocks = Batch::new();
        rocks.add(&boulders, &spots[..spots.len().min(5)], |p| {
            (p.seed * 4.0) as usize % 4
        });

        let trees: Vec<Shape> = (0..3).map(|_| pine(&mut rng)).collect();
        let sites = scatter(
            &mut rng,
            14.0,
            1.5,
            900,
            island,
            |x, z| {
                let r = x.hypot(z);
                (r > 2.5 && r < island_radius(z.atan2(x)) - 0.4 && z < 0.0).then_some(1.0)
            },
            (0.7, 1.3),
        );
        let mut pines = Batch::new();
        pines.add(&trees, &sites[..sites.len().min(6)], |p| {
            (p.seed * 3.0) as usize % 3
        });

        let outcrops: Vec<Placement> = [
            (-8.5, -3.0, 1.2),
            (7.5, -5.0, 1.0),
            (-13.0, -10.0, 2.0),
            (13.0, -13.0, 2.2),
            (3.0, -18.0, 1.8),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(x, z, scale))| Placement {
            position: Vec3::new(x, WATER - 0.2, z),
            yaw: i as f32 * 1.3,
            scale,
            seed: i as f32 * 0.19,
        })
        .collect();
        rocks.add(&boulders, &outcrops, |p| (p.seed * 5.0) as usize % 4);

        let (mut cabin, windows) = cabin();
        for _ in 0..7 {
            let (spot, _) = rim_point(&mut rng);
            if (spot.x + 1.15).abs() < 0.9 && (spot.z - 1.2).abs() < 0.7 {
                continue;
            }
            let size = rng.range(0.15, 0.3);
            cabin.append(
                &Shape::icosphere(
                    size,
                    1,
                    0.3,
                    rng.range(0.0, 20.0),
                    Vec3::new(0.32, 0.29, 0.4),
                )
                .transformed(&Mat4::translation_vec(
                    spot + Vec3::new(0.0, size * 0.3, 0.0),
                )),
            );
        }

        Self {
            island: island_shape,
            rocks: rocks.finish(),
            pines: pines.finish(),
            water,
            cabin,
            windows,
            birds: Shape::new(),
            rock_pipeline: pipeline(ToonShader, CullingMode::None),
            lake: pipeline(LakeShader, CullingMode::BackFace),
            glow: pipeline(GlowShader, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, RangesSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

/// A flock: each bird a pair of wings that beat out of step as the flock crosses the valley.
fn flock(time: f32) -> Shape {
    let mut flock = Shape::new();
    let ink = Vec3::new(0.05, 0.04, 0.1);
    for k in 0..9 {
        let phase = k as f32 * 0.9;
        let lead = (k % 3) as f32 * 3.2 + (k / 3) as f32 * 1.5;
        let x = (time * 2.2 + k as f32 * 17.0 + 70.0).rem_euclid(150.0) - 75.0 + lead;
        let y = FLOOR + 9.0 + 3.0 * (k / 3) as f32 + (time * 0.5 + phase).sin() * 0.6;
        let z = -34.0 - (k / 3) as f32 * 4.0 - (k % 3) as f32 * 2.0;
        let beat = (time * 5.5 + phase).sin();
        let tip = |side: f32| Vec3::new(x - 0.2, y + 0.9 * beat, z + side * 1.5);
        let (nose, tail) = (Vec3::new(x + 0.6, y, z), Vec3::new(x - 0.5, y, z));
        flock.triangle(nose, tail, tip(1.0), ink);
        flock.triangle(nose, tip(-1.0), tail, ink);
    }
    flock
}

impl Stage for LowpolyPeaks {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Horn],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::BirdCall),
                0.0,
                0.45,
                0.09,
                0.35,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.birds = flock(scene.local_time);
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

        self.island.draw(
            pass,
            &self.rock_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.rocks.draw(
            pass,
            &self.rock_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.pines.draw(
            pass,
            &self.rock_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.birds.draw(
            pass,
            &self.rock_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.water
            .draw(pass, &self.lake, motion(Motion::Still), view.clone());

        scene.chip.draw(pass, view, &look());
        self.cabin
            .draw(pass, &self.rock_pipeline, motion(Motion::OnChip), toon);
        self.windows
            .draw(pass, &self.glow, motion(Motion::OnChip), view.clone());

        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| view.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.3,
            vignette: 0.45,
            grain: 0.012,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

/// The sky and ranges as seen over the island.
struct RangesSky;

impl FragmentShaderSimd<ScreenVaryings> for RangesSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let origin = splat3(view.eye);
        to_colour(
            dusk(origin, ray, s(view.frame.time), view.pixel_angle),
            s(SKY_DISTANCE),
        )
    }
}
