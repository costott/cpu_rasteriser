//! A polar night: the chip frosted and crowned with ice on a wind-carved snow plain, pines and a
//! fox against curtains of green and violet aurora, snow glittering in the moonlight.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{F, clamp01, hash2, mix3, mul3, reflect, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonShader,
    ToonUniforms, pipeline,
};
use crate::themes::kit::dress::{chip_shadow, rim_point};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar, value};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shadow::GroundShadow;
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::sky::stars;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The height of the ground where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
const MOON: Vec3 = Vec3 {
    x: -0.55,
    y: 0.3,
    z: 0.62,
};
const WIND: Vec2 = Vec2 { x: 0.95, y: 0.3 };
const GREEN: Vec3 = Vec3 {
    x: 0.12,
    y: 1.0,
    z: 0.4,
};
const VIOLET: Vec3 = Vec3 {
    x: 0.6,
    y: 0.18,
    z: 1.0,
};

/// The ground: a flat patch round the chip, drifts with wind-carved crests stretched along the
/// wind, and snowy mountains beyond the pines for the horizon.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let open = ((r - 3.0) / 9.0).clamp(0.0, 1.0);
    let drift = (fbm_scalar(x * 0.035 + 2.0, z * 0.05, 3) - 0.5) * 3.2;
    let along = x * WIND.x + z * WIND.y;
    let across = -x * WIND.y + z * WIND.x;
    let crest = fbm_scalar(along * 0.05 + 7.0, across * 0.28, 3);
    let carved = (1.0 - (crest * 2.0 - 1.0).abs()).powi(3) * 0.55;
    let ridge_edge = ((r - 70.0) / 70.0).clamp(0.0, 1.0);
    let mountains =
        ridge_edge * ridge_edge * (6.0 + 36.0 * fbm_scalar(x * 0.018 + 3.0, z * 0.018 + 8.0, 4));
    FLOOR + open * (open * drift + carved) + mountains
}

fn lighting() -> Lighting {
    Lighting {
        sun: MOON.normalise(),
        sun_colour: Vec3::new(0.6, 0.72, 1.15),
        sky: Vec3::new(0.13, 0.22, 0.42),
        ground: Vec3::new(0.04, 0.07, 0.11),
        bands: 0.0,
        rim: Vec3::new(0.05, 0.14, 0.18),
        fog: Fog {
            colour: Vec3::new(0.025, 0.06, 0.12),
            density: 0.009,
            falloff: 0.04,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.62, 0.78, 0.92),
        die: Vec3::new(0.03, 0.1, 0.2),
        gold: Vec3::new(0.45, 0.85, 0.95),
        streams: Vec3::new(0.3, 1.0, 0.6),
        key: light.sun,
        key_colour: light.sun_colour * 1.5,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.05, 0.12, 0.22),
    }
}

fn terrain_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let r = (centroid.x * centroid.x + centroid.z * centroid.z).sqrt();
    let snow = Vec3::new(0.8, 0.9, 1.0);
    let shade = Vec3::new(0.52, 0.66, 0.88);
    let patch = fbm_scalar(centroid.x * 0.2 + 5.0, centroid.z * 0.2, 3);
    let mut c = lerp3(shade, snow, (patch * 1.6).clamp(0.0, 1.0));
    let rock = Vec3::new(0.2, 0.26, 0.36);
    c = lerp3(
        c,
        rock,
        ((0.8 - normal.y) * 4.0).clamp(0.0, 0.85) * ((r - 40.0) / 40.0).clamp(0.0, 1.0),
    );
    c * (0.95 + 0.1 * crate::simd::hash2_scalar(centroid.x, centroid.z))
}

/// The aurora's brightness and colour in the direction `ray`, as curtains of vertical rays
/// that fold and drift.
#[inline(always)]
fn aurora(ray: Vec3Simd, time: F) -> Vec3Simd {
    let toward = (-ray.z).fast_max(s(0.25));
    let u = ray.x / toward;
    let mut total = rgb(0.0, 0.0, 0.0);
    for k in 0..3 {
        let kf = k as f32;
        let fold = (u * s(1.3) + time * s(0.07) + s(kf * 2.1)).sin() * s(0.07)
            + (fbm(u * s(0.8) + s(kf * 5.0), time * s(0.04), 3) - s(0.5)) * s(0.12);
        let d = ray.y - s(0.07 + 0.06 * kf) - fold;
        let rays = value(
            u * s(34.0 + 6.0 * kf) + time * s(0.15),
            s(kf * 3.7) + time * s(0.2),
        );
        let reach = s(3.0) + (s(1.0) - rays) * s(7.0);
        let below = smoothstep(-0.02, 0.012, d);
        let height = d.fast_max(s(0.0));
        let profile = below * (-height * reach).exp();
        let strength = s(0.3)
            + s(0.7)
                * smoothstep(
                    0.3,
                    0.8,
                    fbm(u * s(1.5) + s(kf * 3.0) + time * s(0.05), s(1.0), 3),
                );
        let body = mix3(
            rgb(GREEN.x, GREEN.y, GREEN.z),
            rgb(VIOLET.x, VIOLET.y, VIOLET.z),
            smoothstep(0.02, 0.4, height),
        );
        total =
            total + body * (profile * strength * (s(0.35) + rays * s(0.9)) * s(1.15 - 0.25 * kf));
    }
    total
}

#[derive(Clone)]
struct SkyUniforms {
    view: ViewRef,
}

/// The polar sky: deep gradient, stars, and the aurora.
struct SkyShader;

impl FragmentShaderSimd<ScreenVaryings> for SkyShader {
    type Uniforms = SkyUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &SkyUniforms) -> ColourSimd {
        let [x, y] = v.ndc();
        let ray = view_ray(&u.view, x, y);
        let time = s(u.view.frame.time);
        let up = ray.y.fast_max(s(0.0));
        let mut colour = mix3(
            rgb(0.04, 0.1, 0.18),
            rgb(0.004, 0.01, 0.045),
            smoothstep(0.0, 0.8, up).sqrt(),
        );
        let glow = aurora(ray, time);
        colour = colour + glow;
        let dim = clamp01(s(1.0) - (glow.x + glow.y + glow.z) * s(1.5));
        colour = colour + rgb(0.85, 0.92, 1.0) * (stars(ray.x, ray.y, ray.z, 1.0, time) * dim);
        to_colour(colour, s(SKY_DISTANCE))
    }
}

struct GroundUniforms {
    toon: ToonUniforms,
    shadow: Arc<GroundShadow>,
}

/// Snow: wind ripples in the normal, the baked shadows of pines and ice, the chip's shadow, a
/// green wash from the aurora and glitter that fades with distance.
struct SnowShader;

impl FragmentShaderSimd<KitVaryings> for SnowShader {
    type Uniforms = GroundUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GroundUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = crate::simd::length(to_eye);
        let mut normal = facing(v.normal().normalise(), to_eye);
        let along = world.x * s(WIND.x) + world.z * s(WIND.y);
        let across = world.z * s(WIND.x) - world.x * s(WIND.y);
        let ripple = fbm(along * s(0.5), across * s(3.2), 2) - s(0.5);
        let fade = smoothstep(40.0, 8.0, distance);
        normal = (normal
            + Vec3Simd::new(s(WIND.y) * ripple, s(0.0), -s(WIND.x) * ripple) * (s(0.7) * fade))
            .normalise();
        let grain = fbm(world.x * s(1.6), world.z * s(1.6), 2);
        let albedo = v.colour() * (s(0.9) + grain * s(0.2));
        let time = view.frame.time;
        let shadow = u.shadow.lit(world.x, world.z)
            * (s(1.0) - chip_shadow(view, u.toon.lighting.sun, world));
        let mut colour = u
            .toon
            .lighting
            .shade(view, world, normal, albedo, clamp01(shadow));
        let pulse = 0.65 + 0.35 * (time * 0.5).sin();
        colour = colour + mul3(albedo, rgb(0.0, 0.1, 0.045)) * s(pulse);
        let cell = (world.x * s(34.0)).floor();
        let row = (world.z * s(34.0)).floor();
        let h = hash2(cell, row);
        let glint = smoothstep(0.985, 1.0, h)
            * (s(0.5) + s(0.5) * (s(time * 2.4) + h * s(200.0)).sin().fast_max(s(0.0)));
        colour = colour + rgb(0.9, 1.1, 1.3) * (glint * fade * s(1.6) * clamp01(shadow + s(0.3)));
        scene_colour(view, colour, world)
    }
}

/// Ice: a dim body lit by the moon, deepening to cyan towards its edges, with the aurora
/// mirrored in its facets and a hard glint.
struct IceShader;

impl FragmentShaderSimd<KitVaryings> for IceShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let eye_dir = to_eye.normalise();
        let normal = facing(v.normal().normalise(), to_eye);
        let edge = s(1.0) - clamp01(normal.dot(eye_dir));
        let fresnel = edge * edge * edge;
        let body = u.lighting.shade(view, world, normal, v.colour(), s(1.0));
        let depth = rgb(0.03, 0.2, 0.34) * (s(0.35) + edge * s(1.3));
        let mirrored = reflect(-eye_dir, normal);
        let sheen = aurora(
            Vec3Simd::new(mirrored.x, mirrored.y.abs(), mirrored.z),
            s(view.frame.time),
        );
        let glint = reflect(-splat3(u.lighting.sun), normal)
            .dot(eye_dir)
            .fast_max(s(0.0));
        let glint = glint * glint;
        let glint = glint * glint * glint * glint;
        let colour = body * s(0.55)
            + depth
            + sheen * (fresnel * s(1.8) + s(0.12))
            + rgb(0.8, 1.0, 1.3) * (glint * s(2.0));
        scene_colour(view, colour, world)
    }
}

fn pine(detail: bool, rng: &mut Rng) -> Shape {
    let segments = if detail { 8 } else { 6 };
    let bark = Vec3::new(0.1, 0.07, 0.06);
    let mut tree = Shape::cylinder(0.3, 0.14, 2.0, 5, bark);
    let tiers = if detail { 5 } else { 4 };
    for k in 0..tiers {
        let t = k as f32 / (tiers - 1) as f32;
        let radius = 2.4 * (1.0 - 0.66 * t);
        let base = 1.3 + k as f32 * 1.7;
        let cone = Shape::cone(radius, 3.2 - 0.4 * t, segments, Vec3::new(0.03, 0.09, 0.08))
            .transformed(
                &(Mat4::translation(0.0, base, 0.0) * Mat4::rotate_y(rng.range(0.0, TAU))),
            );
        tree.append(&cone);
    }
    let snow = Vec3::new(0.42, 0.55, 0.78);
    tree.colour_by(|p, n| {
        if p.y < 1.8 {
            bark
        } else if n.y > 0.42 {
            snow
        } else {
            lerp3(
                Vec3::new(0.02, 0.07, 0.07),
                Vec3::new(0.05, 0.13, 0.1),
                (p.y / 9.0).clamp(0.0, 1.0),
            )
        }
    })
    .facet_variation(rng, 0.1)
    .sway_by_height(2.0, 9.0, 1.6)
}

fn ice_block(rng: &mut Rng) -> Shape {
    let height = rng.range(0.9, 2.2);
    let radius = rng.range(0.5, 1.1);
    let profile = [
        (0.0, 0.0),
        (radius, 0.0),
        (radius * rng.range(0.8, 1.05), height * 0.8),
        (radius * 0.55, height),
        (0.0, height),
    ];
    let tilt = Mat4::rotate_x(rng.range(-0.35, 0.35)) * Mat4::rotate_z(rng.range(-0.35, 0.35));
    Shape::lathe(&profile, 6, Vec3::new(0.45, 0.78, 0.96))
        .transformed(
            &(Mat4::translation(0.0, -0.3, 0.0)
                * tilt
                * Mat4::scaling(1.0, 1.0, rng.range(0.7, 1.0))),
        )
        .facet_variation(rng, 0.18)
}

fn snow_rock(rng: &mut Rng) -> Shape {
    let rock = Vec3::new(0.2, 0.23, 0.3);
    Shape::icosphere(1.0, 1, 0.3, rng.range(0.0, 90.0), rock)
        .transformed(&Mat4::scaling(
            rng.range(0.9, 1.4),
            rng.range(0.55, 0.85),
            rng.range(0.8, 1.2),
        ))
        .colour_by(|_, n| {
            if n.y > 0.45 {
                Vec3::new(0.68, 0.8, 0.95)
            } else {
                rock
            }
        })
        .facet_variation(rng, 0.1)
}

/// A fox sitting side-on, in black, whose tail sways.
fn fox() -> Shape {
    let fur = Vec3::new(0.025, 0.03, 0.045);
    let mut fox = Shape::new();
    let blob = |centre: Vec3, scale: Vec3| {
        Shape::icosphere(1.0, 1, 0.08, centre.x * 9.0, fur).transformed(
            &(Mat4::translation_vec(centre) * Mat4::scaling(scale.x, scale.y, scale.z)),
        )
    };
    fox.append(&blob(
        Vec3::new(0.0, 0.55, 0.0),
        Vec3::new(0.62, 0.42, 0.34),
    ));
    fox.append(&blob(
        Vec3::new(0.62, 0.95, 0.0),
        Vec3::new(0.3, 0.27, 0.24),
    ));
    fox.append(&Shape::limb(
        Vec3::new(0.82, 0.93, 0.0),
        Vec3::new(1.2, 0.84, 0.0),
        0.15,
        0.03,
        5,
        fur,
    ));
    for side in [-0.1, 0.1] {
        fox.append(
            &Shape::cone(0.11, 0.34, 4, fur).transformed(&Mat4::translation(
                0.58,
                1.12,
                side * 1.5,
            )),
        );
    }
    fox.append(&Shape::limb(
        Vec3::new(0.35, 0.2, 0.12),
        Vec3::new(0.4, 0.0, 0.14),
        0.09,
        0.06,
        5,
        fur,
    ));
    fox.append(&Shape::limb(
        Vec3::new(0.35, 0.2, -0.12),
        Vec3::new(0.4, 0.0, -0.14),
        0.09,
        0.06,
        5,
        fur,
    ));
    fox.append(&Shape::limb(
        Vec3::new(-0.25, 0.15, 0.0),
        Vec3::new(-0.2, 0.0, 0.0),
        0.2,
        0.08,
        5,
        fur,
    ));
    let tail = [
        Vec3::new(-0.5, 0.3, 0.0),
        Vec3::new(-1.0, 0.18, 0.0),
        Vec3::new(-1.5, 0.3, 0.0),
        Vec3::new(-1.8, 0.5, 0.0),
    ];
    for pair in tail.windows(2) {
        let radius = 0.16 * (1.0 - (pair[1].x / -2.4).clamp(0.0, 1.0) * 0.4);
        fox.append(&Shape::limb(pair[0], pair[1], radius, radius * 0.8, 5, fur));
    }
    fox.append(
        &Shape::cone(0.12, 0.3, 5, Vec3::new(0.55, 0.62, 0.7))
            .transformed(&(Mat4::translation(-1.8, 0.5, 0.0) * Mat4::rotate_z(FRAC_PI_2 * 0.9))),
    );
    fox.sway_by(|p| ((-p.x - 0.7) / 1.1).clamp(0.0, 1.0))
}

/// Frost on the chip: snow banked along its edge, ice crystals growing off the rim and
/// icicles hanging from it, in the chip's own space. Returns (snow, ice).
fn chip_frost(rng: &mut Rng) -> (Shape, Shape) {
    let mut snow = Shape::new();
    for _ in 0..30 {
        let (spot, _) = rim_point(rng);
        let radius = rng.range(0.14, 0.4);
        snow.append(
            &Shape::icosphere(
                1.0,
                1,
                0.25,
                rng.range(0.0, 40.0),
                Vec3::new(0.82, 0.92, 1.0),
            )
            .transformed(
                &(Mat4::translation_vec(spot) * Mat4::scaling(radius * 1.2, radius * 0.45, radius)),
            )
            .facet_variation(rng, 0.1),
        );
    }
    let mut ice = Shape::new();
    for _ in 0..22 {
        let (spot, outward) = rim_point(rng);
        let height = rng.range(0.35, 1.0);
        let lean = outward * rng.range(0.1, 0.5)
            + Vec3::new(rng.range(-0.15, 0.15), 0.0, rng.range(-0.15, 0.15));
        let tip = spot + Vec3::new(0.0, height, 0.0) + lean;
        let crystal = Shape::limb(
            spot,
            tip,
            rng.range(0.06, 0.12),
            0.0,
            6,
            Vec3::new(0.55, 0.85, 1.0),
        );
        ice.append(&crystal.facet_variation(rng, 0.2));
    }
    for _ in 0..18 {
        let (spot, outward) = rim_point(rng);
        let from = spot + outward * rng.range(0.3, 0.5);
        let length = rng.range(0.35, 1.0);
        ice.append(&Shape::limb(
            from,
            from + Vec3::new(0.0, -length, 0.0),
            0.07,
            0.0,
            5,
            Vec3::new(0.7, 0.92, 1.0),
        ));
    }
    (snow, ice)
}

pub struct ArcticAurora {
    terrain: Shape,
    trees: Shape,
    ice: Shape,
    rocks: Shape,
    fox: Shape,
    frost_snow: Shape,
    frost_ice: Shape,
    shadow: Arc<GroundShadow>,
    flurry: Particles,
    spindrift: Particles,
    snow_pipeline: SimdPipeline<KitVertexShader, SnowShader>,
    ice_pipeline: SimdPipeline<KitVertexShader, IceShader>,
    plain: SimdPipeline<KitVertexShader, ToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyShader>,
}

impl ArcticAurora {
    pub fn new() -> Self {
        let mut rng = Rng::new(7117);
        let terrain = terrain(&ring_radii(1.5, 260.0, 46), 56, ground, terrain_colour);

        let pines: Vec<Shape> = (0..3).map(|_| pine(true, &mut rng)).collect();
        let far_pines: Vec<Shape> = (0..3).map(|_| pine(false, &mut rng)).collect();
        let clear_of_chip = |x: f32, z: f32| {
            let r = (x * x + z * z).sqrt();
            if r < 24.0 || (x.abs() < 4.0 && z > 0.0) || z > 16.0 {
                return None;
            }
            Some(if z > -4.0 { 1.5 } else { 1.0 })
        };
        let all = scatter(&mut rng, 90.0, 7.5, 9000, ground, clear_of_chip, (0.6, 1.2));
        let (near, far): (Vec<Placement>, Vec<Placement>) = all
            .into_iter()
            .partition(|p| (p.position.x.powi(2) + p.position.z.powi(2)).sqrt() < 34.0);
        let mut trees = Batch::new();
        trees.add(&pines, &near, |p| (p.seed * 13.0) as usize % 3);
        trees.add(&far_pines, &far, |p| (p.seed * 13.0) as usize % 3);
        let trees = trees.finish();

        let blocks: Vec<Shape> = (0..5).map(|_| ice_block(&mut rng)).collect();
        let mut block_spots = scatter(
            &mut rng,
            30.0,
            5.0,
            600,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 5.5 || (x.abs() < 3.5 && z > 0.0) {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.8, 2.2),
        );
        block_spots.truncate(16);
        let rocks_list: Vec<Shape> = (0..4).map(|_| snow_rock(&mut rng)).collect();
        let rock_spots = scatter(
            &mut rng,
            40.0,
            6.0,
            400,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 7.0 || (x.abs() < 3.5 && z > 0.0) {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.5, 1.8),
        );
        let mut rocks = Batch::new();
        rocks.add(&rocks_list, &rock_spots[..rock_spots.len().min(14)], |p| {
            (p.seed * 11.0) as usize % 4
        });
        let rocks = rocks.finish();
        let mut ice = Batch::new();
        ice.add(&blocks, &block_spots, |p| (p.seed * 11.0) as usize % 5);
        let ice = ice.finish();

        let fox_at = Vec3::new(21.0, ground(21.0, -31.0), -31.0);
        let fox = fox().transformed(
            &(Mat4::translation_vec(fox_at) * Mat4::rotate_y(0.9) * Mat4::scaling(1.6, 1.6, 1.6)),
        );

        let shadow = Arc::new(GroundShadow::bake(
            Vec2::new(-80.0, -80.0),
            160.0,
            1024,
            MOON.normalise(),
            ground,
            &[&trees, &rocks, &ice],
            2,
        ));
        let (frost_snow, frost_ice) = chip_frost(&mut rng);
        Self {
            terrain,
            trees,
            ice,
            rocks,
            fox,
            frost_snow,
            frost_ice,
            shadow,
            flurry: Particles::new(260, 41),
            spindrift: Particles::new(140, 73),
            snow_pipeline: pipeline(SnowShader, CullingMode::BackFace),
            ice_pipeline: pipeline(IceShader, CullingMode::BackFace),
            plain: pipeline(ToonShader, CullingMode::BackFace),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for ArcticAurora {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Glockenspiel],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::IcyGust),
                0.0,
                0.45,
                0.16,
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
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let tree_sway = Motion::Sway(Sway {
            amplitude: 0.18,
            speed: 0.9,
            wind: Vec2::new(WIND.x / WIND.length(), WIND.y / WIND.length()),
        });
        let tail_sway = Motion::Sway(Sway {
            amplitude: 0.12,
            speed: 1.4,
            wind: Vec2::new(0.0, 1.0),
        });

        self.terrain.draw(
            pass,
            &self.snow_pipeline,
            motion(Motion::Still),
            GroundUniforms {
                toon: toon.clone(),
                shadow: self.shadow.clone(),
            },
        );
        self.trees
            .draw(pass, &self.plain, motion(tree_sway), toon.clone());
        self.rocks
            .draw(pass, &self.plain, motion(Motion::Still), toon.clone());
        self.ice.draw(
            pass,
            &self.ice_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.fox
            .draw(pass, &self.plain, motion(tail_sway), toon.clone());

        scene.chip.draw(pass, view, &look());
        self.frost_snow
            .draw(pass, &self.plain, motion(Motion::OnChip), toon.clone());
        self.frost_ice.draw(
            pass,
            &self.ice_pipeline,
            motion(Motion::OnChip),
            toon.clone(),
        );

        let sky = SkyUniforms { view: view.clone() };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| sky.clone());

        self.flurry.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.0, 2.0),
                half_extent: Vec3::new(16.0, 5.0, 13.0),
                velocity: Vec3::new(0.7, -0.35, 0.25),
                wander: 0.6,
                size: (0.018, 0.045),
                colour: Vec3::new(0.9, 1.1, 1.4),
                twinkle: 2.0,
            },
        );
        self.spindrift.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, FLOOR + 0.5, 2.0),
                half_extent: Vec3::new(20.0, 0.7, 14.0),
                velocity: Vec3::new(3.2, 0.0, 1.0),
                wander: 0.25,
                size: (0.02, 0.05),
                colour: Vec3::new(0.5, 0.65, 0.9),
                twinkle: 0.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.05,
            bloom: 0.75,
            vignette: 0.42,
            grain: 0.014,
            aberration: 0.0012,
            desaturate: 0.0,
        }
    }
}
