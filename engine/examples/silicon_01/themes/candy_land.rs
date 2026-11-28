//! candy-land: glossy pastels. Frosting hills crossed by a syrup river, lollipop trees with
//! spirals that turn, gumdrops and candy canes, sprinkles drifting in the air, and the chip
//! glazed like a sweet.

use std::f32::consts::{PI, TAU};

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::ScreenVertexShader;
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{
    F, clamp01, fract, hash2, length, mix3, pow_highlight, rgb, s, smoothstep, splat3,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonUniforms,
    pipeline,
};
use crate::themes::kit::dress::rim_point;
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shape::{KitVertex, Shape, lerp3};
use crate::themes::kit::sky::{Clouds, Sky, SkyShader, SkyUniforms};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.5,
    y: 0.55,
    z: 0.45,
};
const WIND: Vec2 = Vec2 { x: 0.8, y: 0.6 };
/// The syrup's surface, a little below the ground beside it.
const SYRUP: f32 = FLOOR - 0.35;

/// Where the river runs at depth `z`: it winds from the foreground, past the chip's right
/// and behind it, away to the left.
fn river(z: f32) -> f32 {
    9.0 * (z * 0.1 + 2.2).sin() + 3.0
}

/// How far into the river's bed a point is, 0 on the bank to 1 in the channel.
fn channel(x: f32, z: f32) -> f32 {
    1.0 - smooth((x - river(z)).abs(), 2.4, 5.5)
}

fn smooth(x: f32, low: f32, high: f32) -> f32 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rolling icing hills round a flat clearing, with a channel cut for the river.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let clearing = (r / 8.0).clamp(0.0, 1.0);
    let swell = (fbm_scalar(x * 0.05 + 2.0, z * 0.06, 3) - 0.35).max(0.0) * 3.2;
    let ridge_edge = ((r - 55.0) / 60.0).clamp(0.0, 1.0);
    let ridge =
        ridge_edge * ridge_edge * (5.0 + 20.0 * fbm_scalar(x * 0.02 + 3.0, z * 0.02 + 7.0, 3));
    let bank = FLOOR + clearing * clearing * swell + ridge;
    let bed = FLOOR - 0.7;
    bank + (bed - bank) * channel(x, z)
}

fn frosting(centroid: Vec3, normal: Vec3) -> Vec3 {
    let drift = fbm_scalar(centroid.x * 0.07 + 5.0, centroid.z * 0.07, 3);
    let pink = Vec3::new(0.85, 0.28, 0.5);
    let cream = Vec3::new(0.95, 0.62, 0.62);
    let lilac = Vec3::new(0.8, 0.68, 0.95);
    let mut c = lerp3(pink, cream, smooth(drift, 0.45, 0.65));
    let r = (centroid.x * centroid.x + centroid.z * centroid.z).sqrt();
    c = lerp3(c, lilac, smooth(r, 40.0, 130.0) * 0.8);
    lerp3(
        c,
        Vec3::new(0.84, 0.46, 0.62),
        ((0.95 - normal.y) * 4.0).clamp(0.0, 0.5),
    )
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.2, 1.08, 0.98),
        sky: Vec3::new(0.42, 0.4, 0.62),
        ground: Vec3::new(0.32, 0.22, 0.28),
        bands: 0.0,
        rim: Vec3::new(0.2, 0.16, 0.26),
        fog: Fog {
            colour: Vec3::new(0.85, 0.6, 0.78),
            density: 0.0025,
            falloff: 0.04,
            base: FLOOR,
        },
    }
}

fn sky() -> Sky {
    Sky {
        zenith: Vec3::new(0.3, 0.42, 0.95),
        horizon: Vec3::new(1.1, 0.72, 0.85),
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.6, 1.3, 1.0),
        sun_size: 0.04,
        glow: 0.8,
        clouds: Some(Clouds {
            height: 320.0,
            coverage: 0.5,
            scale: 0.005,
            wind: Vec2::new(6.0, 2.0),
            lit: Vec3::new(1.6, 1.3, 1.45),
            shade: Vec3::new(0.85, 0.68, 0.95),
        }),
        stars: 0.0,
    }
}

/// The chip as a sweet: a glazed pink package, a mint die and gold sugar pins.
fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.95, 0.5, 0.65),
        die: Vec3::new(0.22, 0.7, 0.52),
        gold: Vec3::new(1.0, 0.78, 0.3),
        streams: Vec3::new(1.0, 0.55, 0.85),
        key: light.sun,
        key_colour: light.sun_colour * 0.65,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.5, 0.4, 0.5),
    }
}

/// Pairs of candy colours: a stripe colour and its cream.
const PALETTES: [(Vec3, Vec3); 5] = [
    (
        Vec3 {
            x: 0.98,
            y: 0.4,
            z: 0.6,
        },
        Vec3 {
            x: 1.0,
            y: 0.94,
            z: 0.9,
        },
    ),
    (
        Vec3 {
            x: 0.35,
            y: 0.85,
            z: 0.7,
        },
        Vec3 {
            x: 0.98,
            y: 1.0,
            z: 0.96,
        },
    ),
    (
        Vec3 {
            x: 1.0,
            y: 0.84,
            z: 0.3,
        },
        Vec3 {
            x: 1.0,
            y: 0.97,
            z: 0.86,
        },
    ),
    (
        Vec3 {
            x: 0.66,
            y: 0.5,
            z: 0.95,
        },
        Vec3 {
            x: 0.98,
            y: 0.95,
            z: 1.0,
        },
    ),
    (
        Vec3 {
            x: 1.0,
            y: 0.58,
            z: 0.3,
        },
        Vec3 {
            x: 1.0,
            y: 0.95,
            z: 0.88,
        },
    ),
];

/// A striped pole of a lollipop, twisting like a barley sugar.
fn stem(palette: (Vec3, Vec3)) -> Shape {
    let rings: Vec<(f32, f32)> = std::iter::once((0.0, 0.0))
        .chain((0..=12).map(|i| (0.13 - 0.01 * (i as f32 / 12.0), i as f32 * 0.25)))
        .chain(std::iter::once((0.0, 3.0)))
        .collect();
    Shape::lathe(&rings, 7, palette.1).colour_by(move |p, _| {
        let twist = (p.z.atan2(p.x) / TAU * 2.0 + p.y * 0.55).rem_euclid(1.0);
        if twist < 0.5 { palette.0 } else { palette.1 }
    })
}

fn gumdrop(colour: Vec3) -> Shape {
    Shape::lathe(
        &[
            (0.0, 0.0),
            (0.62, 0.0),
            (0.64, 0.12),
            (0.54, 0.46),
            (0.3, 0.7),
            (0.0, 0.78),
        ],
        10,
        colour,
    )
}

/// A candy cane: a straight shaft and a hooked top, in alternating stripes.
fn candy_cane(palette: (Vec3, Vec3)) -> Shape {
    let height = 2.4;
    let hook = 0.55;
    let mut points: Vec<Vec3> = (0..=6)
        .map(|i| Vec3::new(0.0, height * i as f32 / 6.0, 0.0))
        .collect();
    points.extend((1..=6).map(|i| {
        let angle = PI * i as f32 / 6.0;
        Vec3::new(hook * (1.0 - angle.cos()), height + hook * angle.sin(), 0.0)
    }));
    let mut cane = Shape::new();
    for (i, pair) in points.windows(2).enumerate() {
        let colour = if i % 2 == 0 { palette.0 } else { palette.1 };
        cane.append(&Shape::limb(pair[0], pair[1], 0.11, 0.11, 5, colour));
    }
    cane
}

/// A sprinkle: a short rounded stick, tilted a random way.
fn sprinkle(colour: Vec3, rng: &mut Rng) -> Shape {
    let tilt = Mat4::rotate_y(rng.range(0.0, TAU)) * Mat4::rotate_z(rng.range(0.0, PI));
    Shape::limb(
        Vec3::new(-0.16, 0.0, 0.0),
        Vec3::new(0.16, 0.0, 0.0),
        0.05,
        0.05,
        4,
        colour,
    )
    .transformed(&tilt)
}

/// Dollops of icing and sprinkles along the chip's edge, in its own space.
fn chip_icing(rng: &mut Rng) -> Shape {
    let mut icing = Shape::new();
    for _ in 0..26 {
        let (spot, _) = rim_point(rng);
        let radius = rng.range(0.14, 0.3);
        let palette = PALETTES[rng.index(PALETTES.len())];
        let dollop = Shape::icosphere(1.0, 1, 0.12, rng.range(0.0, 40.0), palette.1).transformed(
            &(Mat4::translation_vec(spot) * Mat4::scaling(radius, radius * 0.6, radius)),
        );
        icing.append(&dollop);
        icing.append(
            &sprinkle(palette.0, rng).transformed(&Mat4::translation_vec(
                spot + Vec3::new(0.0, radius * 0.7, 0.0),
            )),
        );
    }
    icing
}

struct GlossUniforms {
    toon: ToonUniforms,
    /// How strong the sun's highlight is.
    gloss: f32,
    /// 1 for the frosted ground, with swirls of cream icing and glints of sugar; 0 for smooth candy.
    frosted: f32,
}

/// Glazed candy: lit by the sun and sky with a hard highlight and a bright edge, and optionally
/// strewn with glints of sugar.
struct GlossShader;

impl FragmentShaderSimd<KitVaryings> for GlossShader {
    type Uniforms = GlossUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GlossUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let normal = facing(v.normal().normalise(), to_eye);
        let icing = smoothstep(
            0.5,
            0.62,
            fbm(
                world.x * s(0.12) + s(2.0),
                world.z * s(0.12) + fbm(world.x * s(0.4), world.z * s(0.4), 2) * s(1.5),
                3,
            ),
        );
        let albedo = mix3(v.colour(), rgb(1.0, 0.8, 0.78), icing * s(0.3 * u.frosted));
        let mut colour = u.toon.lighting.shade(view, world, normal, albedo, s(1.0));
        colour = colour
            + glaze(
                &u.toon.lighting,
                normal,
                to_eye * (s(1.0) / distance),
                distance,
                u.gloss,
            );
        if u.frosted > 0.0 {
            let t = view.frame.time;
            let (cx, cz) = ((world.x * s(24.0)).floor(), (world.z * s(24.0)).floor());
            let seed = hash2(cx, cz);
            let on = smoothstep(0.97, 0.99, seed);
            let blink = (s(t * 2.5) + seed * s(60.0)).sin() * s(0.5) + s(0.5);
            let near = s(1.0) - smoothstep(10.0, 45.0, distance);
            colour = colour + rgb(1.4, 1.3, 1.25) * (on * blink * blink * near * s(u.frosted));
        }
        scene_colour(view, colour, world)
    }
}

/// The sun's hard highlight and a pale glow where the surface turns away, the look of a glaze.
#[inline(always)]
fn glaze(
    light: &Lighting,
    normal: Vec3Simd,
    to_eye: Vec3Simd,
    distance: F,
    gloss: f32,
) -> Vec3Simd {
    let half = (splat3(light.sun) + to_eye).normalise();
    let highlight = pow_highlight(normal.dot(half).fast_max(s(0.0)), 64.0);
    let turned = s(1.0) - clamp01(normal.dot(to_eye));
    let edge = turned * turned * turned;
    let fade = (-distance * s(0.012)).exp();
    (splat3(light.sun_colour) * highlight * s(gloss) + rgb(0.35, 0.3, 0.4) * edge * s(gloss)) * fade
}

#[derive(Clone)]
struct HeadUniforms {
    view: ViewRef,
    lighting: Lighting,
    centre: Vec3,
    radius: f32,
    yaw: f32,
    palette: (Vec3, Vec3),
    /// How many spiral arms, and how fast they turn.
    arms: f32,
    phase: f32,
}

/// How thin a lollipop's head is compared with its width.
const HEAD_FLAT: f32 = 0.42;

#[derive(Interpolate, SimdInterpolate)]
struct HeadVaryings {
    local: Vec3,
    world: Vec3,
}

/// Places a unit sphere as a lollipop's flattened head: scaled, turned about the vertical and
/// bobbing a little.
struct HeadVertexShader;

impl VertexShader for HeadVertexShader {
    type Vertex = KitVertex;
    type Uniforms = HeadUniforms;
    type Varyings = HeadVaryings;

    fn shade(&self, vertex: KitVertex, u: &HeadUniforms) -> (Vec4, HeadVaryings) {
        let time = u.view.frame.time;
        let p = vertex.position;
        let (sin, cos) = u.yaw.sin_cos();
        let (x, z) = (p.x * u.radius, p.z * u.radius * HEAD_FLAT);
        let bob = (time * 0.9 + u.phase).sin() * 0.06 * u.radius;
        let world =
            u.centre + Vec3::new(x * cos + z * sin, p.y * u.radius + bob, -x * sin + z * cos);
        (
            u.view.view_projection * world.to_point4(),
            HeadVaryings { local: p, world },
        )
    }
}

/// A lollipop's head: a smooth glossy ellipsoid carrying a spiral of two candy colours,
/// computed per pixel so it stays crisp and turns slowly.
struct HeadShader;

impl FragmentShaderSimd<HeadVaryings> for HeadShader {
    type Uniforms = HeadUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: HeadVaryingsSimd, u: &HeadUniforms) -> ColourSimd {
        let local = Vec3Simd::new(v.local[0], v.local[1], v.local[2]);
        let world = Vec3Simd::new(v.world[0], v.world[1], v.world[2]);
        let to_eye = splat3(u.view.eye) - world;
        let distance = length(to_eye);

        let (sin, cos) = u.yaw.sin_cos();
        let squash = s(1.0 / (HEAD_FLAT * HEAD_FLAT));
        let (nx, ny, nz) = (local.x, local.y, local.z * squash);
        let normal =
            Vec3Simd::new(nx * s(cos) + nz * s(sin), ny, -nx * s(sin) + nz * s(cos)).normalise();

        let radius = (local.x * local.x + local.y * local.y).sqrt();
        let turn = local.y.atan2(local.x) / s(TAU) * s(u.arms) + radius * s(1.6)
            - s(u.view.frame.time * 0.12 + u.phase);
        let wave = (fract(turn) - s(0.5)).abs() * s(2.0);
        let stripe = smoothstep(0.42, 0.58, wave);
        let rim = smoothstep(0.82, 0.9, radius);
        let (a, b) = u.palette;
        let body = mix3(splat3(a), splat3(b), stripe);
        let albedo = mix3(body, splat3(b), rim);

        let mut colour = u.lighting.shade(&u.view, world, normal, albedo, s(1.0));
        colour = colour
            + glaze(
                &u.lighting,
                normal,
                to_eye * (s(1.0) / distance),
                distance,
                1.0,
            );
        scene_colour(&u.view, colour, world)
    }
}

/// The syrup: deep colour under a bright sky reflection, with ripples and slow bands of sheen
/// that run with the current.
struct SyrupShader;

impl FragmentShaderSimd<KitVaryings> for SyrupShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let t = view.frame.time;
        let flow = s(t * 0.9);
        let ripple_x = fbm(world.x * s(0.9), world.z * s(0.35) + flow, 3) - s(0.5);
        let ripple_z = fbm(
            world.x * s(0.9) + s(7.0),
            world.z * s(0.35) + flow * s(0.8),
            3,
        ) - s(0.5);
        let normal = Vec3Simd::new(ripple_x * s(0.5), s(1.0), ripple_z * s(0.5)).normalise();

        let view_dir = to_eye * (s(1.0) / distance);
        let turned = s(1.0) - clamp01(normal.dot(view_dir));
        let fresnel = turned * turned * turned * s(0.8) + s(0.12);
        let depth = fbm(world.x * s(0.3), world.z * s(0.12) + flow * s(0.3), 3);
        let syrup = mix3(
            rgb(0.55, 0.06, 0.2),
            rgb(0.95, 0.4, 0.18),
            smoothstep(0.35, 0.7, depth),
        );
        let reflection = mix3(
            rgb(1.0, 0.8, 0.9),
            rgb(0.7, 0.75, 1.1),
            clamp01(normal.y * s(0.5) + s(0.4)),
        );
        let base = mix3(syrup, reflection, fresnel);

        let half = (splat3(u.lighting.sun) + view_dir).normalise();
        let glint = pow_highlight(normal.dot(half).fast_max(s(0.0)), 120.0) * s(2.2);
        let sheen = smoothstep(
            0.55,
            0.75,
            fbm(world.x * s(0.5), world.z * s(0.12) - flow * s(0.6), 3),
        );
        let colour = base + splat3(u.lighting.sun_colour) * (glint * (s(0.4) + sheen));
        let hazy = u.lighting.fog.apply(colour, world, distance);
        scene_colour(view, hazy, world)
    }
}

struct Lolly {
    foot: Vec3,
    scale: f32,
    yaw: f32,
    palette: usize,
    arms: f32,
    phase: f32,
}

pub struct CandyLand {
    land: Shape,
    river: Shape,
    stems: Shape,
    treats: Shape,
    sprinkles: Shape,
    icing: Shape,
    heads: Shape,
    lollies: Vec<Lolly>,
    sparkle: Particles,
    gloss: SimdPipeline<KitVertexShader, GlossShader>,
    syrup: SimdPipeline<KitVertexShader, SyrupShader>,
    head: SimdPipeline<HeadVertexShader, HeadShader>,
    sky: SimdPipeline<ScreenVertexShader, SkyShader>,
}

impl CandyLand {
    pub fn new() -> Self {
        let mut rng = Rng::new(1977);
        let land = terrain(&ring_radii(1.5, 230.0, 46), 64, ground, frosting);

        let spine: Vec<Vec3> = (0..90)
            .map(|i| {
                let z = 22.0 - i as f32 * 2.0;
                Vec3::new(river(z), SYRUP, z)
            })
            .collect();
        let widths = vec![9.0; spine.len()];
        let river_surface = Shape::ribbon(&spine, &widths, Vec3::new(1.0, 0.0, 0.0), Vec3::ONE);

        let free = |x: f32, z: f32| {
            (x * x + z * z).sqrt() > 5.0
                && !(x.abs() < 3.4 && z > -1.0)
                && z < 12.0
                && (x - river(z)).abs() > 6.5
        };
        let spots = scatter(
            &mut rng,
            40.0,
            6.0,
            5000,
            ground,
            |x, z| if free(x, z) { Some(1.0) } else { None },
            (0.8, 1.5),
        );
        let lollies: Vec<Lolly> = spots
            .iter()
            .take(34)
            .map(|p| Lolly {
                foot: p.position,
                scale: p.scale,
                yaw: (p.seed * 9.0).fract() * 1.2 - 0.6,
                palette: (p.seed * 5.0) as usize % PALETTES.len(),
                arms: 2.0 + ((p.seed * 7.0) as usize % 3) as f32,
                phase: p.seed * TAU,
            })
            .collect();
        let stems: Vec<Shape> = PALETTES.iter().map(|&p| stem(p)).collect();
        let mut stem_batch = Batch::new();
        let stem_places: Vec<Placement> = spots
            .iter()
            .take(34)
            .map(|p| Placement { yaw: 0.0, ..*p })
            .collect();
        stem_batch.add(&stems, &stem_places, |p| {
            (p.seed * 5.0) as usize % PALETTES.len()
        });

        let drops: Vec<Shape> = PALETTES.iter().map(|&(a, _)| gumdrop(a)).collect();
        let drop_spots = scatter(
            &mut rng,
            32.0,
            2.6,
            4000,
            ground,
            |x, z| if free(x, z) { Some(1.0) } else { None },
            (0.7, 2.2),
        );
        let canes: Vec<Shape> = PALETTES.iter().take(3).map(|&p| candy_cane(p)).collect();
        let cane_spots = scatter(
            &mut rng,
            30.0,
            5.0,
            3000,
            ground,
            |x, z| if free(x, z) { Some(1.0) } else { None },
            (0.9, 1.5),
        );
        let mut treats = Batch::new();
        treats.add(&drops, &drop_spots[..drop_spots.len().min(50)], |p| {
            (p.seed * 9.0) as usize % PALETTES.len()
        });
        treats.add(&canes, &cane_spots[..cane_spots.len().min(16)], |p| {
            (p.seed * 9.0) as usize % 3
        });

        let sticks: Vec<Shape> = PALETTES
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .map(|c| sprinkle(c, &mut rng))
            .collect();
        let airborne: Vec<Placement> = (0..150)
            .filter_map(|_| {
                let (x, y, z) = (
                    rng.range(-18.0, 18.0),
                    rng.range(-1.2, 6.0),
                    rng.range(-34.0, 9.0),
                );
                (!(x.abs() < 2.8 && z > -3.0)).then_some(Placement {
                    position: Vec3::new(x, y, z),
                    yaw: rng.range(0.0, TAU),
                    scale: rng.range(0.9, 1.6),
                    seed: rng.unit(),
                })
            })
            .collect();
        let mut sprinkle_batch = Batch::new();
        sprinkle_batch.add(&sticks, &airborne, |p| (p.seed * 31.0) as usize % 10);

        Self {
            land,
            river: river_surface,
            stems: stem_batch.finish(),
            treats: treats.finish(),
            sprinkles: sprinkle_batch.finish(),
            icing: chip_icing(&mut rng),
            heads: Shape::icosphere(1.0, 2, 0.0, 0.0, Vec3::ONE),
            lollies,
            sparkle: Particles::new(160, 31),
            gloss: pipeline(GlossShader, CullingMode::None),
            syrup: pipeline(SyrupShader, CullingMode::None),
            head: SimdPipeline::new(HeadVertexShader, HeadShader)
                .with_culling_mode(CullingMode::BackFace)
                .with_depth_state(DepthState::DEFAULT),
            sky: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for CandyLand {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Glockenspiel],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::RisingPop),
                0.0,
                0.45,
                0.11,
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
        let glossy = |gloss, frosted| GlossUniforms {
            toon: toon.clone(),
            gloss,
            frosted,
        };
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let drift = Motion::Sway(Sway {
            amplitude: 0.55,
            speed: 0.6,
            wind: WIND,
        });

        self.land
            .draw(pass, &self.gloss, motion(Motion::Still), glossy(0.25, 1.0));
        self.river
            .draw(pass, &self.syrup, motion(Motion::Still), toon.clone());
        self.stems
            .draw(pass, &self.gloss, motion(Motion::Still), glossy(0.8, 0.0));
        self.treats
            .draw(pass, &self.gloss, motion(Motion::Still), glossy(0.9, 0.0));
        self.sprinkles
            .draw(pass, &self.gloss, motion(drift), glossy(0.9, 0.0));
        for lolly in &self.lollies {
            let (a, b) = PALETTES[lolly.palette];
            let radius = 1.5 * lolly.scale;
            let uniforms = HeadUniforms {
                view: view.clone(),
                lighting: lighting(),
                centre: lolly.foot + Vec3::new(0.0, 3.0 * lolly.scale + radius * 0.8, 0.0),
                radius,
                yaw: lolly.yaw,
                palette: (a, b),
                arms: lolly.arms,
                phase: lolly.phase,
            };
            self.heads
                .draw(pass, &self.head, uniforms.clone(), uniforms);
        }

        scene.chip.draw(pass, view, &look());
        self.icing
            .draw(pass, &self.gloss, motion(Motion::OnChip), glossy(0.9, 0.0));

        let sky = SkyUniforms {
            view: view.clone(),
            sky: sky(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| sky.clone());
        self.sparkle.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, -2.0),
                half_extent: Vec3::new(16.0, 4.5, 14.0),
                velocity: Vec3::new(0.3, 0.2, -0.1),
                wander: 0.6,
                size: (0.02, 0.05),
                colour: Vec3::new(1.9, 1.5, 1.7),
                twinkle: 2.2,
            },
        );
    }

    fn caption_ink(&self) -> Vec3 {
        Vec3::new(0.38, 0.09, 0.33)
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.4,
            vignette: 0.3,
            grain: 0.008,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
