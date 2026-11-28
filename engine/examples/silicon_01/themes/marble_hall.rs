//! marble-hall: a white gallery under tall windows. Fluted columns, a bust and a torso on plinths
//! beside the chip, a polished veined floor that mirrors the columns and windows, shafts of
//! light with dust turning in them, and the chip as a marble slab with a black inlay.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{
    F, clamp01, fract, length, mix3, mul3, pow_highlight, rgb, s, select3, smoothstep, splat3,
    to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonUniforms, pipeline,
};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::fbm3;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::shadow::GroundShadow;
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// The plane of the back wall.
const WALL: f32 = -18.0;
/// The gap between the back wall's windows.
const BAY: f32 = 7.0;
const WINDOW_HALF_WIDTH: f32 = 1.7;
const WINDOW_BOTTOM: f32 = 2.8;
const WINDOW_SPRING: f32 = 8.6;
const COLUMN_Z: f32 = -9.0;
const COLUMN_RADIUS: f32 = 0.62;
const COLUMN_HEIGHT: f32 = 12.5;
/// The columns' x positions, one between each pair of windows.
const COLUMNS: [f32; 6] = [-17.5, -10.5, -3.5, 3.5, 10.5, 17.5];
/// The floor's tiles are this wide.
const TILE: f32 = 2.2;
/// The direction towards the light that comes through the windows, from behind and to the left.
const SUN: Vec3 = Vec3 {
    x: -0.5,
    y: 0.62,
    z: -0.6,
};
/// The window light's colour, as it shows in the glass.
const GLASS: Vec3 = Vec3 {
    x: 2.1,
    y: 2.15,
    z: 2.3,
};

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.5, 1.35, 1.1),
        sky: Vec3::new(0.4, 0.43, 0.52),
        ground: Vec3::new(0.34, 0.31, 0.27),
        bands: 0.0,
        rim: Vec3::new(0.12, 0.11, 0.1),
        fog: Fog {
            colour: Vec3::new(0.5, 0.5, 0.52),
            density: 0.004,
            falloff: 0.0,
            base: FLOOR,
        },
    }
}

/// The chip as a slab of marble with a black inlay and brass contacts.
fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.78, 0.77, 0.73),
        die: Vec3::new(0.004, 0.005, 0.009),
        gold: Vec3::new(0.55, 0.4, 0.2),
        streams: Vec3::new(1.0, 0.82, 0.5),
        key: light.sun,
        key_colour: light.sun_colour * 0.7,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.7, 0.7, 0.72),
    }
}

/// Veined marble at `p`: how much of a vein is there (0 to 1) and a slow tone drift (0 to 1).
#[inline(always)]
fn marble(p: Vec3Simd) -> (F, F) {
    let warp = fbm3(p.x * s(0.45), p.y * s(0.45), p.z * s(0.45), 4);
    let line = |phase: F, width: f32| {
        let t = (fract(phase) - s(0.5)).abs() * s(2.0);
        s(1.0) - smoothstep(0.0, width, s(1.0) - t)
    };
    let main = line(
        p.x * s(0.17) + p.y * s(0.11) + p.z * s(0.07) + warp * s(2.6),
        0.035,
    );
    let fine = line(
        p.x * s(0.43) - p.z * s(0.31) + warp * s(5.5) + s(0.37),
        0.02,
    );
    let tone = fbm3(
        p.x * s(0.15) + s(5.0),
        p.y * s(0.15),
        p.z * s(0.15) + s(9.0),
        3,
    );
    ((main * s(0.9) + fine * s(0.35)).fast_min(s(1.0)), tone)
}

/// The colour of white marble with its veins, grey and faintly green.
#[inline(always)]
fn marble_colour(vein: F, tone: F) -> Vec3Simd {
    let stone = mix3(
        rgb(0.86, 0.85, 0.81),
        rgb(0.74, 0.76, 0.76),
        smoothstep(0.35, 0.7, tone),
    );
    mix3(stone, rgb(0.36, 0.38, 0.4), vein * s(0.8))
}

/// The back wall at (`x`, `y`): plaster between tall arched windows full of light.
#[inline(always)]
fn wall_colour(x: F, y: F) -> Vec3Simd {
    let across = (fract(x * s(1.0 / BAY) + s(0.5)) - s(0.5)) * s(BAY);
    let height = y - s(FLOOR);
    let arch = (s(WINDOW_HALF_WIDTH * WINDOW_HALF_WIDTH) - across * across)
        .fast_max(s(0.0))
        .sqrt();
    let inside = (s(1.0) - smoothstep(WINDOW_HALF_WIDTH - 0.05, WINDOW_HALF_WIDTH, across.abs()))
        * smoothstep(WINDOW_BOTTOM - 0.05, WINDOW_BOTTOM, height)
        * (s(1.0) - smoothstep(0.0, 0.08, height - s(WINDOW_SPRING) - arch));
    let bar = (s(1.0) - smoothstep(0.03, 0.07, across.abs()))
        .fast_max(s(1.0) - smoothstep(0.03, 0.07, (height - s(5.7)).abs()))
        .fast_max(s(1.0) - smoothstep(0.03, 0.07, (height - s(WINDOW_SPRING)).abs()));
    let glass = splat3(GLASS) * (s(1.0) - bar * s(0.85));
    let panel = smoothstep(0.1, 0.5, across.abs() - s(WINDOW_HALF_WIDTH + 0.5));
    let dado = smoothstep(1.4, 1.5, height);
    let plaster = mix3(
        rgb(0.3, 0.29, 0.27),
        rgb(0.5, 0.48, 0.45),
        dado * (s(0.7) + panel * s(0.3)),
    );
    mix3(plaster, glass, inside)
}

/// A fluted column: a base, a shaft that swells slightly and a capital.
fn column() -> Shape {
    let top = COLUMN_HEIGHT;
    let r = COLUMN_RADIUS;
    let profile = [
        (0.0, 0.0),
        (r * 1.5, 0.0),
        (r * 1.5, 0.3),
        (r * 1.25, 0.45),
        (r * 1.1, 0.6),
        (r * 0.95, 0.9),
        (r * 1.0, 3.0),
        (r * 0.92, top - 1.2),
        (r * 0.82, top - 0.9),
        (r * 1.15, top - 0.75),
        (r * 1.4, top - 0.45),
        (r * 1.55, top - 0.3),
        (r * 1.55, top),
        (0.0, top),
    ];
    Shape::lathe(&profile, 24, Vec3::ONE)
}

fn plinth(x: f32, z: f32, height: f32) -> Shape {
    let mut shape = Shape::cuboid(
        Vec3::new(x - 0.8, FLOOR, z - 0.8),
        Vec3::new(x + 0.8, FLOOR + 0.2, z + 0.8),
        Vec3::ONE,
    );
    shape.append(&Shape::cuboid(
        Vec3::new(x - 0.65, FLOOR + 0.2, z - 0.65),
        Vec3::new(x + 0.65, FLOOR + height - 0.18, z + 0.65),
        Vec3::ONE,
    ));
    shape.append(&Shape::cuboid(
        Vec3::new(x - 0.78, FLOOR + height - 0.18, z - 0.78),
        Vec3::new(x + 0.78, FLOOR + height, z + 0.78),
        Vec3::ONE,
    ));
    shape
}

fn ellipsoid(centre: Vec3, radii: Vec3) -> Shape {
    Shape::icosphere(1.0, 3, 0.0, 0.0, Vec3::ONE)
        .transformed(&(Mat4::translation_vec(centre) * Mat4::scaling(radii.x, radii.y, radii.z)))
}

/// A bust on a plinth at (`x`, `z`): shoulders, a neck, and a head with a brow and a nose.
fn bust(x: f32, z: f32, height: f32) -> Shape {
    let base = Vec3::new(x, FLOOR + height, z);
    let mut shape = plinth(x, z, height);
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 0.32, 0.0),
        Vec3::new(0.82, 0.42, 0.36),
    ));
    shape.append(&Shape::limb(
        base + Vec3::new(0.0, 0.5, 0.0),
        base + Vec3::new(0.0, 1.05, 0.02),
        0.22,
        0.19,
        12,
        Vec3::ONE,
    ));
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 1.42, 0.0),
        Vec3::new(0.3, 0.42, 0.34),
    ));
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 1.36, 0.3),
        Vec3::new(0.06, 0.14, 0.1),
    ));
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 1.56, 0.2),
        Vec3::new(0.26, 0.06, 0.14),
    ));
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 1.55, -0.07),
        Vec3::new(0.32, 0.3, 0.32),
    ));
    shape
}

/// A headless torso on a plinth: chest, waist, hips and the stumps of the arms.
fn torso(x: f32, z: f32, height: f32) -> Shape {
    let base = Vec3::new(x, FLOOR + height, z);
    let mut shape = plinth(x, z, height);
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 0.1, 0.0),
        Vec3::new(0.52, 0.42, 0.3),
    ));
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 0.62, 0.0),
        Vec3::new(0.5, 0.4, 0.3),
    ));
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 1.2, 0.0),
        Vec3::new(0.7, 0.48, 0.34),
    ));
    for side in [-1.0, 1.0] {
        shape.append(&Shape::limb(
            base + Vec3::new(side * 0.66, 1.25, 0.0),
            base + Vec3::new(side * 0.8, 0.55, 0.04),
            0.2,
            0.16,
            10,
            Vec3::ONE,
        ));
    }
    shape.append(&ellipsoid(
        base + Vec3::new(0.0, 1.62, 0.0),
        Vec3::new(0.2, 0.2, 0.2),
    ));
    shape
}

fn floor() -> Shape {
    terrain(
        &ring_radii(1.5, 70.0, 30),
        56,
        |_, _| FLOOR,
        |_, _| Vec3::ONE,
    )
}

/// The back wall, a quad behind the whole gallery.
fn wall() -> Shape {
    let (left, right, top) = (-70.0, 70.0, FLOOR + 22.0);
    let mut shape = Shape::new();
    shape.quad(
        [
            Vec3::new(left, FLOOR, WALL),
            Vec3::new(right, FLOOR, WALL),
            Vec3::new(right, top, WALL),
            Vec3::new(left, top, WALL),
        ],
        Vec3::ONE,
    );
    shape
}

/// Where the window light enters, as the centre of each shaft's mouth, and which way it falls.
fn shafts() -> Vec<Shape> {
    let direction = (-SUN).normalise();
    let centre_height = FLOOR + 0.5 * (WINDOW_BOTTOM + WINDOW_SPRING);
    [-BAY, 0.0, BAY]
        .iter()
        .map(|&x| {
            let mouth = Vec3::new(x, centre_height, WALL);
            let mut shape = Shape::new();
            let length = (centre_height - FLOOR) / direction.y.abs();
            for k in 0..10 {
                let at = mouth + direction * (length * (k as f32 + 0.5) / 10.0);
                let h = 3.2;
                shape.quad(
                    [
                        Vec3::new(at.x - h, at.y - h, at.z),
                        Vec3::new(at.x + h, at.y - h, at.z),
                        Vec3::new(at.x + h, at.y + h, at.z),
                        Vec3::new(at.x - h, at.y + h, at.z),
                    ],
                    Vec3::ONE,
                );
            }
            shape
        })
        .collect()
}

/// A column of window light drifting through haze: bright along its axis, soft towards its
/// edges, fading towards the floor, and mottled by the dust.
struct BeamShader;

#[derive(Clone)]
struct BeamUniforms {
    view: ViewRef,
    mouth: Vec3,
}

impl FragmentShaderSimd<KitVaryings> for BeamShader {
    type Uniforms = BeamUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &BeamUniforms) -> ColourSimd {
        let world = v.world();
        let axis = (-SUN).normalise();
        let from = world - splat3(u.mouth);
        let along = from.dot(splat3(axis));
        let off = from - splat3(axis) * along;
        let spread = length(off);
        let radius = s(WINDOW_HALF_WIDTH * 0.9) + along * s(0.04);
        let core = (-(spread / radius) * (spread / radius) * s(1.6)).exp();
        let reach = (s(1.0) - smoothstep(6.0, 14.0, along)) * smoothstep(0.0, 2.0, along);
        let t = u.view.frame.time;
        let drift = fbm3(
            world.x * s(0.35) + s(t * 0.05),
            world.y * s(0.35) - s(t * 0.08),
            world.z * s(0.35),
            3,
        );
        let mottle = s(0.55) + drift * s(0.9);
        let ground = smoothstep(0.0, 0.9, world.y - s(FLOOR));
        to_colour(
            rgb(1.0, 0.9, 0.72) * (core * reach * mottle * ground * s(0.12)),
            s(0.0),
        )
    }
}

/// Marble stone: veins, a soft wrap of light, a polished sheen.
struct StoneShader;

impl FragmentShaderSimd<KitVaryings> for StoneShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let view_dir = to_eye * (s(1.0) / distance);
        let normal = facing(v.normal().normalise(), to_eye);
        let (vein, tone) = marble(world);
        let wrap = (normal.dot(splat3(u.lighting.sun)) + s(0.35)) * s(1.0 / 1.35);
        let albedo = marble_colour(vein, tone);
        let shadowless = Lighting {
            sun_colour: Vec3::ZERO,
            ..u.lighting
        };
        let ambient = shadowless.shade(view, world, normal, albedo, s(1.0));
        let direct = mul3(albedo, splat3(u.lighting.sun_colour)) * (wrap.fast_max(s(0.0)) * s(0.8));
        let occlusion = s(0.5) + s(0.5) * smoothstep(0.0, 1.4, world.y - s(FLOOR));
        let half = (splat3(u.lighting.sun) + view_dir).normalise();
        let polish = pow_highlight(normal.dot(half).fast_max(s(0.0)), 64.0) * s(0.35);
        let colour = (ambient + direct) * occlusion + splat3(u.lighting.sun_colour) * polish;
        scene_colour(view, colour, world)
    }
}

/// The back wall, plaster and windows; the windows are lit, not shaded.
struct WallShader;

impl FragmentShaderSimd<KitVaryings> for WallShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let world = v.world();
        let distance = length(splat3(u.view.eye) - world);
        let colour = u
            .lighting
            .fog
            .apply(wall_colour(world.x, world.y), world, distance);
        scene_colour(&u.view, colour, world)
    }
}

#[derive(Clone)]
struct FloorUniforms {
    toon: ToonUniforms,
    shadow: Arc<GroundShadow>,
}

/// The polished floor: veined tiles of two stones, shadowed by the columns and the chip, mirroring
/// the columns and the back wall's windows with more of the reflection at grazing angles.
struct FloorShader;

impl FragmentShaderSimd<KitVaryings> for FloorShader {
    type Uniforms = FloorUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &FloorUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let lighting = &u.toon.lighting;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let view_dir = to_eye * (s(1.0) / distance);

        let (tx, tz) = (
            (world.x * s(1.0 / TILE)).floor(),
            (world.z * s(1.0 / TILE)).floor(),
        );
        let dark = (tx + tz - ((tx + tz) * s(0.5)).floor() * s(2.0)).abs();
        let (vein, tone) = marble(world * s(0.9));
        let light_stone = marble_colour(vein, tone);
        let dark_stone = mix3(rgb(0.1, 0.12, 0.12), rgb(0.2, 0.22, 0.22), tone)
            + rgb(0.5, 0.5, 0.5) * (vein * s(0.25));
        let (lx, lz) = (
            (fract(world.x * s(1.0 / TILE)) - s(0.5)).abs(),
            (fract(world.z * s(1.0 / TILE)) - s(0.5)).abs(),
        );
        let joint =
            smoothstep(0.492, 0.5, lx.fast_max(lz)) * (s(1.0) - smoothstep(30.0, 60.0, distance));
        let albedo = mix3(light_stone, dark_stone, dark) * (s(1.0) - joint * s(0.5));

        let lit = u.shadow.lit(world.x, world.z);
        let blocked = chip_shadow(view, lighting.sun, world);
        let shadow = lit * (s(1.0) - blocked * s(0.8));
        let up = Vec3Simd::new(s(0.0), s(1.0), s(0.0));
        let base = lighting.shade(view, world, up, albedo, shadow);

        let reflected = Vec3Simd::new(-view_dir.x, view_dir.y, -view_dir.z);
        let to_wall = (s(WALL) - world.z) / reflected.z.fast_min(s(-1e-3));
        let wall = wall_colour(
            world.x + reflected.x * to_wall,
            world.y + reflected.y * to_wall,
        );
        let mut mirror = wall * s(0.9);
        let mut nearest = to_wall;
        for &cx in &COLUMNS {
            let (ox, oz) = (world.x - s(cx), world.z - s(COLUMN_Z));
            let (dx, dz) = (reflected.x, reflected.z);
            let a = (dx * dx + dz * dz).fast_max(s(1e-6));
            let b = ox * dx + oz * dz;
            let c = ox * ox + oz * oz - s(COLUMN_RADIUS * COLUMN_RADIUS);
            let disc = b * b - a * c;
            let t = (-b - disc.fast_max(s(0.0)).sqrt()) / a;
            let hit_y = world.y + reflected.y * t;
            let hit = disc.simd_gt(s(0.0))
                & t.simd_gt(s(0.0))
                & t.simd_lt(nearest)
                & hit_y.simd_lt(s(FLOOR + COLUMN_HEIGHT));
            let (nx, nz) = (
                (ox + dx * t) * s(1.0 / COLUMN_RADIUS),
                (oz + dz * t) * s(1.0 / COLUMN_RADIUS),
            );
            let facing_light =
                (nx * s(SUN.normalise().x) + nz * s(SUN.normalise().z)).fast_max(s(0.0));
            let stone = rgb(0.62, 0.61, 0.57) * (s(0.35) + facing_light * s(0.7));
            mirror = select3(hit, stone, mirror);
            nearest = hit.select(t, nearest);
        }
        let grazing = s(1.0) - clamp01(view_dir.y * s(1.6));
        let strength = (s(0.08) + grazing * grazing * grazing * s(0.5))
            * (s(1.0) - dark * s(0.35)).fast_max(s(0.0))
            + dark * s(0.12);
        let sheen = pow_highlight(clamp01(reflected.dot(splat3(lighting.sun))), 256.0) * s(0.6);
        let colour = mix3(base, mirror, strength.fast_min(s(0.55)))
            + splat3(lighting.sun_colour) * (sheen * shadow);
        let hazy = lighting.fog.apply(colour, world, distance);
        scene_colour(view, hazy, world)
    }
}

pub struct MarbleHall {
    floor: Shape,
    wall: Shape,
    columns: Shape,
    statues: Shape,
    shafts: Vec<Shape>,
    shadow: Arc<GroundShadow>,
    dust: Particles,
    stone: SimdPipeline<KitVertexShader, StoneShader>,
    wall_pipeline: SimdPipeline<KitVertexShader, WallShader>,
    floor_pipeline: SimdPipeline<KitVertexShader, FloorShader>,
    beam: SimdPipeline<KitVertexShader, BeamShader>,
}

impl MarbleHall {
    pub fn new() -> Self {
        let mut columns = Shape::new();
        for &x in &COLUMNS {
            columns.append(&column().transformed(&Mat4::translation(x, FLOOR, COLUMN_Z)));
        }
        let mut statues = bust(-4.1, -1.3, 1.9);
        statues.append(&torso(4.3, -2.2, 1.7));
        let shadow = GroundShadow::bake(
            Vec2::new(-35.0, -35.0),
            70.0,
            700,
            SUN.normalise(),
            |_, _| FLOOR,
            &[&columns, &statues],
            2,
        );
        Self {
            floor: floor(),
            wall: wall(),
            columns,
            statues,
            shafts: shafts(),
            shadow: Arc::new(shadow),
            dust: Particles::new(120, 77),
            stone: pipeline(StoneShader, CullingMode::BackFace),
            wall_pipeline: pipeline(WallShader, CullingMode::None),
            floor_pipeline: pipeline(FloorShader, CullingMode::None),
            beam: SimdPipeline::new(KitVertexShader, BeamShader)
                .with_culling_mode(CullingMode::None)
                .with_blend_state(BlendState::ADDITIVE)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for MarbleHall {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Piano],
            treatment: Treatment::Vast,
            ambience: vec![Ambience::once(
                Source::Synth(Accent::StoneChime),
                0.0,
                0.45,
                0.1,
                -0.3,
            )],
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let still = || KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };

        self.floor.draw(
            pass,
            &self.floor_pipeline,
            still(),
            FloorUniforms {
                toon: toon.clone(),
                shadow: self.shadow.clone(),
            },
        );
        self.wall
            .draw(pass, &self.wall_pipeline, still(), toon.clone());
        self.columns.draw(pass, &self.stone, still(), toon.clone());
        self.statues.draw(pass, &self.stone, still(), toon);
        scene.chip.draw(pass, view, &look());

        let centre_height = FLOOR + 0.5 * (WINDOW_BOTTOM + WINDOW_SPRING);
        for (shaft, x) in self.shafts.iter().zip([-BAY, 0.0, BAY]) {
            let uniforms = BeamUniforms {
                view: view.clone(),
                mouth: Vec3::new(x, centre_height, WALL),
            };
            shaft.draw(pass, &self.beam, still(), uniforms);
        }
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, -7.0),
                half_extent: Vec3::new(16.0, 6.0, 9.0),
                velocity: Vec3::new(0.06, -0.1, 0.05),
                wander: 0.5,
                size: (0.012, 0.035),
                colour: Vec3::new(1.5, 1.3, 1.0),
                twinkle: 1.3,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.45,
            vignette: 0.3,
            grain: 0.008,
            aberration: 0.0,
            desaturate: 0.12,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_column_is_as_tall_as_the_hall_says() {
        let top = column()
            .vertices()
            .iter()
            .fold(f32::MIN, |top, v| top.max(v.position.y));
        assert!((top - COLUMN_HEIGHT).abs() < 1e-4);
    }

    #[test]
    fn every_column_stands_between_two_windows() {
        for x in COLUMNS {
            let across = (x / BAY + 0.5).rem_euclid(1.0) - 0.5;
            assert!(
                across.abs() * BAY > WINDOW_HALF_WIDTH + COLUMN_RADIUS,
                "column at {x}"
            );
        }
    }
}
