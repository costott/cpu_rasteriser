//! A candlelit baroque hall: the chip as a gilded jewel box on a black marble floor inlaid with
//! gold, a colonnade receding into the dark, floating gold orbs and a chandelier of twelve candles
//! whose light the marble mirrors.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::chip::{HALF, Look};
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, View, ViewRef, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, hash2, length, mix3, mul3, reflect, rgb, s, smoothstep, splat3, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Orbit, Sway, pipeline,
};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::light::facing;
use crate::themes::kit::noise::fbm;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::{Scene, Stage};

/// The height of the marble floor: the chip's lowest corner rests on it.
const FLOOR: f32 = -1.95;
/// The middle of the chandelier's candle ring.
const LAMP: Vec3 = Vec3 {
    x: 0.0,
    y: 5.0,
    z: -10.0,
};
/// A second, dimmer pool of candlelight at the front of the hall, which keeps the chip's face lit.
const FILL: Vec3 = Vec3 {
    x: 2.0,
    y: 4.0,
    z: 14.0,
};
const CANDLES: usize = 12;
const CANDLE_RING: f32 = 1.8;
/// A column's x, z and radius, and the height all share above the floor.
const COLUMNS: [(f32, f32, f32); 8] = [
    (-7.5, 6.0, 0.85),
    (7.5, 6.0, 0.85),
    (-7.5, -7.0, 0.85),
    (7.5, -7.0, 0.85),
    (-7.5, -20.0, 0.85),
    (7.5, -20.0, 0.85),
    (-7.5, -33.0, 0.85),
    (7.5, -33.0, 0.85),
];
const COLUMN_HEIGHT: f32 = 19.0;
/// The slab size of the floor, in world units.
const SLAB: f32 = 3.0;
const HAZE: Vec3 = Vec3 {
    x: 0.034,
    y: 0.015,
    z: 0.009,
};
const GOLD: Vec3 = Vec3 {
    x: 0.95,
    y: 0.62,
    z: 0.2,
};

fn candle_tips() -> [Vec3; CANDLES] {
    std::array::from_fn(|k| {
        let angle = k as f32 / CANDLES as f32 * TAU;
        LAMP + Vec3::new(CANDLE_RING * angle.cos(), 0.95, CANDLE_RING * angle.sin())
    })
}

/// The chandelier's flicker at `time`: a slow breathing with quick wavers, around 1.
#[inline(always)]
fn flicker(time: f32, phase: f32) -> f32 {
    1.0 + 0.05 * (time * 5.3 + phase).sin() + 0.03 * (time * 11.7 + phase * 2.1).sin()
}

/// How brightly candlelight lights a surface at `world` facing `normal`: the chandelier and the
/// dimmer fill, each a diffuse term with inverse-square falloff.
#[inline(always)]
fn lamp_light(world: Vec3Simd, normal: Vec3Simd, flick: f32) -> F {
    let mut total = s(0.0);
    for (position, power) in [(LAMP, 190.0), (FILL, 70.0)] {
        let to_lamp = splat3(position) - world;
        let distance = length(to_lamp);
        let diffuse = normal.dot(to_lamp * (s(1.0) / distance)).fast_max(s(0.0));
        total += diffuse * s(flick * power) / (distance * distance + s(60.0));
    }
    total
}

/// The hall's haze, which the backdrop and the fog fade into: a warm glow low in the dark.
#[inline(always)]
fn hall(ray: Vec3Simd) -> Vec3Simd {
    let up = ray.y.fast_max(s(0.0));
    let glow = (-ray.x * ray.x * s(3.0)).exp() * (-up * s(5.0)).exp();
    let base = mix3(
        splat3(HAZE),
        rgb(0.006, 0.004, 0.006),
        smoothstep(0.0, 0.6, up).sqrt(),
    );
    base + rgb(0.12, 0.055, 0.02) * glow
}

/// What a ray leaving the marble meets in the colonnade: the gilded column it strikes first, if
/// any (the mask is 1 where it does), shaded by the chandelier.
#[inline(always)]
fn column_reflection(origin: Vec3Simd, dir: Vec3Simd, flick: f32) -> (F, Vec3Simd) {
    let mut nearest = s(1.0e9);
    let mut normal = rgb(0.0, 0.0, 1.0);
    let mut height = s(0.0);
    let a = dir.x * dir.x + dir.z * dir.z + s(1.0e-6);
    for &(cx, cz, radius) in &COLUMNS {
        let ox = origin.x - s(cx);
        let oz = origin.z - s(cz);
        let b = ox * dir.x + oz * dir.z;
        let c = ox * ox + oz * oz - s(radius * radius);
        let disc = b * b - a * c;
        let t = (-b - disc.fast_max(s(0.0)).sqrt()) / a;
        let y = origin.y + dir.y * t;
        let hit = disc.simd_gt(s(0.0))
            & t.simd_gt(s(0.0))
            & t.simd_lt(nearest)
            & y.simd_lt(s(FLOOR + COLUMN_HEIGHT));
        nearest = hit.select(t, nearest);
        normal = Vec3Simd::new(
            hit.select((ox + dir.x * t) / s(radius), normal.x),
            s(0.0),
            hit.select((oz + dir.z * t) / s(radius), normal.z),
        );
        height = hit.select(y, height);
    }
    let found = nearest.simd_lt(s(1.0e8));
    let point = origin + dir * nearest;
    let ring = smoothstep(0.0, 0.5, (height - s(FLOOR)) * s(0.5)).fast_min(s(1.0));
    let lit = lamp_light(point, normal, flick).fast_min(s(1.5));
    let gilt =
        (height - s(FLOOR)).simd_lt(s(1.3)) | (height - s(FLOOR)).simd_gt(s(COLUMN_HEIGHT - 1.8));
    let albedo = gilt.select(s(1.0), s(0.1));
    let gold = rgb(GOLD.x, GOLD.y, GOLD.z)
        * ((lit * s(1.4) + s(0.03)) * (s(0.6) + ring * s(0.5)) * albedo);
    let fade = (-nearest * s(0.018)).exp();
    (found.select(s(1.0), s(0.0)), gold * fade)
}

/// The soft glints of the candle flames in a ray leaving the marble; `blur` widens them as the
/// polish clouds.
#[inline(always)]
fn candle_reflection(origin: Vec3Simd, dir: Vec3Simd, blur: F, flick: f32) -> Vec3Simd {
    let mut glow = s(0.0);
    for tip in candle_tips() {
        let to = splat3(tip) - origin;
        let along = to.dot(dir);
        let perp2 = (to.dot(to) - along * along).fast_max(s(0.0));
        let angular = perp2 / (along * along + s(1.0));
        let visible = along.simd_gt(s(0.0)).select(s(1.0), s(0.0));
        glow += visible / (angular * s(260.0) + blur * s(40.0) + s(0.08));
    }
    rgb(1.0, 0.62, 0.26) * (glow * s(0.03 * flick))
}

struct FloorUniforms {
    view: ViewRef,
}

/// Black marble: gold inlay between slabs, veins, the chip's shadow and a polished reflection of
/// the colonnade and the candle flames.
struct MarbleShader;

impl FragmentShaderSimd<KitVaryings> for MarbleShader {
    type Uniforms = FloorUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &FloorUniforms) -> ColourSimd {
        let view: &View = &u.view;
        let world = v.world();
        let time = view.frame.time;
        let flick = flicker(time, 0.0);
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let eye_dir = to_eye * (s(1.0) / distance);

        let sx = world.x / s(SLAB) + s(0.5);
        let sz = world.z / s(SLAB) + s(0.5);
        let (cell_x, cell_z) = (sx.floor(), sz.floor());
        let (fx, fz) = (sx - cell_x, sz - cell_z);
        let edge = (fx - s(0.5)).abs().fast_max((fz - s(0.5)).abs());
        let inlay_width = s(0.005) + distance * s(0.00035);
        let inlay = smoothstep(
            0.0,
            1.0,
            clamp01((edge - s(0.5) + inlay_width * s(2.0)) / inlay_width),
        );
        let inner = smoothstep(
            0.0,
            1.0,
            clamp01((edge - s(0.42) + inlay_width) / inlay_width),
        ) * (s(1.0) - smoothstep(0.0, 1.0, clamp01((edge - s(0.42)) / inlay_width)))
            * s(0.8);
        let slab = hash2(cell_x, cell_z);

        let warp = fbm(
            sx * s(0.9) + fbm(sz * s(0.8), sx * s(0.7), 2) * s(1.6),
            sz * s(0.9),
            3,
        );
        let ridge = s(1.0) - (warp * s(2.0) - s(1.0)).abs();
        let vein = smoothstep(0.93, 0.995, ridge);
        let fine = smoothstep(
            0.95,
            0.995,
            s(1.0) - (fbm(sx * s(3.4), sz * s(3.4) + slab * s(9.0), 3) * s(2.0) - s(1.0)).abs(),
        );
        let distant = (-distance * s(0.03)).exp();
        let mut base = rgb(0.03, 0.028, 0.034) * (s(0.6) + slab * s(0.8));
        base = base + rgb(0.5, 0.42, 0.3) * ((vein * s(0.5) + fine * s(0.18)) * distant);

        let rough = (fbm(world.x * s(2.2), world.z * s(0.35), 2) - s(0.5)) * s(0.05);
        let normal = Vec3Simd::new(rough, s(1.0), rough * s(0.7)).normalise();
        let facing_eye = clamp01(normal.dot(eye_dir));
        let grazing = s(1.0) - facing_eye;
        let fresnel = s(0.1) + s(0.9) * (grazing * grazing * grazing * grazing * grazing);

        let metal = clamp01(inlay + inner);
        let albedo = mix3(base, rgb(GOLD.x, GOLD.y, GOLD.z) * s(0.35), metal);
        let shadow = s(1.0) - chip_shadow(view, Vec3::new(0.0, 1.0, 0.0), world) * s(0.7);
        let lit = lamp_light(world, normal, flick) * shadow;
        let diffuse = albedo * (lit + s(0.015));

        let ray = reflect(-eye_dir, normal);
        let blur = (s(1.0) - facing_eye) * s(0.8) + (s(1.0) - fresnel);
        let (found, columns) = column_reflection(world, ray, flick);
        let environment = mix3(hall(ray), columns, found);
        let mirrored = environment + candle_reflection(world, ray, blur * s(0.3), flick);
        let reflectance = mix3(
            splat3(Vec3::ONE) * (fresnel * s(0.85)),
            rgb(GOLD.x, GOLD.y, GOLD.z) * s(0.5),
            metal,
        );
        let colour = diffuse + mul3(mirrored, reflectance) * shadow.fast_max(s(0.25));
        let mist = s(1.0) - (-distance * s(0.016)).exp();
        let colour = mix3(colour, hall(Vec3Simd::new(ray.x, s(0.0), ray.z)), mist);
        scene_colour(view, colour, world)
    }
}

struct GildedUniforms {
    view: ViewRef,
    /// 1 for polished gold, 0 for a matt surface, such as candle wax.
    metal: f32,
}

/// Gilded ornament and wax, lit by the chandelier: polished gold reflects its warm room, with a
/// hard highlight where the lamp mirrors in a facet.
struct GildedShader;

impl FragmentShaderSimd<KitVaryings> for GildedShader {
    type Uniforms = GildedUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GildedUniforms) -> ColourSimd {
        let view: &View = &u.view;
        let world = v.world();
        let flick = flicker(view.frame.time, 0.0);
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let eye_dir = to_eye * (s(1.0) / distance);
        let normal = facing(v.normal().normalise(), to_eye);

        let to_lamp = splat3(LAMP) - world;
        let lamp_distance = length(to_lamp);
        let lamp_dir = to_lamp * (s(1.0) / lamp_distance);
        let direct = lamp_light(world, normal, flick);

        let mirror = reflect(-eye_dir, normal);
        let toward_lamp = mirror.dot(lamp_dir).fast_max(s(0.0));
        let toward_lamp2 = toward_lamp * toward_lamp;
        let highlight = toward_lamp2 * toward_lamp2 * toward_lamp2 * toward_lamp2;
        let bounce = smoothstep(-0.4, 0.9, mirror.y);
        let room = rgb(0.03, 0.02, 0.015)
            + rgb(0.38, 0.22, 0.09) * bounce
            + rgb(0.5, 0.3, 0.12) * (toward_lamp2 * s(0.8));

        let albedo = v.colour();
        let metal = s(u.metal);
        let metallic = mul3(
            albedo,
            room * s(1.5) + splat3(Vec3::ONE) * (direct * s(0.5)),
        ) + rgb(1.0, 0.82, 0.5) * (highlight * s(1.8 * flick));
        let matt = mul3(
            albedo,
            splat3(Vec3::ONE * 0.012) + splat3(Vec3::ONE) * (direct * s(0.9)),
        );
        let colour = mix3(matt, metallic, metal);
        let mist = s(1.0) - (-distance * s(0.016)).exp();
        scene_colour(view, mix3(colour, hall(-eye_dir), mist), world)
    }
}

/// Candle flames and anything else that glows by itself, flickering with the instance.
struct FlameShader;

impl FragmentShaderSimd<KitVaryings> for FlameShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let time = s(view.frame.time);
        let phase = v.seed() * s(40.0);
        let flick = s(1.0)
            + (time * s(9.0) + phase).sin() * s(0.12)
            + (time * s(17.3) + phase * s(1.7)).sin() * s(0.07);
        scene_colour(view, v.colour() * flick, v.world())
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
}

/// The far hall: warm haze, pilasters in the dark, and the bokeh of countless distant candles.
struct BackdropShader;

impl FragmentShaderSimd<ScreenVaryings> for BackdropShader {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let [x, y] = v.ndc();
        let ray = view_ray(&u.view, x, y);
        let time = s(u.view.frame.time);
        let mut colour = hall(ray);

        let toward = (-ray.z).fast_max(s(0.2));
        let (px, py) = (ray.x / toward, ray.y / toward);
        let pilaster = smoothstep(
            0.0,
            1.0,
            clamp01((((px * s(0.16)).sin() * s(0.5) + s(0.5)) - s(0.93)) * s(14.0)),
        );
        let band = smoothstep(0.0, 0.35, py) * (s(1.0) - smoothstep(0.35, 0.7, py));
        colour = colour + rgb(0.1, 0.055, 0.02) * (pilaster * band);

        for (scale, strength, offset) in [(7.0f32, 1.0f32, 0.0f32), (13.0, 0.55, 5.3)] {
            let (gx, gy) = (px * s(scale) + s(offset), py * s(scale));
            let (cell_x, cell_y) = (gx.floor(), gy.floor());
            let h = hash2(cell_x, cell_y + s(offset));
            let pick = smoothstep(0.55, 0.62, h);
            let centre_x = cell_x + s(0.3) + hash2(cell_x + s(3.0), cell_y) * s(0.4);
            let centre_y = cell_y + s(0.3) + hash2(cell_x, cell_y + s(7.0)) * s(0.4);
            let (dx, dy) = (gx - centre_x, gy - centre_y);
            let disc = s(1.0) - smoothstep(0.0, 0.17, (dx * dx + dy * dy).sqrt());
            let flick = s(0.75) + s(0.25) * (time * s(2.5) + h * s(80.0)).sin();
            let low = smoothstep(-0.05, 0.1, py) * (s(1.0) - smoothstep(0.35, 0.8, py));
            colour = colour
                + rgb(1.0, 0.58, 0.22) * (disc * disc * pick * flick * low * s(strength * 0.8));
        }
        to_colour(colour, s(SKY_DISTANCE))
    }
}

/// A volute: an Archimedean spiral in the xy plane that starts at (`radius`, 0), winds `turns`
/// times anticlockwise and tapers to a pearl at its centre.
fn volute(radius: f32, turns: f32, thickness: f32) -> Shape {
    let steps = (turns * 10.0).ceil() as usize;
    let at = |k: usize| {
        let t = k as f32 / steps as f32;
        let angle = t * turns * TAU;
        let r = radius * (1.0 - 0.9 * t);
        (
            Vec3::new(r * angle.cos(), r * angle.sin(), 0.0),
            thickness * (1.0 - 0.65 * t),
        )
    };
    let mut shape = Shape::new();
    for k in 0..steps {
        let ((from, r0), (to, r1)) = (at(k), at(k + 1));
        shape.append(&Shape::limb(from, to, r0, r1, 5, GOLD));
    }
    let (end, r) = at(steps);
    shape.append(
        &Shape::icosphere(r * 1.8, 1, 0.0, 1.0, GOLD).transformed(&Mat4::translation_vec(end)),
    );
    shape
}

/// A fluted column standing on y = 0: plinth, base mouldings, a tapering shaft with a neck ring,
/// a flared capital and an abacus.
fn column(radius: f32, height: f32) -> Shape {
    let r = radius;
    let profile = [
        (0.0, 0.0),
        (r * 1.45, 0.0),
        (r * 1.45, 0.45),
        (r * 1.2, 0.5),
        (r * 1.3, 0.75),
        (r * 1.05, 0.95),
        (r * 1.0, 1.15),
        (r * 0.9, height - 1.7),
        (r * 1.05, height - 1.5),
        (r * 1.05, height - 1.35),
        (r * 1.4, height - 0.9),
        (r * 1.7, height - 0.45),
        (r * 1.75, height - 0.3),
        (r * 1.75, height),
        (0.0, height),
    ];
    Shape::lathe(&profile, 14, GOLD).colour_by(|p, _| {
        if p.y < 1.3 || p.y > height - 1.8 {
            GOLD
        } else {
            Vec3::new(0.1, 0.095, 0.1)
        }
    })
}

fn colonnade() -> Shape {
    let mut shape = Shape::new();
    let post = column(1.0, COLUMN_HEIGHT);
    for &(x, z, radius) in &COLUMNS {
        shape.append(
            &post.transformed(
                &(Mat4::translation(x, FLOOR, z) * Mat4::scaling(radius, 1.0, radius)),
            ),
        );
    }
    shape
}

/// The chandelier and its candles. Returns (gold, wax, flames).
fn chandelier(rng: &mut Rng) -> (Shape, Shape, Shape) {
    let mut gold = Shape::new();
    let mut wax = Shape::new();
    let mut flames = Shape::new();
    let add_limb = |shape: &mut Shape, from: Vec3, to: Vec3, radius: f32| {
        shape.append(&Shape::limb(from, to, radius, radius, 5, GOLD));
    };
    add_limb(
        &mut gold,
        Vec3::new(0.0, 1.8, 0.0),
        Vec3::new(0.0, 14.0, 0.0),
        0.05,
    );
    let body = [
        (0.0, -1.7),
        (0.18, -1.5),
        (0.5, -1.0),
        (0.3, -0.5),
        (0.65, -0.2),
        (0.35, 0.3),
        (0.2, 1.0),
        (0.0, 1.8),
    ];
    gold.append(&Shape::lathe(&body, 10, GOLD));
    let ring: Vec<Vec3> = (0..=24)
        .map(|k| {
            let angle = k as f32 / 24.0 * TAU;
            Vec3::new(CANDLE_RING * angle.cos(), 0.0, CANDLE_RING * angle.sin())
        })
        .collect();
    for pair in ring.windows(2) {
        add_limb(&mut gold, pair[0], pair[1], 0.07);
    }
    for k in 0..CANDLES {
        let angle = k as f32 / CANDLES as f32 * TAU;
        let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
        let at = direction * CANDLE_RING;
        let arm = [
            Vec3::new(0.0, -0.2, 0.0) + direction * 0.55,
            direction * 1.0 + Vec3::new(0.0, -0.75, 0.0),
            at + Vec3::new(0.0, -0.1, 0.0),
        ];
        for pair in arm.windows(2) {
            add_limb(&mut gold, pair[0], pair[1], 0.055);
        }
        gold.append(
            &Shape::lathe(
                &[
                    (0.0, 0.0),
                    (0.12, 0.0),
                    (0.09, 0.12),
                    (0.2, 0.2),
                    (0.13, 0.2),
                    (0.0, 0.2),
                ],
                8,
                GOLD,
            )
            .transformed(&Mat4::translation_vec(at)),
        );
        let height = rng.range(0.55, 0.75);
        wax.append(
            &Shape::cylinder(0.08, 0.075, height, 8, Vec3::new(0.95, 0.88, 0.7))
                .transformed(&Mat4::translation_vec(at + Vec3::new(0.0, 0.2, 0.0))),
        );
        let top = at + Vec3::new(0.0, 0.2 + height, 0.0);
        flames.append(
            &Shape::lathe(
                &[(0.0, 0.0), (0.07, 0.04), (0.055, 0.14), (0.0, 0.3)],
                7,
                Vec3::new(5.0, 2.7, 0.9),
            )
            .transformed(&Mat4::translation_vec(top))
            .sway_by_height(0.0, 0.3, 1.0)
            .with_seed(k as f32 / CANDLES as f32),
        );
        let drop_from = direction * (CANDLE_RING * 0.82) + Vec3::new(0.0, -0.9, 0.0);
        add_limb(&mut gold, direction * (CANDLE_RING * 0.82), drop_from, 0.02);
        gold.append(
            &Shape::icosphere(0.1, 1, 0.0, 2.0, Vec3::new(1.0, 0.82, 0.5))
                .transformed(&Mat4::translation_vec(drop_from)),
        );
    }
    let lift = Mat4::translation_vec(LAMP);
    (
        gold.transformed(&lift),
        wax.transformed(&lift),
        flames.transformed(&lift),
    )
}

/// Gold orbs, each its own instance so the orbit motion carries it along a path of its own.
fn orbs() -> Shape {
    let mut shape = Shape::new();
    let count = 8;
    for k in 0..count {
        let size = 0.18 + 0.07 * ((k * 5) % 4) as f32;
        shape.append(
            &Shape::icosphere(size, 3, 0.0, k as f32, GOLD)
                .with_seed((k as f32 + 0.37) / count as f32),
        );
    }
    shape
}

/// Gilt for the chip, in its own space: a bezel along the package's edge with pearls, bosses at
/// the corners and a cartouche at the middle of each side.
fn chip_gilding() -> Shape {
    let mut shape = Shape::new();
    let (outer, width, height) = (HALF - 0.05, 0.14, 0.1);
    let inner = outer - width;
    for side in [-1.0f32, 1.0] {
        let edge = side * outer;
        let edge_in = side * inner;
        let (z0, z1) = (edge.min(edge_in), edge.max(edge_in));
        shape.append(&Shape::cuboid(
            Vec3::new(-outer, 0.0, z0),
            Vec3::new(outer, height, z1),
            GOLD,
        ));
        shape.append(&Shape::cuboid(
            Vec3::new(z0, 0.0, -inner),
            Vec3::new(z1, height, inner),
            GOLD,
        ));
        for k in -4..=4 {
            let along = k as f32 * 0.4;
            let pearl = Shape::icosphere(0.065, 1, 0.0, 1.0, Vec3::new(1.0, 0.9, 0.7));
            shape.append(&pearl.transformed(&Mat4::translation(
                along,
                height + 0.03,
                side * (outer - width * 0.5),
            )));
            shape.append(&pearl.transformed(&Mat4::translation(
                side * (outer - width * 0.5),
                height + 0.03,
                along,
            )));
        }
        for end in [-1.0f32, 1.0] {
            shape.append(&Shape::icosphere(0.2, 1, 0.0, 3.0, GOLD).transformed(
                &(Mat4::translation(side * outer, height, end * outer)
                    * Mat4::scaling(1.0, 0.8, 1.0)),
            ));
        }
        shape.append(&Shape::icosphere(0.26, 1, 0.0, 4.0, GOLD).transformed(
            &(Mat4::translation(0.0, height, side * outer) * Mat4::scaling(2.4, 0.7, 0.9)),
        ));
        shape.append(&Shape::icosphere(0.26, 1, 0.0, 5.0, GOLD).transformed(
            &(Mat4::translation(side * outer, height, 0.0) * Mat4::scaling(0.9, 0.7, 2.4)),
        ));
    }
    shape
}

/// Scrollwork over the lintel of the hall's far end: a pair of mirrored volutes either side of a
/// cartouche.
fn crest() -> Shape {
    let mut shape = Shape::new();
    for side in [-1.0f32, 1.0] {
        shape.append(&volute(2.2, 2.0, 0.3).transformed(
            &(Mat4::translation(side * 2.1, 0.0, 0.0) * Mat4::scaling(-side, 1.0, 1.0)),
        ));
        shape.append(&volute(1.3, 1.6, 0.22).transformed(
            &(Mat4::translation(side * 4.6, 0.6, 0.0)
                * Mat4::scaling(side, 1.0, 1.0)
                * Mat4::rotate_z(0.8)),
        ));
    }
    shape.append(
        &Shape::icosphere(1.0, 2, 0.0, 6.0, GOLD).transformed(&Mat4::scaling(1.1, 1.5, 0.35)),
    );
    shape
}

pub struct BaroqueGold {
    floor: Shape,
    columns: Shape,
    crest: Shape,
    chandelier_gold: Shape,
    chandelier_wax: Shape,
    flames: Shape,
    orbs: Shape,
    gilding: Shape,
    dust: Particles,
    sparks: Particles,
    marble: SimdPipeline<KitVertexShader, MarbleShader>,
    gilded: SimdPipeline<KitVertexShader, GildedShader>,
    flame: SimdPipeline<KitVertexShader, FlameShader>,
    backdrop: SimdPipeline<ScreenVertexShader, BackdropShader>,
}

impl BaroqueGold {
    pub fn new() -> Self {
        let mut rng = Rng::new(1618);
        let mut floor = Shape::new();
        let far = 400.0;
        floor.quad(
            [
                Vec3::new(-far, FLOOR, far),
                Vec3::new(far, FLOOR, far),
                Vec3::new(far, FLOOR, -far),
                Vec3::new(-far, FLOOR, -far),
            ],
            Vec3::ONE * 0.02,
        );
        let (chandelier_gold, chandelier_wax, flames) = chandelier(&mut rng);
        Self {
            floor,
            columns: colonnade(),
            crest: crest().transformed(&Mat4::translation(0.0, 11.0, -40.0)),
            chandelier_gold,
            chandelier_wax,
            flames,
            orbs: orbs(),
            gilding: chip_gilding(),
            dust: Particles::new(300, 12),
            sparks: Particles::new(60, 29),
            marble: pipeline(MarbleShader, CullingMode::None),
            gilded: pipeline(GildedShader, CullingMode::BackFace),
            flame: pipeline(FlameShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, BackdropShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

fn look() -> Look {
    Look {
        substrate: Vec3::new(0.1, 0.08, 0.075),
        die: Vec3::new(0.02, 0.42, 0.2),
        gold: Vec3::new(1.0, 0.78, 0.35),
        streams: Vec3::new(1.0, 0.8, 0.45),
        key: Vec3::new(0.1, 0.9, 0.35).normalise(),
        key_colour: Vec3::new(1.4, 0.9, 0.5),
        sky: Vec3::new(0.45, 0.26, 0.1),
        ground: Vec3::new(0.02, 0.014, 0.012),
        softbox: Vec3::new(0.7, 0.45, 0.2),
    }
}

impl Stage for BaroqueGold {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Violins],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::SmallBell),
                0.0,
                0.45,
                0.1,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let still = KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };
        let gilded = |metal| GildedUniforms {
            view: view.clone(),
            metal,
        };

        self.floor.draw(
            pass,
            &self.marble,
            still.clone(),
            FloorUniforms { view: view.clone() },
        );
        self.columns
            .draw(pass, &self.gilded, still.clone(), gilded(1.0));
        self.crest
            .draw(pass, &self.gilded, still.clone(), gilded(1.0));
        self.chandelier_gold
            .draw(pass, &self.gilded, still.clone(), gilded(1.0));
        self.chandelier_wax
            .draw(pass, &self.gilded, still.clone(), gilded(0.0));
        let flicker_sway = Motion::Sway(Sway {
            amplitude: 0.03,
            speed: 9.0,
            wind: Vec2::new(1.0, 0.0),
        });
        self.flames.draw(
            pass,
            &self.flame,
            KitUniforms {
                view: view.clone(),
                motion: flicker_sway,
            },
            view.clone(),
        );
        let orbit = Motion::Orbit(Orbit {
            centre: Vec3::new(0.0, 0.0, 0.0),
            radius: (4.5, 7.5),
            height: (-0.4, 4.0),
            speed: (0.3, 0.6),
            wiggle: 0.0,
        });
        self.orbs.draw(
            pass,
            &self.gilded,
            KitUniforms {
                view: view.clone(),
                motion: orbit,
            },
            gilded(1.0),
        );

        scene.chip.draw(pass, view, &look());
        self.gilding.draw(
            pass,
            &self.gilded,
            KitUniforms {
                view: view.clone(),
                motion: Motion::OnChip,
            },
            gilded(1.0),
        );

        let backdrop = BackdropUniforms { view: view.clone() };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| backdrop.clone());

        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 2.0, 0.0),
                half_extent: Vec3::new(14.0, 6.0, 14.0),
                velocity: Vec3::new(0.05, 0.12, 0.0),
                wander: 0.8,
                size: (0.012, 0.035),
                colour: Vec3::new(1.3, 0.85, 0.4),
                twinkle: 1.6,
            },
        );
        self.sparks.draw(
            pass,
            view,
            Drift {
                centre: LAMP + Vec3::new(0.0, -3.0, 0.0),
                half_extent: Vec3::new(3.0, 4.0, 3.0),
                velocity: Vec3::new(0.0, -0.4, 0.0),
                wander: 0.3,
                size: (0.02, 0.05),
                colour: Vec3::new(2.0, 1.2, 0.5),
                twinkle: 4.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.1,
            bloom: 0.9,
            vignette: 0.55,
            grain: 0.012,
            aberration: 0.0008,
            desaturate: 0.0,
        }
    }
}
