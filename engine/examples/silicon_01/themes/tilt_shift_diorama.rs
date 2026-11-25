//! A toy town on a board, photographed with a tilt-shift lens: the chip a miniature power plant
//! with a striped chimney and cooling towers, saturated blocks, tiny cars looping round the
//! roads, a park with a pond, and a depth-of-field blur that shrinks everything to model size.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::{HALF, Look};
use crate::shaders::composite::Grade;
use crate::shaders::scene_colour;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::simd::{
    F, clamp01, from_colour, hash2, hash2_scalar, length, mix3, reflect, rgb, s, select,
    smoothstep, splat3, step_lt, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonShader,
    ToonUniforms, pipeline,
};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::fbm;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::ScreenPass;
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, scatter};
use crate::themes::kit::shadow::GroundShadow;
use crate::themes::kit::shape::Shape;
use crate::themes::kit::sky::{Clouds, Sky, SkyShader, SkyUniforms};
use crate::themes::{Scene, Stage};

/// The height of the board: the chip's lowest corner rests on it.
const FLOOR: f32 = -1.95;
/// How far the roads, pavements and lots rise above the board, so each layer sits clear of the
/// one beneath it.
const ROAD_RISE: f32 = 0.03;
const PAVEMENT_RISE: f32 = 0.06;
const LOT_RISE: f32 = 0.1;
const SUN: Vec3 = Vec3 {
    x: -0.45,
    y: 0.7,
    z: 0.52,
};
/// The road centre lines along each axis, which divide the board into seven by seven blocks.
const EDGES: [f32; 8] = [-30.0, -21.5, -13.0, -4.5, 4.5, 13.0, 21.5, 30.0];
const ROAD_HALF: f32 = 1.2;
const PAVEMENT: f32 = 0.45;
const BOARD_HALF: f32 = 33.0;
/// The nearest block row that the camera, standing at z = 15, can't see anything in front of.
const FRONT_LIMIT: f32 = 12.5;
const WINDOW: (f32, f32) = (1.0, 0.95);
const POND: (f32, f32, f32) = (-8.75, 0.0, 2.2);

const WALLS: [Vec3; 7] = [
    Vec3 {
        x: 0.86,
        y: 0.22,
        z: 0.17,
    },
    Vec3 {
        x: 0.97,
        y: 0.7,
        z: 0.12,
    },
    Vec3 {
        x: 0.12,
        y: 0.62,
        z: 0.62,
    },
    Vec3 {
        x: 0.95,
        y: 0.46,
        z: 0.12,
    },
    Vec3 {
        x: 0.96,
        y: 0.9,
        z: 0.76,
    },
    Vec3 {
        x: 0.2,
        y: 0.46,
        z: 0.85,
    },
    Vec3 {
        x: 0.9,
        y: 0.38,
        z: 0.52,
    },
];
const ROOFS: [Vec3; 4] = [
    Vec3 {
        x: 0.22,
        y: 0.26,
        z: 0.32,
    },
    Vec3 {
        x: 0.72,
        y: 0.3,
        z: 0.18,
    },
    Vec3 {
        x: 0.62,
        y: 0.64,
        z: 0.66,
    },
    Vec3 {
        x: 0.18,
        y: 0.36,
        z: 0.3,
    },
];

/// What a block between four roads holds.
#[derive(Clone, Copy, PartialEq)]
enum Block {
    Plaza,
    Pond,
    Park,
    Lots,
}

fn block_kind(i: usize, j: usize) -> Block {
    match (i, j) {
        (3, 3) => Block::Plaza,
        (2, 3) => Block::Pond,
        _ if j >= 4 || hash2_scalar(i as f32 * 7.1, j as f32 * 3.3) < 0.14 => Block::Park,
        _ => Block::Lots,
    }
}

/// The ground of block (`i`, `j`) clear of its roads: (x0, x1, z0, z1).
fn block_bounds(i: usize, j: usize, inset: f32) -> (f32, f32, f32, f32) {
    let margin = ROAD_HALF + inset;
    (
        EDGES[i] + margin,
        EDGES[i + 1] - margin,
        EDGES[j] + margin,
        EDGES[j + 1] - margin,
    )
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.3, 1.16, 0.95),
        sky: Vec3::new(0.42, 0.52, 0.68),
        ground: Vec3::new(0.3, 0.27, 0.22),
        bands: 0.0,
        rim: Vec3::new(0.02, 0.03, 0.05),
        fog: Fog {
            colour: Vec3::new(0.55, 0.7, 0.88),
            density: 0.003,
            falloff: 0.08,
            base: FLOOR,
        },
    }
}

fn sky() -> Sky {
    Sky {
        zenith: Vec3::new(0.22, 0.48, 0.95),
        horizon: Vec3::new(0.7, 0.85, 1.0),
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.6, 1.4, 1.1),
        sun_size: 0.03,
        glow: 0.3,
        clouds: Some(Clouds {
            height: 220.0,
            coverage: 0.45,
            scale: 0.006,
            wind: Vec2::new(4.0, 1.0),
            lit: Vec3::new(1.5, 1.45, 1.4),
            shade: Vec3::new(0.7, 0.78, 0.92),
        }),
        stars: 0.0,
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.58, 0.58, 0.56),
        die: Vec3::new(0.7, 0.2, 0.12),
        gold: Vec3::new(1.0, 0.75, 0.22),
        streams: Vec3::new(1.0, 0.85, 0.3),
        key: light.sun,
        key_colour: light.sun_colour * 1.2,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.8, 0.85, 0.9),
    }
}

/// The distance from `a` to the nearest road centre line, along one axis.
#[inline(always)]
fn road_distance(a: F) -> F {
    let a = a.abs();
    EDGES[4..].iter().fold(s(1.0e3), |nearest, &line| {
        nearest.fast_min((a - s(line)).abs())
    })
}

/// 1 where `kind` lies in [`lo`, `hi`), else 0.
#[inline(always)]
fn within(kind: F, lo: f32, hi: f32) -> F {
    step_lt(kind, s(hi)) - step_lt(kind, s(lo))
}

/// How much of a feature `width` wide, centred where `x` is 0, covers a pixel `aa` wide.
#[inline(always)]
fn line(x: F, width: f32, aa: F) -> F {
    smoothstep(
        0.0,
        1.0,
        clamp01((s(width) + aa - x.abs()) / (aa * s(2.0) + s(1.0e-4))),
    )
}

/// A horizontal quad at height `y`, facing up.
fn flat(shape: &mut Shape, (x0, x1, z0, z1): (f32, f32, f32, f32), y: f32, colour: Vec3) {
    shape.quad(
        [
            Vec3::new(x0, y, z1),
            Vec3::new(x1, y, z1),
            Vec3::new(x1, y, z0),
            Vec3::new(x0, y, z0),
        ],
        colour,
    );
}

fn disc(shape: &mut Shape, centre: (f32, f32), radius: f32, y: f32, colour: Vec3) {
    let segments = 32;
    let point = |k: usize| {
        let angle = TAU * k as f32 / segments as f32;
        Vec3::new(
            centre.0 + radius * angle.cos(),
            y,
            centre.1 + radius * angle.sin(),
        )
    };
    for k in 0..segments {
        shape.triangle(
            Vec3::new(centre.0, y, centre.1),
            point(k + 1),
            point(k),
            colour,
        );
    }
}

/// The board's flat layers, each tagged by its seed with the kind of surface the shader draws.
fn ground() -> Shape {
    let (board, road, pavement, lot, park, water, plaza) = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9);
    let mut shape = Shape::new();
    let layer = |colour: Vec3, kind: f32, quads: &dyn Fn(&mut Shape)| {
        let mut part = Shape::new();
        quads(&mut part);
        part.colour_by(move |_, _| colour).with_seed(kind)
    };
    shape.append(&layer(Vec3::new(0.6, 0.45, 0.28), board, &|part| {
        flat(
            part,
            (-BOARD_HALF, BOARD_HALF, -BOARD_HALF, BOARD_HALF),
            FLOOR,
            Vec3::ONE,
        )
    }));
    let reach = EDGES[7] + ROAD_HALF;
    shape.append(&layer(Vec3::new(0.2, 0.2, 0.22), road, &|part| {
        for &line in &EDGES {
            flat(
                part,
                (line - ROAD_HALF, line + ROAD_HALF, -reach, reach),
                FLOOR + ROAD_RISE,
                Vec3::ONE,
            );
            flat(
                part,
                (-reach, reach, line - ROAD_HALF, line + ROAD_HALF),
                FLOOR + ROAD_RISE,
                Vec3::ONE,
            );
        }
    }));
    let blocks = || (0..7).flat_map(|i| (0..7).map(move |j| (i, j)));
    shape.append(&layer(Vec3::new(0.78, 0.76, 0.72), pavement, &|part| {
        for (i, j) in blocks() {
            flat(
                part,
                block_bounds(i, j, 0.0),
                FLOOR + PAVEMENT_RISE,
                Vec3::ONE,
            );
        }
    }));
    let inner = |wanted: fn(Block) -> bool| {
        move |part: &mut Shape| {
            for (i, j) in (0..7).flat_map(|i| (0..7).map(move |j| (i, j))) {
                if wanted(block_kind(i, j)) {
                    flat(
                        part,
                        block_bounds(i, j, PAVEMENT),
                        FLOOR + LOT_RISE,
                        Vec3::ONE,
                    );
                }
            }
        }
    };
    shape.append(&layer(
        Vec3::new(0.74, 0.7, 0.64),
        lot,
        &inner(|b| b == Block::Lots),
    ));
    shape.append(&layer(
        Vec3::new(0.42, 0.72, 0.28),
        park,
        &inner(|b| matches!(b, Block::Park | Block::Pond)),
    ));
    shape.append(&layer(
        Vec3::new(0.85, 0.62, 0.42),
        plaza,
        &inner(|b| b == Block::Plaza),
    ));
    shape.append(&layer(Vec3::new(0.1, 0.4, 0.6), water, &|part| {
        disc(
            part,
            (POND.0, POND.1),
            POND.2,
            FLOOR + LOT_RISE + 0.03,
            Vec3::ONE,
        )
    }));
    shape
}

struct Footprint {
    centre: (f32, f32),
    size: (f32, f32),
}

/// Divides a block's lot into one, two or four buildings with gaps between.
fn footprints(rng: &mut Rng, (x0, x1, z0, z1): (f32, f32, f32, f32)) -> Vec<Footprint> {
    let gap = 0.25;
    let part = |a0: f32, a1: f32, b0: f32, b1: f32| Footprint {
        centre: ((a0 + a1) / 2.0, (b0 + b1) / 2.0),
        size: (a1 - a0, b1 - b0),
    };
    let (mx, mz) = (
        x0 + (x1 - x0) * rng.range(0.4, 0.6),
        z0 + (z1 - z0) * rng.range(0.4, 0.6),
    );
    let pick = rng.unit();
    if pick < 0.25 {
        vec![part(x0, x1, z0, z1)]
    } else if pick < 0.6 {
        if rng.unit() < 0.5 {
            vec![
                part(x0, mx - gap / 2.0, z0, z1),
                part(mx + gap / 2.0, x1, z0, z1),
            ]
        } else {
            vec![
                part(x0, x1, z0, mz - gap / 2.0),
                part(x0, x1, mz + gap / 2.0, z1),
            ]
        }
    } else {
        vec![
            part(x0, mx - gap / 2.0, z0, mz - gap / 2.0),
            part(mx + gap / 2.0, x1, z0, mz - gap / 2.0),
            part(x0, mx - gap / 2.0, mz + gap / 2.0, z1),
            part(mx + gap / 2.0, x1, mz + gap / 2.0, z1),
        ]
    }
}

/// A building on the lot, as a flat roof, a gabled roof or a stepped tower; the seed is the
/// building's identity for its windows.
fn building(rng: &mut Rng, footprint: &Footprint, height: f32, tower: bool) -> Shape {
    let (cx, cz) = footprint.centre;
    let (w, d) = (footprint.size.0 - 0.1, footprint.size.1 - 0.1);
    let y0 = FLOOR + LOT_RISE;
    let wall = WALLS[rng.index(WALLS.len())];
    let roof = ROOFS[rng.index(ROOFS.len())];
    let trim = wall * 0.4 + Vec3::ONE * 0.55;
    let mut shape = Shape::new();
    let cuboid = |x: f32, z: f32, w: f32, d: f32, y: f32, h: f32, colour: Vec3| {
        Shape::cuboid(
            Vec3::new(x - w / 2.0, y, z - d / 2.0),
            Vec3::new(x + w / 2.0, y + h, z + d / 2.0),
            colour,
        )
    };
    if tower {
        let lower = height * 0.6;
        shape.append(&cuboid(cx, cz, w, d, y0, lower, wall));
        shape.append(&cuboid(cx, cz, w + 0.14, d + 0.14, y0 + lower, 0.12, trim));
        shape.append(&cuboid(
            cx,
            cz,
            w * 0.7,
            d * 0.7,
            y0 + lower + 0.12,
            height - lower,
            wall,
        ));
        shape.append(&cuboid(
            cx,
            cz,
            w * 0.7 + 0.12,
            d * 0.7 + 0.12,
            y0 + height + 0.12,
            0.1,
            trim,
        ));
        shape.append(&Shape::limb(
            Vec3::new(cx, y0 + height + 0.2, cz),
            Vec3::new(cx, y0 + height + 1.2, cz),
            0.025,
            0.015,
            4,
            Vec3::new(0.3, 0.3, 0.32),
        ));
    } else if rng.unit() < 0.5 {
        shape.append(&cuboid(cx, cz, w, d, y0, height, wall));
        shape.append(&cuboid(cx, cz, w + 0.14, d + 0.14, y0 + height, 0.12, trim));
        let (bx, bz) = (rng.range(-0.25, 0.25) * w, rng.range(-0.25, 0.25) * d);
        shape.append(&cuboid(
            cx + bx,
            cz + bz,
            w * 0.3,
            d * 0.3,
            y0 + height + 0.12,
            0.3,
            Vec3::new(0.6, 0.62, 0.64),
        ));
    } else {
        shape.append(&cuboid(cx, cz, w, d, y0, height, wall));
        let (hw, hd, eave) = (w / 2.0 + 0.1, d / 2.0 + 0.1, y0 + height);
        let rise = hw.min(hd) * 0.9;
        let along_x = w >= d;
        let (ridge_a, ridge_b) = if along_x {
            (
                Vec3::new(cx - hw, eave + rise, cz),
                Vec3::new(cx + hw, eave + rise, cz),
            )
        } else {
            (
                Vec3::new(cx, eave + rise, cz - hd),
                Vec3::new(cx, eave + rise, cz + hd),
            )
        };
        let corner = |sx: f32, sz: f32| Vec3::new(cx + sx * hw, eave, cz + sz * hd);
        if along_x {
            shape.quad(
                [corner(-1.0, -1.0), corner(1.0, -1.0), ridge_b, ridge_a],
                roof,
            );
            shape.quad(
                [corner(1.0, 1.0), corner(-1.0, 1.0), ridge_a, ridge_b],
                roof,
            );
            shape.triangle(corner(-1.0, -1.0), ridge_a, corner(-1.0, 1.0), wall);
            shape.triangle(corner(1.0, 1.0), ridge_b, corner(1.0, -1.0), wall);
        } else {
            shape.quad(
                [corner(1.0, -1.0), corner(1.0, 1.0), ridge_b, ridge_a],
                roof,
            );
            shape.quad(
                [corner(-1.0, 1.0), corner(-1.0, -1.0), ridge_a, ridge_b],
                roof,
            );
            shape.triangle(corner(-1.0, -1.0), corner(1.0, -1.0), ridge_a, wall);
            shape.triangle(corner(1.0, 1.0), corner(-1.0, 1.0), ridge_b, wall);
        }
    }
    shape.with_seed(rng.unit())
}

/// The town: every lot's buildings, with the tallest kept to the edges so the chip stays the
/// tallest thing near the middle.
fn town(rng: &mut Rng) -> Shape {
    let mut shape = Shape::new();
    for i in 0..7 {
        for j in 0..7 {
            if block_kind(i, j) != Block::Lots {
                continue;
            }
            let centre = (
                (EDGES[i] + EDGES[i + 1]) / 2.0,
                (EDGES[j] + EDGES[j + 1]) / 2.0,
            );
            let near_chip = centre.0.abs() < 13.0 && centre.1.abs() < 5.0;
            for footprint in footprints(rng, block_bounds(i, j, PAVEMENT)) {
                let tall =
                    !near_chip && (centre.0.abs() > 12.0 || centre.1 < -12.0) && rng.unit() < 0.35;
                let height = if tall {
                    rng.range(4.5, 7.5)
                } else if near_chip {
                    rng.range(1.4, 3.0)
                } else {
                    rng.range(1.6, 3.6)
                };
                shape.append(&building(rng, &footprint, height, tall));
            }
        }
    }
    shape
}

fn tree_variants(rng: &mut Rng) -> Vec<Shape> {
    let trunk_colour = Vec3::new(0.42, 0.28, 0.16);
    let greens = [
        Vec3::new(0.2, 0.6, 0.22),
        Vec3::new(0.34, 0.72, 0.24),
        Vec3::new(0.12, 0.5, 0.3),
        Vec3::new(0.9, 0.52, 0.14),
    ];
    let mut variants = Vec::new();
    for (k, green) in greens.iter().enumerate() {
        let mut tree = Shape::new();
        tree.append(&Shape::cylinder(0.09, 0.07, 0.6, 6, trunk_colour));
        if k == 2 {
            tree.append(
                &Shape::cone(0.5, 1.0, 7, *green).transformed(&Mat4::translation(0.0, 0.35, 0.0)),
            );
            tree.append(
                &Shape::cone(0.38, 0.8, 7, *green).transformed(&Mat4::translation(0.0, 0.9, 0.0)),
            );
        } else {
            let crown =
                Shape::icosphere(0.5, 1, 0.12, k as f32 * 3.1, *green).facet_variation(rng, 0.08);
            tree.append(&crown.transformed(&Mat4::translation(0.0, 0.95, 0.0)));
        }
        variants.push(tree.sway_by_height(0.5, 1.5, 1.0));
    }
    variants
}

/// The trees of the parks, kept off the pond and clear of the camera's view of the chip.
fn trees(rng: &mut Rng) -> Shape {
    let in_park = |x: f32, z: f32| {
        let cell = |a: f32| EDGES.windows(2).position(|e| a >= e[0] && a < e[1]);
        let (Some(i), Some(j)) = (cell(x), cell(z)) else {
            return false;
        };
        let (x0, x1, z0, z1) = block_bounds(i, j, PAVEMENT + 0.3);
        matches!(block_kind(i, j), Block::Park | Block::Pond)
            && (x0..x1).contains(&x)
            && (z0..z1).contains(&z)
            && z < FRONT_LIMIT
            && !(z > 5.0 && x.abs() < 3.0)
            && (x - POND.0).hypot(z - POND.1) > POND.2 + 0.5
    };
    let spots = scatter(
        rng,
        31.0,
        1.5,
        9000,
        |_, _| FLOOR + LOT_RISE,
        |x, z| in_park(x, z).then_some(1.0),
        (0.75, 1.25),
    );
    let variants = tree_variants(rng);
    let mut batch = Batch::new();
    batch.add(&variants, &spots, |p| {
        (p.seed * 17.0) as usize % variants.len()
    });
    batch.finish()
}

fn car(body: Vec3) -> Shape {
    let mut shape = Shape::new();
    shape.append(&Shape::cuboid(
        Vec3::new(-0.5, 0.06, -0.24),
        Vec3::new(0.5, 0.3, 0.24),
        body,
    ));
    shape.append(&Shape::cuboid(
        Vec3::new(-0.3, 0.3, -0.2),
        Vec3::new(0.2, 0.5, 0.2),
        Vec3::new(0.85, 0.92, 0.98),
    ));
    let wheel = Vec3::new(0.08, 0.08, 0.09);
    for x in [-0.32, 0.32] {
        for z in [-0.25, 0.25] {
            shape.append(&Shape::cuboid(
                Vec3::new(x - 0.1, 0.0, z - 0.04),
                Vec3::new(x + 0.1, 0.2, z + 0.04),
                wheel,
            ));
        }
    }
    shape
}

/// A rounded rectangle on the board, ridden by cars in one direction.
struct Track {
    half: (f32, f32),
    corner: f32,
    reversed: bool,
    /// How fast every car on the lane drives, in world units per second.
    speed: f32,
}

impl Track {
    fn length(&self) -> f32 {
        4.0 * (self.half.0 - self.corner) + 4.0 * (self.half.1 - self.corner) + TAU * self.corner
    }

    /// Where a car `distance` along the track is, and its heading, as (x, z) pairs.
    fn pose(&self, distance: f32) -> ((f32, f32), (f32, f32)) {
        let along = distance.rem_euclid(self.length());
        let (position, heading) = self.forward(if self.reversed {
            self.length() - along
        } else {
            along
        });
        (
            position,
            if self.reversed {
                (-heading.0, -heading.1)
            } else {
                heading
            },
        )
    }

    fn forward(&self, mut d: f32) -> ((f32, f32), (f32, f32)) {
        let (hx, hz, r) = (self.half.0, self.half.1, self.corner);
        let (side_z, side_x, arc) = (2.0 * (hz - r), 2.0 * (hx - r), FRAC_PI_2 * r);
        let (a, b) = (hx - r, hz - r);
        if d < side_z {
            return ((hx, b - d), (0.0, -1.0));
        }
        d -= side_z;
        if d < arc {
            let phi = d / r;
            return (
                (a + r * phi.cos(), -b - r * phi.sin()),
                (-phi.sin(), -phi.cos()),
            );
        }
        d -= arc;
        if d < side_x {
            return ((a - d, -hz), (-1.0, 0.0));
        }
        d -= side_x;
        if d < arc {
            let phi = d / r;
            return (
                (-a - r * phi.sin(), -b - r * phi.cos()),
                (-phi.cos(), phi.sin()),
            );
        }
        d -= arc;
        if d < side_z {
            return ((-hx, -b + d), (0.0, 1.0));
        }
        d -= side_z;
        if d < arc {
            let phi = d / r;
            return (
                (-a - r * phi.cos(), b + r * phi.sin()),
                (phi.sin(), phi.cos()),
            );
        }
        d -= arc;
        if d < side_x {
            return ((-a + d, hz), (1.0, 0.0));
        }
        d -= side_x;
        let phi = d / r;
        (
            (a + r * phi.sin(), b + r * phi.cos()),
            (phi.cos(), -phi.sin()),
        )
    }
}

struct Car {
    shape: Shape,
    track: usize,
    offset: f32,
}

/// The two lanes of each ring road, and the cars on them.
fn traffic(rng: &mut Rng) -> (Vec<Track>, Vec<Car>) {
    let body = [
        Vec3::new(0.9, 0.15, 0.12),
        Vec3::new(0.98, 0.78, 0.1),
        Vec3::new(0.15, 0.5, 0.9),
        Vec3::new(0.95, 0.95, 0.92),
        Vec3::new(0.15, 0.7, 0.4),
        Vec3::new(0.95, 0.5, 0.1),
    ];
    let mut tracks = Vec::new();
    let mut cars = Vec::new();
    for (ring, count) in [(EDGES[4], 3), (EDGES[5], 5), (EDGES[6], 7)] {
        for (lane, reversed) in [(0.55, false), (-0.55, true)] {
            let half = ring + lane;
            let corner = if reversed { 0.7 } else { 1.7 };
            tracks.push(Track {
                half: (half, half),
                corner,
                reversed,
                speed: rng.range(2.0, 2.8),
            });
            let track = tracks.len() - 1;
            for k in 0..count {
                cars.push(Car {
                    shape: car(body[rng.index(body.len())]),
                    track,
                    offset: (k as f32 + rng.range(-0.3, 0.3)) / count as f32
                        * tracks[track].length(),
                });
            }
        }
    }
    (tracks, cars)
}

/// A hyperboloid cooling tower on the y axis, banded in red near the rim.
fn cooling_tower() -> Shape {
    let profile = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.46, 0.5),
        (0.36, 1.1),
        (0.34, 1.4),
        (0.4, 1.9),
        (0.42, 2.0),
        (0.38, 2.0),
        (0.0, 1.9),
    ];
    Shape::lathe(&profile, 14, Vec3::new(0.9, 0.9, 0.88)).colour_by(|p, _| {
        if (1.7..1.88).contains(&p.y) {
            Vec3::new(0.85, 0.2, 0.15)
        } else {
            Vec3::new(0.9, 0.9, 0.88)
        }
    })
}

/// The power plant's additions, in the chip's own space: a striped chimney and two cooling
/// towers standing off the package's top edge.
fn plant() -> Shape {
    let up = Mat4::rotate_x(-FRAC_PI_2);
    let mut shape = Shape::new();
    let stripes = 6;
    let mut profile = vec![(0.0, 0.0)];
    profile.extend((0..=stripes).map(|k| {
        (
            0.14 - 0.05 * k as f32 / stripes as f32,
            1.9 * k as f32 / stripes as f32,
        )
    }));
    profile.push((0.0, 1.9));
    let chimney = Shape::lathe(&profile, 10, Vec3::ONE).colour_by(|p, _| {
        if (p.y / (1.9 / stripes as f32)).floor() as i32 % 2 == 0 {
            Vec3::new(0.85, 0.18, 0.14)
        } else {
            Vec3::new(0.95, 0.95, 0.93)
        }
    });
    shape.append(&chimney.transformed(&(Mat4::translation(1.3, 0.17, -HALF + 0.9) * up)));
    for x in [-1.5, -0.4] {
        shape.append(&cooling_tower().transformed(
            &(Mat4::translation(x, 0.3, -HALF + 0.9) * up * Mat4::scaling(0.75, 0.7, 0.75)),
        ));
    }
    shape
}

struct GroundUniforms {
    toon: ToonUniforms,
    shadow: Arc<GroundShadow>,
}

/// The board's surfaces: asphalt with dashed lane lines, paved pavements, brick paving, mown
/// grass with flowers and the pond, which mirrors the sky in ripples and glints.
struct GroundShader;

impl FragmentShaderSimd<KitVaryings> for GroundShader {
    type Uniforms = GroundUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GroundUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let time = view.frame.time;
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let eye_dir = to_eye * (s(1.0) / distance);
        let kind = v.seed();
        let (x, z) = (world.x, world.z);
        let aa = distance * s(view.pixel_angle * 1.4);
        let near = smoothstep(55.0, 18.0, distance);

        let (board, asphalt, pavement, lot, park, water, plaza) = (
            within(kind, -1.0, 0.05),
            within(kind, 0.05, 0.15),
            within(kind, 0.15, 0.25),
            within(kind, 0.25, 0.4),
            within(kind, 0.4, 0.6),
            within(kind, 0.6, 0.8),
            within(kind, 0.8, 1.1),
        );
        let grain = fbm(x * s(0.9), z * s(0.9), 3);
        let base = v.colour();

        let (dx, dz) = (road_distance(x), road_distance(z));
        let clear_x = smoothstep(ROAD_HALF + 0.2, ROAD_HALF + 0.9, dz);
        let clear_z = smoothstep(ROAD_HALF + 0.2, ROAD_HALF + 0.9, dx);
        let dash_z = step_lt((z * s(0.5)) - (z * s(0.5)).floor(), s(0.55));
        let dash_x = step_lt((x * s(0.5)) - (x * s(0.5)).floor(), s(0.55));
        let marks =
            (line(dx, 0.06, aa) * clear_x * dash_z + line(dz, 0.06, aa) * clear_z * dash_x) * near;
        let road_colour = mix3(
            base * (s(0.85) + grain * s(0.3)),
            rgb(0.92, 0.9, 0.8),
            marks * s(0.9),
        );

        let slab = |a: F| {
            let f = (a * s(1.0 / 0.7)) - (a * s(1.0 / 0.7)).floor();
            line(f.fast_min(s(1.0) - f) * s(0.7), 0.02, aa)
        };
        let pavement_colour = base
            * (s(0.92) + grain * s(0.1) - (slab(x) + slab(z)).fast_min(s(1.0)) * s(0.12) * near);

        let lot_colour = base * (s(0.9) + grain * s(0.2));

        let stripe = step_lt(((x + z) * s(0.35)) - ((x + z) * s(0.35)).floor(), s(0.5));
        let (bed_x, bed_z) = ((x * s(2.0)).floor(), (z * s(2.0)).floor());
        let seed_x = hash2(bed_x, bed_z);
        let seed_z = hash2(bed_z + s(5.3), bed_x);
        let spot_x = bed_x + s(0.25) + seed_x * s(0.5);
        let spot_z = bed_z + s(0.25) + seed_z * s(0.5);
        let dot = length(Vec3Simd::new(
            x * s(2.0) - spot_x,
            s(0.0),
            z * s(2.0) - spot_z,
        ));
        let petal = mix3(
            rgb(1.0, 0.95, 0.5),
            rgb(0.95, 0.3, 0.4),
            step_lt(hash2(bed_z, bed_x + s(2.0)), s(0.5)),
        );
        let flower = (s(1.0) - smoothstep(0.08, 0.14, dot))
            * step_lt(hash2(bed_x + s(9.0), bed_z), s(0.4))
            * near;
        let walk = ((z * s(0.45)).sin() * s(1.2) - x).abs();
        let path = (s(1.0) - smoothstep(0.45, 0.6, walk)) * step_lt(s(4.6), z);
        let grass = base * (s(0.88) + stripe * s(0.12) + grain * s(0.16));
        let park_colour = mix3(
            mix3(grass, petal, flower),
            rgb(0.82, 0.72, 0.52) * (s(0.9) + grain * s(0.2)),
            path,
        );

        let brick_row = (z * s(2.6)).floor();
        let shifted = x * s(2.6) + brick_row * s(0.5);
        let (brick_u, brick_v) = (shifted - shifted.floor(), z * s(2.6) - brick_row);
        let mortar = (line(brick_u - s(0.5), 0.46, aa * s(2.6))
            * line(brick_v - s(0.5), 0.44, aa * s(2.6)))
        .fast_min(s(1.0));
        let brick_h = hash2(shifted.floor(), brick_row);
        let plaza_colour = base * (s(0.8) + brick_h * s(0.35)) * (s(0.7) + mortar * s(0.3));

        let albedo = road_colour * (asphalt + board * s(0.0))
            + base * (board * (s(0.9) + grain * s(0.2)))
            + pavement_colour * pavement
            + lot_colour * lot
            + park_colour * park
            + plaza_colour * plaza;

        let shadow =
            clamp01(u.shadow.lit(x, z) * (s(1.0) - chip_shadow(view, u.toon.lighting.sun, world)));
        let normal = Vec3Simd::new(s(0.0), s(1.0), s(0.0));
        let ground = u.toon.lighting.shade(view, world, normal, albedo, shadow);

        let ripple_x = fbm(x * s(3.0) + s(time * 0.35), z * s(3.0), 2) - s(0.5);
        let ripple_z = fbm(x * s(3.0), z * s(3.0) - s(time * 0.3), 2) - s(0.5);
        let rippled = Vec3Simd::new(ripple_x * s(0.35), s(1.0), ripple_z * s(0.35)).normalise();
        let facing_eye = clamp01(rippled.dot(eye_dir));
        let grazing = s(1.0) - facing_eye;
        let grazing2 = grazing * grazing;
        let fresnel = s(0.15) + s(0.85) * grazing2 * grazing2;
        let mirror = reflect(-eye_dir, rippled);
        let sky_colour = mix3(
            rgb(0.7, 0.85, 1.0),
            rgb(0.3, 0.55, 1.0),
            smoothstep(0.0, 0.7, mirror.y),
        );
        let glint = mirror.dot(splat3(u.toon.lighting.sun)).fast_max(s(0.0));
        let glint = glint * glint * glint * glint;
        let glint = glint * glint * glint * glint;
        let pond = mix3(base * s(0.55) + rgb(0.0, 0.05, 0.05), sky_colour, fresnel)
            + splat3(u.toon.lighting.sun_colour) * (glint * s(1.2));

        let colour = mix3(ground, pond, water);
        scene_colour(view, colour, world)
    }
}

/// Buildings: painted walls with a grid of windows that mirror the sky, ground floors with
/// large shopfronts, and the odd glint of the sun in the glass.
struct TownShader;

impl FragmentShaderSimd<KitVaryings> for TownShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let eye_dir = to_eye * (s(1.0) / distance);
        let n = facing(v.normal().normalise(), to_eye);
        let wall = smoothstep(0.5, 0.2, n.y.abs());
        let along = select(n.x.abs().simd_lt(n.z.abs()), world.x, world.z);
        let height = world.y - s(FLOOR + LOT_RISE);
        let (cu, cv) = (along * s(1.0 / WINDOW.0), height * s(1.0 / WINDOW.1));
        let (iu, iv) = (cu.floor(), cv.floor());
        let (fu, fv) = (cu - iu, cv - iv);
        let footprint = distance * s(view.pixel_angle * 1.4);
        let band = |a: F, lo: f32, hi: f32, aa: F| {
            smoothstep(0.0, 1.0, clamp01((a - s(lo)) / aa + s(0.5)))
                - smoothstep(0.0, 1.0, clamp01((a - s(hi)) / aa + s(0.5)))
        };
        let (aau, aav) = (
            footprint * s(1.0 / WINDOW.0) + s(1.0e-3),
            footprint * s(1.0 / WINDOW.1) + s(1.0e-3),
        );
        let shop = step_lt(iv, s(1.0));
        let pane_v = band(fv, 0.3, 0.82, aav) * (s(1.0) - shop) + band(fv, 0.12, 0.8, aav) * shop;
        let pane = band(fu, 0.2, 0.8, aau) * pane_v * wall;
        let frame_v = band(fv, 0.24, 0.88, aav);
        let frame = (band(fu, 0.14, 0.86, aau) * frame_v - pane).fast_max(s(0.0)) * wall;
        let detail = smoothstep(70.0, 32.0, distance);

        let identity = hash2(iu + v.seed() * s(61.0), iv + shop * s(3.0));
        let reflection = mix3(
            rgb(0.07, 0.15, 0.24),
            rgb(0.45, 0.66, 0.86),
            clamp01(fv * s(1.1)),
        );
        let curtain = smoothstep(0.7, 0.72, identity) * (s(1.0) - shop);
        let glass = mix3(reflection, rgb(0.9, 0.66, 0.32), curtain * s(0.85));

        let wall_colour = v.colour() * (s(1.0) + frame * s(0.18) * detail);
        let albedo = mix3(wall_colour, glass, pane * detail);
        let mut colour = u.lighting.shade(view, world, n, albedo, s(1.0));
        let mirror = reflect(-eye_dir, n);
        let toward_sun = mirror.dot(splat3(u.lighting.sun)).fast_max(s(0.0));
        let toward_sun =
            toward_sun * toward_sun * toward_sun * toward_sun * toward_sun * toward_sun;
        colour = colour
            + splat3(u.lighting.sun_colour)
                * (toward_sun * pane * detail * (s(1.0) - curtain) * s(0.7));
        scene_colour(view, colour, world)
    }
}

#[derive(Clone)]
struct BlurUniforms {
    source: Arc<TextureSampler>,
    unit: f32,
}

/// Depth of field with a focus band: pixels blur in proportion to how far their view distance
/// is from the focus, gathering only neighbours that are themselves blurred as widely, so
/// sharp foreground edges don't bleed into the blur behind them. Saturation is lifted to the
/// colours of a toy.
struct TiltShiftShader;

impl TiltShiftShader {
    const FOCUS: f32 = 15.0;
    const BAND: f32 = 4.0;
    const REACH: f32 = 26.0;
    const RADIUS: f32 = 7.5;
    const TAPS: usize = 28;
    const SATURATION: f32 = 1.2;
}

#[inline(always)]
fn circle_of_confusion(distance: F, unit: f32) -> F {
    smoothstep(
        TiltShiftShader::BAND,
        TiltShiftShader::REACH,
        (distance - s(TiltShiftShader::FOCUS)).abs(),
    ) * s(TiltShiftShader::RADIUS * unit)
}

impl FragmentShaderSimd<ScreenVaryings> for TiltShiftShader {
    type Uniforms = BlurUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BlurUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (width, height) = (u.source.width() as f32, u.source.height() as f32);
        let centre = u.source.sample_linear_clamp_simd([x, y]);
        let radius = circle_of_confusion(centre.a, u.unit);

        let mut sum = from_colour(centre);
        let mut weight = s(1.0);
        for k in 0..Self::TAPS {
            let t = (k as f32 + 0.5) / Self::TAPS as f32;
            let angle = k as f32 * 2.399_963_1;
            let reach = t.sqrt();
            let (ox, oy) = (angle.cos() * reach, angle.sin() * reach);
            let offset = radius * s(reach);
            let tap = u.source.sample_linear_clamp_simd([
                x + radius * s(ox / width),
                y + radius * s(oy / height),
            ]);
            let tap_radius = circle_of_confusion(tap.a, u.unit);
            let w = clamp01((tap_radius + s(0.75)) / (offset + s(0.75)));
            sum = sum + from_colour(tap) * w;
            weight += w;
        }
        let colour = sum * (s(1.0) / weight);
        let luma = colour.x * s(0.2126) + colour.y * s(0.7152) + colour.z * s(0.0722);
        let vivid =
            splat3(Vec3::ONE) * luma + (colour - splat3(Vec3::ONE) * luma) * s(Self::SATURATION);
        to_colour(
            Vec3Simd::new(
                vivid.x.fast_max(s(0.0)),
                vivid.y.fast_max(s(0.0)),
                vivid.z.fast_max(s(0.0)),
            ),
            centre.a,
        )
    }
}

pub struct TiltShiftDiorama {
    ground: Shape,
    town: Shape,
    trees: Shape,
    plant: Shape,
    tracks: Vec<Track>,
    cars: Vec<Car>,
    traffic: Shape,
    shadow: Arc<GroundShadow>,
    steam: Particles,
    ground_pipeline: SimdPipeline<KitVertexShader, GroundShader>,
    town_pipeline: SimdPipeline<KitVertexShader, TownShader>,
    plain: SimdPipeline<KitVertexShader, ToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyShader>,
    blur: ScreenPass<TiltShiftShader>,
}

impl TiltShiftDiorama {
    pub fn new() -> Self {
        let mut rng = Rng::new(2024);
        let town = town(&mut rng);
        let trees = trees(&mut rng);
        let (tracks, cars) = traffic(&mut rng);
        let shadow = Arc::new(GroundShadow::bake(
            Vec2::new(-BOARD_HALF, -BOARD_HALF),
            2.0 * BOARD_HALF,
            1024,
            SUN.normalise(),
            |_, _| FLOOR + LOT_RISE,
            &[&town, &trees],
            2,
        ));
        Self {
            ground: ground(),
            town,
            trees,
            plant: plant(),
            tracks,
            cars,
            traffic: Shape::new(),
            shadow,
            steam: Particles::new(70, 505),
            ground_pipeline: pipeline(GroundShader, CullingMode::None),
            town_pipeline: pipeline(TownShader, CullingMode::None),
            plain: pipeline(ToonShader, CullingMode::None),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            blur: ScreenPass::new(TiltShiftShader),
        }
    }
}

impl Stage for TiltShiftDiorama {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Xylophone],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::ToyHorn),
                0.0,
                0.45,
                0.08,
                0.4,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let time = scene.view.frame.time;
        let mut traffic = Shape::new();
        for car in &self.cars {
            let (position, heading) =
                self.tracks[car.track].pose(car.offset + time * self.tracks[car.track].speed);
            let angle = (-heading.1).atan2(heading.0);
            let pose = Mat4::translation(position.0, FLOOR + ROAD_RISE, position.1)
                * Mat4::rotate_y(angle);
            traffic.append(&car.shape.transformed(&pose));
        }
        self.traffic = traffic;
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
        let breeze = Motion::Sway(Sway {
            amplitude: 0.05,
            speed: 1.3,
            wind: Vec2::new(1.0, 0.0),
        });

        self.ground.draw(
            pass,
            &self.ground_pipeline,
            motion(Motion::Still),
            GroundUniforms {
                toon: toon.clone(),
                shadow: self.shadow.clone(),
            },
        );
        self.town.draw(
            pass,
            &self.town_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.trees
            .draw(pass, &self.plain, motion(breeze), toon.clone());
        self.traffic
            .draw(pass, &self.plain, motion(Motion::Still), toon.clone());

        scene.chip.draw(pass, view, &look());
        self.plant
            .draw(pass, &self.plain, motion(Motion::OnChip), toon.clone());

        let sky = SkyUniforms {
            view: view.clone(),
            sky: sky(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| sky.clone());

        self.steam.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(-0.95, 2.9, -1.4),
                half_extent: Vec3::new(0.7, 1.0, 0.4),
                velocity: Vec3::new(0.25, 0.7, 0.0),
                wander: 0.5,
                size: (0.1, 0.28),
                colour: Vec3::new(0.7, 0.7, 0.72),
                twinkle: 0.0,
            },
        );
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let uniforms = BlurUniforms {
            source: image,
            unit: scene.extent.height as f32 / 720.0,
        };
        Some(self.blur.run(context, scene, uniforms))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 0.95,
            bloom: 0.2,
            vignette: 0.28,
            grain: 0.0,
            aberration: 0.0,
            desaturate: -0.08,
        }
    }
}
