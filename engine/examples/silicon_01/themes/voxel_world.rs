//! A blocky daylight world: a voxel landscape of grass, sand, stone and water with cube trees and
//! cube clouds, blocky animals walking round the chip, and the chip itself rebuilt from voxels.
//!
//! Terrain and chip share one mesher: a [`HeightField`] of columns, merged into long quads along
//! each row, with faces hidden by neighbouring columns left out. Block textures are drawn in the
//! fragment shader as a 16x16 texel pattern per block face, fading to flat colour with distance.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::HALF;
use crate::shaders::composite::Grade;
use crate::shaders::screen::ScreenVertexShader;
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{hash2, hash2_scalar, length, rgb, s, select, select3, smoothstep, splat3};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, pipeline,
};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::light::{Fog, facing};
use crate::themes::kit::noise::{fbm_scalar, value, value_scalar};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::sky::{Sky, SkyShader, SkyUniforms};
use crate::themes::{Scene, Stage};

/// The height of the grass plain the chip hovers above.
const FLOOR: f32 = -1.95;
/// The edge of a world block, and the number of blocks across the map.
const BLOCK: f32 = 0.75;
const CELLS: usize = 160;
/// How deep the map's edge cliffs and a chip's sides go, in blocks.
const DROP: i32 = -8;
/// Water's surface, and the heights above the plain where sand ends and stone and snow begin.
const WATER: f32 = -2.2;
const STONE: f32 = 14.0;
const SNOW: f32 = 26.0;
const CHIP_CELLS: usize = 32;
const CLOUDS: usize = 16;

/// What a block is made of, which sets its colour and how its faces are drawn.
#[derive(Clone, Copy, PartialEq)]
enum Block {
    Grass,
    Dirt,
    Stone,
    Sand,
    Snow,
    Water,
    Wood,
    Leaves,
    Cloud,
    Package,
    Die,
    Core,
    Gold,
    Marker,
}

impl Block {
    fn colour(self) -> Vec3 {
        match self {
            Block::Grass => Vec3::new(0.3, 0.6, 0.16),
            Block::Dirt => Vec3::new(0.42, 0.29, 0.18),
            Block::Stone => Vec3::new(0.5, 0.5, 0.52),
            Block::Sand => Vec3::new(0.88, 0.8, 0.52),
            Block::Snow => Vec3::new(0.94, 0.97, 1.0),
            Block::Water => Vec3::new(0.1, 0.36, 0.85),
            Block::Wood => Vec3::new(0.4, 0.27, 0.13),
            Block::Leaves => Vec3::new(0.15, 0.44, 0.11),
            Block::Cloud => Vec3::new(1.05, 1.05, 1.05),
            Block::Package => Vec3::new(0.42, 0.43, 0.47),
            Block::Die => Vec3::new(0.05, 0.22, 0.12),
            Block::Core => Vec3::new(0.11, 0.46, 0.24),
            Block::Gold => Vec3::new(0.95, 0.74, 0.2),
            Block::Marker => Vec3::new(0.95, 0.95, 0.9),
        }
    }

    /// Passed to the shader in each vertex's seed: grass has a lip on its sides, water moves.
    fn kind(self) -> f32 {
        match self {
            Block::Grass => 0.2,
            Block::Water => 0.4,
            _ => 0.0,
        }
    }

    /// The block `depth` layers below the surface of a column topped with `self`.
    fn below(self, depth: i32) -> Block {
        match self {
            Block::Grass if depth <= 2 => Block::Dirt,
            Block::Sand if depth <= 3 => Block::Sand,
            Block::Water if depth <= 2 => Block::Dirt,
            Block::Package | Block::Die | Block::Core | Block::Gold | Block::Marker => {
                Block::Package
            }
            _ => Block::Stone,
        }
    }
}

/// A grid of columns, each a height in blocks and the block on top.
struct HeightField {
    cells: usize,
    size: f32,
    /// The world position of the grid's (0, 0) corner, in x and z, and the y of height 0.
    origin: (f32, f32),
    base: f32,
    tops: Vec<(i32, Block)>,
}

impl HeightField {
    fn new(
        cells: usize,
        size: f32,
        origin: (f32, f32),
        base: f32,
        column: impl Fn(usize, usize) -> (i32, Block),
    ) -> Self {
        let tops = (0..cells * cells)
            .map(|k| column(k % cells, k / cells))
            .collect();
        Self {
            cells,
            size,
            origin,
            base,
            tops,
        }
    }

    /// The column at `(i, j)`; beyond the grid it is the bottom of the edge cliffs.
    fn column(&self, i: isize, j: isize) -> (i32, Block) {
        let range = 0..self.cells as isize;
        if range.contains(&i) && range.contains(&j) {
            self.tops[j as usize * self.cells + i as usize]
        } else {
            (DROP, Block::Stone)
        }
    }

    fn y(&self, height: i32) -> f32 {
        self.base + height as f32 * self.size
    }

    fn mesh(&self) -> Shape {
        let mut shape = Shape::new();
        let n = self.cells as isize;
        let (x0, z0) = self.origin;
        for j in 0..n {
            let mut i = 0;
            while i < n {
                let column = self.column(i, j);
                let mut end = i + 1;
                while end < n && self.column(end, j) == column {
                    end += 1;
                }
                let (xa, xb) = (x0 + i as f32 * self.size, x0 + end as f32 * self.size);
                let (za, zb) = (z0 + j as f32 * self.size, z0 + (j + 1) as f32 * self.size);
                let y = self.y(column.0);
                let mut quad = Shape::new();
                quad.quad(
                    [
                        Vec3::new(xa, y, zb),
                        Vec3::new(xb, y, zb),
                        Vec3::new(xb, y, za),
                        Vec3::new(xa, y, za),
                    ],
                    column.1.colour(),
                );
                shape.append(&quad.with_seed(column.1.kind()));
                i = end;
            }
        }
        for (step_i, step_j) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            self.walls(&mut shape, step_i, step_j);
        }
        shape
    }

    /// The faces of columns that stand above the neighbour in direction `(di, dj)`, merged along
    /// the row they face.
    fn walls(&self, shape: &mut Shape, di: isize, dj: isize) {
        let n = self.cells as isize;
        let along = |k: isize, row: isize| if di == 0 { (k, row) } else { (row, k) };
        let key = |k: isize, row: isize| {
            let (i, j) = along(k, row);
            let (h, top) = self.column(i, j);
            (h, self.column(i + di, j + dj).0, top)
        };
        for row in 0..n {
            let mut k = 0;
            while k < n {
                let (h, lower, top) = key(k, row);
                let mut end = k + 1;
                while end < n && key(end, row) == (h, lower, top) {
                    end += 1;
                }
                if h > lower {
                    self.wall(shape, (di, dj), (k, end), row, (h, lower, top));
                }
                k = end;
            }
        }
    }

    fn wall(
        &self,
        shape: &mut Shape,
        (di, dj): (isize, isize),
        (k0, k1): (isize, isize),
        row: isize,
        (h, lower, top): (i32, i32, Block),
    ) {
        let (x0, z0) = self.origin;
        let (a, b) = (k0 as f32 * self.size, k1 as f32 * self.size);
        let plane = (row + isize::from(di + dj > 0)) as f32 * self.size;
        let corner = |along: f32, y: f32| {
            if di == 0 {
                Vec3::new(x0 + along, y, z0 + plane)
            } else {
                Vec3::new(x0 + plane, y, z0 + along)
            }
        };
        let mut layer = h;
        while layer > lower {
            let block = if layer == h {
                top
            } else {
                top.below(h - layer)
            };
            let mut bottom = layer - 1;
            while bottom > lower && top.below(h - bottom) == block {
                bottom -= 1;
            }
            let (ya, yb) = (self.y(bottom), self.y(layer));
            let mut quad = Shape::new();
            quad.quad(
                [corner(a, ya), corner(b, ya), corner(b, yb), corner(a, yb)],
                block.colour(),
            );
            shape.append(&quad.with_seed(block.kind()));
            layer = bottom;
        }
    }
}

fn step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The landscape's height above the plain: a flat clearing round the chip with a pond in it, rolling hills with
/// lakes in the hollows, and mountain ranges closing the horizon.
fn relief(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let open = step(10.0, 18.0, r);
    let hills = (fbm_scalar(x * 0.03 + 11.0, z * 0.03 + 4.0, 4) - 0.5) * 22.0;
    let ranges =
        step(34.0, 62.0, r) * (5.0 + 16.0 * fbm_scalar(x * 0.045 + 3.0, z * 0.045 + 8.0, 3));
    let pond = (Vec3::new(x - 8.0, 0.0, z + 11.0).length() - 3.0) / 3.5;
    open * (hills + 1.5) + ranges - 5.5 * (1.0 - pond.clamp(0.0, 1.0))
}

fn column_at(x: f32, z: f32) -> (i32, Block) {
    let y = relief(x, z);
    if y < WATER {
        return ((WATER / BLOCK).round() as i32, Block::Water);
    }
    let top = if y > SNOW {
        Block::Snow
    } else if y > STONE {
        Block::Stone
    } else if y < WATER + 1.3 {
        Block::Sand
    } else {
        Block::Grass
    };
    ((y / BLOCK).round() as i32, top)
}

fn landscape() -> HeightField {
    let half = CELLS as f32 * BLOCK * 0.5;
    HeightField::new(CELLS, BLOCK, (-half, -half), FLOOR, |i, j| {
        column_at(
            -half + (i as f32 + 0.5) * BLOCK,
            -half + (j as f32 + 0.5) * BLOCK,
        )
    })
}

/// A voxel chip in the chip's own space: package, die, core, bond traces and pins.
fn voxel_chip() -> HeightField {
    let size = 2.0 * HALF / CHIP_CELLS as f32;
    let ring = |c: usize| !(2..CHIP_CELLS - 2).contains(&c);
    HeightField::new(CHIP_CELLS, size, (-HALF, -HALF), 0.0, |i, j| {
        let on_die = (8..24).contains(&i) && (8..24).contains(&j);
        let on_core = (12..20).contains(&i) && (12..20).contains(&j);
        let cell = hash2_scalar(i as f32 * 1.7, j as f32 * 2.3);
        if on_core {
            return (2, if cell < 0.3 { Block::Die } else { Block::Core });
        }
        if on_die {
            return (1, if cell < 0.18 { Block::Core } else { Block::Die });
        }
        if (3..6).contains(&i) && (3..6).contains(&j) {
            return (1, Block::Marker);
        }
        if (ring(i) && j % 2 == 0 && !ring(j)) || (ring(j) && i % 2 == 0 && !ring(i)) {
            return (1, Block::Gold);
        }
        let trace = [10, 14, 17, 21].iter().any(|&line| {
            (i == line && !(8..24).contains(&j) && !ring(j))
                || (j == line && !(8..24).contains(&i) && !ring(i))
        });
        (0, if trace { Block::Gold } else { Block::Package })
    })
}

/// Cube trees on the grass, a trunk of four blocks under a crown of leaves, and blocky flowers.
fn scenery(field: &HeightField) -> Shape {
    let mut shape = Shape::new();
    let half = CELLS as f32 * BLOCK * 0.5;
    for j in 0..CELLS {
        for i in 0..CELLS {
            let (h, top) = field.tops[j * CELLS + i];
            let (x, z) = (
                -half + (i as f32 + 0.5) * BLOCK,
                -half + (j as f32 + 0.5) * BLOCK,
            );
            let r2 = x * x + z * z;
            let bloom = hash2_scalar(i as f32 * 1.13, j as f32 * 0.57);
            if top == Block::Grass
                && (30.0..900.0).contains(&r2)
                && bloom < 0.035 * step(0.4, 0.6, value_scalar(x * 0.2 + 5.0, z * 0.2)) + 0.004
            {
                let petals = [
                    Vec3::new(0.95, 0.15, 0.15),
                    Vec3::new(1.0, 0.85, 0.15),
                    Vec3::new(0.95, 0.95, 0.95),
                    Vec3::new(0.3, 0.45, 1.0),
                ];
                let colour = petals[(bloom * 4000.0) as usize % petals.len()];
                let base = field.y(h);
                let w = BLOCK * 0.1;
                shape.append(&Shape::cuboid(
                    Vec3::new(x - w, base, z - w),
                    Vec3::new(x + w, base + BLOCK * 0.5, z + w),
                    Block::Leaves.colour(),
                ));
                shape.append(&Shape::cuboid(
                    Vec3::new(x - 2.0 * w, base + BLOCK * 0.5, z - 2.0 * w),
                    Vec3::new(x + 2.0 * w, base + BLOCK * 0.5 + 4.0 * w, z + 2.0 * w),
                    colour,
                ));
            }
            let forest = step(0.42, 0.58, value_scalar(x * 0.06 + 2.0, z * 0.06));
            if top != Block::Grass
                || x * x + z * z < 400.0
                || (x.abs() < 8.0 && z > -40.0)
                || hash2_scalar(i as f32 * 0.73, j as f32 * 1.91) > 0.05 * forest
            {
                continue;
            }
            let base = field.y(h);
            let (lo, hi) = (
                Vec3::new(x - BLOCK * 0.5, base, z - BLOCK * 0.5),
                Vec3::new(x + BLOCK * 0.5, base + 4.0 * BLOCK, z + BLOCK * 0.5),
            );
            shape.append(&Shape::cuboid(lo, hi, Block::Wood.colour()));
            let crown = |reach: f32, bottom: f32, top: f32| {
                Shape::cuboid(
                    Vec3::new(x - reach * BLOCK, base + bottom * BLOCK, z - reach * BLOCK),
                    Vec3::new(x + reach * BLOCK, base + top * BLOCK, z + reach * BLOCK),
                    Block::Leaves.colour(),
                )
            };
            shape.append(&crown(1.5, 3.0, 5.0));
            shape.append(&crown(0.5, 5.0, 6.0));
        }
    }
    shape
}

/// The blocky animals at `time`: pigs and sheep pacing circles on the plain round the chip.
fn animals(time: f32) -> Shape {
    let mut shape = Shape::new();
    for k in 0..5 {
        let radius = 4.6 + 0.85 * k as f32;
        let speed = (0.22 + 0.04 * (k % 3) as f32) * if k % 2 == 0 { 1.0 } else { -1.0 };
        let angle = hash2_scalar(k as f32, 9.0) * TAU + speed * time / radius * 2.2;
        let direction = if speed > 0.0 { 1.0 } else { -1.0 };
        let position = Vec3::new(radius * angle.cos(), FLOOR, radius * angle.sin());
        let heading = (-direction * angle.cos()).atan2(-direction * angle.sin());
        let stride = speed.abs() * time * 9.0 + k as f32;
        let placed = Mat4::translation_vec(position)
            * Mat4::rotate_y(heading)
            * Mat4::scaling(1.25, 1.25, 1.25);
        shape.append(&animal(k % 2 == 0, stride).transformed(&placed));
    }
    shape
}

/// A pig or a sheep facing +x with its feet on y = 0, its legs swinging with `stride`.
fn animal(pig: bool, stride: f32) -> Shape {
    let (body, head, leg) = if pig {
        (
            Vec3::new(0.95, 0.6, 0.62),
            Vec3::new(0.95, 0.62, 0.64),
            Vec3::new(0.9, 0.55, 0.58),
        )
    } else {
        (
            Vec3::new(0.93, 0.93, 0.9),
            Vec3::new(0.2, 0.18, 0.17),
            Vec3::new(0.2, 0.18, 0.17),
        )
    };
    let mut shape = Shape::new();
    shape.append(&Shape::cuboid(
        Vec3::new(-0.42, 0.28, -0.26),
        Vec3::new(0.42, 0.78, 0.26),
        body,
    ));
    shape.append(&Shape::cuboid(
        Vec3::new(0.38, 0.42, -0.2),
        Vec3::new(0.8, 0.84, 0.2),
        head,
    ));
    if pig {
        shape.append(&Shape::cuboid(
            Vec3::new(0.8, 0.5, -0.1),
            Vec3::new(0.9, 0.66, 0.1),
            Vec3::new(1.0, 0.74, 0.76),
        ));
    }
    for (n, (x, z)) in [(0.3, 0.15), (0.3, -0.15), (-0.3, 0.15), (-0.3, -0.15)]
        .into_iter()
        .enumerate()
    {
        let swing = 0.55
            * (stride
                + if n == 1 || n == 2 {
                    std::f32::consts::PI
                } else {
                    0.0
                })
            .sin();
        let hip = Vec3::new(x, 0.3, z);
        let limb = Shape::cuboid(
            Vec3::new(x - 0.07, 0.0, z - 0.07),
            Vec3::new(x + 0.07, 0.3, z + 0.07),
            leg,
        );
        shape.append(&limb.transformed(
            &(Mat4::translation_vec(hip) * Mat4::rotate_z(swing) * Mat4::translation_vec(-hip)),
        ));
    }
    shape
}

/// Cube clouds drifting across the sky: a few merged slabs each, wrapping round the map.
fn clouds(time: f32) -> Shape {
    const CUBE: f32 = 3.0;
    const SPAN: f32 = 150.0;
    let mut shape = Shape::new();
    for k in 0..CLOUDS {
        let h = |n: f32| hash2_scalar(k as f32 * 3.7 + n * 1.3, n * 5.1 + k as f32);
        let x = (h(0.0) * SPAN + time * 1.4).rem_euclid(SPAN) - SPAN * 0.5;
        let (z, y) = (-110.0 + h(1.0) * 130.0, 26.0 + h(2.0) * 12.0);
        for m in 0..3 {
            let (w, d) = (
                (2.0 + (h(3.0 + m as f32) * 3.0).floor()) * CUBE,
                (1.0 + (h(6.0 + m as f32) * 2.0).floor()) * CUBE,
            );
            let (ox, oz) = (
                (h(9.0 + m as f32) - 0.5) * 2.0 * CUBE,
                (m as f32 - 1.0) * CUBE,
            );
            shape.append(&Shape::cuboid(
                Vec3::new(x + ox, y, z + oz),
                Vec3::new(x + ox + w, y + CUBE, z + oz + d),
                Block::Cloud.colour(),
            ));
        }
    }
    shape
}

fn fog() -> Fog {
    Fog {
        colour: Vec3::new(0.58, 0.8, 1.0),
        density: 0.0045,
        falloff: 0.02,
        base: FLOOR,
    }
}

/// The sun's direction `time` seconds into the world: an arc across the shot.
fn sun(time: f32) -> Vec3 {
    let azimuth = 3.9 + 0.12 * time;
    let elevation: f32 = 0.62;
    Vec3::new(
        azimuth.cos() * elevation.cos(),
        elevation.sin(),
        azimuth.sin() * elevation.cos(),
    )
}

fn sky(sun: Vec3) -> Sky {
    Sky {
        zenith: Vec3::new(0.08, 0.32, 0.95),
        horizon: Vec3::new(0.55, 0.78, 1.0),
        sun,
        sun_colour: Vec3::new(2.0, 1.9, 1.5),
        sun_size: 0.05,
        glow: 0.6,
        clouds: None,
        stars: 0.0,
    }
}

#[derive(Clone)]
struct BlockUniforms {
    view: ViewRef,
    sun: Vec3,
    /// The edge of the blocks being drawn, which sets the texture's scale.
    block: f32,
}

/// Block faces: a flat tint per face direction, a 16x16 texel pattern and per-block variation
/// that fade out with distance, grass lips, moving water, the sun's shadow of the chip, and fog.
struct BlockShader;

impl FragmentShaderSimd<KitVaryings> for BlockShader {
    type Uniforms = BlockUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &BlockUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let n = facing(v.normal().normalise(), to_eye);
        let kind = v.seed();
        let block = s(u.block);

        let p = (world - n * (block * s(0.25))) * (s(1.0) / block);
        let (cx, cy, cz) = (p.x.floor(), p.y.floor(), p.z.floor());
        let (fx, fy, fz) = (p.x - cx, p.y - cy, p.z - cz);
        let (tx, ty, tz) = (
            (fx * s(16.0)).floor(),
            (fy * s(16.0)).floor(),
            (fz * s(16.0)).floor(),
        );
        let cell = hash2(cx * s(3.1) + cz * s(17.3), cy * s(5.7) + s(0.5));
        let texel = hash2(
            cx * s(7.0) + tx * s(1.37) + cz * s(13.0),
            cy * s(11.0) + ty * s(1.71) + tz * s(2.9),
        );

        let footprint = distance * s(view.pixel_angle);
        let texel_fade = s(1.0) - smoothstep(0.4, 1.0, footprint / (block * s(1.0 / 16.0)));
        let cell_fade = s(1.0) - smoothstep(0.5, 1.5, footprint / block);
        let variation = s(1.0)
            + (texel - s(0.5)) * s(0.22) * texel_fade
            + (cell - s(0.5)) * s(0.14) * cell_fade;

        let top = n.y.simd_gt(s(0.5));
        let is_grass = kind.simd_gt(s(0.1)) & kind.simd_lt(s(0.3));
        let is_water = kind.simd_gt(s(0.3));
        let lip_edge =
            s(0.68) - hash2(cx * s(5.0) + tx * s(1.0) + tz * s(1.0), s(2.0)) * s(0.12) * texel_fade;
        let lip = is_grass & !top & fy.simd_gt(lip_edge);
        let dirt = rgb(0.42, 0.29, 0.18);
        let patch = hash2((p.x * s(0.25)).floor(), (p.z * s(0.25)).floor());
        let patched = v.colour() * select(is_grass & top, s(0.88) + patch * s(0.24), s(1.0));
        let mut albedo = select3(is_grass & !top & !lip, dirt, patched);

        let t = s(view.frame.time);
        let ripple = value(world.x * s(1.6) + t * s(0.7), world.z * s(1.6) - t * s(0.5));
        let glint = smoothstep(
            0.82,
            0.9,
            value(world.x * s(5.0) + t * s(1.3), world.z * s(5.0) - t * s(0.9)),
        );
        let water = albedo * (s(0.85) + ripple * s(0.3))
            + splat3(Vec3::new(0.5, 0.55, 0.6)) * (glint * s(0.35));
        albedo = select3(is_water, water, albedo);

        let tint = n.y.fast_max(s(0.0))
            + n.y.fast_min(s(0.0)).abs() * s(0.5)
            + n.x.abs() * s(0.62)
            + n.z.abs() * s(0.8);
        let shadow = chip_shadow(view, u.sun, world) * select(top, s(1.0), s(0.0));
        let light = tint * (s(1.0) - shadow * s(0.45)) * s(1.12);
        let colour = albedo * (variation * light);
        let colour = fog().apply(colour, world, distance);
        scene_colour(view, colour, world)
    }
}

pub struct VoxelWorld {
    landscape: Shape,
    scenery: Shape,
    chip: Shape,
    animals: Shape,
    clouds: Shape,
    blocks: SimdPipeline<KitVertexShader, BlockShader>,
    sky: SimdPipeline<ScreenVertexShader, SkyShader>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        let field = landscape();
        Self {
            landscape: field.mesh(),
            scenery: scenery(&field),
            chip: voxel_chip().mesh(),
            animals: Shape::new(),
            clouds: Shape::new(),
            blocks: pipeline(BlockShader, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for VoxelWorld {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Marimba],
            ambience: vec![Ambience::bed(Source::Synth(Accent::BlockPops), 0.1, 0.25)],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.animals = animals(scene.local_time);
        self.clouds = clouds(scene.local_time);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let sun = sun(scene.local_time);
        let uniforms = |block| BlockUniforms {
            view: view.clone(),
            sun,
            block,
        };
        let still = KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };

        self.landscape
            .draw(pass, &self.blocks, still.clone(), uniforms(BLOCK));
        self.scenery
            .draw(pass, &self.blocks, still.clone(), uniforms(BLOCK));
        self.animals
            .draw(pass, &self.blocks, still.clone(), uniforms(BLOCK / 2.0));
        self.clouds
            .draw(pass, &self.blocks, still, uniforms(BLOCK * 4.0));
        self.chip.draw(
            pass,
            &self.blocks,
            KitUniforms {
                view: view.clone(),
                motion: Motion::OnChip,
            },
            uniforms(2.0 * HALF / CHIP_CELLS as f32),
        );

        let sky = SkyUniforms {
            view: view.clone(),
            sky: sky(sun),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| sky.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.25,
            vignette: 0.25,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
