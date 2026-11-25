//! A neon city street at night: the chip stands on a rain-wet road between towers of lit windows,
//! blade signs and neon trim, light-trail traffic streaming past, the whole street mirrored in
//! the wet tarmac, rain and haze in the air.
//!
//! The reflection is real geometry: the towers, signs and traffic are drawn a second time flipped
//! under the road, and the road is blended over them by how mirror-like the tarmac is there.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::chip::{HALF, Look};
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, fract, hash2, hash2_scalar, length, mix3, pow2n, rgb, s, select, smoothstep,
    splat3, to_colour,
};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonShader, ToonUniforms,
    pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::noise::{fbm, value};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::{Scene, Stage};

/// The height of the road where the chip stands.
const FLOOR: f32 = -1.95;
/// Where the towers' faces stand, either side of the street's centre line.
const FACADE: f32 = 9.5;
/// The width of the tarmac; pavement runs from there to the facades.
const TARMAC: f32 = 8.0;
/// The towers' spacing along the street, which runs away from the camera along -z.
const PITCH: f32 = 9.0;
const TOWERS: usize = 26;
/// Where the first tower starts, and the far end of the street.
const NEAR: f32 = 13.0;
const FAR: f32 = NEAR - PITCH * TOWERS as f32;
const LANES: [f32; 4] = [-5.4, -3.2, 3.2, 5.4];

const MAGENTA: Vec3 = Vec3 {
    x: 1.0,
    y: 0.1,
    z: 0.55,
};
const CYAN: Vec3 = Vec3 {
    x: 0.08,
    y: 0.85,
    z: 1.0,
};
const AMBER: Vec3 = Vec3 {
    x: 1.0,
    y: 0.5,
    z: 0.1,
};
const VIOLET: Vec3 = Vec3 {
    x: 0.5,
    y: 0.22,
    z: 1.0,
};
/// The neon colours signs, shopfronts and windows are drawn from: mostly magenta and cyan.
const NEON: [Vec3; 6] = [MAGENTA, CYAN, MAGENTA, CYAN, AMBER, VIOLET];

fn haze() -> Vec3 {
    Vec3::new(0.2, 0.05, 0.19)
}

fn fog() -> Fog {
    Fog {
        colour: haze(),
        density: 0.0105,
        falloff: 0.012,
        base: FLOOR,
    }
}

fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(0.3, 0.45, 0.84).normalise(),
        sun_colour: Vec3::new(0.18, 0.1, 0.3),
        sky: Vec3::new(0.05, 0.06, 0.16),
        ground: Vec3::new(0.12, 0.03, 0.09),
        bands: 0.0,
        rim: Vec3::new(0.35, 0.08, 0.3),
        fog: fog(),
    }
}

fn look() -> Look {
    Look {
        substrate: Vec3::new(0.07, 0.08, 0.11),
        die: Vec3::new(0.004, 0.012, 0.03),
        gold: Vec3::new(0.9, 0.1, 0.5),
        streams: Vec3::new(0.1, 0.9, 1.0),
        key: Vec3::new(-0.5, 0.35, 0.8).normalise(),
        key_colour: Vec3::new(1.0, 0.25, 0.8),
        sky: Vec3::new(0.03, 0.04, 0.12),
        ground: Vec3::new(0.12, 0.03, 0.1),
        softbox: Vec3::new(0.05, 0.15, 0.25),
    }
}

/// One of the neon colours for `t` in [0, 1).
#[inline(always)]
fn neon(t: F) -> Vec3Simd {
    let mut colour = splat3(NEON[0]);
    for (i, c) in NEON.iter().enumerate().skip(1) {
        colour = crate::simd::select3(
            crate::simd::ge(t, s(i as f32 / NEON.len() as f32)),
            splat3(*c),
            colour,
        );
    }
    colour
}

fn neon_scalar(t: f32) -> Vec3 {
    NEON[((t * NEON.len() as f32) as usize).min(NEON.len() - 1)]
}

/// A box with no underside: a tower's footprint is never seen, and flipped under the road it
/// would lie in the road's plane.
fn block(min: Vec3, max: Vec3, colour: Vec3) -> Shape {
    let corner = |i: u8| {
        Vec3::new(
            if i & 1 != 0 { max.x } else { min.x },
            if i & 2 != 0 { max.y } else { min.y },
            if i & 4 != 0 { max.z } else { min.z },
        )
    };
    let mut shape = Shape::new();
    for face in [
        [4, 5, 7, 6],
        [1, 0, 2, 3],
        [5, 1, 3, 7],
        [0, 4, 6, 2],
        [6, 7, 3, 2],
    ] {
        shape.quad(face.map(corner), colour);
    }
    shape
}

fn spanning(a: f32, b: f32) -> (f32, f32) {
    (a.min(b), a.max(b))
}

/// Everything the street is made of, as separate meshes because each takes its own shader.
struct City {
    /// Towers, bridges: lit-window facades.
    towers: Shape,
    /// Steady and flickering neon: tubes along the towers, trim and beacons.
    tubes: Shape,
    /// Blade signs' glowing faces and fascia boards.
    faces: Shape,
    /// Dark metal: sign frames, brackets, masts, rooftop plant.
    metal: Shape,
    beacons: Shape,
}

impl City {
    fn build(rng: &mut Rng) -> Self {
        let mut city = Self {
            towers: Shape::new(),
            tubes: Shape::new(),
            faces: Shape::new(),
            metal: Shape::new(),
            beacons: Shape::new(),
        };
        for k in 0..TOWERS {
            for side in [-1.0_f32, 1.0] {
                city.tower(rng, k, side);
            }
        }
        city.bridge(-33.0, 15.0);
        city.bridge(-79.0, 19.5);
        city
    }

    fn tower(&mut self, rng: &mut Rng, k: usize, side: f32) {
        let key = k as f32 + 0.5;
        let hash = |n: f32| hash2_scalar(key * 1.37 + n * 11.1, side * 3.7 + n * 5.3);
        let seed = hash(0.0);
        let z_hi = NEAR - PITCH * k as f32 - 0.35;
        let z_lo = NEAR - PITCH * (k + 1) as f32 + 0.35;
        let recess = k >= 3 && hash(1.0) < 0.1;
        let face = FACADE + if recess { 6.5 } else { 0.0 };
        let depth = 15.0;
        let hue = neon_scalar(fract_scalar(seed * 7.13));

        let mut top = 12.0 + 26.0 * hash(2.0) + k as f32 * 0.35;
        let mut inset = 0.0;
        let tiers = if hash(3.0) < 0.3 { 2 } else { 3 };
        for tier in 0..tiers {
            let (x0, x1) = spanning(side * (face + inset), side * (face + depth));
            let (z0, z1) = (z_lo + inset * 0.5, z_hi - inset * 0.5);
            let body = Vec3::new(0.02, 0.022, 0.04);
            self.towers.append(
                &block(
                    Vec3::new(x0, FLOOR, z0),
                    Vec3::new(x1, FLOOR + top, z1),
                    body,
                )
                .with_seed(seed),
            );
            if tier == 0 || hash(4.0 + tier as f32) < 0.6 {
                let x = side * (face + inset - 0.1);
                let (tx0, tx1) = spanning(x, x - side * 0.14);
                let tube = hue * 2.6;
                let band = FLOOR + top - 0.25;
                self.tubes.append(
                    &Shape::cuboid(
                        Vec3::new(tx0, band, z0 + 0.1),
                        Vec3::new(tx1, band + 0.14, z1 - 0.1),
                        tube,
                    )
                    .with_seed(0.1),
                );
                if hash(9.0 + tier as f32) < 0.55 {
                    for z in [z0 + 0.12, z1 - 0.12] {
                        self.tubes.append(
                            &Shape::cuboid(
                                Vec3::new(tx0, FLOOR + 0.3, z - 0.07),
                                Vec3::new(tx1, FLOOR + top - 0.3, z + 0.07),
                                hue * 1.8,
                            )
                            .with_seed(0.1 + 0.9 * hash(12.0)),
                        );
                    }
                }
            }
            inset += 1.8;
            top += 6.0 + 28.0 * hash(5.0 + tier as f32);
        }
        let roof = FLOOR + top - (6.0 + 28.0 * hash(5.0 + (tiers - 1) as f32));
        let (rx, rz) = (side * (face + inset + 3.0), (z_hi + z_lo) * 0.5);
        self.roof(rng, Vec3::new(rx, roof, rz), seed);

        if k < TOWERS - 3 {
            self.shopfront(rng, side, face, z_lo, z_hi, seed);
            let count = if k < 8 { 2 } else { 1 };
            for _ in 0..count {
                if hash(20.0) < 0.85 && !recess {
                    let z = z_lo + 1.0 + rng.range(0.0, PITCH - 3.0);
                    self.blade(rng, side, z);
                }
            }
        }
    }

    fn roof(&mut self, rng: &mut Rng, at: Vec3, seed: f32) {
        let mast = rng.range(4.0, 12.0);
        self.metal.append(&Shape::cuboid(
            at + Vec3::new(-0.08, 0.0, -0.08),
            at + Vec3::new(0.08, mast, 0.08),
            Vec3::new(0.05, 0.05, 0.07),
        ));
        for _ in 0..3 {
            let (w, d, h) = (
                rng.range(0.8, 1.8),
                rng.range(0.8, 1.8),
                rng.range(0.5, 1.4),
            );
            let o = Vec3::new(rng.range(-2.5, 2.5), 0.0, rng.range(-2.5, 2.5));
            self.metal.append(&Shape::cuboid(
                at + o,
                at + o + Vec3::new(w, h, d),
                Vec3::new(0.04, 0.045, 0.06),
            ));
        }
        let blink = 0.2 + 0.8 * seed;
        let tip = at + Vec3::new(0.0, mast, 0.0);
        self.beacons.append(
            &Shape::cuboid(
                tip - Vec3::new(0.14, 0.0, 0.14),
                tip + Vec3::new(0.14, 0.3, 0.14),
                Vec3::new(6.0, 0.2, 0.2),
            )
            .with_seed(blink),
        );
    }

    fn shopfront(&mut self, rng: &mut Rng, side: f32, face: f32, z_lo: f32, z_hi: f32, seed: f32) {
        let x = side * (face - 0.18);
        let (x0, x1) = spanning(x, x + side * 0.18);
        let colour = neon_scalar(rng.unit());
        let y = FLOOR + 4.3;
        self.faces.append(
            &Shape::cuboid(
                Vec3::new(x0, y, z_lo + 0.4),
                Vec3::new(x1, y + 0.8, z_hi - 0.4),
                colour * 1.8,
            )
            .with_seed(0.4 + 0.6 * seed),
        );
        self.metal.append(&Shape::cuboid(
            Vec3::new(x0 - side * 0.04, y - 0.12, z_lo + 0.3),
            Vec3::new(x1, y + 0.92, z_hi - 0.3),
            Vec3::new(0.03, 0.03, 0.05),
        ));
    }

    fn blade(&mut self, rng: &mut Rng, side: f32, z: f32) {
        let width = rng.range(0.9, 1.5);
        let height = rng.range(2.4, 5.5);
        let reach = rng.range(1.3, 2.0);
        let y = FLOOR + rng.range(5.6, 12.0);
        let x = side * (FACADE - reach);
        let colour = neon_scalar(rng.unit());
        let (x0, x1) = (x - width * 0.5, x + width * 0.5);
        let seed = rng.unit();
        let front = [
            Vec3::new(x0, y, z + 0.06),
            Vec3::new(x1, y, z + 0.06),
            Vec3::new(x1, y + height, z + 0.06),
            Vec3::new(x0, y + height, z + 0.06),
        ];
        let back = [front[1], front[0], front[3], front[2]].map(|p| Vec3::new(p.x, p.y, z - 0.06));
        let mut face = Shape::new();
        face.quad(front, colour * 2.2).quad(back, colour * 2.2);
        self.faces.append(&face.with_seed(seed));
        self.metal.append(&Shape::cuboid(
            Vec3::new(x0 - 0.07, y - 0.07, z - 0.05),
            Vec3::new(x1 + 0.07, y + height + 0.07, z + 0.05),
            Vec3::new(0.03, 0.03, 0.05),
        ));
        let (a0, a1) = spanning(side * FACADE, x);
        self.metal.append(&Shape::cuboid(
            Vec3::new(a0, y + height * 0.5 - 0.05, z - 0.04),
            Vec3::new(a1, y + height * 0.5 + 0.05, z + 0.04),
            Vec3::new(0.03, 0.03, 0.05),
        ));
    }

    fn bridge(&mut self, z: f32, height: f32) {
        let seed = hash2_scalar(z, 4.4);
        self.towers.append(
            &block(
                Vec3::new(-FACADE - 0.3, FLOOR + height, z - 2.4),
                Vec3::new(FACADE + 0.3, FLOOR + height + 3.6, z + 2.4),
                Vec3::new(0.025, 0.027, 0.045),
            )
            .with_seed(seed),
        );
        let hue = neon_scalar(fract_scalar(seed * 7.13));
        for (y, z_offset) in [(height - 0.12, 2.5), (height + 3.7, 2.5)] {
            self.tubes.append(
                &Shape::cuboid(
                    Vec3::new(-FACADE, FLOOR + y, z - z_offset),
                    Vec3::new(FACADE, FLOOR + y + 0.13, z + z_offset),
                    hue * 2.4,
                )
                .with_seed(0.1),
            );
        }
    }
}

fn fract_scalar(x: f32) -> f32 {
    x - x.floor()
}

#[derive(Clone)]
struct SurfaceUniforms {
    view: ViewRef,
    fog: Fog,
    /// 1 for the street, -1 for its reflection, whose shapes are flipped under the road.
    mirror: f32,
}

impl SurfaceUniforms {
    /// The height above the road of the real surface at `world`, which may be a reflection.
    #[inline(always)]
    fn height(&self, world: Vec3Simd) -> F {
        (world.y - s(FLOOR)) * s(self.mirror)
    }

    #[inline(always)]
    fn finish(&self, colour: Vec3Simd, world: Vec3Simd) -> ColourSimd {
        let distance = length(world - splat3(self.view.eye));
        let real = Vec3Simd::new(world.x, s(FLOOR) + self.height(world), world.z);
        scene_colour(&self.view, self.fog.apply(colour, real, distance), world)
    }
}

/// A rectangle's mask along one axis of a cell: 1 inside `half` of the cell's middle, with a soft
/// edge `soft` wide.
#[inline(always)]
fn span(f: F, half: f32, soft: F) -> F {
    clamp01((s(half) - (f - s(0.5)).abs()) / soft + s(0.5))
}

/// The lit windows of a facade: a grid of cells, some lit in the tower's colour, warm or cool
/// white, a few switching on and off.
#[inline(always)]
fn windows(across: F, height: F, seed: F, time: f32, footprint: F) -> Vec3Simd {
    const WIDTH: f32 = 1.1;
    const STOREY: f32 = 1.5;
    let (gx, gy) = (across / s(WIDTH), height / s(STOREY));
    let (ix, iy) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - ix, gy - iy);
    let h = hash2(ix + seed * s(131.0), iy + seed * s(57.0));
    let h2 = hash2(ix * s(1.7) + seed * s(31.0), iy * s(2.3) + s(5.0));
    let h3 = hash2(ix * s(3.3) + seed * s(5.1), iy * s(1.9) + s(2.0));
    let toggle = (s(time * 0.3) + h * s(9.0)).floor();
    let toggled = hash2(ix + toggle, iy + seed * s(11.0)).simd_lt(s(0.5));
    let density = s(0.1) + hash2(seed * s(7.7), s(1.3)) * s(0.3);
    let steady = select(h.simd_lt(density), s(1.0), s(0.0));
    let lit = select(h2.simd_lt(s(0.14)), select(toggled, s(1.0), s(0.0)), steady);
    let aa = clamp01(footprint / s(WIDTH));
    let soft = s(0.04) + aa * s(0.5);
    let rect = span(fx, 0.36, soft) * span(fy, 0.3, soft * s(WIDTH / STOREY));
    let tower = neon(fract(seed * s(7.13)));
    let warm = rgb(1.0, 0.75, 0.45);
    let cool = rgb(0.7, 0.88, 1.0);
    let tint = crate::simd::select3(
        h3.simd_lt(s(0.3)),
        tower,
        crate::simd::select3(h3.simd_lt(s(0.78)), warm, cool),
    );
    let glow = tint * ((s(0.35) + h) * s(0.9));
    let detail = glow * (rect * lit);
    let mean = tint * (density * s(0.2));
    mix3(detail, mean, smoothstep(0.3, 0.8, aa))
}

/// Shopfronts along the first storeys: wide panels of coloured light with mullions.
#[inline(always)]
fn shopfronts(across: F, height: F, seed: F, footprint: F) -> Vec3Simd {
    const WIDTH: f32 = 5.0;
    let cell = across / s(WIDTH);
    let (index, f) = (cell.floor(), fract(cell));
    let h = hash2(index * s(1.3) + seed * s(17.0), s(3.0));
    let h2 = hash2(index * s(2.1) + seed * s(9.0), s(7.0));
    let aa = clamp01(footprint / s(WIDTH));
    let soft = s(0.03) + aa * s(0.5);
    let panel = span(f, 0.44, soft) * span((height - s(0.4)) / s(3.2), 0.5, soft);
    let open = select(h2.simd_lt(s(0.6)), s(1.0), s(0.1));
    let tower = neon(fract(seed * s(7.13)));
    let hue = mix3(
        tower,
        rgb(1.0, 0.72, 0.42),
        select(h.simd_lt(s(0.3)), s(0.9), s(0.0)),
    );
    let level = (s(0.25) + h2 * s(0.35)) * open;
    let glass = mix3(
        hue * level,
        hue * (s(0.18) * s(0.6)),
        smoothstep(0.4, 0.9, aa),
    );
    let fade = (s(1.0) - (height - s(1.0)) / s(3.5) * s(0.35)).fast_max(s(0.5));
    glass * (panel * fade)
}

/// Towers and bridges: dark concrete with lit windows, shopfronts below and the street's neon
/// spilling up the walls.
struct FacadeShader;

impl FragmentShaderSimd<KitVaryings> for FacadeShader {
    type Uniforms = SurfaceUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &SurfaceUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let height = u.height(world);
        let normal = v.normal().normalise();
        let seed = v.seed();
        let to_surface = world - splat3(view.eye);
        let distance = length(to_surface);
        let grazing = (normal.dot(to_surface) / distance).abs().fast_max(s(0.06));
        let footprint = distance * s(view.pixel_angle) / grazing;
        let facing_x = normal.x.abs().simd_gt(normal.z.abs());
        let across = select(facing_x, world.z, world.x);
        let roof = normal.y.abs().simd_gt(s(0.7));

        let pilaster = smoothstep(0.0, 0.12, fract(across * s(0.45)));
        let base = rgb(0.014, 0.016, 0.034) * (s(0.7) + pilaster * s(0.5));
        let spill_hue = mix3(
            splat3(MAGENTA),
            splat3(CYAN),
            value(across * s(0.06) + s(seed_offset()), s(2.0)),
        );
        let spill = spill_hue
            * ((-height / s(8.0)).exp()
                * s(0.2)
                * (s(0.6) + value(across * s(0.4), height * s(0.25)) * s(0.8)));
        let t = view.frame.time;
        let upper = windows(across, height, seed, t, footprint);
        let lower = shopfronts(across, height, seed, footprint);
        let shop = s(1.0) - smoothstep(4.0, 4.5, height);
        let glass = mix3(upper, lower, shop);
        let wall = crate::simd::select3(roof, rgb(0.02, 0.02, 0.03), base + spill + glass);
        u.finish(wall, world)
    }
}

fn seed_offset() -> f32 {
    3.1
}

#[derive(Clone, Copy)]
enum Style {
    /// Mostly on, now and then stuttering off.
    Flicker,
    /// A glowing sign: pseudo-lettering on a dim ground.
    Sign,
    /// Slow red blink.
    Beacon,
    /// A pulse running round, by the vertex's seed.
    Chase,
}

#[derive(Clone)]
struct NeonUniforms {
    surface: SurfaceUniforms,
    style: Style,
}

/// Emissive things: neon tubes, signs, beacons and the chip's trim.
struct NeonShader;

impl FragmentShaderSimd<KitVaryings> for NeonShader {
    type Uniforms = NeonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &NeonUniforms) -> ColourSimd {
        let surface = &u.surface;
        let world = v.world();
        let seed = v.seed();
        let t = s(surface.view.frame.time);
        let level = match u.style {
            Style::Flicker => {
                let slot = (t * s(9.0) + seed * s(61.0)).floor();
                let dip = hash2(slot, seed * s(13.0)).simd_gt(s(0.94));
                select(dip, s(0.15), s(1.0))
            }
            Style::Beacon => {
                let phase = fract(t * s(0.7) + seed * s(3.0));
                s(0.08)
                    + smoothstep(0.0, 0.08, phase)
                        * (s(1.0) - smoothstep(0.12, 0.4, phase))
                        * s(1.0)
            }
            Style::Chase => {
                s(0.35)
                    + s(0.65)
                        * smoothstep(
                            0.55,
                            1.0,
                            (t * s(2.2) - seed * s(TAU * 3.0)).sin() * s(0.5) + s(0.5),
                        )
            }
            Style::Sign => {
                let normal = v.normal().normalise();
                let height = surface.height(world);
                let across = select(normal.x.abs().simd_gt(normal.z.abs()), world.z, world.x);
                let (gx, gy) = (across * s(2.3), height * s(1.7));
                let (cx, cy) = (gx.floor(), gy.floor());
                let (fx, fy) = (gx - cx, gy - cy);
                let (sx, sy) = ((fx * s(3.0)).floor(), (fy * s(5.0)).floor());
                let stroke = hash2(
                    cx * s(7.1) + sx * s(1.3) + seed * s(77.0),
                    cy * s(3.7) + sy * s(2.9) + seed * s(19.0),
                );
                let margin = span(fx, 0.38, s(0.05)) * span(fy, 0.4, s(0.05));
                let bar = seed.simd_lt(s(0.25));
                let letters = select(stroke.simd_lt(s(0.55)), s(1.0), s(0.0)) * margin;
                let ink = select(bar, span(fy, 0.3, s(0.05)), letters) + s(0.1);
                let slot = (t * s(7.0) + seed * s(40.0)).floor();
                let dip = hash2(slot, seed * s(5.0)).simd_gt(s(0.95));
                ink * select(dip, s(0.2), s(1.0))
            }
        };
        surface.finish(v.colour() * level, world)
    }
}

/// The road: dark tarmac with lane markings, puddles and drizzle ripples, lit by the street's
/// neon and a glow round the chip, blended over the street's reflection by how mirror-like it is.
struct RoadShader;

impl FragmentShaderSimd<KitVaryings> for RoadShader {
    type Uniforms = SurfaceUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &SurfaceUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let (x, z) = (world.x, world.z);
        let ax = x.abs();
        let t = view.frame.time;
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let cos_view = clamp01(to_eye.y / distance);
        let fresnel = s(0.03) + s(0.97) * pow2n(s(1.0) - cos_view, 2) * (s(1.0) - cos_view);

        let puddles = fbm(x * s(0.33) + s(4.0), z * s(0.2), 3);
        let wet = s(0.2) + smoothstep(0.4, 0.55, puddles) * s(0.8);
        let tracks = (-(ax - s(3.2)).abs() * s(1.3)).exp() + (-(ax - s(5.4)).abs() * s(1.3)).exp();
        let grain = value(x * s(5.0), z * s(5.0));
        let tarmac = rgb(0.03, 0.032, 0.04) * (s(0.7) + grain * s(0.6));

        let dash = smoothstep(0.0, 0.1, fract(z * s(0.16)) - s(0.0))
            * (s(1.0) - smoothstep(0.38, 0.48, fract(z * s(0.16))));
        let centre = (s(1.0) - smoothstep(0.03, 0.06, (ax - s(0.17)).abs())) * s(1.0);
        let divider = (s(1.0) - smoothstep(0.04, 0.08, (ax - s(4.3)).abs())) * dash;
        let edge = s(1.0) - smoothstep(0.05, 0.09, (ax - s(6.0)).abs());
        let crossing_at = |c: f32| {
            let band =
                smoothstep(0.0, 0.08, z - s(c)) * (s(1.0) - smoothstep(0.0, 0.08, z - s(c + 2.6)));
            band * smoothstep(0.35, 0.45, fract(x * s(0.62)))
                * (s(1.0) - smoothstep(0.72, 0.8, fract(x * s(0.62))))
                * (s(1.0) - smoothstep(TARMAC - 0.15, TARMAC, ax))
        };
        let zebra = crossing_at(-31.0) + crossing_at(-62.0) + crossing_at(-97.0);
        let yellow = rgb(0.9, 0.62, 0.1) * centre;
        let white = rgb(0.75, 0.78, 0.82) * (divider + edge * s(0.8) + zebra);
        let paint = (centre + divider + edge * s(0.8) + zebra).fast_min(s(1.0));

        let kerb = smoothstep(TARMAC, TARMAC + 0.12, ax);
        let slab = fract(z * s(0.5)).simd_lt(s(0.04)) | fract(ax * s(0.5)).simd_lt(s(0.04));
        let pavement = rgb(0.06, 0.06, 0.075) * select(slab, s(0.55), s(1.0));
        let surface = mix3(tarmac * (s(1.0) + tracks * s(0.25)), pavement, kerb);
        let albedo = mix3(surface, yellow + white, paint * (s(1.0) - kerb) * s(0.45));

        let shop_phase = value(z * s(0.12), s(1.0));
        let shop_hue = mix3(splat3(MAGENTA), splat3(CYAN), shop_phase);
        let spill = shop_hue
            * ((-(s(FACADE) - ax).fast_max(s(0.0)) / s(3.2)).exp()
                * s(0.9)
                * (s(0.4) + value(z * s(0.5), s(8.0)) * s(0.9)));
        let (cx, cz) = (x, z);
        let pool_r = (cx * cx + cz * cz * s(0.7)).sqrt();
        let pulse = s(0.85) + s(0.15) * (s(t * 1.6)).sin();
        let pool = (splat3(CYAN) * s(0.55) + splat3(MAGENTA) * s(0.45))
            * ((-pool_r * pool_r / s(11.0)).exp() * pulse * s(1.3));
        let mut colour = crate::simd::mul3(albedo, rgb(0.25, 0.25, 0.3) + spill * s(2.0))
            + spill * s(0.12)
            + pool * s(0.35);

        let ripple = {
            let g = (x * s(1.5), z * s(1.5));
            let cell = (g.0.floor(), g.1.floor());
            let jitter = |k: f32| hash2(cell.0 + s(k), cell.1 + s(k * 2.0)) * s(0.2) - s(0.1);
            let (dx, dz) = (
                fract(g.0) - s(0.5) - jitter(3.0),
                fract(g.1) - s(0.5) - jitter(7.0),
            );
            let phase = fract(s(t * 0.8) + hash2(cell.0, cell.1));
            let radius = phase * s(0.4);
            let off = ((dx * dx + dz * dz).sqrt() - radius) * s(26.0);
            let ring = (-(off * off)).exp();
            ring * (s(1.0) - phase) * select(hash2(cell.1, cell.0).simd_lt(s(0.45)), s(1.0), s(0.0))
        };
        let near_rain = (s(1.0) - smoothstep(4.0, 30.0, distance)) * smoothstep(0.55, 0.9, wet);
        let ripple_tint = mix3(splat3(MAGENTA), splat3(CYAN), smoothstep(-2.0, 2.0, x));
        colour = colour + ripple_tint * (ripple * near_rain * s(0.55));

        let mirror =
            (fresnel * wet * s(0.97) * (s(1.0) - kerb * s(0.8)) * (s(1.0) - paint * s(0.5)))
                .fast_min(s(0.97));
        let colour = u.fog.apply(colour, world, distance);
        to_colour(colour, s(1.0) - mirror)
    }
}

/// The light-trail vehicles' vertex: each is a pair of lights dragging a streak behind them along
/// the street, wrapping round the street's length.
#[derive(Clone, Copy)]
struct StreakVertex {
    /// 0 to 1 along the trail from the head, and -1 to 1 across it.
    corner: Vec2,
    /// The lane's x, the height above the road and the signed speed along z.
    lane: Vec3,
    /// The trail's length and width, and the vehicle's phase round the loop.
    trail: Vec3,
    colour: Vec3,
}

#[derive(Interpolate, SimdInterpolate)]
struct StreakVaryings {
    corner: Vec2,
    colour: Vec3,
}

#[derive(Clone)]
struct StreakUniforms {
    view: ViewRef,
    mirror: f32,
}

struct StreakVertexShader;

impl VertexShader for StreakVertexShader {
    type Vertex = StreakVertex;
    type Uniforms = StreakUniforms;
    type Varyings = StreakVaryings;

    fn shade(&self, v: StreakVertex, u: &StreakUniforms) -> (Vec4, StreakVaryings) {
        const SPAN: f32 = 175.0;
        let view = &u.view;
        let time = view.frame.time;
        let direction = v.lane.z.signum();
        let z = 16.0 - (v.trail.z * SPAN + (-v.lane.z) * time).rem_euclid(SPAN);
        let height = FLOOR + v.lane.y;
        let head = Vec3::new(v.lane.x, height, z);
        let back = Vec3::new(0.0, 0.0, -direction);
        let centre = head + back * (v.corner.x * v.trail.x);
        let look = centre - view.eye;
        let across = Vec3::new(0.0, 0.0, 1.0).cross(look).normalise();
        let mut position = centre + across * (v.corner.y * v.trail.y);
        position.y = FLOOR + (position.y - FLOOR) * u.mirror;
        let far = ((z - FAR * 0.55) / 40.0).clamp(0.0, 1.0);
        let near = 1.0 - ((z + 6.0) / 8.0).clamp(0.0, 1.0);
        (
            view.view_projection * position.to_point4(),
            StreakVaryings {
                corner: v.corner,
                colour: v.colour * (far * near),
            },
        )
    }
}

struct StreakShader;

impl FragmentShaderSimd<StreakVaryings> for StreakShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: StreakVaryingsSimd, _: &()) -> ColourSimd {
        let (along, across) = (v.corner_x(), v.corner_y());
        let tail = (s(1.0) - along).fast_max(s(0.0));
        let body = tail * tail * (-across * across * s(3.0)).exp();
        let core = (-along * s(14.0)).exp() * (-across * across * s(1.2)).exp() * s(2.0);
        to_colour(v.colour() * (body + core), s(0.0))
    }
}

impl StreakVaryingsSimd {
    #[inline(always)]
    fn corner_x(&self) -> F {
        self.corner[0]
    }

    #[inline(always)]
    fn corner_y(&self) -> F {
        self.corner[1]
    }

    #[inline(always)]
    fn colour(&self) -> Vec3Simd {
        Vec3Simd::new(self.colour[0], self.colour[1], self.colour[2])
    }
}

/// Builds the traffic: pairs of lights in the four lanes (headlights towards the camera on the
/// left, tail lights away on the right) and a few craft flying low between the towers.
fn traffic(rng: &mut Rng) -> (Vec<StreakVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut add = |lane: f32,
                   height: f32,
                   speed: f32,
                   length: f32,
                   width: f32,
                   colour: Vec3,
                   rng: &mut Rng| {
        let phase = rng.unit();
        for side in [-0.75, 0.75] {
            let base = vertices.len() as u32;
            for corner in [(0.0, -1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 1.0)] {
                vertices.push(StreakVertex {
                    corner: Vec2::new(corner.0, corner.1),
                    lane: Vec3::new(lane + side, height, speed),
                    trail: Vec3::new(length, width, phase),
                    colour,
                });
            }
            indices.extend([0, 1, 2, 0, 2, 3].map(|k| base + k));
        }
    };
    for lane in LANES {
        let towards = lane < 0.0;
        for _ in 0..9 {
            let speed = rng.range(13.0, 34.0) * if towards { 1.0 } else { -1.0 };
            let colour = if towards {
                Vec3::new(3.0, 2.7, 2.2)
            } else {
                Vec3::new(3.4, 0.25, 0.3)
            };
            let length = rng.range(6.0, 16.0);
            add(lane, rng.range(0.35, 0.6), speed, length, 0.1, colour, rng);
        }
    }
    for _ in 0..7 {
        let lane = rng.range(-4.5, 4.5);
        let towards = rng.unit() < 0.5;
        let speed = rng.range(8.0, 18.0) * if towards { 1.0 } else { -1.0 };
        let colour = if rng.unit() < 0.5 {
            MAGENTA * 2.4
        } else {
            CYAN * 2.4
        };
        add(
            lane,
            rng.range(5.0, 13.0),
            speed,
            rng.range(8.0, 20.0),
            0.14,
            colour,
            rng,
        );
    }
    (vertices, indices)
}

/// A mirror-flipped copy of a shape, under the road.
fn reflected(shape: &Shape) -> Shape {
    shape.transformed(&(Mat4::translation(0.0, 2.0 * FLOOR, 0.0) * Mat4::scaling(1.0, -1.0, 1.0)))
}

/// Chip trim in the chip's own space: a cyan line round the package and magenta leads along each
/// side.
fn chip_trim() -> Shape {
    let mut trim = Shape::new();
    let inner = HALF - 0.07;
    let line = CYAN * 3.6;
    for (a, b) in [
        (
            Vec3::new(-inner, -0.03, -inner),
            Vec3::new(inner, 0.02, -inner + 0.045),
        ),
        (
            Vec3::new(-inner, -0.03, inner - 0.045),
            Vec3::new(inner, 0.02, inner),
        ),
        (
            Vec3::new(-inner, -0.03, -inner),
            Vec3::new(-inner + 0.045, 0.02, inner),
        ),
        (
            Vec3::new(inner - 0.045, -0.03, -inner),
            Vec3::new(inner, 0.02, inner),
        ),
    ] {
        trim.append(&Shape::cuboid(a, b, line).with_seed(0.1));
    }
    let leads = 18;
    for side in 0..4 {
        for i in 0..leads {
            let t = (i as f32 + 0.5) / leads as f32;
            let along = (t * 2.0 - 1.0) * (HALF - 0.28);
            let seed = (side as f32 * leads as f32 + i as f32) / (4 * leads) as f32;
            let (centre, outward) = match side {
                0 => (Vec3::new(along, -0.07, -HALF), Vec3::new(0.0, 0.0, -1.0)),
                1 => (Vec3::new(along, -0.07, HALF), Vec3::new(0.0, 0.0, 1.0)),
                2 => (Vec3::new(-HALF, -0.07, along), Vec3::new(-1.0, 0.0, 0.0)),
                _ => (Vec3::new(HALF, -0.07, along), Vec3::new(1.0, 0.0, 0.0)),
            };
            let tangent = Vec3::new(outward.z.abs(), 0.0, outward.x.abs()) * 0.065;
            let reach = outward * 0.34;
            let lo = centre - tangent;
            let hi = centre + tangent + reach;
            let (min, max) = (
                Vec3::new(lo.x.min(hi.x), -0.1, lo.z.min(hi.z)),
                Vec3::new(lo.x.max(hi.x), 0.0, lo.z.max(hi.z)),
            );
            trim.append(&Shape::cuboid(min, max, MAGENTA * 3.4).with_seed(seed));
        }
    }
    trim
}

/// The sky above the street and its reflection in the road: a deep indigo that glows magenta
/// towards the horizon, with low cloud lit from below and searchlights sweeping it.
struct SkyShader;

impl FragmentShaderSimd<ScreenVaryings> for SkyShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [x, y] = v.ndc();
        let ray = view_ray(view, x, y);
        let t = s(view.frame.time);
        let up = ray.y.abs();
        let glow = (-up * s(5.5)).exp();
        let mut colour = mix3(rgb(0.015, 0.02, 0.09), splat3(haze()) * s(1.35), glow);
        let reach = s(1.0) / up.fast_max(s(0.04));
        let (px, pz) = (ray.x * reach + t * s(0.35), ray.z * reach + t * s(0.12));
        let cloud = smoothstep(0.42, 0.78, fbm(px * s(0.55), pz * s(0.55), 4));
        let lit = (-up * s(3.0)).exp();
        colour =
            colour + rgb(0.5, 0.12, 0.42) * (cloud * lit * smoothstep(0.02, 0.2, up) * s(0.55));
        for (i, hue) in [CYAN, MAGENTA].into_iter().enumerate() {
            let sweep = (t * s(0.23 + 0.09 * i as f32) + s(i as f32 * 2.1)).sin() * s(0.55);
            let across = (ray.x / up.fast_max(s(0.05)) * s(0.12) - sweep).abs();
            let beam = (-across * across * s(260.0)).exp()
                * smoothstep(0.02, 0.5, up)
                * (s(1.0) - smoothstep(0.5, 0.95, up));
            colour = colour + splat3(hue) * (beam * s(0.45));
        }
        to_colour(colour, s(SKY_DISTANCE))
    }
}

/// Rain falling in front of everything: layers of thin streaks, nearer ones longer and brighter.
struct RainShader;

impl FragmentShaderSimd<ScreenVaryings> for RainShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [x, y] = v.ndc();
        let t = view.frame.time;
        let mut total = s(0.0);
        for (layer, (columns, speed, length, weight)) in [
            (90.0_f32, 2.6_f32, 0.07_f32, 0.2_f32),
            (55.0, 3.4, 0.11, 0.3),
            (32.0, 4.4, 0.18, 0.42),
        ]
        .into_iter()
        .enumerate()
        {
            let k = layer as f32 * 17.0;
            let slant = x + y * s(0.12);
            let column = slant * s(columns);
            let (index, f) = (column.floor(), fract(column));
            let h = hash2(index + s(k), s(1.7));
            let fall = fract(y * s(0.5) + s(t * speed * 0.5) + h);
            let streak =
                (s(1.0) - (fall * s(2.0) - s(1.0)).abs() * s(1.0 / length * 0.5)).fast_max(s(0.0));
            let thin = (s(1.0) - (f - s(0.5)).abs() * s(7.0)).fast_max(s(0.0));
            let present = select(hash2(index, s(k + 3.0)).simd_lt(s(0.42)), s(1.0), s(0.0));
            total += streak * thin * present * s(weight) * (s(0.7) + h * s(0.6));
        }
        to_colour(rgb(0.5, 0.65, 1.0) * total * s(0.12), s(0.0))
    }
}

pub struct NightCity {
    city: City,
    mirrored: City,
    road: Shape,
    trim: Shape,
    traffic: (Vec<StreakVertex>, Vec<u32>),
    motes: Particles,
    glints: Particles,
    facade: SimdPipeline<KitVertexShader, FacadeShader>,
    facade_mirror: SimdPipeline<KitVertexShader, FacadeShader>,
    neon: SimdPipeline<KitVertexShader, NeonShader>,
    neon_mirror: SimdPipeline<KitVertexShader, NeonShader>,
    metal: SimdPipeline<KitVertexShader, ToonShader>,
    metal_mirror: SimdPipeline<KitVertexShader, ToonShader>,
    road_pipeline: SimdPipeline<KitVertexShader, RoadShader>,
    streaks: SimdPipeline<StreakVertexShader, StreakShader>,
    sky: SimdPipeline<ScreenVertexShader, SkyShader>,
    rain: SimdPipeline<ScreenVertexShader, RainShader>,
}

impl NightCity {
    pub fn new() -> Self {
        let mut rng = Rng::new(808);
        let city = City::build(&mut rng);
        let mirrored = City {
            towers: reflected(&city.towers),
            tubes: reflected(&city.tubes),
            faces: reflected(&city.faces),
            metal: reflected(&city.metal),
            beacons: reflected(&city.beacons),
        };
        let mut road = Shape::new();
        let (half, rows) = (FACADE + 15.5, 34);
        let mut z1 = 20.0_f32;
        let mut step = 1.2;
        for _ in 0..rows {
            let z0 = (z1 - step).max(FAR - 12.0);
            road.quad(
                [
                    Vec3::new(-half, FLOOR, z1),
                    Vec3::new(half, FLOOR, z1),
                    Vec3::new(half, FLOOR, z0),
                    Vec3::new(-half, FLOOR, z0),
                ],
                Vec3::ONE,
            );
            z1 = z0;
            step *= 1.12;
        }
        let blended = |culling| {
            SimdPipeline::new(KitVertexShader, RoadShader)
                .with_culling_mode(culling)
                .with_depth_state(DepthState::DEFAULT)
                .with_blend_state(BlendState::ALPHA_BLEND)
        };
        Self {
            city,
            mirrored,
            road,
            trim: chip_trim(),
            traffic: traffic(&mut rng),
            motes: Particles::new(260, 31),
            glints: Particles::new(120, 32),
            facade: pipeline(FacadeShader, CullingMode::BackFace),
            facade_mirror: pipeline(FacadeShader, CullingMode::None),
            neon: pipeline(NeonShader, CullingMode::None),
            neon_mirror: pipeline(NeonShader, CullingMode::None),
            metal: pipeline(ToonShader, CullingMode::BackFace),
            metal_mirror: pipeline(ToonShader, CullingMode::None),
            road_pipeline: blended(CullingMode::None),
            streaks: SimdPipeline::new(StreakVertexShader, StreakShader)
                .with_culling_mode(CullingMode::None)
                .with_blend_state(BlendState::ADDITIVE)
                .with_depth_state(DepthState::READ_ONLY),
            sky: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DEFAULT),
            rain: SimdPipeline::new(ScreenVertexShader, RainShader)
                .with_culling_mode(CullingMode::None)
                .with_blend_state(BlendState::ADDITIVE)
                .with_depth_state(DepthState::DISABLED),
        }
    }

    fn draw_street<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, view: &ViewRef, mirror: f32) {
        let (city, facade, neon, metal) = if mirror > 0.0 {
            (&self.city, &self.facade, &self.neon, &self.metal)
        } else {
            (
                &self.mirrored,
                &self.facade_mirror,
                &self.neon_mirror,
                &self.metal_mirror,
            )
        };
        let surface = SurfaceUniforms {
            view: view.clone(),
            fog: fog(),
            mirror,
        };
        let motion = KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let tube = |style| NeonUniforms {
            surface: surface.clone(),
            style,
        };

        city.towers
            .draw(pass, facade, motion.clone(), surface.clone());
        city.metal.draw(pass, metal, motion.clone(), toon.clone());
        city.tubes
            .draw(pass, neon, motion.clone(), tube(Style::Flicker));
        city.faces
            .draw(pass, neon, motion.clone(), tube(Style::Sign));
        city.beacons.draw(pass, neon, motion, tube(Style::Beacon));
    }

    fn draw_traffic<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, view: &ViewRef, mirror: f32) {
        pass.draw_simd(
            &self.streaks,
            DrawCall::new(
                &self.traffic.0,
                &self.traffic.1,
                PrimitiveMode::TRIANGLES,
                (),
            ),
            StreakUniforms {
                view: view.clone(),
                mirror,
            },
        );
    }
}

impl Stage for NightCity {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Piano],
            treatment: Treatment::LoFi,
            ambience: vec![
                Ambience::bed(Source::Recording("rain", 0.0), 0.22, 0.0),
                Ambience::once(Source::Recording("car", 0.0), 0.1, 1.3, 0.18, 0.4),
            ],
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;

        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| view.clone());

        self.draw_street(pass, view, -1.0);
        self.draw_traffic(pass, view, -1.0);

        let surface = SurfaceUniforms {
            view: view.clone(),
            fog: fog(),
            mirror: 1.0,
        };
        let motion = KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };
        self.road.draw(pass, &self.road_pipeline, motion, surface);

        self.draw_street(pass, view, 1.0);
        scene.chip.draw(pass, view, &look());
        self.trim.draw(
            pass,
            &self.neon,
            KitUniforms {
                view: view.clone(),
                motion: Motion::OnChip,
            },
            NeonUniforms {
                surface: SurfaceUniforms {
                    view: view.clone(),
                    fog: fog(),
                    mirror: 1.0,
                },
                style: Style::Chase,
            },
        );
        self.draw_traffic(pass, view, 1.0);

        self.motes.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 2.0, -8.0),
                half_extent: Vec3::new(7.0, 6.0, 20.0),
                velocity: Vec3::new(0.0, 0.5, 0.0),
                wander: 0.6,
                size: (0.02, 0.05),
                colour: Vec3::new(1.4, 0.35, 1.0),
                twinkle: 1.8,
            },
        );
        self.glints.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 0.5, 0.0),
                half_extent: Vec3::new(8.0, 5.0, 12.0),
                velocity: Vec3::new(-0.4, -0.6, 0.0),
                wander: 0.3,
                size: (0.015, 0.03),
                colour: Vec3::new(0.4, 1.0, 1.4),
                twinkle: 2.4,
            },
        );
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.rain, (), |_| view.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.95,
            vignette: 0.5,
            grain: 0.02,
            aberration: 0.0025,
            desaturate: -0.12,
        }
    }
}
