//! steampunk-works: the wall of a boiler house. Riveted iron plate behind meshing brass gear
//! trains, copper pipes with flanges and swinging pressure dials, and steam that leaks from
//! the valves, all in lamplight. A riveted iron floor gives the horizon.
//!
//! Like `papercraft`, the world is one full-screen shader of layers on planes behind the chip.
//! Every metal part is a signed distance function whose gradient near the edge bevels it, so one
//! brass shader lights gears, pipes and flanges alike.

use std::f32::consts::{PI, TAU};

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, View, ViewRef, view_ray};
use crate::simd::{
    F, clamp01, fract, hash2, mix3, mul3, pow_highlight, rgb, s, select, smoothstep, splat3,
    to_colour,
};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::noise::{fbm, value};
use crate::themes::kit::post::expand;
use crate::themes::{Scene, Stage};

/// The direction towards the lamp that lights the metal and casts the chip's shadow.
const LIGHT: Vec3 = Vec3 {
    x: -0.35,
    y: 0.5,
    z: 0.8,
};
/// The screen-plane direction towards the light, which shadows fall away from.
const LIGHT_SCREEN: (f32, f32) = (-0.57, 0.82);
const FAR: f32 = 10.0;
/// The height of the floor's top edge, in screen units.
const FLOOR_TOP: f32 = -0.5;

/// A gear's teeth, pitch circle and spokes. A gear cut by `mesh` from another turns the opposite
/// way, with its teeth in the other's gaps.
#[derive(Clone, Copy)]
struct Gear {
    x: f32,
    y: f32,
    teeth: f32,
    module: f32,
    phase: f32,
    speed: f32,
    spokes: f32,
}

impl Gear {
    fn new(x: f32, y: f32, teeth: u32, module: f32, speed: f32, spokes: u32) -> Self {
        Self {
            x,
            y,
            teeth: teeth as f32,
            module,
            phase: 0.0,
            speed,
            spokes: spokes as f32,
        }
    }

    fn pitch_radius(&self) -> f32 {
        self.teeth * self.module / 2.0
    }

    /// A gear of `teeth` teeth whose pitch circle touches this one's, in direction `angle`.
    fn mesh(&self, angle: f32, teeth: u32, spokes: u32) -> Self {
        let mut next = Self::new(0.0, 0.0, teeth, self.module, 0.0, spokes);
        let distance = self.pitch_radius() + next.pitch_radius();
        next.x = self.x + angle.cos() * distance;
        next.y = self.y + angle.sin() * distance;
        let ratio = self.teeth / next.teeth;
        next.phase = angle + PI - PI / next.teeth + (angle - self.phase) * ratio;
        next.speed = -self.speed * ratio;
        next
    }

    #[inline(always)]
    fn distance(&self, a: F, b: F, time: f32) -> F {
        let (dx, dy) = (a - s(self.x), b - s(self.y));
        let len = (dx * dx + dy * dy).sqrt();
        let rot = self.phase + self.speed * time;
        let angle = dy.atan2(dx) - s(rot);
        let along = fract(angle * s(self.teeth / TAU) + s(0.5)) - s(0.5);
        let flank = clamp01((s(0.36) - along.abs()) * s(4.0));
        let root = self.pitch_radius() - 0.55 * self.module;
        let outline = len - (s(root) + flank * s(1.05 * self.module));
        let mut d = outline * s(0.75);
        if self.spokes > 0.0 {
            let (hub, rim) = (0.2 * root, 0.82 * root);
            let across = (fract(angle * s(self.spokes / TAU) + s(0.5)) - s(0.5)).abs()
                * len
                * s(TAU / self.spokes);
            let window = (s(hub) - len)
                .fast_max(len - s(rim))
                .fast_max(s(0.07 * root) - across);
            d = d.fast_max(-window);
        }
        d
    }
}

/// One metal part of a layer.
#[derive(Clone, Copy)]
enum Part {
    Gear(Gear),
    /// A disc with a bevel as wide as `bevel`: caps, bolts and elbows.
    Disc {
        x: f32,
        y: f32,
        radius: f32,
        bevel: f32,
    },
    /// A pipe: a capsule from one end to the other, bevelled across its whole width.
    Pipe {
        from: (f32, f32),
        to: (f32, f32),
        radius: f32,
    },
    /// A flange or plate: a box with rounded corners.
    Plate {
        x: f32,
        y: f32,
        half: (f32, f32),
        bevel: f32,
    },
}

/// A part's distance, how far in its bevel reaches, and the metal it is made of.
struct Hit {
    d: F,
    bevel: F,
    tint: Vec3Simd,
}

impl Part {
    #[inline(always)]
    fn distance(&self, a: F, b: F, time: f32) -> (F, f32) {
        match *self {
            Part::Gear(gear) => (gear.distance(a, b, time), 0.012),
            Part::Disc {
                x,
                y,
                radius,
                bevel,
            } => {
                let (dx, dy) = (a - s(x), b - s(y));
                ((dx * dx + dy * dy).sqrt() - s(radius), bevel)
            }
            Part::Pipe { from, to, radius } => {
                let (ex, ey) = (to.0 - from.0, to.1 - from.1);
                let (px, py) = (a - s(from.0), b - s(from.1));
                let t = clamp01((px * s(ex) + py * s(ey)) / s(ex * ex + ey * ey));
                let (qx, qy) = (px - t * s(ex), py - t * s(ey));
                ((qx * qx + qy * qy).sqrt() - s(radius), radius)
            }
            Part::Plate { x, y, half, bevel } => {
                let (dx, dy) = (
                    (a - s(x)).abs() - s(half.0 - bevel),
                    (b - s(y)).abs() - s(half.1 - bevel),
                );
                let outside = (dx.fast_max(s(0.0)) * dx.fast_max(s(0.0))
                    + dy.fast_max(s(0.0)) * dy.fast_max(s(0.0)))
                .sqrt();
                (outside + dx.fast_max(dy).fast_min(s(0.0)) - s(bevel), bevel)
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Dial {
    x: f32,
    y: f32,
    radius: f32,
    /// How fast and how far the needle swings, and where it starts.
    rate: f32,
    sweep: f32,
    rest: f32,
}

/// Everything in the world, owned by the stage so each shader invocation reads it by reference.
struct Machine {
    /// Each layer's depth, tint of its parts and part list: layers[i] is back to front.
    layers: Vec<(f32, Vec<(Part, Vec3)>)>,
    dials: Vec<Dial>,
    /// Where steam leaks, in the pipe layer's coordinates, and its drift to the side.
    vents: Vec<(f32, f32, f32)>,
}

const PIPE_LAYER: usize = 1;
const FLOOR_LAYER: usize = 3;

fn machine() -> Machine {
    let bronze = Vec3::new(0.34, 0.22, 0.1);
    let brass = Vec3::new(0.72, 0.5, 0.2);
    let copper = Vec3::new(0.7, 0.34, 0.2);
    let iron = Vec3::new(0.34, 0.32, 0.31);
    let steel = Vec3::new(0.55, 0.55, 0.56);

    let mut back = Vec::new();
    let big = Gear::new(-1.3, 0.3, 30, 0.028, 0.25, 6);
    let chain = [
        big,
        big.mesh(-0.5, 16, 4),
        big.mesh(0.62, 12, 0),
        big.mesh(-2.4, 20, 5),
    ];
    let second = Gear::new(1.5, 0.5, 28, 0.028, -0.28, 5);
    let second_chain = [
        second,
        second.mesh(-1.7, 14, 0),
        second.mesh(-2.9, 18, 4),
        second.mesh(0.3, 12, 0),
    ];
    for gear in chain.iter().chain(second_chain.iter()) {
        back.push((Part::Gear(*gear), bronze));
        back.push((
            Part::Disc {
                x: gear.x,
                y: gear.y,
                radius: 0.2 * gear.pitch_radius().min(0.4) + 0.02,
                bevel: 0.02,
            },
            iron,
        ));
    }

    let pipe =
        |from: (f32, f32), to: (f32, f32), radius: f32| (Part::Pipe { from, to, radius }, copper);
    let flange = |x: f32, y: f32, half: (f32, f32)| {
        (
            Part::Plate {
                x,
                y,
                half,
                bevel: 0.012,
            },
            brass,
        )
    };
    let mut pipes = vec![
        pipe((-2.2, 0.82), (2.2, 0.82), 0.045),
        pipe((-0.95, 0.82), (-0.95, -0.55), 0.055),
        pipe((1.05, 0.82), (1.05, -0.55), 0.065),
        pipe((-0.95, -0.12), (-1.6, -0.12), 0.035),
        pipe((1.05, -0.2), (1.75, -0.2), 0.04),
        pipe((1.75, -0.2), (1.75, 0.3), 0.04),
    ];
    pipes.push((
        Part::Disc {
            x: 1.75,
            y: -0.2,
            radius: 0.058,
            bevel: 0.03,
        },
        brass,
    ));
    for &(x, y) in &[(-0.95, 0.82), (1.05, 0.82)] {
        pipes.push((
            Part::Disc {
                x,
                y,
                radius: 0.085,
                bevel: 0.04,
            },
            brass,
        ));
    }
    for &(x, y) in &[(-0.95, 0.5), (-0.95, -0.2), (1.05, 0.55), (1.05, -0.25)] {
        pipes.push(flange(x, y, (0.1, 0.022)));
    }
    for &(x, y) in &[
        (-1.35, 0.82),
        (0.0, 0.82),
        (1.55, 0.82),
        (-1.6, -0.12),
        (1.45, -0.2),
    ] {
        pipes.push(flange(x, y, (0.022, 0.075)));
    }
    pipes.push((
        Part::Disc {
            x: -0.95,
            y: 0.1,
            radius: 0.075,
            bevel: 0.035,
        },
        steel,
    ));
    pipes.push((
        Part::Disc {
            x: 1.05,
            y: 0.1,
            radius: 0.09,
            bevel: 0.04,
        },
        steel,
    ));

    let wheel = Gear::new(1.05, 0.1, 10, 0.034, 0.18, 4);
    pipes.push((Part::Gear(wheel), iron));

    let mut front = Vec::new();
    let fore = Gear::new(-1.5, -0.6, 22, 0.04, -0.14, 5);
    for gear in [fore, fore.mesh(0.15, 12, 4)] {
        front.push((Part::Gear(gear), brass));
        front.push((
            Part::Disc {
                x: gear.x,
                y: gear.y,
                radius: 0.1 * gear.pitch_radius() + 0.04,
                bevel: 0.03,
            },
            bronze,
        ));
    }
    let right = Gear::new(1.55, -0.65, 24, 0.04, 0.12, 6);
    for gear in [right, right.mesh(2.5, 14, 0)] {
        front.push((Part::Gear(gear), brass));
        front.push((
            Part::Disc {
                x: gear.x,
                y: gear.y,
                radius: 0.1 * gear.pitch_radius() + 0.04,
                bevel: 0.03,
            },
            bronze,
        ));
    }

    let floor = vec![
        (
            Part::Plate {
                x: 0.0,
                y: FLOOR_TOP - 1.5,
                half: (4.0, 1.5),
                bevel: 0.02,
            },
            iron,
        ),
        (
            Part::Plate {
                x: 0.0,
                y: FLOOR_TOP - 0.03,
                half: (4.0, 0.03),
                bevel: 0.014,
            },
            brass,
        ),
    ];

    Machine {
        layers: vec![(-11.0, back), (-9.0, pipes), (-6.0, front), (-3.5, floor)],
        dials: vec![
            Dial {
                x: -1.5,
                y: 0.45,
                radius: 0.15,
                rate: 1.4,
                sweep: 0.7,
                rest: 0.4,
            },
            Dial {
                x: 1.55,
                y: 0.3,
                radius: 0.19,
                rate: 0.9,
                sweep: 0.6,
                rest: -0.3,
            },
            Dial {
                x: 0.55,
                y: 0.82,
                radius: 0.1,
                rate: 2.1,
                sweep: 0.9,
                rest: 0.0,
            },
        ],
        vents: vec![(-0.95, -0.12, -1.0), (1.05, -0.35, 1.0), (1.75, 0.3, 0.6)],
    }
}

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
    machine: std::sync::Arc<Machine>,
}

struct Plane {
    t: F,
    x: F,
    y: F,
    a: F,
    b: F,
    scale: f32,
}

impl Plane {
    #[inline(always)]
    fn new(view: &View, ray: Vec3Simd, z: f32, time: f32) -> Self {
        let t = s(z - view.eye.z) / ray.z;
        let x = s(view.eye.x) + ray.x * t;
        let y = s(view.eye.y) + ray.y * t;
        let scale = view.tan_half_fov * (view.eye.z - z);
        let truck = 0.35 * (time * 0.3).sin();
        Self {
            t,
            x,
            y,
            a: (x + s(truck)) / s(scale),
            b: (y - s(view.eye.y)) / s(scale),
            scale,
        }
    }
}

struct World;

impl FragmentShaderSimd<ScreenVaryings> for World {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let time = u.time;
        let machine = &*u.machine;

        let mut colour = rgb(0.0, 0.0, 0.0);
        let mut remaining = s(1.0);
        let mut distance = s(SKY_DISTANCE);
        let mut nearer: Option<(Plane, usize)> = None;
        for index in (0..machine.layers.len()).rev() {
            if !remaining.simd_gt(s(0.002)).any() {
                break;
            }
            let (z, parts) = &machine.layers[index];
            let plane = Plane::new(view, ray, *z, time);
            let hit = layer_hit(parts, &plane, time);
            let pixel = plane.t * s(view.pixel_angle) / s(plane.scale);
            let cover = smoothstep(0.0, 1.0, clamp01(s(0.5) - hit.d / (pixel * s(2.0))));
            let steam = if index == PIPE_LAYER {
                steam(machine, &plane, time)
            } else {
                s(0.0)
            };
            let any_cover = cover.fast_max(steam);
            if any_cover.simd_gt(s(0.0)).any() {
                let mut metal = metal_layer(index, &hit, &plane, pixel, u, nearer.as_ref());
                if index == PIPE_LAYER {
                    metal = dials(machine, &plane, metal, time);
                }
                let steam_colour = rgb(1.0, 0.93, 0.8) * (s(0.6) + s(0.4) * ny);
                let layer_colour = mix3(steam_colour * s(1.6), metal, cover) * any_cover;
                let weight = remaining;
                colour = colour + layer_colour * weight;
                distance = select(
                    remaining.simd_gt(s(0.5)) & cover.simd_gt(s(0.5)),
                    plane.t,
                    distance,
                );
                remaining *= s(1.0) - any_cover;
            }
            nearer = Some((plane, index));
        }
        let wall_plane = Plane::new(view, ray, -15.0, time);
        let wall = wall(&wall_plane, time);
        colour = colour + wall * remaining;
        to_colour(expand(colour), distance)
    }
}

/// The nearest part under each pixel of a layer, with its distance, bevel and tint.
#[inline(always)]
fn layer_hit(parts: &[(Part, Vec3)], plane: &Plane, time: f32) -> Hit {
    let mut hit = Hit {
        d: s(FAR),
        bevel: s(0.02),
        tint: rgb(0.0, 0.0, 0.0),
    };
    for (part, tint) in parts {
        let (d, bevel) = part.distance(plane.a, plane.b, time);
        let inside = d.simd_lt(s(0.0));
        let take = inside | d.simd_lt(hit.d);
        let bounded = select(inside, d, d.fast_min(hit.d));
        hit.bevel = select(take, s(bevel), hit.bevel);
        hit.tint = crate::simd::select3(take, splat3(*tint), hit.tint);
        hit.d = bounded;
    }
    hit
}

/// Brass or iron lit by the lamp: `slope` is the surface's tilt away from facing the viewer.
#[inline(always)]
fn metal(tint: Vec3Simd, slope: (F, F), grain: F, sheen: f32) -> Vec3Simd {
    let tilt = (slope.0 * slope.0 + slope.1 * slope.1).fast_min(s(0.98));
    let nz = (s(1.0) - tilt).sqrt();
    let light = LIGHT.normalise();
    let diffuse = clamp01(slope.0 * s(light.x) + slope.1 * s(light.y) + nz * s(light.z));
    let half = Vec3::new(light.x, light.y, light.z + 1.0).normalise();
    let glint = clamp01(slope.0 * s(half.x) + slope.1 * s(half.y) + nz * s(half.z));
    let specular =
        pow_highlight(glint, 32.0) * s(sheen * 0.6) + pow_highlight(glint, 4.0) * s(0.05 * sheen);
    let environment = mix3(
        rgb(0.1, 0.06, 0.04),
        rgb(1.0, 0.85, 0.6),
        smoothstep(-0.7, 0.8, slope.1),
    );
    let reflectivity = s(0.4) * (s(1.0) - nz);
    let base = tint * (s(0.18) + diffuse * s(0.95)) * (s(1.0) + grain);
    mix3(base, mul3(tint, environment) * s(1.4), reflectivity) + rgb(1.0, 0.82, 0.55) * specular
}

/// Brushed streaks and pitting in a part's own surface, as an offset from 1.
#[inline(always)]
fn grain(plane: &Plane) -> F {
    let streak = value(plane.a * s(5.0), plane.b * s(90.0)) - s(0.5);
    let pit = value(plane.a * s(70.0), plane.b * s(70.0)) - s(0.5);
    let patina = fbm(plane.a * s(3.0) + s(4.0), plane.b * s(3.0), 3) - s(0.5);
    streak * s(0.1) + pit * s(0.08) + patina * s(0.35)
}

#[inline(always)]
fn metal_layer(
    index: usize,
    hit: &Hit,
    plane: &Plane,
    pixel: F,
    u: &Uniforms,
    nearer: Option<&(Plane, usize)>,
) -> Vec3Simd {
    let (time, view, machine) = (u.time, &*u.view, &*u.machine);
    let eps = pixel.fast_max(s(0.0015));
    let (z, parts) = &machine.layers[index];
    let gx = (layer_hit(parts, &shifted(plane, eps, s(0.0)), time).d - hit.d) / eps;
    let gy = (layer_hit(parts, &shifted(plane, s(0.0), eps), time).d - hit.d) / eps;
    let depth = (-hit.d / hit.bevel).fast_max(s(0.0));
    let rim = clamp01(s(1.0) - depth);
    let weight = rim * rim * s(0.95);
    let mut slope = (gx * weight, gy * weight);
    if index == FLOOR_LAYER {
        let (bump_x, bump_y) = diamond_plate(plane);
        slope = (slope.0 + bump_x, slope.1 + bump_y);
    }
    let mut colour = metal(
        hit.tint,
        slope,
        grain(plane),
        if index == FLOOR_LAYER { 0.35 } else { 0.8 },
    );

    let mut shade = s(0.0);
    if let Some((front, caster_index)) = nearer {
        let (caster_z, caster_parts) = &machine.layers[*caster_index];
        let gap = caster_z - z;
        let offset = 0.012 + 0.003 * gap;
        let shadow_plane = shifted(
            front,
            s(LIGHT_SCREEN.0 * offset),
            s(LIGHT_SCREEN.1 * offset),
        );
        let ds = layer_hit(caster_parts, &shadow_plane, time).d;
        let blur = 0.01 + 0.004 * gap;
        shade = (s(1.0) - smoothstep(-blur * 0.4, blur, ds)) * s(0.55);
    }
    if *z > -12.0 {
        let world = Vec3Simd::new(plane.x, plane.y, s(*z));
        shade += chip_shadow(view, LIGHT.normalise(), world) * s(0.5);
    }
    let glow = lamp(plane.a, plane.b);
    colour = colour * (s(1.0) - clamp01(shade) * s(0.75)) * (s(0.55) + glow * s(0.7));
    if index == FLOOR_LAYER {
        colour = floor_panels(colour, plane);
    }
    colour
}

/// Joins between the floor's big plates and the brass rivets that hold them.
#[inline(always)]
fn floor_panels(colour: Vec3Simd, plane: &Plane) -> Vec3Simd {
    let (cu, cv) = (plane.a / s(0.7), (plane.b - s(FLOOR_TOP)) / s(0.42));
    let (u, v) = (fract(cu), fract(cv));
    let edge_u = u.fast_min(s(1.0) - u) * s(0.7);
    let edge_v = v.fast_min(s(1.0) - v) * s(0.42);
    let seam = s(1.0) - smoothstep(0.003, 0.012, edge_u.fast_min(edge_v));
    let (ru, rv) = (edge_u - s(0.035), edge_v - s(0.035));
    let dome = clamp01(s(1.0) - (ru * ru + rv * rv).sqrt() / s(0.016));
    let rivet = rgb(0.75, 0.52, 0.22) * (s(0.5) + dome * s(0.6));
    mix3(
        colour * (s(1.0) - seam * s(0.75)),
        rivet,
        smoothstep(0.0, 0.3, dome),
    )
}

/// The slope of the raised diamonds on the floor's tread plate, as a tilt in x and y.
#[inline(always)]
fn diamond_plate(plane: &Plane) -> (F, F) {
    let (p, q) = ((plane.a + plane.b) * s(16.0), (plane.a - plane.b) * s(16.0));
    let (fp, fq) = (fract(p) - s(0.5), fract(q) - s(0.5));
    let cell = hash2(p.floor(), q.floor());
    let flip = select(cell.simd_gt(s(0.5)), s(1.0), s(-1.0));
    let lozenge = s(1.0)
        - smoothstep(
            0.2,
            0.6,
            (fp.abs() * s(2.0) + fq.abs() * s(0.8)).fast_max(fp.abs() * s(0.8) + fq.abs() * s(2.0)),
        );
    (
        (fp - fq) * lozenge * s(1.6) * flip,
        (fp + fq) * lozenge * s(1.6),
    )
}

/// The same plane, moved by (`da`, `db`) in layer coordinates.
#[inline(always)]
fn shifted(plane: &Plane, da: F, db: F) -> Plane {
    Plane {
        t: plane.t,
        x: plane.x,
        y: plane.y,
        a: plane.a + da,
        b: plane.b + db,
        scale: plane.scale,
    }
}

/// The lamp's pool of light on the wall, brightest behind the chip.
#[inline(always)]
fn lamp(a: F, b: F) -> F {
    let (da, db) = (a * s(0.6), (b - s(0.1)) * s(0.75));
    s(1.0) - smoothstep(0.0, 1.5, (da * da + db * db).sqrt())
}

/// Steam from the vents: puffs that rise, drift and spread, as opacity.
#[inline(always)]
fn steam(machine: &Machine, plane: &Plane, time: f32) -> F {
    let mut total = s(0.0);
    for (n, &(x, y, side)) in machine.vents.iter().enumerate() {
        for k in 0..6 {
            let life = (time * 0.32 + k as f32 / 6.0 + n as f32 * 0.37).fract();
            let sway = ((time * 0.7 + k as f32 * 2.3 + n as f32) * 1.3).sin() * 0.05 * life;
            let (cx, cy) = (x + side * life * 0.35 + sway, y + life * 0.55);
            let radius = 0.03 + 0.14 * life;
            let (dx, dy) = (plane.a - s(cx), plane.b - s(cy));
            let edge = fbm(
                plane.a * s(9.0) + s(k as f32 * 3.1),
                plane.b * s(9.0) - s(time * 0.6),
                2,
            ) - s(0.5);
            let d = (dx * dx + dy * dy).sqrt() - s(radius) + edge * s(0.05 * life + 0.01);
            total += (s(1.0) - smoothstep(-radius * 0.9, 0.0, d))
                * s((1.0 - life) * (life * 6.0).min(1.0) * 0.8);
        }
    }
    clamp01(total)
}

/// Draws the pressure dials over the pipe layer's metal: a bezel, a cream face with ticks and a
/// needle that swings.
#[inline(always)]
fn dials(machine: &Machine, plane: &Plane, metal: Vec3Simd, time: f32) -> Vec3Simd {
    let mut colour = metal;
    for (n, dial) in machine.dials.iter().enumerate() {
        let (dx, dy) = (plane.a - s(dial.x), plane.b - s(dial.y));
        let len = (dx * dx + dy * dy).sqrt() / s(dial.radius);
        if len.simd_lt(s(1.0)).any() {
            let angle = dy.atan2(dx);
            let needle_angle = dial.rest
                + dial.sweep * (time * dial.rate + n as f32 * 1.9).sin()
                + 0.06 * (time * 9.0 + n as f32).sin();
            let (nx, ny) = (needle_angle.sin(), needle_angle.cos());
            let along = (dx * s(nx) + dy * s(ny)) / s(dial.radius);
            let across = (dx * s(ny) - dy * s(nx)).abs() / s(dial.radius);
            let needle = (s(1.0) - smoothstep(0.02, 0.05, across))
                * (s(1.0) - smoothstep(0.7, 0.75, along))
                * gt_zero(along + s(0.15));
            let face = rgb(0.93, 0.87, 0.7) * (s(0.75) + s(0.25) * (s(1.0) - len));
            let tick = fract((angle - s(PI / 2.0)) * s(14.0 / TAU) + s(0.5)) - s(0.5);
            let ticks = (s(1.0) - smoothstep(0.03, 0.07, tick.abs() * s(1.0)))
                * smoothstep(0.7, 0.74, len)
                * (s(1.0) - smoothstep(0.88, 0.9, len));
            let arc = (s(1.0) - smoothstep(0.0, 0.05, (len - s(0.62)).abs() - s(0.015)))
                * gt_zero(s(2.4) - (angle - s(PI / 2.0)).abs());
            let ink = rgb(0.1, 0.07, 0.05);
            let mut face = mix3(face, ink, clamp01(ticks + arc * s(0.6)));
            face = mix3(face, rgb(0.55, 0.1, 0.06), needle);
            let hub = s(1.0) - smoothstep(0.1, 0.13, len);
            face = mix3(face, rgb(0.15, 0.12, 0.1), hub);
            let inside = s(1.0) - smoothstep(0.76, 0.8, len);
            let ring = clamp01((s(1.0) - len) / s(0.22));
            let bezel = metal * (s(0.7) + s(0.5) * (s(1.0) - (s(2.0) * ring - s(1.0)).abs()));
            let dial_colour = mix3(bezel, face, inside);
            let mask = s(1.0) - smoothstep(0.96, 1.0, len);
            colour = mix3(colour, dial_colour, mask);
        }
    }
    colour
}

#[inline(always)]
fn gt_zero(x: F) -> F {
    select(x.simd_gt(s(0.0)), s(1.0), s(0.0))
}

/// The iron wall behind everything: riveted plates with dark seams, streaked with soot and
/// lit by the lamp.
#[inline(always)]
fn wall(plane: &Plane, time: f32) -> Vec3Simd {
    let (pw, ph) = (0.85, 0.62);
    let (cu, cv) = (plane.a / s(pw), plane.b / s(ph));
    let (cell_u, cell_v) = (cu.floor(), cv.floor());
    let (u, v) = (cu - cell_u, cv - cell_v);
    let tone = hash2(cell_u, cell_v * s(1.7)) - s(0.5);
    let edge_u = u.fast_min(s(1.0) - u) * s(pw);
    let edge_v = v.fast_min(s(1.0) - v) * s(ph);
    let edge = edge_u.fast_min(edge_v);

    let base = rgb(0.26, 0.2, 0.16) * (s(1.0) + tone * s(0.35));
    let streak = value(plane.a * s(14.0), plane.b * s(1.4) + s(7.0)) - s(0.5);
    let soot = fbm(plane.a * s(2.0), plane.b * s(2.0) + s(3.0), 4);
    let brushed = value(plane.a * s(2.0), plane.b * s(120.0)) - s(0.5);
    let mut colour = base * (s(0.8) + soot * s(0.5) + streak * s(0.55) + brushed * s(0.18));

    let bevel = smoothstep(0.0, 0.025, edge);
    let lit = ((u - s(0.5)) * s(-1.0) + (v - s(0.5)) * s(1.0))
        .fast_min(s(0.5))
        .fast_max(s(-0.5));
    let seam = s(1.0) - smoothstep(0.006, 0.014, edge);
    colour = colour * (s(0.55) + bevel * s(0.45) + (s(1.0) - bevel) * lit * s(0.7));
    colour = colour * (s(1.0) - seam * s(0.8));

    let (ru, rv) = (
        u.fast_min(s(1.0) - u) * s(pw) - s(0.055),
        v.fast_min(s(1.0) - v) * s(ph) - s(0.055),
    );
    let rd = (ru * ru + rv * rv).sqrt();
    let dome = clamp01(s(1.0) - rd / s(0.016));
    let sign_u = select(u.simd_lt(s(0.5)), s(1.0), s(-1.0));
    let sign_v = select(v.simd_lt(s(0.5)), s(1.0), s(-1.0));
    let highlight =
        clamp01(s(0.5) - (ru * sign_u * s(0.5) - rv * sign_v * s(0.5)) / s(0.016) * s(0.5));
    let rivet = rgb(0.62, 0.44, 0.26) * (s(0.35) + dome * s(0.6) + highlight * dome * s(0.7));
    colour = mix3(colour, rivet, smoothstep(0.0, 0.25, dome));

    let flicker = 1.0 + 0.04 * (time * 7.0).sin() * (time * 2.3).sin();
    let glow = lamp(plane.a, plane.b);
    let warm = rgb(1.0, 0.68, 0.38);
    mul3(
        colour * (s(0.25) + glow * s(flicker) * s(1.35)),
        rgb(0.7, 0.7, 0.7) + warm * s(0.5),
    )
}

pub struct SteampunkWorks {
    look: Look,
    machine: std::sync::Arc<Machine>,
    pipeline: SimdPipeline<ScreenVertexShader, World>,
}

impl SteampunkWorks {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.45, 0.3, 0.14),
                die: Vec3::new(0.16, 0.09, 0.05),
                gold: Vec3::new(0.85, 0.6, 0.25),
                streams: Vec3::new(1.0, 0.55, 0.18),
                key: LIGHT.normalise(),
                key_colour: Vec3::new(1.5, 1.15, 0.8),
                sky: Vec3::new(0.5, 0.33, 0.2),
                ground: Vec3::new(0.05, 0.035, 0.025),
                softbox: Vec3::ZERO,
            },
            machine: std::sync::Arc::new(machine()),
            pipeline: SimdPipeline::new(ScreenVertexShader, World)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for SteampunkWorks {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Trumpet, Instrument::Trombone],
            ambience: vec![
                Ambience::bed(Source::Recording("ratchet", 0.0), 0.13, 0.3),
                Ambience::once(Source::Recording("steam", 0.0), 0.35, 0.8, 0.15, -0.35),
            ],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        let uniforms = Uniforms {
            view: scene.view.clone(),
            time: scene.local_time,
            machine: self.machine.clone(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.pipeline, (), |_| uniforms.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.3,
            vignette: 0.4,
            grain: 0.015,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
