//! cave-wall: a cave lit by one flickering torch. Ochre hand stencils, charcoal animals that
//! run and a spiral sun are painted on a rough rock wall; the torch sweeps along the wall and
//! the chip, painted red ochre, throws its shadow across the paintings.

use std::f32::consts::{PI, TAU};

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{F, clamp01, fract, hash2, length, mix3, mul3, rgb, s, smoothstep, splat3};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, pipeline,
};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::light::facing;
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, scatter};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// Where the wall stands behind the chip.
const WALL: f32 = -9.0;

const OCHRE_RED: Vec3 = Vec3 {
    x: 0.5,
    y: 0.13,
    z: 0.06,
};
const OCHRE_YELLOW: Vec3 = Vec3 {
    x: 0.78,
    y: 0.5,
    z: 0.12,
};
const CHARCOAL: Vec3 = Vec3 {
    x: 0.025,
    y: 0.02,
    z: 0.018,
};

/// The torch's position and brightness at `time`: it is carried back and forth along the wall,
/// and its flame never burns steadily.
fn torch(time: f32) -> (Vec3, f32) {
    let jitter = (time * 21.0).sin() * 0.12;
    let position = Vec3::new(
        9.0 * (time * 0.4).sin() + jitter,
        1.6 + 0.35 * (time * 0.9).sin(),
        2.5,
    );
    let flicker = 1.0
        + 0.16 * (time * 13.0).sin() * (time * 7.3 + 1.0).sin()
        + 0.1 * (time * 29.0 + 3.0).sin();
    (position, flicker)
}

/// The cave floor: uneven rock, with the chip's standing place levelled.
fn floor(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let flat = (r / 6.0).clamp(0.0, 1.0);
    FLOOR + flat * flat * ((fbm_scalar(x * 0.18 + 3.0, z * 0.18, 3) - 0.45) * 2.2)
}

/// The wall's depth at (`x`, `y`): rough rock that curves in towards the viewer at the sides
/// and overhangs towards the top, like the inside of a cave.
fn wall(x: f32, y: f32) -> f32 {
    let lumps = (fbm_scalar(x * 0.22, y * 0.22, 4) - 0.5) * 2.4
        + (fbm_scalar(x * 0.9 + 5.0, y * 0.9, 2) - 0.5) * 0.45;
    WALL + 0.012 * x * x + 0.1 * (y - 3.0).max(0.0).powi(2) + lumps
}

fn wall_mesh() -> Shape {
    let (nx, ny) = (150, 40);
    let point = |i: usize, j: usize| {
        let x = -48.0 + 96.0 * i as f32 / nx as f32;
        let y = FLOOR - 3.0 + 18.0 * (j as f32 / ny as f32).powf(1.15);
        Vec3::new(x, y, wall(x, y))
    };
    let mut shape = Shape::new();
    for j in 0..ny {
        for i in 0..nx {
            let (a, b, c, d) = (
                point(i, j),
                point(i + 1, j),
                point(i + 1, j + 1),
                point(i, j + 1),
            );
            if (i + j) % 2 == 0 {
                shape
                    .triangle(a, b, c, Vec3::ONE)
                    .triangle(a, c, d, Vec3::ONE);
            } else {
                shape
                    .triangle(a, b, d, Vec3::ONE)
                    .triangle(b, c, d, Vec3::ONE);
            }
        }
    }
    shape
}

fn stalagmite(rng: &mut Rng) -> Shape {
    Shape::lathe(
        &[
            (0.0, 0.0),
            (0.7, 0.0),
            (0.45, 0.8),
            (0.3, 1.8),
            (0.12, 2.7),
            (0.0, 3.2),
        ],
        6,
        Vec3::ONE,
    )
    .facet_variation(rng, 0.2)
}

fn boulder(rng: &mut Rng) -> Shape {
    Shape::icosphere(1.0, 1, 0.3, rng.range(0.0, 99.0), Vec3::ONE)
        .transformed(&Mat4::scaling(
            rng.range(1.0, 1.5),
            rng.range(0.6, 0.9),
            rng.range(0.9, 1.3),
        ))
        .facet_variation(rng, 0.15)
}

/// The chip painted in red ochre and charcoal, lit by the torch.
fn look(torch: Vec3, flicker: f32) -> Look {
    Look {
        substrate: Vec3::new(0.5, 0.17, 0.08),
        die: Vec3::new(0.04, 0.032, 0.028),
        gold: Vec3::new(0.85, 0.58, 0.22),
        streams: Vec3::new(1.0, 0.5, 0.18),
        key: torch.normalise(),
        key_colour: Vec3::new(3.4, 1.7, 0.65) * flicker,
        sky: Vec3::new(0.2, 0.11, 0.07),
        ground: Vec3::new(0.12, 0.06, 0.03),
        softbox: Vec3::new(0.3, 0.14, 0.06),
    }
}

/// The distance from (`px`, `py`) to the segment from (`ax`, `ay`) to (`bx`, `by`), less `r`.
#[inline(always)]
fn capsule(px: F, py: F, a: (f32, f32), b: (f32, f32), r: f32) -> F {
    let (pax, pay) = (px - s(a.0), py - s(a.1));
    let (bax, bay) = (b.0 - a.0, b.1 - a.1);
    let h = clamp01((pax * s(bax) + pay * s(bay)) / s(bax * bax + bay * bay));
    let (dx, dy) = (pax - h * s(bax), pay - h * s(bay));
    (dx * dx + dy * dy).sqrt() - s(r)
}

/// A hand with fingers up, palm at the origin, about 0.9 tall.
#[inline(always)]
fn hand(px: F, py: F) -> F {
    let fingers = [
        ((-0.12, 0.16), (-0.2, 0.46)),
        ((-0.04, 0.2), (-0.06, 0.56)),
        ((0.05, 0.2), (0.09, 0.54)),
        ((0.13, 0.16), (0.24, 0.44)),
        ((0.15, -0.02), (0.38, 0.2)),
    ];
    let mut d = capsule(px, py, (0.0, -0.4), (0.0, -0.05), 0.16);
    d = d.fast_min(capsule(px, py, (-0.1, 0.0), (0.1, 0.0), 0.2));
    for (base, tip) in fingers {
        d = d.fast_min(capsule(px, py, base, tip, 0.05));
    }
    d
}

/// A running animal in profile, nose along +x, about 1.6 long; its legs swing with `phase`.
#[inline(always)]
fn animal(px: F, py: F, phase: F) -> F {
    let mut d = capsule(px, py, (-0.45, 0.0), (0.35, 0.03), 0.1);
    d = d.fast_min(capsule(px, py, (0.38, 0.05), (0.62, 0.34), 0.06));
    d = d.fast_min(capsule(px, py, (0.62, 0.34), (0.84, 0.29), 0.05));
    for (base, tip) in [
        ((0.66, 0.36), (0.6, 0.6)),
        ((0.6, 0.6), (0.72, 0.7)),
        ((0.64, 0.38), (0.5, 0.58)),
    ] {
        d = d.fast_min(capsule(px, py, base, tip, 0.02));
    }
    d = d.fast_min(capsule(px, py, (-0.45, 0.02), (-0.63, -0.08), 0.025));
    for (hip, offset) in [
        ((0.3, -0.02), 0.0),
        ((0.3, -0.02), 0.7),
        ((-0.4, -0.02), PI),
        ((-0.4, -0.02), PI + 0.7),
    ] {
        let upper = (phase + s(offset)).sin() * s(0.75);
        let knee = (
            s(hip.0) + upper.sin() * s(0.36),
            s(hip.1) - upper.cos() * s(0.36),
        );
        let lower = upper - s(0.9) + (phase + s(offset + 1.0)).sin() * s(0.4);
        let foot = (
            knee.0 + lower.sin() * s(0.34),
            knee.1 - lower.cos() * s(0.34),
        );
        d = d.fast_min(segment(px, py, (s(hip.0), s(hip.1)), knee, 0.04));
        d = d.fast_min(segment(px, py, knee, foot, 0.03));
    }
    d
}

/// [`capsule`] between points that are themselves per-pixel values.
#[inline(always)]
fn segment(px: F, py: F, a: (F, F), b: (F, F), r: f32) -> F {
    let (pax, pay) = (px - a.0, py - a.1);
    let (bax, bay) = (b.0 - a.0, b.1 - a.1);
    let h = clamp01((pax * bax + pay * bay) / (bax * bax + bay * bay + s(1e-6)));
    let (dx, dy) = (pax - h * bax, pay - h * bay);
    (dx * dx + dy * dy).sqrt() - s(r)
}

/// An ochre stencil of a hand centred at (`cx`, `cy`): bare rock where the hand lay, sprayed
/// paint in a stippled halo round it. `flip` mirrors it for a left hand. Returns the paint's
/// coverage.
#[inline(always)]
fn hand_stencil(x: F, y: F, cx: f32, cy: f32, scale: f32, flip: f32, grain: F) -> F {
    let (lx, ly) = ((x - s(cx)) * s(flip / scale), (y - s(cy)) / s(scale));
    let d = hand(lx, ly) * s(scale);
    let density = (s(1.0) - smoothstep(0.0, 0.3, d)) * smoothstep(-0.005, 0.01, d);
    let dots = hash2((x * s(26.0)).floor(), (y * s(26.0)).floor());
    (density * s(1.2) - dots)
        .fast_max(s(0.0))
        .fast_min(s(1.0))
        .ceil()
        * (s(0.7) + grain * s(0.6)).fast_min(s(1.0))
}

/// What is painted at (`x`, `y`) on the wall: the colour and how much of the rock it covers.
#[inline(always)]
fn paintings(x: F, y: F, time: F) -> (Vec3Simd, F) {
    let grain = fbm(x * s(7.0), y * s(7.0), 2);
    let rough = (grain - s(0.5)) * s(0.05);

    let mut colour = splat3(OCHRE_RED);
    let mut cover = s(0.0);
    for (cx, cy, scale, flip) in [
        (-9.2, 2.4, 1.1, 1.0),
        (-7.4, 3.5, 1.0, -1.0),
        (-5.6, 2.3, 1.15, 1.0),
        (-8.0, 0.4, 0.95, -1.0),
        (7.6, 4.2, 1.0, 1.0),
    ] {
        cover = cover.fast_max(hand_stencil(x, y, cx, cy, scale, flip, grain));
    }

    let mut animals = s(0.0);
    let mut red_animals = s(0.0);
    for (home, base, scale, speed, facing, offset, red) in [
        (5.0, 1.2, 2.3, 0.9, 1.0, 0.0, false),
        (8.8, 0.2, 1.7, 1.2, 1.0, 1.7, true),
        (-3.0, 4.3, 1.9, 1.0, -1.0, 3.1, false),
        (-9.0, -0.4, 1.5, 1.3, -1.0, 4.4, true),
    ] {
        let span = 36.0;
        let travelled = s(home) + time * s(speed * facing);
        let along = ((x - travelled) / s(span) + s(0.5)).floor();
        let lx = (x - travelled - along * s(span)) * s(facing / scale);
        let ly = (y - s(base)) / s(scale);
        let d = animal(lx, ly, time * s(speed * 3.2) + s(offset)) * s(scale);
        let body = s(1.0) - smoothstep(0.0, 0.012, d + rough);
        if red {
            red_animals = red_animals.fast_max(body);
        } else {
            animals = animals.fast_max(body);
        }
    }

    let (sx, sy) = (x - s(5.6), y - s(5.0));
    let radius = (sx * sx + sy * sy).sqrt();
    let turns = fract(sy.atan2(sx) / s(TAU) + radius * s(2.6));
    let spiral = (s(1.0) - smoothstep(0.1, 0.16, (turns - s(0.5)).abs() * s(1.0)))
        * (s(1.0) - smoothstep(0.78, 0.84, radius));
    let sun = spiral.fast_max(s(1.0) - smoothstep(0.0, 0.02, (radius - s(0.92)).abs()));

    let chalk = smoothstep(0.3, 0.6, grain + s(0.2));
    colour = mix3(colour, splat3(OCHRE_YELLOW), sun);
    colour = mix3(colour, splat3(OCHRE_RED), red_animals);
    colour = mix3(colour, splat3(CHARCOAL), animals);
    cover = cover.fast_max(red_animals).fast_max(animals).fast_max(sun);
    (colour, cover * chalk)
}

#[derive(Clone)]
struct CaveUniforms {
    view: ViewRef,
    torch: Vec3,
    flicker: f32,
    /// 1 on the painted wall, 0 on the floor and the rocks.
    painted: f32,
}

/// Rock lit only by the torch: grained and cracked, its relief bumped per pixel, shadowed by the
/// chip, and on the wall painted with the cave's figures.
struct RockShader;

impl FragmentShaderSimd<KitVaryings> for RockShader {
    type Uniforms = CaveUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &CaveUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let flat = facing(v.normal().normalise(), to_eye);

        let wall = s(u.painted);
        let (x, y) = (world.x, world.y * wall + world.z * (s(1.0) - wall));
        let step = 0.06;
        let height = |dx: f32, dy: f32| fbm(x * s(1.6) + s(dx), y * s(1.6) + s(dy), 3);
        let h = height(0.0, 0.0);
        let (gx, gy) = (
            (height(step, 0.0) - h) / s(step),
            (height(0.0, step) - h) / s(step),
        );
        let bump = s(0.35);
        let normal = Vec3Simd::new(
            flat.x - gx * bump,
            flat.y - gy * bump * wall,
            flat.z - gy * bump * (s(1.0) - wall),
        )
        .normalise();

        let strata = fbm(x * s(0.35), y * s(2.2), 3);
        let rock = mix3(rgb(0.2, 0.16, 0.13), rgb(0.4, 0.33, 0.26), strata);
        let crack = (s(1.0)
            - smoothstep(
                0.0,
                0.02,
                (fbm(x * s(1.1) + s(9.0), y * s(1.1), 3) - s(0.5)).abs(),
            ))
            * wall;
        let contrast = s(0.75) + wall * s(-0.2) + h * (s(0.5) + wall * s(0.4));
        let mut albedo = rock * contrast * (s(1.0) - crack * s(0.45));

        let t = view.frame.time;
        let (painted, cover) = paintings(x, y, s(t));
        albedo = mix3(albedo, mul3(painted, rock * s(1.8)), cover * wall);

        let to_torch = splat3(u.torch) - world;
        let distance = length(to_torch);
        let direction = to_torch * (s(1.0) / distance);
        let shadow = s(1.0) - chip_shadow(view, u.torch.normalise(), world);
        let diffuse = normal.dot(direction).fast_max(s(0.0));
        let reach = s(1.0) / (s(1.0) + (distance / s(7.5)) * (distance / s(7.5)));
        let torch_light = splat3(Vec3::new(5.0, 3.2, 1.7) * u.flicker) * (diffuse * reach * shadow);
        let ambient = rgb(0.03, 0.026, 0.03) * (s(0.6) + normal.y * s(0.4));
        scene_colour(view, mul3(albedo, torch_light + ambient), world)
    }
}

pub struct CaveWall {
    wall: Shape,
    floor: Shape,
    rocks: Shape,
    dust: Particles,
    rock: SimdPipeline<KitVertexShader, RockShader>,
}

impl CaveWall {
    pub fn new() -> Self {
        let mut rng = Rng::new(15000);
        let floor = terrain(&ring_radii(1.5, 60.0, 34), 56, floor, |_, _| Vec3::ONE);

        let pillars: Vec<Shape> = (0..3).map(|_| stalagmite(&mut rng)).collect();
        let stones: Vec<Shape> = (0..3).map(|_| boulder(&mut rng)).collect();
        let clear = |x: f32, z: f32| {
            (x * x + z * z).sqrt() > 5.5 && !(x.abs() < 3.4 && z > -1.0) && z > WALL + 1.5
        };
        let pillar_spots = scatter(
            &mut rng,
            30.0,
            9.0,
            3000,
            floor_height,
            |x, z| if clear(x, z) { Some(1.0) } else { None },
            (0.8, 1.5),
        );
        let stone_spots = scatter(
            &mut rng,
            26.0,
            4.0,
            3000,
            floor_height,
            |x, z| if clear(x, z) { Some(1.0) } else { None },
            (0.5, 1.4),
        );
        let mut rocks = Batch::new();
        rocks.add(&pillars, &pillar_spots[..pillar_spots.len().min(7)], |p| {
            (p.seed * 7.0) as usize % 3
        });
        rocks.add(&stones, &stone_spots[..stone_spots.len().min(14)], |p| {
            (p.seed * 7.0) as usize % 3
        });

        Self {
            wall: wall_mesh(),
            floor,
            rocks: rocks.finish(),
            dust: Particles::new(140, 99),
            rock: pipeline(RockShader, CullingMode::None),
        }
    }
}

fn floor_height(x: f32, z: f32) -> f32 {
    floor(x, z) - 0.1
}

impl Stage for CaveWall {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::FrameDrum],
            ambience: vec![Ambience::once(
                Source::Recording("fire", 0.0),
                0.0,
                0.45,
                0.18,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let (position, flicker) = torch(view.frame.time);
        let uniforms = |painted| CaveUniforms {
            view: view.clone(),
            torch: position,
            flicker,
            painted,
        };
        let still = || KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };

        self.wall.draw(pass, &self.rock, still(), uniforms(1.0));
        self.floor.draw(pass, &self.rock, still(), uniforms(0.0));
        self.rocks.draw(pass, &self.rock, still(), uniforms(0.0));
        scene.chip.draw(pass, view, &look(position, flicker));
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.0, -2.0),
                half_extent: Vec3::new(14.0, 4.0, 10.0),
                velocity: Vec3::new(0.25, 0.12, 0.05),
                wander: 0.5,
                size: (0.015, 0.04),
                colour: Vec3::new(1.5, 0.85, 0.4) * flicker,
                twinkle: 1.4,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.5,
            vignette: 0.55,
            grain: 0.03,
            aberration: 0.001,
            desaturate: -0.05,
        }
    }
}
