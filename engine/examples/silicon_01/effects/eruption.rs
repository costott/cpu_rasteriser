//! The eruption: rays of light leave the die, shock rings sweep the frame, and the renderer's
//! primitives burst out of the chip and fill the screen. They then settle into a grid across the
//! screen, pulsing, flashing and changing shape with packets racing along its lines, until the
//! first world cuts in. Before it, the die starts to throw thin rays as the spin reaches its limit.

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::director::{Phase, chip_matrix, ramp, smootherstep};
use crate::effects::primitives::{
    Kind, PRIMITIVES, Sprite, SpriteVaryings, SpriteVertex, expand as expand_sprite, hidden,
    sprite_corners,
};
use crate::effects::streak::{
    Streak, StreakVaryings, StreakVertex, expand, hidden as hidden_streak, streak_corners,
};
use crate::effects::{
    DIE_TOP, GOLD, GREEN_BRIGHT, WARM, die_centre, quad_model, random, random_direction,
};
use crate::shaders::{View, ViewRef};

const PRIMITIVE_COUNT: usize = 1800;
const RAY_COUNT: usize = 72;
/// The grid the burst settles into: cells about 51 pixels across at 720p, near the size of a tile.
const COLUMNS: usize = 25;
const ROWS: usize = 14;
/// How far in front of the camera the grid hangs, in world units: nearer than the chip, so the
/// chip glows through it.
const GRID_DISTANCE: f32 = 6.0;
/// Seconds after the eruption when the primitives start to settle into the grid, the most a
/// primitive's start is staggered by, and how long each takes to fly to its place (set by eye,
/// provisional).
pub const SETTLE: f32 = 0.65;
pub const SETTLE_STAGGER: f32 = 0.3;
pub const SETTLE_TIME: f32 = 0.6;

/// What the burst's vertex shaders share: the view, and the die's place at the eruption, where
/// everything leaves from.
#[derive(Clone)]
pub struct BurstUniforms {
    view: ViewRef,
    origin: Vec3,
}

impl BurstUniforms {
    pub fn new(view: &ViewRef) -> Self {
        let chip = chip_matrix(Phase::Eruption.start());
        Self {
            view: view.clone(),
            origin: (chip * Vec4::new(0.0, DIE_TOP, 0.0, 1.0)).xyz(),
        }
    }

    fn since(&self) -> f32 {
        self.view.frame.time - Phase::Eruption.start()
    }
}

/// One primitive of the burst.
#[derive(Clone, Copy)]
pub struct Burst {
    direction: Vec3,
    speed: f32,
    delay: f32,
    size: f32,
    spin: f32,
    kind: Kind,
    seed: f32,
    colour: Vec3,
    role: Role,
    /// Seconds of extra delay before it starts to settle.
    stagger: f32,
}

/// Where a primitive of the burst settles.
#[derive(Clone, Copy)]
enum Role {
    /// In a cell of the grid, counted from the top left.
    Cell { column: f32, row: f32 },
    /// Racing along one of the grid's lines, `offset` of the way along at the eruption, `speed`
    /// screen widths a second (negative runs backwards).
    Packet {
        line: f32,
        across: bool,
        offset: f32,
        speed: f32,
    },
}

pub struct BurstVertexShader;

impl VertexShader for BurstVertexShader {
    type Vertex = SpriteVertex<Burst>;
    type Uniforms = BurstUniforms;
    type Varyings = SpriteVaryings;

    fn shade(&self, vertex: SpriteVertex<Burst>, u: &BurstUniforms) -> (Vec4, SpriteVaryings) {
        let burst = &vertex.data;
        let since = u.since();
        let age = since - burst.delay;
        if age < 0.0 || since >= Phase::Worlds.start() - Phase::Eruption.start() {
            return hidden();
        }
        const DRAG: f32 = 1.3;
        let distance = burst.speed * (1.0 - (-DRAG * age).exp()) / DRAG;
        let free = u.origin + burst.direction * distance;
        let settled =
            smootherstep(((since - SETTLE - burst.stagger) / SETTLE_TIME).clamp(0.0, 1.0));
        let place = place(&u.view, burst);
        let shimmer = 0.8 + 0.2 * (age * 17.0 + burst.seed * 40.0).sin();
        let (kind, seed) = if settled < 0.5 {
            (burst.kind, burst.seed)
        } else {
            (place.kind, place.seed)
        };
        // The grid builds towards the white-out the first world emerges from.
        let build = 1.0
            + 2.5
                * ramp(
                    since,
                    SETTLE + SETTLE_TIME,
                    Phase::Worlds.start() - Phase::Eruption.start(),
                );
        let light = (1.0 + (place.brightness - 1.0) * settled) * (1.0 + (build - 1.0) * settled);
        let sprite = Sprite {
            centre: free + (place.centre - free) * settled,
            size: burst.size * (0.6 + 0.4 * (age * 3.0).min(1.0)) * (1.0 - settled)
                + place.size * (1.0 + 0.1 * (build - 1.0)) * settled,
            angle: (burst.spin * age + burst.seed * 6.0) * (1.0 - settled),
            kind,
            seed,
            colour: (burst.colour + (Vec3::ONE - burst.colour) * (place.whiten * settled)) * light,
            fade: (age / 0.06).min(1.0) * shimmer,
        };
        expand_sprite(&u.view, &sprite, vertex.corner)
    }
}

/// Where a settled primitive is in the grid at this moment, and how it shows there.
struct Place {
    centre: Vec3,
    size: f32,
    kind: Kind,
    seed: f32,
    /// Multiplies its light.
    brightness: f32,
    /// How far its colour is pushed to white, 0 to 1.
    whiten: f32,
}

/// The grid hangs square to the camera, so it covers the screen however the camera moves. Cells
/// pulse in waves that sweep diagonally, flash white at random and change kind every few frames;
/// packets race along the lines between them.
fn place(view: &View, burst: &Burst) -> Place {
    let time = view.frame.time;
    let half_height = GRID_DISTANCE * view.tan_half_fov;
    let half_width = half_height * view.aspect;
    let at = |x: f32, y: f32| {
        view.eye
            + view.forward * GRID_DISTANCE
            + view.right * (x * half_width)
            + view.up * (y * half_height)
    };
    let cell = 2.0 * half_height / ROWS as f32;
    match burst.role {
        Role::Cell { column, row } => {
            let x = (column + 0.5) / COLUMNS as f32 * 2.0 - 1.0;
            let y = 1.0 - (row + 0.5) / ROWS as f32 * 2.0;
            let wave = 0.5 + 0.5 * (time * 7.0 - (x * 4.0 + y * 2.5)).sin();
            let crest = wave.powi(12);
            let flashing = random(burst.seed * 97.0, (time * 12.0).floor()) > 0.96;
            let change = (time * 6.0 + burst.seed * 5.0).floor();
            let pick = random(burst.seed * 31.0, change) * PRIMITIVES.len() as f32;
            Place {
                centre: at(x, y),
                size: cell * 0.42,
                kind: PRIMITIVES[pick as usize % PRIMITIVES.len()],
                seed: random(burst.seed, change + 7.0),
                brightness: 0.8 + 0.8 * wave + 2.4 * crest + if flashing { 2.0 } else { 0.0 },
                whiten: if flashing { 0.8 } else { 0.5 * crest },
            }
        }
        Role::Packet {
            line,
            across,
            offset,
            speed,
        } => {
            let along = (offset + time * speed).rem_euclid(1.0) * 2.2 - 1.1;
            let centre = if across {
                at(along, 1.0 - line / ROWS as f32 * 2.0)
            } else {
                at(line / COLUMNS as f32 * 2.0 - 1.0, along)
            };
            Place {
                centre,
                size: cell * 0.16,
                kind: Kind::Glow,
                seed: burst.seed,
                brightness: 2.2,
                whiten: 0.3,
            }
        }
    }
}

/// The burst's primitives: mostly green, white and gold, with warm white among them.
pub fn primitives() -> Model<SpriteVertex<Burst>> {
    quad_model((0..PRIMITIVE_COUNT).map(|i| {
        let id = i as f32;
        let pick = |salt: f32| random(id, salt + 200.0);
        let direction = random_direction(id * 0.73, 3.0);
        let direction = Vec3::new(
            direction.x * 1.5,
            direction.y * 0.85 + 0.35,
            direction.z * 0.9,
        )
        .normalise();
        let tint = pick(4.0);
        let colour = if tint < 0.45 {
            GREEN_BRIGHT * 1.1
        } else if tint < 0.6 {
            Vec3::new(0.9, 0.95, 0.95)
        } else if tint < 0.85 {
            GOLD * 1.1
        } else {
            WARM * 1.1
        };
        sprite_corners(Burst {
            direction,
            speed: 4.0 + 24.0 * pick(1.0) * pick(1.0).sqrt(),
            delay: 0.14 * pick(2.0),
            size: 0.07 + 0.22 * pick(3.0) * pick(3.0),
            spin: (pick(5.0) - 0.5) * 2.0,
            kind: PRIMITIVES[(pick(6.0) * PRIMITIVES.len() as f32) as usize % PRIMITIVES.len()],
            seed: pick(7.0),
            colour,
            role: role(i, &pick),
            stagger: SETTLE_STAGGER * pick(8.0),
        })
    }))
}

/// The first primitives fill the grid's cells, one each in a scattered order so their flights
/// cross; the rest become packets on its lines.
fn role(i: usize, pick: &impl Fn(f32) -> f32) -> Role {
    const CELLS: usize = COLUMNS * ROWS;
    if i < CELLS {
        // 389 is coprime with CELLS, so every cell is taken once.
        let cell = (i * 389) % CELLS;
        Role::Cell {
            column: (cell % COLUMNS) as f32,
            row: (cell / COLUMNS) as f32,
        }
    } else {
        let across = pick(9.0) < 0.6;
        let lines = if across { ROWS } else { COLUMNS };
        let speed = 0.4 + 1.2 * pick(10.0);
        Role::Packet {
            line: (pick(11.0) * lines as f32).floor(),
            across,
            offset: pick(12.0),
            speed: if pick(13.0) < 0.5 { speed } else { -speed },
        }
    }
}

/// A ray from the die: pre-eruption it flickers short, then at the eruption it shoots out.
#[derive(Clone, Copy)]
pub struct Radial {
    direction: Vec3,
    length: f32,
    order: f32,
    colour: Vec3,
}

pub struct RadialVertexShader;

impl VertexShader for RadialVertexShader {
    type Vertex = StreakVertex<Radial>;
    type Uniforms = BurstUniforms;
    type Varyings = StreakVaryings;

    fn shade(&self, vertex: StreakVertex<Radial>, u: &BurstUniforms) -> (Vec4, StreakVaryings) {
        let ray = &vertex.data;
        let time = u.view.frame.time;
        let age = u.since();
        let streak = if age < 0.0 {
            let build = ramp(time, Phase::Eruption.start() - 2.2, Phase::Eruption.start());
            if build <= ray.order {
                return hidden_streak();
            }
            let centre = die_centre(&u.view.frame);
            let flicker = 0.5 + 0.5 * (time * 41.0 + ray.order * 90.0).sin().abs();
            let reach = (1.0 + 0.5 * ray.length * build) * flicker;
            Streak {
                tail: centre + ray.direction * 0.15,
                head: centre + ray.direction * reach,
                width: 0.9,
                colour: ray.colour * (0.5 + 2.0 * build * build),
                falloff: 1.0,
            }
        } else {
            let along = |t: f32| ray.length * (1.0 - (-4.2 * t.max(0.0)).exp());
            Streak {
                tail: u.origin + ray.direction * along(age - 0.28),
                head: u.origin + ray.direction * along(age),
                width: 1.2,
                colour: ray.colour * (2.4 * (-age * 1.7).exp()),
                falloff: 1.2,
            }
        };
        expand(&u.view, &streak, vertex.corner)
    }
}

pub fn rays() -> Vec<[StreakVertex<Radial>; 4]> {
    (0..RAY_COUNT)
        .map(|i| {
            let id = i as f32;
            let pick = |salt: f32| random(id, salt + 300.0);
            let direction = random_direction(id * 1.31, 5.0);
            let direction =
                Vec3::new(direction.x * 1.6, direction.y * 0.9, direction.z * 0.6).normalise();
            streak_corners(Radial {
                direction,
                length: 8.0 + 20.0 * pick(1.0) * pick(1.0),
                order: pick(2.0),
                colour: if pick(3.0) < 0.55 { WARM } else { GREEN_BRIGHT },
            })
        })
        .collect()
}

/// A shock ring expanding about the die, flat to the camera.
#[derive(Clone, Copy)]
pub struct Shock {
    delay: f32,
}

pub struct ShockVertexShader;

impl VertexShader for ShockVertexShader {
    type Vertex = SpriteVertex<Shock>;
    type Uniforms = BurstUniforms;
    type Varyings = SpriteVaryings;

    fn shade(&self, vertex: SpriteVertex<Shock>, u: &BurstUniforms) -> (Vec4, SpriteVaryings) {
        let age = u.since() - vertex.data.delay;
        if age < 0.0 {
            return hidden();
        }
        let sprite = Sprite {
            centre: u.origin,
            size: 0.5 + 24.0 * (1.0 - (-2.4 * age).exp()),
            angle: 0.0,
            kind: Kind::Ring,
            seed: 0.0,
            colour: WARM * 2.0,
            fade: (-age * 1.7).exp(),
        };
        expand_sprite(&u.view, &sprite, vertex.corner)
    }
}

pub fn shocks() -> Model<SpriteVertex<Shock>> {
    quad_model([0.0, 0.2].map(|delay| sprite_corners(Shock { delay })))
}
