//! The build-up's tension: light and primitives drawn in towards the die, thin at first and
//! thickening as the chip spins up, and the glow that gathers on the die from the hologram to the
//! eruption.

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::director::{Phase, ramp, smootherstep};
use crate::effects::frustum;
use crate::effects::primitives::{
    Kind, PRIMITIVES, Sprite, SpriteVaryings, SpriteVertex, expand as expand_sprite, hidden,
    sprite_corners,
};
use crate::effects::streak::{
    Streak, StreakVaryings, StreakVertex, expand, hidden as hidden_streak, streak_corners,
};
use crate::effects::{GREEN_BRIGHT, WARM, die_centre, quad_model, random, random_direction};
use crate::shaders::ViewRef;

const STREAKS: usize = 110;
const SPRITES: usize = 130;

/// How much is being drawn in, 0 to 1, growing slowly and then fast.
fn density(time: f32) -> f32 {
    ramp(
        time,
        Phase::BuildUp.start() + 0.6,
        Phase::Eruption.start() - 0.1,
    )
    .powi(2)
}

/// The colour of what is drawn in: the data green, going to warm white as the eruption nears.
fn light(time: f32) -> Vec3 {
    let warm = ramp(time, Phase::Eruption.start() - 1.8, Phase::Eruption.start());
    GREEN_BRIGHT + (WARM - GREEN_BRIGHT) * warm
}

/// A line of approach to the die, travelled over and over, differently each time.
#[derive(Clone, Copy)]
pub struct Inflow {
    seed: f32,
    radius: f32,
    period: f32,
    phase: f32,
    /// How far into the build-up it starts to show, 0 to 1.
    threshold: f32,
}

impl Inflow {
    fn new(id: f32, salt: f32) -> Self {
        let pick = |k: f32| random(id + salt, k);
        Self {
            seed: pick(1.0) * 100.0,
            radius: 4.0 + 7.0 * pick(2.0),
            period: 0.55 + 0.6 * pick(3.0),
            phase: pick(4.0) * 20.0,
            threshold: pick(5.0),
        }
    }

    /// How far into this approach it is (0 to 1), where it started, and how far from the die.
    fn at(&self, time: f32) -> (f32, Vec3, f32, f32) {
        let cycles = time / self.period + self.phase;
        let (cycle, u) = (cycles.floor(), cycles.fract());
        let direction = random_direction(self.seed + cycle * 0.37, cycle);
        let start = self.radius * (0.7 + 0.6 * random(self.seed, cycle + 0.5));
        (u, direction, start, (1.0 - u).powf(1.8))
    }
}

pub struct InflowVertexShader;

impl VertexShader for InflowVertexShader {
    type Vertex = StreakVertex<Inflow>;
    type Uniforms = ViewRef;
    type Varyings = StreakVaryings;

    fn shade(&self, vertex: StreakVertex<Inflow>, view: &ViewRef) -> (Vec4, StreakVaryings) {
        let frame = &view.frame;
        let inflow = &vertex.data;
        let build = density(frame.time);
        if build <= inflow.threshold * 0.9 {
            return hidden_streak();
        }
        let (u, direction, start, head) = inflow.at(frame.time);
        let tail = (1.0 - (u - 0.16).max(0.0)).powf(1.8);
        let centre = die_centre(frame);
        let fade = smootherstep((u / 0.1).min(1.0))
            * (1.0 - smootherstep(((u - 0.88) / 0.12).clamp(0.0, 1.0)));
        let streak = Streak {
            tail: centre + direction * (start * tail),
            head: centre + direction * (start * head),
            width: 1.0,
            colour: light(frame.time) * ((0.3 + 0.9 * build) * fade),
            falloff: 1.4,
        };
        expand(view, &streak, vertex.corner)
    }
}

pub fn streaks() -> Vec<[StreakVertex<Inflow>; 4]> {
    (0..STREAKS)
        .map(|i| streak_corners(Inflow::new(i as f32, 0.0)))
        .collect()
}

/// A primitive being drawn in: the inflow's path, a kind and a size.
#[derive(Clone, Copy)]
pub struct Drawn {
    inflow: Inflow,
    kind: Kind,
    size: f32,
    spin: f32,
}

pub struct DrawnVertexShader;

impl VertexShader for DrawnVertexShader {
    type Vertex = SpriteVertex<Drawn>;
    type Uniforms = ViewRef;
    type Varyings = SpriteVaryings;

    fn shade(&self, vertex: SpriteVertex<Drawn>, view: &ViewRef) -> (Vec4, SpriteVaryings) {
        let frame = &view.frame;
        let drawn = &vertex.data;
        let build = density(frame.time);
        if build <= 0.15 + 0.8 * drawn.inflow.threshold {
            return hidden();
        }
        let (u, direction, start, head) = drawn.inflow.at(frame.time);
        let fade = smootherstep((u / 0.1).min(1.0))
            * (1.0 - smootherstep(((u - 0.9) / 0.1).clamp(0.0, 1.0)));
        let sprite = Sprite {
            centre: die_centre(frame) + direction * (start * head),
            size: drawn.size * (0.3 + 0.7 * head),
            angle: drawn.spin * frame.time,
            kind: drawn.kind,
            seed: drawn.inflow.seed.fract(),
            colour: light(frame.time) * 0.8,
            fade: fade * 0.9,
        };
        expand_sprite(view, &sprite, vertex.corner)
    }
}

pub fn primitives() -> Model<SpriteVertex<Drawn>> {
    quad_model((0..SPRITES).map(|i| {
        let id = i as f32;
        sprite_corners(Drawn {
            inflow: Inflow::new(id, 500.0),
            kind: PRIMITIVES
                [(random(id, 9.0) * PRIMITIVES.len() as f32) as usize % PRIMITIVES.len()],
            size: 0.12 + 0.1 * random(id, 10.0),
            spin: (random(id, 11.0) - 0.5) * 3.0,
        })
    }))
}

pub struct GlowVertexShader;

impl VertexShader for GlowVertexShader {
    type Vertex = SpriteVertex<()>;
    type Uniforms = ViewRef;
    type Varyings = SpriteVaryings;

    fn shade(&self, vertex: SpriteVertex<()>, view: &ViewRef) -> (Vec4, SpriteVaryings) {
        let frame = &view.frame;
        let time = frame.time;
        let eruption = Phase::Eruption.start();
        let level = if time < eruption {
            frustum::die_glow(time) * 0.3
                + 0.3 * ramp(time, Phase::BuildUp.start(), eruption - 3.0)
                + 2.5 * ramp(time, eruption - 2.2, eruption).powi(2)
        } else {
            5.0 * (-(time - eruption) * 3.2).exp()
        };
        let warm = ramp(time, eruption - 1.8, eruption);
        let sprite = Sprite {
            centre: die_centre(frame),
            size: 0.8 + 0.7 * level,
            angle: 0.0,
            kind: Kind::Glow,
            seed: 0.0,
            colour: (GREEN_BRIGHT + (WARM - GREEN_BRIGHT) * warm) * level,
            fade: 1.0,
        };
        expand_sprite(view, &sprite, vertex.corner)
    }
}

pub fn glow() -> Model<SpriteVertex<()>> {
    quad_model([sprite_corners(())])
}
