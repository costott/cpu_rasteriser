//! What leaks out of the frustum's borders: tiny triangles, wireframe fragments, depth bands and
//! hex digits that creep out of its edges, drift, and fade.

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::director::Phase;
use crate::effects::frustum::{corner, volume};
use crate::effects::primitives::{
    Kind, Sprite, SpriteVaryings, SpriteVertex, expand, hidden, sprite_corners,
};
use crate::effects::{GREEN_BRIGHT, quad_model, random};
use crate::shaders::ViewRef;

const COUNT: usize = 420;
/// When the first leak is born and when the last is.
const BORN: (f32, f32) = (
    Phase::Hologram.start() + 2.4,
    Phase::Hologram.start() + 11.2,
);
/// The kinds that leak, in proportion.
const KINDS: [Kind; 20] = [
    Kind::Triangle,
    Kind::Triangle,
    Kind::Triangle,
    Kind::Triangle,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::DepthBands,
    Kind::DepthBands,
    Kind::Hex,
    Kind::Hex,
    Kind::Hex,
    Kind::Hex,
    Kind::Hex,
    Kind::Hex,
    Kind::Tile,
    Kind::Normal,
];

/// One leaking fragment: where on the frustum's border it starts, which way it creeps, and its
/// life.
#[derive(Clone, Copy)]
pub struct Leak {
    anchor: Vec3,
    outward: Vec3,
    born: f32,
    life: f32,
    kind: Kind,
    seed: f32,
    size: f32,
    spin: f32,
    colour: Vec3,
}

pub struct LeakVertexShader;

impl VertexShader for LeakVertexShader {
    type Vertex = SpriteVertex<Leak>;
    type Uniforms = ViewRef;
    type Varyings = SpriteVaryings;

    fn shade(&self, vertex: SpriteVertex<Leak>, view: &ViewRef) -> (Vec4, SpriteVaryings) {
        let frame = &view.frame;
        let leak = &vertex.data;
        let age = frame.time - leak.born;
        if age < 0.0 || age > leak.life {
            return hidden();
        }
        let wobble = Vec3::new(
            (age * 1.7 + leak.seed * 20.0).sin(),
            (age * 1.3 + leak.seed * 31.0).sin() * 0.5,
            (age * 1.1 + leak.seed * 11.0).cos(),
        ) * 0.07;
        let local = leak.anchor + leak.outward * (0.3 * age + 0.06 * age * age) + wobble;
        let appear = (age / 0.3).min(1.0);
        let vanish = ((leak.life - age) / (leak.life * 0.5)).clamp(0.0, 1.0);
        let flicker = 0.85 + 0.15 * (age * 23.0 + leak.seed * 50.0).sin();
        let sprite = Sprite {
            centre: (frame.chip * local.to_point4()).xyz(),
            size: leak.size,
            angle: leak.spin * age + leak.seed * 6.0,
            kind: leak.kind,
            seed: leak.seed,
            colour: leak.colour,
            fade: appear * vanish * volume(frame.time) * flicker,
        };
        expand(view, &sprite, vertex.corner)
    }
}

/// Every leak, as a model of one quad each.
pub fn leaks() -> Model<SpriteVertex<Leak>> {
    quad_model((0..COUNT).map(|i| {
        let id = i as f32;
        let pick = |salt: f32| random(id, salt);
        let on_rim = pick(1.0) < 0.45;
        let side = (pick(2.0) * 4.0) as usize % 4;
        let (anchor, outward) = if on_rim {
            let (a, b) = (corner(side, 1.0), corner(side + 1, 1.0));
            let middle = (a + b) * 0.5;
            let away = Vec3::new(middle.x, 0.0, middle.z).normalise();
            (a + (b - a) * pick(3.0), away + Vec3::new(0.0, 0.5, 0.0))
        } else {
            let level = 0.08 + 0.92 * pick(3.0);
            let c = corner(side, level);
            let away = Vec3::new(c.x, 0.0, c.z).normalise();
            (c, away + Vec3::new(0.0, 0.35, 0.0))
        };
        let kind = KINDS[(pick(4.0) * KINDS.len() as f32) as usize % KINDS.len()];
        let tint = pick(5.0);
        let colour = if tint < 0.6 {
            GREEN_BRIGHT * 0.7
        } else if tint < 0.8 {
            Vec3::new(0.7, 0.75, 0.75)
        } else {
            Vec3::new(0.94, 0.84, 0.54) * 0.6
        };
        sprite_corners(Leak {
            anchor,
            outward,
            born: BORN.0 + (BORN.1 - BORN.0) * (id + pick(6.0)) / COUNT as f32,
            life: 2.0 + 1.8 * pick(7.0),
            kind,
            seed: pick(8.0),
            size: 0.05 + 0.05 * pick(9.0),
            spin: (pick(10.0) - 0.5) * 0.8,
            colour,
        })
    }))
}
