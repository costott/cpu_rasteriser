use std::ops::Add;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vertex2D {
    pub position: Vec2,
    pub colour: Colour,
}
impl Vertex2D {
    pub fn new(position: Vec2, colour: Colour) -> Self {
        Self { position, colour }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            colour: self.colour.lerp(&other.colour, t),
        }
    }
}

impl From<(f32, f32, Colour)> for Vertex2D {
    fn from((x, y, colour): (f32, f32, Colour)) -> Self {
        Self {
            position: Vec2::new(x, y),
            colour,
        }
    }
}
