use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vertex2D {
    pub position: Vec2,
    pub colour: Colour,
    pub depth: f32,
}
impl Vertex2D {
    pub fn new(position: Vec2, colour: Colour, depth: f32) -> Self {
        Self {
            position,
            colour,
            depth,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            colour: self.colour.lerp(&other.colour, t),
            depth: self.depth * (1.0 - t) + other.depth * t,
        }
    }
}

impl From<(f32, f32, Colour, f32)> for Vertex2D {
    fn from((x, y, colour, depth): (f32, f32, Colour, f32)) -> Self {
        Self {
            position: Vec2::new(x, y),
            colour,
            depth,
        }
    }
}
