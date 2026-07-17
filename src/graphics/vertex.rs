use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipVertex {
    pub position: Vec4,
    pub colour: Colour,
}
impl ClipVertex {
    pub fn new(position: Vec4, colour: Colour) -> Self {
        Self { position, colour }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            self.position.lerp(&other.position, t),
            self.colour.lerp(&other.colour, t),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex3D {
    pub position: Vec3,
    pub colour: Colour,
}
impl Vertex3D {
    pub fn new(position: Vec3, colour: Colour) -> Self {
        Self { position, colour }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            colour: self.colour.lerp(&other.colour, t),
        }
    }
}

impl From<(f32, f32, f32, Colour)> for Vertex3D {
    fn from((x, y, z, colour): (f32, f32, f32, Colour)) -> Self {
        Self {
            position: Vec3::new(x, y, z),
            colour,
        }
    }
}
