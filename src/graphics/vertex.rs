use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex2D {
    pub position: Vec2,
    pub colour: Colour,
    pub normal: Vec3,
    pub depth: f32,
}
impl Vertex2D {
    pub fn new(position: Vec2, colour: Colour, normal: Vec3, depth: f32) -> Self {
        Self {
            position,
            colour,
            normal,
            depth,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            colour: self.colour.lerp(&other.colour, t),
            normal: self.normal.lerp(&other.normal, t),
            depth: self.depth * (1.0 - t) + other.depth * t,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipVertex {
    pub position: Vec4,
    pub colour: Colour,
    pub normal: Vec3,
}
impl ClipVertex {
    pub fn new(position: Vec4, colour: Colour, normal: Vec3) -> Self {
        Self {
            position,
            colour,
            normal,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            self.position.lerp(&other.position, t),
            self.colour.lerp(&other.colour, t),
            self.normal.lerp(&other.normal, t),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex3D {
    pub position: Vec3,
    pub colour: Colour,
    pub normal: Vec3,
}
impl Vertex3D {
    pub fn new(position: Vec3, colour: Colour) -> Self {
        Self {
            position,
            colour,
            normal: Vec3::ZERO,
        }
    }
}
