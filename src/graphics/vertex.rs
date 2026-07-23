use crate::prelude::*;

/// Represents a 2D vertex with position, colour, normal, and depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex2D {
    pub position: Vec2,
    pub world_position: Vec3,
    pub colour: Colour,
    pub normal: Vec3,
    pub depth: f32,

    pub inv_w: f32,
}
impl Vertex2D {
    pub fn new(
        position: Vec2,
        world_position: Vec3,
        colour: Colour,
        normal: Vec3,
        depth: f32,
        inv_w: f32,
    ) -> Self {
        Self {
            position,
            world_position,
            colour,
            normal,
            depth,
            inv_w,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            world_position: self.world_position.lerp(&other.world_position, t),
            colour: self.colour.lerp(&other.colour, t),
            normal: self.normal.lerp(&other.normal, t),
            depth: self.depth * (1.0 - t) + other.depth * t,
            inv_w: self.inv_w * (1.0 - t) + other.inv_w * t,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipVertex {
    pub position: Vec4,
    pub world_position: Vec3,
    pub colour: Colour,
    pub normal: Vec3,
}
impl ClipVertex {
    pub fn new(position: Vec4, world_position: Vec3, colour: Colour, normal: Vec3) -> Self {
        Self {
            position,
            world_position,
            colour,
            normal,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            self.position.lerp(&other.position, t),
            self.world_position.lerp(&other.world_position, t),
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

    pub fn new_with_normal(position: Vec3, colour: Colour, normal: Vec3) -> Self {
        Self {
            position,
            colour,
            normal,
        }
    }
}
