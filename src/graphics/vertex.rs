use std::ops::{Add, Mul, Sub};

use crate::prelude::*;

/// Represents a 2D vertex with position, colour, normal, and depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterVertex {
    pub position: Vec2,
    pub varyings: RasterVaryings,
}
impl RasterVertex {
    pub fn new(position: Vec2, varyings: RasterVaryings) -> Self {
        Self { position, varyings }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            varyings: self.varyings.lerp(&other.varyings, t),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterVaryings {
    pub world_position: Vec3,
    pub colour: Vec3,
    pub normal: Vec3,
    pub depth: f32,

    pub inv_w: f32,
}
impl RasterVaryings {
    pub fn new(world_position: Vec3, colour: Vec3, normal: Vec3, depth: f32, inv_w: f32) -> Self {
        Self {
            world_position,
            colour,
            normal,
            depth,
            inv_w,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        *self * (1.0 - t) + *other * t
    }
}
impl Add for RasterVaryings {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            world_position: self.world_position + other.world_position,
            colour: self.colour + other.colour,
            normal: self.normal + other.normal,
            depth: self.depth + other.depth,
            inv_w: self.inv_w + other.inv_w,
        }
    }
}
impl Sub for RasterVaryings {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            world_position: self.world_position - other.world_position,
            colour: self.colour - other.colour,
            normal: self.normal - other.normal,
            depth: self.depth - other.depth,
            inv_w: self.inv_w - other.inv_w,
        }
    }
}
impl Mul<f32> for RasterVaryings {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            world_position: self.world_position * scalar,
            colour: self.colour * scalar,
            normal: self.normal * scalar,
            depth: self.depth * scalar,
            inv_w: self.inv_w * scalar,
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
}
