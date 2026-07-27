use std::ops::{Add, Mul, Sub};

use crate::prelude::*;

/// Represents a 2D vertex in screen space
#[derive(Clone)]
pub struct RasterVertex<V>
where
    V: Interpolate,
{
    pub position: Vec2,
    pub depth: f32,
    pub inv_w: f32,
    pub varyings: V,
}
impl<V: Interpolate> RasterVertex<V> {
    pub fn new(position: Vec2, depth: f32, inv_w: f32, varyings: V) -> Self {
        Self {
            position,
            depth,
            inv_w,
            varyings,
        }
    }
}

#[derive(Clone)]
pub struct ClipVertex<V>
where
    V: Interpolate,
{
    pub position: Vec4,
    pub varyings: V,
}
impl<V: Interpolate> ClipVertex<V> {
    pub fn new(position: Vec4, varyings: V) -> Self {
        Self { position, varyings }
    }

    pub fn interpolate(&self, other: &Self, t: f32) -> Self
    where
        V: Interpolate,
    {
        Self {
            position: self.position.lerp(&other.position, t),
            varyings: self.varyings.interpolate(&other.varyings, t),
        }
    }
}

#[derive(Clone)]
pub struct ObjVertex {
    pub position: Vec3,
    pub colour: Colour,
    pub normal: Vec3,
    // TODO: UV
}
impl ObjVertex {
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
