use wide::f32x8;

use crate::prelude::*;

pub struct Fragment<V> {
    pub position: Vec2,
    pub depth: f32,
    pub varyings: V,
}
impl<V> Fragment<V> {
    pub fn new(position: Vec2, depth: f32, varyings: V) -> Self {
        Self {
            position,
            depth,
            varyings,
        }
    }
}

pub struct FragmentSimd<V: SimdInterpolate> {
    pub x_start: i32,
    pub y: i32,
    pub depth: f32x8,
    pub varyings: V::Simd,
}
impl<V: SimdInterpolate> FragmentSimd<V> {
    pub fn new(x_start: i32, y: i32, depth: f32x8, varyings: V::Simd) -> Self {
        Self {
            x_start,
            y,
            depth,
            varyings,
        }
    }
}
