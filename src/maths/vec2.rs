use std::ops::{Add, Mul};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}
impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        *self * (1.0 - t) + *other * t
    }
}
impl Add for Vec2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl From<(f32, f32)> for Vec2 {
    fn from(tuple: (f32, f32)) -> Self {
        Self {
            x: tuple.0,
            y: tuple.1,
        }
    }
}
impl From<(usize, usize)> for Vec2 {
    fn from(tuple: (usize, usize)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
        }
    }
}
impl From<(i32, i32)> for Vec2 {
    fn from(tuple: (i32, i32)) -> Self {
        Self {
            x: tuple.0 as f32,
            y: tuple.1 as f32,
        }
    }
}
