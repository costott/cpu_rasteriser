use crate::prelude::*;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

#[derive(Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Colour {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}
impl Colour {
    pub const WHITE: Colour = Colour {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Colour = Colour {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const RED: Colour = Colour {
        r: 255,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const GREEN: Colour = Colour {
        r: 0,
        g: 255,
        b: 0,
        a: 255,
    };
    pub const BLUE: Colour = Colour {
        r: 0,
        g: 0,
        b: 255,
        a: 255,
    };

    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn to_u32(&self) -> u32 {
        ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32) | ((self.a as u32) << 24)
    }

    pub fn from_u32(colour: u32) -> Self {
        let r = ((colour >> 16) & 0xFF) as u8;
        let g = ((colour >> 8) & 0xFF) as u8;
        let b = (colour & 0xFF) as u8;
        let a = ((colour >> 24) & 0xFF) as u8;
        Self { r, g, b, a }
    }

    pub fn from_f32(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (g.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (b.clamp(0.0, 1.0) * 255.0) as u8;
        let a = (a.clamp(0.0, 1.0) * 255.0) as u8;
        Self { r, g, b, a }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        *self * (1.0 - t) + *other * t
    }
}
impl Add for Colour {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let r = self.r.saturating_add(other.r);
        let g = self.g.saturating_add(other.g);
        let b = self.b.saturating_add(other.b);
        let a = self.a.saturating_add(other.a);
        Colour { r, g, b, a }
    }
}
impl AddAssign for Colour {
    fn add_assign(&mut self, other: Self) {
        self.r = self.r.saturating_add(other.r);
        self.g = self.g.saturating_add(other.g);
        self.b = self.b.saturating_add(other.b);
        self.a = self.a.saturating_add(other.a);
    }
}
impl Sub for Colour {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let r = self.r.saturating_sub(other.r);
        let g = self.g.saturating_sub(other.g);
        let b = self.b.saturating_sub(other.b);
        let a = self.a.saturating_sub(other.a);
        Colour { r, g, b, a }
    }
}
impl Mul<f32> for Colour {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        let r = (self.r as f32 * scalar).clamp(0.0, 255.0) as u8;
        let g = (self.g as f32 * scalar).clamp(0.0, 255.0) as u8;
        let b = (self.b as f32 * scalar).clamp(0.0, 255.0) as u8;
        let a = (self.a as f32 * scalar).clamp(0.0, 255.0) as u8;
        Self::new(r, g, b, a)
    }
}
impl Mul for Colour {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let r = ((self.r as u16 * other.r as u16) / 255) as u8;
        let g = ((self.g as u16 * other.g as u16) / 255) as u8;
        let b = ((self.b as u16 * other.b as u16) / 255) as u8;
        let a = ((self.a as u16 * other.a as u16) / 255) as u8;
        Self::new(r, g, b, a)
    }
}
impl Div<f32> for Colour {
    type Output = Self;

    fn div(self, scalar: f32) -> Self {
        let r = (self.r as f32 / scalar).clamp(0.0, 255.0) as u8;
        let g = (self.g as f32 / scalar).clamp(0.0, 255.0) as u8;
        let b = (self.b as f32 / scalar).clamp(0.0, 255.0) as u8;
        let a = (self.a as f32 / scalar).clamp(0.0, 255.0) as u8;
        Self::new(r, g, b, a)
    }
}

impl From<Vec4> for Colour {
    fn from(vec: Vec4) -> Self {
        Self::new(
            vec.x.clamp(0.0, 255.0) as u8,
            vec.y.clamp(0.0, 255.0) as u8,
            vec.z.clamp(0.0, 255.0) as u8,
            vec.w.clamp(0.0, 255.0) as u8,
        )
    }
}
impl From<Colour> for Vec4 {
    fn from(colour: Colour) -> Self {
        Self::new(
            colour.r as f32,
            colour.g as f32,
            colour.b as f32,
            colour.a as f32,
        )
    }
}
