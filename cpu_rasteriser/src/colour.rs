use wide::f32x8;

use crate::prelude::*;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub};

// Basically a wrapper over a vec4 that doesn't affect the alpha channel when doing operations like addition,
// subtraction, multiplication, and division. This is useful for colour blending where we don't want to change
// the alpha channel when blending colours together.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Colour {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}
impl Colour {
    pub const WHITE: Colour = Colour {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Colour = Colour {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const RED: Colour = Colour {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Colour = Colour {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Colour = Colour {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    pub const TRANSPARENT: Colour = Colour {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    #[inline(always)]
    pub fn to_u32(self) -> u32 {
        let r = (self.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.b.clamp(0.0, 1.0) * 255.0) as u32;
        let a = (self.a.clamp(0.0, 1.0) * 255.0) as u32;

        (a << 24) | (r << 16) | (g << 8) | b
    }

    #[inline(always)]
    pub fn from_u32(value: u32) -> Self {
        let r = (value & 0xff) as f32 / 255.0;
        let g = ((value >> 8) & 0xff) as f32 / 255.0;
        let b = ((value >> 16) & 0xff) as f32 / 255.0;
        let a = ((value >> 24) & 0xff) as f32 / 255.0;

        Self::new(r, g, b, a)
    }

    pub fn scale_all(self, scale: f32) -> Self {
        Self::new(
            self.r * scale,
            self.g * scale,
            self.b * scale,
            self.a * scale,
        )
    }

    pub fn clamp(self, min: f32, max: f32) -> Self {
        Self::new(
            self.r.clamp(min, max),
            self.g.clamp(min, max),
            self.b.clamp(min, max),
            self.a.clamp(min, max),
        )
    }

    pub fn clamp_rgb(self, min: f32, max: f32) -> Self {
        Self::new(
            self.r.clamp(min, max),
            self.g.clamp(min, max),
            self.b.clamp(min, max),
            self.a,
        )
    }

    pub fn luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    #[inline(always)]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )
    }
}
impl Add for Colour {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a,
        }
    }
}
impl AddAssign for Colour {
    fn add_assign(&mut self, other: Self) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}
impl Sub for Colour {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
            a: self.a,
        }
    }
}
impl Mul<f32> for Colour {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            r: self.r * scalar,
            g: self.g * scalar,
            b: self.b * scalar,
            a: self.a,
        }
    }
}
impl Mul for Colour {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
            a: self.a,
        }
    }
}
impl MulAssign<f32> for Colour {
    fn mul_assign(&mut self, scalar: f32) {
        self.r *= scalar;
        self.g *= scalar;
        self.b *= scalar;
    }
}
impl Div<f32> for Colour {
    type Output = Self;

    fn div(self, scalar: f32) -> Self {
        Self {
            r: self.r / scalar,
            g: self.g / scalar,
            b: self.b / scalar,
            a: self.a,
        }
    }
}
impl DivAssign<f32> for Colour {
    fn div_assign(&mut self, scalar: f32) {
        self.r /= scalar;
        self.g /= scalar;
        self.b /= scalar;
    }
}

impl From<Vec4> for Colour {
    fn from(vec: Vec4) -> Self {
        Self::new(vec.x, vec.y, vec.z, vec.w)
    }
}
impl From<Colour> for Vec4 {
    fn from(colour: Colour) -> Self {
        Self::new(colour.r, colour.g, colour.b, colour.a)
    }
}
impl From<Vec3> for Colour {
    fn from(vec: Vec3) -> Self {
        Self::new(vec.x, vec.y, vec.z, 1.0)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ColourSimd {
    pub r: f32x8,
    pub g: f32x8,
    pub b: f32x8,
    pub a: f32x8,
}
impl ColourSimd {
    #[inline(always)]
    pub fn splat(colour: Colour) -> Self {
        Self {
            r: f32x8::splat(colour.r),
            g: f32x8::splat(colour.g),
            b: f32x8::splat(colour.b),
            a: f32x8::splat(colour.a),
        }
    }

    #[inline(always)]
    pub fn lerp(self, other: Self, t: f32x8) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}
