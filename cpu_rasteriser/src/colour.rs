use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub};

use wide::f32x8;

use crate::prelude::*;

/// An RGBA colour with one `f32` per channel.
///
/// Channels are nominally in `0.0..=1.0`, but they are not clamped: values above 1 (HDR) and below
/// 0 are kept through shading and blending. Clamping happens only when a colour is packed into
/// bytes by [`Colour::to_u32`], or explicitly with [`Colour::clamp`] and [`Colour::clamp_rgb`].
/// No colour space is implied; the crate treats channels as linear values. Alpha is straight, not
/// premultiplied.
///
/// # Arithmetic keeps the left-hand alpha
///
/// The operators combine only the red, green, and blue channels, component-wise, and the result
/// takes the alpha of the left-hand operand:
///
/// - `a + b`, `a - b`, `a += b`: per-channel sum or difference of r, g, and b.
/// - `a * b`: per-channel (modulating) product of r, g, and b.
/// - `a * s`, `a / s`, `a *= s`, `a /= s`: r, g, and b scaled by the scalar `s`.
///
/// To scale alpha as well, use [`Colour::scale_all`]; [`Colour::lerp`] also interpolates alpha.
///
/// # Conversions
///
/// `From<Vec4>` and `Into<Vec4>` map `(r, g, b, a)` to `(x, y, z, w)`, and `From<Vec3>` sets
/// alpha to 1. Shaders usually pass colours between stages as [`Vec4`] varyings, because `Colour`
/// does not implement [`Interpolate`](crate::Interpolate).
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let base = Colour::new(0.8, 0.6, 0.4, 0.5);
/// let light = Colour::rgb(0.5, 0.5, 0.5);
///
/// // Modulation and scaling keep the base colour's alpha.
/// let lit = base * light * 2.0;
/// assert_eq!(lit, Colour::new(0.8, 0.6, 0.4, 0.5));
///
/// // Values above 1 survive until they are packed for display.
/// let bright = base * 4.0;
/// assert!(bright.r > 1.0);
/// assert_eq!(bright.to_u32(), 0x7f_ff_ff_ff);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Colour {
    /// The red channel.
    pub r: f32,
    /// The green channel.
    pub g: f32,
    /// The blue channel.
    pub b: f32,
    /// The alpha (opacity) channel, where 0 is fully transparent and 1 fully opaque.
    pub a: f32,
}
impl Colour {
    /// Opaque white, `(1, 1, 1, 1)`.
    pub const WHITE: Colour = Colour {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    /// Opaque black, `(0, 0, 0, 1)`.
    pub const BLACK: Colour = Colour {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    /// Opaque red, `(1, 0, 0, 1)`.
    pub const RED: Colour = Colour {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    /// Opaque green, `(0, 1, 0, 1)`.
    pub const GREEN: Colour = Colour {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    /// Opaque blue, `(0, 0, 1, 1)`.
    pub const BLUE: Colour = Colour {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    /// Fully transparent black, `(0, 0, 0, 0)`.
    pub const TRANSPARENT: Colour = Colour {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Creates a colour from its four channels.
    ///
    /// The values are stored as given, without clamping.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let translucent_orange = Colour::new(1.0, 0.5, 0.0, 0.25);
    /// assert_eq!(translucent_orange.g, 0.5);
    /// assert_eq!(translucent_orange.a, 0.25);
    /// ```
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Creates an opaque colour (alpha 1) from its red, green, and blue channels.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Colour::rgb(1.0, 0.0, 0.0), Colour::RED);
    /// ```
    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    /// Unpacks a colour from a `u32` holding one byte per channel, as `0xAABBGGRR`.
    ///
    /// Red is read from the lowest byte, then green, blue and, in the highest byte, alpha. Each
    /// byte is divided by 255. A value written without an alpha byte, such as `0xff0000`, has
    /// alpha 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Colour::from_u32(0xff_00_00_ff), Colour::RED);
    /// assert_eq!(Colour::from_u32(0xff_ff_00_00), Colour::BLUE);
    /// ```
    ///
    /// # Known issues
    ///
    /// - The byte order is not the `0xAARRGGBB` that [`Colour::to_u32`] writes, so
    ///   `Colour::from_u32(c.to_u32())` swaps red and blue
    ///   ([#22](https://github.com/costott/cpu_rasteriser/issues/22)).
    #[inline(always)]
    pub fn from_u32(value: u32) -> Self {
        let r = (value & 0xff) as f32 / 255.0;
        let g = ((value >> 8) & 0xff) as f32 / 255.0;
        let b = ((value >> 16) & 0xff) as f32 / 255.0;
        let a = ((value >> 24) & 0xff) as f32 / 255.0;

        Self::new(r, g, b, a)
    }

    /// Packs the colour into a `u32` as `0xAARRGGBB`: alpha in the highest byte, then red, green,
    /// and blue in the lowest.
    ///
    /// Each channel is clamped to `0.0..=1.0`, multiplied by 255, and truncated towards zero, so
    /// HDR values saturate at 255 and a NaN channel becomes 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Colour::RED.to_u32(), 0xff_ff_00_00);
    /// assert_eq!(Colour::new(2.0, -1.0, 0.5, 1.0).to_u32(), 0xff_ff_00_7f);
    /// ```
    #[inline(always)]
    pub fn to_u32(self) -> u32 {
        let r = (self.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.b.clamp(0.0, 1.0) * 255.0) as u32;
        let a = (self.a.clamp(0.0, 1.0) * 255.0) as u32;

        (a << 24) | (r << 16) | (g << 8) | b
    }

    /// Multiplies all four channels, alpha included, by `scale`.
    ///
    /// `self * scale` leaves alpha unchanged; use this to fade a colour out entirely or to
    /// premultiply by a coverage factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// assert_eq!(Colour::WHITE.scale_all(0.5), Colour::new(0.5, 0.5, 0.5, 0.5));
    /// assert_eq!((Colour::WHITE * 0.5).a, 1.0);
    /// ```
    pub fn scale_all(self, scale: f32) -> Self {
        Self::new(
            self.r * scale,
            self.g * scale,
            self.b * scale,
            self.a * scale,
        )
    }

    /// Clamps all four channels, alpha included, to `min..=max`.
    ///
    /// A NaN channel stays NaN, as with [`f32::clamp`].
    ///
    /// # Panics
    ///
    /// Panics if `min > max`, or if `min` or `max` is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let hdr = Colour::new(1.5, 0.5, -0.25, 2.0);
    /// assert_eq!(hdr.clamp(0.0, 1.0), Colour::new(1.0, 0.5, 0.0, 1.0));
    /// ```
    pub fn clamp(self, min: f32, max: f32) -> Self {
        Self::new(
            self.r.clamp(min, max),
            self.g.clamp(min, max),
            self.b.clamp(min, max),
            self.a.clamp(min, max),
        )
    }

    /// Clamps the red, green, and blue channels to `min..=max`, leaving alpha unchanged.
    ///
    /// A NaN channel stays NaN, as with [`f32::clamp`].
    ///
    /// # Panics
    ///
    /// Panics if `min > max`, or if `min` or `max` is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let hdr = Colour::new(1.5, 0.5, -0.25, 2.0);
    /// assert_eq!(hdr.clamp_rgb(0.0, 1.0), Colour::new(1.0, 0.5, 0.0, 2.0));
    /// ```
    pub fn clamp_rgb(self, min: f32, max: f32) -> Self {
        Self::new(
            self.r.clamp(min, max),
            self.g.clamp(min, max),
            self.b.clamp(min, max),
            self.a,
        )
    }

    /// Returns the relative luminance, `0.2126 r + 0.7152 g + 0.0722 b`.
    ///
    /// The weights are the Rec. 709 coefficients and assume linear channel values. Alpha is
    /// ignored, and channels are not clamped first.
    ///
    /// # Examples
    ///
    /// Converting a colour to greyscale:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let y = Colour::GREEN.luminance();
    /// let grey = Colour::rgb(y, y, y);
    ///
    /// assert_eq!(grey.g, 0.7152);
    /// assert!(Colour::GREEN.luminance() > Colour::RED.luminance());
    /// ```
    pub fn luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Linearly interpolates all four channels from `self` (at `t = 0`) to `other` (at `t = 1`).
    ///
    /// `t` is not clamped, so values outside `0.0..=1.0` extrapolate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let half_faded = Colour::BLACK.lerp(Colour::TRANSPARENT, 0.5);
    /// assert_eq!(half_faded, Colour::new(0.0, 0.0, 0.0, 0.5));
    /// ```
    #[inline(always)]
    pub fn lerp(self, other: Self, t: f32) -> Self {
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

/// Eight RGBA colours in struct-of-arrays form, one [`f32x8`] per channel.
///
/// Lane `i` of `r`, `g`, `b`, and `a` together form the `i`th colour, with the channels
/// represented as in [`Colour`]. [`FragmentShaderSimd`](crate::FragmentShaderSimd) describes how
/// the lanes map to pixels.
///
/// # Examples
///
/// Shading a horizontal gradient across the eight lanes:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use cpu_rasteriser::wide::f32x8;
///
/// let t = f32x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / f32x8::splat(7.0);
/// let black = ColourSimd::splat(Colour::BLACK);
/// let white = ColourSimd::splat(Colour::WHITE);
/// let gradient = black.lerp(white, t);
///
/// assert_eq!(gradient.r.to_array()[0], 0.0);
/// assert_eq!(gradient.r.to_array()[7], 1.0);
/// assert_eq!(gradient.a, f32x8::splat(1.0));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ColourSimd {
    /// The red channel of each lane.
    pub r: f32x8,
    /// The green channel of each lane.
    pub g: f32x8,
    /// The blue channel of each lane.
    pub b: f32x8,
    /// The alpha channel of each lane.
    pub a: f32x8,
}
impl ColourSimd {
    /// Creates a `ColourSimd` with `colour` in every lane.
    ///
    /// Use it for a colour that is constant across the group, such as a solid fill or a uniform
    /// tint.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use cpu_rasteriser::wide::f32x8;
    ///
    /// let red = ColourSimd::splat(Colour::RED);
    /// assert_eq!(red.r, f32x8::splat(1.0));
    /// assert_eq!(red.g, f32x8::splat(0.0));
    /// ```
    #[inline(always)]
    pub fn splat(colour: Colour) -> Self {
        Self {
            r: f32x8::splat(colour.r),
            g: f32x8::splat(colour.g),
            b: f32x8::splat(colour.b),
            a: f32x8::splat(colour.a),
        }
    }

    /// Linearly interpolates each lane's four channels from `self` (at `t = 0`) to `other` (at
    /// `t = 1`), using that lane of `t`.
    ///
    /// `t` is not clamped, so values outside `0.0..=1.0` extrapolate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use cpu_rasteriser::wide::f32x8;
    ///
    /// let red = ColourSimd::splat(Colour::RED);
    /// let blue = ColourSimd::splat(Colour::BLUE);
    /// let t = f32x8::from([0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]);
    /// let mixed = red.lerp(blue, t);
    ///
    /// assert_eq!(mixed.r.to_array()[1], 0.75);
    /// assert_eq!(mixed.b.to_array()[1], 0.25);
    /// ```
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
