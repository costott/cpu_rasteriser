//! Textures, and samplers that read them at texture coordinates.
//!
//! A [`Texture`] holds an image's pixels; a [`TextureSampler`] made from it reads them in a
//! shader, with a [`WrapMode`] and a [`FilterMode`].
//!
//! # Coordinate conventions
//!
//! - Pixels are stored row by row, and row 0 is the top of the image.
//! - Texture coordinates (`uv`) have their origin at the bottom-left of the image, with u
//!   increasing to the right and v upwards.
//! - u = 0 and u = 1 are the centres of the first and last columns, and v = 0 and v = 1 the
//!   centres of the bottom and top rows: the texel position is `u * (width - 1)` across and
//!   `(1 - v) * (height - 1)` down. This differs from the GPU convention, which puts 0 and 1 at
//!   the image's outer edges.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use cpu_rasteriser::wide::f32x8;

/// An image held in memory, to be read through a [`TextureSampler`].
///
/// Load one from an image file with [`Texture::from_image`], or copy a render target's pixels
/// with [`Texture::from_render_target`] to sample an earlier render pass. The pixels are shared:
/// cloning the texture or making a sampler doesn't copy them.
///
/// [`pixels`](Texture::pixels) should hold `width * height` colours, in the order the
/// [module documentation](self) describes. This isn't checked until a sampler is made.
///
/// # Examples
///
/// See [`Texture::from_image`] and [`Texture::update_from`].
#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    /// The width in pixels.
    pub width: u32,
    /// The height in pixels.
    pub height: u32,
    /// The pixels, row by row from the top.
    pub pixels: Arc<[Colour]>,
}
impl Texture {
    /// Loads a texture from an image file, in any format the `image` crate supports.
    ///
    /// Each channel is the file's 8-bit value divided by 255, so the colours lie in `0..=1`. No
    /// colour-space conversion is made, so an sRGB image's colours stay sRGB-encoded.
    ///
    /// # Errors
    ///
    /// Returns [`TextureError::Image`] if the file can't be opened or decoded.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use engine::prelude::*;
    ///
    /// let texture = Texture::from_image("../assets/dice/texture.png")?;
    /// let sampler = texture.sampler(WrapMode::Repeat, FilterMode::Linear);
    ///
    /// assert_eq!((sampler.width(), sampler.height()), (320, 320));
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_image(path: impl AsRef<std::path::Path>) -> Result<Self, TextureError> {
        let path = path.as_ref();
        let image = image::open(path)
            .map_err(|source| TextureError::Image {
                path: path.to_path_buf(),
                source,
            })?
            .to_rgba8();

        let pixels = image
            .pixels()
            .map(|pixel| {
                Colour::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                    pixel[3] as f32 / 255.0,
                )
            })
            .collect::<Vec<_>>()
            .into();

        Ok(Self {
            width: image.width(),
            height: image.height(),
            pixels,
        })
    }

    /// Creates a texture holding a copy of the render target's current pixels.
    ///
    /// To copy from the same target repeatedly, keep the texture and refill it with
    /// [`Texture::update_from`].
    ///
    /// # Examples
    ///
    /// See [`Texture::update_from`].
    pub fn from_render_target(render_target: &RenderTarget) -> Self {
        Self {
            width: render_target.width() as u32,
            height: render_target.height() as u32,
            pixels: Self::copy_pixels(render_target),
        }
    }

    /// Replaces the texture's contents with the render target's current pixels.
    ///
    /// Samplers and clones made earlier keep the pixels they were made with. This allocates new
    /// storage if the size has changed or any of them is still alive, so drop them first to
    /// refill the texture without allocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let target = RenderTarget::new(Extent::new(4, 2));
    /// let mut texture = Texture::from_render_target(&target);
    ///
    /// for _frame in 0..3 {
    ///     // ... render into `target` ...
    ///     texture.update_from(&target);
    ///     let sampler = texture.sampler(WrapMode::Clamp, FilterMode::Linear);
    ///     assert_eq!((sampler.width(), sampler.height()), (4, 2));
    /// }
    /// ```
    pub fn update_from(&mut self, render_target: &RenderTarget) {
        let len = render_target.width() * render_target.height();

        if let Some(pixels) = Arc::get_mut(&mut self.pixels).filter(|pixels| pixels.len() == len) {
            render_target.write_pixels(pixels);
        } else {
            self.pixels = Self::copy_pixels(render_target);
        }

        self.width = render_target.width() as u32;
        self.height = render_target.height() as u32;
    }

    fn copy_pixels(render_target: &RenderTarget) -> Arc<[Colour]> {
        let len = render_target.width() * render_target.height();
        let mut pixels: Arc<[Colour]> = std::iter::repeat_n(Colour::BLACK, len).collect();
        render_target.write_pixels(Arc::get_mut(&mut pixels).expect("a new Arc is unique"));
        pixels
    }

    /// Returns a sampler that reads the texture's pixels with the given wrap and filter modes.
    ///
    /// The sampler keeps the pixels as they are now: a later
    /// [`update_from`](Texture::update_from) doesn't change what it reads.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is zero or greater than 2^24, or if `pixels` holds fewer than
    /// `width * height` colours, as [`TextureSampler::new`] does.
    ///
    /// # Examples
    ///
    /// See [`Texture::from_image`].
    pub fn sampler(&self, wrap_mode: WrapMode, filter_mode: FilterMode) -> TextureSampler {
        TextureSampler::new(
            self.pixels.clone(),
            self.width,
            self.height,
            wrap_mode,
            filter_mode,
        )
    }
}

/// Reads a texture's pixels at texture coordinates, with a [`WrapMode`] and a [`FilterMode`].
///
/// [`sample`](TextureSampler::sample) applies the sampler's own modes. The `_clamp` methods always
/// clamp, with a fixed filter, whatever the sampler's modes; the `_simd` ones sample eight
/// coordinates at once for a [`FragmentShaderSimd`]. See the [module documentation](self) for
/// how coordinates map to texels.
///
/// A sampler is cheap to clone.
///
/// # Examples
///
/// A 1×2 texture, red on top of blue:
///
/// ```
/// use std::sync::Arc;
///
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let pixels: Arc<[Colour]> = Arc::from([Colour::RED, Colour::BLUE]);
/// let sampler = TextureSampler::new(pixels, 1, 2, WrapMode::Clamp, FilterMode::Linear);
///
/// assert_eq!(sampler.sample(Vec2::new(0.0, 1.0)), Colour::RED);
/// assert_eq!(sampler.sample(Vec2::new(0.0, 0.0)), Colour::BLUE);
/// assert_eq!(sampler.sample(Vec2::new(0.0, 0.5)), Colour::new(0.5, 0.0, 0.5, 1.0));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TextureSampler {
    pixels: Arc<[Colour]>,
    width: u32,
    height: u32,
    wrap_mode: WrapMode,
    filter_mode: FilterMode,
}
impl TextureSampler {
    /// The largest supported width or height. Up to this value every `width - 1` and
    /// `height - 1` converts to `f32` exactly, which keeps the clamped texel coordinates
    /// computed in `f32` inside the texture.
    const MAX_DIMENSION: u32 = 1 << f32::MANTISSA_DIGITS;

    /// Creates a sampler of `width` by `height` pixels, stored row by row from the top.
    ///
    /// Pixels beyond the first `width * height` are ignored.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is zero or greater than 2^24, or if `pixels` holds fewer
    /// than `width * height` pixels.
    ///
    /// # Examples
    ///
    /// See [`TextureSampler`].
    pub fn new(
        pixels: Arc<[Colour]>,
        width: u32,
        height: u32,
        wrap_mode: WrapMode,
        filter_mode: FilterMode,
    ) -> Self {
        assert!(
            (1..=Self::MAX_DIMENSION).contains(&width)
                && (1..=Self::MAX_DIMENSION).contains(&height),
            "texture size {width}x{height} must be between 1 and {} in each dimension",
            Self::MAX_DIMENSION,
        );
        assert!(
            (width as usize)
                .checked_mul(height as usize)
                .is_some_and(|len| pixels.len() >= len),
            "{} pixels are too few for a {width}x{height} texture",
            pixels.len(),
        );

        Self {
            pixels,
            width,
            height,
            wrap_mode,
            filter_mode,
        }
    }

    /// Returns the texture's width in pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let pixels: Arc<[Colour]> = Arc::from([Colour::RED; 6]);
    /// let sampler = TextureSampler::new(pixels, 3, 2, WrapMode::Clamp, FilterMode::Nearest);
    ///
    /// assert_eq!(sampler.width(), 3);
    /// ```
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the texture's height in pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let pixels: Arc<[Colour]> = Arc::from([Colour::RED; 6]);
    /// let sampler = TextureSampler::new(pixels, 3, 2, WrapMode::Clamp, FilterMode::Nearest);
    ///
    /// assert_eq!(sampler.height(), 2);
    /// ```
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the colour at `uv`, applying the sampler's wrap mode and then its filter mode.
    ///
    /// # Examples
    ///
    /// A repeating texture tiles: coordinates a whole number apart give the same colour.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let pixels: Arc<[Colour]> = Arc::from([Colour::RED, Colour::GREEN, Colour::BLUE]);
    /// let sampler = TextureSampler::new(pixels, 3, 1, WrapMode::Repeat, FilterMode::Nearest);
    ///
    /// assert_eq!(sampler.sample(Vec2::new(0.5, 0.0)), Colour::GREEN);
    /// assert_eq!(sampler.sample(Vec2::new(2.5, 0.0)), Colour::GREEN);
    /// ```
    #[inline(always)]
    pub fn sample(&self, uv: Vec2) -> Colour {
        let uv = self.apply_wrap_mode(uv);

        self.apply_filter_mode(uv)
    }

    #[inline(always)]
    fn apply_wrap_mode(&self, uv: Vec2) -> Vec2 {
        match self.wrap_mode {
            WrapMode::Repeat => Vec2::new(uv.x.fract(), uv.y.fract()),
            WrapMode::Clamp => Vec2::new(uv.x.clamp(0.0, 1.0), uv.y.clamp(0.0, 1.0)),
        }
    }

    #[inline(always)]
    fn apply_filter_mode(&self, uv: Vec2) -> Colour {
        let x = uv.x * (self.width as f32 - 1.0);
        let y = (1.0 - uv.y) * (self.height as f32 - 1.0);

        match self.filter_mode {
            FilterMode::Nearest => {
                let x = x.round() as u32;
                let y = y.round() as u32;
                self.get_pixel(x, y)
            }
            FilterMode::Linear => {
                let x0 = x.floor() as u32;
                let x1 = x.ceil() as u32;
                let y0 = y.floor() as u32;
                let y1 = y.ceil() as u32;

                let c00 = self.get_pixel(x0, y0);
                let c10 = self.get_pixel(x1, y0);
                let c01 = self.get_pixel(x0, y1);
                let c11 = self.get_pixel(x1, y1);

                let tx = x - x0 as f32;
                let ty = y - y0 as f32;

                let c0 = Colour::lerp(c00, c10, tx);
                let c1 = Colour::lerp(c01, c11, tx);

                Colour::lerp(c0, c1, ty)
            }
        }
    }

    /// Returns the colour at `uv` with [`WrapMode::Clamp`] and [`FilterMode::Linear`], whatever
    /// the sampler's own modes.
    ///
    /// It gives the same result as [`sample`](TextureSampler::sample) on a clamping, linear
    /// sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let pixels: Arc<[Colour]> = Arc::from([Colour::BLACK, Colour::WHITE]);
    /// let sampler = TextureSampler::new(pixels, 2, 1, WrapMode::Clamp, FilterMode::Linear);
    ///
    /// for u in [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0] {
    ///     let uv = Vec2::new(u, 0.0);
    ///     assert_eq!(sampler.sample_linear_clamp(uv), sampler.sample(uv));
    /// }
    /// ```
    #[inline(always)]
    pub fn sample_linear_clamp(&self, uv: Vec2) -> Colour {
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);

        let x_f = u * (self.width - 1) as f32;
        let y_f = (1.0 - v) * (self.height - 1) as f32;

        let x0 = x_f as u32;
        let y0 = y_f as u32;

        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let tx = x_f - x0 as f32;
        let ty = y_f - y0 as f32;

        // SAFETY: `u` and `v` are clamped to [0, 1] (NaN converts to 0), and `new` limits the size
        // so that `width - 1` and `height - 1` are exact in `f32`. So `x0 <= width - 1` and
        // `y0 <= height - 1`, and `x1` and `y1` are clamped to the same limits.
        let [c00, c10, c01, c11] = unsafe {
            [
                self.get_pixel_unchecked(x0, y0),
                self.get_pixel_unchecked(x1, y0),
                self.get_pixel_unchecked(x0, y1),
                self.get_pixel_unchecked(x1, y1),
            ]
        };

        let c0 = Colour::lerp(c00, c10, tx);
        let c1 = Colour::lerp(c01, c11, tx);

        Colour::lerp(c0, c1, ty)
    }

    /// Returns the colour at `uv` with [`WrapMode::Clamp`] and [`FilterMode::Nearest`], whatever
    /// the sampler's own modes.
    ///
    /// It gives the same result as [`sample`](TextureSampler::sample) on a clamping, nearest
    /// sampler. A coordinate exactly between two texels takes the one further from 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let pixels: Arc<[Colour]> = Arc::from([Colour::BLACK, Colour::WHITE]);
    /// let sampler = TextureSampler::new(pixels, 2, 1, WrapMode::Repeat, FilterMode::Linear);
    ///
    /// assert_eq!(sampler.sample_nearest_clamp(Vec2::new(0.4, 0.0)), Colour::BLACK);
    /// assert_eq!(sampler.sample_nearest_clamp(Vec2::new(0.5, 0.0)), Colour::WHITE);
    /// assert_eq!(sampler.sample_nearest_clamp(Vec2::new(1.5, 0.0)), Colour::WHITE);
    /// ```
    #[inline(always)]
    pub fn sample_nearest_clamp(&self, uv: Vec2) -> Colour {
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);

        let x = (u * (self.width - 1) as f32).round() as u32;
        let y = ((1.0 - v) * (self.height - 1) as f32).round() as u32;

        self.get_pixel(x, y)
    }

    #[inline(always)]
    fn get_pixel(&self, x: u32, y: u32) -> Colour {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);

        self.pixels[(y * self.width + x) as usize]
    }

    /// Returns the colours at eight coordinates with [`WrapMode::Clamp`] and
    /// [`FilterMode::Linear`], whatever the sampler's own modes.
    ///
    /// `uv[0]` holds the eight u coordinates and `uv[1]` the eight v coordinates, one per lane;
    /// [`FragmentShaderSimd`] describes how lanes map to pixels. Each lane is sampled
    /// independently, and lane `i` of the result is
    /// [`sample_linear_clamp`](TextureSampler::sample_linear_clamp) of lane `i` of `uv`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use cpu_rasteriser::wide::f32x8;
    /// use engine::prelude::*;
    ///
    /// let pixels: Arc<[Colour]> = Arc::from([Colour::BLACK, Colour::WHITE]);
    /// let sampler = TextureSampler::new(pixels, 2, 1, WrapMode::Clamp, FilterMode::Linear);
    ///
    /// let us = [-0.5, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5];
    /// let colours = sampler.sample_linear_clamp_simd([f32x8::new(us), f32x8::splat(0.0)]);
    ///
    /// let reds = colours.r.to_array();
    /// for (u, red) in us.into_iter().zip(reds) {
    ///     assert!((red - sampler.sample_linear_clamp(Vec2::new(u, 0.0)).r).abs() < 1e-6);
    /// }
    /// ```
    #[inline(always)]
    pub fn sample_linear_clamp_simd(&self, uv: [f32x8; 2]) -> ColourSimd {
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let u = uv[0].fast_max(zero).fast_min(one);
        let v = uv[1].fast_max(zero).fast_min(one);

        let x_f = u * f32x8::splat((self.width - 1) as f32);

        let y_f = (one - v) * f32x8::splat((self.height - 1) as f32);

        let x = x_f.to_array();
        let y = y_f.to_array();

        let mut x0 = [0u32; 8];
        let mut x1 = [0u32; 8];
        let mut y0 = [0u32; 8];
        let mut y1 = [0u32; 8];

        for lane in 0..8 {
            x0[lane] = x[lane] as u32;
            y0[lane] = y[lane] as u32;

            x1[lane] = (x0[lane] + 1).min(self.width - 1);

            y1[lane] = (y0[lane] + 1).min(self.height - 1);
        }

        let tx = x_f - f32x8::new(x0.map(|x| x as f32));

        let ty = y_f - f32x8::new(y0.map(|y| y as f32));

        // SAFETY: the coordinates are in range for the same reasons as in `sample_linear_clamp`.
        // `fast_max` and `fast_min` may pass a NaN through, but NaN still converts to 0.
        let [c00, c10, c01, c11] = unsafe {
            [
                self.gather8(x0, y0),
                self.gather8(x1, y0),
                self.gather8(x0, y1),
                self.gather8(x1, y1),
            ]
        };

        let c0 = c00.lerp(c10, tx);
        let c1 = c01.lerp(c11, tx);

        c0.lerp(c1, ty)
    }

    /// Returns the colours at eight coordinates with [`WrapMode::Clamp`] and
    /// [`FilterMode::Nearest`], whatever the sampler's own modes.
    ///
    /// `uv[0]` holds the eight u coordinates and `uv[1]` the eight v coordinates, one per lane;
    /// [`FragmentShaderSimd`] describes how lanes map to pixels. Each lane is sampled
    /// independently, and lane `i` of the result is
    /// [`sample_nearest_clamp`](TextureSampler::sample_nearest_clamp) of lane `i` of `uv`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use cpu_rasteriser::wide::f32x8;
    /// use engine::prelude::*;
    ///
    /// let pixels: Arc<[Colour]> = Arc::from([Colour::BLACK, Colour::WHITE]);
    /// let sampler = TextureSampler::new(pixels, 2, 1, WrapMode::Clamp, FilterMode::Linear);
    ///
    /// // Each lane takes the nearer texel, so the colours are only black or white.
    /// let us = [-0.5, 0.0, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5];
    /// let colours = sampler.sample_nearest_clamp_simd([f32x8::new(us), f32x8::splat(0.0)]);
    ///
    /// let reds = colours.r.to_array();
    /// assert_eq!(reds, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// ```
    #[inline(always)]
    pub fn sample_nearest_clamp_simd(&self, uv: [f32x8; 2]) -> ColourSimd {
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let u = uv[0].fast_max(zero).fast_min(one);
        let v = uv[1].fast_max(zero).fast_min(one);

        let x = (u * f32x8::splat((self.width - 1) as f32)).to_array();

        let y = ((one - v) * f32x8::splat((self.height - 1) as f32)).to_array();

        let mut xs = [0u32; 8];
        let mut ys = [0u32; 8];

        for lane in 0..8 {
            xs[lane] = x[lane].round() as u32;
            ys[lane] = y[lane].round() as u32;
        }

        // SAFETY: as in `sample_linear_clamp`, each coordinate is at most `width - 1` or
        // `height - 1`, both exact integers in `f32`, so rounding can't exceed them.
        unsafe { self.gather8(xs, ys) }
    }

    /// # Safety
    ///
    /// Every element of `xs` is below `self.width` and every element of `ys` below `self.height`.
    #[inline(always)]
    unsafe fn gather8(&self, xs: [u32; 8], ys: [u32; 8]) -> ColourSimd {
        // SAFETY: the caller guarantees every coordinate is inside the texture.
        let [p0, p1, p2, p3, p4, p5, p6, p7] = unsafe {
            [
                self.get_pixel_unchecked(xs[0], ys[0]),
                self.get_pixel_unchecked(xs[1], ys[1]),
                self.get_pixel_unchecked(xs[2], ys[2]),
                self.get_pixel_unchecked(xs[3], ys[3]),
                self.get_pixel_unchecked(xs[4], ys[4]),
                self.get_pixel_unchecked(xs[5], ys[5]),
                self.get_pixel_unchecked(xs[6], ys[6]),
                self.get_pixel_unchecked(xs[7], ys[7]),
            ]
        };

        ColourSimd {
            r: f32x8::new([p0.r, p1.r, p2.r, p3.r, p4.r, p5.r, p6.r, p7.r]),
            g: f32x8::new([p0.g, p1.g, p2.g, p3.g, p4.g, p5.g, p6.g, p7.g]),
            b: f32x8::new([p0.b, p1.b, p2.b, p3.b, p4.b, p5.b, p6.b, p7.b]),
            a: f32x8::new([p0.a, p1.a, p2.a, p3.a, p4.a, p5.a, p6.a, p7.a]),
        }
    }

    /// # Safety
    ///
    /// `x < self.width` and `y < self.height`.
    #[inline(always)]
    unsafe fn get_pixel_unchecked(&self, x: u32, y: u32) -> Colour {
        // SAFETY: the caller guarantees `x` and `y` are inside the texture, so the index is below
        // `width * height`, and `new` ensures `pixels` holds at least that many pixels. If the
        // index overflows `u32` in a release build, `width * height` exceeds `u32::MAX`, so the
        // wrapped index is still below `pixels.len()`.
        unsafe { *self.pixels.get_unchecked((y * self.width + x) as usize) }
    }
}

/// An error loading a [`Texture`].
///
/// Its [`Display`](std::fmt::Display) form names the image file; the cause is its
/// [`source`](std::error::Error::source).
///
/// # Examples
///
/// ```
/// use engine::prelude::*;
///
/// let error = Texture::from_image("missing.png").unwrap_err();
///
/// assert_eq!(error.to_string(), "failed to load the image `missing.png`");
/// assert!(matches!(error, TextureError::Image { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TextureError {
    /// The image file couldn't be opened or decoded.
    #[error("failed to load the image `{}`", path.display())]
    Image {
        /// The image file's path.
        path: std::path::PathBuf,
        /// Why it couldn't be opened or decoded.
        #[source]
        source: image::ImageError,
    },
}

/// How a [`TextureSampler`] treats texture coordinates outside [0, 1].
///
/// # Examples
///
/// See [`TextureSampler::sample`].
///
/// # Known issues
///
/// - [`WrapMode::Repeat`] doesn't wrap negative coordinates: they clamp to the texture's edge
///   instead ([#24](https://github.com/costott/cpu_rasteriser/issues/24)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    /// Tiles the texture by wrapping each coordinate into [0, 1), so 1 reads the same texels as 0.
    Repeat,
    /// Clamps each coordinate to [0, 1], so the edge texels extend outwards.
    Clamp,
}

/// How a [`TextureSampler`] turns a texture coordinate into a colour.
///
/// # Examples
///
/// See [`TextureSampler`] and [`TextureSampler::sample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Takes the texel nearest the coordinate, rounding halfway cases away from 0.
    Nearest,
    /// Interpolates bilinearly between the four texels around the coordinate.
    Linear,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(extent: Extent, colour: Colour) -> RenderTarget {
        let mut target = RenderTarget::new(extent);
        let mut renderer = Renderer::new().unwrap();
        renderer
            .begin_render_pass(
                &mut target,
                RenderPassDescriptor {
                    viewport: Viewport::full(&extent),
                    colour_load_op: LoadOp::Clear(colour),
                    depth_load_op: None,
                },
            )
            .finish();
        target
    }

    #[test]
    fn update_from_never_changes_a_live_sampler_and_reuses_unshared_storage() {
        let extent = Extent::new(10, 4);
        let mut texture = Texture::from_render_target(&filled(extent, Colour::RED));
        let first = texture.sampler(WrapMode::Clamp, FilterMode::Linear);

        texture.update_from(&filled(extent, Colour::GREEN));
        assert_ne!(texture.pixels.as_ptr(), first.pixels.as_ptr());
        assert!(first.pixels.iter().all(|&c| c == Colour::RED));
        assert!(texture.pixels.iter().all(|&c| c == Colour::GREEN));

        let unshared = texture.pixels.as_ptr();
        texture.update_from(&filled(extent, Colour::BLUE));
        assert_eq!(texture.pixels.as_ptr(), unshared);
        assert!(texture.pixels.iter().all(|&c| c == Colour::BLUE));
        assert!(first.pixels.iter().all(|&c| c == Colour::RED));

        texture.update_from(&filled(Extent::new(6, 3), Colour::BLUE));
        assert_eq!(
            (texture.width, texture.height, texture.pixels.len()),
            (6, 3, 18)
        );
    }

    #[test]
    #[should_panic(expected = "too few")]
    fn sampler_rejects_fewer_pixels_than_its_size() {
        TextureSampler::new(
            Arc::from([Colour::BLACK; 3]),
            2,
            2,
            WrapMode::Clamp,
            FilterMode::Linear,
        );
    }

    #[test]
    #[should_panic(expected = "must be between")]
    fn sampler_rejects_an_empty_size() {
        Texture {
            width: 0,
            height: 0,
            pixels: Arc::from([]),
        }
        .sampler(WrapMode::Clamp, FilterMode::Linear);
    }

    #[test]
    #[should_panic(expected = "must be between")]
    fn sampler_rejects_a_size_beyond_exact_f32_texel_coordinates() {
        TextureSampler::new(
            Arc::from([]),
            (1 << 24) + 1,
            1,
            WrapMode::Clamp,
            FilterMode::Linear,
        );
    }
}
