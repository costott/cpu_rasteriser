use cpu_rasteriser::{prelude::*, wide::f32x8};

use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<[Colour]>,
}
impl Texture {
    pub fn from_image(path: impl AsRef<std::path::Path>) -> Result<Self, TextureError> {
        let image = image::open(path).map_err(TextureError::Image)?.to_rgba8();

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

#[derive(Clone, Debug, PartialEq)]
pub struct TextureSampler {
    pixels: Arc<[Colour]>,
    width: u32,
    height: u32,
    wrap_mode: WrapMode,
    filter_mode: FilterMode,
}
impl TextureSampler {
    pub fn new(
        pixels: Arc<[Colour]>,
        width: u32,
        height: u32,
        wrap_mode: WrapMode,
        filter_mode: FilterMode,
    ) -> Self {
        Self {
            pixels,
            width,
            height,
            wrap_mode,
            filter_mode,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

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

                let c0 = Colour::lerp(&c00, &c10, tx);
                let c1 = Colour::lerp(&c01, &c11, tx);

                Colour::lerp(&c0, &c1, ty)
            }
        }
    }

    #[inline(always)]
    fn get_pixel(&self, x: u32, y: u32) -> Colour {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);

        self.pixels[(y * self.width + x) as usize]
    }

    #[inline(always)]
    unsafe fn get_pixel_unchecked(&self, x: u32, y: u32) -> Colour {
        unsafe { *self.pixels.get_unchecked((y * self.width + x) as usize) }
    }

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

        let c00 = unsafe { self.get_pixel_unchecked(x0, y0) };
        let c10 = unsafe { self.get_pixel_unchecked(x1, y0) };
        let c01 = unsafe { self.get_pixel_unchecked(x0, y1) };
        let c11 = unsafe { self.get_pixel_unchecked(x1, y1) };

        let c0 = Colour::lerp(&c00, &c10, tx);
        let c1 = Colour::lerp(&c01, &c11, tx);

        Colour::lerp(&c0, &c1, ty)
    }

    #[inline(always)]
    pub fn sample_nearest_clamp(&self, uv: Vec2) -> Colour {
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);

        let x = (u * (self.width - 1) as f32).round() as u32;
        let y = ((1.0 - v) * (self.height - 1) as f32).round() as u32;

        self.get_pixel(x, y)
    }

    /// Samples eight horizontally adjacent fragments using bilinear filtering.
    ///
    /// The eight lanes are expected to correspond to consecutive screen-space
    /// fragments: x, x + 1, ..., x + 7.
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

        let c00 = self.gather8(x0, y0);
        let c10 = self.gather8(x1, y0);
        let c01 = self.gather8(x0, y1);
        let c11 = self.gather8(x1, y1);

        let c0 = c00.lerp(c10, tx);
        let c1 = c01.lerp(c11, tx);

        c0.lerp(c1, ty)
    }

    #[inline(always)]
    fn gather8(&self, xs: [u32; 8], ys: [u32; 8]) -> ColourSimd {
        let p0 = unsafe { self.get_pixel_unchecked(xs[0], ys[0]) };
        let p1 = unsafe { self.get_pixel_unchecked(xs[1], ys[1]) };
        let p2 = unsafe { self.get_pixel_unchecked(xs[2], ys[2]) };
        let p3 = unsafe { self.get_pixel_unchecked(xs[3], ys[3]) };
        let p4 = unsafe { self.get_pixel_unchecked(xs[4], ys[4]) };
        let p5 = unsafe { self.get_pixel_unchecked(xs[5], ys[5]) };
        let p6 = unsafe { self.get_pixel_unchecked(xs[6], ys[6]) };
        let p7 = unsafe { self.get_pixel_unchecked(xs[7], ys[7]) };

        ColourSimd {
            r: f32x8::new([p0.r, p1.r, p2.r, p3.r, p4.r, p5.r, p6.r, p7.r]),
            g: f32x8::new([p0.g, p1.g, p2.g, p3.g, p4.g, p5.g, p6.g, p7.g]),
            b: f32x8::new([p0.b, p1.b, p2.b, p3.b, p4.b, p5.b, p6.b, p7.b]),
            a: f32x8::new([p0.a, p1.a, p2.a, p3.a, p4.a, p5.a, p6.a, p7.a]),
        }
    }

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

        self.gather8(xs, ys)
    }
}

pub fn render_target_sampler(render_target: &RenderTarget) -> TextureSampler {
    TextureSampler::new(
        render_target.pixels().into(),
        render_target.width() as u32,
        render_target.height() as u32,
        WrapMode::Clamp,
        FilterMode::Linear,
    )
}

#[derive(Debug)]
pub enum TextureError {
    Image(image::ImageError),
}
impl std::fmt::Display for TextureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureError::Image(e) => write!(f, "Image error: {}", e),
        }
    }
}
impl std::error::Error for TextureError {}

#[derive(Debug, Clone, PartialEq)]
pub enum WrapMode {
    Repeat,
    Clamp,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterMode {
    Nearest,
    Linear,
}
