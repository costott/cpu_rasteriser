use cpu_rasteriser::prelude::*;

use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<[u32]>,
}
impl Texture {
    pub fn from_image(path: impl AsRef<std::path::Path>) -> Result<Self, TextureError> {
        let image = image::open(path).map_err(TextureError::Image)?.to_rgba8();

        let pixels = image
            .pixels()
            .map(|pixel| u32::from_be_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]))
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
    pixels: Arc<[u32]>,
    width: u32,
    height: u32,
    wrap_mode: WrapMode,
    filter_mode: FilterMode,
}
impl TextureSampler {
    pub fn new(
        pixels: Arc<[u32]>,
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

        Colour::from_u32(self.pixels[(y * self.width + x) as usize])
    }

    #[inline(always)]
    pub fn sample_linear_clamp(&self, uv: Vec2) -> Colour {
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);

        let x_f = u * (self.width - 1) as f32;
        let y_f = (1.0 - v) * (self.height - 1) as f32;

        let x0 = x_f as u32;
        let y0 = y_f as u32;

        let tx = ((x_f - x0 as f32) * 256.0) as u32;
        let ty = ((y_f - y0 as f32) * 256.0) as u32;

        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let stride = self.width as usize;
        let c00 = self.pixels[(y0 as usize * stride) + x0 as usize];
        let c10 = self.pixels[(y0 as usize * stride) + x1 as usize];
        let c01 = self.pixels[(y1 as usize * stride) + x0 as usize];
        let c11 = self.pixels[(y1 as usize * stride) + x1 as usize];

        let inv_tx = 256 - tx;
        let inv_ty = 256 - ty;

        let w00 = inv_tx * inv_ty;
        let w10 = tx * inv_ty;
        let w01 = inv_tx * ty;
        let w11 = tx * ty;

        #[inline(always)]
        fn interpolate(
            c00: u32,
            c10: u32,
            c01: u32,
            c11: u32,
            shift: u32,
            w00: u32,
            w10: u32,
            w01: u32,
            w11: u32,
        ) -> u8 {
            let c00 = (c00 >> shift) & 0xff;
            let c10 = (c10 >> shift) & 0xff;
            let c01 = (c01 >> shift) & 0xff;
            let c11 = (c11 >> shift) & 0xff;
            ((c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11) >> 16) as u8
        }

        Colour::new(
            interpolate(c00, c10, c01, c11, 16, w00, w10, w01, w11),
            interpolate(c00, c10, c01, c11, 8, w00, w10, w01, w11),
            interpolate(c00, c10, c01, c11, 0, w00, w10, w01, w11),
            interpolate(c00, c10, c01, c11, 24, w00, w10, w01, w11),
        )
    }

    #[inline(always)]
    pub fn sample_nearest_clamp(&self, uv: Vec2) -> Colour {
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);

        let x = (u * (self.width - 1) as f32).round() as u32;
        let y = ((1.0 - v) * (self.height - 1) as f32).round() as u32;

        Colour::from_u32(self.pixels[(y * self.width + x) as usize])
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
