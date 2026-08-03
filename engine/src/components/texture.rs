use cpu_rasteriser::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Colour>,

    pub wrap_mode: WrapMode,
    pub filter_mode: FilterMode,
}
impl Texture {
    pub fn from_image(path: impl AsRef<std::path::Path>) -> Result<Self, TextureError> {
        let image = image::open(path).map_err(TextureError::Image)?.to_rgba8();

        let pixels = image
            .pixels()
            .map(|pixel| Colour {
                r: pixel[0],
                g: pixel[1],
                b: pixel[2],
                a: pixel[3],
            })
            .collect();

        Ok(Self {
            width: image.width(),
            height: image.height(),
            pixels,
            wrap_mode: WrapMode::Repeat,
            filter_mode: FilterMode::Linear,
        })
    }

    pub fn sample(&self, uv: Vec2) -> Colour {
        let uv = self.apply_wrap_mode(uv);

        self.apply_filter_mode(uv)
    }

    fn apply_wrap_mode(&self, uv: Vec2) -> Vec2 {
        match self.wrap_mode {
            WrapMode::Repeat => Vec2::new(uv.x.fract(), uv.y.fract()),
            WrapMode::Clamp => Vec2::new(uv.x.clamp(0.0, 1.0), uv.y.clamp(0.0, 1.0)),
        }
    }

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

                let tx = x - x.floor();
                let ty = y - y.floor();

                let c0 = Colour::lerp(&c00, &c10, tx);
                let c1 = Colour::lerp(&c01, &c11, tx);

                Colour::lerp(&c0, &c1, ty)
            }
        }
    }

    fn get_pixel(&self, x: u32, y: u32) -> Colour {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        self.pixels[(y * self.width + x) as usize]
    }
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
