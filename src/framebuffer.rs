use crate::colour::Colour;
use crate::maths::Vec2;

pub struct FrameBuffer {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
}
impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; width * height],
        }
    }

    pub fn clear(&mut self, colour: Colour) {
        self.pixels.fill(colour.to_u32());
    }

    pub fn set_pixel(&mut self, p: Vec2, colour: Colour) {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return;
        }
        let index = (y as usize) * self.width + (x as usize);
        self.pixels[index] = colour.to_u32();
    }

    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }
}
