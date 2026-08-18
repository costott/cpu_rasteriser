use crate::prelude::*;

pub struct FrameBuffer {
    extent: Extent,
    pixels: Vec<u32>,
}
impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            extent: Extent::new(width, height),
            pixels: vec![0; width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.extent.width
    }

    pub fn height(&self) -> usize {
        self.extent.height
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn clear(&mut self, colour: Colour) {
        self.pixels.fill(colour.to_u32());
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.extent = Extent::new(width, height);
        self.pixels.resize(width * height, 0);
    }

    pub fn set_pixel(&mut self, p: Vec2, colour: Colour) {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width() as i32 || y < 0 || y >= self.height() as i32 {
            return;
        }
        let index = (y as usize) * self.width() + (x as usize);
        self.pixels[index] = colour.to_u32();
    }

    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    pub fn get_pixel(&self, p: Vec2) -> Option<Colour> {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width() as i32 || y < 0 || y >= self.height() as i32 {
            return None;
        }
        let index = (y as usize) * self.width() + (x as usize);
        Some(Colour::from_u32(self.pixels[index]))
    }
}
