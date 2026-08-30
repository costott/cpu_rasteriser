use crate::prelude::*;

pub struct FrameBuffer {
    extent: Extent,
    pixels: Vec<Colour>,
}
impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            extent: Extent::new(width, height),
            pixels: vec![Colour::BLACK; width * height],
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

    pub fn pixels(&self) -> &[Colour] {
        &self.pixels
    }

    pub fn pixels_u32(&self) -> Vec<u32> {
        self.pixels.iter().copied().map(Colour::to_u32).collect()
    }

    pub fn clear(&mut self, colour: Colour) {
        self.pixels.fill(colour);
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.extent = Extent::new(width, height);
        self.pixels.resize(width * height, Colour::BLACK);
    }

    pub fn set_pixel(&mut self, p: Vec2, colour: Colour) {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width() as i32 || y < 0 || y >= self.height() as i32 {
            return;
        }
        let index = (y as usize) * self.width() + (x as usize);
        self.pixels[index] = colour;
    }

    pub unsafe fn set_pixel_unchecked(&mut self, p: Vec2, colour: Colour) {
        let index = (p.y as usize) * self.width() + (p.x as usize);
        unsafe {
            *self.pixels.get_unchecked_mut(index) = colour;
        }
    }

    pub fn get_pixel(&self, p: Vec2) -> Option<Colour> {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width() as i32 || y < 0 || y >= self.height() as i32 {
            return None;
        }
        let index = (y as usize) * self.width() + (x as usize);
        Some(self.pixels[index])
    }

    pub unsafe fn get_pixel_unchecked(&self, p: Vec2) -> Colour {
        let index = (p.y as usize) * self.width() + (p.x as usize);

        unsafe { *self.pixels.get_unchecked(index) }
    }
}
