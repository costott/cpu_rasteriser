use wide::f32x8;

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

    pub unsafe fn set_pixel_index_unchecked(&mut self, index: usize, colour: Colour) {
        unsafe {
            *self.pixels.get_unchecked_mut(index) = colour;
        }
    }

    #[inline(always)]
    pub unsafe fn set8_r_unchecked(&mut self, index: usize, reds: f32x8) {
        debug_assert!(index + 8 <= self.pixels.len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = reds.to_array();
        unsafe {
            *self.pixels.get_unchecked_mut(index) = Colour::new(values[0], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 1) = Colour::new(values[1], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 2) = Colour::new(values[2], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 3) = Colour::new(values[3], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 4) = Colour::new(values[4], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 5) = Colour::new(values[5], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 6) = Colour::new(values[6], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 7) = Colour::new(values[7], 0.0, 0.0, 1.0);
        }
    }

    #[inline(always)]
    pub unsafe fn set8_r_unchecked_with_mask(&mut self, index: usize, reds: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.pixels.len());

        let current = unsafe { self.get8_r_unchecked(index) };
        let new_pixels = mask.select(reds, current);

        unsafe { self.set8_r_unchecked(index, new_pixels) };
    }
    #[inline(always)]
    pub unsafe fn set8_g_unchecked(&mut self, index: usize, greens: f32x8) {
        debug_assert!(index + 8 <= self.pixels.len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = greens.to_array();
        unsafe {
            *self.pixels.get_unchecked_mut(index) = Colour::new(values[0], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 1) = Colour::new(values[1], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 2) = Colour::new(values[2], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 3) = Colour::new(values[3], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 4) = Colour::new(values[4], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 5) = Colour::new(values[5], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 6) = Colour::new(values[6], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 7) = Colour::new(values[7], 0.0, 0.0, 1.0);
        }
    }

    #[inline(always)]
    pub unsafe fn set8_g_unchecked_with_mask(&mut self, index: usize, greens: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.pixels.len());

        let current = unsafe { self.get8_g_unchecked(index) };
        let new_pixels = mask.select(greens, current);

        unsafe { self.set8_g_unchecked(index, new_pixels) };
    }
    #[inline(always)]
    pub unsafe fn set8_b_unchecked(&mut self, index: usize, blues: f32x8) {
        debug_assert!(index + 8 <= self.pixels.len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = blues.to_array();
        unsafe {
            *self.pixels.get_unchecked_mut(index) = Colour::new(values[0], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 1) = Colour::new(values[1], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 2) = Colour::new(values[2], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 3) = Colour::new(values[3], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 4) = Colour::new(values[4], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 5) = Colour::new(values[5], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 6) = Colour::new(values[6], 0.0, 0.0, 1.0);
            *self.pixels.get_unchecked_mut(index + 7) = Colour::new(values[7], 0.0, 0.0, 1.0);
        }
    }

    #[inline(always)]
    pub unsafe fn set8_b_unchecked_with_mask(&mut self, index: usize, blues: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.pixels.len());

        let current = unsafe { self.get8_b_unchecked(index) };
        let new_pixels = mask.select(blues, current);

        unsafe { self.set8_b_unchecked(index, new_pixels) };
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

    #[inline(always)]
    pub unsafe fn get_pixel_index_unchecked(&self, index: usize) -> Colour {
        unsafe { *self.pixels.get_unchecked(index) }
    }

    #[inline(always)]
    pub unsafe fn get8_r_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.pixels.len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let values = [
            unsafe { self.pixels.get_unchecked(index).r },
            unsafe { self.pixels.get_unchecked(index + 1).r },
            unsafe { self.pixels.get_unchecked(index + 2).r },
            unsafe { self.pixels.get_unchecked(index + 3).r },
            unsafe { self.pixels.get_unchecked(index + 4).r },
            unsafe { self.pixels.get_unchecked(index + 5).r },
            unsafe { self.pixels.get_unchecked(index + 6).r },
            unsafe { self.pixels.get_unchecked(index + 7).r },
        ];
        f32x8::new(values)
    }
    #[inline(always)]
    pub unsafe fn get8_g_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.pixels.len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let values = [
            unsafe { self.pixels.get_unchecked(index).r },
            unsafe { self.pixels.get_unchecked(index + 1).g },
            unsafe { self.pixels.get_unchecked(index + 2).g },
            unsafe { self.pixels.get_unchecked(index + 3).g },
            unsafe { self.pixels.get_unchecked(index + 4).g },
            unsafe { self.pixels.get_unchecked(index + 5).g },
            unsafe { self.pixels.get_unchecked(index + 6).g },
            unsafe { self.pixels.get_unchecked(index + 7).g },
        ];
        f32x8::new(values)
    }
    #[inline(always)]
    pub unsafe fn get8_b_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.pixels.len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let values = [
            unsafe { self.pixels.get_unchecked(index).r },
            unsafe { self.pixels.get_unchecked(index + 1).b },
            unsafe { self.pixels.get_unchecked(index + 2).b },
            unsafe { self.pixels.get_unchecked(index + 3).b },
            unsafe { self.pixels.get_unchecked(index + 4).b },
            unsafe { self.pixels.get_unchecked(index + 5).b },
            unsafe { self.pixels.get_unchecked(index + 6).b },
            unsafe { self.pixels.get_unchecked(index + 7).b },
        ];
        f32x8::new(values)
    }
}
