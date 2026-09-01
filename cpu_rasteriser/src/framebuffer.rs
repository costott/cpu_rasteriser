use wide::f32x8;

use crate::prelude::*;

pub struct FrameBuffer {
    extent: Extent,
    stride: usize,       // physical row stride, including SIMD padding
    pixels: Vec<Colour>, // TODO: change to SoA layout for better SIMD performance - separate Vec<f32> for each channel, as rn profiling shows performance is dominated by branch misprediction in scalar loop of rasteriser callback, so need a set8_unchecked_with_mask
}
impl FrameBuffer {
    const SIMD_WIDTH: usize = 8;

    pub fn new(width: usize, height: usize) -> Self {
        let stride = (width + Self::SIMD_WIDTH - 1).next_multiple_of(Self::SIMD_WIDTH);

        Self {
            extent: Extent::new(width, height),
            stride,
            pixels: vec![Colour::BLACK; stride * height],
        }
    }

    #[inline(always)]
    pub fn width(&self) -> usize {
        self.extent.width
    }

    #[inline(always)]
    pub fn height(&self) -> usize {
        self.extent.height
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn pixels(&self) -> &[Colour] {
        &self.pixels
    }

    pub fn pixels_compact(&self) -> Vec<Colour> {
        let mut pixels = Vec::with_capacity(self.width() * self.height());

        for y in 0..self.height() {
            let start = y * self.stride;
            let end = start + self.width();

            pixels.extend_from_slice(&self.pixels[start..end]);
        }

        pixels
    }

    pub fn pixels_u32(&self) -> Vec<u32> {
        // Returns only the logical pixels of the framebuffer, not the padded pixels at the end of each row
        let mut pixels = Vec::with_capacity(self.width() * self.height());

        for y in 0..self.height() {
            let start = y * self.stride;
            let end = start + self.width();

            pixels.extend(self.pixels[start..end].iter().copied().map(Colour::to_u32));
        }

        pixels
    }

    pub fn clear(&mut self, colour: Colour) {
        self.pixels.fill(colour);
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.extent = Extent::new(width, height);
        self.stride = (width + Self::SIMD_WIDTH - 1).next_multiple_of(Self::SIMD_WIDTH);
        self.pixels.resize(self.stride * height, Colour::BLACK);
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.extent.width);
        debug_assert!(y < self.extent.height);

        y * self.stride + x
    }

    #[inline(always)]
    pub fn index_unchecked(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }

    pub fn set_pixel(&mut self, p: Vec2, colour: Colour) {
        let x = p.x as usize;
        let y = p.y as usize;

        if x >= self.width() || y >= self.height() {
            return;
        }

        let index = self.index(x, y);
        self.pixels[index] = colour;
    }

    #[inline(always)]
    pub unsafe fn set_pixel_unchecked(&mut self, p: Vec2, colour: Colour) {
        let index = self.index(p.x as usize, p.y as usize);
        unsafe {
            *self.pixels.get_unchecked_mut(index) = colour;
        }
    }

    pub unsafe fn set_pixel_index_unchecked(&mut self, index: usize, colour: Colour) {
        debug_assert!(index < self.pixels.len());

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

    #[inline(always)]
    pub fn get_pixel(&self, p: Vec2) -> Option<Colour> {
        let x = p.x as usize;
        let y = p.y as usize;

        if x >= self.width() || y >= self.height() {
            return None;
        }

        let index = self.index(x, y);
        Some(self.pixels[index])
    }

    #[inline(always)]
    pub unsafe fn get_pixel_unchecked(&self, p: Vec2) -> Colour {
        let index = self.index(p.x as usize, p.y as usize);

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
            unsafe { self.pixels.get_unchecked(index).g },
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
            unsafe { self.pixels.get_unchecked(index).b },
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

    #[inline(always)]
    pub unsafe fn get8_unchecked(&self, index: usize) -> ColourSimd {
        debug_assert!(index + 8 <= self.pixels.len());

        // TODO: look into different storage layout of framebuffer to make this more efficient, e.g. SoA instead of AoS
        let p0 = unsafe { self.pixels.get_unchecked(index) };
        let p1 = unsafe { self.pixels.get_unchecked(index + 1) };
        let p2 = unsafe { self.pixels.get_unchecked(index + 2) };
        let p3 = unsafe { self.pixels.get_unchecked(index + 3) };
        let p4 = unsafe { self.pixels.get_unchecked(index + 4) };
        let p5 = unsafe { self.pixels.get_unchecked(index + 5) };
        let p6 = unsafe { self.pixels.get_unchecked(index + 6) };
        let p7 = unsafe { self.pixels.get_unchecked(index + 7) };

        ColourSimd {
            r: f32x8::new([p0.r, p1.r, p2.r, p3.r, p4.r, p5.r, p6.r, p7.r]),
            g: f32x8::new([p0.g, p1.g, p2.g, p3.g, p4.g, p5.g, p6.g, p7.g]),
            b: f32x8::new([p0.b, p1.b, p2.b, p3.b, p4.b, p5.b, p6.b, p7.b]),
            a: f32x8::new([p0.a, p1.a, p2.a, p3.a, p4.a, p5.a, p6.a, p7.a]),
        }
    }
}
