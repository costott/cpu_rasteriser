use wide::{f32x8, i32x8};

use crate::grid::Grid;
use crate::prelude::*;

pub struct FrameBuffer {
    pixels: Grid<Colour>, // TODO: change to SoA layout for better SIMD performance - separate Vec<f32> for each channel, as rn profiling shows performance is dominated by branch misprediction in scalar loop of rasteriser callback, so need a set8_unchecked_with_mask
}
impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: Grid::new(width, height, Colour::BLACK),
        }
    }

    #[inline(always)]
    pub fn width(&self) -> usize {
        self.pixels.width()
    }

    #[inline(always)]
    pub fn height(&self) -> usize {
        self.pixels.height()
    }

    pub fn extent(&self) -> Extent {
        Extent::new(self.width(), self.height())
    }

    pub fn pixels_compact(&self) -> Vec<Colour> {
        let mut pixels = Vec::with_capacity(self.width() * self.height());

        for y in 0..self.height() {
            pixels.extend_from_slice(self.pixels.row(y));
        }

        pixels
    }

    pub(crate) fn write_pixels(&self, out: &mut [Colour]) {
        let width = self.width();
        assert!(out.len() >= width * self.height());

        for y in 0..self.height() {
            out[y * width..(y + 1) * width].copy_from_slice(self.pixels.row(y));
        }
    }

    pub fn pixels_u32(&self) -> Vec<u32> {
        let mut pixels = vec![0; self.width() * self.height()];
        self.write_pixels_u32(&mut pixels);
        pixels
    }

    pub(crate) fn write_pixels_u32(&self, out: &mut [u32]) {
        let width = self.width();
        assert!(out.len() >= width * self.height());

        for y in 0..self.height() {
            let src = self.pixels.row(y);
            let dst = &mut out[y * width..(y + 1) * width];

            let mut src_chunks = src.chunks_exact(8);
            let mut dst_chunks = dst.chunks_exact_mut(8);

            for (s, d) in (&mut src_chunks).zip(&mut dst_chunks) {
                let r = f32x8::new([
                    s[0].r, s[1].r, s[2].r, s[3].r, s[4].r, s[5].r, s[6].r, s[7].r,
                ]);
                let g = f32x8::new([
                    s[0].g, s[1].g, s[2].g, s[3].g, s[4].g, s[5].g, s[6].g, s[7].g,
                ]);
                let b = f32x8::new([
                    s[0].b, s[1].b, s[2].b, s[3].b, s[4].b, s[5].b, s[6].b, s[7].b,
                ]);
                let a = f32x8::new([
                    s[0].a, s[1].a, s[2].a, s[3].a, s[4].a, s[5].a, s[6].a, s[7].a,
                ]);

                // max (not fast_max) picks the non-NaN operand on every target, so NaN becomes 0 as in to_u32
                let quantise = |v: f32x8| {
                    (v.max(f32x8::splat(0.0)).fast_min(f32x8::splat(1.0)) * f32x8::splat(255.0))
                        .trunc_int()
                };

                let packed = (quantise(a) * i32x8::splat(1 << 24))
                    | (quantise(r) * i32x8::splat(1 << 16))
                    | (quantise(g) * i32x8::splat(1 << 8))
                    | quantise(b);

                for (d, p) in d.iter_mut().zip(packed.to_array()) {
                    *d = p as u32;
                }
            }

            for (s, d) in src_chunks
                .remainder()
                .iter()
                .zip(dst_chunks.into_remainder())
            {
                *d = s.to_u32();
            }
        }
    }

    pub fn clear(&mut self, colour: Colour) {
        self.pixels.fill(colour);
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.pixels.resize(width, height, Colour::BLACK);
    }

    pub(crate) fn blit_from(
        &mut self,
        dst_origin: (usize, usize),
        src: &FrameBuffer,
        src_origin: (usize, usize),
        size: Extent,
    ) {
        self.pixels
            .blit_from(dst_origin, &src.pixels, src_origin, size);
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        self.pixels.index(x, y)
    }

    #[inline(always)]
    pub fn index_unchecked(&self, x: usize, y: usize) -> usize {
        self.pixels.index_unchecked(x, y)
    }

    pub fn set_pixel(&mut self, p: Vec2, colour: Colour) {
        let x = p.x as usize;
        let y = p.y as usize;

        if x >= self.width() || y >= self.height() {
            return;
        }

        let index = self.index(x, y);
        self.pixels.as_mut_slice()[index] = colour;
    }

    /// # Safety
    ///
    /// `p` must lie inside the framebuffer: `p.x as usize < self.width()` and
    /// `p.y as usize < self.height()`.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn set_pixel_unchecked(&mut self, p: Vec2, colour: Colour) {
        let index = self.index(p.x as usize, p.y as usize);
        // SAFETY: the caller guarantees `p` lies inside the framebuffer, so `index` is in bounds.
        unsafe {
            *self.pixels.as_mut_slice().get_unchecked_mut(index) = colour;
        }
    }

    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y) + i` for some `x < self.width()`,
    /// `y < self.height()` and `i < 8`. An index past the end of row `y` writes into the row's
    /// padding, which isn't part of the image.
    pub unsafe fn set_pixel_index_unchecked(&mut self, index: usize, colour: Colour) {
        debug_assert!(index < self.pixels.as_slice().len());

        // SAFETY: the caller guarantees `index` is one of the eight elements starting at an
        // in-bounds pixel, and `Grid` pads every row so that all eight are inside the allocation.
        unsafe {
            *self.pixels.as_mut_slice().get_unchecked_mut(index) = colour;
        }
    }

    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` write into the row's padding, which
    /// isn't part of the image.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn set8_r_unchecked(&mut self, index: usize, reds: f32x8) {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = reds.to_array();
        let pixels = self.pixels.as_mut_slice();
        // SAFETY: the caller guarantees `index` is an in-bounds pixel, and `Grid` pads every row so
        // that the eight elements starting at any in-bounds pixel are inside the allocation.
        unsafe {
            *pixels.get_unchecked_mut(index) = Colour::new(values[0], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 1) = Colour::new(values[1], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 2) = Colour::new(values[2], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 3) = Colour::new(values[3], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 4) = Colour::new(values[4], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 5) = Colour::new(values[5], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 6) = Colour::new(values[6], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 7) = Colour::new(values[7], 0.0, 0.0, 1.0);
        }
    }

    /// # Safety
    ///
    /// The same as [`set8_r_unchecked`](Self::set8_r_unchecked).
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn set8_r_unchecked_with_mask(&mut self, index: usize, reds: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // SAFETY: `get8_r_unchecked` and `set8_r_unchecked` share this function's contract.
        let current = unsafe { self.get8_r_unchecked(index) };
        let new_pixels = mask.select(reds, current);

        // SAFETY: as above.
        unsafe { self.set8_r_unchecked(index, new_pixels) };
    }
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` write into the row's padding, which
    /// isn't part of the image.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn set8_g_unchecked(&mut self, index: usize, greens: f32x8) {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = greens.to_array();
        let pixels = self.pixels.as_mut_slice();
        // SAFETY: the caller guarantees `index` is an in-bounds pixel, and `Grid` pads every row so
        // that the eight elements starting at any in-bounds pixel are inside the allocation.
        unsafe {
            *pixels.get_unchecked_mut(index) = Colour::new(values[0], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 1) = Colour::new(values[1], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 2) = Colour::new(values[2], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 3) = Colour::new(values[3], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 4) = Colour::new(values[4], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 5) = Colour::new(values[5], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 6) = Colour::new(values[6], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 7) = Colour::new(values[7], 0.0, 0.0, 1.0);
        }
    }

    /// # Safety
    ///
    /// The same as [`set8_g_unchecked`](Self::set8_g_unchecked).
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn set8_g_unchecked_with_mask(&mut self, index: usize, greens: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // SAFETY: `get8_g_unchecked` and `set8_g_unchecked` share this function's contract.
        let current = unsafe { self.get8_g_unchecked(index) };
        let new_pixels = mask.select(greens, current);

        // SAFETY: as above.
        unsafe { self.set8_g_unchecked(index, new_pixels) };
    }
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` write into the row's padding, which
    /// isn't part of the image.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn set8_b_unchecked(&mut self, index: usize, blues: f32x8) {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = blues.to_array();
        let pixels = self.pixels.as_mut_slice();
        // SAFETY: the caller guarantees `index` is an in-bounds pixel, and `Grid` pads every row so
        // that the eight elements starting at any in-bounds pixel are inside the allocation.
        unsafe {
            *pixels.get_unchecked_mut(index) = Colour::new(values[0], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 1) = Colour::new(values[1], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 2) = Colour::new(values[2], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 3) = Colour::new(values[3], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 4) = Colour::new(values[4], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 5) = Colour::new(values[5], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 6) = Colour::new(values[6], 0.0, 0.0, 1.0);
            *pixels.get_unchecked_mut(index + 7) = Colour::new(values[7], 0.0, 0.0, 1.0);
        }
    }

    /// # Safety
    ///
    /// The same as [`set8_b_unchecked`](Self::set8_b_unchecked).
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn set8_b_unchecked_with_mask(&mut self, index: usize, blues: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // SAFETY: `get8_b_unchecked` and `set8_b_unchecked` share this function's contract.
        let current = unsafe { self.get8_b_unchecked(index) };
        let new_pixels = mask.select(blues, current);

        // SAFETY: as above.
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
        Some(self.pixels.as_slice()[index])
    }

    /// # Safety
    ///
    /// `p` must lie inside the framebuffer: `p.x as usize < self.width()` and
    /// `p.y as usize < self.height()`.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn get_pixel_unchecked(&self, p: Vec2) -> Colour {
        let index = self.index(p.x as usize, p.y as usize);

        // SAFETY: the caller guarantees `p` lies inside the framebuffer, so `index` is in bounds.
        unsafe { *self.pixels.as_slice().get_unchecked(index) }
    }

    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` read the row's padding, whose values
    /// are unspecified.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn get8_r_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let pixels = self.pixels.as_slice();
        // SAFETY: the caller guarantees `index` is an in-bounds pixel, and `Grid` pads every row so
        // that the eight elements starting at any in-bounds pixel are inside the allocation.
        let values = unsafe {
            [
                pixels.get_unchecked(index).r,
                pixels.get_unchecked(index + 1).r,
                pixels.get_unchecked(index + 2).r,
                pixels.get_unchecked(index + 3).r,
                pixels.get_unchecked(index + 4).r,
                pixels.get_unchecked(index + 5).r,
                pixels.get_unchecked(index + 6).r,
                pixels.get_unchecked(index + 7).r,
            ]
        };
        f32x8::new(values)
    }
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` read the row's padding, whose values
    /// are unspecified.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn get8_g_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let pixels = self.pixels.as_slice();
        // SAFETY: the caller guarantees `index` is an in-bounds pixel, and `Grid` pads every row so
        // that the eight elements starting at any in-bounds pixel are inside the allocation.
        let values = unsafe {
            [
                pixels.get_unchecked(index).g,
                pixels.get_unchecked(index + 1).g,
                pixels.get_unchecked(index + 2).g,
                pixels.get_unchecked(index + 3).g,
                pixels.get_unchecked(index + 4).g,
                pixels.get_unchecked(index + 5).g,
                pixels.get_unchecked(index + 6).g,
                pixels.get_unchecked(index + 7).g,
            ]
        };
        f32x8::new(values)
    }
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` read the row's padding, whose values
    /// are unspecified.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "internals"),
        expect(dead_code, reason = "unused; kept rather than deleted for now")
    )]
    pub unsafe fn get8_b_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let pixels = self.pixels.as_slice();
        // SAFETY: the caller guarantees `index` is an in-bounds pixel, and `Grid` pads every row so
        // that the eight elements starting at any in-bounds pixel are inside the allocation.
        let values = unsafe {
            [
                pixels.get_unchecked(index).b,
                pixels.get_unchecked(index + 1).b,
                pixels.get_unchecked(index + 2).b,
                pixels.get_unchecked(index + 3).b,
                pixels.get_unchecked(index + 4).b,
                pixels.get_unchecked(index + 5).b,
                pixels.get_unchecked(index + 6).b,
                pixels.get_unchecked(index + 7).b,
            ]
        };
        f32x8::new(values)
    }

    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` read the row's padding, whose values
    /// are unspecified.
    #[inline(always)]
    pub unsafe fn get8_unchecked(&self, index: usize) -> ColourSimd {
        debug_assert!(index + 8 <= self.pixels.as_slice().len());

        // TODO: look into different storage layout of framebuffer to make this more efficient, e.g. SoA instead of AoS
        let pixels = self.pixels.as_slice();
        // SAFETY: the caller guarantees `index` is an in-bounds pixel, and `Grid` pads every row so
        // that the eight elements starting at any in-bounds pixel are inside the allocation.
        let [p0, p1, p2, p3, p4, p5, p6, p7] = unsafe {
            [
                pixels.get_unchecked(index),
                pixels.get_unchecked(index + 1),
                pixels.get_unchecked(index + 2),
                pixels.get_unchecked(index + 3),
                pixels.get_unchecked(index + 4),
                pixels.get_unchecked(index + 5),
                pixels.get_unchecked(index + 6),
                pixels.get_unchecked(index + 7),
            ]
        };

        ColourSimd {
            r: f32x8::new([p0.r, p1.r, p2.r, p3.r, p4.r, p5.r, p6.r, p7.r]),
            g: f32x8::new([p0.g, p1.g, p2.g, p3.g, p4.g, p5.g, p6.g, p7.g]),
            b: f32x8::new([p0.b, p1.b, p2.b, p3.b, p4.b, p5.b, p6.b, p7.b]),
            a: f32x8::new([p0.a, p1.a, p2.a, p3.a, p4.a, p5.a, p6.a, p7.a]),
        }
    }
}

/// Accessors that only the benches use, through the `internals` feature.
#[cfg(feature = "internals")]
impl FrameBuffer {
    pub fn pixels(&self) -> &[Colour] {
        self.pixels.as_slice()
    }

    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y) + i` for some `x < self.width()`,
    /// `y < self.height()` and `i < 8`. An index past the end of row `y` reads the row's
    /// padding, whose value is unspecified.
    #[inline(always)]
    pub unsafe fn get_pixel_index_unchecked(&self, index: usize) -> Colour {
        // SAFETY: the caller guarantees `index` is one of the eight elements starting at an
        // in-bounds pixel, and `Grid` pads every row so that all eight are inside the allocation.
        unsafe { *self.pixels.as_slice().get_unchecked(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(width: usize, height: usize) -> FrameBuffer {
        let mut framebuffer = FrameBuffer::new(width, height);
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 3.0 - 1.0
        };

        for y in 0..height {
            for x in 0..width {
                let mut colour = Colour::new(next(), next(), next(), next());
                if (x + y) % 17 == 0 {
                    colour.g = f32::NAN;
                }
                framebuffer.set_pixel(Vec2::new(x as f32, y as f32), colour);
            }
        }

        framebuffer
    }

    fn bits(pixels: &[Colour]) -> Vec<[u32; 4]> {
        pixels
            .iter()
            .map(|c| [c.r, c.g, c.b, c.a].map(f32::to_bits))
            .collect()
    }

    #[test]
    fn write_pixels_u32_matches_scalar_to_u32() {
        for (w, h) in [(64, 3), (37, 5), (7, 2), (1, 1), (640, 4)] {
            let framebuffer = filled(w, h);

            let expected: Vec<u32> = framebuffer
                .pixels_compact()
                .into_iter()
                .map(Colour::to_u32)
                .collect();

            assert_eq!(framebuffer.pixels_u32(), expected, "{w}x{h}");
        }
    }

    #[test]
    fn write_pixels_writes_compact_rows_and_leaves_the_rest() {
        let framebuffer = filled(37, 5);
        let mut out = vec![Colour::RED; 37 * 5 + 3];

        framebuffer.write_pixels(&mut out);

        assert_eq!(bits(&out[..37 * 5]), bits(&framebuffer.pixels_compact()));
        assert_eq!(bits(&out[37 * 5..]), bits(&[Colour::RED; 3]));
    }
}
