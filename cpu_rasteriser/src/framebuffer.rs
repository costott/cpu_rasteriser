use wide::{f32x8, i32x8};

use crate::grid::Grid;
use crate::prelude::*;

/// A 2D buffer of [`Colour`] pixels: the colour attachment of a render target, and of each tile
/// while a pass renders.
///
/// Pixels are stored row by row, top row first. Each row is padded to a stride of at least
/// `width + 7` elements, a multiple of 8, so that an 8-wide access starting at any pixel of the row
/// stays inside the allocation; the unchecked 8-wide methods rely on this. Padding elements aren't
/// part of the buffer's contents and hold unspecified values.
///
/// Methods that take a [`Vec2`](crate::Vec2) position truncate its coordinates towards zero, so
/// a position addresses the pixel that contains it.
pub struct FrameBuffer {
    pixels: Grid<Colour>, // TODO: change to SoA layout for better SIMD performance - separate Vec<f32> for each channel, as rn profiling shows performance is dominated by branch misprediction in scalar loop of rasteriser callback, so need a set8_unchecked_with_mask
}
impl FrameBuffer {
    /// Creates a `width` × `height` framebuffer with every pixel
    /// [`Colour::BLACK`](crate::Colour::BLACK).
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: Grid::new(width, height, Colour::BLACK),
        }
    }

    /// Returns the width in pixels.
    #[inline(always)]
    pub fn width(&self) -> usize {
        self.pixels.width()
    }

    /// Returns the height in pixels.
    #[inline(always)]
    pub fn height(&self) -> usize {
        self.pixels.height()
    }

    /// Returns the width and height as an [`Extent`](crate::Extent).
    pub fn extent(&self) -> Extent {
        Extent::new(self.width(), self.height())
    }

    /// Returns a copy of the pixels without the row padding: `width * height` colours, row by row
    /// from the top.
    pub fn pixels_compact(&self) -> Vec<Colour> {
        let mut pixels = Vec::with_capacity(self.width() * self.height());

        for y in 0..self.height() {
            pixels.extend_from_slice(self.pixels.row(y));
        }

        pixels
    }

    /// Returns the pixels without the row padding, each packed by
    /// [`Colour::to_u32`](crate::Colour::to_u32): `width * height` values, row by row from the top.
    pub fn pixels_u32(&self) -> Vec<u32> {
        let mut pixels = vec![0; self.width() * self.height()];
        self.write_pixels_u32(&mut pixels);
        pixels
    }

    /// Sets every pixel, and the row padding, to `colour`.
    pub fn clear(&mut self, colour: Colour) {
        self.pixels.fill(colour);
    }

    /// Changes the size to `width` × `height`, reusing the allocation where it is large enough.
    ///
    /// The pixels' contents are unspecified afterwards; clear the buffer before reading it.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.pixels.resize(width, height, Colour::BLACK);
    }

    /// Returns the storage index of pixel (`x`, `y`) without checking that it is inside the buffer.
    ///
    /// The result is the `index` argument of the unchecked 8-wide methods.
    #[inline(always)]
    pub fn index_unchecked(&self, x: usize, y: usize) -> usize {
        self.pixels.index_unchecked(x, y)
    }

    /// Sets the pixel at `p` to `colour`, or does nothing if `p` is outside the buffer.
    ///
    /// # Known issues
    ///
    /// - A negative or NaN coordinate truncates to 0, so the call sets a pixel in column or row 0
    ///   instead of doing nothing ([#60](https://github.com/costott/cpu_rasteriser/issues/60)).
    pub fn set_pixel(&mut self, p: Vec2, colour: Colour) {
        let x = p.x as usize;
        let y = p.y as usize;

        if x >= self.width() || y >= self.height() {
            return;
        }

        let index = self.index(x, y);
        self.pixels.as_mut_slice()[index] = colour;
    }

    /// Sets the element at storage `index` to `colour`, without bounds checks.
    ///
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y) + i` for some `x < self.width()`,
    /// `y < self.height()` and `i < 8`. An index past the end of row `y` writes into the row's
    /// padding, which isn't part of the buffer's contents.
    pub unsafe fn set_pixel_index_unchecked(&mut self, index: usize, colour: Colour) {
        debug_assert!(index < self.pixels.as_slice().len());

        // SAFETY: the caller guarantees `index` is one of the eight elements starting at an
        // in-bounds pixel, and `Grid` pads every row so that all eight are inside the allocation.
        unsafe {
            *self.pixels.as_mut_slice().get_unchecked_mut(index) = colour;
        }
    }

    /// Returns the colour of the pixel at `p`, or `None` if `p` is outside the buffer.
    ///
    /// # Known issues
    ///
    /// - A negative or NaN coordinate truncates to 0, so the call reads a pixel in column or row 0
    ///   instead of returning `None` ([#60](https://github.com/costott/cpu_rasteriser/issues/60)).
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

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        self.pixels.index(x, y)
    }

    /// Returns the eight consecutive elements starting at storage `index`, one per lane, without
    /// bounds checks.
    ///
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

    /// Copies the `size` region of `src` whose top-left corner is `src_origin` to `dst_origin`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if the region lies outside either buffer. In release builds, a
    /// region that leaves a buffer is copied through the row padding and the next row, and the call
    /// panics only if the region runs past the end of a buffer's storage.
    pub fn blit_from(
        &mut self,
        dst_origin: (usize, usize),
        src: &FrameBuffer,
        src_origin: (usize, usize),
        size: Extent,
    ) {
        self.pixels
            .blit_from(dst_origin, &src.pixels, src_origin, size);
    }

    pub(crate) fn write_pixels(&self, out: &mut [Colour]) {
        let width = self.width();
        assert!(out.len() >= width * self.height());

        for y in 0..self.height() {
            out[y * width..(y + 1) * width].copy_from_slice(self.pixels.row(y));
        }
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
}
/// Unused methods, kept for now.
impl FrameBuffer {
    /// Sets the pixel at `p` to `colour`, without bounds checks.
    ///
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

    /// Writes the lanes of `reds` to the red channels of the eight consecutive elements starting at
    /// storage `index`, without bounds checks.
    ///
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` write into the row's padding, which
    /// isn't part of the buffer's contents.
    ///
    /// # Known issues
    ///
    /// - The other channels are overwritten too: green and blue become 0 and alpha 1
    ///   ([#23](https://github.com/costott/cpu_rasteriser/issues/23)).
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

    /// Like [`set8_r_unchecked`](Self::set8_r_unchecked), but lanes whose `mask` is zero keep their
    /// current red value.
    ///
    /// Mask lanes must be all ones or all zeros.
    ///
    /// # Safety
    ///
    /// The same as [`set8_r_unchecked`](Self::set8_r_unchecked).
    ///
    /// # Known issues
    ///
    /// - Every lane, masked or not, has its green and blue set to 0 and its alpha to 1
    ///   ([#23](https://github.com/costott/cpu_rasteriser/issues/23)).
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
    /// Writes the lanes of `greens` to the eight consecutive elements starting at storage `index`,
    /// without bounds checks.
    ///
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` write into the row's padding, which
    /// isn't part of the buffer's contents.
    ///
    /// # Known issues
    ///
    /// - Each lane is written to the red channel, not green, and green and blue become 0 and
    ///   alpha 1 ([#23](https://github.com/costott/cpu_rasteriser/issues/23)).
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

    /// Like [`set8_g_unchecked`](Self::set8_g_unchecked), but lanes whose `mask` is zero take their
    /// current green value instead of the lane of `greens`.
    ///
    /// Mask lanes must be all ones or all zeros.
    ///
    /// # Safety
    ///
    /// The same as [`set8_g_unchecked`](Self::set8_g_unchecked).
    ///
    /// # Known issues
    ///
    /// - Each selected value is written to the red channel, not green, and green and blue become 0
    ///   and alpha 1 ([#23](https://github.com/costott/cpu_rasteriser/issues/23)).
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
    /// Writes the lanes of `blues` to the eight consecutive elements starting at storage `index`,
    /// without bounds checks.
    ///
    /// # Safety
    ///
    /// `index` must be `self.index_unchecked(x, y)` for some `x < self.width()` and
    /// `y < self.height()`. Lanes past the end of row `y` write into the row's padding, which
    /// isn't part of the buffer's contents.
    ///
    /// # Known issues
    ///
    /// - Each lane is written to the red channel, not blue, and green and blue become 0 and alpha 1
    ///   ([#23](https://github.com/costott/cpu_rasteriser/issues/23)).
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

    /// Like [`set8_b_unchecked`](Self::set8_b_unchecked), but lanes whose `mask` is zero take their
    /// current blue value instead of the lane of `blues`.
    ///
    /// Mask lanes must be all ones or all zeros.
    ///
    /// # Safety
    ///
    /// The same as [`set8_b_unchecked`](Self::set8_b_unchecked).
    ///
    /// # Known issues
    ///
    /// - Each selected value is written to the red channel, not blue, and green and blue become 0
    ///   and alpha 1 ([#23](https://github.com/costott/cpu_rasteriser/issues/23)).
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

    /// Returns the colour of the pixel at `p`, without bounds checks.
    ///
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

    /// Returns the red channels of the eight consecutive elements starting at storage `index`,
    /// without bounds checks.
    ///
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
    /// Returns the green channels of the eight consecutive elements starting at storage `index`,
    /// without bounds checks.
    ///
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
    /// Returns the blue channels of the eight consecutive elements starting at storage `index`,
    /// without bounds checks.
    ///
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
}
impl std::fmt::Debug for FrameBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameBuffer")
            .field("width", &self.width())
            .field("height", &self.height())
            .finish_non_exhaustive()
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
