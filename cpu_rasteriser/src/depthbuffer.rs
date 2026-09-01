use wide::f32x8;

use crate::maths::Vec2;

pub struct DepthBuffer {
    width: usize,
    stride: usize, // physical row stride, including SIMD padding
    height: usize,
    buffer: Vec<f32>,
}
impl DepthBuffer {
    const SIMD_WIDTH: usize = 8;

    pub fn new(width: usize, height: usize) -> Self {
        let stride = (width + Self::SIMD_WIDTH - 1).next_multiple_of(Self::SIMD_WIDTH);

        Self {
            width,
            stride,
            height,
            buffer: vec![1.0; stride * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn clear(&mut self, depth: f32) {
        self.buffer.fill(depth);
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.stride = (width + Self::SIMD_WIDTH - 1).next_multiple_of(Self::SIMD_WIDTH);
        self.buffer.resize(self.stride * height, 1.0);
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        y * self.stride + x
    }

    #[inline(always)]
    pub fn index_unchecked(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }

    #[inline(always)]
    pub fn set_depth(&mut self, position: Vec2, depth: f32) {
        let x = position.x as usize;
        let y = position.y as usize;

        let index = self.index(x, y);
        self.buffer[index] = depth;
    }

    pub unsafe fn set_depth_unchecked(&mut self, p: Vec2, depth: f32) {
        let index = self.index_unchecked(p.x as usize, p.y as usize);

        unsafe {
            *self.buffer.get_unchecked_mut(index) = depth;
        }
    }

    #[inline(always)]
    pub unsafe fn set8_unchecked(&mut self, index: usize, depth: f32x8) {
        debug_assert!(index + 8 <= self.buffer.len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = depth.to_array();
        unsafe {
            *self.buffer.get_unchecked_mut(index) = values[0];
            *self.buffer.get_unchecked_mut(index + 1) = values[1];
            *self.buffer.get_unchecked_mut(index + 2) = values[2];
            *self.buffer.get_unchecked_mut(index + 3) = values[3];
            *self.buffer.get_unchecked_mut(index + 4) = values[4];
            *self.buffer.get_unchecked_mut(index + 5) = values[5];
            *self.buffer.get_unchecked_mut(index + 6) = values[6];
            *self.buffer.get_unchecked_mut(index + 7) = values[7];
        }
    }

    pub unsafe fn set8_unchecked_with_mask(&mut self, index: usize, depth: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.buffer.len());

        let current_depth = unsafe { self.get8_unchecked(index) };
        let new_depth = mask.select(depth, current_depth);

        unsafe { self.set8_unchecked(index, new_depth) };
    }

    #[inline(always)]
    pub fn get(&self, position: Vec2) -> f32 {
        let x = position.x as usize;
        let y = position.y as usize;

        self.buffer[self.index(x, y)]
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, p: Vec2) -> f32 {
        let index = self.index_unchecked(p.x as usize, p.y as usize);

        unsafe { *self.buffer.get_unchecked(index) }
    }

    #[inline(always)]
    pub unsafe fn get8_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.buffer.len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let values = [
            unsafe { *self.buffer.get_unchecked(index) },
            unsafe { *self.buffer.get_unchecked(index + 1) },
            unsafe { *self.buffer.get_unchecked(index + 2) },
            unsafe { *self.buffer.get_unchecked(index + 3) },
            unsafe { *self.buffer.get_unchecked(index + 4) },
            unsafe { *self.buffer.get_unchecked(index + 5) },
            unsafe { *self.buffer.get_unchecked(index + 6) },
            unsafe { *self.buffer.get_unchecked(index + 7) },
        ];
        f32x8::new(values)
    }
}
