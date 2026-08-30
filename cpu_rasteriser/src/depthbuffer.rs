use wide::f32x8;

use crate::maths::Vec2;

pub struct DepthBuffer {
    width: usize,
    height: usize,
    buffer: Vec<f32>,
}
impl DepthBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![1.0; width * height],
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
        self.buffer.resize(width * height, 1.0);
    }

    pub fn set_depth(&mut self, p: Vec2, depth: f32) {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return;
        }
        let index = (y as usize) * self.width + (x as usize);
        self.buffer[index] = depth;
    }

    pub unsafe fn set_depth_unchecked(&mut self, p: Vec2, depth: f32) {
        let index = (p.y as usize) * self.width + (p.x as usize);

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

    pub fn get(&self, p: Vec2) -> f32 {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return 1.0;
        }
        let index = (y as usize) * self.width + (x as usize);
        self.buffer[index]
    }

    pub unsafe fn get_unchecked(&self, p: Vec2) -> f32 {
        let index = (p.y as usize) * self.width + (p.x as usize);

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
