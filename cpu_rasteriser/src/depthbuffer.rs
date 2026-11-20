use wide::f32x8;

use crate::extent::Extent;
use crate::grid::Grid;
use crate::maths::Vec2;

pub struct DepthBuffer {
    depths: Grid<f32>,
}
impl DepthBuffer {
    const FAR: f32 = 1.0;

    pub fn new(width: usize, height: usize) -> Self {
        Self {
            depths: Grid::new(width, height, Self::FAR),
        }
    }

    pub fn width(&self) -> usize {
        self.depths.width()
    }

    pub fn height(&self) -> usize {
        self.depths.height()
    }

    pub fn clear(&mut self, depth: f32) {
        self.depths.fill(depth);
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.depths.resize(width, height, Self::FAR);
    }

    pub(crate) fn blit_from(
        &mut self,
        dst_origin: (usize, usize),
        src: &DepthBuffer,
        src_origin: (usize, usize),
        size: Extent,
    ) {
        self.depths
            .blit_from(dst_origin, &src.depths, src_origin, size);
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        self.depths.index(x, y)
    }

    #[inline(always)]
    pub fn index_unchecked(&self, x: usize, y: usize) -> usize {
        self.depths.index_unchecked(x, y)
    }

    #[inline(always)]
    pub fn set_depth(&mut self, position: Vec2, depth: f32) {
        let x = position.x as usize;
        let y = position.y as usize;

        let index = self.index(x, y);
        self.depths.as_mut_slice()[index] = depth;
    }

    pub unsafe fn set_depth_unchecked(&mut self, p: Vec2, depth: f32) {
        let index = self.index_unchecked(p.x as usize, p.y as usize);

        unsafe {
            *self.depths.as_mut_slice().get_unchecked_mut(index) = depth;
        }
    }

    #[inline(always)]
    pub unsafe fn set8_unchecked(&mut self, index: usize, depth: f32x8) {
        debug_assert!(index + 8 <= self.depths.as_slice().len());

        // check if stores auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD store
        let values = depth.to_array();
        let buffer = self.depths.as_mut_slice();
        unsafe {
            *buffer.get_unchecked_mut(index) = values[0];
            *buffer.get_unchecked_mut(index + 1) = values[1];
            *buffer.get_unchecked_mut(index + 2) = values[2];
            *buffer.get_unchecked_mut(index + 3) = values[3];
            *buffer.get_unchecked_mut(index + 4) = values[4];
            *buffer.get_unchecked_mut(index + 5) = values[5];
            *buffer.get_unchecked_mut(index + 6) = values[6];
            *buffer.get_unchecked_mut(index + 7) = values[7];
        }
    }

    pub unsafe fn set8_unchecked_with_mask(&mut self, index: usize, depth: f32x8, mask: f32x8) {
        debug_assert!(index + 8 <= self.depths.as_slice().len());

        let current_depth = unsafe { self.get8_unchecked(index) };
        let new_depth = mask.select(depth, current_depth);

        unsafe { self.set8_unchecked(index, new_depth) };
    }

    #[inline(always)]
    pub fn get(&self, position: Vec2) -> f32 {
        let x = position.x as usize;
        let y = position.y as usize;

        self.depths.as_slice()[self.index(x, y)]
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, p: Vec2) -> f32 {
        let index = self.index_unchecked(p.x as usize, p.y as usize);

        unsafe { *self.depths.as_slice().get_unchecked(index) }
    }

    #[inline(always)]
    pub unsafe fn get8_unchecked(&self, index: usize) -> f32x8 {
        debug_assert!(index + 8 <= self.depths.as_slice().len());

        // check if loads auto vectorise with LLVM, otherwise look into finding out how to get as 1 SIMD load
        let buffer = self.depths.as_slice();
        let values = [
            unsafe { *buffer.get_unchecked(index) },
            unsafe { *buffer.get_unchecked(index + 1) },
            unsafe { *buffer.get_unchecked(index + 2) },
            unsafe { *buffer.get_unchecked(index + 3) },
            unsafe { *buffer.get_unchecked(index + 4) },
            unsafe { *buffer.get_unchecked(index + 5) },
            unsafe { *buffer.get_unchecked(index + 6) },
            unsafe { *buffer.get_unchecked(index + 7) },
        ];
        f32x8::new(values)
    }
}
