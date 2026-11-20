use crate::extent::Extent;

/// A 2D grid whose rows are padded for SIMD access.
pub(crate) struct Grid<T> {
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<T>,
}

impl<T: Copy> Grid<T> {
    const SIMD_WIDTH: usize = 8;

    /// Leaves room for a full SIMD group starting on the last pixel of a row, so unchecked
    /// 8-wide loads and stores there stay inside the grid.
    fn stride_for(width: usize) -> usize {
        (width + Self::SIMD_WIDTH - 1).next_multiple_of(Self::SIMD_WIDTH)
    }

    pub(crate) fn new(width: usize, height: usize, fill: T) -> Self {
        let stride = Self::stride_for(width);

        Self {
            width,
            height,
            stride,
            data: vec![fill; stride * height],
        }
    }

    #[inline(always)]
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    #[inline(always)]
    pub(crate) fn height(&self) -> usize {
        self.height
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// The visible pixels of row `y`, without the SIMD padding.
    #[inline(always)]
    pub(crate) fn row(&self, y: usize) -> &[T] {
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    #[inline(always)]
    pub(crate) fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        y * self.stride + x
    }

    #[inline(always)]
    pub(crate) fn index_unchecked(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }

    pub(crate) fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    /// Contents are unspecified afterwards.
    pub(crate) fn resize(&mut self, width: usize, height: usize, fill: T) {
        self.width = width;
        self.height = height;
        self.stride = Self::stride_for(width);
        self.data.resize(self.stride * height, fill);
    }

    /// Copies the `size` region of `src` whose top-left corner is `src_origin` to `dst_origin`.
    pub(crate) fn blit_from(
        &mut self,
        dst_origin: (usize, usize),
        src: &Grid<T>,
        src_origin: (usize, usize),
        size: Extent,
    ) {
        let ((dst_x, dst_y), (src_x, src_y)) = (dst_origin, src_origin);
        let Extent { width, height } = size;
        debug_assert!(dst_x + width <= self.width && dst_y + height <= self.height);
        debug_assert!(src_x + width <= src.width && src_y + height <= src.height);

        for row in 0..height {
            let s = (src_y + row) * src.stride + src_x;
            let d = (dst_y + row) * self.stride + dst_x;

            self.data[d..d + width].copy_from_slice(&src.data[s..s + width]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(width: usize, height: usize) -> Grid<u32> {
        let mut grid = Grid::new(width, height, 0);
        for y in 0..height {
            for x in 0..width {
                let index = grid.index(x, y);
                grid.as_mut_slice()[index] = (y * 1000 + x) as u32;
            }
        }
        grid
    }

    #[test]
    fn rows_leave_room_for_a_simd_group_on_the_last_pixel() {
        for width in [1, 7, 8, 37, 64, 640] {
            let grid = Grid::new(width, 3, 0u32);
            let last_group_end = grid.index(width - 1, 2) + 8;

            assert!(last_group_end <= grid.as_slice().len(), "width {width}");
            assert_eq!(grid.row(2).len(), width);
        }
    }

    #[test]
    fn blit_from_copies_only_the_requested_region() {
        let src = numbered(20, 10);
        let mut dst = Grid::new(30, 12, 7u32);

        dst.blit_from((5, 3), &src, (2, 1), Extent::new(10, 6));

        for y in 0..12 {
            for x in 0..30 {
                let got = dst.row(y)[x];

                if (5..15).contains(&x) && (3..9).contains(&y) {
                    assert_eq!(got, ((y - 3 + 1) * 1000 + (x - 5 + 2)) as u32, "({x},{y})");
                } else {
                    assert_eq!(got, 7, "({x},{y}) should be untouched");
                }
            }
        }
    }

    #[test]
    fn resize_reuses_the_allocation_within_its_capacity() {
        let mut grid = Grid::new(64, 64, 0u32);
        let (data, capacity) = (grid.data.as_ptr(), grid.data.capacity());

        grid.resize(40, 30, 0);
        grid.fill(9);

        assert_eq!((grid.width(), grid.height()), (40, 30));
        assert_eq!((grid.data.as_ptr(), grid.data.capacity()), (data, capacity));
        assert!(grid.as_slice().iter().all(|&v| v == 9));

        grid.resize(64, 64, 0);

        assert_eq!((grid.width(), grid.height()), (64, 64));
        assert_eq!((grid.data.as_ptr(), grid.data.capacity()), (data, capacity));
    }
}
