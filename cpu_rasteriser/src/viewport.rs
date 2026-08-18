use crate::extent::Extent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}
impl Viewport {
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    // pub fn full(render_target: &RenderTarget) -> Self {
    //     Self {
    //         x: 0,
    //         y: 0,
    //         width: render_target.width(),
    //         height: render_target.height(),
    //     }
    // }

    pub fn full(extent: &Extent) -> Self {
        Self {
            x: 0,
            y: 0,
            width: extent.width,
            height: extent.height,
        }
    }
}
