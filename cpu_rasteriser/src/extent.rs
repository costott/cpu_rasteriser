/// A width and height in pixels.
///
/// An extent sizes a [`RenderTarget`](crate::RenderTarget) and gives the area a
/// [`Viewport::full`](crate::Viewport::full) covers. A zero width or height is allowed and
/// describes an empty area.
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let extent = Extent::new(320, 240);
/// let target = RenderTarget::new(extent);
///
/// assert_eq!(target.extent(), extent);
/// assert_eq!(Viewport::full(&extent), Viewport::new(0, 0, 320, 240));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// The width in pixels.
    pub width: usize,
    /// The height in pixels.
    pub height: usize,
}
impl Extent {
    /// Creates an extent of `width` by `height` pixels.
    ///
    /// See the [type-level example](Extent#examples).
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}
