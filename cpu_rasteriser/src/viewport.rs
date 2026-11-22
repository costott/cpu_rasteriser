use crate::extent::Extent;

/// The rectangle of a render target that normalised device coordinates map to, in pixels.
///
/// After the perspective divide, a vertex at NDC `(x, y)` lands at the screen position
///
/// ```text
/// screen_x = viewport.x + (x + 1) / 2 * viewport.width
/// screen_y = viewport.y + (1 - y) / 2 * viewport.height
/// ```
///
/// so NDC `(-1, 1)` is the viewport's top-left corner and `(1, -1)` its bottom-right corner.
/// Screen y points down, and `(x, y)` is the top-left pixel corner of the rectangle.
///
/// The viewport only maps coordinates; it is not a scissor test. Geometry is clipped to the clip
/// volume before mapping, which keeps it within the viewport, and nothing is drawn outside the
/// render target, so a viewport that extends past the target is cut off at its edges.
///
/// A [`RenderPassDescriptor`](crate::RenderPassDescriptor) carries the viewport for every draw in
/// the pass.
///
/// # Examples
///
/// Splitting a target into two side-by-side views, for example for split-screen rendering, uses
/// one viewport per half:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// let extent = Extent::new(640, 480);
///
/// let left = Viewport::new(0, 0, extent.width / 2, extent.height);
/// let right = Viewport::new(extent.width / 2, 0, extent.width / 2, extent.height);
///
/// assert_eq!(right.x, 320);
/// assert_eq!(left.width + right.width, Viewport::full(&extent).width);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    /// The x coordinate of the left edge, in pixels.
    pub x: usize,
    /// The y coordinate of the top edge, in pixels.
    pub y: usize,
    /// The width in pixels.
    pub width: usize,
    /// The height in pixels.
    pub height: usize,
}
impl Viewport {
    /// Creates a viewport whose top-left corner is at `(x, y)` and which is `width` by `height`
    /// pixels.
    ///
    /// See the [type-level example](Viewport#examples).
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

    /// Creates a viewport that covers the whole of `extent`, starting at `(0, 0)`.
    ///
    /// This is the usual viewport for a render pass that draws to an entire target.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    ///
    /// let target = RenderTarget::new(Extent::new(800, 600));
    /// let viewport = Viewport::full(&target.extent());
    ///
    /// assert_eq!(viewport, Viewport::new(0, 0, 800, 600));
    /// ```
    pub fn full(extent: &Extent) -> Self {
        Self {
            x: 0,
            y: 0,
            width: extent.width,
            height: extent.height,
        }
    }
}
