use cpu_rasteriser::prelude::*;

use crate::shaders::Vertex;

/// The target of every case except the resolution ones.
pub(crate) const EXTENT: Extent = Extent {
    width: 1280,
    height: 720,
};

/// The colour every frame starts from. No palette colour, and no blend of palette colours over
/// this one, equals it, so a pixel that differs from it was drawn.
pub(crate) const CLEAR: Colour = Colour {
    r: 0.1,
    g: 0.1,
    b: 0.1,
    a: 1.0,
};

/// The depth every frame starts from: the far plane.
pub(crate) const FAR: f32 = 1.0;

/// Four distinct colours with every channel above `CLEAR`'s.
const PALETTE: [[f32; 3]; 4] = [
    [0.9, 0.2, 0.2],
    [0.2, 0.8, 0.3],
    [0.2, 0.3, 0.9],
    [0.9, 0.8, 0.2],
];

/// The alpha of every colour in a blended variant. Provisional: any value strictly between 0 and
/// 1 blends; a half weights source and destination equally, so every layer shows in the result.
const BLEND_ALPHA: f32 = 0.5;

/// The size in pixels of the cells of `small_triangles`. Provisional: two triangles per 5×12
/// cell give 30,720 triangles on `EXTENT`, near the intended 30,000. Issue #35 tunes it.
const SMALL_CELL: (usize, usize) = (5, 12);

/// The depth of the `small_triangles` layer, inside the clip volume and nearer than `FAR`.
const SMALL_DEPTH: f32 = 0.0;

/// The depth of each `large_triangles` layer, in drawing order. They aren't sorted by depth: the
/// nearest layer, the second, is neither first nor last, so depth testing keeps it where drawing
/// order alone would keep the fourth. All lie inside the clip volume, and `f32` represents them
/// exactly.
const LARGE_DEPTHS: [f32; 4] = [0.5, -0.25, 0.75, 0.0];

/// The indices of a cell's two triangles, relative to its first vertex.
const CELL_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// The fixed-function state a workload is drawn with.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Variant {
    Opaque,
    Blended,
    DepthOpaque,
    DepthBlended,
}
impl Variant {
    /// Whether fragments are depth tested and write their depth.
    pub(crate) fn depth(self) -> bool {
        matches!(self, Self::DepthOpaque | Self::DepthBlended)
    }

    /// Whether fragments are alpha blended onto the target rather than replacing it.
    pub(crate) fn blended(self) -> bool {
        matches!(self, Self::Blended | Self::DepthBlended)
    }

    fn alpha(self) -> f32 {
        if self.blended() { BLEND_ALPHA } else { 1.0 }
    }
}

/// The passes of one frame, each a list of layers drawn in order.
///
/// Every layer tiles the whole target. A pass after the first loads the colour, and the depth if
/// the variant has it, so the passes together draw the layers in order, as one pass would.
pub(crate) struct Workload {
    pub(crate) extent: Extent,
    pub(crate) variant: Variant,
    pub(crate) passes: Vec<Vec<Layer>>,
}
impl Workload {
    /// A layer of 30,720 small triangles, to stress per-triangle costs: geometry processing,
    /// binning and tile set-up.
    pub(crate) fn small_triangles(variant: Variant) -> Self {
        let alpha = variant.alpha();
        let layer = grid(EXTENT, SMALL_CELL, SMALL_DEPTH, |column, row| {
            palette(column + row, alpha)
        });

        Self {
            extent: EXTENT,
            variant,
            passes: vec![vec![layer]],
        }
    }

    /// The opaque `small_triangles` frame drawn with one draw call per cell, 15,360 calls of two
    /// triangles each, to add per-draw costs: queuing the draw, its boxed command and its uniforms.
    /// It draws the same frame as `small_triangles` with `Variant::Opaque`, so the per-draw cost is
    /// the difference between the two cases' times.
    pub(crate) fn many_draws() -> Self {
        let mut workload = Self::small_triangles(Variant::Opaque);
        for layer in workload.passes.iter_mut().flatten() {
            layer.indices_per_draw = CELL_INDICES.len();
        }
        workload
    }

    /// Four overlapping layers of two large triangles each, to stress per-pixel costs.
    pub(crate) fn large_triangles(extent: Extent, variant: Variant) -> Self {
        Self {
            extent,
            variant,
            passes: vec![large_layers(extent, variant)],
        }
    }

    /// The layers of the depth-tested, blended `large_triangles` split over two passes. The
    /// second loads what the first drew and its depth, and the depth test rejects both of its
    /// layers, so only a pass that loaded both leaves the frame as a single pass would.
    pub(crate) fn multipass() -> Self {
        let variant = Variant::DepthBlended;
        let mut first = large_layers(EXTENT, variant);
        let second = first.split_off(2);

        Self {
            extent: EXTENT,
            variant,
            passes: vec![first, second],
        }
    }

    pub(crate) fn pixel_count(&self) -> usize {
        self.extent.width * self.extent.height
    }

    pub(crate) fn layers(&self) -> impl Iterator<Item = &Layer> {
        self.passes.iter().flatten()
    }
}

/// Flat-coloured cells of equal area that tile the target at one depth.
pub(crate) struct Layer {
    pub(crate) depth: f32,
    /// The colour of each cell.
    pub(crate) colours: Vec<Colour>,
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
    /// The number of indices in each of the draw calls the layer is drawn with.
    pub(crate) indices_per_draw: usize,
}

fn large_layers(extent: Extent, variant: Variant) -> Vec<Layer> {
    let alpha = variant.alpha();

    LARGE_DEPTHS
        .iter()
        .enumerate()
        .map(|(index, &depth)| {
            grid(extent, (extent.width, extent.height), depth, |_, _| {
                palette(index, alpha)
            })
        })
        .collect()
}

/// Tiles `extent` with cells of `cell` pixels at `depth`, each two counter-clockwise triangles
/// of the colour `colour(column, row)` that split it along a diagonal.
///
/// A pixel centre lies exactly on the diagonal of a `w`×`h` cell only if `(2j + 1)w = (2i + 1)h`
/// for some pixel `(i, j)`, which needs `w` and `h` to be divisible by the same power of two. The
/// assertion rules that out, so every pixel lies strictly inside one triangle and no
/// tie-breaking rule can drop it or draw it twice.
fn grid(
    extent: Extent,
    cell: (usize, usize),
    depth: f32,
    colour: impl Fn(usize, usize) -> Colour,
) -> Layer {
    let (width, height) = cell;
    assert!(
        extent.width.is_multiple_of(width) && extent.height.is_multiple_of(height),
        "a {width}×{height} cell doesn't tile {}×{}",
        extent.width,
        extent.height,
    );
    assert_ne!(
        width.trailing_zeros(),
        height.trailing_zeros(),
        "pixel centres would lie on the diagonal of a {width}×{height} cell",
    );

    let (columns, rows) = (extent.width / width, extent.height / height);
    let mut colours = Vec::with_capacity(columns * rows);
    let mut vertices = Vec::with_capacity(4 * columns * rows);
    let mut indices = Vec::with_capacity(6 * columns * rows);

    for row in 0..rows {
        for column in 0..columns {
            let cell_colour = colour(column, row);
            let (left, top) = (column * width, row * height);
            let (right, bottom) = (left + width, top + height);
            let first = vertices.len() as u32;

            colours.push(cell_colour);
            vertices.extend(
                [(left, top), (left, bottom), (right, bottom), (right, top)].map(|(x, y)| Vertex {
                    position: clip_position(extent, x, y, depth),
                    colour: cell_colour.into(),
                }),
            );
            indices.extend(CELL_INDICES.map(|index| first + index));
        }
    }

    Layer {
        depth,
        colours,
        indices_per_draw: indices.len(),
        vertices,
        indices,
    }
}

fn clip_position(extent: Extent, x: usize, y: usize, depth: f32) -> Vec4 {
    Vec4::new(
        2.0 * x as f32 / extent.width as f32 - 1.0,
        1.0 - 2.0 * y as f32 / extent.height as f32,
        depth,
        1.0,
    )
}

fn palette(index: usize, alpha: f32) -> Colour {
    let [r, g, b] = PALETTE[index % PALETTE.len()];
    Colour::new(r, g, b, alpha)
}
