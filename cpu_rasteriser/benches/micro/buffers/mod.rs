use std::array;
use std::hint::black_box;
use std::iter::StepBy;
use std::ops::{BitXor, Range};

use cpu_rasteriser::internals::{DepthBuffer, FrameBuffer};
use cpu_rasteriser::prelude::*;
use cpu_rasteriser::wide::f32x8;
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group};

use crate::fixtures::{EDGE_HEIGHT, TARGET_EXTENT, TILE, Variant, register, unit};

mod depthbuffer;
mod framebuffer;

/// The top-left corner of an interior tile of `TARGET`, where the tile-sized blits merge.
const TILE_ORIGIN: (usize, usize) = (3 * TILE, 2 * TILE);

const _: () = assert!(
    TILE_ORIGIN.0 + TILE <= TARGET_EXTENT.width && TILE_ORIGIN.1 + TILE <= TARGET_EXTENT.height
);

const LANES: usize = 8;
const GROUPS_PER_ROW: usize = TILE / LANES;

/// The first column of a row's span cycles through `MIN_SPAN_START..=MAX_SPAN_START`: every
/// offset from a multiple of `LANES`, so no row's groups are aligned.
const MIN_SPAN_START: usize = 1;
const MAX_SPAN_START: usize = LANES - 1;

/// Every span has exactly `GROUPS_PER_ROW` groups, and the last starts inside the tile, so its
/// lanes can hang over into the row's padding but not past it.
const _: () = assert!(
    TILE.is_multiple_of(LANES)
        && MAX_SPAN_START + LANES * (GROUPS_PER_ROW - 1) < TILE
        && MIN_SPAN_START + LANES * GROUPS_PER_ROW >= TILE
);

/// The pixels a tile-sized case visits in one iteration: each row's span, from `span_start` to
/// the tile's right edge.
const PIXELS: usize = {
    let mut count = 0;
    let mut y = 0;
    while y < TILE {
        count += TILE - span_start(y);
        y += 1;
    }
    count
};

fn buffers_group(c: &mut Criterion) -> BenchmarkGroup<'_, WallTime> {
    let mut group = c.benchmark_group("buffers");
    group.throughput(Throughput::Elements(PIXELS as u64));
    group
}

/// Benchmarks `load` over every position of `spans` (`pixel_spans` or `group_spans`), folding each
/// result into one value. `load` receives a buffer of `TILE` × `TILE` pixels and the column
/// `x < TILE` and row `y < TILE` of a pixel inside it, or of a group's first pixel, whose other
/// lanes can reach into the row's padding.
///
/// Folding with XOR keeps each result alive without a floating-point sum's dependency chain, which
/// would bound the loop at the add's latency.
fn bench_loads<B, A, S, X>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    variant: Variant,
    buffer: &B,
    spans: fn() -> S,
    load: impl Fn(&B, usize, usize) -> A,
) where
    B: TileBuffer,
    A: Copy + Default + BitXor<Output = A>,
    S: Iterator<Item = (usize, X)>,
    X: Iterator<Item = usize>,
{
    assert_tile(buffer);
    register(group, case, variant, |bench| {
        bench.iter(|| {
            let buffer = black_box(buffer);
            let mut folded = A::default();
            for_each(spans(), |x, y| {
                folded = folded ^ load(buffer, x, y);
            });
            black_box(folded)
        })
    });
}

/// Benchmarks `store` over every position of `spans`, passing each the next of `inputs` (from
/// `inputs` over the same spans). `store` receives the buffer and a position as for
/// [`bench_loads`].
fn bench_stores<B, T, S, X>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    variant: Variant,
    buffer: &mut B,
    spans: fn() -> S,
    inputs: &[T],
    store: impl Fn(&mut B, usize, usize, T),
) where
    B: TileBuffer,
    T: Copy,
    S: Iterator<Item = (usize, X)>,
    X: ExactSizeIterator<Item = usize>,
{
    assert_tile(buffer);
    register(group, case, variant, |bench| {
        bench.iter(|| {
            for_each_with_inputs(spans(), black_box(inputs), |x, y, input| {
                store(buffer, x, y, input);
            });
            black_box(&*buffer);
        })
    });
}

/// A buffer the tile-sized cases can walk. They require it to be `TILE` × `TILE`, which the
/// `unsafe` accesses in the cases rely on.
trait TileBuffer {
    fn extent(&self) -> Extent;
}

impl TileBuffer for FrameBuffer {
    fn extent(&self) -> Extent {
        FrameBuffer::extent(self)
    }
}

impl TileBuffer for DepthBuffer {
    fn extent(&self) -> Extent {
        Extent::new(self.width(), self.height())
    }
}

fn assert_tile(buffer: &impl TileBuffer) {
    assert_eq!(
        buffer.extent(),
        Extent::new(TILE, TILE),
        "a tile-sized case needs a tile-sized buffer"
    );
}

/// Calls `visit(x, y, input)` for each (`x`, `y`) of `spans`, with the next of `inputs`.
///
/// # Panics
///
/// Panics if `inputs` has fewer elements than `spans` yields positions.
fn for_each_with_inputs<T: Copy, S: ExactSizeIterator<Item = usize>>(
    spans: impl Iterator<Item = (usize, S)>,
    inputs: &[T],
    mut visit: impl FnMut(usize, usize, T),
) {
    let mut remaining = inputs;
    for (y, xs) in spans {
        let (row, rest) = remaining.split_at(xs.len());
        remaining = rest;
        for (x, &input) in xs.zip(row) {
            visit(x, y, input);
        }
    }
}

/// Returns `input(x, y)` for each position of `spans`, in its order.
fn inputs<T: Copy, S: Iterator<Item = (usize, X)>, X: Iterator<Item = usize>>(
    spans: fn() -> S,
    input: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let mut inputs = Vec::new();
    for_each(spans(), |x, y| inputs.push(input(x, y)));
    inputs
}

/// Calls `visit(x, y)` for each (`x`, `y`) of `spans`, row by row.
fn for_each<S: Iterator<Item = usize>>(
    spans: impl Iterator<Item = (usize, S)>,
    mut visit: impl FnMut(usize, usize),
) {
    for (y, xs) in spans {
        for x in xs {
            visit(x, y);
        }
    }
}

/// Every row's span of pixels, as its row `y` and columns, from `span_start(y)` to the tile's right
/// edge. Always `x < TILE` and `y < TILE`.
fn pixel_spans() -> impl Iterator<Item = (usize, Range<usize>)> {
    (0..TILE).map(|y| (y, span_start(y)..TILE))
}

/// Every row's span of 8-wide groups, as its row `y` and the column of each group's first pixel.
/// They cover `PIXELS`. Always `x < TILE` and `y < TILE`.
fn group_spans() -> impl Iterator<Item = (usize, StepBy<Range<usize>>)> {
    (0..TILE).map(|y| (y, (span_start(y)..TILE).step_by(LANES)))
}

const fn span_start(y: usize) -> usize {
    MIN_SPAN_START + y % (MAX_SPAN_START - MIN_SPAN_START + 1)
}

/// Returns the position the scalar accessors take for pixel (`x`, `y`).
fn position(x: usize, y: usize) -> Vec2 {
    Vec2::new(x as f32, y as f32)
}

/// Returns `value` for the eight pixels of the group starting at (`x`, `y`), one per lane. Lanes
/// past the tile's right edge take the values their positions would have.
fn lanes(x: usize, y: usize, value: impl Fn(usize, usize) -> f32) -> f32x8 {
    f32x8::from(array::from_fn(|lane| value(x + lane, y)))
}

/// Whether a masked store writes pixel (`x`, `y`): scattered, for about half of the pixels.
///
/// Provisional: a rate at which neither outcome dominates, so a branch on it is unpredictable. The
/// rate in a real scene depends on the scene. A modern predictor may still learn a sequence of
/// this length when the same one repeats every iteration.
fn passes(x: usize, y: usize) -> bool {
    unit(2 * (y * TILE + x) + 1) < 0.5
}

/// Returns the lane mask of the group starting at (`x`, `y`): all ones in the lanes whose pixel is
/// inside the tile and `passes`, all zeros in the rest, as the rasteriser's masks are.
fn group_mask(x: usize, y: usize) -> f32x8 {
    f32x8::from(array::from_fn(|lane| {
        if x + lane < TILE && passes(x + lane, y) {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        }
    }))
}

criterion_group!(
    benches,
    framebuffer::benchmark_framebuffer_access,
    framebuffer::benchmark_framebuffer_storage,
    depthbuffer::benchmark_depthbuffer_access,
    depthbuffer::benchmark_depthbuffer_storage,
);
