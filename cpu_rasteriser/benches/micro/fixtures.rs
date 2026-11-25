use std::array;
use std::hint::black_box;

use cpu_rasteriser::internals::{ClipVertex, RasterVertex, Rect, TileBinner, Triangle2D};
use cpu_rasteriser::prelude::*;
use cpu_rasteriser::wide::f32x8;
use criterion::measurement::WallTime;
use criterion::{Bencher, BenchmarkGroup, BenchmarkId};

/// Items per batch for operations that take nanoseconds each.
///
/// Provisional starting size: single operations of 1–10 ns varied by up to 2× between identical
/// runs. At 1024 many cases still take only 0.1–1 µs per iteration, notably the SIMD ones, which
/// process `BATCH / 8` operations, and the cheap scalar operators. Timings that short are
/// layout-sensitive: byte-identical code measured 0.70 and 1.02 µs in two builds that differed
/// only in other functions of `mat4.rs`. Issue #35 sizes batches from measured variation.
pub const BATCH: usize = 1024;

/// The render target the rasterisation and buffer cases work on.
pub const TARGET: Rect = Rect {
    min_x: 0,
    min_y: 0,
    max_x: 640,
    max_y: 360,
};

/// The size of `TARGET` in pixels.
pub const TARGET_EXTENT: Extent = Extent {
    width: TARGET.max_x as usize,
    height: TARGET.max_y as usize,
};

/// The width and height of a tile in pixels.
pub const TILE: usize = TileBinner::TILE_SIZE as usize;

/// The height of the bottom row of tiles of `TARGET`, which its height isn't a multiple of.
pub const EDGE_HEIGHT: usize = TARGET.max_y as usize % TILE;

/// Varyings with one field of each supported type, standing for a typical shader's.
#[derive(Debug, Interpolate, SimdInterpolate)]
pub struct Varyings {
    pub colour: Vec3,
    pub uv: Vec2,
    pub light: f32,
}

impl Varyings {
    /// Returns the sum of every component, so a benchmark consumes all of them.
    pub fn sum(&self) -> f32 {
        self.colour.x + self.colour.y + self.colour.z + self.uv.x + self.uv.y + self.light
    }
}

impl VaryingsSimd {
    /// Returns the per-lane sum of every component; the lane-wise form of [`Varyings::sum`].
    pub fn sum(&self) -> f32x8 {
        let [r, g, b] = self.colour;
        let [u, v] = self.uv;
        r + g + b + u + v + self.light
    }
}

/// Returns varyings with every component scattered in `[0, 1)`, a different set for each `i`.
pub fn varyings(i: usize) -> Varyings {
    Varyings {
        colour: Vec3::new(unit(6 * i), unit(6 * i + 1), unit(6 * i + 2)),
        uv: Vec2::new(unit(6 * i + 3), unit(6 * i + 4)),
        light: unit(6 * i + 5),
    }
}

/// Returns a deterministic, evenly scattered value in `[0, 1)` for index `i`.
pub fn unit(i: usize) -> f32 {
    let hash = (i as u32).wrapping_mul(2_654_435_761);
    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// Returns a screen-space triangle with varying colours, uvs, light, depth and `1/w`.
pub fn triangle(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> Triangle2D<Varyings> {
    Triangle2D::new(
        vertex(
            a,
            0.2,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec2::new(0.0, 0.0),
            0.2,
        ),
        vertex(
            b,
            0.5,
            0.8,
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::new(1.0, 0.0),
            0.6,
        ),
        vertex(
            c,
            0.8,
            0.6,
            Vec3::new(0.0, 0.0, 1.0),
            Vec2::new(0.0, 1.0),
            1.0,
        ),
    )
}

fn vertex(
    position: (f32, f32),
    depth: f32,
    inv_w: f32,
    colour: Vec3,
    uv: Vec2,
    light: f32,
) -> RasterVertex<Varyings> {
    let varyings = Varyings { colour, uv, light };
    RasterVertex::new(
        Vec2::new(position.0, position.1),
        depth,
        inv_w,
        varyings.scale(inv_w),
    )
}

/// A pass-through stage in place of a vertex shader, which is user code and not what is measured:
/// its vertices already carry a clip-space position and varyings.
pub struct PassThrough;
impl VertexShader for PassThrough {
    type Vertex = ClipVertex<Varyings>;
    type Uniforms = ();
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, _: &()) -> (Vec4, Varyings) {
        (vertex.position, vertex.varyings)
    }
}

/// Which implementation of a case a benchmark runs; it decides the benchmark's id.
#[derive(Clone, Copy)]
pub enum Variant {
    Only,
    Scalar,
    Simd,
}

/// Benchmarks `op` over each of `inputs`, writing every result to an output buffer.
pub fn bench_map<A: Copy, R: Copy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    variant: Variant,
    inputs: &[A],
    op: impl Fn(A) -> R,
) {
    let mut outputs = vec![op(inputs[0]); inputs.len()];
    register(group, case, variant, |bench| {
        bench.iter(|| {
            for (output, &input) in outputs.iter_mut().zip(black_box(inputs)) {
                *output = op(input);
            }
            black_box(&outputs);
        })
    });
}

/// Benchmarks `op` over each pair of `first` and `second`, writing every result to an output
/// buffer.
pub fn bench_zip<A: Copy, B: Copy, R: Copy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    variant: Variant,
    first: &[A],
    second: &[B],
    op: impl Fn(A, B) -> R,
) {
    let mut outputs = vec![op(first[0], second[0]); first.len()];
    register(group, case, variant, |bench| {
        bench.iter(|| {
            for ((output, &a), &b) in outputs
                .iter_mut()
                .zip(black_box(first))
                .zip(black_box(second))
            {
                *output = op(a, b);
            }
            black_box(&outputs);
        })
    });
}

pub fn register(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    variant: Variant,
    routine: impl FnMut(&mut Bencher<'_, WallTime>),
) {
    match variant {
        Variant::Only => group.bench_function(case, routine),
        Variant::Scalar => group.bench_function(BenchmarkId::new(case, "scalar"), routine),
        Variant::Simd => group.bench_function(BenchmarkId::new(case, "simd"), routine),
    };
}

/// Returns `BATCH` scale factors in `[0.5, 1.5)`, so they are never zero.
pub fn scalars() -> Vec<f32> {
    (0..BATCH).map(|i| 0.5 + unit(i)).collect()
}

/// Pairs each of `values` with an interpolation parameter in `[0, 1)`.
pub fn with_fractions<T: Copy>(values: &[T]) -> Vec<(T, f32)> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| (value, unit(i)))
        .collect()
}

/// Returns `BATCH` colours with every channel, alpha included, in `[0, 1)`; `stream` selects an
/// independent set.
pub fn colours(stream: usize) -> Vec<Colour> {
    (0..BATCH)
        .map(|i| {
            let base = 4 * (stream * BATCH + i);
            Colour::new(unit(base), unit(base + 1), unit(base + 2), unit(base + 3))
        })
        .collect()
}

/// Packs consecutive groups of eight colours into lanes.
pub fn to_simd(colours: &[Colour]) -> Vec<ColourSimd> {
    colours
        .chunks_exact(8)
        .map(|group| ColourSimd {
            r: f32x8::from(array::from_fn(|lane| group[lane].r)),
            g: f32x8::from(array::from_fn(|lane| group[lane].g)),
            b: f32x8::from(array::from_fn(|lane| group[lane].b)),
            a: f32x8::from(array::from_fn(|lane| group[lane].a)),
        })
        .collect()
}

/// Packs consecutive groups of eight values into lanes.
pub fn to_lanes(values: &[f32]) -> Vec<f32x8> {
    values
        .chunks_exact(8)
        .map(|group| f32x8::from(array::from_fn(|lane| group[lane])))
        .collect()
}
