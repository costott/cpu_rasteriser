use std::hint::black_box;

use cpu_rasteriser::internals::{Rect, TileBinner, Triangle2D};
use cpu_rasteriser::wide::f32x8;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};

use crate::fixtures::{TARGET, Varyings, triangle};

fn benchmark_rasterise(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise");

    for case in cases() {
        group.throughput(Throughput::Elements(pixel_count(&case)));
        group.bench_function(BenchmarkId::new(case.name, "scalar"), |bench| {
            bench.iter(|| rasterise_scalar(black_box(&case.triangle), black_box(case.bounds)))
        });
        group.bench_function(BenchmarkId::new(case.name, "simd"), |bench| {
            bench.iter(|| rasterise_simd(black_box(&case.triangle), black_box(case.bounds)))
        });
    }

    group.finish();
}

struct Case {
    name: &'static str,
    triangle: Triangle2D<Varyings>,
    bounds: Rect,
}

/// Each case isolates one cost: per-row set-up (`small`, `thin`), per-pixel work (`large`,
/// `wide`), a fixed area at three scales (`small`, `medium`, `large`), and how bounds change
/// where rows start (`clamped`, `window`). `tile_inside` and `tile_edge` use the bounds a render
/// pass rasterises with, one tile: a tile inside the triangle, and one that its left edge crosses.
fn cases() -> [Case; 9] {
    let window = Rect {
        min_x: 200,
        min_y: 100,
        max_x: 456,
        max_y: 228,
    };
    let large = || triangle((64.0, 32.0), (576.0, 96.0), (320.0, 328.0));

    [
        Case {
            name: "small",
            triangle: triangle((300.0, 170.0), (340.0, 180.0), (320.0, 210.0)),
            bounds: TARGET,
        },
        Case {
            name: "medium",
            triangle: triangle((220.0, 120.0), (380.0, 140.0), (260.0, 260.0)),
            bounds: TARGET,
        },
        Case {
            name: "large",
            triangle: large(),
            bounds: TARGET,
        },
        Case {
            name: "wide",
            triangle: triangle((32.0, 150.0), (608.0, 156.0), (320.0, 210.0)),
            bounds: TARGET,
        },
        Case {
            name: "thin",
            triangle: triangle((40.0, 40.0), (46.0, 40.0), (43.0, 300.0)),
            bounds: TARGET,
        },
        Case {
            name: "clamped",
            triangle: triangle((-120.0, -60.0), (760.0, 40.0), (300.0, 520.0)),
            bounds: TARGET,
        },
        Case {
            name: "window",
            triangle: large(),
            bounds: window,
        },
        Case {
            name: "tile_inside",
            triangle: large(),
            bounds: tile(4, 2),
        },
        Case {
            name: "tile_edge",
            triangle: large(),
            bounds: tile(2, 2),
        },
    ]
}

/// Returns the bounds of the tile in column `x` and row `y`.
fn tile(x: i32, y: i32) -> Rect {
    let size = TileBinner::TILE_SIZE;
    Rect {
        min_x: x * size,
        min_y: y * size,
        max_x: (x + 1) * size,
        max_y: (y + 1) * size,
    }
}

fn pixel_count(case: &Case) -> u64 {
    let mut count = 0;
    case.triangle.rasterise_segment(case.bounds, |_| count += 1);
    count
}

fn rasterise_scalar(triangle: &Triangle2D<Varyings>, bounds: Rect) -> f32 {
    let mut total = 0.0;
    triangle.rasterise_segment(bounds, |fragment| {
        total += fragment.depth + fragment.varyings.sum();
    });
    black_box(total)
}

fn rasterise_simd(triangle: &Triangle2D<Varyings>, bounds: Rect) -> f32x8 {
    let mut total = f32x8::ZERO;
    triangle.rasterise_segment_simd(bounds, |fragment| {
        total += (fragment.depth + fragment.varyings.sum()) & fragment.mask;
    });
    black_box(total)
}

criterion_group!(benches, benchmark_rasterise);
