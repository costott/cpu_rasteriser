use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cpu_rasteriser::prelude::*;

// -----------------------------------------------------------------------------
// Test types
// -----------------------------------------------------------------------------

#[derive(Debug, Interpolate)]
struct TestVaryings {
    colour: Vec3,
}

fn vertex(x: f32, y: f32, colour: Vec3) -> RasterVertex<TestVaryings> {
    RasterVertex {
        position: Vec2::new(x, y),
        depth: 0.5,
        inv_w: 1.0,
        varyings: TestVaryings { colour },
    }
}

fn triangle(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> Triangle2D<TestVaryings> {
    Triangle2D::new(
        vertex(a.0, a.1, Vec3::new(1.0, 0.0, 0.0)),
        vertex(b.0, b.1, Vec3::new(0.0, 1.0, 0.0)),
        vertex(c.0, c.1, Vec3::new(0.0, 0.0, 1.0)),
    )
}

// -----------------------------------------------------------------------------
// Bounds
// -----------------------------------------------------------------------------

fn bounds(width: i32, height: i32) -> Rect {
    Rect {
        min_x: 0,
        min_y: 0,
        max_x: width,
        max_y: height,
    }
}

// -----------------------------------------------------------------------------
// Triangles
// -----------------------------------------------------------------------------

/// Large triangle covering a significant portion of the render target.
///
/// This is the most important benchmark for SIMD because the inner x-loop
/// processes a large number of fragments.
fn large_triangle() -> Triangle2D<TestVaryings> {
    triangle((64.0, 32.0), (576.0, 96.0), (320.0, 328.0))
}

/// Roughly half-screen triangle.
fn medium_triangle() -> Triangle2D<TestVaryings> {
    triangle((100.0, 80.0), (500.0, 120.0), (180.0, 300.0))
}

/// Small triangle. This is useful for seeing the overhead of SIMD setup when
/// there aren't many pixels to process.
fn small_triangle() -> Triangle2D<TestVaryings> {
    triangle((300.0, 170.0), (340.0, 180.0), (320.0, 210.0))
}

/// Very wide and shallow triangle.
fn wide_triangle() -> Triangle2D<TestVaryings> {
    triangle((32.0, 150.0), (608.0, 155.0), (320.0, 210.0))
}

/// Tall and narrow triangle.
fn tall_triangle() -> Triangle2D<TestVaryings> {
    triangle((300.0, 16.0), (340.0, 180.0), (320.0, 344.0))
}

/// Triangle that extends beyond the render target. This exercises the
/// x/y bounds clamping paths.
fn clipped_triangle() -> Triangle2D<TestVaryings> {
    triangle((-100.0, -50.0), (740.0, 80.0), (320.0, 500.0))
}

/// A triangle with non-trivial depth and perspective values.
///
/// This makes sure the benchmark isn't accidentally representative only of
/// the particularly cheap inv_w = 1.0 case.
fn perspective_triangle() -> Triangle2D<TestVaryings> {
    Triangle2D::new(
        RasterVertex {
            position: Vec2::new(64.0, 32.0),
            depth: 0.1,
            inv_w: 0.25,
            varyings: TestVaryings {
                colour: Vec3::new(1.0, 0.0, 0.0),
            },
        },
        RasterVertex {
            position: Vec2::new(576.0, 96.0),
            depth: 0.8,
            inv_w: 1.5,
            varyings: TestVaryings {
                colour: Vec3::new(0.0, 1.0, 0.0),
            },
        },
        RasterVertex {
            position: Vec2::new(320.0, 328.0),
            depth: 0.4,
            inv_w: 0.5,
            varyings: TestVaryings {
                colour: Vec3::new(0.0, 0.0, 1.0),
            },
        },
    )
}

// -----------------------------------------------------------------------------
// Benchmark helper
// -----------------------------------------------------------------------------

#[inline(never)]
fn benchmark_triangle(triangle: &Triangle2D<TestVaryings>, bounds: Rect) -> usize {
    let mut fragment_count = 0usize;

    triangle.rasterise_segment(black_box(bounds), |fragment| {
        // Consume the fragment so the compiler can't eliminate the
        // rasterisation work.
        black_box(fragment);
        fragment_count += 1;
    });

    black_box(fragment_count)
}

// -----------------------------------------------------------------------------
// Individual benchmark groups
// -----------------------------------------------------------------------------

fn bench_triangle_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/size");

    let bounds = bounds(640, 360);

    let triangles = [
        ("small", small_triangle()),
        ("medium", medium_triangle()),
        ("large", large_triangle()),
    ];

    for (name, triangle) in triangles {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| benchmark_triangle(triangle, bounds));
            },
        );
    }

    group.finish();
}

fn bench_triangle_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/shape");

    let bounds = bounds(640, 360);

    let triangles = [
        ("large", large_triangle()),
        ("wide", wide_triangle()),
        ("tall", tall_triangle()),
        ("clipped", clipped_triangle()),
        ("perspective", perspective_triangle()),
    ];

    for (name, triangle) in triangles {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| benchmark_triangle(triangle, bounds));
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Bounds benchmarks
// -----------------------------------------------------------------------------

fn bench_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/bounds");

    let triangle = large_triangle();

    let bounds_cases = [
        ("full", bounds(640, 360)),
        (
            "half",
            Rect {
                min_x: 0,
                min_y: 0,
                max_x: 320,
                max_y: 180,
            },
        ),
        (
            "small",
            Rect {
                min_x: 280,
                min_y: 140,
                max_x: 360,
                max_y: 220,
            },
        ),
    ];

    for (name, bounds) in bounds_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &bounds, |b, bounds| {
            b.iter(|| benchmark_triangle(&triangle, *bounds));
        });
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Fragment throughput benchmark
// -----------------------------------------------------------------------------

fn bench_fragment_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/throughput");

    let bounds = bounds(640, 360);
    let triangle = large_triangle();

    let fragment_count = benchmark_triangle(&triangle, bounds);

    group.throughput(criterion::Throughput::Elements(fragment_count as u64));

    group.bench_function("large_triangle", |b| {
        b.iter(|| benchmark_triangle(&triangle, bounds));
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Criterion
// -----------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_triangle_sizes,
    bench_triangle_shapes,
    bench_bounds,
    bench_fragment_throughput
);

criterion_main!(benches);
