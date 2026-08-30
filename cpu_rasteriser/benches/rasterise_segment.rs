use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cpu_rasteriser::{depthbuffer::DepthBuffer, framebuffer::FrameBuffer, prelude::*};
use wide::f32x8;

// -----------------------------------------------------------------------------
// Test varying / geometry
// -----------------------------------------------------------------------------

#[derive(Debug, Interpolate)]
struct TestVaryings {
    colour: Vec3,
}

impl SimdInterpolate for TestVaryings {
    type Simd = TestVaryingsSimd;

    fn simd_step(value: &Self, step: &Self, lanes: f32x8) -> Self::Simd {
        TestVaryingsSimd {
            colour: [
                f32x8::splat(value.colour.x) + lanes * f32x8::splat(step.colour.x),
                f32x8::splat(value.colour.y) + lanes * f32x8::splat(step.colour.y),
                f32x8::splat(value.colour.z) + lanes * f32x8::splat(step.colour.z),
            ],
        }
    }

    fn simd_add_scaled(value: &Self::Simd, step: &Self, scale: f32x8) -> Self::Simd {
        TestVaryingsSimd {
            colour: [
                value.colour[0] + f32x8::splat(step.colour.x) * scale,
                value.colour[1] + f32x8::splat(step.colour.y) * scale,
                value.colour[2] + f32x8::splat(step.colour.z) * scale,
            ],
        }
    }

    fn simd_perspective(value: Self::Simd, perspective: f32x8) -> Self::Simd {
        TestVaryingsSimd {
            colour: [
                value.colour[0] * perspective,
                value.colour[1] * perspective,
                value.colour[2] * perspective,
            ],
        }
    }

    fn simd_extract(value: &Self::Simd, lane: usize) -> Self {
        let r = value.colour[0].to_array();
        let g = value.colour[1].to_array();
        let b = value.colour[2].to_array();

        Self {
            colour: Vec3::new(r[lane], g[lane], b[lane]),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TestVaryingsSimd {
    pub colour: [f32x8; 3],
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

fn small_triangle() -> Triangle2D<TestVaryings> {
    triangle((300.0, 170.0), (340.0, 180.0), (320.0, 210.0))
}

fn medium_triangle() -> Triangle2D<TestVaryings> {
    triangle((100.0, 80.0), (500.0, 120.0), (180.0, 300.0))
}

fn large_triangle() -> Triangle2D<TestVaryings> {
    triangle((64.0, 32.0), (576.0, 96.0), (320.0, 328.0))
}

fn wide_triangle() -> Triangle2D<TestVaryings> {
    triangle((32.0, 150.0), (608.0, 155.0), (320.0, 210.0))
}

fn tall_triangle() -> Triangle2D<TestVaryings> {
    triangle((300.0, 16.0), (340.0, 180.0), (320.0, 344.0))
}

fn clipped_triangle() -> Triangle2D<TestVaryings> {
    triangle((-100.0, -50.0), (740.0, 80.0), (320.0, 500.0))
}

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
// Bounds
// -----------------------------------------------------------------------------

fn full_bounds() -> Rect {
    Rect {
        min_x: 0,
        min_y: 0,
        max_x: 640,
        max_y: 360,
    }
}

fn half_bounds() -> Rect {
    Rect {
        min_x: 0,
        min_y: 0,
        max_x: 320,
        max_y: 180,
    }
}

fn small_bounds() -> Rect {
    Rect {
        min_x: 280,
        min_y: 140,
        max_x: 360,
        max_y: 220,
    }
}

// -----------------------------------------------------------------------------
// Callback path
// -----------------------------------------------------------------------------

#[inline(always)]
fn shade(varyings: &TestVaryings) -> Colour {
    Colour::new(varyings.colour.x, varyings.colour.y, varyings.colour.z, 1.0)
}

fn rasterise_with_callback(
    triangle: &Triangle2D<TestVaryings>,
    framebuffer: &mut FrameBuffer,
    depthbuffer: Option<&mut DepthBuffer>,
    bounds: Rect,
    depth_test: bool,
) {
    let mut depthbuffer = depthbuffer;

    triangle.rasterise_segment(bounds, |mut fragment| {
        fragment.position.x -= bounds.min_x as f32;
        fragment.position.y -= bounds.min_y as f32;

        if depth_test {
            let depthbuffer = depthbuffer
                .as_deref_mut()
                .expect("depth testing enabled but no depth buffer");

            if fragment.depth >= depthbuffer.get(fragment.position) {
                return;
            }
        }

        let src = shade(&fragment.varyings);

        framebuffer.set_pixel(fragment.position, src);
    });
}

// -----------------------------------------------------------------------------
// Shared benchmark case lists
// -----------------------------------------------------------------------------

fn size_cases() -> [(&'static str, Triangle2D<TestVaryings>); 3] {
    [
        ("small", small_triangle()),
        ("medium", medium_triangle()),
        ("large", large_triangle()),
    ]
}

fn shape_cases() -> [(&'static str, Triangle2D<TestVaryings>); 5] {
    [
        ("large", large_triangle()),
        ("wide", wide_triangle()),
        ("tall", tall_triangle()),
        ("clipped", clipped_triangle()),
        ("perspective", perspective_triangle()),
    ]
}

// -----------------------------------------------------------------------------
// Rasteriser: overhead
//
// The fragment itself is deliberately not consumed. This measures the
// scan-conversion / control-flow side with the produced fragment allowed to
// be discarded.
//
// -----------------------------------------------------------------------------

fn bench_rasteriser_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/overhead");

    let bounds = full_bounds();

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    let mut fragment_count = 0usize;

                    triangle.rasterise_segment(black_box(bounds), |_fragment| {
                        fragment_count += 1;
                    });

                    black_box(fragment_count);
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Rasteriser: compute
//
// Consume the actual interpolated values, but don't black_box the entire
// Fragment. This keeps the interpolation/perspective work observable without
// introducing a black_box barrier for every individual fragment.
// -----------------------------------------------------------------------------

fn bench_rasteriser_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/compute");

    let bounds = full_bounds();

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    let mut checksum = 0.0f32;
                    let mut fragment_count = 0usize;

                    triangle.rasterise_segment(black_box(bounds), |fragment| {
                        checksum += fragment.depth;
                        checksum += fragment.varyings.colour.x;
                        checksum += fragment.varyings.colour.y;
                        checksum += fragment.varyings.colour.z;
                        fragment_count += 1;
                    });

                    black_box(checksum);
                    black_box(fragment_count);
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Rasteriser: compute / shape
// -----------------------------------------------------------------------------

fn bench_rasteriser_compute_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/compute_shape");

    let bounds = full_bounds();

    for (name, triangle) in shape_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    let mut checksum = 0.0f32;
                    let mut fragment_count = 0usize;

                    triangle.rasterise_segment(black_box(bounds), |fragment| {
                        checksum += fragment.depth;
                        checksum += fragment.varyings.colour.x;
                        checksum += fragment.varyings.colour.y;
                        checksum += fragment.varyings.colour.z;
                        fragment_count += 1;
                    });

                    black_box(checksum);
                    black_box(fragment_count);
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Rasteriser: overhead / shape
// -----------------------------------------------------------------------------

fn bench_rasteriser_overhead_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/overhead_shape");

    let bounds = full_bounds();

    for (name, triangle) in shape_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    let mut fragment_count = 0usize;

                    triangle.rasterise_segment(black_box(bounds), |_fragment| {
                        fragment_count += 1;
                    });

                    black_box(fragment_count);
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Rasteriser: overhead / bounds
// -----------------------------------------------------------------------------

fn bench_rasteriser_overhead_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/overhead_bounds");

    let triangle = large_triangle();

    let cases = [
        ("full", full_bounds()),
        ("half", half_bounds()),
        ("small", small_bounds()),
    ];

    for (name, bounds) in cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    let mut fragment_count = 0usize;

                    triangle.rasterise_segment(black_box(bounds), |_fragment| {
                        fragment_count += 1;
                    });

                    black_box(fragment_count);
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Rasteriser: compute / bounds
// -----------------------------------------------------------------------------

fn bench_rasteriser_compute_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/compute_bounds");

    let triangle = large_triangle();

    let cases = [
        ("full", full_bounds()),
        ("half", half_bounds()),
        ("small", small_bounds()),
    ];

    for (name, bounds) in cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    let mut checksum = 0.0f32;
                    let mut fragment_count = 0usize;

                    triangle.rasterise_segment(black_box(bounds), |fragment| {
                        checksum += fragment.depth;
                        checksum += fragment.varyings.colour.x;
                        checksum += fragment.varyings.colour.y;
                        checksum += fragment.varyings.colour.z;
                        fragment_count += 1;
                    });

                    black_box(checksum);
                    black_box(fragment_count);
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Callback: size
//
// Depth test is enabled, but depth writes are disabled. The depth buffer stays
// at 1.0 and fragment depth is 0.5, so every fragment follows the same path.
// -----------------------------------------------------------------------------

fn bench_callback_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/callback/size");

    let bounds = full_bounds();

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    depthbuffer.clear(1.0);

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    rasterise_with_callback(
                        triangle,
                        &mut framebuffer,
                        Some(&mut depthbuffer),
                        bounds,
                        true,
                    );

                    black_box(framebuffer.pixels());
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Callback: shape
// -----------------------------------------------------------------------------

fn bench_callback_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/callback/shape");

    let bounds = full_bounds();

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    depthbuffer.clear(1.0);

    for (name, triangle) in shape_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    rasterise_with_callback(
                        triangle,
                        &mut framebuffer,
                        Some(&mut depthbuffer),
                        bounds,
                        true,
                    );

                    black_box(framebuffer.pixels());
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Callback: bounds
// -----------------------------------------------------------------------------

fn bench_callback_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/callback/bounds");

    let triangle = large_triangle();

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    depthbuffer.clear(1.0);

    let cases = [
        ("full", full_bounds()),
        ("half", half_bounds()),
        ("small", small_bounds()),
    ];

    for (name, bounds) in cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &bounds, |b, bounds| {
            b.iter(|| {
                rasterise_with_callback(
                    &triangle,
                    &mut framebuffer,
                    Some(&mut depthbuffer),
                    *bounds,
                    true,
                );

                black_box(framebuffer.pixels());
            });
        });
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Throughput
// -----------------------------------------------------------------------------

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/throughput");

    let triangle = large_triangle();
    let bounds = full_bounds();

    let mut fragment_count = 0usize;

    triangle.rasterise_segment(bounds, |_| {
        fragment_count += 1;
    });

    group.throughput(criterion::Throughput::Elements(fragment_count as u64));

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    depthbuffer.clear(1.0);

    group.bench_function("overhead", |b| {
        b.iter(|| {
            let mut fragment_count = 0usize;

            triangle.rasterise_segment(black_box(bounds), |_fragment| {
                fragment_count += 1;
            });

            black_box(fragment_count);
        });
    });

    group.bench_function("compute", |b| {
        b.iter(|| {
            let mut checksum = 0.0f32;
            let mut fragment_count = 0usize;

            triangle.rasterise_segment(black_box(bounds), |fragment| {
                checksum += fragment.depth;
                checksum += fragment.varyings.colour.x;
                checksum += fragment.varyings.colour.y;
                checksum += fragment.varyings.colour.z;
                fragment_count += 1;
            });

            black_box(checksum);
            black_box(fragment_count);
        });
    });

    group.bench_function("callback", |b| {
        b.iter(|| {
            rasterise_with_callback(
                &triangle,
                &mut framebuffer,
                Some(&mut depthbuffer),
                bounds,
                true,
            );

            black_box(framebuffer.pixels());
        });
    });

    group.finish();
}

fn bench_rasteriser_simd_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/simd_compute");

    let bounds = full_bounds();

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    let checksum = std::cell::Cell::new(0.0f32);

                    triangle.rasterise_segment_simd(
                        black_box(bounds),
                        |_, _, depth, inv_w, varyings| {
                            let perspective = inv_w.recip();
                            let varyings = TestVaryings::simd_perspective(varyings, perspective);

                            checksum.set(
                                checksum.get()
                                    + depth.reduce_add()
                                    + varyings.colour[0].reduce_add()
                                    + varyings.colour[1].reduce_add()
                                    + varyings.colour[2].reduce_add(),
                            );
                        },
                        |fragment| {
                            checksum.set(
                                checksum.get()
                                    + fragment.depth
                                    + fragment.varyings.colour.x
                                    + fragment.varyings.colour.y
                                    + fragment.varyings.colour.z,
                            );
                        },
                    );

                    black_box(checksum);
                });
            },
        );
    }

    group.finish();
}

// -----------------------------------------------------------------------------
// Criterion
// -----------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_rasteriser_overhead,
    bench_rasteriser_compute,
    bench_rasteriser_overhead_shape,
    bench_rasteriser_compute_shape,
    bench_rasteriser_overhead_bounds,
    bench_rasteriser_compute_bounds,
    bench_callback_size,
    bench_callback_shape,
    bench_callback_bounds,
    bench_throughput,
    bench_rasteriser_simd_compute
);

criterion_main!(benches);
