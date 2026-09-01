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

    fn simd_extract_all(value: &Self::Simd) -> [Self; 8] {
        let r = value.colour[0].to_array();
        let g = value.colour[1].to_array();
        let b = value.colour[2].to_array();

        [
            Self {
                colour: Vec3::new(r[0], g[0], b[0]),
            },
            Self {
                colour: Vec3::new(r[1], g[1], b[1]),
            },
            Self {
                colour: Vec3::new(r[2], g[2], b[2]),
            },
            Self {
                colour: Vec3::new(r[3], g[3], b[3]),
            },
            Self {
                colour: Vec3::new(r[4], g[4], b[4]),
            },
            Self {
                colour: Vec3::new(r[5], g[5], b[5]),
            },
            Self {
                colour: Vec3::new(r[6], g[6], b[6]),
            },
            Self {
                colour: Vec3::new(r[7], g[7], b[7]),
            },
        ]
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
// Shaders
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, Default)]
struct TestUniforms;

struct TestShader;

impl FragmentShader<TestVaryings> for TestShader {
    type Uniforms = TestUniforms;

    #[inline(always)]
    fn shade(&self, varyings: TestVaryings, _uniforms: &Self::Uniforms) -> Colour {
        let mut r = varyings.colour.x;
        let mut g = varyings.colour.y;
        let mut b = varyings.colour.z;

        for _ in 0..16 {
            r = (r * 1.37 + g * 0.21).max(0.0);
            g = (g * 0.91 + b * 0.34).max(0.0);
            b = (b * 1.13 + r * 0.17).max(0.0);
        }

        Colour::new(r, g, b, 1.0)
    }
}

impl FragmentShaderSimd<TestVaryings> for TestShader {
    type Uniforms = TestUniforms;

    #[inline(always)]
    fn shade_simd(&self, varyings: TestVaryingsSimd, _uniforms: &Self::Uniforms) -> ColourSimd {
        let mut r = varyings.colour[0];
        let mut g = varyings.colour[1];
        let mut b = varyings.colour[2];

        let c137 = f32x8::splat(1.37);
        let c021 = f32x8::splat(0.21);
        let c091 = f32x8::splat(0.91);
        let c034 = f32x8::splat(0.34);
        let c113 = f32x8::splat(1.13);
        let c017 = f32x8::splat(0.17);
        let zero = f32x8::splat(0.0);

        for _ in 0..16 {
            r = (r * c137 + g * c021).fast_max(zero);
            g = (g * c091 + b * c034).fast_max(zero);
            b = (b * c113 + r * c017).fast_max(zero);
        }

        ColourSimd {
            r,
            g,
            b,
            a: f32x8::splat(1.0),
        }
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

fn bench_callback_depthwrite(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/callback_depthwrite");

    let bounds = full_bounds();

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    depthbuffer.clear(1.0);

                    triangle.rasterise_segment(bounds, |mut fragment| {
                        fragment.position.x -= bounds.min_x as f32;
                        fragment.position.y -= bounds.min_y as f32;

                        if fragment.depth >= depthbuffer.get(fragment.position) {
                            return;
                        }

                        let src = shade(&fragment.varyings);

                        framebuffer.set_pixel(fragment.position, src);

                        depthbuffer.set_depth(fragment.position, fragment.depth);
                    });

                    black_box(framebuffer.pixels());
                    black_box(&depthbuffer);
                });
            },
        );
    }

    group.finish();
}

fn bench_callback_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/callback_full");

    let bounds = full_bounds();

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    let shader = TestShader;
    let uniforms = TestUniforms;

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    depthbuffer.clear(1.0);

                    triangle.rasterise_segment(bounds, |mut fragment| {
                        fragment.position.x -= bounds.min_x as f32;
                        fragment.position.y -= bounds.min_y as f32;

                        if fragment.depth >= depthbuffer.get(fragment.position) {
                            return;
                        }

                        let src = shader.shade(fragment.varyings, &uniforms);

                        framebuffer.set_pixel(fragment.position, src);

                        depthbuffer.set_depth(fragment.position, fragment.depth);
                    });

                    black_box(framebuffer.pixels());
                    black_box(&depthbuffer);
                });
            },
        );
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

                    triangle.rasterise_segment_simd(black_box(bounds), |fragment_simd| {
                        checksum.set(
                            checksum.get()
                                + fragment_simd.depth.reduce_add()
                                + fragment_simd.varyings.colour[0].reduce_add()
                                + fragment_simd.varyings.colour[1].reduce_add()
                                + fragment_simd.varyings.colour[2].reduce_add(),
                        );
                    });

                    black_box(checksum);
                });
            },
        );
    }

    group.finish();
}

fn bench_rasteriser_simd_callback(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/simd_callback");

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
                    triangle.rasterise_segment_simd(black_box(bounds), |fragment_simd| {
                        let index = depthbuffer.index_unchecked(
                            (fragment_simd.x_start - bounds.min_x) as usize,
                            (fragment_simd.y - bounds.min_y) as usize,
                        );

                        let stored = unsafe { depthbuffer.get8_unchecked(index) };

                        let pass = fragment_simd.depth.simd_lt(stored);

                        let mask = pass.to_bitmask();

                        if mask == 0 {
                            return;
                        }

                        let base = framebuffer.index_unchecked(
                            (fragment_simd.x_start - bounds.min_x) as usize,
                            (fragment_simd.y - bounds.min_y) as usize,
                        );

                        let r = fragment_simd.varyings.colour[0].to_array();
                        let g = fragment_simd.varyings.colour[1].to_array();
                        let b = fragment_simd.varyings.colour[2].to_array();

                        for lane in 0..8 {
                            if mask & (1 << lane) == 0 {
                                continue;
                            }

                            let src = Colour::new(r[lane], g[lane], b[lane], 1.0);

                            unsafe {
                                framebuffer.set_pixel_index_unchecked(base + lane, src);
                            }
                        }
                    });

                    black_box(framebuffer.pixels());
                });
            },
        );
    }

    group.finish();
}

fn bench_rasteriser_simd_depthwrite(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/simd_depthwrite");

    let bounds = full_bounds();

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    depthbuffer.clear(1.0);

                    triangle.rasterise_segment_simd(black_box(bounds), |fragment_simd| {
                        let index = depthbuffer.index_unchecked(
                            (fragment_simd.x_start - bounds.min_x) as usize,
                            (fragment_simd.y - bounds.min_y) as usize,
                        );

                        let stored = unsafe { depthbuffer.get8_unchecked(index) };

                        let pass = fragment_simd.depth.simd_lt(stored);

                        if !pass.any() {
                            return;
                        }

                        let mask = pass.to_bitmask();

                        let base = framebuffer.index_unchecked(
                            (fragment_simd.x_start - bounds.min_x) as usize,
                            (fragment_simd.y - bounds.min_y) as usize,
                        );

                        let r = fragment_simd.varyings.colour[0].to_array();
                        let g = fragment_simd.varyings.colour[1].to_array();
                        let b = fragment_simd.varyings.colour[2].to_array();

                        for lane in 0..8 {
                            if mask & (1 << lane) == 0 {
                                continue;
                            }

                            let colour = Colour::new(r[lane], g[lane], b[lane], 1.0);

                            unsafe {
                                framebuffer.set_pixel_index_unchecked(base + lane, colour);
                            }
                        }

                        unsafe {
                            depthbuffer.set8_unchecked_with_mask(index, fragment_simd.depth, pass);
                        }
                    });

                    black_box(framebuffer.pixels());
                    black_box(&depthbuffer);
                });
            },
        );
    }

    group.finish();
}

fn bench_rasteriser_simd_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/simd_full");

    let bounds = full_bounds();

    let mut framebuffer = FrameBuffer::new(640, 360);
    let mut depthbuffer = DepthBuffer::new(640, 360);

    let shader = TestShader;
    let uniforms = TestUniforms;

    for (name, triangle) in size_cases() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &triangle,
            |b, triangle| {
                b.iter(|| {
                    depthbuffer.clear(1.0);

                    triangle.rasterise_segment_simd(black_box(bounds), |fragment_simd| {
                        let index = depthbuffer.index_unchecked(
                            (fragment_simd.x_start - bounds.min_x) as usize,
                            (fragment_simd.y - bounds.min_y) as usize,
                        );

                        let stored = unsafe { depthbuffer.get8_unchecked(index) };

                        let depth_mask = fragment_simd.depth.simd_lt(stored);

                        // Only lanes that are both covered and pass depth.
                        let mask = fragment_simd.mask & depth_mask;

                        if !mask.any() {
                            return;
                        }

                        let base = framebuffer.index_unchecked(
                            (fragment_simd.x_start - bounds.min_x) as usize,
                            (fragment_simd.y - bounds.min_y) as usize,
                        );

                        let colour = shader.shade_simd(fragment_simd.varyings, &uniforms);

                        let r = colour.r.to_array();
                        let g = colour.g.to_array();
                        let b = colour.b.to_array();

                        let mask_bits = mask.to_bitmask();

                        for lane in 0..8 {
                            if mask_bits & (1 << lane) == 0 {
                                continue;
                            }

                            let frag_colour = Colour::new(r[lane], g[lane], b[lane], 1.0);

                            unsafe {
                                framebuffer.set_pixel_index_unchecked(base + lane, frag_colour);
                            }
                        }

                        unsafe {
                            depthbuffer.set8_unchecked_with_mask(index, fragment_simd.depth, mask);
                        }
                    });

                    black_box(framebuffer.pixels());
                    black_box(&depthbuffer);
                });
            },
        );
    }

    group.finish();
}

fn bench_shader(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterise_segment/shader");

    let shader = TestShader;
    let uniforms = TestUniforms;

    let mut input = 0.1234567f32;

    group.bench_function("scalar", |b| {
        b.iter(|| {
            input = input.mul_add(1.000001, 0.12345);

            let varyings = TestVaryings {
                colour: Vec3::new(input, input * 0.73, input * 1.17),
            };

            let colour = shader.shade(varyings, &uniforms);

            black_box(colour.r + colour.g + colour.b + colour.a);
        });
    });

    group.bench_function("simd", |b| {
        b.iter(|| {
            input = input.mul_add(1.000001, 0.12345);

            let value = f32x8::splat(input);

            let varyings = TestVaryingsSimd {
                colour: [
                    value,
                    value * f32x8::splat(0.73),
                    value * f32x8::splat(1.17),
                ],
            };

            let colour = shader.shade_simd(varyings, &uniforms);

            black_box(
                colour.r.reduce_add()
                    + colour.g.reduce_add()
                    + colour.b.reduce_add()
                    + colour.a.reduce_add(),
            );
        });
    });

    group.finish();
}

fn bench_framebuffer_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("framebuffer/set");

    let mut framebuffer = FrameBuffer::new(640, 360);

    let colours = [
        Colour::new(0.1, 0.2, 0.3, 1.0),
        Colour::new(0.2, 0.3, 0.4, 1.0),
        Colour::new(0.3, 0.4, 0.5, 1.0),
        Colour::new(0.4, 0.5, 0.6, 1.0),
        Colour::new(0.5, 0.6, 0.7, 1.0),
        Colour::new(0.6, 0.7, 0.8, 1.0),
        Colour::new(0.7, 0.8, 0.9, 1.0),
        Colour::new(0.8, 0.9, 1.0, 1.0),
    ];

    group.throughput(criterion::Throughput::Elements(8));

    group.bench_function("8_scalar_indexed", |b| {
        b.iter(|| {
            let index = 100_000;

            for lane in 0..8 {
                unsafe {
                    framebuffer.set_pixel_index_unchecked(index + lane, colours[lane]);
                }
            }

            black_box(framebuffer.pixels());
        });
    });

    group.bench_function("8_scalar_checked", |b| {
        b.iter(|| {
            let x = 100;
            let y = 150;

            for lane in 0..8 {
                framebuffer.set_pixel(Vec2::new((x + lane) as f32, y as f32), colours[lane]);
            }

            black_box(framebuffer.pixels());
        });
    });

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
    bench_callback_depthwrite,
    bench_callback_shape,
    bench_callback_bounds,
    bench_callback_full,
    bench_throughput,
    bench_rasteriser_simd_compute,
    bench_rasteriser_simd_callback,
    bench_rasteriser_simd_depthwrite,
    bench_rasteriser_simd_full,
    bench_shader,
    bench_framebuffer_set
);

criterion_main!(benches);
