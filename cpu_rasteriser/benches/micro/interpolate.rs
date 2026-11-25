use std::hint::black_box;

use cpu_rasteriser::prelude::*;
use cpu_rasteriser::wide::f32x8;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};

use crate::fixtures::{BATCH, Varyings, VaryingsSimd, to_lanes, unit, varyings};

/// Pixels stepped in one iteration of `step`.
///
/// Provisional: stepping is a dependent chain of about 4 cycles per scalar operation, so a SIMD
/// case of `BATCH / 8` operations would take under a microsecond, where timings are
/// layout-sensitive; eight times as many pixels keeps it above. Issue #35 tunes it from measured
/// variation.
const STEP_PIXELS: usize = 8 * BATCH;

/// Each case is one cost of the rasteriser's inner loops. Per span, both rasterisers set up the
/// scalar interpolation (`span_setup`), and the SIMD one then expands it to eight lanes
/// (`simd_step`, which has no scalar counterpart). Per pixel or group of eight, they advance the
/// varyings (`step`, a dependent chain as in the rasteriser) and apply perspective correction
/// (`perspective`, independent per pixel). `SimdInterpolate::simd_extract_all` is not benchmarked
/// because the rasteriser never calls it: it is for checking and debugging a SIMD shader.
fn benchmark_interpolate(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate");

    let spans = spans();
    let mut totals = vec![0.0; BATCH];
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function("span_setup", |bench| {
        bench.iter(|| {
            for (total, span) in totals.iter_mut().zip(black_box(&spans)) {
                let varyings = span.left.interpolate(&span.right, span.t);
                let step = span.right.difference(&span.left).scale(span.x_step);
                *total = varyings.sum() + step.sum();
            }
            black_box(&totals);
        })
    });

    let lanes = f32x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let steps: Vec<Varyings> = spans
        .iter()
        .map(|span| span.right.difference(&span.left).scale(span.x_step))
        .collect();
    let mut lane_totals = vec![f32x8::ZERO; BATCH];
    group.bench_function("simd_step", |bench| {
        bench.iter(|| {
            for ((total, span), step) in lane_totals
                .iter_mut()
                .zip(black_box(&spans))
                .zip(black_box(&steps))
            {
                *total = Varyings::simd_step(&span.left, step, lanes).sum();
            }
            black_box(&lane_totals);
        })
    });

    let start = varyings(0);
    let step = varyings(1).scale(1e-4);
    let start_simd = Varyings::simd_step(&start, &step, lanes);
    group.throughput(Throughput::Elements(STEP_PIXELS as u64));
    group.bench_function(BenchmarkId::new("step", "scalar"), |bench| {
        bench.iter(|| {
            let step = black_box(&step);
            let mut varyings = black_box(&start).clone();
            for _ in 0..STEP_PIXELS {
                varyings = varyings.add_scaled(step, 1.0);
            }
            black_box(varyings.sum())
        })
    });
    group.bench_function(BenchmarkId::new("step", "simd"), |bench| {
        bench.iter(|| {
            let step = black_box(&step);
            let eight = f32x8::splat(8.0);
            let mut varyings = black_box(start_simd);
            for _ in 0..STEP_PIXELS / 8 {
                varyings = Varyings::simd_add_scaled(&varyings, step, eight);
            }
            black_box(varyings.sum())
        })
    });

    let groups = groups();
    let pixels: Vec<Varyings> = groups.iter().flat_map(Varyings::simd_extract_all).collect();
    let perspective: Vec<f32> = (0..BATCH).map(|i| 1.0 + unit(i)).collect();
    let perspective_simd = to_lanes(&perspective);
    let mut corrected = vec![pixels[0].clone(); BATCH];
    let mut corrected_simd = vec![groups[0]; BATCH / 8];
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function(BenchmarkId::new("perspective", "scalar"), |bench| {
        bench.iter(|| {
            for ((out, pixel), &factor) in corrected
                .iter_mut()
                .zip(black_box(&pixels))
                .zip(black_box(&perspective))
            {
                *out = pixel.scale(factor);
            }
            black_box(&corrected);
        })
    });
    group.bench_function(BenchmarkId::new("perspective", "simd"), |bench| {
        bench.iter(|| {
            for ((out, &pixel), &factor) in corrected_simd
                .iter_mut()
                .zip(black_box(&groups))
                .zip(black_box(&perspective_simd))
            {
                *out = Varyings::simd_perspective(pixel, factor);
            }
            black_box(&corrected_simd);
        })
    });

    group.finish();
}

struct Span {
    left: Varyings,
    right: Varyings,
    t: f32,
    x_step: f32,
}

fn spans() -> Vec<Span> {
    (0..BATCH)
        .map(|i| {
            let width = 16.0 + unit(3 * i) * 584.0;
            Span {
                left: varyings(2 * i),
                right: varyings(2 * i + 1),
                t: unit(3 * i + 1),
                x_step: 1.0 / width,
            }
        })
        .collect()
}

/// Returns `BATCH / 8` groups of eight consecutive pixels' varyings.
fn groups() -> Vec<VaryingsSimd> {
    let lanes = f32x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    (0..BATCH / 8)
        .map(|i| {
            let step = varyings(2 * i + 1).scale(1e-2);
            Varyings::simd_step(&varyings(2 * i), &step, lanes)
        })
        .collect()
}

criterion_group!(benches, benchmark_interpolate);
