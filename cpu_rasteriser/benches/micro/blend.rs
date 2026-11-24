use std::hint::black_box;

use cpu_rasteriser::prelude::*;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};

use crate::fixtures::{BATCH, colours, to_simd};

/// Each case takes a different code path through blending. `additive` (constant weights) is the
/// floor. `alpha_blend` weights by the source alpha and its complement, and `multiply` by the
/// destination colour itself with a zero weight. `one_minus_src_colour` and `one_minus_dst_colour`
/// weigh by the complement of a colour, each with its own arm of `BlendFactor::resolve`. The
/// remaining cases keep the additive weights and change only the operation, so each isolates one
/// `BlendOp` arm: `subtract`, `reverse_subtract`, `min` and `max` (`Add` is `additive`). The other
/// factors are not repeated because their arms run the same code on another operand: `SrcColor`
/// passes a colour through as `DstColor` does, `DstAlpha` broadcasts an alpha as `SrcAlpha` does,
/// and `OneMinusDstAlpha` broadcasts a complement as `OneMinusSrcAlpha` does.
fn benchmark_blend(c: &mut Criterion) {
    let mut group = c.benchmark_group("blend");
    group.throughput(Throughput::Elements(BATCH as u64));

    let sources = colours(0);
    let destinations = colours(1);
    let sources_simd = to_simd(&sources);
    let destinations_simd = to_simd(&destinations);

    let mut blended = vec![Colour::BLACK; BATCH];
    let mut blended_simd = vec![ColourSimd::splat(Colour::BLACK); BATCH / 8];
    for (name, state) in cases() {
        group.bench_function(BenchmarkId::new(name, "scalar"), |bench| {
            bench.iter(|| {
                let state = black_box(state);
                for ((out, &src), &dst) in blended
                    .iter_mut()
                    .zip(black_box(&sources))
                    .zip(black_box(&destinations))
                {
                    *out = state.apply(src, dst);
                }
                black_box(&blended);
            })
        });
        group.bench_function(BenchmarkId::new(name, "simd"), |bench| {
            bench.iter(|| {
                let state = black_box(state);
                for ((out, &src), &dst) in blended_simd
                    .iter_mut()
                    .zip(black_box(&sources_simd))
                    .zip(black_box(&destinations_simd))
                {
                    *out = state.apply_simd(src, dst);
                }
                black_box(&blended_simd);
            })
        });
    }

    group.finish();
}

fn cases() -> [(&'static str, BlendState); 9] {
    let additive = |op| BlendState {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        op,
    };

    [
        ("additive", BlendState::ADDITIVE),
        ("alpha_blend", BlendState::ALPHA_BLEND),
        (
            "multiply",
            BlendState {
                src_factor: BlendFactor::DstColor,
                dst_factor: BlendFactor::Zero,
                op: BlendOp::Add,
            },
        ),
        (
            "one_minus_src_colour",
            BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcColor,
                op: BlendOp::Add,
            },
        ),
        (
            "one_minus_dst_colour",
            BlendState {
                src_factor: BlendFactor::OneMinusDstColor,
                dst_factor: BlendFactor::One,
                op: BlendOp::Add,
            },
        ),
        ("subtract", additive(BlendOp::Subtract)),
        ("reverse_subtract", additive(BlendOp::ReverseSubtract)),
        ("min", additive(BlendOp::Min)),
        ("max", additive(BlendOp::Max)),
    ]
}

criterion_group!(benches, benchmark_blend);
