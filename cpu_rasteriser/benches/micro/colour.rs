use cpu_rasteriser::prelude::*;
use criterion::{Criterion, Throughput, criterion_group};

use crate::fixtures::{
    BATCH, Variant, bench_map, bench_zip, colours, scalars, to_lanes, to_simd, unit, with_fractions,
};

/// Every computing operation of `Colour` and `ColourSimd` has a case, so a code-generation change
/// to any of them shows. `lerp` is the only one with both forms: one scalar pass over `BATCH`
/// colours against `BATCH / 8` `ColourSimd` operations over the same colours and parameters.
/// `splat` has no scalar counterpart and processes `BATCH / 8` colours. `clamp` and `clamp_rgb`
/// take colours that need clamping, and the others take colours in `[0, 1)`. Each operator and
/// each assign form is its own code, so each has a case. `new`, `rgb`, the constants and the
/// `Vec3` and `Vec4` conversions only place fields.
fn benchmark_colour(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour");
    group.throughput(Throughput::Elements(BATCH as u64));

    let a = colours(0);
    let b = colours(1);
    let unclamped = unclamped_colours();
    let packed: Vec<u32> = a.iter().map(|colour| colour.to_u32()).collect();
    let factors = scalars();

    bench_map(
        &mut group,
        "from_u32",
        Variant::Only,
        &packed,
        Colour::from_u32,
    );
    bench_map(&mut group, "to_u32", Variant::Only, &a, Colour::to_u32);
    bench_zip(
        &mut group,
        "scale_all",
        Variant::Only,
        &a,
        &factors,
        Colour::scale_all,
    );
    bench_map(&mut group, "clamp", Variant::Only, &unclamped, |colour| {
        colour.clamp(0.0, 1.0)
    });
    bench_map(
        &mut group,
        "clamp_rgb",
        Variant::Only,
        &unclamped,
        |colour| colour.clamp_rgb(0.0, 1.0),
    );
    bench_map(
        &mut group,
        "luminance",
        Variant::Only,
        &a,
        Colour::luminance,
    );

    let lerps: Vec<_> = a.iter().copied().zip(with_fractions(&b)).collect();
    bench_map(
        &mut group,
        "lerp",
        Variant::Scalar,
        &lerps,
        |(from, (to, t))| from.lerp(to, t),
    );
    let fractions: Vec<f32> = (0..BATCH).map(unit).collect();
    let lerps_simd: Vec<_> = to_simd(&a)
        .into_iter()
        .zip(to_simd(&b).into_iter().zip(to_lanes(&fractions)))
        .collect();
    bench_map(
        &mut group,
        "lerp",
        Variant::Simd,
        &lerps_simd,
        |(from, (to, t))| from.lerp(to, t),
    );

    bench_zip(&mut group, "add", Variant::Only, &a, &b, |a, b| a + b);
    bench_zip(
        &mut group,
        "add_assign",
        Variant::Only,
        &a,
        &b,
        |mut a, b| {
            a += b;
            a
        },
    );
    bench_zip(&mut group, "sub", Variant::Only, &a, &b, |a, b| a - b);
    bench_zip(&mut group, "mul", Variant::Only, &a, &b, |a, b| a * b);
    bench_zip(
        &mut group,
        "mul_scalar",
        Variant::Only,
        &a,
        &factors,
        |a, s| a * s,
    );
    bench_zip(
        &mut group,
        "mul_assign",
        Variant::Only,
        &a,
        &factors,
        |mut a, s| {
            a *= s;
            a
        },
    );
    bench_zip(
        &mut group,
        "div_scalar",
        Variant::Only,
        &a,
        &factors,
        |a, s| a / s,
    );
    bench_zip(
        &mut group,
        "div_assign",
        Variant::Only,
        &a,
        &factors,
        |mut a, s| {
            a /= s;
            a
        },
    );

    group.throughput(Throughput::Elements(BATCH as u64 / 8));
    bench_map(
        &mut group,
        "splat",
        Variant::Only,
        &a[..BATCH / 8],
        ColourSimd::splat,
    );

    group.finish();
}

/// Returns `BATCH` colours with every channel in `[-0.5, 1.5)`, so about half of the channels
/// need clamping to `[0, 1]`.
fn unclamped_colours() -> Vec<Colour> {
    (0..BATCH)
        .map(|i| {
            let channel = |c: usize| unit(4 * i + c) * 2.0 - 0.5;
            Colour::new(channel(0), channel(1), channel(2), channel(3))
        })
        .collect()
}

criterion_group!(benches, benchmark_colour);
