use criterion::Criterion;

use crate::fixtures::{Variant, bench_map, bench_zip, scalars, with_fractions};
use crate::maths::{maths_group, vec2s};

/// `Vec2` has no SIMD form. Every operator and computing method has a case; `new`, the `From`
/// conversions and the constants only copy fields.
pub(super) fn benchmark_vec2(c: &mut Criterion) {
    let mut group = maths_group(c);

    let a = vec2s(0);
    let b = vec2s(1);
    let factors = scalars();
    let lerps: Vec<_> = a.iter().copied().zip(with_fractions(&b)).collect();

    bench_zip(&mut group, "vec2_add", Variant::Only, &a, &b, |a, b| a + b);
    bench_zip(&mut group, "vec2_sub", Variant::Only, &a, &b, |a, b| a - b);
    bench_zip(&mut group, "vec2_mul", Variant::Only, &a, &b, |a, b| a * b);
    bench_zip(
        &mut group,
        "vec2_mul_scalar",
        Variant::Only,
        &a,
        &factors,
        |a, s| a * s,
    );
    bench_map(
        &mut group,
        "vec2_lerp",
        Variant::Only,
        &lerps,
        |(a, (b, t))| a.lerp(b, t),
    );
    bench_zip(&mut group, "vec2_cross", Variant::Only, &a, &b, |a, b| {
        a.cross(b)
    });
    bench_map(&mut group, "vec2_length", Variant::Only, &a, |a| a.length());

    group.finish();
}
