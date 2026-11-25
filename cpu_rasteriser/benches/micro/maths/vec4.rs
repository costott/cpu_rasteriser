use criterion::Criterion;

use crate::fixtures::{Variant, bench_map, bench_zip, scalars, with_fractions};
use crate::maths::{maths_group, vec4s};

/// `Vec4` has no SIMD form. Every operator and computing method has a case; `new`, the `From`
/// conversions, `xyz` and `xy` only copy fields.
pub(super) fn benchmark_vec4(c: &mut Criterion) {
    let mut group = maths_group(c);

    let a = vec4s(0);
    let b = vec4s(1);
    let factors = scalars();
    let lerps: Vec<_> = a.iter().copied().zip(with_fractions(&b)).collect();

    bench_zip(&mut group, "vec4_add", Variant::Only, &a, &b, |a, b| a + b);
    bench_zip(
        &mut group,
        "vec4_add_assign",
        Variant::Only,
        &a,
        &b,
        |mut a, b| {
            a += b;
            a
        },
    );
    bench_zip(&mut group, "vec4_sub", Variant::Only, &a, &b, |a, b| a - b);
    bench_zip(
        &mut group,
        "vec4_mul_scalar",
        Variant::Only,
        &a,
        &factors,
        |a, s| a * s,
    );
    bench_zip(
        &mut group,
        "vec4_div_scalar",
        Variant::Only,
        &a,
        &factors,
        |a, s| a / s,
    );
    bench_map(
        &mut group,
        "vec4_lerp",
        Variant::Only,
        &lerps,
        |(a, (b, t))| a.lerp(b, t),
    );
    bench_zip(&mut group, "vec4_dot", Variant::Only, &a, &b, |a, b| {
        a.dot(b)
    });
    bench_map(&mut group, "vec4_normalise", Variant::Only, &a, |a| {
        a.normalise()
    });
    bench_map(&mut group, "vec4_homogenize", Variant::Only, &a, |a| {
        a.homogenize()
    });
    bench_map(
        &mut group,
        "vec4_homogenize_to_vec3",
        Variant::Only,
        &a,
        |a| a.homogenize_to_vec3(),
    );

    group.finish();
}
