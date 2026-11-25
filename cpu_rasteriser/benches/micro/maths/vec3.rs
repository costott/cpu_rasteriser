use std::array;

use cpu_rasteriser::prelude::*;
use cpu_rasteriser::wide::f32x8;
use criterion::Criterion;

use crate::fixtures::{Variant, bench_map, bench_zip, scalars, to_lanes};
use crate::maths::{maths_group, vec3s};

/// Every operator and computing method has a case. Those `Vec3Simd` also has (`Add`, `Sub`, `Neg`,
/// `Mul<f32x8>`, `dot`, `normalise`) pair one scalar pass over `BATCH` vectors with `BATCH / 8`
/// operations over the same vectors; `Mul<f32>` pairs with `Mul<f32x8>`, scaling by the same
/// factors. `Div<f32>`, the assign forms, `cross` and `length` have no `Vec3Simd` form. `new`, the
/// `From` conversions, the constants, `to_point4`, `to_direction4` and `xy` only copy fields.
pub(super) fn benchmark_vec3(c: &mut Criterion) {
    let mut group = maths_group(c);

    let a = vec3s(0);
    let b = vec3s(1);
    let factors = scalars();
    let (a_simd, b_simd) = (to_simd(&a), to_simd(&b));
    let factors_simd = to_lanes(&factors);

    bench_zip(&mut group, "vec3_add", Variant::Scalar, &a, &b, |a, b| {
        a + b
    });
    bench_zip(
        &mut group,
        "vec3_add",
        Variant::Simd,
        &a_simd,
        &b_simd,
        |a, b| a + b,
    );
    bench_zip(&mut group, "vec3_sub", Variant::Scalar, &a, &b, |a, b| {
        a - b
    });
    bench_zip(
        &mut group,
        "vec3_sub",
        Variant::Simd,
        &a_simd,
        &b_simd,
        |a, b| a - b,
    );
    bench_map(&mut group, "vec3_neg", Variant::Scalar, &a, |a| -a);
    bench_map(&mut group, "vec3_neg", Variant::Simd, &a_simd, |a| -a);
    bench_zip(
        &mut group,
        "vec3_mul_scalar",
        Variant::Scalar,
        &a,
        &factors,
        |a, s| a * s,
    );
    bench_zip(
        &mut group,
        "vec3_mul_scalar",
        Variant::Simd,
        &a_simd,
        &factors_simd,
        |a, s| a * s,
    );
    bench_zip(
        &mut group,
        "vec3_div_scalar",
        Variant::Only,
        &a,
        &factors,
        |a, s| a / s,
    );
    bench_zip(
        &mut group,
        "vec3_add_assign",
        Variant::Only,
        &a,
        &b,
        |mut a, b| {
            a += b;
            a
        },
    );
    bench_zip(
        &mut group,
        "vec3_sub_assign",
        Variant::Only,
        &a,
        &b,
        |mut a, b| {
            a -= b;
            a
        },
    );
    bench_zip(&mut group, "vec3_dot", Variant::Scalar, &a, &b, |a, b| {
        a.dot(b)
    });
    bench_zip(
        &mut group,
        "vec3_dot",
        Variant::Simd,
        &a_simd,
        &b_simd,
        |a, b| a.dot(b),
    );
    bench_map(&mut group, "vec3_normalise", Variant::Scalar, &a, |a| {
        a.normalise()
    });
    bench_map(&mut group, "vec3_normalise", Variant::Simd, &a_simd, |a| {
        a.normalise()
    });
    bench_zip(&mut group, "vec3_cross", Variant::Only, &a, &b, |a, b| {
        a.cross(b)
    });
    bench_map(&mut group, "vec3_length", Variant::Only, &a, |a| a.length());

    group.finish();
}

/// Packs consecutive groups of eight vectors into lanes.
fn to_simd(vectors: &[Vec3]) -> Vec<Vec3Simd> {
    vectors
        .chunks_exact(8)
        .map(|group| {
            Vec3Simd::new(
                f32x8::from(array::from_fn(|lane| group[lane].x)),
                f32x8::from(array::from_fn(|lane| group[lane].y)),
                f32x8::from(array::from_fn(|lane| group[lane].z)),
            )
        })
        .collect()
}
