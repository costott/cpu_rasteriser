use std::array;
use std::f32::consts::TAU;
use std::hint::black_box;

use cpu_rasteriser::prelude::*;
use criterion::Criterion;

use crate::fixtures::{BATCH, Variant, bench_map, bench_zip, unit};
use crate::maths::{maths_group, vec3s, vec4s};

/// The operators, `inverse` and `transpose` each have their own loops, and their indexed form is
/// what the `needless_range_loop` expectations in `Mat4` guard (iterator forms measured slower, as
/// the reasons on those expectations say), so each case guards its loop's code generation.
/// `Mat4 * Vec4` transforms vertices by one matrix; the other cases combine a fixed matrix with
/// each of a batch of matrices. Every constructor that computes has a case; `new`, `identity`,
/// `translation`, `scaling`, `reflect_*`, `shear` and the `_vec` forms only place their arguments.
pub(super) fn benchmark_mat4(c: &mut Criterion) {
    let mut group = maths_group(c);

    let fixed = black_box(Mat4::perspective(1.0, 16.0 / 9.0, 0.1, 100.0) * Mat4::rotate_y(0.3));
    let matrices = matrices();

    bench_map(&mut group, "mat4_mul_vec4", Variant::Only, &vec4s(0), |v| {
        fixed * v
    });
    bench_map(&mut group, "mat4_mul_mat4", Variant::Only, &matrices, |m| {
        fixed * m
    });
    bench_map(
        &mut group,
        "mat4_mul_scalar",
        Variant::Only,
        &matrices,
        |m| m * 2.0,
    );
    bench_map(&mut group, "mat4_add", Variant::Only, &matrices, |m| {
        fixed + m
    });
    bench_map(&mut group, "mat4_sub", Variant::Only, &matrices, |m| {
        fixed - m
    });
    bench_map(
        &mut group,
        "mat4_transpose",
        Variant::Only,
        &matrices,
        |m| m.transpose(),
    );
    bench_map(&mut group, "mat4_inverse", Variant::Only, &matrices, |m| {
        m.inverse()
    });

    let angles: Vec<f32> = (0..BATCH).map(|i| unit(i) * TAU).collect();
    let rotations: Vec<Vec3> = vec3s(2).into_iter().map(|v| v * TAU).collect();
    let eyes: Vec<Vec3> = vec3s(3).into_iter().map(|v| v * 10.0).collect();
    let targets = vec3s(4);
    let perspectives: Vec<[f32; 4]> = (0..BATCH)
        .map(|i| {
            [
                0.5 + unit(4 * i) * 1.5,
                1.0 + unit(4 * i + 1),
                0.05 + unit(4 * i + 2) * 0.2,
                50.0 + unit(4 * i + 3) * 100.0,
            ]
        })
        .collect();
    let orthographics: Vec<[f32; 6]> = (0..BATCH)
        .map(|i| {
            let [width, height, depth] = array::from_fn(|c| 1.0 + unit(3 * i + c) * 10.0);
            [-width, width, -height, height, 0.1, depth + 1.0]
        })
        .collect();

    bench_map(
        &mut group,
        "mat4_rotate_x",
        Variant::Only,
        &angles,
        Mat4::rotate_x,
    );
    bench_map(
        &mut group,
        "mat4_rotate_y",
        Variant::Only,
        &angles,
        Mat4::rotate_y,
    );
    bench_map(
        &mut group,
        "mat4_rotate_z",
        Variant::Only,
        &angles,
        Mat4::rotate_z,
    );
    bench_map(
        &mut group,
        "mat4_rotate_vec",
        Variant::Only,
        &rotations,
        Mat4::rotate_vec,
    );
    bench_zip(
        &mut group,
        "mat4_look_at",
        Variant::Only,
        &eyes,
        &targets,
        |eye, target| Mat4::look_at(eye, target, Vec3::new(0.0, 1.0, 0.0)),
    );
    bench_map(
        &mut group,
        "mat4_perspective",
        Variant::Only,
        &perspectives,
        |[fov, aspect, near, far]| Mat4::perspective(fov, aspect, near, far),
    );
    bench_map(
        &mut group,
        "mat4_orthographic",
        Variant::Only,
        &orthographics,
        |[left, right, bottom, top, near, far]| {
            Mat4::orthographic(left, right, bottom, top, near, far)
        },
    );

    group.finish();
}

/// Returns `BATCH` invertible matrices shaped like model or view transforms: a rotation about
/// two axes and a translation.
fn matrices() -> Vec<Mat4> {
    (0..BATCH)
        .map(|i| {
            Mat4::translation(
                unit(4 * i) * 10.0,
                unit(4 * i + 1) * 10.0,
                unit(4 * i + 2) * 10.0,
            ) * Mat4::rotate_x(unit(4 * i + 3) * TAU)
                * Mat4::rotate_y(unit(4 * i + 2) * TAU)
        })
        .collect()
}
