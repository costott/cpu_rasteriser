use std::array;

use cpu_rasteriser::prelude::*;
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group};

use crate::fixtures::{BATCH, unit};

mod mat4;
mod vec2;
mod vec3;
mod vec4;

fn maths_group(c: &mut Criterion) -> BenchmarkGroup<'_, WallTime> {
    let mut group = c.benchmark_group("maths");
    group.throughput(Throughput::Elements(BATCH as u64));
    group
}

/// Returns `BATCH` arrays of components in `[-1, 1)`; `stream` selects an independent set.
fn components<const N: usize>(stream: usize) -> Vec<[f32; N]> {
    (0..BATCH)
        .map(|i| array::from_fn(|c| unit(N * (stream * BATCH + i) + c) * 2.0 - 1.0))
        .collect()
}

fn vec2s(stream: usize) -> Vec<Vec2> {
    components::<2>(stream)
        .into_iter()
        .map(|[x, y]| Vec2::new(x, y))
        .collect()
}

fn vec3s(stream: usize) -> Vec<Vec3> {
    components::<3>(stream)
        .into_iter()
        .map(|[x, y, z]| Vec3::new(x, y, z))
        .collect()
}

/// Returns clip-space-like vectors: `w` is in `[0.5, 1.5)`, so it is never zero.
fn vec4s(stream: usize) -> Vec<Vec4> {
    components::<4>(stream)
        .into_iter()
        .enumerate()
        .map(|(i, [x, y, z, _])| Vec4::new(x, y, z, 0.5 + unit(stream * BATCH + i)))
        .collect()
}

criterion_group!(
    benches,
    vec2::benchmark_vec2,
    vec3::benchmark_vec3,
    vec4::benchmark_vec4,
    mat4::benchmark_mat4,
);
