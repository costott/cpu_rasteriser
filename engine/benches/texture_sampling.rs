use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cpu_rasteriser::prelude::*;

use engine::components::texture::{FilterMode, TextureSampler, WrapMode};

const SCREEN_WIDTH: usize = 1920;
const SCREEN_HEIGHT: usize = 1080;

fn make_texture(width: u32, height: u32) -> Arc<[u32]> {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let r = x * 255 / width;
            let g = y * 255 / height;
            let b = (x + y) * 255 / (width + height);

            let colour = (255 << 24) | (r << 16) | (g << 8) | b;

            pixels.push(colour);
        }
    }

    pixels.into()
}

fn make_sampler(
    width: u32,
    height: u32,
    wrap_mode: WrapMode,
    filter_mode: FilterMode,
) -> TextureSampler {
    TextureSampler::new(
        make_texture(width, height),
        width,
        height,
        wrap_mode,
        filter_mode,
    )
}

fn make_fullscreen_uvs(width: usize, height: usize) -> Vec<Vec2> {
    let mut uvs = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / (width - 1) as f32;
            let v = y as f32 / (height - 1) as f32;

            uvs.push(Vec2::new(u, v));
        }
    }

    uvs
}

fn make_random_uvs(count: usize) -> Vec<Vec2> {
    let mut state = 0x12345678u32;
    let mut uvs = Vec::with_capacity(count);

    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        let u = state as f32 / u32::MAX as f32;

        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        let v = state as f32 / u32::MAX as f32;

        uvs.push(Vec2::new(u, v));
    }

    uvs
}

fn benchmark_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("texture_sampling");

    let uvs = make_fullscreen_uvs(SCREEN_WIDTH, SCREEN_HEIGHT);

    let sampler_512_nearest = make_sampler(512, 512, WrapMode::Clamp, FilterMode::Nearest);

    let sampler_512_linear = make_sampler(512, 512, WrapMode::Clamp, FilterMode::Linear);

    let sampler_1024_nearest = make_sampler(1024, 1024, WrapMode::Clamp, FilterMode::Nearest);

    let sampler_1024_linear = make_sampler(1024, 1024, WrapMode::Clamp, FilterMode::Linear);

    let sampler_2048_nearest = make_sampler(2048, 2048, WrapMode::Clamp, FilterMode::Nearest);

    let sampler_2048_linear = make_sampler(2048, 2048, WrapMode::Clamp, FilterMode::Linear);

    group.bench_function("nearest_512", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler_512_nearest.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("linear_512", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler_512_linear.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("nearest_1024", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler_1024_nearest.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("linear_1024", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler_1024_linear.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("nearest_2048", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler_2048_nearest.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("linear_2048", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler_2048_linear.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_linear_implementations(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_implementation");

    let sampler = make_sampler(1920, 1080, WrapMode::Clamp, FilterMode::Linear);

    let uvs = make_fullscreen_uvs(SCREEN_WIDTH, SCREEN_HEIGHT);

    group.bench_function("old_sample", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("sample_linear_clamp", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler.sample_linear_clamp(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("sample_nearest_clamp", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &uvs {
                result = sampler.sample_nearest_clamp(black_box(uv));
            }

            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_uv_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("uv_pattern");

    let sampler = make_sampler(2048, 2048, WrapMode::Clamp, FilterMode::Linear);

    let fullscreen_uvs = make_fullscreen_uvs(SCREEN_WIDTH, SCREEN_HEIGHT);

    let random_uvs = make_random_uvs(SCREEN_WIDTH * SCREEN_HEIGHT);

    group.bench_function("fullscreen_coherent", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &fullscreen_uvs {
                result = sampler.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("random", |b| {
        b.iter(|| {
            let mut result = Colour::new(0, 0, 0, 255);

            for &uv in &random_uvs {
                result = sampler.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_sampling,
    benchmark_linear_implementations,
    benchmark_uv_patterns,
);

criterion_main!(benches);
