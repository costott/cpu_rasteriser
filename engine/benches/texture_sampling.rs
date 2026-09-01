use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cpu_rasteriser::{prelude::*, wide::f32x8};

use engine::components::texture::{FilterMode, TextureSampler, WrapMode};

const SCREEN_WIDTH: usize = 1920;
const SCREEN_HEIGHT: usize = 1080;

fn make_texture(width: u32, height: u32) -> Arc<[Colour]> {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let r = x as f32 * 1.0 / width as f32;
            let g = y as f32 * 1.0 / height as f32;
            let b = (x + y) as f32 * 1.0 / (width + height) as f32;

            let colour = Colour::new(r, g, b, 1.0);

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
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &uvs {
                result = sampler_512_nearest.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("linear_512", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &uvs {
                result = sampler_512_linear.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("nearest_1024", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &uvs {
                result = sampler_1024_nearest.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("linear_1024", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &uvs {
                result = sampler_1024_linear.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("nearest_2048", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &uvs {
                result = sampler_2048_nearest.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("linear_2048", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

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
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &uvs {
                result = sampler.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("sample_linear_clamp", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &uvs {
                result = sampler.sample_linear_clamp(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("sample_nearest_clamp", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

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
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &fullscreen_uvs {
                result = sampler.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.bench_function("random", |b| {
        b.iter(|| {
            let mut result = Colour::new(0.0, 0.0, 0.0, 1.0);

            for &uv in &random_uvs {
                result = sampler.sample(black_box(uv));
            }

            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_simd_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("texture_sampling_simd");

    let uvs = make_fullscreen_uvs(SCREEN_WIDTH, SCREEN_HEIGHT);

    let sampler_512_linear = make_sampler(512, 512, WrapMode::Clamp, FilterMode::Linear);

    let sampler_1024_linear = make_sampler(1024, 1024, WrapMode::Clamp, FilterMode::Linear);

    let sampler_2048_linear = make_sampler(2048, 2048, WrapMode::Clamp, FilterMode::Linear);

    for (name, sampler) in [
        ("linear_512", &sampler_512_linear),
        ("linear_1024", &sampler_1024_linear),
        ("linear_2048", &sampler_2048_linear),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), sampler, |b, sampler| {
            b.iter(|| {
                let mut checksum = ColourSimd::splat(Colour::BLACK);

                for uvs in uvs.chunks_exact(8) {
                    let uv_x = f32x8::new([
                        uvs[0].x, uvs[1].x, uvs[2].x, uvs[3].x, uvs[4].x, uvs[5].x, uvs[6].x,
                        uvs[7].x,
                    ]);

                    let uv_y = f32x8::new([
                        uvs[0].y, uvs[1].y, uvs[2].y, uvs[3].y, uvs[4].y, uvs[5].y, uvs[6].y,
                        uvs[7].y,
                    ]);

                    let colour = sampler.sample_linear_clamp_simd([uv_x, uv_y]);

                    checksum.r += colour.r;
                    checksum.g += colour.g;
                    checksum.b += colour.b;
                    checksum.a += colour.a;
                }

                black_box(checksum);
            });
        });
    }

    group.finish();
}

fn benchmark_simd_linear_implementations(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_implementation_simd");

    let sampler = make_sampler(1920, 1080, WrapMode::Clamp, FilterMode::Linear);

    let uvs = make_fullscreen_uvs(SCREEN_WIDTH, SCREEN_HEIGHT);

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let mut checksum = Colour::BLACK;

            for &uv in &uvs {
                let colour = sampler.sample_linear_clamp(black_box(uv));

                checksum.r += colour.r;
                checksum.g += colour.g;
                checksum.b += colour.b;
                checksum.a += colour.a;
            }

            black_box(checksum);
        });
    });

    group.bench_function("simd", |b| {
        b.iter(|| {
            let mut checksum = ColourSimd::splat(Colour::BLACK);

            for uvs in uvs.chunks_exact(8) {
                let uv_x = f32x8::new([
                    uvs[0].x, uvs[1].x, uvs[2].x, uvs[3].x, uvs[4].x, uvs[5].x, uvs[6].x, uvs[7].x,
                ]);

                let uv_y = f32x8::new([
                    uvs[0].y, uvs[1].y, uvs[2].y, uvs[3].y, uvs[4].y, uvs[5].y, uvs[6].y, uvs[7].y,
                ]);

                let colour = sampler.sample_linear_clamp_simd([uv_x, uv_y]);

                checksum.r += colour.r;
                checksum.g += colour.g;
                checksum.b += colour.b;
                checksum.a += colour.a;
            }

            black_box(checksum);
        });
    });

    group.finish();
}

fn benchmark_simd_uv_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("uv_pattern_simd");

    let sampler = make_sampler(2048, 2048, WrapMode::Clamp, FilterMode::Linear);

    let fullscreen_uvs = make_fullscreen_uvs(SCREEN_WIDTH, SCREEN_HEIGHT);

    let random_uvs = make_random_uvs(SCREEN_WIDTH * SCREEN_HEIGHT);

    for (name, uvs) in [
        ("fullscreen_coherent", &fullscreen_uvs),
        ("random", &random_uvs),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), uvs, |b, uvs| {
            b.iter(|| {
                let mut checksum = ColourSimd::splat(Colour::BLACK);

                for uvs in uvs.chunks_exact(8) {
                    let uv_x = f32x8::new([
                        uvs[0].x, uvs[1].x, uvs[2].x, uvs[3].x, uvs[4].x, uvs[5].x, uvs[6].x,
                        uvs[7].x,
                    ]);

                    let uv_y = f32x8::new([
                        uvs[0].y, uvs[1].y, uvs[2].y, uvs[3].y, uvs[4].y, uvs[5].y, uvs[6].y,
                        uvs[7].y,
                    ]);

                    let colour = sampler.sample_linear_clamp_simd([uv_x, uv_y]);

                    checksum.r += colour.r;
                    checksum.g += colour.g;
                    checksum.b += colour.b;
                    checksum.a += colour.a;
                }

                black_box(checksum);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // benchmark_sampling,
    // benchmark_linear_implementations,
    // benchmark_uv_patterns,
    benchmark_simd_sampling,
    benchmark_simd_linear_implementations,
    benchmark_simd_uv_patterns,
);

criterion_main!(benches);
