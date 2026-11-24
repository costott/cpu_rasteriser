use std::hint::black_box;

use cpu_rasteriser::internals::FrameBuffer;
use cpu_rasteriser::prelude::*;
use cpu_rasteriser::wide::f32x8;
use criterion::{BatchSize, Criterion, Throughput};

use crate::buffers::{
    EDGE_HEIGHT, TILE, TILE_ORIGIN, bench_loads, bench_stores, buffers_group, group_spans, inputs,
    pixel_spans, position,
};
use crate::fixtures::{TARGET_EXTENT, Variant, register, unit};

/// `FrameBuffer`'s accesses over a tile, a pixel or an 8-wide group at a time, row by row from an
/// unaligned start (see `pixel_spans` and `group_spans`).
///
/// Only the accesses the renderer calls have cases. `framebuffer_load` is the one pair:
/// `get_pixel` against `get8_unchecked` over the same pixels. `set_pixel` and
/// `set_pixel_index_unchecked` (the SIMD path's colour store) have no counterpart that does the
/// same job, so each has one case. `index_unchecked` is part of every by-index case, as in the
/// rasteriser. `width`, `height` and `extent` only read fields, and `pixels_compact`,
/// `pixels_u32`, `write_pixels` and `write_pixels_u32` run the same code as the `output` group's
/// `RenderTarget` reads. The accessors nothing in the crate calls (`get_pixel_unchecked`,
/// `set_pixel_unchecked` and the per-channel 8-wide ones) have no case.
pub(super) fn benchmark_framebuffer_access(c: &mut Criterion) {
    let mut group = buffers_group(c);

    let mut framebuffer = tile_framebuffer();
    let colours = inputs(pixel_spans, colour_at);

    bench_loads(
        &mut group,
        "framebuffer_load",
        Variant::Scalar,
        &framebuffer,
        pixel_spans,
        |framebuffer, x, y| {
            fold(
                framebuffer
                    .get_pixel(position(x, y))
                    .unwrap_or(Colour::BLACK),
            )
        },
    );
    bench_loads(
        &mut group,
        "framebuffer_load",
        Variant::Simd,
        &framebuffer,
        group_spans,
        |framebuffer, x, y| {
            let index = framebuffer.index_unchecked(x, y);
            // SAFETY: `x < TILE` and `y < TILE` for a `TILE` × `TILE` buffer, so `index` is
            // `index_unchecked(x, y)` of an in-bounds pixel.
            fold_simd(unsafe { framebuffer.get8_unchecked(index) })
        },
    );
    bench_stores(
        &mut group,
        "framebuffer_set",
        Variant::Only,
        &mut framebuffer,
        pixel_spans,
        &colours,
        |framebuffer, x, y, colour| framebuffer.set_pixel(position(x, y), colour),
    );
    bench_stores(
        &mut group,
        "framebuffer_set_index_unchecked",
        Variant::Only,
        &mut framebuffer,
        pixel_spans,
        &colours,
        |framebuffer, x, y, colour| {
            let index = framebuffer.index_unchecked(x, y);
            // SAFETY: `x < TILE` and `y < TILE` for a `TILE` × `TILE` buffer, so `index` is
            // `index_unchecked(x, y)` of an in-bounds pixel, plus 0.
            unsafe { framebuffer.set_pixel_index_unchecked(index, colour) }
        },
    );

    group.finish();
}

/// `FrameBuffer`'s whole-buffer operations, at the size a pass uses them. `clear` is a tile's
/// colour load; `new` includes the allocation and its release; `resize_within_capacity` shrinks a
/// tile to the height of the bottom row of tiles and grows it back, as a pooled buffer is reused
/// by tiles of different heights; `resize_beyond_capacity` grows a buffer made for that bottom row
/// to a full tile, which reallocates. `blit_from` has a case for a tile merged into the target
/// (`blit_tile`, always the same interior tile) and one for a target-sized copy (`blit_target`).
/// Loading a tile from the target runs the same code with the roles swapped, so it has no case.
pub(super) fn benchmark_framebuffer_storage(c: &mut Criterion) {
    let mut group = buffers_group(c);
    group.throughput(Throughput::Elements((TILE * TILE) as u64));

    let mut framebuffer = FrameBuffer::new(TILE, TILE);
    register(&mut group, "framebuffer_clear", Variant::Only, |bench| {
        bench.iter(|| {
            framebuffer.clear(black_box(Colour::new(0.25, 0.5, 0.75, 1.0)));
            black_box(&framebuffer);
        })
    });
    register(&mut group, "framebuffer_new", Variant::Only, |bench| {
        bench.iter(|| black_box(FrameBuffer::new(black_box(TILE), black_box(TILE))))
    });
    register(
        &mut group,
        "framebuffer_resize_within_capacity",
        Variant::Only,
        |bench| {
            bench.iter(|| {
                framebuffer.resize(black_box(TILE), black_box(EDGE_HEIGHT));
                framebuffer.resize(black_box(TILE), black_box(TILE));
                black_box(&framebuffer);
            })
        },
    );
    register(
        &mut group,
        "framebuffer_resize_beyond_capacity",
        Variant::Only,
        |bench| {
            bench.iter_batched(
                || FrameBuffer::new(TILE, EDGE_HEIGHT),
                |mut framebuffer| {
                    framebuffer.resize(black_box(TILE), black_box(TILE));
                    framebuffer
                },
                BatchSize::LargeInput,
            )
        },
    );

    let mut target = FrameBuffer::new(TARGET_EXTENT.width, TARGET_EXTENT.height);
    let tile = tile_framebuffer();
    register(
        &mut group,
        "framebuffer_blit_tile",
        Variant::Only,
        |bench| {
            bench.iter(|| {
                target.blit_from(
                    black_box(TILE_ORIGIN),
                    black_box(&tile),
                    (0, 0),
                    Extent::new(TILE, TILE),
                );
                black_box(&target);
            })
        },
    );

    group.throughput(Throughput::Elements(
        (TARGET_EXTENT.width * TARGET_EXTENT.height) as u64,
    ));
    let source = FrameBuffer::new(TARGET_EXTENT.width, TARGET_EXTENT.height);
    register(
        &mut group,
        "framebuffer_blit_target",
        Variant::Only,
        |bench| {
            bench.iter(|| {
                target.blit_from((0, 0), black_box(&source), (0, 0), TARGET_EXTENT);
                black_box(&target);
            })
        },
    );

    group.finish();
}

/// Returns a tile-sized framebuffer with scattered colours.
fn tile_framebuffer() -> FrameBuffer {
    let mut framebuffer = FrameBuffer::new(TILE, TILE);
    for y in 0..TILE {
        for x in 0..TILE {
            framebuffer.set_pixel(position(x, y), colour_at(x, y));
        }
    }
    framebuffer
}

fn colour_at(x: usize, y: usize) -> Colour {
    let i = 4 * (y * TILE + x);
    Colour::new(unit(i), unit(i + 1), unit(i + 2), unit(i + 3))
}

/// Folds the bits of every channel into one value.
fn fold(colour: Colour) -> u32 {
    (colour.r.to_bits() ^ colour.g.to_bits()) ^ (colour.b.to_bits() ^ colour.a.to_bits())
}

fn fold_simd(colour: ColourSimd) -> f32x8 {
    (colour.r ^ colour.g) ^ (colour.b ^ colour.a)
}
