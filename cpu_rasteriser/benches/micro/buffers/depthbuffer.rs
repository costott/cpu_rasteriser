use std::hint::black_box;

use cpu_rasteriser::internals::DepthBuffer;
use cpu_rasteriser::prelude::*;
use criterion::{BatchSize, Criterion, Throughput};

use crate::buffers::{
    EDGE_HEIGHT, TILE, TILE_ORIGIN, bench_loads, bench_stores, buffers_group, group_mask,
    group_spans, inputs, lanes, passes, pixel_spans, position,
};
use crate::fixtures::{TARGET_EXTENT, Variant, register, unit};

/// `DepthBuffer`'s accesses over a tile, a pixel or an 8-wide group at a time, row by row from an
/// unaligned start (see `pixel_spans` and `group_spans`).
///
/// Three pairs set the 8-wide operation against the scalar ones that do its job over the same
/// pixels. `depthbuffer_load` is `get8_unchecked` against `get`, `depthbuffer_store` is
/// `set8_unchecked` against `set_depth`, and `depthbuffer_store_masked` is
/// `set8_unchecked_with_mask` against a `set_depth` for each pixel whose lane is set. The scalar
/// `set_depth` takes a branch on the same scattered outcomes as the mask, which the SIMD form
/// avoids. `index_unchecked` is part of every by-index case, as in the rasteriser. `width` and
/// `height` only read fields, and `get_unchecked` and `set_depth_unchecked`, which nothing in the
/// crate calls, have no case.
pub(super) fn benchmark_depthbuffer_access(c: &mut Criterion) {
    let mut group = buffers_group(c);

    let mut depthbuffer = tile_depthbuffer();
    let depths = inputs(pixel_spans, depth_at);
    let depth_lanes = inputs(group_spans, |x, y| lanes(x, y, depth_at));
    let depths_and_passes = inputs(pixel_spans, |x, y| (depth_at(x, y), passes(x, y)));
    let depth_lanes_and_masks = inputs(group_spans, |x, y| {
        (lanes(x, y, depth_at), group_mask(x, y))
    });

    bench_loads(
        &mut group,
        "depthbuffer_load",
        Variant::Scalar,
        &depthbuffer,
        pixel_spans,
        |depthbuffer, x, y| depthbuffer.get(position(x, y)).to_bits(),
    );
    bench_loads(
        &mut group,
        "depthbuffer_load",
        Variant::Simd,
        &depthbuffer,
        group_spans,
        |depthbuffer, x, y| {
            let index = depthbuffer.index_unchecked(x, y);
            // SAFETY: `x < TILE` and `y < TILE` for a `TILE` × `TILE` buffer, so `index` is
            // `index_unchecked(x, y)` of an in-bounds pixel.
            unsafe { depthbuffer.get8_unchecked(index) }
        },
    );

    bench_stores(
        &mut group,
        "depthbuffer_store",
        Variant::Scalar,
        &mut depthbuffer,
        pixel_spans,
        &depths,
        |depthbuffer, x, y, depth| depthbuffer.set_depth(position(x, y), depth),
    );
    bench_stores(
        &mut group,
        "depthbuffer_store",
        Variant::Simd,
        &mut depthbuffer,
        group_spans,
        &depth_lanes,
        |depthbuffer, x, y, depth| {
            let index = depthbuffer.index_unchecked(x, y);
            // SAFETY: `x < TILE` and `y < TILE` for a `TILE` × `TILE` buffer, so `index` is
            // `index_unchecked(x, y)` of an in-bounds pixel.
            unsafe { depthbuffer.set8_unchecked(index, depth) }
        },
    );
    bench_stores(
        &mut group,
        "depthbuffer_store_masked",
        Variant::Scalar,
        &mut depthbuffer,
        pixel_spans,
        &depths_and_passes,
        |depthbuffer, x, y, (depth, pass)| {
            if pass {
                depthbuffer.set_depth(position(x, y), depth);
            }
        },
    );
    bench_stores(
        &mut group,
        "depthbuffer_store_masked",
        Variant::Simd,
        &mut depthbuffer,
        group_spans,
        &depth_lanes_and_masks,
        |depthbuffer, x, y, (depth, mask)| {
            let index = depthbuffer.index_unchecked(x, y);
            // SAFETY: `x < TILE` and `y < TILE` for a `TILE` × `TILE` buffer, so `index` is
            // `index_unchecked(x, y)` of an in-bounds pixel. `group_mask` lanes are all ones or
            // all zeros, as the method requires.
            unsafe { depthbuffer.set8_unchecked_with_mask(index, depth, mask) }
        },
    );

    group.finish();
}

/// `DepthBuffer`'s whole-buffer operations, at the size a pass uses them, with the cases and
/// reasons of `benchmark_framebuffer_storage`.
pub(super) fn benchmark_depthbuffer_storage(c: &mut Criterion) {
    let mut group = buffers_group(c);
    group.throughput(Throughput::Elements((TILE * TILE) as u64));

    let mut depthbuffer = DepthBuffer::new(TILE, TILE);
    register(&mut group, "depthbuffer_clear", Variant::Only, |bench| {
        bench.iter(|| {
            depthbuffer.clear(black_box(0.5));
            black_box(&depthbuffer);
        })
    });
    register(&mut group, "depthbuffer_new", Variant::Only, |bench| {
        bench.iter(|| black_box(DepthBuffer::new(black_box(TILE), black_box(TILE))))
    });
    register(
        &mut group,
        "depthbuffer_resize_within_capacity",
        Variant::Only,
        |bench| {
            bench.iter(|| {
                depthbuffer.resize(black_box(TILE), black_box(EDGE_HEIGHT));
                depthbuffer.resize(black_box(TILE), black_box(TILE));
                black_box(&depthbuffer);
            })
        },
    );
    register(
        &mut group,
        "depthbuffer_resize_beyond_capacity",
        Variant::Only,
        |bench| {
            bench.iter_batched(
                || DepthBuffer::new(TILE, EDGE_HEIGHT),
                |mut depthbuffer| {
                    depthbuffer.resize(black_box(TILE), black_box(TILE));
                    depthbuffer
                },
                BatchSize::LargeInput,
            )
        },
    );

    let mut target = DepthBuffer::new(TARGET_EXTENT.width, TARGET_EXTENT.height);
    let tile = tile_depthbuffer();
    register(
        &mut group,
        "depthbuffer_blit_tile",
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
    let source = DepthBuffer::new(TARGET_EXTENT.width, TARGET_EXTENT.height);
    register(
        &mut group,
        "depthbuffer_blit_target",
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

/// Returns a tile-sized depth buffer with scattered depths.
fn tile_depthbuffer() -> DepthBuffer {
    let mut depthbuffer = DepthBuffer::new(TILE, TILE);
    for y in 0..TILE {
        for x in 0..TILE {
            depthbuffer.set_depth(position(x, y), depth_at(x, y));
        }
    }
    depthbuffer
}

fn depth_at(x: usize, y: usize) -> f32 {
    unit(2 * (y * TILE + x))
}
