use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpu_rasteriser::internals::{
    DepthBuffer, FrameBuffer, RasterCommand, Rect, TileBinner, TilePool, Triangle2D, render_tile,
};
use cpu_rasteriser::prelude::*;
use criterion::measurement::WallTime;
use criterion::{Bencher, Criterion, Throughput, criterion_group};

use crate::fixtures::{BATCH, EDGE_HEIGHT, TARGET, TARGET_EXTENT, TILE, Varyings, triangle, unit};

const TILES_X: usize = (TARGET.max_x as usize).div_ceil(TILE);
const TILES_Y: usize = (TARGET.max_y as usize).div_ceil(TILE);

/// The tiles of `TARGET`: the work of one render pass's tile set-up and merge.
const TILES: usize = TILES_X * TILES_Y;

/// The bounds of one tile, for the intersection cases.
const TILE_BOUNDS: Rect = Rect {
    min_x: 2 * TileBinner::TILE_SIZE,
    min_y: 2 * TileBinner::TILE_SIZE,
    max_x: 3 * TileBinner::TILE_SIZE,
    max_y: 3 * TileBinner::TILE_SIZE,
};

/// Triangles are 24 px across, the size of a small triangle on a 640×360 target.
const TRIANGLE_SIZE: f32 = 24.0;

/// `Triangle2D::bounding_box` and `Triangle2D::intersects_rect` have one case each, over triangles
/// scattered across `TARGET` for the first and across a band around the rectangle for the second,
/// so that about 40% overlap it and the branches of its chain of four comparisons are
/// unpredictable. Outcomes with predictable branches (every triangle inside, or every one outside)
/// measured the same, so they have no cases of their own.
fn benchmark_triangle_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("binning");
    group.throughput(Throughput::Elements(BATCH as u64));

    let scattered = small_triangles((0.0, TARGET.max_x as f32), (0.0, TARGET.max_y as f32));
    group.bench_function("bounding_box", |bench| {
        bench.iter(|| {
            let mut total = 0.0;
            for triangle in black_box(&scattered) {
                let (min, max) = triangle.bounding_box();
                total += min.x + min.y + max.x + max.y;
            }
            black_box(total)
        })
    });

    let mixed = small_triangles(
        (
            TILE_BOUNDS.min_x as f32 - 76.0,
            TILE_BOUNDS.max_x as f32 + 76.0,
        ),
        (TILE_BOUNDS.min_y as f32, TILE_BOUNDS.max_y as f32),
    );
    group.bench_function("intersects_rect", |bench| {
        bench.iter(|| {
            let rect = black_box(TILE_BOUNDS);
            let mut overlapping = 0;
            for triangle in black_box(&mixed) {
                overlapping += triangle.intersects_rect(rect) as u32;
            }
            black_box(overlapping)
        })
    });

    group.finish();
}

/// Returns `BATCH` triangles, each with its corners within `TRIANGLE_SIZE` of a centre scattered
/// over `x` and `y`.
fn small_triangles(x: (f32, f32), y: (f32, f32)) -> Vec<Triangle2D<Varyings>> {
    (0..BATCH)
        .map(|i| {
            let centre_x = x.0 + (x.1 - x.0) * unit(8 * i);
            let centre_y = y.0 + (y.1 - y.0) * unit(8 * i + 1);
            let corner = |c: usize| {
                (
                    centre_x + TRIANGLE_SIZE * (unit(8 * i + 2 + 2 * c) - 0.5),
                    centre_y + TRIANGLE_SIZE * (unit(8 * i + 3 + 2 * c) - 0.5),
                )
            };
            triangle(corner(0), corner(1), corner(2))
        })
        .collect()
}

/// `TileBinner::bin_command` has a case for how many tiles a command's bounding box covers, which
/// is the number of `intersects` tests and pushes it makes: `bin_none` (beyond the target, so only
/// the bounding box and range arithmetic run), `bin_one_tile`, `bin_many_tiles` (3×3) and
/// `bin_whole_target` (every tile, with the range clamped at the target's edge). Each iteration
/// bins `BATCH` commands into a binner whose tiles hold none. The batch is cloned before timing,
/// drained inside it, and its emptied allocation released after it; each tile's list is dropped
/// after timing too, so that it grows from empty as in a pass.
fn benchmark_bin_command(c: &mut Criterion) {
    let mut group = c.benchmark_group("binning");
    group.throughput(Throughput::Elements(BATCH as u64));

    let cases: [(&str, BoundsOf); 4] = [
        ("bin_none", beyond_target),
        ("bin_one_tile", one_tile),
        ("bin_many_tiles", many_tiles),
        ("bin_whole_target", whole_target),
    ];
    for (name, bounds) in cases {
        let commands = commands(bounds);
        group.bench_function(name, |bench| {
            iter_cycle(
                bench,
                TileBinner::new(
                    &RenderTarget::new(TARGET_EXTENT),
                    &LoadOp::Clear(Colour::BLACK),
                    &None,
                    &mut TilePool::default(),
                ),
                |_| commands.clone(),
                |binner, mut batch| {
                    for command in batch.drain(..) {
                        binner.bin_command(command);
                    }
                    black_box(&binner.tiles);
                    batch
                },
                |binner, _batch| {
                    for tile in &mut binner.tiles {
                        tile.commands = Vec::new();
                    }
                },
            )
        });
    }

    group.finish();
}

/// A command's bounding box as its minimum and maximum corners, in pixels.
type Bounds = ((f32, f32), (f32, f32));

/// Returns the bounding box of the `i`th command of a batch.
type BoundsOf = fn(usize) -> Bounds;

fn commands(bounds: BoundsOf) -> Vec<Arc<dyn RasterCommand>> {
    (0..BATCH)
        .map(|i| {
            let ((min_x, min_y), (max_x, max_y)) = bounds(i);
            let shape = triangle((min_x, min_y), (max_x, min_y), (min_x, max_y));
            Arc::new(BoundsCommand(shape)) as Arc<dyn RasterCommand>
        })
        .collect()
}

/// Stands in for the real raster commands, which are private to the crate. Binning never calls
/// `rasterise`, so the stand-in's does nothing; its bounding box and intersection test are the
/// triangle's, as in the real commands.
struct BoundsCommand(Triangle2D<Varyings>);
impl RasterCommand for BoundsCommand {
    fn rasterise(&self, _: &mut FrameBuffer, _: Option<&mut DepthBuffer>, _: Rect) {}

    fn bounding_box(&self) -> (Vec2, Vec2) {
        self.0.bounding_box()
    }

    fn intersects(&self, rect: Rect) -> bool {
        self.0.intersects_rect(rect)
    }
}

fn beyond_target(i: usize) -> Bounds {
    let min = (
        TARGET.max_x as f32 + 16.0 + 100.0 * unit(2 * i),
        300.0 * unit(2 * i + 1),
    );
    (min, (min.0 + TRIANGLE_SIZE, min.1 + TRIANGLE_SIZE))
}

fn one_tile(i: usize) -> Bounds {
    within_tiles(i, (1, 1), 40.0)
}

fn many_tiles(i: usize) -> Bounds {
    within_tiles(i, (3, 3), 144.0)
}

/// Returns a box of `size` that starts 8 to 16 px into a random tile. It spans exactly `tiles`
/// tiles in each direction when `size` is in `[(tiles - 1) * TILE - 8, tiles * TILE - 16)`, which
/// puts its far edge in the tile `tiles - 1` past the first. Both sizes used, 40 for one tile and
/// 144 for three, are in their range.
fn within_tiles(i: usize, tiles: (usize, usize), size: f32) -> Bounds {
    let tile_x = (unit(5 * i) * (TILES_X - tiles.0 + 1) as f32) as usize;
    let tile_y = (unit(5 * i + 1) * (TILES_Y - tiles.1 + 1) as f32) as usize;
    let min = (
        (tile_x * TILE) as f32 + 8.0 + 8.0 * unit(5 * i + 2),
        (tile_y * TILE) as f32 + 8.0 + 8.0 * unit(5 * i + 3),
    );
    (min, (min.0 + size, min.1 + size))
}

fn whole_target(i: usize) -> Bounds {
    (
        (8.0 * unit(4 * i), 8.0 * unit(4 * i + 1)),
        (
            TARGET.max_x as f32 - 8.0 + 16.0 * unit(4 * i + 2),
            TARGET.max_y as f32 - 8.0 + 16.0 * unit(4 * i + 3),
        ),
    )
}

/// The work a render pass does per tile outside rasterising, over all of `TARGET`'s tiles.
///
/// `TileBinner::new` runs `Tile::new` for every tile, its only caller, so it covers both. Its cases
/// are the tile buffers' source (`fresh` allocates them, `recycled` reuses a warm pool's) and the
/// load operations: `tile_new_recycled` clears the colour buffer, `tile_new_recycled_depth` also
/// clears a depth buffer, and the `load` pair copy from the target instead. `TilePool` has a case
/// for `take_framebuffer` and `take_depthbuffer` together on an empty pool (`pool_take_fresh`), on
/// a pool of buffers of the right size (`pool_take_recycled`) and on one of buffers last used by
/// the shorter bottom-row tiles, which grow within their capacity (`pool_take_resized`); a tile
/// always takes both when it has depth. `TilePool::recycle` has no case of its own: it is two
/// pushes per tile, below what a clock read per iteration resolves, and `merge_tile` times it in
/// context. `Renderer::merge_tile` has a case with depth and one without; its results are released
/// after timing, with the emptied vector that held them. Neither `render_tile` nor the raster
/// commands are benchmarked here: the frame benchmarks time them.
fn benchmark_tiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("binning");

    group.throughput(Throughput::Elements(pixel_count()));
    let load_cases = [
        (
            "tile_new_fresh",
            LoadOp::Clear(Colour::BLACK),
            None,
            Source::Fresh,
        ),
        (
            "tile_new_recycled",
            LoadOp::Clear(Colour::BLACK),
            None,
            Source::Recycled,
        ),
        (
            "tile_new_recycled_depth",
            LoadOp::Clear(Colour::BLACK),
            Some(LoadOp::Clear(1.0)),
            Source::Recycled,
        ),
        (
            "tile_new_recycled_load",
            LoadOp::Load,
            None,
            Source::Recycled,
        ),
        (
            "tile_new_recycled_load_depth",
            LoadOp::Load,
            Some(LoadOp::Load),
            Source::Recycled,
        ),
    ];
    for (name, colour_load_op, depth_load_op, source) in load_cases {
        group.bench_function(name, |bench| {
            let target = RenderTarget::new(TARGET_EXTENT).with_depth();
            let mut pool = TilePool::default();
            if let Source::Recycled = source {
                let warm = TileBinner::new(&target, &colour_load_op, &depth_load_op, &mut pool);
                recycle(warm, &mut pool);
            }
            iter_cycle(
                bench,
                (target, pool),
                |_| (),
                |(target, pool), ()| match source {
                    Source::Fresh => TileBinner::new(
                        target,
                        &colour_load_op,
                        &depth_load_op,
                        &mut TilePool::default(),
                    ),
                    Source::Recycled => {
                        TileBinner::new(target, &colour_load_op, &depth_load_op, pool)
                    }
                },
                |(_, pool), binner| {
                    if let Source::Recycled = source {
                        recycle(binner, pool);
                    }
                },
            )
        });
    }

    group.throughput(Throughput::Elements(TILES as u64));
    group.bench_function("pool_take_fresh", |bench| {
        iter_cycle(
            bench,
            (TilePool::default(), Vec::with_capacity(TILES)),
            |_| (),
            |(pool, tiles), ()| {
                take_buffers(pool, tiles, TILE);
                black_box(&tiles);
            },
            |(_, tiles), ()| tiles.clear(),
        )
    });
    group.bench_function("pool_take_recycled", |bench| {
        let mut state = (TilePool::default(), Vec::with_capacity(TILES));
        stock(&mut state, TILE);
        iter_cycle(
            bench,
            state,
            |_| (),
            |(pool, tiles), ()| {
                take_buffers(pool, tiles, TILE);
                black_box(&tiles);
            },
            |(pool, tiles), ()| recycle_buffers(pool, tiles),
        )
    });
    group.bench_function("pool_take_resized", |bench| {
        let mut state = (TilePool::default(), Vec::with_capacity(TILES));
        stock(&mut state, EDGE_HEIGHT);
        iter_cycle(
            bench,
            state,
            |_| (),
            |(pool, tiles), ()| {
                take_buffers(pool, tiles, TILE);
                black_box(&tiles);
            },
            |state, ()| {
                recycle_buffers(&mut state.0, &mut state.1);
                stock(state, EDGE_HEIGHT);
            },
        )
    });

    group.throughput(Throughput::Elements(pixel_count()));
    let merge_cases = [
        ("merge_tile", None),
        ("merge_tile_depth", Some(LoadOp::Clear(1.0))),
    ];
    for (name, depth_load_op) in merge_cases {
        group.bench_function(name, |bench| {
            let target = match depth_load_op {
                Some(_) => RenderTarget::new(TARGET_EXTENT).with_depth(),
                None => RenderTarget::new(TARGET_EXTENT),
            };
            let renderer = Renderer::new().expect("the available parallelism is known");
            iter_cycle(
                bench,
                (renderer, target),
                |(renderer, target)| {
                    let binner = TileBinner::new(
                        target,
                        &LoadOp::Clear(Colour::BLACK),
                        &depth_load_op,
                        renderer.tile_pool_mut(),
                    );
                    binner
                        .tiles
                        .into_iter()
                        .map(render_tile)
                        .collect::<Vec<_>>()
                },
                |(renderer, target), mut results| {
                    for result in results.drain(..) {
                        renderer.merge_tile_result(result, target);
                    }
                    black_box(&*target);
                    results
                },
                |_, _results| (),
            )
        });
    }

    group.finish();
}

/// Where a tile's buffers come from.
#[derive(Clone, Copy)]
enum Source {
    Fresh,
    Recycled,
}

fn pixel_count() -> u64 {
    (TARGET.max_x * TARGET.max_y) as u64
}

/// Returns a binner's buffers to `pool` in reverse, so the next binner's tile `i` takes the buffers
/// tile `i` held, with the same size.
fn recycle(binner: TileBinner, pool: &mut TilePool) {
    for tile in binner.tiles.into_iter().rev() {
        pool.recycle(tile.framebuffer, tile.depthbuffer);
    }
}

/// The buffers of one tile with a depth buffer.
type Buffers = (FrameBuffer, DepthBuffer);

/// Takes the buffers of `TILES` tiles of `height` rows from `pool`.
fn take_buffers(pool: &mut TilePool, tiles: &mut Vec<Buffers>, height: usize) {
    for _ in 0..TILES {
        tiles.push((
            pool.take_framebuffer(TILE, height),
            pool.take_depthbuffer(TILE, height),
        ));
    }
}

fn recycle_buffers(pool: &mut TilePool, tiles: &mut Vec<Buffers>) {
    for (framebuffer, depthbuffer) in tiles.drain(..) {
        pool.recycle(framebuffer, Some(depthbuffer));
    }
}

/// Fills an empty pool with `TILES` buffer pairs whose current size is `height` rows, and whose
/// allocations hold full tiles.
fn stock((pool, tiles): &mut (TilePool, Vec<Buffers>), height: usize) {
    take_buffers(pool, tiles, TILE);
    recycle_buffers(pool, tiles);
    take_buffers(pool, tiles, height);
    recycle_buffers(pool, tiles);
}

/// Times one iteration at a time, so that each can be prepared before the clock starts and
/// restored after it stops. `prepare` and `restore` see `state`; `measured` is the timed work.
///
/// Used where an iteration changes what the next one starts from, and rebuilding that outside the
/// timing is cheaper than a fresh state each time. One `Instant::now` pair is read per iteration,
/// and `Instant` ticks at about 100 ns on Windows, so a few percent of bias remains for the
/// shortest case that uses it, `pool_take_recycled` at about 1.7 µs. Issue #35 revisits this.
fn iter_cycle<T, P, M>(
    bench: &mut Bencher<'_, WallTime>,
    mut state: T,
    mut prepare: impl FnMut(&mut T) -> P,
    mut measured: impl FnMut(&mut T, P) -> M,
    mut restore: impl FnMut(&mut T, M),
) {
    bench.iter_custom(|iterations| {
        let mut elapsed = Duration::ZERO;
        for _ in 0..iterations {
            let input = prepare(&mut state);
            let start = Instant::now();
            let output = measured(&mut state, input);
            black_box(&output);
            elapsed += start.elapsed();
            restore(&mut state, output);
        }
        elapsed
    })
}

criterion_group!(
    benches,
    benchmark_triangle_bounds,
    benchmark_bin_command,
    benchmark_tiles
);
