---
sources:
  - cpu_rasteriser/src/renderer.rs
  - cpu_rasteriser/Cargo.toml
verified: 530e0dd
---

# Tiling and threading

All of this lives in `renderer.rs` (`TileBinner`, `Tile`, `render_tile`, `Renderer::render_tiles`, `Renderer::merge_tile`).

## Tiles

- **Layout:** `TileBinner::TILE_SIZE = 64`. Tiles cover the **whole render target**, not the viewport. Edge tiles are smaller. Each `Rect` bound has an exclusive max.
- **Ownership:** each `Tile` owns its own `FrameBuffer` and optional `DepthBuffer`, with the tile's dimensions and its own padded stride. Workers never share mutable memory. This is why rasterisation needs no locks.
- **Allocation:** tiles are created in `begin_render_pass`. Every pass allocates every tile buffer afresh.
- **Load operations:** these seed the tiles at that point.
  - `LoadOp::Clear` fills the tile.
  - `LoadOp::Load` copies the target's pixels and depth into the tile pixel by pixel.
  - With `depth_load_op: None`, tiles have no depth buffer. Any pipeline that tests or writes depth will then panic inside the worker thread (`expect` in the raster command; [#27](https://github.com/costott/cpu_rasteriser/issues/27)).
  - A `depth_load_op` of `Some(_)` on a target built without `with_depth()` panics earlier, on the calling thread, in `begin_render_pass` (`Tile::new`).

## Binning (`TileBinner::bin_command`)

- **Selection:** the command's screen bounding box selects a range of tiles. `RasterCommand::intersects` then filters them, but that test is bounding box against rectangle, inclusive, and conservative. A large thin diagonal triangle is therefore binned into tiles it doesn't touch, which then do empty scanline work.
- **Sharing:** commands are `Arc<dyn RasterCommand>`, shared across tiles, not copied into each one.
- **Ordering invariant:** binning preserves submission order within each tile, and a tile is rasterised by exactly one job, sequentially. **Draw order per pixel therefore equals submission order.** Blending and depth results are deterministic regardless of thread scheduling. Any change to scheduling must keep this.

## Execution (`Renderer::render_tiles`)

- **Pool:** `Renderer::new` creates a `threadpool::ThreadPool` with `available_parallelism()` workers. It lives as long as the renderer.
- **Jobs:** each tile is moved into its own job. A job runs `render_tile` and sends a `TileResult` back over an `mpsc` channel.
- **Merging:** the calling thread receives results in completion order. It merges each one into the target pixel by pixel with `set_pixel`/`set_depth`, which are per-pixel calls and not row copies (see `buffers-and-colour.md` for how far each is bounds-checked). The tiles don't overlap, so merge order doesn't matter.
- **Waiting:** `thread_pool.join()` then waits for the pool to go idle.

## Performance characteristics

Only rasterisation and fragment shading run in parallel. The calling thread does the rest serially, on every pass:
- geometry processing for every draw;
- allocating every tile buffer;
- per-pixel load and merge copies.

Reading the result back with `pixels_u32` also allocates on every call.
