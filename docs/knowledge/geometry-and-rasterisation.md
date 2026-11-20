---
sources:
  - cpu_rasteriser/src/graphics/geometry_processing.rs
  - cpu_rasteriser/src/graphics/clipping.rs
  - cpu_rasteriser/src/graphics/triangle.rs
  - cpu_rasteriser/src/graphics/vertex.rs
  - cpu_rasteriser/src/graphics/interpolate.rs
  - cpu_rasteriser/src/graphics/circle.rs
  - cpu_rasteriser/src/graphics/line.rs
  - cpu_rasteriser/src/viewport.rs
  - cpu_rasteriser/src/maths/mat4.rs
  - cpu_rasteriser/src/maths/vec2.rs
  - cpu_rasteriser/src/maths/vec3.rs
  - cpu_rasteriser/src/maths/vec4.rs
  - cpu_rasteriser/src/colour.rs
  - cpu_rasteriser/src/renderer.rs
  - cpu_rasteriser/src/depthbuffer.rs
verified: 530e0dd
---

# Geometry processing and rasterisation

## Coordinate conventions

- **Matrices:** `Mat4` is stored row-major and multiplies column vectors (`M * v`). `Mat4::perspective` and `orthographic` are OpenGL-style. The view space is right-handed, looking down −z. Clip-space z maps to NDC z in [−1, 1].
- **Vertex types:** `ClipVertex` is the output of the vertex shader (`Vec4` clip position and varyings). `RasterVertex` is screen space: `position` in pixels, `depth`, `inv_w`, and the varyings pre-divided by w.
- **Viewport mapping** (`GeometryProcessor::clip_to_screen`): `x = vp.x + (ndc.x + 1)/2 · vp.width`, and `y = vp.y + (1 − ndc.y)/2 · vp.height`. Screen y points down.
- **Depth:** the value stored is NDC z. The depth test passes when `fragment.depth < stored` (scalar `>=` rejects; SIMD uses `simd_lt`). Depth buffers start at 1.0, so clear depth with `LoadOp::Clear(1.0)`.

## Stages (`GeometryProcessor::process_triangle`)

1. **Vertex stage:** the vertex shader runs once per triangle corner. Indexed vertices are cloned per triangle (`Primitive::triangles`), so shared vertices are shaded again for every triangle that uses them. There is no post-transform cache.
2. **Clipping** (`clip_triangle`): Sutherland–Hodgman in homogeneous clip space against six planes, in the order left, right, top, bottom, near, far. Signed distances are `w ± x`, `w ± y` and `w ± z`, so the near plane is `z = −w`.
   - Each plane's output polygon is re-triangulated before the next plane (`triangulate`, a fan for 4 vertices). Clipping one triangle against one plane yields 0, 3 or 4 vertices, so other counts are discarded.
   - Varyings are interpolated linearly in clip space, which is correct before the perspective divide.
3. **To screen:** divide by w, apply the viewport mapping, and store `inv_w = 1/w` and `varyings · inv_w`.
4. **Culling:** with `CullingMode::BackFace`, triangles whose screen-space cross product `(b−a)×(c−a) ≥ 0` are removed. Because screen y is flipped, **counter-clockwise in NDC is front-facing**. Culling happens after clipping.
   - Zero-area triangles are removed only when back-face culling is on (the `>= 0` test).
   - With `CullingMode::None` they reach the rasteriser, which emits no fragments for zero-height triangles and none for zero-width spans.

## Scanline rasterisation (`Triangle2D::rasterise_segment`)

- **Approach:** vertices are sorted by y. A long edge (v0→v2) and a short edge (v0→v1, then v1→v2) are stepped incrementally (`Edge::step`). Each scanline interpolates x, depth, `inv_w` and the varyings incrementally from left to right.
- **Coverage is tested at pixel centres with half-open ranges.** Rows run from `ceil(y_top − 0.5)` to `ceil(y_bottom − 0.5)`, exclusive. Columns run from `ceil(x_left − 0.5)` to `ceil(x_right − 0.5)`, exclusive. Shared edges are therefore drawn exactly once. `rasterise_uses_pixel_centres_for_coverage` pins this.
- **Bounds:** `bounds: Rect` restricts output to one tile, with `max_x`/`max_y` exclusive. Fragments carry absolute screen positions. The raster command converts them to tile-local coordinates.
- **Perspective correction:** varyings are pre-divided by w. They are interpolated affinely along with `inv_w`, and each fragment multiplies by `1/inv_w`. Depth (NDC z) is affine in screen space, so it is interpolated without correction.

## Interpolation trait

`Interpolate` (`interpolate`, `difference`, `scale`, `add_scaled`) is what the edge and span steppers need. The manual implementation for `f32` is the only one written by hand. `Vec2`, `Vec3` and `Vec4` use `#[derive(Interpolate)]`. `Colour` does **not** implement `Interpolate`, so shaders pass colours as `Vec4` varyings.

The derive also generates `Clone`. A type deriving `Interpolate` must not derive `Clone` as well.

## Other primitives

- `PrimitiveMode::TRIANGLES` is the only topology.
- `graphics::circle` (midpoint circle) writes straight into a `FrameBuffer` and isn't part of the pipeline.
- `graphics::line` is entirely commented out ([#28](https://github.com/costott/cpu_rasteriser/issues/28)).
