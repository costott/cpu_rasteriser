---
sources:
  - cpu_rasteriser/src/framebuffer.rs
  - cpu_rasteriser/src/depthbuffer.rs
  - cpu_rasteriser/src/colour.rs
  - cpu_rasteriser/src/renderer.rs
verified: 530e0dd
---

# Buffers, colour and blending

## `FrameBuffer` and `DepthBuffer`

- **Layout:** both are row-major `Vec`s with a padded stride (see `simd.md` for the invariant). `FrameBuffer` stores `Colour`, so its layout is an array of structs (AoS). `DepthBuffer` stores `f32`.
- **Duplication:** the two are structurally duplicated: stride computation, `index`/`index_unchecked`, the `*8_unchecked` accessors, and `resize`.
- **Planned layout change:** `FrameBuffer` carries a TODO to move to a struct-of-arrays layout (separate channel `Vec`s) for SIMD loads and stores. A shared abstraction must not block that.
- **Accessor families:**
  - Bounds-checked `get_pixel`/`set_pixel`. These ignore out-of-range writes and return `None` on out-of-range reads.
  - `index_unchecked`, plus `unsafe` single-element and 8-wide accessors.
  - `DepthBuffer::get`/`set_depth` aren't bounds-checked against the logical size in release builds. Only `index`'s `debug_assert`s check it, so release builds panic only beyond the allocation.
- **Positions:** positions are passed as `Vec2` and truncated with `as usize`.
- **Dead accessors:** `set8_{r,g,b}_unchecked(_with_mask)` and `get8_{r,g,b}_unchecked` are used nowhere outside `framebuffer.rs`. The `g` and `b` setters write into the red channel ([#23](https://github.com/costott/cpu_rasteriser/issues/23)).
- **Resizing:** `resize` keeps the old allocation and recomputes the stride. Existing contents become meaningless.
- **Reading pixels back:** `pixels_compact` and `pixels_u32` strip the padding and allocate a new `Vec` on every call.

## `RenderTarget`

A `RenderTarget` is a `FrameBuffer` plus an optional `DepthBuffer` (`with_depth()`). It is the only public way to get pixels out.

## `Colour`

- **Format:** four `f32` channels, unclamped. HDR values are deliberate (commit `55bfb21`). Clamping happens only in `to_u32`, which packs `0xAARRGGBB` for minifb and softbuffer.
- **Arithmetic keeps the left-hand operand's alpha.** `Add`, `Sub`, `Mul<Colour>`, `Mul<f32>` and `Div<f32>` combine only r, g and b, and the result takes `self.a`. The comment in `colour.rs` says this is intended for blending. `scale_all` and `lerp` do touch alpha.
- **Byte order:** `from_u32` reads red from the **low** byte, so it is not the inverse of `to_u32` ([#22](https://github.com/costott/cpu_rasteriser/issues/22)).
- **`ColourSimd`:** a struct of four `f32x8` (r, g, b, a), with `splat` and `lerp`.

## Blending (`BlendState`, `BlendFactor`, `BlendOp`)

- **Formula:** `result = op(src · src_factor, dst · dst_factor)`. There are 10 factors and 5 operations. A pipeline without a blend state overwrites the destination.
- **Presets:** `BlendState::ALPHA_BLEND` and `ADDITIVE`. `ALPHA_BLEND` carries the comment `TODO: currently broken`.
- **The two paths disagree on alpha:**
  - Scalar `apply` uses `Colour` arithmetic, so its factors and the `Add`, `Subtract` and `ReverseSubtract` results follow the left-hand-alpha rule above. `Colour::WHITE - src` yields alpha 1 rather than `1 − src.a`. `Min` and `Max` take all four channels component-wise.
  - `apply_simd` computes all four channels arithmetically.
  - `blend_factor_resolve` and `blend_state_reverse_subtract` pin the arithmetic interpretation and currently fail against the scalar path ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).
  - **Inferred:** this alpha handling is why `ALPHA_BLEND` is marked broken.
- **Destination with blending:** the scalar path reads the tile's current pixel (`unwrap_or(Colour::BLACK)`). The SIMD path reads 8 pixels unchecked.
