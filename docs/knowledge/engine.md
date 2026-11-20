---
sources:
  - engine/src/app.rs
  - engine/src/engine.rs
  - engine/src/input.rs
  - engine/src/components/camera.rs
  - engine/src/components/model.rs
  - engine/src/components/mesh.rs
  - engine/src/components/material.rs
  - engine/src/components/texture.rs
  - engine/src/loaders/obj.rs
  - engine/src/loaders/mtl.rs
verified: 4e5fb07
---

# Engine crate

The user-facing API is described in `engine/README.md`. This document covers internals.

## Backends (`engine.rs`)

- **Common interface:** `EngineBackend::run(app)` is implemented by `WinitEngine` (winit with a softbuffer surface) and `MinifbEngine`. Each backend owns the `Renderer` and a presentation `RenderTarget` that has a depth buffer.
- **Frame flow:** compute `dt`, call `app.update(dt)`, then `app.render(&mut RenderContext)`, then `presentation_target.write_pixels_u32` packs the pixels straight into the window buffer without allocating.
- **Winit specifics:**
  - The presentation target starts at 640×360 whatever the window attributes say. It is resized on `resumed` and `Resized`. Zero-size resizes are ignored.
  - Rendering is driven by `RedrawRequested`, and `about_to_wait` requests the next redraw.
  - Raw `DeviceEvent::MouseMotion` supplies `InputEvent::MouseMotion`.
- **Minifb specifics:**
  - The loop runs at 60 FPS (`set_target_fps(60)`).
  - Input comes from polling. Key and button state is compared with the previous frame to synthesise press and release events.
  - The resize check runs every frame.
- **Input mapping:** both backends map only a fixed set of keys (arrows, WASD, Space, LeftShift, Escape) and mouse buttons to `InputKey`/`MouseButton`. Adding a key needs a change in both mappers.
- **Window state:** `Application::window_state` is applied every frame (cursor visibility and grab). Minifb ignores cursor grab.

## `RenderContext` and `PresentedFrame` (`app.rs`)

`Application::render` must return a `PresentedFrame`. Only `PresentationPass::finish` can construct one (it has a private field). This is a compile-time proof that the presentation target was rendered.

The default `render` implementation does construct one without rendering. **Inferred:** it is a convenience for apps that don't render.

## Models, meshes and materials

- **`Model<V>`:** holds meshes, materials and a `ModelTransform` (translation × rotation × scale).
- **Validation:** `Model::new` checks that each mesh's material index is in range.
- **Drawing:** `draw_calls` maps each mesh to a `DrawCall`, with per-mesh fragment uniforms from a closure. `draw_to_render_pass(_simd)` submits those draw calls and clones the vertex uniforms for each mesh.
- **Shading:** the engine ships no shaders. Examples define their own (Gouraud, Phong, post-processing, bloom).

## Textures (`texture.rs`)

- **Storage:** `Texture` holds `Arc<[Colour]>` in image row order. Row 0 is the top of the image.
- **UV mapping:** `TextureSampler` maps `v` to a row with `(1 − v)`, so the UV origin is the bottom-left.
- **Wrap modes:** `Repeat` uses `fract()`, which doesn't wrap negative coordinates ([#24](https://github.com/costott/cpu_rasteriser/issues/24)), and `Clamp` clamps to [0, 1].
- **Scalar sampling:** `sample` dispatches on wrap and filter mode at run time. The specialised `sample_linear_clamp` and `sample_nearest_clamp` skip that dispatch, and `sample_linear_clamp` reads texels unchecked.
- **SIMD sampling:** `sample_linear_clamp_simd` and `sample_nearest_clamp_simd` clamp, then gather 8 texels with scalar unchecked loads. The bilinear variant also clamps its `x1`/`y1` to the edge, so every index stays in bounds.
- **Render targets as textures:** `Texture::from_render_target` copies a target's pixels into a texture, and `Texture::update_from` refills it with `write_pixels`. This is how multipass post-processing samples an earlier pass. `update_from` overwrites the storage in place only when it is the right size and `Arc::get_mut` shows no clone or sampler shares it, so a live sampler's pixels never change; otherwise new storage replaces it. Reuse across frames therefore needs one app-held `Texture` per place a target is sampled, with that place's samplers dropped before the next update: bloom samples `bloom_half` twice per frame and holds two. Engine-owned storage, so apps hold nothing, is [#32](https://github.com/costott/cpu_rasteriser/issues/32).

## Loaders

`load_obj` and `load_mtl` are hand-written parsers. They produce `Model<ObjVertex>` and `Vec<Material>`, with typed errors (`ObjError`, `MtlError`).
