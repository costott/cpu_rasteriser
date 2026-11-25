# CPU rasteriser

A software implementation of a modern GPU-style graphics pipeline in Rust. It is a Cargo workspace of three crates:

| Crate | Role |
|---|---|
| `cpu_rasteriser` | The low-level rendering API: pipelines, shaders, draw calls, render passes and render targets. This is the core library. |
| `rasteriser_macros` | `#[derive(Interpolate)]` and `#[derive(SimdInterpolate)]` for varying structs. |
| `engine` | Built on `cpu_rasteriser`: the application framework (winit/minifb backends), cameras, models, materials, textures, OBJ/MTL loading. |

The public API, its features and the examples are documented in `README.md`, `cpu_rasteriser/README.md` and `engine/README.md`. Don't duplicate them here or in the knowledge base.

## Knowledge base

`docs/knowledge/` is the technical reference for internals. Read the documents relevant to the area you're about to change, not all of them.

| Document | Read when working on |
|---|---|
| `architecture.md` | Anything non-trivial. It explains how the crates and the render-pass lifecycle fit together. |
| `geometry-and-rasterisation.md` | Vertex processing, clipping, culling, viewport mapping, scanline rasterisation, interpolation, depth conventions |
| `tiling-and-threading.md` | Render passes, tile binning, the thread pool, tile merging, load operations |
| `simd.md` | The SIMD pipeline, `FragmentShaderSimd`, `SimdInterpolate` and its derive macro, unchecked 8-wide buffer access |
| `buffers-and-colour.md` | `FrameBuffer`, `DepthBuffer`, `RenderTarget`, `Colour`/`ColourSimd`, blending |
| `engine.md` | The engine crate: backends, presentation, textures, models, loaders |
| `testing-and-benchmarks.md` | Tests, benches, build flags, performance measurement |
| `decisions.md` | Before revisiting an architectural choice. It records why things are the way they are. |

Every document has `sources` (the files it describes) and `verified` (the commit it was last checked against) in its frontmatter.
- Find out whether a document may be stale with `git log --oneline <verified>..HEAD -- <sources>`. Commits that also update the document are the sync itself and don't count.
- When a change alters anything a document describes, update that document in the same commit, and set `verified` to the base commit of the change.
- Documents describe the current design and behaviour of `master` and the reasoning behind it, never what a change replaced or how things used to be. Bugs live in GitHub issues: where behaviour departs from intent, link the issue in one line instead of describing the bug, and never link an issue the change resolves, its own ticket included. Never refer to unmerged branches, local work or work in progress.
- Documents record invariants, conventions and reasoning. They refer to symbols rather than line numbers, and they never restate code.
- Claims are marked **Inferred** where they couldn't be confirmed from the code or history.

## Commands

Run these from the workspace root.

| Task | Command |
|---|---|
| Format | `cargo fmt --all` |
| Lint | `cargo clippy --workspace --all-targets --features cpu_rasteriser/internals -- -D warnings` (the workspace is clippy-clean and must stay so; the feature includes the benches) |
| Test | `cargo test --workspace --no-fail-fast` |
| Bench | `cargo bench -p cpu_rasteriser --features internals --bench micro` (microbenchmarks), `cargo bench -p cpu_rasteriser --bench frame` (whole frames, with output checks), and `-p engine --bench texture_sampling`; add a Criterion filter after `--` to select cases |
| Example | `cargo run --release -p <crate> --example <name>` (see the READMEs) |

`target-cpu=native` applies to every build in the workspace, whichever directory cargo runs from; see `testing-and-benchmarks.md`. Compare performance only between builds with the same build configuration.

**Test baseline**, so new failures can be told apart from old ones: every test and doctest in the workspace passes, and none are ignored, except these known failures in `cpu_rasteriser`:
- `graphics::triangle::tests::rasterise_simd_matches_scalar` ([#21](https://github.com/costott/cpu_rasteriser/issues/21)).
- `renderer::tests::blend_factor_resolve` and `renderer::tests::blend_state_reverse_subtract` ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).

Update this list when a known failure is fixed or a new one is accepted.

## Project conventions

- Update the READMEs for important user-facing changes. Keep the examples compiling and in step with the API.
- Breaking public API changes bump the crate versions according to semver, and the commit message mentions the new versions.
- Code that must stay fast is marked `#[inline(always)]` and uses unchecked indexing only behind a documented invariant; see `simd.md`. Every hot-path change needs benchmark or profiling evidence.
- Worktrees need no extra setup. All assets are tracked, and the first build fills that worktree's own `target/`.
