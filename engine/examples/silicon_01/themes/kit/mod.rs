//! The shared toolkit themes are built from.
//!
//! # Building a 3D world
//!
//! A world is a handful of [`Shape`](shape::Shape)s drawn with kit pipelines, a backdrop and some
//! particles. The usual recipe, as `forest`, `reef` and `earth` follow it:
//!
//! 1. **Geometry** ([`shape`], [`terrain`], [`scatter`]). Generate low-poly shapes in code
//!    ([`Shape::lathe`](shape::Shape::lathe), `cone`, `icosphere`, `limb`, `ribbon`, ...), colour
//!    them per vertex (`gradient_by_height`, `colour_by`, `facet_variation`) and give them sway
//!    weights (`sway_by_height`). Lay out the ground with [`terrain::terrain`]: a polar heightfield
//!    whose rings widen with distance, so it reaches the horizon cheaply. Place props with
//!    [`scatter::scatter`] (seeded dart throwing, with a callback to thin or exclude areas) and
//!    bake the copies into one mesh with [`scatter::Batch`]. Keep the ground's `height(x, z)` as
//!    a function so props can stand on it.
//! 2. **Drawing** ([`draw`], [`light`]). [`draw::KitVertexShader`] draws a shape in world space and
//!    animates it with a [`draw::Motion`]: `Sway` bends vertices by their sway weight (trees,
//!    kelp), `Orbit` circles each instance by its seed (a shoal), `OnChip` poses things attached
//!    to the chip. Pair it with a fragment shader on [`draw::KitVaryings`]: [`draw::ToonShader`]
//!    for plain [`light::Lighting`] (a sun, sky and ground ambient, optional banding and rim,
//!    height-dependent [`light::Fog`]), [`draw::VariedToonShader`] to vary each instance's colour
//!    by seed, or your own, reading the varyings through the accessors on
//!    [`draw::KitVaryingsSimd`]. Build the pipeline with [`draw::pipeline`], and draw each shape with
//!    [`Shape::draw`](shape::Shape::draw). Use `CullingMode::None` for one-sided foliage.
//! 3. **Shading detail** ([`noise`], [`shadow`]). Analytic SIMD noise (value, fractal, cellular
//!    for caustics and cells) and its scalar twins for building geometry. [`shadow::GroundShadow`]
//!    bakes a static scene's long sun shadows into a texture once, for one lookup per ground pixel.
//! 4. **Backdrops** ([`sky`]). [`sky::SkyShader`] is a full-screen sky: gradient, sun disc and glow,
//!    drifting clouds, stars. Draw it after the opaque geometry with `DepthState::READ_ONLY` so it
//!    fills only uncovered pixels. [`sky::stars`] and [`sky::sun_glow`] are free functions for
//!    custom backdrops (water, space).
//! 5. **Particles** ([`particles`]). [`particles::Particles`] animates a field of glowing specks
//!    (pollen, bubbles, plankton) entirely in the vertex shader. Draw after the backdrop.
//! 6. **The chip** ([`dress`]). Restyle it with a [`Look`](crate::chip::Look), add extras built in
//!    the chip's own space ([`dress::rim_point`] picks spots on its package), and let it shade the
//!    ground with [`dress::chip_shadow`].
//!
//! Draw order within a stage: opaque geometry (writes depth), the chip, the backdrop, then
//! particles.

pub mod draw;
pub mod dress;
pub mod light;
pub mod noise;
pub mod particles;
pub mod post;
pub mod random;
pub mod scatter;
pub mod shadow;
pub mod shape;
pub mod sky;
pub mod terrain;
