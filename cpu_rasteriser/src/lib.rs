#![doc = include_str!("../README.md")]
#![warn(missing_debug_implementations, missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

// The `rasteriser_macros` derives emit `::cpu_rasteriser` paths; this lets the unit tests use them.
#[cfg(test)]
extern crate self as cpu_rasteriser;

pub use rasteriser_macros::{Interpolate, SimdInterpolate};
pub use wide;

pub use crate::colour::{Colour, ColourSimd};
pub use crate::extent::Extent;
pub use crate::graphics::fragment_shader::{FragmentShader, FragmentShaderSimd};
pub use crate::graphics::interpolate::{Interpolate, SimdInterpolate};
pub use crate::graphics::triangle::Triangle3D;
pub use crate::graphics::vertex::ObjVertex;
pub use crate::graphics::vertex_shader::VertexShader;
pub use crate::maths::{Mat4, Vec2, Vec3, Vec3Simd, Vec4};
pub use crate::renderer::{
    BlendFactor, BlendOp, BlendState, CullingMode, DepthState, DrawCall, LoadOp, Pipeline,
    PrimitiveMode, RenderPass, RenderPassDescriptor, RenderTarget, Renderer, SimdPipeline,
};
pub use crate::viewport::Viewport;

pub mod prelude;

mod colour;
mod depthbuffer;
mod extent;
mod framebuffer;
mod graphics;
mod grid;
mod maths;
mod renderer;
mod viewport;

/// Internal types re-exported for the crate's benchmarks.
///
/// Compiled only with the `internals` feature. This is not public API: its contents can change or
/// disappear in any release.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod internals {
    pub use crate::depthbuffer::DepthBuffer;
    pub use crate::framebuffer::FrameBuffer;
    pub use crate::graphics::clipping::clip_triangle;
    pub use crate::graphics::geometry_processing::GeometryProcessor;
    pub use crate::graphics::triangle::{Triangle2D, TriangleClip};
    pub use crate::graphics::vertex::{ClipVertex, RasterVertex};
    pub use crate::renderer::{
        RasterCommand, Rect, Tile, TileBinner, TilePool, TileResult, render_tile,
    };
}
