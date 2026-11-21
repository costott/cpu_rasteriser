// The `rasteriser_macros` derives emit `::cpu_rasteriser` paths; this lets the unit tests use them.
#[cfg(test)]
extern crate self as cpu_rasteriser;

pub use colour::{Colour, ColourSimd};
pub use extent::Extent;
pub use graphics::fragment_shader::{FragmentShader, FragmentShaderSimd};
pub use graphics::interpolate::{Interpolate, SimdInterpolate};
pub use graphics::triangle::Triangle3D;
pub use graphics::vertex::ObjVertex;
pub use graphics::vertex_shader::VertexShader;
pub use maths::{Mat4, Vec2, Vec3, Vec3Simd, Vec4};
pub use rasteriser_macros::{Interpolate, SimdInterpolate};
pub use renderer::{
    BlendFactor, BlendOp, BlendState, CullingMode, DepthState, DrawCall, LoadOp, Pipeline,
    PrimitiveMode, RenderPass, RenderPassDescriptor, RenderTarget, Renderer, SimdPipeline,
};
pub use viewport::Viewport;
pub use wide;

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

/// Internal types re-exported for the crate's own benchmarks. Not part of the public API.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod internals {
    pub use crate::depthbuffer::DepthBuffer;
    pub use crate::framebuffer::FrameBuffer;
    pub use crate::graphics::triangle::Triangle2D;
    pub use crate::graphics::vertex::RasterVertex;
    pub use crate::renderer::Rect;
}
