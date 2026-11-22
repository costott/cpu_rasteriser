#![warn(missing_debug_implementations)]

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

#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod internals {
    pub use crate::depthbuffer::DepthBuffer;
    pub use crate::framebuffer::FrameBuffer;
    pub use crate::graphics::triangle::Triangle2D;
    pub use crate::graphics::vertex::RasterVertex;
    pub use crate::renderer::Rect;
}
