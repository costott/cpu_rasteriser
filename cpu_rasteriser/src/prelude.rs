//! Re-exports the crate's types, traits, and derive macros for a single glob import.
//!
//! The prelude holds everything the crate root exports except the [`wide`] re-export. Importing
//! it brings both the [`Interpolate`] and [`SimdInterpolate`] traits and the derive macros of the
//! same names into scope.
//!
//! # Examples
//!
//! ```
//! use cpu_rasteriser::prelude::*;
//!
//! let target = RenderTarget::new(Extent::new(320, 180)).with_depth();
//! # let _ = target;
//! ```

pub use crate::{
    BlendFactor, BlendOp, BlendState, Colour, ColourSimd, CullingMode, DepthState, DrawCall,
    Extent, FragmentShader, FragmentShaderSimd, Interpolate, LoadOp, Mat4, ObjVertex, Pipeline,
    PrimitiveMode, RenderPass, RenderPassDescriptor, RenderTarget, Renderer, SimdInterpolate,
    SimdPipeline, Triangle3D, Vec2, Vec3, Vec3Simd, Vec4, VertexShader, Viewport,
};
