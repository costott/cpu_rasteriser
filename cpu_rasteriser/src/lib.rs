// The `rasteriser_macros` derives emit `::cpu_rasteriser` paths; this lets the unit tests use them.
#[cfg(test)]
extern crate self as cpu_rasteriser;

pub mod colour;
pub mod depthbuffer;
pub mod extent;
pub mod framebuffer;
pub mod graphics;
mod grid;
pub mod maths;
pub mod prelude;
pub mod renderer;
pub mod viewport;

pub use wide;
