//! The building blocks of a scene: cameras, lights, materials, meshes, models, textures,
//! and a frame timer.
//!
//! Everything here except [`timer`] is re-exported by the [`prelude`](crate::prelude). The engine
//! ships no shaders, so these types hold data for the application's own shaders to interpret:
//! a [`Camera`](camera::Camera) supplies the view and projection matrices, a
//! [`Model`](model::Model) the geometry and its [`Material`](material::Material)s, and a
//! [`TextureSampler`](texture::TextureSampler) the texels.

pub mod camera;
pub mod lighting;
pub mod material;
pub mod mesh;
pub mod model;
pub mod texture;
pub mod timer;
