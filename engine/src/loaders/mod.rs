//! Loaders for Wavefront OBJ models and their MTL material libraries.
//!
//! [`load_obj`] reads a model, with the material libraries and textures it names, into a
//! [`Model<ObjVertex>`](crate::components::model::Model). [`load_mtl`] reads a material library
//! on its own. Both read the subset of the formats their documentation lists. Errors report the
//! line of the failing statement and chain their causes through
//! [`source`](std::error::Error::source).
//!
//! Relative paths in a file are resolved against the directory of the file that names them.

pub use crate::loaders::mtl::*;
pub use crate::loaders::obj::*;

pub mod mtl;
pub mod obj;
mod values;
