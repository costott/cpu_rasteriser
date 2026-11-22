//! The engine's commonly used items, for a glob import.
//!
//! It re-exports the application framework, the backends, the input types, the components
//! (cameras and their controllers, lighting, materials, meshes, models, and textures) and the
//! loaders. Shaders, pipelines, and render passes come from `cpu_rasteriser`, so an app usually
//! imports both preludes.
//!
//! # Examples
//!
//! Building a model uses items from both preludes:
//!
//! ```
//! use cpu_rasteriser::prelude::*;
//! use engine::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cube = Mesh::cube(Colour::WHITE, None);
//! let transform = ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE);
//! let model = Model::new(vec![cube], Vec::new(), transform)?;
//! assert_eq!(model.meshes.len(), 1);
//! # Ok(())
//! # }
//! ```

pub use crate::app::*;
pub use crate::components::camera::*;
pub use crate::components::lighting::*;
pub use crate::components::material::*;
pub use crate::components::mesh::*;
pub use crate::components::model::*;
pub use crate::components::texture::*;
pub use crate::engine::*;
pub use crate::input::*;
pub use crate::loaders::*;
