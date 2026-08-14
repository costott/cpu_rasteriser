use crate::prelude::*;

/// A trait for vertex shaders, which process vertices before rasterisation.
///
/// # Example
/// ```ignore
/// #[derive(Clone)]
/// struct Vertex {
///     position: Vec3,
///     colour: Colour,
///     normal: Vec3,
/// }
///
/// struct Uniforms {
///    pub model_matrix: Mat4,
///    pub view_matrix: Mat4,
///    pub projection_matrix: Mat4,
/// }
///
/// #[derive(Interpolate)]
/// struct Varyings {
///     pub world_position: Vec3,
///     pub colour: Colour,
///     pub normal: Vec3,
/// }
///     
/// struct BasicVertexShader;
/// impl VertexShader for BasicVertexShader {
///     type Vertex = Vertex;
///     type Uniforms = Uniforms;
///     type Varyings = Varyings;
///
///     fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
///         let world_position = uniforms.model_matrix * vertex.position.to_point4();
///         let normal_matrix = uniforms.model_matrix.inverse().transpose();
///
///         let view_position = uniforms.view_matrix * world_position;
///         let clip_position = uniforms.projection_matrix * view_position;
///
///         let varyings = Varyings {
///             world_position: world_position.homogenize_to_vec3(),
///             colour: vertex.colour,
///             normal: (normal_matrix * vertex.normal.to_direction4())
///                 .xyz()
///                 .normalise(),
///         };
///
///         (clip_position, varyings)
///     }
/// }
/// ```
pub trait VertexShader: Send + Sync + 'static {
    type Vertex: Clone;
    type Uniforms: Send + Sync + 'static;
    type Varyings: Interpolate + Send + Sync + 'static;

    /// Processes a world-space vertex, before projection and clipping.
    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings);
}
