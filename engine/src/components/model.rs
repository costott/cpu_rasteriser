//! Models: meshes with their materials and a transform, and how to draw them.

use cpu_rasteriser::prelude::*;

use crate::prelude::*;

/// A renderable object: meshes, the materials they use, and where the object is in the world.
///
/// Each mesh's [`material_index`](Mesh::material_index) refers to an entry of
/// [`materials`](Model::materials). [`Model::new`] checks this; the fields are public, so code
/// that changes them afterwards must keep it true. [`load_obj`] builds a `Model<ObjVertex>` from
/// a file.
///
/// The meshes are in model space. Drawing doesn't apply [`transform`](Model::transform): pass its
/// [`model_matrix`](ModelTransform::model_matrix) to the vertex shader in the vertex uniforms, as
/// in [`Model::draw_to_render_pass`].
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let red = Material::named("red".to_owned());
/// let model = Model::new(
///     vec![Mesh::cube(Colour::WHITE, Some(0)), Mesh::sphere(0.5, 8, 16, Colour::WHITE, None)],
///     vec![red],
///     ModelTransform::new(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, Vec3::ONE),
/// )?;
///
/// assert_eq!(model.meshes.len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Model<V: Clone> {
    /// The model's meshes, in model space.
    pub meshes: Vec<Mesh<V>>,
    /// The materials the meshes' material indices refer to.
    pub materials: Vec<Material>,
    /// The model's position, rotation, and scale in the world.
    pub transform: ModelTransform,
}
impl<V: Clone> Model<V> {
    /// Creates a model with the given meshes, materials, and transform.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidMaterialIndex`] for the first mesh whose material index is
    /// out of range for `materials`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let transform = ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE);
    /// let error = Model::new(vec![Mesh::cube(Colour::WHITE, Some(1))], Vec::new(), transform)
    ///     .unwrap_err();
    ///
    /// assert!(matches!(
    ///     error,
    ///     ModelError::InvalidMaterialIndex { mesh: 0, material: 1, material_count: 0 }
    /// ));
    /// ```
    pub fn new(
        meshes: Vec<Mesh<V>>,
        materials: Vec<Material>,
        transform: ModelTransform,
    ) -> Result<Self, ModelError> {
        let model = Self {
            meshes,
            materials,
            transform,
        };
        model.validate()?;
        Ok(model)
    }

    fn validate(&self) -> Result<(), ModelError> {
        for (i, mesh) in self.meshes.iter().enumerate() {
            if let Some(material_index) = mesh.material_index {
                if material_index >= self.materials.len() {
                    return Err(ModelError::InvalidMaterialIndex {
                        mesh: i,
                        material: material_index,
                        material_count: self.materials.len(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Returns one [`DrawCall`] per mesh, as a triangle list, with fragment uniforms made by
    /// `make_uniforms` from the mesh.
    ///
    /// Use this to submit the draw calls yourself, for example to pass different vertex uniforms
    /// per mesh; otherwise [`draw_to_render_pass`](Model::draw_to_render_pass) and
    /// [`draw_to_render_pass_simd`](Model::draw_to_render_pass_simd) submit them for you.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let model = Model::new(
    ///     vec![Mesh::cube(Colour::WHITE, Some(0)), Mesh::cube(Colour::WHITE, None)],
    ///     vec![Material::named("red".to_owned())],
    ///     ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
    /// )?;
    ///
    /// // Meshes without a material are drawn grey.
    /// let draw_calls = model.draw_calls(|mesh| match mesh.material_index {
    ///     Some(index) => model.materials[index].diffuse,
    ///     None => Colour::new(0.5, 0.5, 0.5, 1.0),
    /// });
    /// assert_eq!(draw_calls.count(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn draw_calls<U, F>(&self, make_uniforms: F) -> impl Iterator<Item = DrawCall<'_, V, U>>
    where
        F: Fn(&Mesh<V>) -> U,
    {
        self.meshes.iter().map(move |mesh| {
            DrawCall::new(
                &mesh.vertices,
                &mesh.indices,
                PrimitiveMode::TRIANGLES,
                make_uniforms(mesh),
            )
        })
    }

    /// Queues the model's meshes in a render pass, drawn with a scalar [`Pipeline`].
    ///
    /// Each mesh is drawn with a clone of `vertex_uniforms` and fragment uniforms made by
    /// `make_fragment_uniforms` from the mesh, typically from its material. See
    /// [`RenderPass::draw`].
    ///
    /// # Panics
    ///
    /// Panics if the pipeline tests or writes depth and the render pass has no depth buffer, as
    /// [`RenderPass::draw`] does.
    ///
    /// # Examples
    ///
    /// Draw each mesh in its material's diffuse colour, with the model matrix as the vertex
    /// uniforms:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// struct TransformShader;
    /// impl VertexShader for TransformShader {
    ///     type Vertex = ObjVertex;
    ///     type Uniforms = Mat4;
    ///     type Varyings = f32;
    ///
    ///     fn shade(&self, vertex: ObjVertex, model_matrix: &Mat4) -> (Vec4, f32) {
    ///         (*model_matrix * vertex.position.to_point4(), 0.0)
    ///     }
    /// }
    ///
    /// struct FlatShader;
    /// impl FragmentShader<f32> for FlatShader {
    ///     type Uniforms = Colour;
    ///
    ///     fn shade(&self, _: f32, colour: &Colour) -> Colour {
    ///         *colour
    ///     }
    /// }
    ///
    /// // A cube stretched to cover the whole view.
    /// let red = Material::new_simple(
    ///     "red".to_owned(),
    ///     Colour::BLACK,
    ///     Colour::RED,
    ///     Colour::BLACK,
    ///     1.0,
    /// );
    /// let model = Model::new(
    ///     vec![Mesh::cube(Colour::WHITE, Some(0))],
    ///     vec![red],
    ///     ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::new(4.0, 4.0, 1.0)),
    /// )?;
    /// let pipeline = Pipeline::new(TransformShader, FlatShader);
    ///
    /// let extent = Extent::new(4, 4);
    /// let mut target = RenderTarget::new(extent);
    /// let mut renderer = Renderer::new()?;
    /// let mut pass = renderer.begin_render_pass(
    ///     &mut target,
    ///     RenderPassDescriptor {
    ///         viewport: Viewport::full(&extent),
    ///         colour_load_op: LoadOp::Clear(Colour::BLACK),
    ///         depth_load_op: None,
    ///     },
    /// );
    /// model.draw_to_render_pass(&mut pass, &pipeline, model.transform.model_matrix(), |mesh| {
    ///     model.materials[mesh.material_index.unwrap()].diffuse
    /// });
    /// pass.finish();
    ///
    /// assert!(target.pixels().iter().all(|&pixel| pixel == Colour::RED));
    /// # Ok(())
    /// # }
    /// ```
    pub fn draw_to_render_pass<'pass, VS, FS, F>(
        &'pass self,
        render_pass: &mut RenderPass<'_, 'pass>,
        pipeline: &'pass Pipeline<VS, FS>,
        vertex_uniforms: VS::Uniforms,
        make_fragment_uniforms: F,
    ) where
        VS: VertexShader<Vertex = V>,
        FS: FragmentShader<VS::Varyings>,
        VS::Uniforms: Clone,
        F: Fn(&Mesh<V>) -> FS::Uniforms,
    {
        for draw_call in self.draw_calls(make_fragment_uniforms) {
            render_pass.draw(pipeline, draw_call, vertex_uniforms.clone());
        }
    }

    /// Queues the model's meshes in a render pass, drawn with a [`SimdPipeline`].
    ///
    /// Each mesh is drawn with a clone of `vertex_uniforms` and fragment uniforms made by
    /// `make_fragment_uniforms` from the mesh, typically from its material. See
    /// [`RenderPass::draw_simd`]. The fragment shader implements [`FragmentShaderSimd`], where
    /// [`draw_to_render_pass`](Model::draw_to_render_pass) takes a scalar one.
    ///
    /// # Panics
    ///
    /// Panics if the pipeline tests or writes depth and the render pass has no depth buffer, as
    /// [`RenderPass::draw_simd`] does.
    ///
    /// # Examples
    ///
    /// Draw each mesh in its material's diffuse colour, with the model matrix as the vertex
    /// uniforms:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// #[derive(Interpolate, SimdInterpolate)]
    /// struct Varyings {
    ///     unused: f32,
    /// }
    ///
    /// struct TransformShader;
    /// impl VertexShader for TransformShader {
    ///     type Vertex = ObjVertex;
    ///     type Uniforms = Mat4;
    ///     type Varyings = Varyings;
    ///
    ///     fn shade(&self, vertex: ObjVertex, model_matrix: &Mat4) -> (Vec4, Varyings) {
    ///         (*model_matrix * vertex.position.to_point4(), Varyings { unused: 0.0 })
    ///     }
    /// }
    ///
    /// struct FlatShader;
    /// impl FragmentShaderSimd<Varyings> for FlatShader {
    ///     type Uniforms = Colour;
    ///
    ///     fn shade_simd(&self, _: VaryingsSimd, colour: &Colour) -> ColourSimd {
    ///         ColourSimd::splat(*colour)
    ///     }
    /// }
    ///
    /// // A cube stretched to cover the whole view.
    /// let red = Material::new_simple(
    ///     "red".to_owned(),
    ///     Colour::BLACK,
    ///     Colour::RED,
    ///     Colour::BLACK,
    ///     1.0,
    /// );
    /// let model = Model::new(
    ///     vec![Mesh::cube(Colour::WHITE, Some(0))],
    ///     vec![red],
    ///     ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::new(4.0, 4.0, 1.0)),
    /// )?;
    /// let pipeline = SimdPipeline::new(TransformShader, FlatShader);
    ///
    /// let extent = Extent::new(4, 4);
    /// let mut target = RenderTarget::new(extent);
    /// let mut renderer = Renderer::new()?;
    /// let mut pass = renderer.begin_render_pass(
    ///     &mut target,
    ///     RenderPassDescriptor {
    ///         viewport: Viewport::full(&extent),
    ///         colour_load_op: LoadOp::Clear(Colour::BLACK),
    ///         depth_load_op: None,
    ///     },
    /// );
    /// let model_matrix = model.transform.model_matrix();
    /// model.draw_to_render_pass_simd(&mut pass, &pipeline, model_matrix, |mesh| {
    ///     model.materials[mesh.material_index.unwrap()].diffuse
    /// });
    /// pass.finish();
    ///
    /// assert!(target.pixels().iter().all(|&pixel| pixel == Colour::RED));
    /// # Ok(())
    /// # }
    /// ```
    pub fn draw_to_render_pass_simd<'pass, VS, FS, F>(
        &'pass self,
        render_pass: &mut RenderPass<'_, 'pass>,
        pipeline: &'pass SimdPipeline<VS, FS>,
        vertex_uniforms: VS::Uniforms,
        make_fragment_uniforms: F,
    ) where
        VS: VertexShader<Vertex = V>,
        VS::Varyings: SimdInterpolate,
        FS: FragmentShaderSimd<VS::Varyings> + Send + Sync + 'static,
        VS::Uniforms: Clone,
        F: Fn(&Mesh<V>) -> FS::Uniforms,
    {
        for draw_call in self.draw_calls(make_fragment_uniforms) {
            render_pass.draw_simd(pipeline, draw_call, vertex_uniforms.clone());
        }
    }
}
/// Normals for models of [`ObjVertex`].
impl Model<ObjVertex> {
    /// Calculates the vertex normals of every mesh, with [`Mesh::calculate_vertex_normals`].
    ///
    /// # Panics
    ///
    /// Panics if a mesh has an index out of range for its vertices, or a last triangle with
    /// fewer than three indices.
    ///
    /// # Examples
    ///
    /// Give a model built from primitives normals for lighting:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let mut model = Model::new(
    ///     vec![Mesh::cube(Colour::WHITE, None)],
    ///     Vec::new(),
    ///     ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
    /// )?;
    ///
    /// model.calculate_vertex_normals();
    /// assert!(model.meshes[0].vertices.iter().all(|v| (v.normal.length() - 1.0).abs() < 1e-5));
    /// # Ok(())
    /// # }
    /// ```
    pub fn calculate_vertex_normals(&mut self) {
        for mesh in &mut self.meshes {
            mesh.calculate_vertex_normals();
        }
    }
}

/// An error constructing a [`Model`].
///
/// Its [`Display`](std::fmt::Display) form is a one-line message, such as
/// `mesh 2 uses material 3, but the model has 3 materials`. It has no
/// [`source`](std::error::Error::source).
///
/// # Examples
///
/// See [`Model::new`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// A mesh refers to a material the model doesn't have.
    #[error("mesh {mesh} uses material {material}, but the model has {material_count} materials")]
    InvalidMaterialIndex {
        /// The mesh's index in the model.
        mesh: usize,
        /// The material index the mesh uses.
        material: usize,
        /// The number of materials the model has.
        material_count: usize,
    },
}

/// Where a model is in the world: its position, rotation, and scale.
///
/// The model matrix scales first, then rotates, then translates. The rotation is three Euler
/// angles in radians, about the x, y, and z axes, applied in that order about the world axes
/// (`Rz * Ry * Rx`). Each is counter-clockwise looking from the positive end of its axis towards
/// the origin.
///
/// # Examples
///
/// ```
/// use std::f32::consts::FRAC_PI_2;
///
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// // Doubled in size, turned a quarter turn about y, then moved 10 units along x.
/// let transform = ModelTransform::new(
///     Vec3::new(10.0, 0.0, 0.0),
///     Vec3::new(0.0, FRAC_PI_2, 0.0),
///     Vec3::new(2.0, 2.0, 2.0),
/// );
///
/// let world = transform.model_matrix() * Vec3::new(1.0, 0.0, 0.0).to_point4();
/// assert!((world.xyz() - Vec3::new(10.0, 0.0, -2.0)).length() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelTransform {
    /// The translation, in world units.
    pub position: Vec3,
    /// The Euler angles about the x, y, and z axes, in radians.
    pub rotation: Vec3,
    /// The scale factor along each of the model's axes.
    pub scale: Vec3,
}
impl ModelTransform {
    /// Creates a transform from a position, Euler angles in radians, and a scale.
    ///
    /// `ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE)` is the identity.
    ///
    /// # Examples
    ///
    /// See [`ModelTransform`].
    pub fn new(position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Returns the model matrix, which takes model space to world space: translation × rotation
    /// × scale.
    ///
    /// # Examples
    ///
    /// See [`ModelTransform`].
    pub fn model_matrix(&self) -> Mat4 {
        let translation_matrix = Mat4::translation_vec(self.position);
        let rotation_matrix = Mat4::rotate_vec(self.rotation);
        let scale_matrix = Mat4::scaling_vec(self.scale);

        translation_matrix * rotation_matrix * scale_matrix
    }
}
