use crate::prelude::*;

use cpu_rasteriser::prelude::*;

use cpu_rasteriser::graphics::fragment_shader::FragmentShader;
use cpu_rasteriser::graphics::vertex_shader::VertexShader;
use cpu_rasteriser::renderer::DrawCall;
use cpu_rasteriser::renderer::{Frame, Pipeline};

#[derive(Debug)]
pub struct Model<V: Clone> {
    pub meshes: Vec<Mesh<V>>,
    pub materials: Vec<Material>,
    pub transform: ModelTransform,
}
impl<V: Clone> Model<V> {
    /// Creates a new model with the given meshes, materials, and transform.
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
                    return Err(ModelError::InvalidMaterialIndex(i, material_index));
                }
            }
        }
        Ok(())
    }

    pub fn draw_calls<U, F>(&self, make_uniforms: F) -> impl Iterator<Item = DrawCall<'_, V, U>>
    where
        F: Fn(&Mesh<V>) -> U,
    {
        self.meshes.iter().map(move |mesh| {
            DrawCall::new(
                &mesh.vertices,
                &mesh.indices,
                cpu_rasteriser::renderer::PrimitiveMode::TRIANGLES,
                make_uniforms(mesh),
            )
        })
    }

    pub fn draw_to_frame<VS, FS, F>(
        &self,
        frame: &mut Frame,
        pipeline: &Pipeline<VS, FS>,
        vertex_uniforms: VS::Uniforms,
        make_fragment_uniforms: F,
    ) where
        VS: VertexShader<Vertex = V>,
        FS: FragmentShader<VS::Varyings>,
        VS::Varyings: Interpolate + Send + Sync + 'static,
        FS::Uniforms: Send + Sync + 'static,
        VS::Uniforms: Clone,
        F: Fn(&Mesh<V>) -> FS::Uniforms,
    {
        for draw_call in self.draw_calls(make_fragment_uniforms) {
            frame.draw(pipeline, draw_call, vertex_uniforms.clone());
        }
    }
}
impl Model<ObjVertex> {
    pub fn calculate_vertex_normals(&mut self) {
        for mesh in &mut self.meshes {
            mesh.calculate_vertex_normals();
        }
    }
}

#[derive(Debug)]
pub enum ModelError {
    InvalidMaterialIndex(usize, usize), // (mesh_index, material_index)
}
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::InvalidMaterialIndex(mesh_index, material_index) => write!(
                f,
                "Mesh {} has an invalid material index: {}",
                mesh_index, material_index
            ),
        }
    }
}
impl std::error::Error for ModelError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelTransform {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
}
impl ModelTransform {
    pub fn new(position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn model_matrix(&self) -> Mat4 {
        let translation_matrix = Mat4::translation_vec(self.position);
        let rotation_matrix = Mat4::rotate_vec(self.rotation);
        let scale_matrix = Mat4::scaling_vec(self.scale);

        translation_matrix * rotation_matrix * scale_matrix
    }
}
