use crate::prelude::*;

#[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub transform: ModelTransform,
}
impl Model {
    /// Creates a new model with the given meshes, materials, and transform.
    pub fn new(
        meshes: Vec<Mesh>,
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
