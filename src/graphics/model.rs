use crate::prelude::*;

pub struct Model {
    // pub mesh: Mesh,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub transform: ModelTransform,
}
impl Model {
    pub fn new(meshes: Vec<Mesh>, materials: Vec<Material>, transform: ModelTransform) -> Self {
        Self {
            meshes,
            materials,
            transform,
        }
    }

    pub fn calculate_vertex_normals(&mut self) {
        for mesh in &mut self.meshes {
            mesh.calculate_vertex_normals();
        }
    }
}

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
