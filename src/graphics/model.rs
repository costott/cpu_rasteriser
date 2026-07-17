use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::renderer::Renderer;

pub struct Model {
    pub mesh: Mesh,
    pub transform: ModelTransform,
}
impl Model {
    pub fn new(mesh: Mesh, transform: ModelTransform) -> Self {
        Self { mesh, transform }
    }

    pub fn draw_wireframe(&self, renderer: &mut Renderer, camera: &Camera, viewport: &Viewport) {
        self.mesh
            .draw_wireframe(renderer, self.transform.model_matrix(), camera, viewport);
    }

    pub fn draw_filled(&self, renderer: &mut Renderer, camera: &Camera, viewport: &Viewport) {
        self.mesh
            .draw_filled(renderer, self.transform.model_matrix(), camera, viewport);
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
