use crate::{graphics::camera::Camera, prelude::*};

pub struct GeometryProcessor;
impl GeometryProcessor {
    pub fn process(
        vertex: Vertex3D,
        model_matrix: Mat4,
        camera: &Camera,
        viewport: &Viewport,
    ) -> Vertex2D {
        // Model to World
        let world = model_matrix * vertex.position.to_homogenous();

        // World to View
        let view = camera.view_matrix() * world;

        // Projection from 3D to 2D coordinates
        let clip = camera.projection_matrix() * view;

        // 2D coordiantes to normalised device coordinates
        let ndc = clip / clip.w;

        // Normalised device coordinates to screen coordinates
        let screen = Vec2::new(
            (ndc.x + 1.0) * 0.5 * viewport.width as f32,
            (1.0 - ndc.y) * 0.5 * viewport.height as f32,
        );

        Vertex2D::new(screen, vertex.colour, ndc.z)
    }

    pub fn process_triangle(
        triangle: Triangle3D,
        model_matrix: Mat4,
        camera: &Camera,
        viewport: &Viewport,
    ) -> Triangle {
        let a = Self::process(triangle.a, model_matrix, camera, viewport);
        let b = Self::process(triangle.b, model_matrix, camera, viewport);
        let c = Self::process(triangle.c, model_matrix, camera, viewport);

        Triangle::new(a, b, c)
    }
}
