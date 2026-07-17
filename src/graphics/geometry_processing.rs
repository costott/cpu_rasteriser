use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::graphics::clipping::*;
use crate::graphics::vertex_shader::*;

pub struct GeometryProcessor;
impl GeometryProcessor {
    /// Transforms a 3D vertex into a clip space vertex using the model matrix and camera.
    pub fn transform_vertex(vertex: Vertex3D, model_matrix: Mat4, camera: &Camera) -> ClipVertex {
        // Model to World
        let world = model_matrix * vertex.position.to_homogenous();

        // World to View
        let view = camera.view_matrix() * world;

        // Projection from 3D to 2D coordinates
        let clip = camera.projection_matrix() * view;

        ClipVertex {
            position: clip,
            colour: vertex.colour,
            normal: vertex.normal,
        }
    }

    /// Transforms a triangle of 3D vertices into a triangle of clip space vertices using the model matrix and camera.
    pub fn transform_triangle(
        triangle: Triangle3D,
        shader: &dyn VertexShader,
        model_matrix: Mat4,
        camera: &Camera,
    ) -> TriangleClip {
        let a = Self::transform_vertex(shader.shade(triangle.a), model_matrix, camera);
        let b = Self::transform_vertex(shader.shade(triangle.b), model_matrix, camera);
        let c = Self::transform_vertex(shader.shade(triangle.c), model_matrix, camera);

        TriangleClip { a, b, c }
    }

    /// Projects a clip space vertex into 2D screen space using the viewport dimensions.
    pub fn project_vertex(vertex: ClipVertex, viewport: &Viewport) -> Vertex2D {
        // 2D coordiantes to normalised device coordinates
        let ndc = vertex.position / vertex.position.w;

        // Normalised device coordinates to screen coordinates
        let screen = Vec2::new(
            (ndc.x + 1.0) * 0.5 * viewport.width as f32,
            (1.0 - ndc.y) * 0.5 * viewport.height as f32,
        );

        Vertex2D::new(screen, vertex.colour, vertex.normal, ndc.z)
    }

    /// Projects a triangle of clip space vertices into a triangle of 2D screen space vertices using the viewport dimensions.
    pub fn project_triangle(triangle: TriangleClip, viewport: &Viewport) -> Triangle2D {
        let a = Self::project_vertex(triangle.a, viewport);
        let b = Self::project_vertex(triangle.b, viewport);
        let c = Self::project_vertex(triangle.c, viewport);

        Triangle2D { a, b, c }
    }

    pub fn process_triangle(
        triangle: Triangle3D,
        shader: &dyn VertexShader,
        model_matrix: Mat4,
        camera: &Camera,
        viewport: &Viewport,
    ) -> Vec<Triangle2D> {
        let triangle_clip = Self::transform_triangle(triangle, shader, model_matrix, camera);

        let clipped = clip_triangle(triangle_clip);

        clipped
            .iter()
            .map(|t| Self::project_triangle(*t, viewport))
            .collect()
    }
}
