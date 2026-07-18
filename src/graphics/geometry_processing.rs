use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::graphics::clipping::*;
use crate::graphics::vertex_shader::*;

pub struct GeometryProcessor;
impl GeometryProcessor {
    /// Transforms a 3D vertex into a clip space vertex using the model matrix and camera.
    fn world_to_clip(vertex: Vertex3D, camera: &Camera) -> ClipVertex {
        // World to View
        let view = camera.view_matrix() * vertex.position.to_homogenous();

        // Projection from 3D to 2D coordinates
        let clip = camera.projection_matrix() * view;

        ClipVertex {
            position: clip,
            colour: vertex.colour,
            normal: vertex.normal,
        }
    }

    /// Transforms a 3D vertex into a world space vertex using the model matrix.
    fn model_to_world(vertex: Vertex3D, model_matrix: Mat4) -> Vertex3D {
        let world_pos = model_matrix * vertex.position.to_homogenous();

        let normal_matrix = model_matrix.inverse().transpose();

        Vertex3D {
            position: world_pos.homogenize_to_vec3(),
            colour: vertex.colour,
            normal: (normal_matrix * vertex.normal.to_homogenous())
                .homogenize_to_vec3()
                .normalise(),
        }
    }

    /// Transforms a triangle of 3D vertices into a triangle of clip space vertices using the model matrix and camera.
    fn transform_triangle(
        triangle: Triangle3D,
        shader: &dyn VertexShader,
        model_matrix: Mat4,
        camera: &Camera,
    ) -> TriangleClip {
        let a = Self::model_to_world(triangle.a, model_matrix);
        let b = Self::model_to_world(triangle.b, model_matrix);
        let c = Self::model_to_world(triangle.c, model_matrix);

        let a = Self::world_to_clip(shader.shade(a), camera);
        let b = Self::world_to_clip(shader.shade(b), camera);
        let c = Self::world_to_clip(shader.shade(c), camera);

        TriangleClip { a, b, c }
    }

    /// Projects a clip space vertex into 2D screen space using the viewport dimensions.
    fn project_vertex(vertex: ClipVertex, viewport: &Viewport) -> Vertex2D {
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
    fn project_triangle(triangle: TriangleClip, viewport: &Viewport) -> Triangle2D {
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
