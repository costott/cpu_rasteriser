use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::graphics::clipping::*;
use crate::graphics::vertex_shader::*;
use crate::renderer::CullingMode;

pub struct GeometryProcessor;
impl GeometryProcessor {
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

    /// Transforms a 3D triangle into world space using the model matrix.
    fn triangle_model_to_world(triangle: Triangle3D, model_matrix: Mat4) -> Triangle3D {
        Triangle3D {
            a: Self::model_to_world(triangle.a, model_matrix),
            b: Self::model_to_world(triangle.b, model_matrix),
            c: Self::model_to_world(triangle.c, model_matrix),
        }
    }

    /// Returns true when the triangle faces away from the camera in view space.
    fn is_back_facing(triangle: &Triangle3D, camera: &Camera) -> bool {
        let view = camera.view_matrix();

        let a = (view * triangle.a.position.to_homogenous()).homogenize_to_vec3();
        let b = (view * triangle.b.position.to_homogenous()).homogenize_to_vec3();
        let c = (view * triangle.c.position.to_homogenous()).homogenize_to_vec3();

        let normal = (b - a).cross(&(c - a));
        let centroid = (a + b + c) / 3.0;

        normal.dot(&centroid) <= 0.0
    }

    /// Transforms a triangle of 3D vertices into a triangle of clip space vertices using the model matrix and camera.
    fn triangle_world_to_clip(
        triangle: Triangle3D,
        shader: &dyn VertexShader,
        camera: &Camera,
    ) -> TriangleClip {
        let a = Self::world_to_clip(shader.shade(triangle.a), camera);
        let b = Self::world_to_clip(shader.shade(triangle.b), camera);
        let c = Self::world_to_clip(shader.shade(triangle.c), camera);

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
        culling_mode: CullingMode,
    ) -> Vec<Triangle2D> {
        let triangle = Self::triangle_model_to_world(triangle, model_matrix);

        if matches!(culling_mode, CullingMode::BackFace) && Self::is_back_facing(&triangle, camera)
        {
            return vec![];
        }

        let triangle_clip = Self::triangle_world_to_clip(triangle, shader, camera);

        let clipped = clip_triangle(triangle_clip);

        clipped
            .iter()
            .map(|t| Self::project_triangle(*t, viewport))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::camera::{OrthographicProjection, Projection};
    use crate::graphics::vertex_shader::BasicVertexShader;

    fn test_camera() -> Camera {
        Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Orthographic(OrthographicProjection::new(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0)),
        )
    }

    #[test]
    fn process_triangle_keeps_front_facing_triangles() {
        let viewport = Viewport::new(8, 8);
        let camera = test_camera();
        let shader = BasicVertexShader;
        let triangle = Triangle3D::new(
            Vertex3D::new(Vec3::new(-1.0, -1.0, -2.0), Colour::WHITE),
            Vertex3D::new(Vec3::new(1.0, -1.0, -2.0), Colour::WHITE),
            Vertex3D::new(Vec3::new(0.0, 1.0, -2.0), Colour::WHITE),
        );

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &shader,
            Mat4::identity(),
            &camera,
            &viewport,
            CullingMode::BackFace,
        );

        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn process_triangle_culls_back_facing_triangles() {
        let viewport = Viewport::new(8, 8);
        let camera = test_camera();
        let shader = BasicVertexShader;
        let triangle = Triangle3D::new(
            Vertex3D::new(Vec3::new(-1.0, -1.0, -2.0), Colour::WHITE),
            Vertex3D::new(Vec3::new(0.0, 1.0, -2.0), Colour::WHITE),
            Vertex3D::new(Vec3::new(1.0, -1.0, -2.0), Colour::WHITE),
        );

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &shader,
            Mat4::identity(),
            &camera,
            &viewport,
            CullingMode::BackFace,
        );

        assert!(triangles.is_empty());
    }

    #[test]
    fn process_triangle_keeps_back_facing_triangles_when_culling_is_disabled() {
        let viewport = Viewport::new(8, 8);
        let camera = test_camera();
        let shader = BasicVertexShader;
        let triangle = Triangle3D::new(
            Vertex3D::new(Vec3::new(-1.0, -1.0, -2.0), Colour::WHITE),
            Vertex3D::new(Vec3::new(0.0, 1.0, -2.0), Colour::WHITE),
            Vertex3D::new(Vec3::new(1.0, -1.0, -2.0), Colour::WHITE),
        );

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &shader,
            Mat4::identity(),
            &camera,
            &viewport,
            CullingMode::None,
        );

        assert_eq!(triangles.len(), 1);
    }
}
