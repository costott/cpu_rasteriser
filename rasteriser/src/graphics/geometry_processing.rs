use crate::prelude::*;

use crate::graphics::clipping::*;
use crate::graphics::vertex_shader::*;
use crate::renderer::CullingMode;

pub struct GeometryProcessor;
impl GeometryProcessor {
    pub fn process_triangle<VS>(
        triangle: Triangle3D<VS::Vertex>,
        vertex_shader: &VS,
        vertex_uniforms: &VS::Uniforms,
        viewport: &Viewport,
        culling_mode: CullingMode,
    ) -> Vec<Triangle2D<VS::Varyings>>
    where
        VS: VertexShader,
    {
        let triangle_clip = Self::vertex_stage(triangle, vertex_shader, vertex_uniforms);

        // Clipping and backface culling
        clip_triangle(triangle_clip)
            .into_iter()
            .map(|triangle| Self::triangle_clip_to_screen(triangle, viewport))
            .filter(|triangle| {
                !matches!(culling_mode, CullingMode::BackFace)
                    || !Self::is_back_facing_screen(triangle)
            })
            .collect()
    }

    /// Process a triangle through the vertex shader
    fn vertex_stage<VS>(
        triangle: Triangle3D<VS::Vertex>,
        shader: &VS,
        uniforms: &VS::Uniforms,
    ) -> TriangleClip<VS::Varyings>
    where
        VS: VertexShader,
    {
        let (a_pos, a_var) = shader.shade(triangle.a, uniforms);

        let (b_pos, b_var) = shader.shade(triangle.b, uniforms);

        let (c_pos, c_var) = shader.shade(triangle.c, uniforms);

        TriangleClip {
            a: ClipVertex {
                position: a_pos,
                varyings: a_var,
            },
            b: ClipVertex {
                position: b_pos,
                varyings: b_var,
            },
            c: ClipVertex {
                position: c_pos,
                varyings: c_var,
            },
        }
    }

    fn is_back_facing_screen<V>(triangle: &Triangle2D<V>) -> bool
    where
        V: Interpolate,
    {
        let ab = triangle.b.position - triangle.a.position;
        let ac = triangle.c.position - triangle.a.position;

        ab.cross(&ac) >= 0.0
    }

    fn triangle_clip_to_screen<V>(triangle: TriangleClip<V>, viewport: &Viewport) -> Triangle2D<V>
    where
        V: Interpolate,
    {
        Triangle2D {
            a: Self::clip_to_screen(triangle.a, viewport),
            b: Self::clip_to_screen(triangle.b, viewport),
            c: Self::clip_to_screen(triangle.c, viewport),
        }
    }

    fn clip_to_screen<V>(vertex: ClipVertex<V>, viewport: &Viewport) -> RasterVertex<V>
    where
        V: Interpolate,
    {
        let inv_w = 1.0 / vertex.position.w;

        let ndc = vertex.position * inv_w;

        let screen = Vec2::new(
            (ndc.x + 1.0) * 0.5 * viewport.width as f32,
            (1.0 - ndc.y) * 0.5 * viewport.height as f32,
        );

        RasterVertex {
            position: screen,
            depth: ndc.z,
            inv_w,
            varyings: vertex.varyings.scale(inv_w),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::camera::{OrthographicProjection, Projection};

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
        let vertex_uniforms = VertexUniforms { lights: &[] };

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &shader,
            &vertex_uniforms,
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
        let vertex_uniforms = VertexUniforms { lights: &[] };

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &shader,
            &vertex_uniforms,
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
        let vertex_uniforms = VertexUniforms { lights: &[] };

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &shader,
            &vertex_uniforms,
            Mat4::identity(),
            &camera,
            &viewport,
            CullingMode::None,
        );

        assert_eq!(triangles.len(), 1);
    }
}
