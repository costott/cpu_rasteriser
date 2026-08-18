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
            viewport.x as f32 + (ndc.x + 1.0) * 0.5 * viewport.width as f32,
            viewport.y as f32 + (1.0 - ndc.y) * 0.5 * viewport.height as f32,
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

    #[derive(Clone)]
    struct TestVertex {
        position: Vec4,
    }

    fn vertex(x: f32, y: f32, z: f32) -> TestVertex {
        TestVertex {
            position: Vec4::new(x, y, z, 1.0),
        }
    }

    #[derive(Interpolate)]
    struct TestVaryings {
        pub colour: Vec4,
    }

    struct TestUniforms;

    struct TestVertexShader;

    impl VertexShader for TestVertexShader {
        type Vertex = TestVertex;
        type Uniforms = TestUniforms;
        type Varyings = TestVaryings;

        fn shade(&self, vertex: Self::Vertex, _: &Self::Uniforms) -> (Vec4, Self::Varyings) {
            (
                vertex.position,
                TestVaryings {
                    colour: Colour::WHITE.into(),
                },
            )
        }
    }

    fn front_facing_triangle() -> Triangle3D<TestVertex> {
        Triangle3D::new(
            vertex(-0.5, -0.5, 0.0),
            vertex(0.5, -0.5, 0.0),
            vertex(0.0, 0.5, 0.0),
        )
    }

    fn back_facing_triangle() -> Triangle3D<TestVertex> {
        Triangle3D::new(
            vertex(-0.5, -0.5, 0.0),
            vertex(0.0, 0.5, 0.0),
            vertex(0.5, -0.5, 0.0),
        )
    }

    #[test]
    fn process_triangle_keeps_front_facing_triangles() {
        let viewport = Viewport::new(0, 0, 8, 8);

        let triangles = GeometryProcessor::process_triangle(
            front_facing_triangle(),
            &TestVertexShader,
            &TestUniforms,
            &viewport,
            CullingMode::BackFace,
        );

        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn process_triangle_culls_back_facing_triangles() {
        let viewport = Viewport::new(0, 0, 8, 8);

        let triangles = GeometryProcessor::process_triangle(
            back_facing_triangle(),
            &TestVertexShader,
            &TestUniforms,
            &viewport,
            CullingMode::BackFace,
        );

        assert!(triangles.is_empty());
    }

    #[test]
    fn process_triangle_keeps_back_facing_triangles_when_culling_disabled() {
        let viewport = Viewport::new(0, 0, 8, 8);

        let triangles = GeometryProcessor::process_triangle(
            back_facing_triangle(),
            &TestVertexShader,
            &TestUniforms,
            &viewport,
            CullingMode::None,
        );

        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn process_triangle_converts_vertices_to_screen_space() {
        let viewport = Viewport::new(0, 0, 100, 100);

        let triangles = GeometryProcessor::process_triangle(
            front_facing_triangle(),
            &TestVertexShader,
            &TestUniforms,
            &viewport,
            CullingMode::None,
        );

        let tri = &triangles[0];

        for vertex in [&tri.a, &tri.b, &tri.c] {
            assert!(vertex.position.x >= 0.0);
            assert!(vertex.position.x <= 100.0);
            assert!(vertex.position.y >= 0.0);
            assert!(vertex.position.y <= 100.0);
        }
    }

    #[test]
    fn process_triangle_preserves_triangle_when_fully_inside_frustum() {
        let viewport = Viewport::new(0, 0, 100, 100);

        let triangles = GeometryProcessor::process_triangle(
            front_facing_triangle(),
            &TestVertexShader,
            &TestUniforms,
            &viewport,
            CullingMode::BackFace,
        );

        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn process_triangle_clips_triangle_crossing_near_plane() {
        let viewport = Viewport::new(0, 0, 100, 100);

        let triangle = Triangle3D::new(
            vertex(-1.0, -1.0, -0.05), // outside near plane
            vertex(1.0, -1.0, -2.0),
            vertex(0.0, 1.0, -2.0),
        );

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &TestVertexShader,
            &TestUniforms,
            &viewport,
            CullingMode::None,
        );

        assert!(!triangles.is_empty());
    }

    #[test]
    fn process_triangle_discards_triangle_outside_frustum() {
        let viewport = Viewport::new(0, 0, 100, 100);

        let triangle = Triangle3D::new(
            vertex(100.0, 100.0, -2.0), // outside frustum
            vertex(101.0, 100.0, -2.0), // outside frustum
            vertex(100.0, 101.0, -2.0), // outside frustum
        );

        let triangles = GeometryProcessor::process_triangle(
            triangle,
            &TestVertexShader,
            &TestUniforms,
            &viewport,
            CullingMode::None,
        );

        assert!(triangles.is_empty());
    }
}
