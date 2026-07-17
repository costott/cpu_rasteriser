use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::graphics::geometry_processing::GeometryProcessor;
use crate::graphics::vertex_shader::VertexShader;
use crate::renderer::Renderer;

pub struct Mesh {
    /// The vertices of the mesh.
    vertices: Vec<Vertex3D>,
    /// The indices of the mesh, which define the triangles.
    indices: Vec<u32>,
}
impl Mesh {
    pub fn new(vertices: Vec<Vertex3D>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Creates a white cube mesh with 8 vertices and 12 triangles (36 indices).
    pub fn cube(colour: Colour) -> Self {
        let vertices = vec![
            Vertex3D::new(Vec3::new(-0.5, -0.5, -0.5), colour),
            Vertex3D::new(Vec3::new(0.5, -0.5, -0.5), colour),
            Vertex3D::new(Vec3::new(0.5, 0.5, -0.5), colour),
            Vertex3D::new(Vec3::new(-0.5, 0.5, -0.5), colour),
            Vertex3D::new(Vec3::new(-0.5, -0.5, 0.5), colour),
            Vertex3D::new(Vec3::new(0.5, -0.5, 0.5), colour),
            Vertex3D::new(Vec3::new(0.5, 0.5, 0.5), colour),
            Vertex3D::new(Vec3::new(-0.5, 0.5, 0.5), colour),
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 4, 7, 3, 3, 0, 4, 1, 5, 6, 6, 2, 1, 4, 1, 0, 1, 4,
            5, 7, 6, 2, 2, 3, 7,
        ];

        Self { vertices, indices }
    }

    /// Returns an iterator over the triangles in the mesh, where each triangle is represented by three vertices.
    pub fn triangles(&self) -> impl Iterator<Item = Triangle3D> + '_ {
        self.indices.chunks(3).map(move |chunk| {
            Triangle3D::new(
                self.vertices[chunk[0] as usize],
                self.vertices[chunk[1] as usize],
                self.vertices[chunk[2] as usize],
            )
        })
    }

    pub fn calculate_vertex_normals(&mut self) {
        let mut normals = vec![Vec3::ZERO; self.vertices.len()];

        for triangle in self.indices.chunks(3) {
            let i0 = triangle[0] as usize;
            let i1 = triangle[1] as usize;
            let i2 = triangle[2] as usize;

            let face = Triangle3D::new(self.vertices[i0], self.vertices[i1], self.vertices[i2]);

            let n = face.normal();

            normals[i0] += n;
            normals[i1] += n;
            normals[i2] += n;
        }

        for (vertex, normal) in self.vertices.iter_mut().zip(normals.iter()) {
            vertex.normal = normal.normalise();
        }
    }

    pub fn draw_wireframe(
        &self,
        renderer: &mut Renderer,
        vertex_shader: &dyn VertexShader,
        model_matrix: Mat4,
        camera: &Camera,
        viewport: &Viewport,
    ) {
        for triangle in self.triangles() {
            for triangle_2d in GeometryProcessor::process_triangle(
                triangle,
                vertex_shader,
                model_matrix,
                camera,
                viewport,
            ) {
                triangle_2d.draw(renderer);
            }
        }
    }

    pub fn draw_filled(
        &self,
        renderer: &mut Renderer,
        vertex_shader: &dyn VertexShader,
        model_matrix: Mat4,
        camera: &Camera,
        viewport: &Viewport,
    ) {
        for triangle in self.triangles() {
            for triangle_2d in GeometryProcessor::process_triangle(
                triangle,
                vertex_shader,
                model_matrix,
                camera,
                viewport,
            ) {
                triangle_2d.draw_filled(renderer);
            }
        }
    }
}
