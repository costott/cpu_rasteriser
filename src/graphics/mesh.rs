use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::graphics::geometry_processing::GeometryProcessor;
use crate::renderer::Renderer;

pub struct Mesh {
    vertices: Vec<Vertex3D>,
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

    pub fn draw_wireframe(
        &self,
        renderer: &mut Renderer,
        model_matrix: Mat4,
        camera: &Camera,
        viewport: &Viewport,
    ) {
        for triangle in self.triangles() {
            GeometryProcessor::process_triangle(triangle, model_matrix, camera, viewport)
                .draw(renderer);
        }
    }

    pub fn draw_filled(
        &self,
        renderer: &mut Renderer,
        model_matrix: Mat4,
        camera: &Camera,
        viewport: &Viewport,
    ) {
        for triangle in self.triangles() {
            GeometryProcessor::process_triangle(triangle, model_matrix, camera, viewport)
                .draw_filled(renderer);
        }
    }
}
