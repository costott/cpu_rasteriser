use crate::prelude::*;

pub struct Mesh {
    /// The vertices of the mesh.
    vertices: Vec<Vertex3D>,
    /// The indices of the mesh, which define the triangles.
    indices: Vec<u32>,
    /// The index of the material to use for this mesh.
    pub material_index: usize,
}
impl Mesh {
    pub fn new(vertices: Vec<Vertex3D>, indices: Vec<u32>, material_index: usize) -> Self {
        Self {
            vertices,
            indices,
            material_index,
        }
    }

    /// Creates a white cube mesh with 8 vertices and 12 triangles (36 indices).
    pub fn cube(colour: Colour, material_index: usize) -> Self {
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
            // Back (-Z)
            0, 2, 1, //
            0, 3, 2, //
            // Front (+Z)
            4, 5, 6, //
            4, 6, 7, //
            // Left (-X)
            0, 7, 3, //
            0, 4, 7, //
            // Right (+X)
            1, 2, 6, //
            1, 6, 5, //
            // Bottom (-Y)
            0, 1, 5, //
            0, 5, 4, //
            // Top (+Y)
            3, 7, 6, //
            3, 6, 2, //
        ];

        Self {
            vertices,
            indices,
            material_index,
        }
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
}
