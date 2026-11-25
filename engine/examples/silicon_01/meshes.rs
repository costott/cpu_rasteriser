//! Building meshes in code: flat-shaded primitives appended to one vertex and index list.

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

/// Collects triangles into one mesh.
#[derive(Default)]
pub struct MeshBuilder {
    vertices: Vec<ObjVertex>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    /// Adds a flat-shaded quad with corners in anticlockwise order seen from its front.
    pub fn quad(&mut self, corners: [Vec3; 4], colour: Colour) -> &mut Self {
        let normal = (corners[1] - corners[0])
            .cross(corners[2] - corners[0])
            .normalise();
        let base = self.vertices.len() as u32;
        let uvs = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        for (position, uv) in corners.into_iter().zip(uvs) {
            self.vertices.push(ObjVertex {
                position,
                colour,
                normal,
                uv,
            });
        }
        self.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        self
    }

    /// Adds an axis-aligned, flat-shaded box from `min` to `max`.
    pub fn cuboid(&mut self, min: Vec3, max: Vec3, colour: Colour) -> &mut Self {
        let corner = |x: bool, y: bool, z: bool| {
            Vec3::new(
                if x { max.x } else { min.x },
                if y { max.y } else { min.y },
                if z { max.z } else { min.z },
            )
        };
        let c = |i: u8| corner(i & 1 != 0, i & 2 != 0, i & 4 != 0);
        // Each face's corners, anticlockwise seen from outside.
        for face in [
            [4, 5, 7, 6], // +z
            [1, 0, 2, 3], // -z
            [5, 1, 3, 7], // +x
            [0, 4, 6, 2], // -x
            [6, 7, 3, 2], // +y
            [0, 1, 5, 4], // -y
        ] {
            self.quad(face.map(c), colour);
        }
        self
    }

    /// Adds a vertex with its own normal and returns its index, for smooth-shaded surfaces built
    /// with [`MeshBuilder::triangle`].
    pub fn vertex(&mut self, position: Vec3, normal: Vec3, colour: Colour) -> u32 {
        self.vertices.push(ObjVertex {
            position,
            colour,
            normal,
            uv: Vec2::new(0.0, 0.0),
        });
        (self.vertices.len() - 1) as u32
    }

    /// Adds a triangle between three vertices, anticlockwise seen from its front.
    pub fn triangle(&mut self, a: u32, b: u32, c: u32) -> &mut Self {
        self.indices.extend([a, b, c]);
        self
    }

    pub fn build(self) -> Mesh<ObjVertex> {
        Mesh::new(self.vertices, self.indices, None)
    }
}

/// A model of one mesh with no materials, placed at the origin.
pub fn model(mesh: Mesh<ObjVertex>) -> Model<ObjVertex> {
    Model::new(
        vec![mesh],
        Vec::new(),
        ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
    )
    .expect("a model with one mesh and no materials is always valid")
}
