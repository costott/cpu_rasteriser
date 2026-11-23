//! Indexed triangle meshes.

use cpu_rasteriser::prelude::*;

/// An indexed triangle list drawn with one material.
///
/// Each group of three [`indices`](Mesh::indices) is a triangle, and each index refers to one of
/// the [`vertices`](Mesh::vertices). Vertices shared by several triangles are stored once. The
/// mesh is in model space; a [`Model`](crate::components::model::Model) holds meshes together
/// with their materials and transform.
///
/// Nothing checks that the indices are in range or that there is a whole number of triangles:
/// [`triangles`](Mesh::triangles) and [`calculate_vertex_normals`](Mesh::calculate_vertex_normals)
/// panic if not, and so may drawing the mesh.
///
/// # Examples
///
/// A quad made of two triangles that share an edge:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let corner = |x, y| ObjVertex::simple(Vec3::new(x, y, 0.0), Colour::WHITE);
/// let quad = Mesh::new(
///     vec![corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0)],
///     vec![0, 1, 2, 0, 2, 3],
///     None,
/// );
///
/// assert_eq!(quad.triangles().count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Mesh<V: Clone> {
    /// The vertices the indices refer to.
    pub vertices: Vec<V>,
    /// The triangles, as three indices into [`vertices`](Mesh::vertices) each.
    pub indices: Vec<u32>,
    /// The index of the mesh's material in its model's
    /// [`materials`](crate::components::model::Model::materials), if it has one.
    pub material_index: Option<usize>,
}
impl<V: Clone> Mesh<V> {
    /// Creates a mesh from its vertices, triangle indices, and material index.
    ///
    /// # Examples
    ///
    /// See [`Mesh`].
    pub fn new(vertices: Vec<V>, indices: Vec<u32>, material_index: Option<usize>) -> Self {
        Self {
            vertices,
            indices,
            material_index,
        }
    }

    /// Returns an iterator over the mesh's triangles, each with copies of its three vertices.
    ///
    /// # Panics
    ///
    /// The iterator panics on reaching a triangle with an index that is out of range for
    /// [`vertices`](Mesh::vertices), or a final triangle with fewer than three indices.
    ///
    /// # Examples
    ///
    /// Find the mesh's surface area:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let cube = Mesh::cube(Colour::WHITE, None);
    ///
    /// let area: f32 = cube
    ///     .triangles()
    ///     .map(|t| {
    ///         (t.b.position - t.a.position).cross(t.c.position - t.a.position).length() / 2.0
    ///     })
    ///     .sum();
    /// assert!((area - 6.0).abs() < 1e-5);
    /// ```
    pub fn triangles(&self) -> impl Iterator<Item = Triangle3D<V>> + '_ {
        self.indices.chunks(3).map(move |chunk| {
            Triangle3D::new(
                self.vertices[chunk[0] as usize].clone(),
                self.vertices[chunk[1] as usize].clone(),
                self.vertices[chunk[2] as usize].clone(),
            )
        })
    }
}
/// Primitive shapes, and normals, for meshes of [`ObjVertex`].
impl Mesh<ObjVertex> {
    /// Creates a cube with sides of length 1, centred on the origin, in a single colour.
    ///
    /// It has 8 vertices, one per corner, shared by its 12 triangles (36 indices). Each
    /// triangle winds counter-clockwise seen from outside the cube, so it is front-facing for
    /// back-face culling. The vertices' normals and texture coordinates are zero: call
    /// [`calculate_vertex_normals`](Mesh::calculate_vertex_normals) for normals, which with
    /// shared corners point roughly along the diagonals and give smooth rather than flat
    /// shading.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let cube = Mesh::cube(Colour::RED, Some(0));
    ///
    /// assert_eq!((cube.vertices.len(), cube.indices.len()), (8, 36));
    /// // Every face points away from the centre.
    /// assert!(cube.triangles().all(|t| t.normal().dot(t.a.position) > 0.0));
    /// ```
    pub fn cube(colour: Colour, material_index: Option<usize>) -> Self {
        let vertices = vec![
            ObjVertex::simple(Vec3::new(-0.5, -0.5, -0.5), colour),
            ObjVertex::simple(Vec3::new(0.5, -0.5, -0.5), colour),
            ObjVertex::simple(Vec3::new(0.5, 0.5, -0.5), colour),
            ObjVertex::simple(Vec3::new(-0.5, 0.5, -0.5), colour),
            ObjVertex::simple(Vec3::new(-0.5, -0.5, 0.5), colour),
            ObjVertex::simple(Vec3::new(0.5, -0.5, 0.5), colour),
            ObjVertex::simple(Vec3::new(0.5, 0.5, 0.5), colour),
            ObjVertex::simple(Vec3::new(-0.5, 0.5, 0.5), colour),
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

    /// Creates a UV sphere of the given radius, centred on the origin, in a single colour.
    ///
    /// The sphere is divided into `latitude_segments` bands from the pole at +y to the pole at
    /// −y, and `longitude_segments` slices around the y axis. It has
    /// `(latitude_segments + 1) * (longitude_segments + 1)` vertices, because the poles and the
    /// seam are duplicated, and two triangles for each band and slice (those at the poles are
    /// degenerate). The vertices' normals and texture coordinates are zero.
    ///
    /// `latitude_segments` and `longitude_segments` must be at least 2 and 3 for a closed
    /// shape. A zero for either gives NaN positions; nothing panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let sphere = Mesh::sphere(2.0, 8, 16, Colour::WHITE, None);
    ///
    /// assert_eq!(sphere.vertices.len(), 9 * 17);
    /// assert_eq!(sphere.indices.len(), 8 * 16 * 6);
    /// assert!(sphere.vertices.iter().all(|v| (v.position.length() - 2.0).abs() < 1e-5));
    /// ```
    ///
    /// # Known issues
    ///
    /// - Unlike [`cube`](Mesh::cube), the triangles wind clockwise seen from outside the sphere, so
    ///   back-face culling removes the outside, and
    ///   [`calculate_vertex_normals`](Mesh::calculate_vertex_normals) gives normals that point
    ///   inwards ([#53](https://github.com/costott/cpu_rasteriser/issues/53)).
    pub fn sphere(
        radius: f32,
        latitude_segments: usize,
        longitude_segments: usize,
        colour: Colour,
        material_index: Option<usize>,
    ) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for lat in 0..=latitude_segments {
            let theta = lat as f32 * std::f32::consts::PI / latitude_segments as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for lon in 0..=longitude_segments {
                let phi = lon as f32 * 2.0 * std::f32::consts::PI / longitude_segments as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                let x = radius * sin_theta * cos_phi;
                let y = radius * cos_theta;
                let z = radius * sin_theta * sin_phi;

                vertices.push(ObjVertex::simple(Vec3::new(x, y, z), colour));
            }
        }

        for lat in 0..latitude_segments {
            for lon in 0..longitude_segments {
                let first = (lat * (longitude_segments + 1) + lon) as u32;
                let second = first + longitude_segments as u32 + 1;

                indices.push(first);
                indices.push(second);
                indices.push(first + 1);

                indices.push(second);
                indices.push(second + 1);
                indices.push(first + 1);
            }
        }

        Self {
            vertices,
            indices,
            material_index,
        }
    }

    /// Sets each vertex's normal to the normalised sum of the face normals of the triangles that
    /// use it.
    ///
    /// Each face normal is normalised before summing, so every triangle counts equally whatever
    /// its area. A face normal points to the side from which the triangle winds
    /// counter-clockwise. A vertex that no non-degenerate triangle uses, or whose face normals
    /// cancel out, gets a zero normal. Triangles that share vertices get smooth shading; for flat
    /// shading, give each triangle its own vertices.
    ///
    /// # Panics
    ///
    /// Panics if an index is out of range for [`vertices`](Mesh::vertices), or the last triangle
    /// has fewer than three indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let corner = |x, y| ObjVertex::simple(Vec3::new(x, y, 0.0), Colour::WHITE);
    /// let mut quad = Mesh::new(
    ///     vec![corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0)],
    ///     vec![0, 1, 2, 0, 2, 3],
    ///     None,
    /// );
    ///
    /// quad.calculate_vertex_normals();
    /// assert!(quad.vertices.iter().all(|v| v.normal == Vec3::new(0.0, 0.0, 1.0)));
    /// ```
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
