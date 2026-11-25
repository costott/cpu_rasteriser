//! Low-poly geometry built in code.
//!
//! A [`Shape`] is a flat-shaded triangle soup: every triangle has its own three vertices and one
//! normal, which is what makes a low-poly look. Build shapes from the generators ([`Shape::lathe`],
//! [`Shape::cone`], [`Shape::icosphere`], [`Shape::cuboid`], [`Shape::ribbon`]), move and combine
//! them with [`Shape::transformed`] and [`Shape::append`], and place many copies with
//! [`Batch`](super::scatter::Batch). Draw a shape with [`Shape::draw`] and a kit pipeline.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use super::random::Rng;
use crate::simd::hash2_scalar;

/// A corner of a [`Shape`].
///
/// `sway` is how much the vertex shader's [`Motion`](super::draw::Motion) moves it: 0 for a
/// vertex that stays put (a trunk's base), up to 1 for one that moves freely (a branch tip).
/// `seed` identifies the instance the vertex belongs to, in [0, 1): it phases the motion so
/// copies don't move in step.
#[derive(Clone, Copy, Debug)]
pub struct KitVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub colour: Vec3,
    pub sway: f32,
    pub seed: f32,
}

/// Transforms a point by `m`.
pub fn transform_point(m: &Mat4, p: Vec3) -> Vec3 {
    (*m * p.to_point4()).xyz()
}

/// Linear interpolation between two colours or points.
pub fn lerp3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

/// A flat-shaded triangle soup; see the [module documentation](self).
#[derive(Clone, Debug, Default)]
pub struct Shape {
    vertices: Vec<KitVertex>,
    indices: Vec<u32>,
}

impl Shape {
    /// An empty shape.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of triangles, for staying within the frame's triangle budget.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The vertices, in the order they were added.
    pub fn vertices(&self) -> &[KitVertex] {
        &self.vertices
    }

    /// Adds a triangle whose corners run anticlockwise seen from its front. Triangles with no
    /// area are skipped.
    pub fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, colour: Vec3) -> &mut Self {
        let cross = (b - a).cross(c - a);
        if cross.length() < 1e-9 {
            return self;
        }
        let normal = cross.normalise();
        for position in [a, b, c] {
            self.indices.push(self.vertices.len() as u32);
            self.vertices.push(KitVertex {
                position,
                normal,
                colour,
                sway: 0.0,
                seed: 0.0,
            });
        }
        self
    }

    /// Adds a quad as two triangles, corners anticlockwise seen from the front, so a twisted
    /// quad shows its fold.
    pub fn quad(&mut self, c: [Vec3; 4], colour: Vec3) -> &mut Self {
        self.triangle(c[0], c[1], c[2], colour)
            .triangle(c[0], c[2], c[3], colour)
    }

    /// Adds a copy of `other`.
    pub fn append(&mut self, other: &Shape) -> &mut Self {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
        self
    }

    /// A copy moved by `m`, with its normals recomputed so any transform, scaling included,
    /// keeps the shading right.
    pub fn transformed(&self, m: &Mat4) -> Shape {
        let mut shape = self.clone();
        for v in &mut shape.vertices {
            v.position = transform_point(m, v.position);
        }
        for triangle in shape.vertices.chunks_exact_mut(3) {
            let cross = (triangle[1].position - triangle[0].position)
                .cross(triangle[2].position - triangle[0].position);
            if cross.length() > 1e-12 {
                let normal = cross.normalise();
                triangle.iter_mut().for_each(|v| v.normal = normal);
            }
        }
        shape
    }

    /// Sets every vertex's sway to `weight(position)`.
    pub fn sway_by(mut self, weight: impl Fn(Vec3) -> f32) -> Shape {
        self.vertices
            .iter_mut()
            .for_each(|v| v.sway = weight(v.position));
        self
    }

    /// Sets every vertex's sway from its height: 0 at `base`, 1 at `top`, eased by `power`
    /// (above 1 keeps the lower part stiff).
    pub fn sway_by_height(self, base: f32, top: f32, power: f32) -> Shape {
        self.sway_by(|p| ((p.y - base) / (top - base)).clamp(0.0, 1.0).powf(power))
    }

    /// Colours vertices by height, blending from `bottom` at `y0` to `top` at `y1`, so a blade of
    /// grass or a frond shades smoothly along its length.
    pub fn gradient_by_height(mut self, y0: f32, y1: f32, bottom: Vec3, top: Vec3) -> Shape {
        for v in &mut self.vertices {
            v.colour = lerp3(
                bottom,
                top,
                ((v.position.y - y0) / (y1 - y0)).clamp(0.0, 1.0),
            );
        }
        self
    }

    /// Sets every vertex's sway to `weight`.
    pub fn with_sway(mut self, weight: f32) -> Shape {
        self.vertices.iter_mut().for_each(|v| v.sway = weight);
        self
    }

    /// Sets the instance seed of every vertex.
    pub fn with_seed(mut self, seed: f32) -> Shape {
        self.vertices.iter_mut().for_each(|v| v.seed = seed);
        self
    }

    /// Varies the brightness of each triangle by up to `amount` either way, which makes the
    /// facets read.
    pub fn facet_variation(mut self, rng: &mut Rng, amount: f32) -> Shape {
        for triangle in self.vertices.chunks_exact_mut(3) {
            let k = 1.0 + (rng.unit() - 0.5) * 2.0 * amount;
            triangle.iter_mut().for_each(|v| v.colour = v.colour * k);
        }
        self
    }

    /// Recolours each triangle with `colour(centroid, normal)`, for height and slope based
    /// colouring.
    pub fn colour_by(mut self, colour: impl Fn(Vec3, Vec3) -> Vec3) -> Shape {
        for triangle in self.vertices.chunks_exact_mut(3) {
            let centroid =
                (triangle[0].position + triangle[1].position + triangle[2].position) / 3.0;
            let c = colour(centroid, triangle[0].normal);
            triangle.iter_mut().for_each(|v| v.colour = c);
        }
        self
    }

    /// A solid of revolution about the y axis. `profile` lists (radius, height) from bottom to
    /// top; a radius of 0 closes the shape there. Seen from outside, quads run anticlockwise.
    pub fn lathe(profile: &[(f32, f32)], segments: usize, colour: Vec3) -> Shape {
        let ring = |(radius, y): (f32, f32), i: usize| {
            let angle = TAU * i as f32 / segments as f32;
            Vec3::new(radius * angle.cos(), y, -radius * angle.sin())
        };
        let mut shape = Shape::new();
        for pair in profile.windows(2) {
            for i in 0..segments {
                let (l0, l1) = (ring(pair[0], i), ring(pair[0], i + 1));
                let (u0, u1) = (ring(pair[1], i), ring(pair[1], i + 1));
                match (pair[0].0 == 0.0, pair[1].0 == 0.0) {
                    (true, _) => shape.triangle(l0, u1, u0, colour),
                    (_, true) => shape.triangle(l0, l1, u0, colour),
                    _ => shape.quad([l0, l1, u1, u0], colour),
                };
            }
        }
        shape
    }

    /// A cone standing on the y = 0 plane with a closed base.
    pub fn cone(radius: f32, height: f32, segments: usize, colour: Vec3) -> Shape {
        Shape::lathe(
            &[(0.0, 0.0), (radius, 0.0), (0.0, height)],
            segments,
            colour,
        )
    }

    /// A closed cylinder standing on y = 0, tapering from `bottom` to `top` radius.
    pub fn cylinder(bottom: f32, top: f32, height: f32, segments: usize, colour: Vec3) -> Shape {
        Shape::lathe(
            &[(0.0, 0.0), (bottom, 0.0), (top, height), (0.0, height)],
            segments,
            colour,
        )
    }

    /// An icosphere subdivided `subdivisions` times, each vertex pushed in or out by up to
    /// `roughness` (a fraction of the radius) so it reads as a boulder or a clump of leaves.
    /// The same `seed` gives the same lumps.
    pub fn icosphere(
        radius: f32,
        subdivisions: u32,
        roughness: f32,
        seed: f32,
        colour: Vec3,
    ) -> Shape {
        let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let corners = [
            (-1.0, t, 0.0),
            (1.0, t, 0.0),
            (-1.0, -t, 0.0),
            (1.0, -t, 0.0),
            (0.0, -1.0, t),
            (0.0, 1.0, t),
            (0.0, -1.0, -t),
            (0.0, 1.0, -t),
            (t, 0.0, -1.0),
            (t, 0.0, 1.0),
            (-t, 0.0, -1.0),
            (-t, 0.0, 1.0),
        ]
        .map(|(x, y, z)| Vec3::new(x, y, z).normalise());
        let faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        let mut triangles: Vec<[Vec3; 3]> = faces
            .iter()
            .map(|f| [corners[f[0]], corners[f[1]], corners[f[2]]])
            .collect();
        for _ in 0..subdivisions {
            triangles = triangles
                .iter()
                .flat_map(|&[a, b, c]| {
                    let (ab, bc, ca) = (
                        ((a + b) * 0.5).normalise(),
                        ((b + c) * 0.5).normalise(),
                        ((c + a) * 0.5).normalise(),
                    );
                    [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
                })
                .collect();
        }
        let displace = |d: Vec3| {
            let h = hash2_scalar(
                d.x * 97.3 + seed * 31.7 + d.z * 13.1,
                d.y * 71.9 + d.z * 57.7 - seed * 7.3,
            );
            d * (radius * (1.0 + (h - 0.5) * 2.0 * roughness))
        };
        let mut shape = Shape::new();
        for [a, b, c] in triangles {
            shape.triangle(displace(a), displace(b), displace(c), colour);
        }
        shape
    }

    /// A closed tapered cylinder from point `from` to point `to`, for limbs, twigs and tails.
    pub fn limb(
        from: Vec3,
        to: Vec3,
        radius_from: f32,
        radius_to: f32,
        segments: usize,
        colour: Vec3,
    ) -> Shape {
        let axis = to - from;
        let length = axis.length();
        let d = axis / length;
        let reference = if d.y.abs() < 0.9 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let u = reference.cross(d).normalise();
        let w = u.cross(d);
        let basis = Mat4::new([
            [u.x, d.x, w.x, from.x],
            [u.y, d.y, w.y, from.y],
            [u.z, d.z, w.z, from.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        Shape::cylinder(radius_from, radius_to, length, segments, colour).transformed(&basis)
    }

    /// An axis-aligned box.
    pub fn cuboid(min: Vec3, max: Vec3, colour: Vec3) -> Shape {
        let corner = |i: u8| {
            Vec3::new(
                if i & 1 != 0 { max.x } else { min.x },
                if i & 2 != 0 { max.y } else { min.y },
                if i & 4 != 0 { max.z } else { min.z },
            )
        };
        let mut shape = Shape::new();
        for face in [
            [4, 5, 7, 6],
            [1, 0, 2, 3],
            [5, 1, 3, 7],
            [0, 4, 6, 2],
            [6, 7, 3, 2],
            [0, 1, 5, 4],
        ] {
            shape.quad(face.map(corner), colour);
        }
        shape
    }

    /// A strip along `spine`, `widths[i]` wide at point `i` across the direction `across`, as
    /// for a blade of kelp or a fern frond. One-sided: draw it with culling off.
    pub fn ribbon(spine: &[Vec3], widths: &[f32], across: Vec3, colour: Vec3) -> Shape {
        let edge = |i: usize, side: f32| spine[i] + across * (widths[i] * 0.5 * side);
        let mut shape = Shape::new();
        for i in 0..spine.len() - 1 {
            shape.quad(
                [
                    edge(i, -1.0),
                    edge(i + 1, -1.0),
                    edge(i + 1, 1.0),
                    edge(i, 1.0),
                ],
                colour,
            );
        }
        shape
    }

    /// Queues the shape on `pass`.
    pub fn draw<'a, VS, FS>(
        &'a self,
        pass: &mut RenderPass<'_, 'a>,
        pipeline: &'a SimdPipeline<VS, FS>,
        vertex_uniforms: VS::Uniforms,
        fragment_uniforms: FS::Uniforms,
    ) where
        VS: VertexShader<Vertex = KitVertex>,
        VS::Varyings: SimdInterpolate,
        FS: FragmentShaderSimd<VS::Varyings> + Send + Sync + 'static,
    {
        pass.draw_simd(
            pipeline,
            DrawCall::new(
                &self.vertices,
                &self.indices,
                PrimitiveMode::TRIANGLES,
                fragment_uniforms,
            ),
            vertex_uniforms,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limbs_run_between_their_end_points() {
        let (from, to) = (Vec3::new(1.0, 2.0, 3.0), Vec3::new(-2.0, 5.0, 0.5));
        let limb = Shape::limb(from, to, 0.2, 0.1, 6, Vec3::ONE);
        let along = |p: Vec3| (p - from).dot(to - from) / (to - from).length();
        let (low, high) = limb
            .vertices()
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), v| {
                (l.min(along(v.position)), h.max(along(v.position)))
            });
        assert!(low.abs() < 1e-4 && (high - (to - from).length()).abs() < 1e-4);
        for t in limb.vertices().chunks_exact(3) {
            let centre = (t[0].position + t[1].position + t[2].position) / 3.0;
            let on_axis = from + (to - from) * (along(centre) / (to - from).length());
            assert!(t[0].normal.dot(centre - on_axis) >= -1e-4);
        }
    }

    #[test]
    fn solids_face_outwards() {
        let white = Vec3::ONE;
        for shape in [
            Shape::icosphere(1.0, 1, 0.2, 3.0, white),
            Shape::cone(1.0, 2.0, 8, white).transformed(&Mat4::translation(0.0, -1.0, 0.0)),
            Shape::cuboid(Vec3::new(-1.0, -1.0, -1.0), Vec3::ONE, white),
        ] {
            for t in shape.vertices().chunks_exact(3) {
                let centroid = (t[0].position + t[1].position + t[2].position) / 3.0;
                assert!(t[0].normal.dot(centroid) > 0.0);
            }
        }
    }
}
