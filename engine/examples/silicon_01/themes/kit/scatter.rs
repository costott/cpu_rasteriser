//! Seeded scattering of props over ground, and baking the copies into one mesh.

use cpu_rasteriser::prelude::*;

use super::random::Rng;
use super::shape::Shape;

/// Where and how one copy of a prop stands.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub position: Vec3,
    pub yaw: f32,
    pub scale: f32,
    /// In [0, 1): the instance's identity, for motion phase and for choosing a variant.
    pub seed: f32,
}

/// Scatters points over the ground by dart throwing, no two closer than `spacing`.
///
/// `accept(x, z)` returns the spacing multiplier at a spot (`None` rejects it), so a world can
/// thin props out near the chip or in a path, or let them crowd in thickets. Points come back in
/// the order they were accepted. `height(x, z)` puts each on the surface.
pub fn scatter(
    rng: &mut Rng,
    half_extent: f32,
    spacing: f32,
    attempts: usize,
    height: impl Fn(f32, f32) -> f32,
    accept: impl Fn(f32, f32) -> Option<f32>,
    scale: (f32, f32),
) -> Vec<Placement> {
    let mut placed: Vec<(f32, f32, f32)> = Vec::new();
    let mut out = Vec::new();
    for _ in 0..attempts {
        let (x, z) = (
            rng.range(-half_extent, half_extent),
            rng.range(-half_extent, half_extent),
        );
        let Some(factor) = accept(x, z) else { continue };
        let need = spacing * factor;
        if placed.iter().any(|&(px, pz, r)| {
            let limit = need.max(r);
            (px - x).powi(2) + (pz - z).powi(2) < limit * limit
        }) {
            continue;
        }
        placed.push((x, z, need));
        out.push(Placement {
            position: Vec3::new(x, height(x, z), z),
            yaw: rng.range(0.0, std::f32::consts::TAU),
            scale: rng.range(scale.0, scale.1),
            seed: rng.unit(),
        });
    }
    out
}

/// Many copies of shapes baked into one mesh, for a single draw.
///
/// Each copy is moved to its [`Placement`], and its vertices take the placement's seed, so a
/// [`Motion`](super::draw::Motion) in the vertex shader can sway or swim each copy out of step
/// with the rest.
#[derive(Default)]
pub struct Batch {
    shape: Shape,
}

impl Batch {
    /// An empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a copy of `variants[pick(seed)]` at each placement, where `pick` maps the seed to a
    /// variant index.
    pub fn add(
        &mut self,
        variants: &[Shape],
        placements: &[Placement],
        pick: impl Fn(&Placement) -> usize,
    ) -> &mut Self {
        for p in placements {
            let m = Mat4::translation_vec(p.position)
                * Mat4::rotate_y(p.yaw)
                * Mat4::scaling(p.scale, p.scale, p.scale);
            self.shape
                .append(&variants[pick(p)].transformed(&m).with_seed(p.seed));
        }
        self
    }

    /// Ends the batch and returns the combined mesh.
    pub fn finish(self) -> Shape {
        self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scattered_points_respect_the_spacing() {
        let points = scatter(
            &mut Rng::new(1),
            10.0,
            2.0,
            2000,
            |_, _| 0.0,
            |_, _| Some(1.0),
            (1.0, 1.0),
        );
        assert!(points.len() > 10);
        for (i, a) in points.iter().enumerate() {
            for b in &points[i + 1..] {
                assert!((a.position - b.position).length() >= 2.0 - 1e-4);
            }
        }
    }
}
