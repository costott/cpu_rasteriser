//! Baked shadows on the ground: the sun's shadows of a world's static props, rasterised once into
//! a soft top-down texture that ground shaders sample.
//!
//! This costs one texture lookup per ground pixel and nothing per frame, so a golden-hour world
//! can have long shadows. Moving things (the chip) shade the ground separately; see
//! [`chip_shadow`](super::skin::chip_shadow).

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use super::shape::Shape;
use crate::simd::{F, s};

/// A baked texture of the shadows a static scene casts on the ground. Build it once, then look
/// it up per ground pixel.
pub struct GroundShadow {
    texture: TextureSampler,
    min: Vec2,
    size: f32,
}

impl GroundShadow {
    /// Bakes the shadow of `casters` on the ground over the square from `min` (x, z) and `size`
    /// on a side, in a `resolution` square texture, for the sun in direction `sun` (towards it).
    /// `height(x, z)` is the ground's surface, so shadows follow it; `blur` softens the edges, in
    /// texels.
    pub fn bake(
        min: Vec2,
        size: f32,
        resolution: usize,
        sun: Vec3,
        height: impl Fn(f32, f32) -> f32,
        casters: &[&Shape],
        blur: usize,
    ) -> Self {
        let to_texel = |p: Vec3| {
            (
                (p.x - min.x) / size * resolution as f32,
                (p.z - min.y) / size * resolution as f32,
            )
        };
        let onto_ground = |p: Vec3| {
            let mut ground = height(p.x, p.z);
            let mut q = p;
            for _ in 0..3 {
                q = p - sun * ((p.y - ground) / sun.y);
                ground = height(q.x, q.z);
            }
            q
        };
        let mut cover = vec![0.0_f32; resolution * resolution];
        for shape in casters {
            for triangle in shape.vertices().chunks_exact(3) {
                let corners = [0, 1, 2].map(|i| to_texel(onto_ground(triangle[i].position)));
                fill(&mut cover, resolution, corners);
            }
        }
        for _ in 0..2 {
            cover = box_blur(&cover, resolution, blur, true);
            cover = box_blur(&cover, resolution, blur, false);
        }
        let mut pixels = vec![Colour::new(1.0, 1.0, 1.0, 1.0); resolution * resolution];
        for j in 0..resolution {
            for i in 0..resolution {
                let lit = 1.0 - cover[j * resolution + i].min(1.0);
                pixels[(resolution - 1 - j) * resolution + i] = Colour::new(lit, lit, lit, 1.0);
            }
        }
        let pixels: Arc<[Colour]> = pixels.into();
        let texture = TextureSampler::new(
            pixels,
            resolution as u32,
            resolution as u32,
            WrapMode::Clamp,
            FilterMode::Linear,
        );
        Self { texture, min, size }
    }

    /// The fraction of the sun that reaches the ground at (`x`, `z`): 1 in the open, less in shade.
    #[inline(always)]
    pub fn lit(&self, x: F, z: F) -> F {
        let u = (x - s(self.min.x)) * s(1.0 / self.size);
        let v = (z - s(self.min.y)) * s(1.0 / self.size);
        self.texture.sample_linear_clamp_simd([u, v]).r
    }
}

fn fill(cover: &mut [f32], resolution: usize, c: [(f32, f32); 3]) {
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
    };
    let area = edge(c[0], c[1], c[2]);
    if area.abs() < 1e-6 {
        return;
    }
    let bounds = |axis: fn(&(f32, f32)) -> f32| {
        let values = c.iter().map(axis);
        let low = values.clone().fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let high = (values.fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(resolution);
        (low, high)
    };
    let (x0, x1) = bounds(|p| p.0);
    let (y0, y1) = bounds(|p| p.1);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let w = [
                edge(c[1], c[2], p),
                edge(c[2], c[0], p),
                edge(c[0], c[1], p),
            ]
            .map(|w| w / area);
            if w.iter().all(|&w| w >= -0.02) {
                cover[y * resolution + x] = 1.0;
            }
        }
    }
}

fn box_blur(source: &[f32], resolution: usize, radius: usize, horizontal: bool) -> Vec<f32> {
    let mut out = vec![0.0; source.len()];
    let r = radius as isize;
    for j in 0..resolution {
        for i in 0..resolution {
            let mut sum = 0.0;
            for k in -r..=r {
                let (x, y) = if horizontal {
                    (i as isize + k, j as isize)
                } else {
                    (i as isize, j as isize + k)
                };
                if (0..resolution as isize).contains(&x) && (0..resolution as isize).contains(&y) {
                    sum += source[y as usize * resolution + x as usize];
                }
            }
            out[j * resolution + i] = sum / (2 * radius + 1) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wall_shades_the_ground_away_from_the_sun() {
        let wall = Shape::cuboid(
            Vec3::new(-1.0, 0.0, -0.1),
            Vec3::new(1.0, 2.0, 0.1),
            Vec3::ONE,
        );
        let sun = Vec3::new(0.0, 1.0, 1.0).normalise();
        let shadow = GroundShadow::bake(
            Vec2::new(-8.0, -8.0),
            16.0,
            64,
            sun,
            |_, _| 0.0,
            &[&wall],
            1,
        );
        let at = |x: f32, z: f32| shadow.lit(s(x), s(z)).to_array()[0];
        assert!(at(0.0, -1.5) < 0.2, "behind the wall, away from the sun");
        assert!(at(0.0, 3.0) > 0.9, "on the sun's side");
    }
}
