//! Drifting glowing specks (pollen, bubbles, plankton, dust, snow), all animated in the vertex
//! shader.
//!
//! A [`Particles`] is a set of camera-facing quads, each with a fixed random identity. Each frame
//! the vertex shader places a particle by wrapping its identity plus velocity times time inside a
//! box, so the field loops forever with no per-frame work on the CPU. Draw it after the sky with
//! additive blending and a read-only depth test.

use cpu_rasteriser::prelude::*;

use super::random::Rng;
use crate::shaders::ViewRef;
use crate::simd::{F, s, to_colour};

/// Where the specks live and how they move and look.
#[derive(Clone, Copy, Debug)]
pub struct Drift {
    pub centre: Vec3,
    /// Half the box's size on each axis; particles wrap round its edges, fading out as they go.
    pub half_extent: Vec3,
    /// World units per second.
    pub velocity: Vec3,
    /// How far each speck wanders round its path, in world units.
    pub wander: f32,
    /// Speck radius range, in world units.
    pub size: (f32, f32),
    pub colour: Vec3,
    /// How fast each speck's brightness pulses, in radians per second; 0 for steady.
    pub twinkle: f32,
}

#[derive(Clone, Copy)]
pub struct ParticleVertex {
    corner: Vec2,
    random: Vec3,
    size: f32,
}

#[derive(Clone)]
pub struct ParticleUniforms {
    pub view: ViewRef,
    pub drift: Drift,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct ParticleVaryings {
    corner: Vec2,
    colour: Vec3,
}

/// Moves each particle along its path and expands it to a camera-facing quad.
pub struct ParticleVertexShader;

impl VertexShader for ParticleVertexShader {
    type Vertex = ParticleVertex;
    type Uniforms = ParticleUniforms;
    type Varyings = ParticleVaryings;

    fn shade(&self, v: ParticleVertex, u: &ParticleUniforms) -> (Vec4, ParticleVaryings) {
        let (view, d) = (&u.view, &u.drift);
        let time = view.frame.time;
        let wrap = |random: f32, velocity: f32, half: f32| {
            let t = random + velocity * time / (2.0 * half);
            (t - t.floor()) * 2.0 - 1.0
        };
        let local = Vec3::new(
            wrap(v.random.x, d.velocity.x, d.half_extent.x),
            wrap(v.random.y, d.velocity.y, d.half_extent.y),
            wrap(v.random.z, d.velocity.z, d.half_extent.z),
        );
        let edge = |t: f32| (1.0 - t.abs()).clamp(0.0, 1.0);
        let fade = (edge(local.x) * edge(local.y) * edge(local.z) * 6.0).min(1.0);
        let phase = v.random.x * 97.0 + v.random.z * 41.0;
        let wander = Vec3::new(
            (time * 0.7 + phase).sin(),
            (time * 0.5 + phase * 1.3).sin(),
            (time * 0.6 + phase * 0.7).cos(),
        ) * d.wander;
        let pulse = if d.twinkle > 0.0 {
            0.65 + 0.35 * (time * d.twinkle + phase).sin()
        } else {
            1.0
        };
        let radius = d.size.0 + (d.size.1 - d.size.0) * v.size;
        let centre = d.centre
            + Vec3::new(
                local.x * d.half_extent.x,
                local.y * d.half_extent.y,
                local.z * d.half_extent.z,
            )
            + wander;
        let corner = centre + (view.right * v.corner.x + view.up * v.corner.y) * radius;
        (
            view.view_projection * corner.to_point4(),
            ParticleVaryings {
                corner: v.corner,
                colour: d.colour * (fade * pulse),
            },
        )
    }
}

/// A soft round glow, added to what is behind it.
pub struct ParticleShader;

impl FragmentShaderSimd<ParticleVaryings> for ParticleShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: ParticleVaryingsSimd, _: &()) -> ColourSimd {
        let (x, y) = (v.corner_x(), v.corner_y());
        let d2: F = x * x + y * y;
        let falloff = (s(1.0) - d2).fast_max(s(0.0));
        let glow = falloff * falloff * falloff;
        to_colour(v.colour() * glow, s(0.0))
    }
}

impl ParticleVaryingsSimd {
    #[inline(always)]
    fn corner_x(&self) -> F {
        self.corner[0]
    }

    #[inline(always)]
    fn corner_y(&self) -> F {
        self.corner[1]
    }

    #[inline(always)]
    fn colour(&self) -> Vec3Simd {
        Vec3Simd::new(self.colour[0], self.colour[1], self.colour[2])
    }
}

/// A particle field and its pipeline.
pub struct Particles {
    vertices: Vec<ParticleVertex>,
    indices: Vec<u32>,
    pipeline: SimdPipeline<ParticleVertexShader, ParticleShader>,
}

impl Particles {
    /// `count` particles with identities from `seed`.
    pub fn new(count: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut vertices = Vec::with_capacity(count * 4);
        let mut indices = Vec::with_capacity(count * 6);
        for i in 0..count as u32 {
            let random = Vec3::new(rng.unit(), rng.unit(), rng.unit());
            let size = rng.unit();
            for corner in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                vertices.push(ParticleVertex {
                    corner: Vec2::new(corner.0, corner.1),
                    random,
                    size,
                });
            }
            indices.extend([0, 1, 2, 0, 2, 3].map(|k| i * 4 + k));
        }
        Self {
            vertices,
            indices,
            pipeline: SimdPipeline::new(ParticleVertexShader, ParticleShader)
                .with_culling_mode(CullingMode::None)
                .with_blend_state(BlendState::ADDITIVE)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }

    /// Queues the field on `pass`, which should already hold the opaque world and its backdrop.
    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, view: &ViewRef, drift: Drift) {
        pass.draw_simd(
            &self.pipeline,
            DrawCall::new(&self.vertices, &self.indices, PrimitiveMode::TRIANGLES, ()),
            ParticleUniforms {
                view: view.clone(),
                drift,
            },
        );
    }
}
