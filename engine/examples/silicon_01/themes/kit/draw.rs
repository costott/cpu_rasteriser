//! The kit's vertex shader and toon fragment shader.
//!
//! [`KitVertexShader`] draws a [`Shape`](super::shape::Shape) in world space and animates it with
//! a [`Motion`]: sway for plants, orbiting swimmers for shoals. [`ToonShader`] lights it with a
//! [`Lighting`]. A world with a surface of its own (shadowed ground, caustics) writes its own
//! fragment shader on the same [`KitVaryings`], reading them through the accessors on
//! [`KitVaryingsSimd`], and builds its pipeline with [`pipeline`].

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use super::light::{Lighting, facing};
use super::shape::{KitVertex, transform_point};
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{F, hash2, mul3, s, splat3, v3};

/// How the vertex shader animates a draw's vertices.
#[derive(Clone, Copy, Debug)]
pub enum Motion {
    Still,
    /// Bends vertices by their sway weight, as wind through trees and grass or a current
    /// through kelp.
    Sway(Sway),
    /// Moves each instance round a circle, by its seed, as a shoal of fish.
    Orbit(Orbit),
    /// Treats positions as the chip's own space and poses them with the chip, for things
    /// attached to it.
    OnChip,
}

#[derive(Clone, Copy, Debug)]
pub struct Sway {
    /// The largest displacement, at a sway weight of 1, in world units.
    pub amplitude: f32,
    /// Radians per second of the main swing.
    pub speed: f32,
    /// The horizontal direction of the wind or current (x, z), unit length.
    pub wind: Vec2,
}

/// Instances placed at the origin with their nose along +x, up +y and right +z swim in circles
/// about `centre`: each seed picks its own radius, height, speed and phase from the ranges. The
/// vertex sway weight drives a tail wiggle, strongest where it is 1.
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub centre: Vec3,
    pub radius: (f32, f32),
    pub height: (f32, f32),
    /// Swimming speed in world units per second.
    pub speed: (f32, f32),
    /// The tail's sideways swing, in world units.
    pub wiggle: f32,
}

#[derive(Clone)]
pub struct KitUniforms {
    pub view: ViewRef,
    pub motion: Motion,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct KitVaryings {
    world: Vec3,
    normal: Vec3,
    colour: Vec3,
    /// The instance seed and the sway weight.
    tag: Vec2,
}

impl KitVaryingsSimd {
    /// The world-space position.
    pub fn world(&self) -> Vec3Simd {
        v3(self.world)
    }

    /// The interpolated normal, not yet normalised or turned to face the eye.
    pub fn normal(&self) -> Vec3Simd {
        v3(self.normal)
    }

    /// The interpolated vertex colour, before lighting.
    pub fn colour(&self) -> Vec3Simd {
        v3(self.colour)
    }

    /// The instance seed, in [0, 1).
    pub fn seed(&self) -> F {
        self.tag[0]
    }
}

/// Places a [`KitVertex`] in the world and animates it with the pipeline's [`Motion`].
pub struct KitVertexShader;

impl VertexShader for KitVertexShader {
    type Vertex = KitVertex;
    type Uniforms = KitUniforms;
    type Varyings = KitVaryings;

    fn shade(&self, vertex: KitVertex, u: &KitUniforms) -> (Vec4, KitVaryings) {
        let time = u.view.frame.time;
        let (mut position, mut normal) = (vertex.position, vertex.normal);
        match u.motion {
            Motion::Still => {}
            Motion::OnChip => {
                let chip = u.view.frame.chip;
                position = transform_point(&chip, position);
                normal = (chip * normal.to_direction4()).xyz();
            }
            Motion::Sway(sway) => {
                position = swayed(position, vertex.sway, vertex.seed, time, &sway)
            }
            Motion::Orbit(orbit) => {
                (position, normal) =
                    orbited(position, normal, vertex.sway, vertex.seed, time, &orbit)
            }
        }
        (
            u.view.view_projection * position.to_point4(),
            KitVaryings {
                world: position,
                normal,
                colour: vertex.colour,
                tag: Vec2::new(vertex.seed, vertex.sway),
            },
        )
    }
}

fn swayed(p: Vec3, weight: f32, seed: f32, time: f32, sway: &Sway) -> Vec3 {
    let phase = time * sway.speed + seed * TAU + p.x * 0.31 + p.z * 0.27;
    let swing = phase.sin() + 0.5 * (phase * 2.3 + 1.7).sin();
    let push = weight * sway.amplitude * swing;
    Vec3::new(
        p.x + sway.wind.x * push - sway.wind.y * push * 0.3,
        p.y,
        p.z + sway.wind.y * push + sway.wind.x * push * 0.3,
    )
}

fn orbited(p: Vec3, n: Vec3, weight: f32, seed: f32, time: f32, orbit: &Orbit) -> (Vec3, Vec3) {
    let pick = |k: f32, offset: f32| (seed * k + offset).fract();
    let lerp = |range: (f32, f32), t: f32| range.0 + (range.1 - range.0) * t;
    let radius = lerp(orbit.radius, pick(7.31, 0.0));
    let direction = if pick(29.3, 0.4) < 0.1 { -1.0 } else { 1.0 };
    let angular = lerp(orbit.speed, pick(13.17, 0.3)) / radius * direction;
    let angle = seed * TAU * 5.0 + time * angular;
    let height = lerp(orbit.height, pick(3.91, 0.7)) + 0.25 * (time * 0.9 + seed * 40.0).sin();
    let heading = (-angle.sin() * direction, angle.cos() * direction);
    let wiggle = (time * 7.0 * orbit.speed.0.max(0.5) + p.x * 4.0 + seed * TAU).sin()
        * orbit.wiggle
        * weight;
    let local = (p.x, p.z + wiggle);
    let rotate = |x: f32, z: f32| (x * heading.0 - z * heading.1, x * heading.1 + z * heading.0);
    let (x, z) = rotate(local.0, local.1);
    let (nx, nz) = rotate(n.x, n.z);
    let centre = orbit.centre + Vec3::new(radius * angle.cos(), height, radius * angle.sin());
    (
        Vec3::new(centre.x + x, centre.y + p.y, centre.z + z),
        Vec3::new(nx, n.y, nz),
    )
}

/// A kit pipeline: the kit vertex shader with `fragment`, depth-tested and written.
pub fn pipeline<FS>(fragment: FS, culling: CullingMode) -> SimdPipeline<KitVertexShader, FS>
where
    FS: FragmentShaderSimd<KitVaryings> + Send + Sync + 'static,
{
    SimdPipeline::new(KitVertexShader, fragment)
        .with_culling_mode(culling)
        .with_depth_state(DepthState::DEFAULT)
}

#[derive(Clone)]
pub struct ToonUniforms {
    pub view: ViewRef,
    pub lighting: Lighting,
}

/// Lights surfaces with the world's [`Lighting`], using vertex colour as albedo.
pub struct ToonShader;

impl FragmentShaderSimd<KitVaryings> for ToonShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let world = v.world();
        let to_eye = splat3(u.view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        let colour = u.lighting.shade(&u.view, world, normal, v.colour(), s(1.0));
        scene_colour(&u.view, colour, world)
    }
}

#[derive(Clone)]
pub struct VariedToonUniforms {
    pub toon: ToonUniforms,
    /// How much each instance's colour varies by its seed: brightness, and a shift between warm
    /// and cool. 0 for none.
    pub variation: f32,
}

/// [`ToonShader`] with each instance's colour varied by its seed, so a baked forest of copies
/// doesn't look stamped.
pub struct VariedToonShader;

impl FragmentShaderSimd<KitVaryings> for VariedToonShader {
    type Uniforms = VariedToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &VariedToonUniforms) -> ColourSimd {
        let world = v.world();
        let to_eye = splat3(u.toon.view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        let d = (hash2(v.seed() * s(91.7), s(3.1)) - s(0.5)) * s(2.0 * u.variation);
        let albedo = mul3(
            v.colour(),
            Vec3Simd::new(s(1.0) + d, s(1.0), s(1.0) - d * s(0.6)),
        );
        let colour = u
            .toon
            .lighting
            .shade(&u.toon.view, world, normal, albedo, s(1.0));
        scene_colour(&u.toon.view, colour, world)
    }
}
