//! The studio's spectacle: the teapot projected out of the die, the hologram's light frustum and
//! what leaks out of it, the build-up's tension, and the eruption's burst of the renderer's
//! primitives.
//!
//! Everything is placed in the chip's own space through [`Frame::chip`](crate::shaders::Frame::chip),
//! so it follows the chip's drift and shake. Each effect keeps its shaders in its own module;
//! `streak` and `primitives` hold the two kinds of light they share.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::director::Phase;
use crate::shaders::{Frame, ViewRef};
use crate::simd::hash2_scalar;
use crate::themes::Scene;
use buildup::{DrawnVertexShader, GlowVertexShader, InflowVertexShader};
use eruption::{BurstUniforms, BurstVertexShader, RadialVertexShader, ShockVertexShader};
use frustum::{FaceShader, FaceVertexShader, RayVertexShader, ScanVertexShader};
use hologram::{HologramShader, HologramUniforms, HologramVertexShader};
use leak::LeakVertexShader;
use primitives::{SpriteShader, SpriteVertex};
use streak::{StreakShader, StreakVertex};

mod buildup;
pub mod eruption;
pub mod frustum;
pub mod hologram;
mod leak;
mod primitives;
mod streak;
pub mod teapot;

/// Where the die's face is in the chip's space: its height, and half its width.
const DIE_TOP: f32 = 0.2;
const DIE_HALF: f32 = 0.9;

/// The accent light's body and its hot core (the style brief's data green).
const GREEN: Vec3 = Vec3 {
    x: 0.18,
    y: 0.71,
    z: 0.54,
};
const GREEN_BRIGHT: Vec3 = Vec3 {
    x: 0.43,
    y: 0.91,
    z: 0.72,
};
/// The eruption's warm white, and the palette's gold.
const WARM: Vec3 = Vec3 {
    x: 1.0,
    y: 0.85,
    z: 0.63,
};
const GOLD: Vec3 = Vec3 {
    x: 0.94,
    y: 0.84,
    z: 0.54,
};

/// A pipeline for light: additive, depth tested against the scene but never written.
fn light<VS, FS>(vertex_shader: VS, fragment_shader: FS) -> SimdPipeline<VS, FS>
where
    VS: VertexShader<Varyings: SimdInterpolate>,
    FS: FragmentShaderSimd<VS::Varyings>,
{
    SimdPipeline::new(vertex_shader, fragment_shader)
        .with_culling_mode(CullingMode::None)
        .with_depth_state(DepthState::READ_ONLY)
        .with_blend_state(BlendState::ADDITIVE)
}

pub struct Effects {
    teapot: teapot::Teapot,
    rays: Model<StreakVertex<frustum::Ray>>,
    scans: Model<StreakVertex<frustum::Scan>>,
    faces: Model<frustum::FaceVertex>,
    hologram_cube: Model<ObjVertex>,
    leaks: Model<SpriteVertex<leak::Leak>>,
    inflow: Model<StreakVertex<buildup::Inflow>>,
    drawn: Model<SpriteVertex<buildup::Drawn>>,
    glow: Model<SpriteVertex<()>>,
    radials: Model<StreakVertex<eruption::Radial>>,
    shocks: Model<SpriteVertex<eruption::Shock>>,
    burst: Model<SpriteVertex<eruption::Burst>>,
    ray_pipeline: SimdPipeline<RayVertexShader, StreakShader>,
    scan_pipeline: SimdPipeline<ScanVertexShader, StreakShader>,
    face_pipeline: SimdPipeline<FaceVertexShader, FaceShader>,
    hologram_pipeline: SimdPipeline<HologramVertexShader, HologramShader>,
    leak_pipeline: SimdPipeline<LeakVertexShader, SpriteShader>,
    inflow_pipeline: SimdPipeline<InflowVertexShader, StreakShader>,
    drawn_pipeline: SimdPipeline<DrawnVertexShader, SpriteShader>,
    glow_pipeline: SimdPipeline<GlowVertexShader, SpriteShader>,
    radial_pipeline: SimdPipeline<RadialVertexShader, StreakShader>,
    shock_pipeline: SimdPipeline<ShockVertexShader, SpriteShader>,
    burst_pipeline: SimdPipeline<BurstVertexShader, SpriteShader>,
}

impl Effects {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            teapot: teapot::Teapot::new()?,
            rays: quad_model(frustum::rays()),
            scans: quad_model(frustum::scans()),
            faces: frustum::faces(),
            hologram_cube: Model::new(
                vec![Mesh::cube(Colour::WHITE, None)],
                Vec::new(),
                ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
            )?,
            leaks: leak::leaks(),
            inflow: quad_model(buildup::streaks()),
            drawn: buildup::primitives(),
            glow: buildup::glow(),
            radials: quad_model(eruption::rays()),
            shocks: eruption::shocks(),
            burst: eruption::primitives(),
            ray_pipeline: light(RayVertexShader, StreakShader),
            scan_pipeline: light(ScanVertexShader, StreakShader),
            face_pipeline: light(FaceVertexShader, FaceShader),
            hologram_pipeline: SimdPipeline::new(HologramVertexShader, HologramShader)
                .with_depth_state(DepthState::READ_ONLY)
                .with_blend_state(BlendState::ADDITIVE),
            leak_pipeline: light(LeakVertexShader, SpriteShader),
            inflow_pipeline: light(InflowVertexShader, StreakShader),
            drawn_pipeline: light(DrawnVertexShader, SpriteShader),
            glow_pipeline: light(GlowVertexShader, SpriteShader),
            radial_pipeline: light(RadialVertexShader, StreakShader),
            shock_pipeline: light(ShockVertexShader, SpriteShader),
            burst_pipeline: light(BurstVertexShader, SpriteShader),
        })
    }

    /// Renders anything the effects need before the scene pass.
    pub fn prepare(&mut self, _context: &mut RenderContext<'_>, _scene: &Scene<'_>) {}

    /// Draws the effects' opaque surfaces, which write depth, before the backdrop.
    pub fn draw_opaque<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        self.teapot.draw_opaque(pass, &scene.view);
    }

    /// Draws the effects' light: additive, depth tested but not written, after the backdrop.
    pub fn draw_light<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let time = view.frame.time;
        let during = |start: f32, end: f32| (start..end).contains(&time);
        let hologram = Phase::Hologram.start();
        let eruption = Phase::Eruption.start();

        self.teapot.draw_light(pass, view);
        if during(hologram + 0.2, hologram + 12.5) {
            self.faces
                .draw_to_render_pass_simd(pass, &self.face_pipeline, view.clone(), |_| {
                    view.clone()
                });
            self.rays
                .draw_to_render_pass_simd(pass, &self.ray_pipeline, view.clone(), |_| ());
            self.scans
                .draw_to_render_pass_simd(pass, &self.scan_pipeline, view.clone(), |_| ());
        }
        if during(hologram + 2.2, hologram + 12.5) {
            self.leaks
                .draw_to_render_pass_simd(pass, &self.leak_pipeline, view.clone(), |_| ());
        }
        if during(hologram + 1.9, hologram + 12.3)
            && let Some(uniforms) = HologramUniforms::new(view)
        {
            self.hologram_cube.draw_to_render_pass_simd(
                pass,
                &self.hologram_pipeline,
                uniforms.clone(),
                |_| uniforms.clone(),
            );
        }
        if during(Phase::BuildUp.start(), eruption + 0.1) {
            self.inflow
                .draw_to_render_pass_simd(pass, &self.inflow_pipeline, view.clone(), |_| ());
            self.drawn
                .draw_to_render_pass_simd(pass, &self.drawn_pipeline, view.clone(), |_| ());
        }
        if during(eruption - 2.3, Phase::Worlds.start() + 0.1) {
            let uniforms = BurstUniforms::new(view);
            self.radials.draw_to_render_pass_simd(
                pass,
                &self.radial_pipeline,
                uniforms.clone(),
                |_| (),
            );
            if time >= eruption {
                self.shocks.draw_to_render_pass_simd(
                    pass,
                    &self.shock_pipeline,
                    uniforms.clone(),
                    |_| (),
                );
                self.burst
                    .draw_to_render_pass_simd(pass, &self.burst_pipeline, uniforms, |_| ());
            }
        }
        if during(hologram, eruption + 2.1) {
            self.glow
                .draw_to_render_pass_simd(pass, &self.glow_pipeline, view.clone(), |_| ());
        }
    }
}

/// A pseudo-random number in [0, 1) from two seeds.
fn random(a: f32, b: f32) -> f32 {
    hash2_scalar(a, b)
}

/// A pseudo-random unit vector from two seeds, uniform over the sphere.
fn random_direction(a: f32, b: f32) -> Vec3 {
    let z = random(a, b + 11.0) * 2.0 - 1.0;
    let angle = random(a + 5.0, b + 23.0) * TAU;
    let ring = (1.0 - z * z).sqrt();
    Vec3::new(ring * angle.cos(), ring * angle.sin(), z)
}

/// The middle of the die's face in the world.
fn die_centre(frame: &Frame) -> Vec3 {
    (frame.chip * Vec4::new(0.0, DIE_TOP + 0.03, 0.0, 1.0)).xyz()
}

/// A model of one mesh of `quads`, four corners each in the order the corners of a quad go round.
fn quad_model<V: Clone>(quads: impl IntoIterator<Item = [V; 4]>) -> Model<V> {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for quad in quads {
        let base = vertices.len() as u32;
        vertices.extend(quad);
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Model::new(
        vec![Mesh::new(vertices, indices, None)],
        Vec::new(),
        ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
    )
    .expect("a model with one mesh and no materials is always valid")
}

/// Whether `view`'s frame is in `phase`.
fn in_phase(view: &ViewRef, phase: crate::director::Phase) -> bool {
    view.frame.phase == phase
}
