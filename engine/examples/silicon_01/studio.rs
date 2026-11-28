//! The studio: the chip alone on a dark backdrop, where the film opens, projects, erupts and
//! closes.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::director::ramp;
use crate::effects::Effects;
use crate::shaders::View;
use crate::shaders::screen::{ScreenVaryings, ScreenVertexShader};
use crate::shaders::studio::{
    BackdropShader, BackdropUniforms, DepthOfFieldShader, DepthOfFieldUniforms,
    EdgeSmoothingShader, EdgeSmoothingUniforms,
};
use crate::themes::{Scene, Stage};

/// The circle of confusion's largest radius in the macro shots, in pixels of a 720-line frame:
/// the style brief's 6-8 px.
const BLUR: f32 = 7.0;
/// When the macro shots' depth of field opens out to everything in focus, as the camera pulls
/// back for the reveal.
const FOCUS_PULL: (f32, f32) = (11.5, 14.5);

pub struct Studio {
    look: Look,
    effects: Effects,
    backdrop: SimdPipeline<ScreenVertexShader, BackdropShader>,
    edge_smoothing: SimdPipeline<ScreenVertexShader, EdgeSmoothingShader>,
    depth_of_field: SimdPipeline<ScreenVertexShader, DepthOfFieldShader>,
    /// The scene with its edges smoothed, and blurred by depth, each made when first needed.
    smoothed: Option<(RenderTarget, Texture)>,
    focused: Option<(RenderTarget, Texture)>,
}

impl Studio {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            look: Look::default(),
            effects: Effects::new()?,
            backdrop: SimdPipeline::new(ScreenVertexShader, BackdropShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            edge_smoothing: post(EdgeSmoothingShader),
            depth_of_field: post(DepthOfFieldShader),
            smoothed: None,
            focused: None,
        })
    }
}

impl Stage for Studio {
    fn prepare(&mut self, context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.effects.prepare(context, scene);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        self.effects.draw_opaque(pass, scene);
        // The backdrop after opaque surfaces, so it is shaded only where nothing covers it.
        let view = &scene.view;
        let centre = view.view_projection * (view.frame.chip * Vec4::new(0.0, 0.0, 0.0, 1.0));
        let uniforms = BackdropUniforms {
            centre: Vec2::new(centre.x, centre.y) * (1.0 / centre.w.max(1e-3)),
            aspect: view.aspect,
            light: self.look.streams * view.frame.activity,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), move |_| uniforms);
        self.effects.draw_light(pass, scene);
    }

    /// Smooths the scene's jagged edges, and in the macro shots blurs it by depth.
    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let smoothed = full_screen(
            context,
            scene,
            &self.edge_smoothing,
            EdgeSmoothingUniforms { source: image },
            &mut self.smoothed,
        );
        let view = &scene.view;
        let opening = 1.0 - ramp(view.frame.time, FOCUS_PULL.0, FOCUS_PULL.1);
        if opening <= 0.0 {
            return Some(smoothed);
        }
        let scale = scene.extent.height as f32 / 720.0;
        let uniforms = DepthOfFieldUniforms {
            source: smoothed,
            focus: focus_distance(view),
            aperture: BLUR * 2.0 * scale * opening,
            reach: (BLUR * scale * opening).max(1.0),
        };
        Some(full_screen(
            context,
            scene,
            &self.depth_of_field,
            uniforms,
            &mut self.focused,
        ))
    }
}

/// Renders a full-screen pass into `target` (made, or resized, to the scene's size) and returns
/// a sampler of the result.
fn full_screen<FS>(
    context: &mut RenderContext<'_>,
    scene: &Scene<'_>,
    pipeline: &SimdPipeline<ScreenVertexShader, FS>,
    uniforms: FS::Uniforms,
    target: &mut Option<(RenderTarget, Texture)>,
) -> Arc<TextureSampler>
where
    FS: FragmentShaderSimd<ScreenVaryings> + Send + Sync + 'static,
    FS::Uniforms: Clone,
{
    let (target, texture) = target.get_or_insert_with(|| {
        let target = RenderTarget::new(scene.extent);
        let texture = Texture::from_render_target(&target);
        (target, texture)
    });
    if target.extent() != scene.extent {
        target.resize(scene.extent);
    }
    {
        let mut pass = context.begin_render_pass(
            target,
            RenderPassDescriptor {
                viewport: Viewport::full(&target.extent()),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );
        scene
            .quad
            .draw_to_render_pass_simd(&mut pass, pipeline, (), move |_| uniforms.clone());
        pass.finish();
    }
    texture.update_from(target);
    Arc::new(texture.sampler(WrapMode::Clamp, FilterMode::Linear))
}

/// A pipeline for a full-screen pass over the finished scene.
fn post<FS>(shader: FS) -> SimdPipeline<ScreenVertexShader, FS>
where
    FS: FragmentShaderSimd<ScreenVaryings>,
{
    SimdPipeline::new(ScreenVertexShader, shader)
        .with_culling_mode(CullingMode::None)
        .with_depth_state(DepthState::DISABLED)
}

/// The distance along the view's centre line to the plane of the chip's face, where the macro
/// shots focus.
fn focus_distance(view: &View) -> f32 {
    let chip = &view.frame.chip;
    let centre = *chip * Vec4::new(0.0, 0.0, 0.0, 1.0);
    let up = *chip * Vec4::new(0.0, 1.0, 0.0, 0.0);
    let (centre, normal) = (
        Vec3::new(centre.x, centre.y, centre.z),
        Vec3::new(up.x, up.y, up.z),
    );
    let along = view.forward.dot(normal);
    if along.abs() < 1e-3 {
        return (centre - view.eye).length();
    }
    ((centre - view.eye).dot(normal) / along).max(0.05)
}
