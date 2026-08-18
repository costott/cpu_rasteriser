use cpu_rasteriser::prelude::*;

use crate::input::InputEvent;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorGrab {
    None,
    Confined,
    Locked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowState {
    pub cursor: WindowCursorSettings,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            cursor: WindowCursorSettings::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCursorSettings {
    pub visible: bool,
    pub grab: CursorGrab,
}

impl Default for WindowCursorSettings {
    fn default() -> Self {
        Self {
            visible: true,
            grab: CursorGrab::None,
        }
    }
}

pub trait Application {
    fn update(&mut self, _dt: f32) {}

    fn render<'frame>(&mut self, _context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        PresentedFrame { _private: () }
    }

    fn resize(&mut self, _width: u32, _height: u32) {}

    fn event(&mut self, _event: AppEvent, _handle: &mut AppHandle) {}

    fn window_state(&self) -> WindowState {
        WindowState::default()
    }
}

pub struct RenderContext<'a> {
    renderer: &'a mut Renderer,
    presentation_target: &'a mut RenderTarget,
}
impl<'a> RenderContext<'a> {
    pub fn new(renderer: &'a mut Renderer, presentation_target: &'a mut RenderTarget) -> Self {
        Self {
            renderer,
            presentation_target,
        }
    }

    pub fn presentation_target(&self) -> &RenderTarget {
        self.presentation_target
    }

    /// Begin a render pass that will render directly to the presentation target (the screen).
    pub fn begin_presentation_pass<'pass>(
        &'pass mut self,
        descriptor: RenderPassDescriptor,
    ) -> PresentationPass<'pass, 'pass> {
        PresentationPass::new(
            self.renderer
                .begin_render_pass(self.presentation_target, descriptor),
        )
    }

    /// Begin a render pass that will render to the given render target.
    pub fn begin_render_pass<'pass>(
        &'pass mut self,
        target: &'pass mut RenderTarget,
        descriptor: RenderPassDescriptor,
    ) -> RenderPass<'pass, 'pass> {
        self.renderer.begin_render_pass(target, descriptor)
    }
}

/// A wrapper around the final [`RenderPass`] that will draw to the
/// engine's presentation target.
pub struct PresentationPass<'a, 'b> {
    render_pass: RenderPass<'a, 'b>,
}
impl<'a, 'b> PresentationPass<'a, 'b> {
    pub fn new(render_pass: RenderPass<'a, 'b>) -> Self {
        Self { render_pass }
    }

    pub fn render_pass_mut(&mut self) -> &mut RenderPass<'a, 'b> {
        &mut self.render_pass
    }

    pub fn finish(self) -> PresentedFrame {
        self.render_pass.finish();
        PresentedFrame { _private: () }
    }
}
impl<'a, 'b> std::ops::Deref for PresentationPass<'a, 'b> {
    type Target = RenderPass<'a, 'b>;

    fn deref(&self) -> &Self::Target {
        &self.render_pass
    }
}
impl<'a, 'b> std::ops::DerefMut for PresentationPass<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.render_pass
    }
}
/// Compiler proof token that the [`PresentationPass`] presents a finished pass to the engine's
/// presentation target.
pub struct PresentedFrame {
    _private: (),
}

#[derive(Clone, Copy, Debug)]
pub enum AppEvent {
    CloseRequested,
    Resized { width: u32, height: u32 },
    RedrawRequested,
    Suspended,
    Resumed,
    Input(InputEvent),
}

pub struct AppHandle {
    exit_requested: bool,
}
impl AppHandle {
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    pub fn should_exit(&self) -> bool {
        self.exit_requested
    }
}
impl Default for AppHandle {
    fn default() -> Self {
        Self {
            exit_requested: false,
        }
    }
}
