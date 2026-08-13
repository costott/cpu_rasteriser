use std::time::Duration;

use cpu_rasteriser::{renderer::Frame, viewport::Viewport};

pub trait Application {
    fn update(&mut self, _dt: f32) {}

    fn render<'a>(&'a mut self, _frame: &mut Frame<'a, 'a>, _viewport: &'a Viewport) {}

    fn resize(&mut self, _width: u32, _height: u32) {}

    fn event(&mut self, _event: AppEvent) {}
}

#[derive(Clone, Copy, Debug)]
pub enum AppEvent {
    CloseRequested,
    Resized { width: u32, height: u32 },
    RedrawRequested,
    Suspended,
    Resumed,
}
