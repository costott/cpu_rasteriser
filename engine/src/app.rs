use cpu_rasteriser::{renderer::Frame, viewport::Viewport};

use crate::input::{InputKey, MouseButton, MouseState};

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

    fn render<'frame>(&'frame mut self, _frame: &mut Frame<'_, '_, 'frame>, _viewport: &Viewport) {}

    fn resize(&mut self, _width: u32, _height: u32) {}

    fn event(&mut self, _event: AppEvent) {}

    fn window_state(&self) -> WindowState {
        WindowState::default()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AppEvent {
    CloseRequested,
    Resized {
        width: u32,
        height: u32,
    },
    RedrawRequested,
    Suspended,
    Resumed,
    Key {
        key: InputKey,
        state: MouseState,
    },
    MouseButton {
        button: MouseButton,
        state: MouseState,
    },
    MouseMoved {
        x: f32,
        y: f32,
    },
    MouseMotionDelta {
        dx: f32,
        dy: f32,
    },
    MouseWheel {
        delta_y: f32,
    },
}
