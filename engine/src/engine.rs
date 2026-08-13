use std::collections::HashSet;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use cpu_rasteriser::{renderer::Renderer, viewport::Viewport};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::Window;

use crate::app::{AppEvent, Application, CursorGrab};
use crate::input::{
    InputKey, MouseButton, MouseState, map_winit_element_state, map_winit_key_code,
    map_winit_mouse_button, map_winit_scroll_delta_y,
};

pub trait EngineBackend: Sized {
    fn run<A>(self, app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        A: Application + 'static;
}

pub struct WinitEngine {
    window_attributes: winit::window::WindowAttributes,
}

impl WinitEngine {
    pub fn new() -> Self {
        Self {
            window_attributes: Window::default_attributes()
                .with_title("Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(640.0, 360.0)),
        }
    }

    pub fn with_window_attributes(mut self, attrs: winit::window::WindowAttributes) -> Self {
        self.window_attributes = attrs;
        self
    }
}

impl Default for WinitEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBackend for WinitEngine {
    fn run<A>(self, app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        A: Application + 'static,
    {
        let event_loop = EventLoop::new()?;
        let context = softbuffer::Context::new(event_loop.owned_display_handle())?;

        let mut app_runner = WinitEngineApp::new(context, app, self.window_attributes)?;
        event_loop.run_app(&mut app_runner)?;
        Ok(())
    }
}

pub struct MinifbEngine {
    title: String,
    width: usize,
    height: usize,
    options: minifb::WindowOptions,
}

impl MinifbEngine {
    pub fn new() -> Self {
        Self {
            title: "Engine".into(),
            width: 640,
            height: 360,
            options: minifb::WindowOptions::default(),
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_size(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_options(mut self, options: minifb::WindowOptions) -> Self {
        self.options = options;
        self
    }
}

impl Default for MinifbEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBackend for MinifbEngine {
    fn run<A>(self, app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        A: Application + 'static,
    {
        let mut window = minifb::Window::new(&self.title, self.width, self.height, self.options)?;
        window.set_target_fps(60);

        let mut app_runner = MinifbEngineApp::new(window, app)?;
        app_runner.run()
    }
}

struct WinitEngineApp<A>
where
    A: Application,
{
    app: A,
    context: softbuffer::Context<winit::event_loop::OwnedDisplayHandle>,
    renderer: Renderer,
    viewport: Viewport,
    window_attributes: winit::window::WindowAttributes,
    state: AppState,
    last_frame: Instant,
}

enum AppState {
    Initial,
    Suspended {
        window: Rc<Window>,
    },
    Running {
        surface: softbuffer::Surface<winit::event_loop::OwnedDisplayHandle, Rc<Window>>,
    },
}

impl<A> WinitEngineApp<A>
where
    A: Application,
{
    fn new(
        context: softbuffer::Context<winit::event_loop::OwnedDisplayHandle>,
        app: A,
        window_attributes: winit::window::WindowAttributes,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let viewport = Viewport::new(640, 360);
        Ok(Self {
            app,
            context,
            renderer: Renderer::new(&viewport)?,
            viewport,
            window_attributes,
            state: AppState::Initial,
            last_frame: Instant::now(),
        })
    }

    fn apply_resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        let viewport = Viewport::new(width as usize, height as usize);
        self.viewport = viewport;
        self.renderer.resize(&self.viewport);
        self.app.resize(width, height);
    }

    fn render_frame(&mut self) {
        let AppState::Running { surface } = &mut self.state else {
            return;
        };

        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();

        self.app.update(dt);

        let mut frame = self.renderer.begin_frame(&self.viewport);
        self.app.render(&mut frame, &self.viewport);
        frame.finish();

        let mut buffer = surface.buffer_mut().unwrap();

        buffer.copy_from_slice(self.renderer.pixels());
        buffer.present().unwrap();
    }
}

impl<A> ApplicationHandler for WinitEngineApp<A>
where
    A: Application,
{
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if let StartCause::Init = cause {
            let window = event_loop
                .create_window(self.window_attributes.clone())
                .expect("failed creating window");
            self.state = AppState::Suspended {
                window: Rc::new(window),
            };
        }
    }

    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        let window = match &self.state {
            AppState::Suspended { window } => window.clone(),
            _ => return,
        };

        let size = window.inner_size();
        let surface = softbuffer::Surface::new(&self.context, window.clone())
            .expect("failed creating surface");

        if let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        {
            self.apply_resize(width.get(), height.get());
        }

        apply_winit_cursor_settings(self.app.window_cursor_settings(), window.as_ref());
        self.state = AppState::Running { surface };
        self.app.event(AppEvent::Resumed);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        let AppState::Running { surface } = &mut self.state else {
            return;
        };

        let window = surface.window().clone();
        self.state = AppState::Suspended { window };
        self.app.event(AppEvent::Suspended);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = (match &self.state {
            AppState::Running { surface } => Some(surface.window().clone()),
            AppState::Suspended { window } => Some(window.clone()),
            AppState::Initial => None,
        }) else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::Resized(size) => {
                if let (Some(width), Some(height)) =
                    (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                {
                    if let AppState::Running { surface } = &mut self.state {
                        surface.resize(width, height).unwrap();
                    }
                    self.apply_resize(width.get(), height.get());
                    self.app.event(AppEvent::Resized {
                        width: width.get(),
                        height: height.get(),
                    });
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                self.app.event(AppEvent::RedrawRequested);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
                self.app.event(AppEvent::CloseRequested);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(key) = map_winit_key_code(code) {
                        self.app.event(AppEvent::Key {
                            key,
                            state: map_winit_element_state(event.state),
                        });
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(button) = map_winit_mouse_button(button) {
                    self.app.event(AppEvent::MouseButton {
                        button,
                        state: map_winit_element_state(state),
                    });
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.app.event(AppEvent::MouseMoved {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta_y = map_winit_scroll_delta_y(delta);
                if delta_y != 0.0 {
                    self.app.event(AppEvent::MouseWheel { delta_y });
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let (dx, dy) = delta;
            if dx != 0.0 || dy != 0.0 {
                self.app.event(AppEvent::MouseMotionDelta {
                    dx: dx as f32,
                    dy: dy as f32,
                });
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let AppState::Running { surface } = &self.state {
            let window = surface.window().clone();
            apply_winit_cursor_settings(self.app.window_cursor_settings(), window.as_ref());
            window.request_redraw();
        }
    }
}

struct MinifbEngineApp<A>
where
    A: Application,
{
    app: A,
    renderer: Renderer,
    viewport: Viewport,
    window: minifb::Window,
    last_frame: Instant,
    keys_down: HashSet<InputKey>,
    buttons_down: HashSet<MouseButton>,
}

fn apply_winit_cursor_settings(settings: crate::app::WindowCursorSettings, window: &Window) {
    window.set_cursor_visible(settings.visible);

    let _ = match settings.grab {
        CursorGrab::None => window.set_cursor_grab(winit::window::CursorGrabMode::None),
        CursorGrab::Confined => window.set_cursor_grab(winit::window::CursorGrabMode::Confined),
        CursorGrab::Locked => window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined)),
    };
}

impl<A> MinifbEngineApp<A>
where
    A: Application,
{
    fn new(window: minifb::Window, app: A) -> Result<Self, Box<dyn std::error::Error>> {
        let (width, height) = window.get_size();
        let viewport = Viewport::new(width, height);
        Ok(Self {
            app,
            renderer: Renderer::new(&viewport)?,
            viewport,
            window,
            last_frame: Instant::now(),
            keys_down: HashSet::new(),
            buttons_down: HashSet::new(),
        })
    }

    fn emit_input_events(&mut self) {
        const TRACKED_KEYS: [(minifb::Key, InputKey); 11] = [
            (minifb::Key::Escape, InputKey::Escape),
            (minifb::Key::Left, InputKey::Left),
            (minifb::Key::Right, InputKey::Right),
            (minifb::Key::Up, InputKey::Up),
            (minifb::Key::Down, InputKey::Down),
            (minifb::Key::W, InputKey::W),
            (minifb::Key::A, InputKey::A),
            (minifb::Key::S, InputKey::S),
            (minifb::Key::D, InputKey::D),
            (minifb::Key::Space, InputKey::Space),
            (minifb::Key::LeftShift, InputKey::LeftShift),
        ];

        const TRACKED_BUTTONS: [(minifb::MouseButton, MouseButton); 3] = [
            (minifb::MouseButton::Left, MouseButton::Left),
            (minifb::MouseButton::Right, MouseButton::Right),
            (minifb::MouseButton::Middle, MouseButton::Middle),
        ];

        for (minifb_key, key) in TRACKED_KEYS {
            let is_down = self.window.is_key_down(minifb_key);
            let was_down = self.keys_down.contains(&key);

            match (was_down, is_down) {
                (false, true) => {
                    self.keys_down.insert(key);
                    self.app.event(AppEvent::Key {
                        key,
                        state: MouseState::Pressed,
                    });
                }
                (true, false) => {
                    self.keys_down.remove(&key);
                    self.app.event(AppEvent::Key {
                        key,
                        state: MouseState::Released,
                    });
                }
                _ => {}
            }
        }

        for (minifb_button, button) in TRACKED_BUTTONS {
            let is_down = self.window.get_mouse_down(minifb_button);
            let was_down = self.buttons_down.contains(&button);

            match (was_down, is_down) {
                (false, true) => {
                    self.buttons_down.insert(button);
                    self.app.event(AppEvent::MouseButton {
                        button,
                        state: MouseState::Pressed,
                    });
                }
                (true, false) => {
                    self.buttons_down.remove(&button);
                    self.app.event(AppEvent::MouseButton {
                        button,
                        state: MouseState::Released,
                    });
                }
                _ => {}
            }
        }

        if let Some((x, y)) = self.window.get_mouse_pos(minifb::MouseMode::Pass) {
            self.app.event(AppEvent::MouseMoved { x, y });
        }

        if let Some((_, y)) = self.window.get_scroll_wheel() {
            if y != 0.0 {
                self.app.event(AppEvent::MouseWheel { delta_y: y });
            }
        }
    }

    fn apply_resize(&mut self) {
        let (width, height) = self.window.get_size();
        if width == 0 || height == 0 {
            return;
        }

        let viewport = Viewport::new(width, height);
        self.viewport = viewport;
        self.renderer.resize(&self.viewport);
        self.app.resize(width as u32, height as u32);
        self.app.event(AppEvent::Resized {
            width: width as u32,
            height: height as u32,
        });
    }

    fn apply_cursor_settings(&mut self) {
        let settings = self.app.window_cursor_settings();
        self.window.set_cursor_visibility(settings.visible);
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();
        self.app.update(dt);

        let mut frame = self.renderer.begin_frame(&self.viewport);
        self.app.render(&mut frame, &self.viewport);
        frame.finish();

        self.window
            .update_with_buffer(
                self.renderer.pixels(),
                self.viewport.width,
                self.viewport.height,
            )
            .map_err(|err| err.into())
    }

    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.app.event(AppEvent::Resumed);

        while self.window.is_open() && !self.window.is_key_down(minifb::Key::Escape) {
            self.emit_input_events();
            self.apply_resize();
            self.apply_cursor_settings();
            self.render_frame()?;
        }

        self.app.event(AppEvent::CloseRequested);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winit_engine_accepts_custom_window_attributes() {
        let attrs = Window::default_attributes()
            .with_title("Custom engine window")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0));

        let _engine = WinitEngine::new().with_window_attributes(attrs);
    }
}
