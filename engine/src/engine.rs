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

use crate::AppHandle;
use crate::app::{AppEvent, Application, CursorGrab};
use crate::input::{ButtonState, InputEvent, InputKey, InputState, MouseButton};

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
    app_handle: AppHandle,
    context: softbuffer::Context<winit::event_loop::OwnedDisplayHandle>,
    renderer: Renderer,
    viewport: Viewport,
    window_attributes: winit::window::WindowAttributes,
    state: AppState,
    last_frame: Instant,
    input_state: InputState,
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
            app_handle: AppHandle::default(),
            context,
            renderer: Renderer::new(&viewport)?,
            viewport,
            window_attributes,
            state: AppState::Initial,
            last_frame: Instant::now(),
            input_state: InputState::default(),
        })
    }

    fn handle_input_event(&mut self, event: InputEvent) {
        self.input_state.handle_event(event);
        self.app.event(AppEvent::Input(event), &mut self.app_handle);
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

        apply_winit_window_state(self.app.window_state(), window.as_ref());
        self.state = AppState::Running { surface };
        self.app.event(AppEvent::Resumed, &mut self.app_handle);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        let AppState::Running { surface } = &mut self.state else {
            return;
        };

        let window = surface.window().clone();
        self.state = AppState::Suspended { window };
        self.app.event(AppEvent::Suspended, &mut self.app_handle);
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
                    self.app.event(
                        AppEvent::Resized {
                            width: width.get(),
                            height: height.get(),
                        },
                        &mut self.app_handle,
                    );
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                self.app
                    .event(AppEvent::RedrawRequested, &mut self.app_handle);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
                self.app
                    .event(AppEvent::CloseRequested, &mut self.app_handle);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(key) = WinitInputMapper::map_winit_key_code(code) {
                        let input_event = InputEvent::Key {
                            key,
                            state: WinitInputMapper::map_winit_element_state(event.state),
                        };
                        self.handle_input_event(input_event);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(button) = WinitInputMapper::map_winit_mouse_button(button) {
                    let input_event = InputEvent::MouseButton {
                        button,
                        state: WinitInputMapper::map_winit_element_state(state),
                    };
                    self.handle_input_event(input_event);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_input_event(InputEvent::MouseMoved {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta_y = WinitInputMapper::map_winit_scroll_delta_y(delta);
                if delta_y != 0.0 {
                    self.handle_input_event(InputEvent::MouseWheel { delta_y });
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
                self.handle_input_event(InputEvent::MouseMotion {
                    dx: dx as f32,
                    dy: dy as f32,
                });
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.app_handle.should_exit() {
            event_loop.exit();
            return;
        }

        if let AppState::Running { surface } = &self.state {
            let window = surface.window().clone();
            apply_winit_window_state(self.app.window_state(), window.as_ref());
            window.request_redraw();
        }
    }
}

struct WinitInputMapper;
impl WinitInputMapper {
    fn map_winit_key_code(key_code: winit::keyboard::KeyCode) -> Option<InputKey> {
        match key_code {
            winit::keyboard::KeyCode::Escape => Some(InputKey::Escape),
            winit::keyboard::KeyCode::ArrowLeft => Some(InputKey::Left),
            winit::keyboard::KeyCode::ArrowRight => Some(InputKey::Right),
            winit::keyboard::KeyCode::ArrowUp => Some(InputKey::Up),
            winit::keyboard::KeyCode::ArrowDown => Some(InputKey::Down),
            winit::keyboard::KeyCode::KeyW => Some(InputKey::W),
            winit::keyboard::KeyCode::KeyA => Some(InputKey::A),
            winit::keyboard::KeyCode::KeyS => Some(InputKey::S),
            winit::keyboard::KeyCode::KeyD => Some(InputKey::D),
            winit::keyboard::KeyCode::Space => Some(InputKey::Space),
            winit::keyboard::KeyCode::ShiftLeft => Some(InputKey::LeftShift),
            _ => None,
        }
    }

    fn map_winit_mouse_button(button: winit::event::MouseButton) -> Option<MouseButton> {
        match button {
            winit::event::MouseButton::Left => Some(MouseButton::Left),
            winit::event::MouseButton::Right => Some(MouseButton::Right),
            winit::event::MouseButton::Middle => Some(MouseButton::Middle),
            _ => None,
        }
    }

    fn map_winit_element_state(state: winit::event::ElementState) -> ButtonState {
        match state {
            winit::event::ElementState::Pressed => ButtonState::Pressed,
            winit::event::ElementState::Released => ButtonState::Released,
        }
    }

    fn map_winit_scroll_delta_y(delta: winit::event::MouseScrollDelta) -> f32 {
        match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        }
    }
}

struct MinifbEngineApp<A>
where
    A: Application,
{
    app: A,
    app_handle: AppHandle,
    renderer: Renderer,
    viewport: Viewport,
    window: minifb::Window,
    last_frame: Instant,
    input_state: InputState,
    previous_keys: HashSet<InputKey>,
    previous_buttons: HashSet<MouseButton>,
}

fn apply_winit_window_state(state: crate::app::WindowState, window: &Window) {
    apply_winit_cursor_settings(state.cursor, window);
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
            app_handle: AppHandle::default(),
            renderer: Renderer::new(&viewport)?,
            viewport,
            window,
            last_frame: Instant::now(),
            input_state: InputState::default(),
            previous_keys: HashSet::new(),
            previous_buttons: HashSet::new(),
        })
    }

    fn handle_input_event(&mut self, event: InputEvent) {
        self.input_state.handle_event(event);
        self.app.event(AppEvent::Input(event), &mut self.app_handle);
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
            let was_down = self.previous_keys.contains(&key);

            match (was_down, is_down) {
                (false, true) => {
                    self.previous_keys.insert(key);
                    self.handle_input_event(InputEvent::Key {
                        key,
                        state: ButtonState::Pressed,
                    });
                }
                (true, false) => {
                    self.previous_keys.remove(&key);
                    self.handle_input_event(InputEvent::Key {
                        key,
                        state: ButtonState::Released,
                    });
                }
                _ => {}
            }
        }

        for (minifb_button, button) in TRACKED_BUTTONS {
            let is_down = self.window.get_mouse_down(minifb_button);
            let was_down = self.previous_buttons.contains(&button);

            match (was_down, is_down) {
                (false, true) => {
                    self.previous_buttons.insert(button);
                    self.handle_input_event(InputEvent::MouseButton {
                        button,
                        state: ButtonState::Pressed,
                    });
                }
                (true, false) => {
                    self.previous_buttons.remove(&button);
                    self.handle_input_event(InputEvent::MouseButton {
                        button,
                        state: ButtonState::Released,
                    });
                }
                _ => {}
            }
        }

        if let Some((x, y)) = self.window.get_mouse_pos(minifb::MouseMode::Pass) {
            self.handle_input_event(InputEvent::MouseMoved { x, y });
        }

        if let Some((_, y)) = self.window.get_scroll_wheel() {
            if y != 0.0 {
                self.handle_input_event(InputEvent::MouseWheel { delta_y: y });
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
        self.app.event(
            AppEvent::Resized {
                width: width as u32,
                height: height as u32,
            },
            &mut self.app_handle,
        );
    }

    fn apply_window_state(&mut self) {
        let state = self.app.window_state();
        self.window.set_cursor_visibility(state.cursor.visible);
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
        self.app.event(AppEvent::Resumed, &mut self.app_handle);

        while self.window.is_open() {
            self.emit_input_events();

            if self.app_handle.should_exit() {
                break;
            }

            self.apply_resize();
            self.apply_window_state();
            self.render_frame()?;
        }

        self.app
            .event(AppEvent::CloseRequested, &mut self.app_handle);
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
