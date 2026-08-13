use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use cpu_rasteriser::{renderer::Renderer, viewport::Viewport};
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

use crate::app::{AppEvent, Application};

pub trait EngineBackend {
    fn run<A>(app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        A: Application + 'static;
}

pub struct Engine;

impl Engine {
    pub fn run<A>(app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        A: Application + 'static,
    {
        WinitEngine::run(app)
    }

    pub fn with_backend<B, A>(app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        B: EngineBackend,
        A: Application + 'static,
    {
        B::run(app)
    }
}

pub struct WinitEngine;

impl EngineBackend for WinitEngine {
    fn run<A>(app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        A: Application + 'static,
    {
        let event_loop = EventLoop::new()?;
        let context = softbuffer::Context::new(event_loop.owned_display_handle())?;

        let mut app_runner = WinitEngineApp::new(context, app)?;
        event_loop.run_app(&mut app_runner)?;
        Ok(())
    }
}

pub struct MinifbEngine;

impl EngineBackend for MinifbEngine {
    fn run<A>(app: A) -> Result<(), Box<dyn std::error::Error>>
    where
        A: Application + 'static,
    {
        let mut window = minifb::Window::new("Engine", 640, 360, minifb::WindowOptions::default())?;
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
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let viewport = Viewport::new(640, 360);
        Ok(Self {
            app,
            context,
            renderer: Renderer::new(&viewport)?,
            viewport,
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

        let start = Instant::now();

        self.app.update(dt);
        let update = start.elapsed();

        let mut frame = self.renderer.begin_frame(&self.viewport);
        self.app.render(&mut frame, &self.viewport);
        frame.finish();
        let render = start.elapsed();

        let mut buffer = surface.buffer_mut().unwrap();
        let buffer_mut = start.elapsed();

        buffer.copy_from_slice(self.renderer.pixels());
        let copy = start.elapsed();

        buffer.present().unwrap();
        let present = start.elapsed();

        println!(
            "update={:?} render={:?} buffer={:?} copy={:?} present={:?}",
            update,
            render - update,
            buffer_mut - render,
            copy - buffer_mut,
            present - copy,
        );
    }
}

impl<A> ApplicationHandler for WinitEngineApp<A>
where
    A: Application,
{
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if let StartCause::Init = cause {
            let attrs = Window::default_attributes()
                .with_title("Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(640.0, 360.0));
            let window = event_loop
                .create_window(attrs)
                .expect("failed creating window");
            self.state = AppState::Suspended {
                window: Rc::new(window),
            };
        }
    }

    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        let AppState::Suspended { window } = &mut self.state else {
            return;
        };

        let size = window.inner_size();
        let surface = softbuffer::Surface::new(&self.context, window.clone())
            .expect("failed creating surface");

        if let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        {
            self.apply_resize(width.get(), height.get());
        }

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
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let AppState::Running { surface } = &mut self.state {
            surface.window().request_redraw();
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
        })
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
            self.apply_resize();
            self.render_frame()?;
        }

        self.app.event(AppEvent::CloseRequested);
        Ok(())
    }
}
