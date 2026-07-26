use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{
        camera::{Camera, Projection},
        fragment_shader::BasicFragmentShader,
        lighting::DirectionalLight,
        vertex_shader::GouraudVertexShader,
    },
    loaders::obj::load_obj,
    renderer::{CullingMode, Renderer},
};

use std::num::NonZeroU32;
use std::rc::Rc;

mod common;
use common::camera_controller::FirstPersonControls;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise winit event loop and window
    let event_loop = winit::event_loop::EventLoop::new()?;
    let context = softbuffer::Context::new(event_loop.owned_display_handle())?;

    let mut camera = Camera::new(
        Vec3::new(0.0, 0.75, 1.25),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Projection::Perspective(
            cpu_rasteriser::graphics::camera::PerspectiveProjection::new(
                90.0,
                WIDTH as f32 / HEIGHT as f32,
                0.01,
                50.0,
            ),
        ),
    );

    let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
    teapot.transform.scale = Vec3::ONE * 0.3;
    teapot.transform.rotation.y = 0_f32.to_radians();

    let mut scene = Scene::new(camera);
    scene.add_light(DirectionalLight::new(
        Vec3::new(0.0, -1.0, -1.0),
        Colour::from_u32(0xfffde8),
    ));
    scene.add_model(teapot);

    let mut app = App {
        context,
        state: AppState::Initial,
        viewport: Viewport::new(WIDTH, HEIGHT),
        renderer: Renderer::new(
            &Viewport::new(WIDTH, HEIGHT),
            Box::new(GouraudVertexShader),
            std::sync::Arc::new(BasicFragmentShader),
        )?,
        scene,
    };
    event_loop.run_app(&mut app)?;

    Ok(())
}

struct App {
    context: softbuffer::Context<winit::event_loop::OwnedDisplayHandle>,
    state: AppState,
    viewport: Viewport,
    renderer: Renderer,
    scene: Scene,
}

#[derive(Debug)]
enum AppState {
    Initial,
    Suspended {
        window: Rc<winit::window::Window>,
    },
    Running {
        surface:
            softbuffer::Surface<winit::event_loop::OwnedDisplayHandle, Rc<winit::window::Window>>,
    },
}

impl winit::application::ApplicationHandler for App {
    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        if let winit::event::StartCause::Init = cause {
            // Create window on startup
            let window_attrs = winit::window::Window::default_attributes()
                .with_inner_size(winit::dpi::LogicalSize::new(WIDTH as f64, HEIGHT as f64));
            let window = event_loop
                .create_window(window_attrs)
                .expect("failed creating window");
            self.state = AppState::Suspended {
                window: Rc::new(window),
            };
        }
    }

    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Create or re-create the surface.
        let AppState::Suspended { window } = &mut self.state else {
            unreachable!("got resumed event while not suspended");
        };
        let mut surface = softbuffer::Surface::new(&self.context, window.clone())
            .expect("failed creating surface");

        let size = window.inner_size();
        if let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        {
            // Resize surface
            surface.resize(width, height).unwrap();
            self.viewport = Viewport::new(width.get() as usize, height.get() as usize);
            self.renderer.resize(&self.viewport);
        }

        self.state = AppState::Running { surface };
    }

    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Drop the surface.
        let AppState::Running { surface } = &mut self.state else {
            unreachable!("got resumed event while not running");
        };
        let window = surface.window().clone();
        self.state = AppState::Suspended { window };
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let AppState::Running { surface } = &mut self.state else {
            unreachable!("got window event while suspended");
        };

        if surface.window().id() != window_id {
            return;
        }

        match event {
            winit::event::WindowEvent::Resized(size) => {
                if let (Some(width), Some(height)) =
                    (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                {
                    // Resize surface
                    surface.resize(width, height).unwrap();
                    self.viewport = Viewport::new(width.get() as usize, height.get() as usize);
                    self.renderer.resize(&self.viewport);
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.renderer.clear(Colour::BLACK);

                self.renderer.draw_scene(&self.scene, &self.viewport);

                // Get the next buffer.
                let mut buffer = surface.buffer_mut().unwrap();

                // Render into the buffer.
                buffer.copy_from_slice(self.renderer.pixels());

                // Send the buffer to the compositor.
                buffer.present().unwrap();
            }
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
    }
}
