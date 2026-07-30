use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{
        camera::{Camera, Projection},
        fragment_shader::FragmentShader,
        vertex_shader::VertexShader,
    },
    loaders::obj::load_obj,
    renderer::Renderer,
};

use std::num::NonZeroU32;
use std::rc::Rc;

mod common;
use common::timer::Timer;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

struct VertexUniforms {
    pub model_matrix: Mat4,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}

#[derive(Interpolate)]
struct Varyings {
    pub colour: Vec3,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_homogenous();

        let view_position = uniforms.view_matrix * world_position;
        let clip_position = uniforms.projection_matrix * view_position;

        let varyings = Varyings {
            colour: vertex.colour.into(),
        };

        (clip_position, varyings)
    }
}

struct FragmentUniforms;

struct BasicFragmentShader;
impl FragmentShader<Varyings> for BasicFragmentShader {
    type Uniforms = FragmentUniforms;

    fn shade(&self, varyings: Varyings, _uniforms: &Self::Uniforms) -> Colour {
        varyings.colour.into()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise winit event loop and window
    let event_loop = winit::event_loop::EventLoop::new()?;
    let context = softbuffer::Context::new(event_loop.owned_display_handle())?;

    let camera = Camera::new(
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

    let mut app = App {
        context,
        state: AppState::Initial,
        viewport: Viewport::new(WIDTH, HEIGHT),
        renderer: Renderer::new(
            &Viewport::new(WIDTH, HEIGHT),
            BasicVertexShader,
            BasicFragmentShader,
        )?,
        teapot,
        camera,
        timer: Timer::new(),
    };
    event_loop.run_app(&mut app)?;

    Ok(())
}

struct App {
    context: softbuffer::Context<winit::event_loop::OwnedDisplayHandle>,
    state: AppState,
    viewport: Viewport,
    renderer: Renderer<BasicVertexShader, BasicFragmentShader>,
    teapot: Model<ObjVertex>,
    camera: Camera,
    timer: Timer,
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
                self.renderer.begin_frame();

                let dt = self.timer.delta();
                self.teapot.transform.rotation.y += 0.5 * dt.as_secs_f32();

                let teapot_vertex_uniforms = VertexUniforms {
                    model_matrix: self.teapot.transform.model_matrix(),
                    view_matrix: self.camera.view_matrix(),
                    projection_matrix: self.camera.projection_matrix(),
                };

                for draw_call in self.teapot.draw_calls(|_| FragmentUniforms) {
                    self.renderer.submit_draw_call(
                        draw_call,
                        &teapot_vertex_uniforms,
                        &self.viewport,
                    );
                }

                self.renderer.submit_frame();

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

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppState::Running { surface } = &mut self.state {
            surface.window().request_redraw();
        }
    }
}
