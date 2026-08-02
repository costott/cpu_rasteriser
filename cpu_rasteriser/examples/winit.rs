use cpu_rasteriser::prelude::*;

use cpu_rasteriser::renderer::DrawCall;
use cpu_rasteriser::{
    graphics::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
    renderer::{CullingMode, Pipeline, Renderer},
};

use std::num::NonZeroU32;
use std::rc::Rc;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[derive(Clone)]
struct Vertex {
    pub position: Vec3,
    pub colour: Vec3,
}

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
    type Vertex = Vertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();
        let view_position = uniforms.view_matrix * world_position;
        let clip_position = uniforms.projection_matrix * view_position;

        let varyings = Varyings {
            colour: vertex.colour,
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

    let mut app = App {
        context,
        state: AppState::Initial,
        viewport: Viewport::new(WIDTH, HEIGHT),
        renderer: Renderer::new(&Viewport::new(WIDTH, HEIGHT))?,
        timer: Timer::new(),
        triangle: vec![
            Vertex {
                position: Vec3::new(-1.0, -1.0, 0.0),
                colour: Colour::from_u32(0xff0000).into(),
            },
            Vertex {
                position: Vec3::new(1.0, -1.0, 0.0),
                colour: Colour::from_u32(0x00ff00).into(),
            },
            Vertex {
                position: Vec3::new(0.0, 1.0, 0.0),
                colour: Colour::from_u32(0x0000ff).into(),
            },
        ],
        triangle_indices: vec![0, 1, 2],
    };
    event_loop.run_app(&mut app)?;

    Ok(())
}

struct Timer {
    start_time: std::time::Instant,
}
impl Timer {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

struct App {
    context: softbuffer::Context<winit::event_loop::OwnedDisplayHandle>,
    state: AppState,
    viewport: Viewport,
    renderer: Renderer,
    timer: Timer,
    triangle: Vec<Vertex>,
    triangle_indices: Vec<u32>,
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
                let simple_pipeline = Pipeline::new(BasicVertexShader, BasicFragmentShader)
                    .with_culling_mode(CullingMode::None);

                let mut frame = self.renderer.begin_frame(&self.viewport);

                let vertex_uniforms = VertexUniforms {
                    model_matrix: Mat4::rotate_y(self.timer.elapsed().as_secs_f32()),
                    view_matrix: Mat4::look_at(
                        Vec3::new(0.0, 0.0, 1.25),
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(0.0, 1.0, 0.0),
                    ),
                    projection_matrix: Mat4::perspective(
                        90.0,
                        WIDTH as f32 / HEIGHT as f32,
                        0.01,
                        50.0,
                    ),
                };

                frame.draw(
                    &simple_pipeline,
                    DrawCall::new(
                        &self.triangle,
                        &self.triangle_indices,
                        cpu_rasteriser::renderer::PrimitiveMode::TRIANGLES,
                        FragmentUniforms,
                    ),
                    &vertex_uniforms,
                );

                frame.finish();

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
