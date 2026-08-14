use engine::prelude::*;

use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
    renderer::{CullingMode, Pipeline},
};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[derive(Clone)]
struct VertexUniforms {
    pub model_matrix: Mat4,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}

#[derive(Interpolate)]
struct Varyings {
    pub colour: Vec4,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();

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

struct TeapotApp {
    camera: Camera,
    teapot: Model<ObjVertex>,
    simple_pipeline: Pipeline<BasicVertexShader, BasicFragmentShader>,
}
impl TeapotApp {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let simple_pipeline = Pipeline::new(BasicVertexShader, BasicFragmentShader)
            .with_culling_mode(CullingMode::None);

        let camera = Camera::new(
            Vec3::new(0.0, 0.75, 1.25),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective(PerspectiveProjection::new(
                90.0,
                WIDTH as f32 / HEIGHT as f32,
                0.01,
                50.0,
            )),
        );

        let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
        teapot.transform.scale = Vec3::ONE * 0.3;

        Ok(Self {
            camera,
            teapot,
            simple_pipeline,
        })
    }
}

impl Application for TeapotApp {
    fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect_ratio(width as f32 / height as f32);
    }

    fn render<'frame>(
        &'frame mut self,
        frame: &mut cpu_rasteriser::renderer::Frame<'_, '_, 'frame>,
        _viewport: &Viewport,
    ) {
        let vertex_uniforms = VertexUniforms {
            model_matrix: self.teapot.transform.model_matrix(),
            view_matrix: self.camera.view_matrix(),
            projection_matrix: self.camera.projection_matrix(),
        };

        self.teapot
            .draw_to_frame(frame, &self.simple_pipeline, vertex_uniforms, |_| {
                FragmentUniforms
            });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    MinifbEngine::new()
        .with_title("Resize Demo - ESC to exit")
        .with_size(WIDTH, HEIGHT)
        .with_options(minifb::WindowOptions {
            resize: true,
            ..Default::default()
        })
        .run(TeapotApp::new()?)
}
