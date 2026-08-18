use engine::prelude::*;

use cpu_rasteriser::prelude::*;

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
    camera_controls: OrbitControls,
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
        let camera_controls = OrbitControls::new(&camera);

        let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
        teapot.transform.scale = Vec3::ONE * 0.3;
        teapot.transform.rotation.y = 90_f32.to_radians();

        Ok(Self {
            camera,
            camera_controls,
            teapot,
            simple_pipeline,
        })
    }
}

impl Application for TeapotApp {
    fn update(&mut self, dt: f32) {
        self.camera_controls
            .update_from_events(&mut self.camera, dt);
    }

    fn event(&mut self, event: AppEvent, handle: &mut AppHandle) {
        self.camera_controls.handle_event(event);

        // Exit the application if the Escape key is pressed
        if let AppEvent::Input(InputEvent::Key {
            key: InputKey::Escape,
            state: ButtonState::Released,
        }) = event
        {
            handle.request_exit();
        }
    }

    fn render<'frame>(&mut self, context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        let extent = context.presentation_target().extent();

        let mut presentation_pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: Some(LoadOp::Clear(1.0)),
        });

        let vertex_uniforms = VertexUniforms {
            model_matrix: self.teapot.transform.model_matrix(),
            view_matrix: self.camera.view_matrix(),
            projection_matrix: self.camera.projection_matrix(),
        };

        self.teapot.draw_to_render_pass(
            &mut presentation_pass,
            &self.simple_pipeline,
            vertex_uniforms,
            |_| FragmentUniforms,
        );

        presentation_pass.finish()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    MinifbEngine::new()
        .with_title("Teapot Demo - ESC to exit")
        .with_size(WIDTH, HEIGHT)
        .run(TeapotApp::new()?)
}
