use engine::prelude::*;

use cpu_rasteriser::prelude::*;

use std::sync::Arc;

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
    pub world_position: Vec3,
    pub colour: Vec4,
    pub normal: Vec3,
    pub uv: Vec2,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();
        let normal_matrix = uniforms.model_matrix.inverse().transpose();

        let view_position = uniforms.view_matrix * world_position;
        let clip_position = uniforms.projection_matrix * view_position;

        let varyings = Varyings {
            world_position: world_position.homogenize_to_vec3(),
            colour: vertex.colour.into(),
            normal: (normal_matrix * vertex.normal.to_direction4())
                .xyz()
                .normalise(),
            uv: vertex.uv,
        };

        (clip_position, varyings)
    }
}

// Uniforms for the scene (don't vary per model/mesh)
struct SceneUniforms {
    camera: Camera,
    lights: Vec<DirectionalLight>,
    ambient_light: Colour,
}

struct FragmentUniforms {
    scene: Arc<SceneUniforms>,
    material: Option<Material>,
}

struct PhongFragmentShader;
impl FragmentShader<Varyings> for PhongFragmentShader {
    type Uniforms = FragmentUniforms;

    fn shade(&self, varyings: Varyings, uniforms: &Self::Uniforms) -> Colour {
        let normal = varyings.normal.normalise();

        let Some(material) = &uniforms.material else {
            return Colour::BLACK;
        };

        let view_dir = (uniforms.scene.camera.eye - varyings.world_position).normalise();

        // Base surface colour
        let albedo = match &material.diffuse_texture {
            Some(texture) => texture.sample(varyings.uv),
            None => material.diffuse,
        };

        // Ambient lighting
        let mut colour = albedo * uniforms.scene.ambient_light;

        for light in &uniforms.scene.lights {
            let light_dir = (-light.direction).normalise();

            // Diffuse
            let diffuse_strength = normal.dot(&light_dir).max(0.0);

            let diffuse = albedo * light.colour * diffuse_strength;

            // Specular
            let reflect_dir = reflect(-light_dir, normal);

            let specular_strength = view_dir.dot(&reflect_dir).max(0.0).powf(material.shininess);

            let specular = match &material.specular_texture {
                Some(texture) => texture.sample(varyings.uv),
                None => material.specular,
            } * light.colour
                * specular_strength;

            colour = colour + diffuse + specular;
        }

        colour
    }
}

/// Reflects a vector around a normal, using the formula: R = V - 2 * (V . N) * N
fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * 2.0 * vector.dot(&normal)
}

struct TexturedCubeApp {
    camera: Camera,
    camera_controls: OrbitControls,
    cube: Model<ObjVertex>,
    phong_pipeline: Pipeline<BasicVertexShader, PhongFragmentShader>,
}
impl TexturedCubeApp {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let phong_pipeline = Pipeline::new(BasicVertexShader, PhongFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DEFAULT);

        let camera = Camera::new(
            Vec3::new(1.25, 0.75, 0.0),
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

        let mut cube = load_obj(std::path::Path::new("assets/dice/cube-tex.obj"))?;
        cube.transform.position = Vec3::new(-0.5, -0.5, -0.5);
        cube.calculate_vertex_normals();

        Ok(Self {
            camera,
            camera_controls,
            cube,
            phong_pipeline,
        })
    }
}

impl Application for TexturedCubeApp {
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

        let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: Some(LoadOp::Clear(1.0)),
        });

        let vertex_uniforms = VertexUniforms {
            model_matrix: self.cube.transform.model_matrix(),
            view_matrix: self.camera.view_matrix(),
            projection_matrix: self.camera.projection_matrix(),
        };

        let scene_uniforms = Arc::new(SceneUniforms {
            camera: self.camera.clone(),
            lights: vec![DirectionalLight::new(
                Vec3::new(-0.5, -1.0, 2.0).normalise(),
                Colour::from_u32(0xfffde8),
            )],
            ambient_light: Colour::from_u32(0x202020),
        });

        self.cube
            .draw_to_render_pass(&mut pass, &self.phong_pipeline, vertex_uniforms, |mesh| {
                FragmentUniforms {
                    scene: scene_uniforms.clone(),
                    material: self
                        .cube
                        .materials
                        .get(mesh.material_index.unwrap())
                        .cloned(),
                }
            });

        pass.finish()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    WinitEngine::new()
        .with_window_attributes(
            winit::window::Window::default_attributes()
                .with_title("Textured Cube Demo - ESC to exit")
                .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
                    WIDTH as u32,
                    HEIGHT as u32,
                ))),
        )
        .run(TexturedCubeApp::new()?)
}
