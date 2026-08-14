use engine::prelude::*;

use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
    renderer::{CullingMode, Pipeline},
};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

/// Reflects a vector around a normal, using the formula: R = V - 2 * (V . N) * N
fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * 2.0 * vector.dot(&normal)
}

#[derive(Clone)]
struct GouraudVertexUniforms {
    pub model_matrix: Mat4,
    pub scene: std::sync::Arc<SceneUniforms>,
}

#[derive(Interpolate)]
struct GouraudVaryings {
    pub colour: Vec4,
}

struct GouraudVertexShader;
impl VertexShader for GouraudVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = GouraudVertexUniforms;
    type Varyings = GouraudVaryings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();

        let view_position = uniforms.scene.camera.view_matrix() * world_position;
        let clip_position = uniforms.scene.camera.projection_matrix() * view_position;

        // Transform normal into world space
        let normal = (uniforms.model_matrix.inverse().transpose() * vertex.normal.to_direction4())
            .xyz()
            .normalise();

        let mut colour = Vec4::new(0.1, 0.1, 0.1, 1.0);

        let world_pos = world_position.homogenize_to_vec3();

        for light in &uniforms.scene.lights {
            let light_dir = -light.direction.normalise();

            // diffuse
            let diffuse = normal.dot(&light_dir).max(0.0);

            colour += Into::<Vec4>::into(light.colour) * diffuse;

            // specular
            let view_dir = (uniforms.scene.camera.eye - world_pos).normalise();

            let reflect_dir = reflect(-light_dir, normal).normalise();
            let specular = view_dir.dot(&reflect_dir).max(0.0).powf(32.0);

            colour += Into::<Vec4>::into(light.colour) * specular;
        }

        let varyings = GouraudVaryings { colour };

        (clip_position, varyings)
    }
}

#[derive(Clone)]
struct PhongVertexUniforms {
    pub model_matrix: Mat4,
    pub scene: std::sync::Arc<SceneUniforms>,
}

#[derive(Interpolate)]
struct PhongVaryings {
    pub world_position: Vec3,
    pub normal: Vec3,
}

struct PhongVertexShader;
impl VertexShader for PhongVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = PhongVertexUniforms;
    type Varyings = PhongVaryings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();

        let view_position = uniforms.scene.camera.view_matrix() * world_position;

        let clip_position = uniforms.scene.camera.projection_matrix() * view_position;

        // Transform normal into world space
        let normal = (uniforms.model_matrix.inverse().transpose() * vertex.normal.to_direction4())
            .xyz()
            .normalise();

        let varyings = PhongVaryings {
            world_position: world_position.homogenize_to_vec3(),
            normal,
        };

        (clip_position, varyings)
    }
}

struct GouraudFragmentUniforms;

struct GouraudFragmentShader;
impl FragmentShader<GouraudVaryings> for GouraudFragmentShader {
    type Uniforms = GouraudFragmentUniforms;

    fn shade(&self, varyings: GouraudVaryings, _uniforms: &Self::Uniforms) -> Colour {
        varyings.colour.into()
    }
}

// Uniforms for the scene (don't vary per model/mesh)
struct SceneUniforms {
    camera: Camera,
    lights: Vec<DirectionalLight>,
}

struct PhongFragmentUniforms {
    scene: std::sync::Arc<SceneUniforms>,
    material: Option<Material>,
}

struct PhongFragmentShader;
impl FragmentShader<PhongVaryings> for PhongFragmentShader {
    type Uniforms = PhongFragmentUniforms;

    fn shade(&self, varyings: PhongVaryings, uniforms: &Self::Uniforms) -> Colour {
        let normal = varyings.normal.normalise();

        let mut colour = Colour::BLACK;

        let view_dir = (uniforms.scene.camera.eye - varyings.world_position).normalise();

        if let Some(material) = &uniforms.material {
            colour = material.ambient;

            for light in &uniforms.scene.lights {
                let light_dir = (-light.direction).normalise();

                // Diffuse
                let diffuse_strength = normal.dot(&light_dir).max(0.0);

                let diffuse = material.diffuse * light.colour * diffuse_strength;

                // Specular
                let reflect_dir = reflect(-light_dir, normal);

                let specular_strength =
                    view_dir.dot(&reflect_dir).max(0.0).powf(material.shininess);

                let specular = material.specular * light.colour * specular_strength;

                colour = colour + diffuse + specular;
            }
        }

        colour
    }
}

struct PipelinesApp {
    camera: Camera,
    camera_controller: OrbitControls,
    lights: Vec<DirectionalLight>,
    gouraud_teapot: Model<ObjVertex>,
    phong_teapot: Model<ObjVertex>,
    gouraud_pipeline: Pipeline<GouraudVertexShader, GouraudFragmentShader>,
    phong_pipeline: Pipeline<PhongVertexShader, PhongFragmentShader>,
    elapsed: f32,
}
impl PipelinesApp {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let gouraud_pipeline = Pipeline::new(GouraudVertexShader, GouraudFragmentShader)
            .with_culling_mode(CullingMode::None);

        let phong_pipeline = Pipeline::new(PhongVertexShader, PhongFragmentShader)
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
        let controls = OrbitControls::new(&camera);

        let light = DirectionalLight::new(Vec3::new(0.0, -1.0, -1.0), Colour::from_u32(0xfffde8));

        let mut gouraud_teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
        gouraud_teapot.transform.scale = Vec3::ONE * 0.2;
        gouraud_teapot.transform.rotation.y = 90_f32.to_radians();
        gouraud_teapot.transform.position.z = -0.75;
        gouraud_teapot.calculate_vertex_normals();

        let mut phong_teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
        phong_teapot.transform.scale = Vec3::ONE * 0.2;
        phong_teapot.transform.rotation.y = 90_f32.to_radians();
        phong_teapot.transform.position.z = 0.75;
        phong_teapot.calculate_vertex_normals();

        // quick hack to add a material to the teapot, since the .obj file doesn't have any materials defined
        let polished_brass = Material::new_simple(
            "Polished Brass".to_string(),
            Colour::from_u32(0x543808),
            Colour::from_u32(0x8b7500),
            Colour::from_u32(0xffffff),
            21.8,
        );
        phong_teapot.materials = vec![polished_brass];
        phong_teapot.meshes[0].material_index = Some(0);

        Ok(Self {
            camera,
            camera_controller: controls,
            lights: vec![light],
            gouraud_teapot,
            phong_teapot,
            gouraud_pipeline,
            phong_pipeline,
            elapsed: 0.0,
        })
    }
}

impl Application for PipelinesApp {
    fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        self.gouraud_teapot.transform.rotation.y = 0.5 * self.elapsed;
        self.phong_teapot.transform.rotation.y = 0.5 * self.elapsed;

        self.camera_controller
            .update_from_events(&mut self.camera, dt);
    }

    fn event(&mut self, event: AppEvent, handle: &mut AppHandle) {
        self.camera_controller.handle_event(event);

        // Exit the application if the Escape key is pressed
        if let AppEvent::Input(InputEvent::Key {
            key: InputKey::Escape,
            state: ButtonState::Released,
        }) = event
        {
            handle.request_exit();
        }
    }

    fn render<'frame>(
        &'frame mut self,
        frame: &mut cpu_rasteriser::renderer::Frame<'_, '_, 'frame>,
        _viewport: &Viewport,
    ) {
        let scene_uniforms = std::sync::Arc::new(SceneUniforms {
            camera: self.camera.clone(),
            lights: self.lights.clone(),
        });

        let gouraud_vertex_uniforms = GouraudVertexUniforms {
            model_matrix: self.gouraud_teapot.transform.model_matrix(),
            scene: scene_uniforms.clone(),
        };

        let phong_vertex_uniforms = PhongVertexUniforms {
            model_matrix: self.phong_teapot.transform.model_matrix(),
            scene: scene_uniforms.clone(),
        };

        self.gouraud_teapot.draw_to_frame(
            frame,
            &self.gouraud_pipeline,
            gouraud_vertex_uniforms.clone(),
            |_| GouraudFragmentUniforms,
        );

        self.phong_teapot.draw_to_frame(
            frame,
            &self.phong_pipeline,
            phong_vertex_uniforms.clone(),
            |mesh| PhongFragmentUniforms {
                scene: scene_uniforms.clone(),
                material: self
                    .phong_teapot
                    .materials
                    .get(mesh.material_index.unwrap())
                    .cloned(),
            },
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    MinifbEngine::new()
        .with_title("Pipelines Demo - ESC to exit")
        .with_size(WIDTH, HEIGHT)
        .run(PipelinesApp::new()?)
}
