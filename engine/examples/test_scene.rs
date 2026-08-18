use engine::prelude::*;

use cpu_rasteriser::prelude::*;

use std::sync::Arc;

struct SceneUniforms {
    camera: Camera,
    lights: Vec<DirectionalLight>,
}

#[derive(Clone)]
struct VertexUniforms {
    pub model_matrix: Mat4,
    pub scene: Arc<SceneUniforms>,
}

#[derive(Interpolate)]
struct Varyings {
    pub world_position: Vec3,
    pub colour: Vec4,
    pub normal: Vec3,
}

struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = VertexUniforms;
    type Varyings = Varyings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let world_position = uniforms.model_matrix * vertex.position.to_point4();
        let normal_matrix = uniforms.model_matrix.inverse().transpose();

        let view_position = uniforms.scene.camera.view_matrix() * world_position;
        let clip_position = uniforms.scene.camera.projection_matrix() * view_position;

        let varyings = Varyings {
            world_position: world_position.homogenize_to_vec3(),
            colour: vertex.colour.into(),
            normal: (normal_matrix * vertex.normal.to_direction4())
                .xyz()
                .normalise(),
        };

        (clip_position, varyings)
    }
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
        let mut colour = Colour::BLACK;
        let view_dir = (uniforms.scene.camera.eye - varyings.world_position).normalise();

        if let Some(material) = &uniforms.material {
            colour = material.ambient;

            for light in &uniforms.scene.lights {
                let light_dir = (-light.direction).normalise();

                let diffuse_strength = normal.dot(&light_dir).max(0.0);
                let diffuse = material.diffuse * light.colour * diffuse_strength;

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

/// Reflects a vector around a normal, using the formula: R = V - 2 * (V . N) * N
fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * 2.0 * vector.dot(&normal)
}

struct TestSceneApp {
    camera: Camera,
    camera_controller: FirstPersonControls,
    lights: Vec<DirectionalLight>,
    floor_model: Model<ObjVertex>,
    cube1: Model<ObjVertex>,
    cube2: Model<ObjVertex>,
    loaded_cube: Model<ObjVertex>,
    phong_pipeline: Pipeline<BasicVertexShader, PhongFragmentShader>,
    elapsed: f32,
}

impl TestSceneApp {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let camera = Camera::new(
            Vec3::new(0.0, 0.75, 1.25),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective(PerspectiveProjection::new(90.0, 640.0 / 360.0, 0.01, 50.0)),
        );
        let camera_controller = FirstPersonControls::new(&camera);

        let floor_material = Material::new_simple(
            "Floor".to_string(),
            Colour::from_u32(0x808080),
            Colour::from_u32(0x404040),
            Colour::from_u32(0xffffff),
            1.0,
        );

        let red_plastic = Material::new_simple(
            "Red Plastic".to_string(),
            Colour::from_u32(0xff0000),
            Colour::from_u32(0x990000),
            Colour::from_u32(0xffffff),
            64.0,
        );

        let polished_brass = Material::new_simple(
            "Polished Brass".to_string(),
            Colour::from_u32(0x543808),
            Colour::from_u32(0x8b7500),
            Colour::from_u32(0xffffff),
            21.8,
        );

        let mut floor_model = Model::new(
            vec![Mesh::cube(Colour::from_u32(0x808080), Some(0))],
            vec![floor_material],
            ModelTransform::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(50.0, 0.1, 50.0),
            ),
        )
        .unwrap();
        floor_model.calculate_vertex_normals();

        let mut cube1 = Model::new(
            vec![Mesh::cube(Colour::WHITE, Some(0))],
            vec![polished_brass],
            ModelTransform::new(
                Vec3::new(-0.8, 0.5, -1.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            ),
        )
        .unwrap();
        cube1.calculate_vertex_normals();

        let mut cube2 = Model::new(
            vec![Mesh::cube(Colour::WHITE, Some(0))],
            vec![red_plastic],
            ModelTransform::new(
                Vec3::new(0.5, 0.0, 0.5),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.5, 0.5, 0.5),
            ),
        )
        .unwrap();
        cube2.calculate_vertex_normals();

        let mut loaded_cube = load_obj(std::path::Path::new("assets/cube/cube.obj"))?;
        loaded_cube.transform.scale = Vec3::ONE * 0.5;

        Ok(Self {
            camera,
            camera_controller,
            lights: vec![DirectionalLight::new(
                Vec3::new(0.0, -1.0, -1.0),
                Colour::from_u32(0xfffde8),
            )],
            floor_model,
            cube1,
            cube2,
            loaded_cube,
            phong_pipeline: Pipeline::new(BasicVertexShader, PhongFragmentShader)
                .with_culling_mode(CullingMode::BackFace)
                .with_depth_state(DepthState::DEFAULT),
            elapsed: 0.0,
        })
    }
}

impl Application for TestSceneApp {
    fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        self.camera_controller
            .update_from_events(&mut self.camera, dt);
    }

    fn event(&mut self, event: AppEvent, _handle: &mut AppHandle) {
        if let AppEvent::Input(InputEvent::Key {
            key: InputKey::Escape,
            state: ButtonState::Released,
        }) = event
        {
            self.camera_controller.toggle_cursor_grabbed();
        }

        self.camera_controller.handle_event(event);
    }

    fn window_state(&self) -> WindowState {
        WindowState {
            cursor: if self.camera_controller.cursor_grabbed() {
                WindowCursorSettings {
                    visible: false,
                    grab: CursorGrab::Locked,
                }
            } else {
                WindowCursorSettings::default()
            },
        }
    }

    fn render<'frame>(&mut self, context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        let extent = context.presentation_target().extent();

        let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: Some(LoadOp::Clear(1.0)),
        });

        let scene_uniforms = Arc::new(SceneUniforms {
            camera: self.camera.clone(),
            lights: self.lights.clone(),
        });

        let floor_vertex_uniforms = VertexUniforms {
            model_matrix: self.floor_model.transform.model_matrix(),
            scene: scene_uniforms.clone(),
        };
        self.floor_model.draw_to_render_pass(
            &mut pass,
            &self.phong_pipeline,
            floor_vertex_uniforms,
            |mesh| FragmentUniforms {
                scene: scene_uniforms.clone(),
                material: self
                    .floor_model
                    .materials
                    .get(mesh.material_index.unwrap())
                    .cloned(),
            },
        );

        let cube1_vertex_uniforms = VertexUniforms {
            model_matrix: self.cube1.transform.model_matrix(),
            scene: scene_uniforms.clone(),
        };
        self.cube1.draw_to_render_pass(
            &mut pass,
            &self.phong_pipeline,
            cube1_vertex_uniforms,
            |mesh| FragmentUniforms {
                scene: scene_uniforms.clone(),
                material: self
                    .cube1
                    .materials
                    .get(mesh.material_index.unwrap())
                    .cloned(),
            },
        );

        let cube2_vertex_uniforms = VertexUniforms {
            model_matrix: self.cube2.transform.model_matrix(),
            scene: scene_uniforms.clone(),
        };
        self.cube2.draw_to_render_pass(
            &mut pass,
            &self.phong_pipeline,
            cube2_vertex_uniforms,
            |mesh| FragmentUniforms {
                scene: scene_uniforms.clone(),
                material: self
                    .cube2
                    .materials
                    .get(mesh.material_index.unwrap())
                    .cloned(),
            },
        );

        let loaded_cube_vertex_uniforms = VertexUniforms {
            model_matrix: self.loaded_cube.transform.model_matrix(),
            scene: scene_uniforms.clone(),
        };
        self.loaded_cube.draw_to_render_pass(
            &mut pass,
            &self.phong_pipeline,
            loaded_cube_vertex_uniforms,
            |mesh| FragmentUniforms {
                scene: scene_uniforms.clone(),
                material: self
                    .loaded_cube
                    .materials
                    .get(mesh.material_index.unwrap())
                    .cloned(),
            },
        );

        pass.finish()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = TestSceneApp::new()?;
    WinitEngine::new()
        .with_window_attributes(
            winit::window::Window::default_attributes()
                .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
        )
        .run(app)
}
