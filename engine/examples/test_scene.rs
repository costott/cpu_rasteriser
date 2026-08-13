use engine::prelude::*;

use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
    renderer::{CullingMode, Frame, Pipeline},
    viewport::Viewport,
};

use std::sync::Arc;

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
        };

        (clip_position, varyings)
    }
}

struct SceneUniforms {
    camera: Camera,
    lights: Vec<DirectionalLight>,
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
    lights: Vec<DirectionalLight>,
    floor_model: Model<ObjVertex>,
    cube1: Model<ObjVertex>,
    cube2: Model<ObjVertex>,
    loaded_cube: Model<ObjVertex>,
    phong_pipeline: Pipeline<BasicVertexShader, PhongFragmentShader>,
    floor_vertex_uniforms: VertexUniforms,
    cube1_vertex_uniforms: VertexUniforms,
    cube2_vertex_uniforms: VertexUniforms,
    loaded_cube_vertex_uniforms: VertexUniforms,
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
                Vec3::new(0.5, 0.25, 0.5),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.5, 0.5, 0.5),
            ),
        )
        .unwrap();
        cube2.calculate_vertex_normals();

        let mut loaded_cube = load_obj(std::path::Path::new("assets/cube/cube.obj"))?;
        loaded_cube.transform.scale = Vec3::ONE * 0.5;

        let floor_vertex_uniforms = VertexUniforms {
            model_matrix: floor_model.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };

        let cube1_vertex_uniforms = VertexUniforms {
            model_matrix: cube1.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };

        let cube2_vertex_uniforms = VertexUniforms {
            model_matrix: cube2.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };

        let loaded_cube_vertex_uniforms = VertexUniforms {
            model_matrix: loaded_cube.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };

        Ok(Self {
            camera,
            lights: vec![DirectionalLight::new(
                Vec3::new(0.0, -1.0, -1.0),
                Colour::from_u32(0xfffde8),
            )],
            floor_model,
            cube1,
            cube2,
            loaded_cube,
            phong_pipeline: Pipeline::new(BasicVertexShader, PhongFragmentShader)
                .with_culling_mode(CullingMode::BackFace),
            floor_vertex_uniforms,
            cube1_vertex_uniforms,
            cube2_vertex_uniforms,
            loaded_cube_vertex_uniforms,
            elapsed: 0.0,
        })
    }
}

impl Application for TestSceneApp {
    fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        self.camera.eye.x = self.elapsed.sin() * 0.25;
        self.camera.eye.z = 1.25 + self.elapsed.cos() * 0.2;
    }

    fn render<'a>(&'a mut self, frame: &mut Frame<'a, 'a>, _viewport: &'a Viewport) {
        let scene_uniforms = Arc::new(SceneUniforms {
            camera: self.camera.clone(),
            lights: self.lights.clone(),
        });

        self.floor_vertex_uniforms = VertexUniforms {
            model_matrix: self.floor_model.transform.model_matrix(),
            view_matrix: self.camera.view_matrix(),
            projection_matrix: self.camera.projection_matrix(),
        };
        for draw_call in self.floor_model.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: self
                .floor_model
                .materials
                .get(mesh.material_index.unwrap())
                .cloned(),
        }) {
            frame.draw(&self.phong_pipeline, draw_call, &self.floor_vertex_uniforms);
        }

        self.cube1_vertex_uniforms = VertexUniforms {
            model_matrix: self.cube1.transform.model_matrix(),
            view_matrix: self.camera.view_matrix(),
            projection_matrix: self.camera.projection_matrix(),
        };
        for draw_call in self.cube1.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: self
                .cube1
                .materials
                .get(mesh.material_index.unwrap())
                .cloned(),
        }) {
            frame.draw(&self.phong_pipeline, draw_call, &self.cube1_vertex_uniforms);
        }

        self.cube2_vertex_uniforms = VertexUniforms {
            model_matrix: self.cube2.transform.model_matrix(),
            view_matrix: self.camera.view_matrix(),
            projection_matrix: self.camera.projection_matrix(),
        };
        for draw_call in self.cube2.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: self
                .cube2
                .materials
                .get(mesh.material_index.unwrap())
                .cloned(),
        }) {
            frame.draw(&self.phong_pipeline, draw_call, &self.cube2_vertex_uniforms);
        }

        self.loaded_cube_vertex_uniforms = VertexUniforms {
            model_matrix: self.loaded_cube.transform.model_matrix(),
            view_matrix: self.camera.view_matrix(),
            projection_matrix: self.camera.projection_matrix(),
        };
        for draw_call in self.loaded_cube.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: self
                .loaded_cube
                .materials
                .get(mesh.material_index.unwrap())
                .cloned(),
        }) {
            frame.draw(
                &self.phong_pipeline,
                draw_call,
                &self.loaded_cube_vertex_uniforms,
            );
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Engine::run(TestSceneApp::new()?)
    Engine::with_backend::<WinitEngine, _>(TestSceneApp::new()?)
}
