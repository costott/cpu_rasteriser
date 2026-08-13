use engine::prelude::*;

use cpu_rasteriser::prelude::*;

use cpu_rasteriser::{
    graphics::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
    renderer::{CullingMode, Pipeline, Renderer},
};

use minifb::{Key, Window, WindowOptions};
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

// Uniforms for the scene (don't vary per model/mesh)
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

/// Reflects a vector around a normal, using the formula: R = V - 2 * (V . N) * N
fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * 2.0 * vector.dot(&normal)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "CPU rasteriser - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    window.set_target_fps(60);
    window.set_cursor_visibility(false);

    let viewport = Viewport::new(WIDTH, HEIGHT);

    let mut renderer = Renderer::new(&viewport)?;

    let phong_pipeline = Pipeline::new(BasicVertexShader, PhongFragmentShader)
        .with_culling_mode(CullingMode::BackFace);

    let mut camera = Camera::new(
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
    let mut controls = FirstPersonControls::new(&camera);

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

    let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;
    teapot.transform.scale = Vec3::ONE * 0.1;

    let mut loaded_cube = load_obj(std::path::Path::new("assets/cube/cube.obj"))?;
    loaded_cube.transform.scale = Vec3::ONE * 0.5;

    // let mut scene = Scene::new(camera);
    // scene.add_light(DirectionalLight::new(
    //     Vec3::new(0.0, -1.0, -1.0),
    //     Colour::from_u32(0xfffde8),
    // ));
    // scene.add_model(floor_model);
    // scene.add_model(cube1);
    // scene.add_model(cube2);
    let lights = vec![DirectionalLight::new(
        Vec3::new(0.0, -1.0, -1.0),
        Colour::from_u32(0xfffde8),
    )];

    // let mut t: f32 = 0.0;
    // let mut angle: f32 = 0.0;

    let mut previous_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let dt = std::time::Instant::now()
            .duration_since(previous_time)
            .as_secs_f32();
        previous_time = std::time::Instant::now();
        // t += dt;

        controls.update(&mut camera, &window, dt);

        let mut frame = renderer.begin_frame(&viewport);

        let scene_uniforms = Arc::new(SceneUniforms {
            camera: camera.clone(),
            lights: lights.to_vec(),
        });

        let floor_vertex_uniforms = VertexUniforms {
            model_matrix: floor_model.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };
        for draw_call in floor_model.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: floor_model
                .materials
                .get(mesh.material_index.unwrap())
                .cloned(),
        }) {
            frame.draw(&phong_pipeline, draw_call, floor_vertex_uniforms.clone());
        }

        let cube1_vertex_uniforms = VertexUniforms {
            model_matrix: cube1.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };
        for draw_call in cube1.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: cube1.materials.get(mesh.material_index.unwrap()).cloned(),
        }) {
            frame.draw(&phong_pipeline, draw_call, cube1_vertex_uniforms.clone());
        }

        let cube2_vertex_uniforms = VertexUniforms {
            model_matrix: cube2.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };
        for draw_call in cube2.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: cube2.materials.get(mesh.material_index.unwrap()).cloned(),
        }) {
            frame.draw(&phong_pipeline, draw_call, cube2_vertex_uniforms.clone());
        }

        let loaded_cube_vertex_uniforms = VertexUniforms {
            model_matrix: loaded_cube.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };
        for draw_call in loaded_cube.draw_calls(|mesh| FragmentUniforms {
            scene: scene_uniforms.clone(),
            material: loaded_cube
                .materials
                .get(mesh.material_index.unwrap())
                .cloned(),
        }) {
            frame.draw(
                &phong_pipeline,
                draw_call,
                loaded_cube_vertex_uniforms.clone(),
            );
        }

        frame.finish();

        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
