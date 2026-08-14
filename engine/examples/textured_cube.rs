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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Textured Cube Demo - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    window.set_target_fps(60);

    let viewport = Viewport::new(WIDTH, HEIGHT);

    let mut renderer = Renderer::new(&viewport)?;

    let phong_pipeline =
        Pipeline::new(BasicVertexShader, PhongFragmentShader).with_culling_mode(CullingMode::None);

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
    let mut controls = OrbitControls::new(&camera);

    let mut cube = load_obj(std::path::Path::new("assets/dice/cube-tex.obj"))?;
    cube.transform.position = Vec3::new(-0.5, -0.5, -0.5);
    cube.calculate_vertex_normals();

    let lights = vec![DirectionalLight::new(
        Vec3::new(-0.5, -1.0, 2.0).normalise(),
        Colour::from_u32(0xfffde8),
    )];
    let ambient_light = Colour::from_u32(0x202020);

    let mut previous_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let dt = std::time::Instant::now()
            .duration_since(previous_time)
            .as_secs_f32();
        previous_time = std::time::Instant::now();

        controls.update(&mut camera, &window, dt);

        let vertex_uniforms = VertexUniforms {
            model_matrix: cube.transform.model_matrix(),
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
        };

        let mut frame = renderer.begin_frame(&viewport);

        let scene_uniforms = Arc::new(SceneUniforms {
            camera: camera.clone(),
            lights: lights.to_vec(),
            ambient_light,
        });

        cube.draw_to_frame(&mut frame, &phong_pipeline, vertex_uniforms, |mesh| {
            FragmentUniforms {
                scene: scene_uniforms.clone(),
                material: cube.materials.get(mesh.material_index.unwrap()).cloned(),
            }
        });

        frame.finish();

        window
            .update_with_buffer(renderer.pixels(), WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
