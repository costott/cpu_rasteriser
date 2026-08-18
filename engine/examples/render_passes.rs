use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

// -----------------------------------------------------------------------------
// Shared scene uniforms
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct SceneUniforms {
    camera: Camera,
    light: DirectionalLight,
}

// -----------------------------------------------------------------------------
// Teapot shader
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct PhongVertexUniforms {
    model_matrix: Mat4,
    scene: Arc<SceneUniforms>,
}

#[derive(Interpolate)]
struct PhongVaryings {
    world_position: Vec3,
    normal: Vec3,
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

        let normal = (uniforms.model_matrix.inverse().transpose() * vertex.normal.to_direction4())
            .xyz()
            .normalise();

        (
            clip_position,
            PhongVaryings {
                world_position: world_position.homogenize_to_vec3(),
                normal,
            },
        )
    }
}

#[derive(Clone)]
struct PhongFragmentUniforms {
    scene: Arc<SceneUniforms>,
    material: Material,
}

struct PhongFragmentShader;

impl FragmentShader<PhongVaryings> for PhongFragmentShader {
    type Uniforms = PhongFragmentUniforms;

    fn shade(&self, varyings: PhongVaryings, uniforms: &Self::Uniforms) -> Colour {
        let normal = varyings.normal.normalise();

        let view_dir = (uniforms.scene.camera.eye - varyings.world_position).normalise();

        let light_dir = (-uniforms.scene.light.direction).normalise();

        // Ambient
        let mut colour = uniforms.material.ambient;

        // Diffuse
        let diffuse_strength = normal.dot(&light_dir).max(0.0);

        colour += uniforms.material.diffuse * uniforms.scene.light.colour * diffuse_strength;

        // Specular
        let reflect_dir = reflect(-light_dir, normal).normalise();

        let specular_strength = view_dir
            .dot(&reflect_dir)
            .max(0.0)
            .powf(uniforms.material.shininess);

        colour += uniforms.material.specular * uniforms.scene.light.colour * specular_strength;

        colour
    }
}

/// Reflects a vector around a normal, using the formula: R = V - 2 * (V . N) * N
fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * 2.0 * vector.dot(&normal)
}

// -----------------------------------------------------------------------------
// Cloud background
// -----------------------------------------------------------------------------

#[derive(Interpolate)]
struct CloudVaryings {
    uv: Vec2,
}

struct CloudVertexShader;

#[derive(Clone)]
struct CloudVertexUniforms;

impl VertexShader for CloudVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = CloudVertexUniforms;
    type Varyings = CloudVaryings;

    fn shade(&self, vertex: Self::Vertex, _uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        (vertex.position.to_point4(), CloudVaryings { uv: vertex.uv })
    }
}

struct CloudFragmentUniforms {
    time: f32,
}

struct CloudFragmentShader;

impl FragmentShader<CloudVaryings> for CloudFragmentShader {
    type Uniforms = CloudFragmentUniforms;

    fn shade(&self, varyings: CloudVaryings, uniforms: &Self::Uniforms) -> Colour {
        let uv = varyings.uv;

        // Very cheap procedural "clouds".
        //
        // This isn't proper Perlin/fBm noise; that's deliberate for an example.
        // Layering a few sine waves gives something cloud-like without needing
        // another noise implementation in the example.
        let t = uniforms.time * 0.04;

        let n1 = ((uv.x * 6.0 + t).sin() + (uv.y * 4.0 - t * 0.7).cos()) * 0.5;

        let n2 = ((uv.x * 13.0 - t * 1.3).sin() * (uv.y * 11.0 + t).cos()) * 0.25;

        let cloud = (n1 + n2 + 0.5).clamp(0.0, 1.0);

        let sky = Colour::lerp(
            &Colour::from_u32(0x0d1e2f),
            &Colour::from_u32(0x4d7ebf),
            uv.y,
        );

        let cloud_colour = Colour::from_u32(0xf2f7ff);

        Colour::lerp(&sky, &cloud_colour, cloud * 0.75)
    }
}

// -----------------------------------------------------------------------------
// Post-processing
// -----------------------------------------------------------------------------

#[derive(Interpolate)]
struct PostProcessVaryings {
    uv: Vec2,
}

struct PostProcessVertexShader;

#[derive(Clone)]
struct PostProcessVertexUniforms;

impl VertexShader for PostProcessVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = PostProcessVertexUniforms;
    type Varyings = PostProcessVaryings;

    fn shade(&self, vertex: Self::Vertex, _uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        (
            vertex.position.to_point4(),
            PostProcessVaryings { uv: vertex.uv },
        )
    }
}

struct PostProcessFragmentUniforms {
    texture: Arc<TextureSampler>,
    time: f32,
}

struct ChromaticAberrationFragmentShader;
impl FragmentShader<PostProcessVaryings> for ChromaticAberrationFragmentShader {
    type Uniforms = PostProcessFragmentUniforms;

    fn shade(&self, varyings: PostProcessVaryings, uniforms: &Self::Uniforms) -> Colour {
        let uv = varyings.uv;

        // Sample the result of the previous passes.
        let source = uniforms.texture.sample_linear_clamp(uv);

        // Slight animated RGB split to make it obvious this is a post-process.
        let offset = 0.0025 * (uniforms.time * 0.8).sin();

        let red_uv = Vec2::new(uv.x + offset, uv.y);
        let blue_uv = Vec2::new(uv.x - offset, uv.y);

        let red = uniforms.texture.sample_linear_clamp(red_uv).r;
        let blue = uniforms.texture.sample_linear_clamp(blue_uv).b;

        Colour {
            r: red,
            g: source.g,
            b: blue,
            a: source.a,
        }
    }
}

struct VignetteFragmentShader;
impl FragmentShader<PostProcessVaryings> for VignetteFragmentShader {
    type Uniforms = PostProcessFragmentUniforms;

    fn shade(&self, varyings: PostProcessVaryings, uniforms: &Self::Uniforms) -> Colour {
        let uv = varyings.uv;

        let source = uniforms.texture.sample_linear_clamp(uv);

        let centre = Vec2::new(0.5, 0.5);
        let distance = (uv - centre).length();

        let vignette_strength = 1.0 - distance * 1.5;

        Colour {
            r: (source.r as f32 * vignette_strength) as u8,
            g: (source.g as f32 * vignette_strength) as u8,
            b: (source.b as f32 * vignette_strength) as u8,
            a: source.a,
        }
    }
}

struct GreyscaleFragmentShader;
impl FragmentShader<PostProcessVaryings> for GreyscaleFragmentShader {
    type Uniforms = PostProcessFragmentUniforms;

    fn shade(
        &self,
        varyings: PostProcessVaryings,
        uniforms: &PostProcessFragmentUniforms,
    ) -> Colour {
        let colour = uniforms.texture.sample_linear_clamp(varyings.uv);

        let luminance = colour.r as f32 * 0.299 + colour.g as f32 * 0.587 + colour.b as f32 * 0.114;

        Colour {
            r: luminance as u8,
            g: luminance as u8,
            b: luminance as u8,
            a: colour.a,
        }
    }
}

struct InvertFragmentShader;
impl FragmentShader<PostProcessVaryings> for InvertFragmentShader {
    type Uniforms = PostProcessFragmentUniforms;

    fn shade(
        &self,
        varyings: PostProcessVaryings,
        uniforms: &PostProcessFragmentUniforms,
    ) -> Colour {
        let colour = uniforms.texture.sample_linear_clamp(varyings.uv);

        Colour {
            r: 255 - colour.r,
            g: 255 - colour.g,
            b: 255 - colour.b,
            a: colour.a,
        }
    }
}

// This is naive bc it's better to do a two-pass separable Gaussian blur, but this is just an example.
struct NaiveGaussianBlurFragmentShader;
impl FragmentShader<PostProcessVaryings> for NaiveGaussianBlurFragmentShader {
    type Uniforms = PostProcessFragmentUniforms;

    fn shade(
        &self,
        varyings: PostProcessVaryings,
        uniforms: &PostProcessFragmentUniforms,
    ) -> Colour {
        let uv = varyings.uv;

        let offsets = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(0.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];

        let weights = [
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            4.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
        ];

        let mut r = 0.0;
        let mut g = 0.0;
        let mut b = 0.0;

        let texel_size = Vec2::new(
            1.0 / uniforms.texture.width() as f32,
            1.0 / uniforms.texture.height() as f32,
        );

        for (offset, weight) in offsets.iter().zip(weights.iter()) {
            let sample_uv = uv + *offset * texel_size;
            let sample_colour = uniforms.texture.sample_nearest_clamp(sample_uv);

            r += sample_colour.r as f32 * weight;
            g += sample_colour.g as f32 * weight;
            b += sample_colour.b as f32 * weight;
        }

        Colour {
            r: r as u8,
            g: g as u8,
            b: b as u8,
            a: 255,
        }
    }
}

struct SobelEdgeDetectionFragmentShader;
impl FragmentShader<PostProcessVaryings> for SobelEdgeDetectionFragmentShader {
    type Uniforms = PostProcessFragmentUniforms;

    fn shade(
        &self,
        varyings: PostProcessVaryings,
        uniforms: &PostProcessFragmentUniforms,
    ) -> Colour {
        let uv = varyings.uv;

        let texel_size = Vec2::new(
            1.0 / uniforms.texture.width() as f32,
            1.0 / uniforms.texture.height() as f32,
        );

        let offsets = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(0.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];

        let gx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];

        let gy = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut gradient_x = 0.0;
        let mut gradient_y = 0.0;

        for i in 0..9 {
            let sample_uv = uv + offsets[i] * texel_size;
            let sample = uniforms.texture.sample_nearest_clamp(sample_uv);

            let luminance =
                0.299 * sample.r as f32 + 0.587 * sample.g as f32 + 0.114 * sample.b as f32;

            gradient_x += luminance * gx[i];
            gradient_y += luminance * gy[i];
        }

        let magnitude = (gradient_x * gradient_x + gradient_y * gradient_y).sqrt();

        let magnitude = magnitude.clamp(0.0, 255.0) as u8;

        Colour::new(magnitude, magnitude, magnitude, 255)
    }
}

/// Small helper trait to represent a post processing pipeline to allow for dynamic dispatch of different post processing effects.
trait PostProcessPipeline {
    fn draw<'a>(
        &'a self,
        pass: &mut RenderPass<'a, 'a>,
        quad: &'a Model<ObjVertex>,
        texture: Arc<TextureSampler>,
        time: f32,
    );
}

impl<F> PostProcessPipeline for Pipeline<PostProcessVertexShader, F>
where
    F: FragmentShader<PostProcessVaryings, Uniforms = PostProcessFragmentUniforms> + 'static,
{
    fn draw<'a>(
        &'a self,
        pass: &mut RenderPass<'a, 'a>,
        quad: &'a Model<ObjVertex>,
        texture: Arc<TextureSampler>,
        time: f32,
    ) {
        quad.draw_to_render_pass(pass, self, PostProcessVertexUniforms, |_| {
            PostProcessFragmentUniforms {
                texture: texture.clone(),
                time,
            }
        });
    }
}

// -----------------------------------------------------------------------------
// Application
// -----------------------------------------------------------------------------

struct MultipassApp {
    camera: Camera,
    camera_controller: OrbitControls,
    light: DirectionalLight,

    teapot: Model<ObjVertex>,

    fullscreen_quad: Model<ObjVertex>,

    scene_target: RenderTarget,

    cloud_pipeline: Pipeline<CloudVertexShader, CloudFragmentShader>,
    teapot_pipeline: Pipeline<PhongVertexShader, PhongFragmentShader>,

    post_processes: Vec<Box<dyn PostProcessPipeline>>,
    post_process_index: usize,

    elapsed: f32,
}

impl MultipassApp {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let camera = Camera::new(
            Vec3::new(0.0, 1.5, 2.8),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective(PerspectiveProjection::new(
                90.0,
                WIDTH as f32 / HEIGHT as f32,
                0.01,
                50.0,
            )),
        );

        let camera_controller = OrbitControls::new(&camera);

        let light = DirectionalLight::new(Vec3::new(-0.4, -1.0, -0.5), Colour::from_u32(0xffffff));

        // ---------------------------------------------------------------------
        // Teapot
        // ---------------------------------------------------------------------

        let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;

        teapot.transform.scale = Vec3::ONE * 0.55;
        teapot.transform.position = Vec3::new(0.0, -0.35, 0.0);
        teapot.calculate_vertex_normals();

        let material = Material::new_simple(
            "Polished Ceramic".to_string(),
            Colour::from_u32(0x080c14),
            Colour::from_u32(0xfcdcb8),
            Colour::from_u32(0xffffff),
            64.0,
        );

        teapot.materials = vec![material];
        teapot.meshes[0].material_index = Some(0);

        // ---------------------------------------------------------------------
        // Fullscreen quad
        // ---------------------------------------------------------------------

        let mut fullscreen_quad = load_obj(std::path::Path::new("assets/fullscreen_quad.obj"))?;

        fullscreen_quad.transform.position = Vec3::ZERO;
        fullscreen_quad.transform.rotation = Vec3::ZERO;
        fullscreen_quad.transform.scale = Vec3::ONE;

        // ---------------------------------------------------------------------
        // Offscreen target
        // ---------------------------------------------------------------------

        //
        // This is the important part of the example:
        //
        //   pass 1 -> clouds -> scene_target
        //   pass 2 -> teapot -> scene_target
        //   scene_target -> Texture
        //   pass 3 -> post process -> presentation target
        //
        let scene_target = RenderTarget::new(Extent::new(WIDTH, HEIGHT)).with_depth();

        let cloud_pipeline = Pipeline::new(CloudVertexShader, CloudFragmentShader)
            .with_culling_mode(CullingMode::None);

        let teapot_pipeline = Pipeline::new(PhongVertexShader, PhongFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DEFAULT);

        let greyscale_pipeline = Pipeline::new(PostProcessVertexShader, GreyscaleFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED);

        let chromatic_aberration_pipeline =
            Pipeline::new(PostProcessVertexShader, ChromaticAberrationFragmentShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED);

        let vignette_pipeline = Pipeline::new(PostProcessVertexShader, VignetteFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED);

        let gaussian_blur_pipeline =
            Pipeline::new(PostProcessVertexShader, NaiveGaussianBlurFragmentShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED);

        let invert_pipeline = Pipeline::new(PostProcessVertexShader, InvertFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED);

        let sobel_edge_detection_pipeline =
            Pipeline::new(PostProcessVertexShader, SobelEdgeDetectionFragmentShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED);

        let post_processes: Vec<Box<dyn PostProcessPipeline>> = vec![
            Box::new(chromatic_aberration_pipeline),
            Box::new(vignette_pipeline),
            Box::new(greyscale_pipeline),
            Box::new(invert_pipeline),
            Box::new(gaussian_blur_pipeline),
            Box::new(sobel_edge_detection_pipeline),
        ];

        Ok(Self {
            camera,
            camera_controller,
            light,

            teapot,
            fullscreen_quad,

            scene_target,

            cloud_pipeline,
            teapot_pipeline,

            post_processes,
            post_process_index: 0,

            elapsed: 0.0,
        })
    }
}

impl Application for MultipassApp {
    fn update(&mut self, dt: f32) {
        self.elapsed += dt;

        self.teapot.transform.rotation.y = self.elapsed * 0.5;

        self.camera_controller
            .update_from_events(&mut self.camera, dt);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect_ratio(width as f32 / height as f32);
        self.scene_target
            .resize(Extent::new(width as usize, height as usize));
    }

    fn event(&mut self, event: AppEvent, handle: &mut AppHandle) {
        self.camera_controller.handle_event(event);

        if let AppEvent::Input(InputEvent::Key {
            key: InputKey::Escape,
            state: ButtonState::Released,
        }) = event
        {
            handle.request_exit();
        }

        if let AppEvent::Input(InputEvent::Key {
            key: InputKey::Space,
            state: ButtonState::Released,
        }) = event
        {
            self.post_process_index = (self.post_process_index + 1) % self.post_processes.len();
        }
    }

    fn render<'frame>(&mut self, context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        // ---------------------------------------------------------------------
        // PASS 1: cloudy background
        // ---------------------------------------------------------------------

        let extent = self.scene_target.extent();

        let mut pass = context.begin_render_pass(
            &mut self.scene_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        self.fullscreen_quad.draw_to_render_pass(
            &mut pass,
            &self.cloud_pipeline,
            CloudVertexUniforms,
            |_| CloudFragmentUniforms { time: self.elapsed },
        );

        pass.finish(); // always remember to finish your passes, or the compiler won't let you start a new one!

        // ---------------------------------------------------------------------
        // PASS 2: teapot over the clouds
        // ---------------------------------------------------------------------

        let mut pass = context.begin_render_pass(
            &mut self.scene_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Load,
                depth_load_op: Some(LoadOp::Clear(1.0)),
            },
        );

        let scene = Arc::new(SceneUniforms {
            camera: self.camera.clone(),
            light: self.light.clone(),
        });

        let uniforms = PhongVertexUniforms {
            model_matrix: self.teapot.transform.model_matrix(),
            scene: scene.clone(),
        };

        self.teapot
            .draw_to_render_pass(&mut pass, &self.teapot_pipeline, uniforms, |mesh| {
                let material = self
                    .teapot
                    .materials
                    .get(mesh.material_index.unwrap())
                    .cloned()
                    .expect("teapot mesh should have a material");

                PhongFragmentUniforms {
                    scene: scene.clone(),
                    material,
                }
            });

        pass.finish();

        // ---------------------------------------------------------------------
        // Convert the result of passes 1 + 2 into a Texture.
        // ---------------------------------------------------------------------

        let scene_texture = Arc::new(render_target_sampler(&self.scene_target));

        // ---------------------------------------------------------------------
        // PASS 3: post-process into the presentation target
        // ---------------------------------------------------------------------

        let extent = context.presentation_target().extent();

        let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: None,
        });

        self.post_processes[self.post_process_index].draw(
            &mut pass,
            &self.fullscreen_quad,
            scene_texture.clone(),
            self.elapsed,
        );

        pass.finish()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    MinifbEngine::new()
        .with_title("Multipass Rendering - SPACE to change post processing - ESC to exit")
        .with_size(WIDTH, HEIGHT)
        .with_options(minifb::WindowOptions {
            resize: true,
            ..Default::default()
        })
        .run(MultipassApp::new()?)
}
