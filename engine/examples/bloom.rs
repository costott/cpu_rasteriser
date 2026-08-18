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
// Teapot
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

        let mut colour = uniforms.material.ambient;

        let diffuse_strength = normal.dot(&light_dir).max(0.0);

        colour += uniforms.material.diffuse * uniforms.scene.light.colour * diffuse_strength;

        let reflect_dir = reflect(-light_dir, normal).normalise();

        let specular_strength = view_dir
            .dot(&reflect_dir)
            .max(0.0)
            .powf(uniforms.material.shininess);

        // Deliberately allow the specular highlight to exceed 1.0.
        // These HDR values are what the bloom pass will extract.
        colour +=
            uniforms.material.specular * uniforms.scene.light.colour * specular_strength * 4.0;

        colour
    }
}

fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * 2.0 * vector.dot(&normal)
}

// -----------------------------------------------------------------------------
// Background
// -----------------------------------------------------------------------------

#[derive(Interpolate)]
struct BackgroundVaryings {
    uv: Vec2,
}

struct BackgroundVertexShader;

#[derive(Clone)]
struct BackgroundVertexUniforms;

impl VertexShader for BackgroundVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = BackgroundVertexUniforms;
    type Varyings = BackgroundVaryings;

    fn shade(&self, vertex: Self::Vertex, _uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        (
            vertex.position.to_point4(),
            BackgroundVaryings { uv: vertex.uv },
        )
    }
}

struct BackgroundFragmentUniforms {
    time: f32,
}

struct BackgroundFragmentShader;

impl FragmentShader<BackgroundVaryings> for BackgroundFragmentShader {
    type Uniforms = BackgroundFragmentUniforms;

    fn shade(&self, varyings: BackgroundVaryings, uniforms: &Self::Uniforms) -> Colour {
        let uv = varyings.uv;

        let sky = Colour::lerp(
            &Colour::new(0.01, 0.025, 0.08, 1.0),
            &Colour::new(0.08, 0.25, 0.65, 1.0),
            uv.y,
        );

        // Animated procedural clouds.
        let t = uniforms.time * 0.03;

        let n1 = ((uv.x * 7.0 + t).sin() + (uv.y * 5.0 - t).cos()) * 0.5;
        let n2 = ((uv.x * 15.0 - t * 1.3).sin() * (uv.y * 12.0 + t).cos()) * 0.25;

        let cloud = (n1 + n2 + 0.5).clamp(0.0, 1.0);

        let cloud_colour = Colour::new(0.7, 0.8, 1.0, 1.0);

        let mut colour = Colour::lerp(&sky, &cloud_colour, cloud * 0.35);

        // A deliberately overbright sun.
        //
        // This is HDR, so values above 1.0 survive into the scene target
        // and will later be extracted by the bloom pass.
        let sun_position = Vec2::new(0.72, 0.8);
        let distance = (uv - sun_position).length();

        let sun = (1.0 - distance * 8.0).max(0.0).powf(4.0);

        colour += Colour::new(5.0, 3.5, 1.5, 1.0) * sun;

        colour
    }
}

// -----------------------------------------------------------------------------
// Fullscreen post-processing geometry
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

// -----------------------------------------------------------------------------
// Bright-pass extraction
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct BrightPassUniforms {
    source: Arc<TextureSampler>,
    threshold: f32,
}

struct BrightPassFragmentShader;

impl FragmentShader<PostProcessVaryings> for BrightPassFragmentShader {
    type Uniforms = BrightPassUniforms;

    fn shade(&self, varyings: PostProcessVaryings, uniforms: &Self::Uniforms) -> Colour {
        let colour = uniforms.source.sample_linear_clamp(varyings.uv);

        if colour.luminance() > uniforms.threshold {
            colour
        } else {
            Colour::BLACK
        }
    }
}

// -----------------------------------------------------------------------------
// Gaussian blur
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct BlurUniforms {
    source: Arc<TextureSampler>,
}

struct GaussianBlurFragmentShader;

impl FragmentShader<PostProcessVaryings> for GaussianBlurFragmentShader {
    type Uniforms = BlurUniforms;

    fn shade(&self, varyings: PostProcessVaryings, uniforms: &Self::Uniforms) -> Colour {
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
            1.0 / uniforms.source.width() as f32,
            1.0 / uniforms.source.height() as f32,
        );

        for (offset, weight) in offsets.iter().zip(weights.iter()) {
            let sample_uv = uv + *offset * texel_size;
            let sample_colour = uniforms.source.sample_nearest_clamp(sample_uv);

            r += sample_colour.r as f32 * weight;
            g += sample_colour.g as f32 * weight;
            b += sample_colour.b as f32 * weight;
        }

        Colour::new(r, g, b, 1.0)
    }
}

// -----------------------------------------------------------------------------
// Bloom composite + tone mapping
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct BloomCompositeUniforms {
    scene: Arc<TextureSampler>,
    bloom: Arc<TextureSampler>,
    bloom_strength: f32,
    exposure: f32,
}

struct BloomCompositeFragmentShader;

impl FragmentShader<PostProcessVaryings> for BloomCompositeFragmentShader {
    type Uniforms = BloomCompositeUniforms;

    fn shade(&self, varyings: PostProcessVaryings, uniforms: &Self::Uniforms) -> Colour {
        let scene = uniforms.scene.sample_linear_clamp(varyings.uv);

        let bloom = uniforms.bloom.sample_linear_clamp(varyings.uv) * uniforms.bloom_strength;

        let hdr = (scene + bloom) * uniforms.exposure;

        // Simple Reinhard tone mapping.
        //
        // HDR:
        //     0.5 -> 0.333
        //     1.0 -> 0.5
        //     5.0 -> 0.833
        //    10.0 -> 0.909
        let mapped = Colour::new(
            hdr.r / (1.0 + hdr.r),
            hdr.g / (1.0 + hdr.g),
            hdr.b / (1.0 + hdr.b),
            hdr.a,
        );

        // The output of tone mapping should already be in displayable range,
        // but clamp as the final safety boundary before the presentation target.
        mapped.clamp_rgb(0.0, 1.0)
    }
}

// -----------------------------------------------------------------------------
// Application
// -----------------------------------------------------------------------------

struct BloomApp {
    camera: Camera,
    camera_controller: OrbitControls,
    light: DirectionalLight,

    teapot: Model<ObjVertex>,
    cube: Model<ObjVertex>,
    fullscreen_quad: Model<ObjVertex>,

    // -------------------------------------------------------------------------
    // HDR render targets
    //
    // scene_target:
    //     scene geometry + background
    //
    // bright_target:
    //     only pixels bright enough to bloom
    //
    // blur_a / blur_b:
    //     ping-pong buffers for separable Gaussian blur
    // -------------------------------------------------------------------------
    scene_target: RenderTarget,
    bright_target: RenderTarget,
    blur_target: RenderTarget,

    background_pipeline: Pipeline<BackgroundVertexShader, BackgroundFragmentShader>,
    teapot_pipeline: Pipeline<PhongVertexShader, PhongFragmentShader>,
    glass_pipeline: Pipeline<PhongVertexShader, PhongFragmentShader>,

    bright_pipeline: Pipeline<PostProcessVertexShader, BrightPassFragmentShader>,
    blur_pipeline: Pipeline<PostProcessVertexShader, GaussianBlurFragmentShader>,
    composite_pipeline: Pipeline<PostProcessVertexShader, BloomCompositeFragmentShader>,

    elapsed: f32,
}

impl BloomApp {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let camera = Camera::new(
            Vec3::new(0.0, 2.5, 3.0),
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

        // Deliberately bright light so the scene produces HDR values.
        let light =
            DirectionalLight::new(Vec3::new(-0.4, -1.0, -0.5), Colour::new(2.5, 2.5, 2.5, 1.0));

        // ---------------------------------------------------------------------
        // Teapot
        // ---------------------------------------------------------------------

        let mut teapot = load_obj(std::path::Path::new("assets/utah_teapot.obj"))?;

        teapot.transform.scale = Vec3::ONE * 0.6;
        teapot.transform.position = Vec3::new(0.0, -0.35, 0.0);

        teapot.calculate_vertex_normals();

        let material = Material::new_simple(
            "Polished Ceramic".to_string(),
            Colour::new(0.01, 0.02, 0.04, 1.0),
            Colour::new(0.08, 0.25, 0.5, 1.0),
            Colour::WHITE,
            64.0,
        );

        teapot.materials = vec![material];
        teapot.meshes[0].material_index = Some(0);

        // ---------------------------------------------------------------------
        // Cube
        // ---------------------------------------------------------------------

        let mut cube = load_obj(std::path::Path::new("assets/cube/cube.obj"))?;

        cube.transform.scale = Vec3::ONE * 2.0;
        cube.transform.position = Vec3::new(0.0, 0.25, 0.0);

        cube.calculate_vertex_normals();

        let glass_material = Material::new_simple(
            "Glass".to_string(),
            Colour::new(0.01, 0.02, 0.04, 0.3),
            Colour::new(0.08, 0.15, 0.2, 0.3),
            Colour::new(1.0, 1.0, 1.0, 0.4),
            128.0,
        );

        cube.materials = vec![glass_material];
        for mesh in cube.meshes.iter_mut() {
            mesh.material_index = Some(0);
        }

        // ---------------------------------------------------------------------
        // Fullscreen quad
        // ---------------------------------------------------------------------

        let fullscreen_quad = load_obj(std::path::Path::new("assets/fullscreen_quad.obj"))?;

        // ---------------------------------------------------------------------
        // Render targets
        // ---------------------------------------------------------------------

        let extent = Extent::new(WIDTH, HEIGHT);

        let scene_target = RenderTarget::new(extent).with_depth();
        let bright_target = RenderTarget::new(extent);
        let blur_target = RenderTarget::new(extent);

        // ---------------------------------------------------------------------
        // Pipelines
        // ---------------------------------------------------------------------

        let background_pipeline = Pipeline::new(BackgroundVertexShader, BackgroundFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED);

        let teapot_pipeline = Pipeline::new(PhongVertexShader, PhongFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DEFAULT);

        let glass_pipeline = Pipeline::new(PhongVertexShader, PhongFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::READ_ONLY)
            .with_blend_state(BlendState::ALPHA_BLEND);

        let bright_pipeline = Pipeline::new(PostProcessVertexShader, BrightPassFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED);

        let blur_pipeline = Pipeline::new(PostProcessVertexShader, GaussianBlurFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED);

        let composite_pipeline =
            Pipeline::new(PostProcessVertexShader, BloomCompositeFragmentShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED);

        Ok(Self {
            camera,
            camera_controller,
            light,

            teapot,
            cube,
            fullscreen_quad,

            scene_target,
            bright_target,
            blur_target,

            background_pipeline,
            teapot_pipeline,
            glass_pipeline,
            bright_pipeline,
            blur_pipeline,
            composite_pipeline,

            elapsed: 0.0,
        })
    }
}

impl Application for BloomApp {
    fn update(&mut self, dt: f32) {
        self.elapsed += dt;

        self.teapot.transform.rotation.y = self.elapsed * 0.5;

        self.camera_controller
            .update_from_events(&mut self.camera, dt);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect_ratio(width as f32 / height as f32);

        let extent = Extent::new(width as usize, height as usize);

        self.scene_target.resize(extent);
        self.bright_target.resize(extent);
        self.blur_target.resize(extent);
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
    }

    fn render<'frame>(&mut self, context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        // =====================================================================
        // PASS 1: Render the HDR scene
        // =====================================================================

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
            &self.background_pipeline,
            BackgroundVertexUniforms,
            |_| BackgroundFragmentUniforms { time: self.elapsed },
        );

        pass.finish();

        // ---------------------------------------------------------------------
        // PASS 2: Render the teapot and glass over the HDR background
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

        self.cube.draw_to_render_pass(
            &mut pass,
            &self.glass_pipeline,
            PhongVertexUniforms {
                model_matrix: self.cube.transform.model_matrix(),
                scene: scene.clone(),
            },
            |mesh| {
                let material = self
                    .cube
                    .materials
                    .get(mesh.material_index.unwrap())
                    .cloned()
                    .expect("cube mesh should have a material");

                PhongFragmentUniforms {
                    scene: scene.clone(),
                    material,
                }
            },
        );

        pass.finish();

        // =====================================================================
        // Convert the HDR scene target to a texture sampler.
        // =====================================================================

        let scene_texture = Arc::new(render_target_sampler(&self.scene_target));

        // =====================================================================
        // PASS 3: Extract bright pixels
        //
        // scene_target -> bright_target
        // =====================================================================

        let extent = self.bright_target.extent();

        let mut pass = context.begin_render_pass(
            &mut self.bright_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        self.fullscreen_quad.draw_to_render_pass(
            &mut pass,
            &self.bright_pipeline,
            PostProcessVertexUniforms,
            |_| BrightPassUniforms {
                source: scene_texture.clone(),
                threshold: 1.0,
            },
        );

        pass.finish();

        // =====================================================================
        // PASS 4: Gaussian blur
        //
        // bright_target -> blur
        // =====================================================================

        let bright_texture = Arc::new(render_target_sampler(&self.bright_target));

        let extent = self.blur_target.extent();

        let mut pass = context.begin_render_pass(
            &mut self.blur_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        self.fullscreen_quad.draw_to_render_pass(
            &mut pass,
            &self.blur_pipeline,
            PostProcessVertexUniforms,
            |_| BlurUniforms {
                source: bright_texture.clone(),
            },
        );

        pass.finish();

        // =====================================================================
        // PASS 5: Composite bloom + tone mapping -> presentation target
        // =====================================================================

        let blur_target_texture = Arc::new(render_target_sampler(&self.blur_target));

        let extent = context.presentation_target().extent();

        let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: None,
        });

        self.fullscreen_quad.draw_to_render_pass(
            &mut pass,
            &self.composite_pipeline,
            PostProcessVertexUniforms,
            |_| BloomCompositeUniforms {
                scene: scene_texture.clone(),
                bloom: blur_target_texture.clone(),
                bloom_strength: 5.0,
                exposure: 3.0,
            },
        );

        pass.finish()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    MinifbEngine::new()
        .with_title("HDR Bloom")
        .with_size(WIDTH, HEIGHT)
        .with_options(minifb::WindowOptions {
            resize: true,
            ..Default::default()
        })
        .run(BloomApp::new()?)
}
