use std::sync::Arc;

use cpu_rasteriser::{prelude::*, wide::f32x8};
use engine::prelude::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[inline(always)]
fn reflect_simd(vector: Vec3Simd, normal: Vec3Simd) -> Vec3Simd {
    vector - normal * (f32x8::splat(2.0) * vector.dot(normal))
}

#[inline(always)]
fn colour_splat(colour: Colour) -> ColourSimd {
    ColourSimd {
        r: f32x8::splat(colour.r),
        g: f32x8::splat(colour.g),
        b: f32x8::splat(colour.b),
        a: f32x8::splat(colour.a),
    }
}

#[inline(always)]
fn colour_lerp(a: ColourSimd, b: ColourSimd, t: f32x8) -> ColourSimd {
    ColourSimd {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

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

#[derive(Interpolate, SimdInterpolate)]
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

impl FragmentShaderSimd<PhongVaryings> for PhongFragmentShader {
    type Uniforms = PhongFragmentUniforms;

    #[inline(always)]
    fn shade_simd(&self, varyings: PhongVaryingsSimd, uniforms: &Self::Uniforms) -> ColourSimd {
        let normal =
            Vec3Simd::new(varyings.normal[0], varyings.normal[1], varyings.normal[2]).normalise();

        let eye = uniforms.scene.camera.eye;

        let view_dir = Vec3Simd::new(
            f32x8::splat(eye.x) - varyings.world_position[0],
            f32x8::splat(eye.y) - varyings.world_position[1],
            f32x8::splat(eye.z) - varyings.world_position[2],
        )
        .normalise();

        let direction = uniforms.scene.light.direction;

        let light_dir = Vec3Simd::new(
            f32x8::splat(-direction.x),
            f32x8::splat(-direction.y),
            f32x8::splat(-direction.z),
        )
        .normalise();

        let mut r = f32x8::splat(uniforms.material.ambient.r);
        let mut g = f32x8::splat(uniforms.material.ambient.g);
        let mut b = f32x8::splat(uniforms.material.ambient.b);

        let light_colour = uniforms.scene.light.colour;

        let diffuse_strength = normal.dot(light_dir).fast_max(f32x8::splat(0.0));

        r += f32x8::splat(uniforms.material.diffuse.r)
            * f32x8::splat(light_colour.r)
            * diffuse_strength;

        g += f32x8::splat(uniforms.material.diffuse.g)
            * f32x8::splat(light_colour.g)
            * diffuse_strength;

        b += f32x8::splat(uniforms.material.diffuse.b)
            * f32x8::splat(light_colour.b)
            * diffuse_strength;

        let reflect_dir = reflect_simd(-light_dir, normal);

        let specular_strength = view_dir
            .dot(reflect_dir)
            .fast_max(f32x8::splat(0.0))
            .powf_simd(f32x8::splat(uniforms.material.shininess));

        let specular_scale = specular_strength * f32x8::splat(4.0);

        r += f32x8::splat(uniforms.material.specular.r)
            * f32x8::splat(light_colour.r)
            * specular_scale;

        g += f32x8::splat(uniforms.material.specular.g)
            * f32x8::splat(light_colour.g)
            * specular_scale;

        b += f32x8::splat(uniforms.material.specular.b)
            * f32x8::splat(light_colour.b)
            * specular_scale;

        ColourSimd {
            r,
            g,
            b,
            a: f32x8::splat(1.0),
        }
    }
}

// -----------------------------------------------------------------------------
// Background
// -----------------------------------------------------------------------------

#[derive(Interpolate, SimdInterpolate)]
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

impl FragmentShaderSimd<BackgroundVaryings> for BackgroundFragmentShader {
    type Uniforms = BackgroundFragmentUniforms;

    #[inline(always)]
    fn shade_simd(
        &self,
        varyings: BackgroundVaryingsSimd,
        uniforms: &Self::Uniforms,
    ) -> ColourSimd {
        let uv = varyings.uv;

        let low = colour_splat(Colour::new(0.01, 0.025, 0.08, 1.0));

        let high = colour_splat(Colour::new(0.08, 0.25, 0.65, 1.0));

        let cloud_colour = colour_splat(Colour::new(0.7, 0.8, 1.0, 1.0));

        let t = f32x8::splat(uniforms.time * 0.03);

        let n1 = ((uv[0] * f32x8::splat(7.0) + t).sin() + (uv[1] * f32x8::splat(5.0) - t).cos())
            * f32x8::splat(0.5);

        let n2 = ((uv[0] * f32x8::splat(15.0) - t * f32x8::splat(1.3)).sin()
            * (uv[1] * f32x8::splat(12.0) + t).cos())
            * f32x8::splat(0.25);

        let cloud = (n1 + n2 + f32x8::splat(0.5))
            .fast_max(f32x8::splat(0.0))
            .fast_min(f32x8::splat(1.0));

        let mut colour = colour_lerp(low, high, uv[1]);

        colour = colour_lerp(colour, cloud_colour, cloud * f32x8::splat(0.35));

        let dx = uv[0] - f32x8::splat(0.72);
        let dy = uv[1] - f32x8::splat(0.8);

        let distance = (dx * dx + dy * dy).sqrt();

        let sun = (f32x8::splat(1.0) - distance * f32x8::splat(8.0)).fast_max(f32x8::splat(0.0));

        let sun = sun.powf_simd(f32x8::splat(4.0));

        colour.r += f32x8::splat(5.0) * sun;
        colour.g += f32x8::splat(3.5) * sun;
        colour.b += f32x8::splat(1.5) * sun;

        colour
    }
}

// -----------------------------------------------------------------------------
// Fullscreen post-processing geometry
// -----------------------------------------------------------------------------

#[derive(Interpolate, SimdInterpolate)]
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

impl FragmentShaderSimd<PostProcessVaryings> for BrightPassFragmentShader {
    type Uniforms = BrightPassUniforms;

    #[inline(always)]
    fn shade_simd(
        &self,
        varyings: PostProcessVaryingsSimd,
        uniforms: &Self::Uniforms,
    ) -> ColourSimd {
        let colour = uniforms.source.sample_linear_clamp_simd(varyings.uv);

        let brightness = colour.r * f32x8::splat(0.2126)
            + colour.g * f32x8::splat(0.7152)
            + colour.b * f32x8::splat(0.0722);

        let knee = f32x8::splat(0.5);

        let contribution = ((brightness - f32x8::splat(uniforms.threshold)) / knee)
            .fast_max(f32x8::splat(0.0))
            .fast_min(f32x8::splat(1.0));

        ColourSimd {
            r: colour.r * contribution,
            g: colour.g * contribution,
            b: colour.b * contribution,
            a: colour.a * contribution,
        }
    }
}

// -----------------------------------------------------------------------------
// Gaussian blur
// -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum BloomScale {
    Downsample,
    Upsample,
}

#[derive(Clone)]
struct BloomScaleUniforms {
    source: Arc<TextureSampler>,
    scale: BloomScale,
}

struct BloomScaleFragmentShader;

impl FragmentShaderSimd<PostProcessVaryings> for BloomScaleFragmentShader {
    type Uniforms = BloomScaleUniforms;

    #[inline(always)]
    fn shade_simd(
        &self,
        varyings: PostProcessVaryingsSimd,
        uniforms: &Self::Uniforms,
    ) -> ColourSimd {
        match uniforms.scale {
            BloomScale::Downsample => {
                let texel_x = f32x8::splat(1.0 / uniforms.source.width() as f32);

                let texel_y = f32x8::splat(1.0 / uniforms.source.height() as f32);

                let offsets = [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)];

                let mut colour = ColourSimd::splat(Colour::BLACK);

                for (ox, oy) in offsets {
                    let sample_uv = [
                        varyings.uv[0] + texel_x * f32x8::splat(ox),
                        varyings.uv[1] + texel_y * f32x8::splat(oy),
                    ];

                    let sample = uniforms.source.sample_linear_clamp_simd(sample_uv);

                    colour.r += sample.r;
                    colour.g += sample.g;
                    colour.b += sample.b;
                    colour.a += sample.a;
                }

                let quarter = f32x8::splat(0.25);

                ColourSimd {
                    r: colour.r * quarter,
                    g: colour.g * quarter,
                    b: colour.b * quarter,
                    a: colour.a * quarter,
                }
            }

            BloomScale::Upsample => uniforms.source.sample_linear_clamp_simd(varyings.uv),
        }
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

impl FragmentShaderSimd<PostProcessVaryings> for BloomCompositeFragmentShader {
    type Uniforms = BloomCompositeUniforms;

    #[inline(always)]
    fn shade_simd(
        &self,
        varyings: PostProcessVaryingsSimd,
        uniforms: &Self::Uniforms,
    ) -> ColourSimd {
        let scene = uniforms.scene.sample_linear_clamp_simd(varyings.uv);

        let bloom = uniforms.bloom.sample_linear_clamp_simd(varyings.uv);

        let bloom_strength = f32x8::splat(uniforms.bloom_strength);

        let exposure = f32x8::splat(uniforms.exposure);

        let hdr = ColourSimd {
            r: (scene.r + bloom.r * bloom_strength) * exposure,

            g: (scene.g + bloom.g * bloom_strength) * exposure,

            b: (scene.b + bloom.b * bloom_strength) * exposure,

            a: scene.a,
        };

        let one = f32x8::splat(1.0);

        let mapped = ColourSimd {
            r: hdr.r / (one + hdr.r),
            g: hdr.g / (one + hdr.g),
            b: hdr.b / (one + hdr.b),
            a: hdr.a,
        };

        ColourSimd {
            r: mapped.r.fast_max(f32x8::splat(0.0)).fast_min(one),

            g: mapped.g.fast_max(f32x8::splat(0.0)).fast_min(one),

            b: mapped.b.fast_max(f32x8::splat(0.0)).fast_min(one),

            a: mapped.a,
        }
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
    // bloom_half:
    //     downsampled + blurred bright_target
    //
    // bloom_quarter:
    //     downsampled + blurred bloom_half
    // -------------------------------------------------------------------------
    scene_target: RenderTarget,
    bright_target: RenderTarget,
    bloom_half: RenderTarget,
    bloom_quarter: RenderTarget,

    background_pipeline: SimdPipeline<BackgroundVertexShader, BackgroundFragmentShader>,
    teapot_pipeline: SimdPipeline<PhongVertexShader, PhongFragmentShader>,
    glass_pipeline: SimdPipeline<PhongVertexShader, PhongFragmentShader>,

    bright_pipeline: SimdPipeline<PostProcessVertexShader, BrightPassFragmentShader>,
    bloom_scale_pipeline: SimdPipeline<PostProcessVertexShader, BloomScaleFragmentShader>,
    composite_pipeline: SimdPipeline<PostProcessVertexShader, BloomCompositeFragmentShader>,

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

        let half_extent = Extent::new(WIDTH / 2, HEIGHT / 2);
        let quarter_extent = Extent::new(WIDTH / 4, HEIGHT / 4);

        let bloom_half = RenderTarget::new(half_extent);
        let bloom_quarter = RenderTarget::new(quarter_extent);

        // ---------------------------------------------------------------------
        // Pipelines
        // ---------------------------------------------------------------------

        let background_pipeline =
            SimdPipeline::new(BackgroundVertexShader, BackgroundFragmentShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED);

        let teapot_pipeline = SimdPipeline::new(PhongVertexShader, PhongFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DEFAULT);

        let glass_pipeline = SimdPipeline::new(PhongVertexShader, PhongFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::READ_ONLY)
            .with_blend_state(BlendState::ADDITIVE);

        let bright_pipeline = SimdPipeline::new(PostProcessVertexShader, BrightPassFragmentShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED);

        let bloom_scale_pipeline =
            SimdPipeline::new(PostProcessVertexShader, BloomScaleFragmentShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED);

        let composite_pipeline =
            SimdPipeline::new(PostProcessVertexShader, BloomCompositeFragmentShader)
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
            bloom_half,
            bloom_quarter,

            background_pipeline,
            teapot_pipeline,
            glass_pipeline,
            bright_pipeline,
            bloom_scale_pipeline,
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

        self.bloom_half
            .resize(Extent::new(width as usize / 2, height as usize / 2));
        self.bloom_quarter
            .resize(Extent::new(width as usize / 4, height as usize / 4));
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

        self.fullscreen_quad.draw_to_render_pass_simd(
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
            .draw_to_render_pass_simd(&mut pass, &self.teapot_pipeline, uniforms, |mesh| {
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

        self.cube.draw_to_render_pass_simd(
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

        self.fullscreen_quad.draw_to_render_pass_simd(
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
        // PASS 4: Downsample + blur to half resolution
        //
        // bright_target 640×360 -> bloom_half 320×180
        // =====================================================================

        let bright_texture = Arc::new(render_target_sampler(&self.bright_target));

        let extent = self.bloom_half.extent();

        let mut pass = context.begin_render_pass(
            &mut self.bloom_half,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        self.fullscreen_quad.draw_to_render_pass_simd(
            &mut pass,
            &self.bloom_scale_pipeline,
            PostProcessVertexUniforms,
            |_| BloomScaleUniforms {
                source: bright_texture.clone(),
                scale: BloomScale::Downsample,
            },
        );

        pass.finish();

        // =====================================================================
        // PASS 5: Downsample + blur to quarter resolution
        //
        // bloom_half 320×180 -> bloom_quarter 160×90
        // =====================================================================

        let bloom_half_texture = Arc::new(render_target_sampler(&self.bloom_half));

        let extent = self.bloom_quarter.extent();

        let mut pass = context.begin_render_pass(
            &mut self.bloom_quarter,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        self.fullscreen_quad.draw_to_render_pass_simd(
            &mut pass,
            &self.bloom_scale_pipeline,
            PostProcessVertexUniforms,
            |_| BloomScaleUniforms {
                source: bloom_half_texture.clone(),
                scale: BloomScale::Downsample,
            },
        );

        pass.finish();

        // =====================================================================
        // PASS 6: Upsample quarter -> half
        //
        // bloom_quarter 160×90 -> bloom_half 320×180
        // =====================================================================

        let bloom_quarter_texture = Arc::new(render_target_sampler(&self.bloom_quarter));

        let extent = self.bloom_half.extent();

        let mut pass = context.begin_render_pass(
            &mut self.bloom_half,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        self.fullscreen_quad.draw_to_render_pass_simd(
            &mut pass,
            &self.bloom_scale_pipeline,
            PostProcessVertexUniforms,
            |_| BloomScaleUniforms {
                source: bloom_quarter_texture.clone(),
                scale: BloomScale::Upsample,
            },
        );

        pass.finish();

        // =====================================================================
        // PASS 7: Upsample + composite bloom + tone mapping
        //
        // bloom_half 320×180 + scene 640×360 -> presentation 640×360
        // =====================================================================

        let bloom_half_texture = Arc::new(render_target_sampler(&self.bloom_half));

        let extent = context.presentation_target().extent();

        let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: None,
        });

        self.fullscreen_quad.draw_to_render_pass_simd(
            &mut pass,
            &self.composite_pipeline,
            PostProcessVertexUniforms,
            |_| BloomCompositeUniforms {
                scene: scene_texture.clone(),
                bloom: bloom_half_texture.clone(),
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
