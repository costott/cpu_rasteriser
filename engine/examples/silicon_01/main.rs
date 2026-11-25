//! SILICON 01: FIRST LIGHT.
//!
//! A looping product film rendered entirely on the CPU. The screen powers on tile by tile over
//! the face of a chip; the chip is revealed and turns under its captions; it projects the Utah
//! teapot out of streaming triangles, then a fractal hologram; it erupts, and passes through
//! world after world, faster and faster, until the tiles shut down.
//!
//! Space pauses the director and lets you orbit (arrow keys or WASD); Left and Right skip
//! between phases; Escape quits. `options` describes the command-line flags, including frame
//! capture for rendering the film as a video.

mod chip;
mod director;
mod effects;
mod meshes;
mod options;
mod shaders;
mod simd;
mod sound;
mod studio;
mod themes;
mod typography;

use std::path::Path;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Chip;
use crate::director::{CAPTIONS, Caption, CaptionKind, LOOP_LENGTH, Phase};
use crate::shaders::composite::{CompositeShader, CompositeUniforms};
use crate::shaders::screen::{
    BrightShader, BrightUniforms, Resample, ResampleShader, ResampleUniforms, ScreenVertexShader,
};
use crate::shaders::{SKY_DISTANCE, View, ViewRef};
use crate::studio::Studio;
use crate::themes::{Scene, Schedule, Stage, Theme};
use crate::typography::{Finish, Line, Typesetter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = options::Options::from_args();
    if let Some((path, span)) = &options.soundtrack {
        return sound::write(path, &themes::all()?, span.clone());
    }
    let (width, height) = options.window;
    let app = SiliconApp::new(options)?;
    MinifbEngine::new()
        .with_title("SILICON 01: FIRST LIGHT")
        .with_size(width, height)
        .with_options(minifb::WindowOptions {
            resize: true,
            ..Default::default()
        })
        .run(app)?;
    Ok(())
}

struct SiliconApp {
    chip: Chip,
    studio: Studio,
    themes: Vec<Theme>,
    schedule: Schedule,
    screen_quad: Model<ObjVertex>,
    pipelines: Pipelines,
    targets: Targets,

    time: f32,
    paused: bool,
    camera: Camera,
    orbit: OrbitControls,
    window: (usize, usize),
    capture: options::Capture,
    captions: bool,
}

impl SiliconApp {
    fn new(options: options::Options) -> Result<Self, Box<dyn std::error::Error>> {
        let (width, height) = options.window;
        let mut themes = themes::all()?;
        if let Some(name) = &options.theme {
            themes.retain(|theme| theme.name == name);
            if themes.is_empty() {
                return Err(format!("there is no theme called {name:?}").into());
            }
        }
        let capture = options.capture;
        let time = capture.start_time();
        let camera = camera_at(time, width as f32 / height as f32);
        Ok(Self {
            chip: Chip::new(),
            studio: Studio::new()?,
            schedule: Schedule::new(themes.len()),
            themes,
            screen_quad: load_obj(Path::new("assets/fullscreen_quad.obj"))?,
            pipelines: Pipelines::new()?,
            targets: Targets::new(width, height, options.scale),
            time,
            paused: false,
            orbit: OrbitControls::new(&camera),
            camera,
            window: (width, height),
            capture,
            captions: options.captions,
        })
    }
}

fn camera_at(time: f32, aspect: f32) -> Camera {
    let shot = director::shot(time);
    Camera::new(
        shot.eye,
        shot.target,
        Vec3::new(0.0, 1.0, 0.0),
        Projection::Perspective(PerspectiveProjection::new(
            shot.fov.to_radians(),
            aspect,
            0.05,
            500.0,
        )),
    )
}

/// The camera's view of the frame, for targets `height` pixels high.
fn view_of(camera: &Camera, frame: Arc<shaders::Frame>, height: usize) -> ViewRef {
    let Projection::Perspective(perspective) = camera.projection else {
        unreachable!("the film's camera is always perspective");
    };
    let forward = (camera.lookat - camera.eye).normalise();
    let right = forward.cross(camera.up).normalise();
    let up = right.cross(forward);
    let tan_half_fov = (perspective.fov * 0.5).tan();
    Arc::new(View {
        view_projection: camera.projection_matrix() * camera.view_matrix(),
        eye: camera.eye,
        forward,
        right,
        up,
        tan_half_fov,
        aspect: perspective.aspect_ratio,
        pixel_angle: 2.0 * tan_half_fov / height as f32,
        line_angle: 2.0 * tan_half_fov / height.min(720) as f32,
        frame,
    })
}

impl Application for SiliconApp {
    fn update(&mut self, dt: f32) {
        let dt = dt.min(0.1);
        if self.paused {
            self.orbit.update_from_events(&mut self.camera, dt);
            return;
        }
        self.time = match self.capture.frame_time() {
            Some(time) => time,
            None => self.time + dt,
        }
        .rem_euclid(LOOP_LENGTH);
        let (width, height) = self.window;
        self.camera = camera_at(self.time, width as f32 / height as f32);
    }

    fn render<'frame>(&mut self, context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        let presented = self.render_frame(context);
        self.capture.save(context.presentation_target(), self.time);
        presented
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.window = (width as usize, height as usize);
        self.camera.set_aspect_ratio(width as f32 / height as f32);
        self.targets.resize(width as usize, height as usize);
    }

    fn event(&mut self, event: AppEvent, handle: &mut AppHandle) {
        if self.paused {
            self.orbit.handle_event(event);
        }
        let AppEvent::Input(InputEvent::Key {
            key,
            state: ButtonState::Pressed,
        }) = event
        else {
            return;
        };
        let phase = Phase::at(self.time);
        match key {
            InputKey::Escape => handle.request_exit(),
            InputKey::Space => {
                self.paused = !self.paused;
                self.orbit = OrbitControls::new(&self.camera);
            }
            InputKey::Right => self.time = phase.next().map_or(0.0, Phase::start),
            InputKey::Left => {
                // Back to this phase's start, or the previous phase's if we're already near it.
                let target = if self.time - phase.start() < 1.0 {
                    phase.previous().unwrap_or(Phase::Shutdown)
                } else {
                    phase
                };
                self.time = target.start();
            }
            _ => {}
        }
    }
}

impl SiliconApp {
    fn render_frame(&mut self, context: &mut RenderContext<'_>) -> PresentedFrame {
        let t = self.time;
        let targets = &mut self.targets;
        let pipes = &self.pipelines;
        let frame = Arc::new(director::frame(t));
        let view = view_of(&self.camera, frame, targets.scene.height());

        let (stage, local_time): (&mut dyn Stage, f32) = match self.schedule.at(t) {
            Some((cut, local)) => (self.themes[self.schedule.theme(cut)].stage.as_mut(), local),
            None => (&mut self.studio, t),
        };
        let scene = Scene {
            view: view.clone(),
            chip: &self.chip,
            quad: &self.screen_quad,
            local_time,
            extent: targets.scene.extent(),
        };

        stage.prepare(context, &scene);
        {
            let target = &mut targets.scene;
            let clear = Colour::new(0.0, 0.0, 0.0, SKY_DISTANCE);
            let mut pass = context.begin_render_pass(target, descriptor(target, clear, true));
            stage.draw(&mut pass, &scene);
            pass.finish();
        }
        let image = sample(
            &mut targets.scene_texture,
            &targets.scene,
            FilterMode::Linear,
        );
        let image = stage
            .stylise(context, &scene, image.clone())
            .unwrap_or(image);
        let grade = stage.grade();
        let word_ink = stage.caption_ink();
        let chip_screen = project_to_uv(&view, view.frame.chip * Vec4::new(0.0, 0.0, 0.0, 1.0));

        // Bloom: bright pass at half resolution, down to an eighth, and back up to a quarter.
        let quad = &self.screen_quad;
        let bright = {
            let target = &mut targets.bright;
            let mut pass =
                context.begin_render_pass(target, descriptor(target, Colour::BLACK, false));
            let uniforms = BrightUniforms {
                source: image.clone(),
                threshold: 0.8,
            };
            post(&mut pass, quad, &pipes.bright, uniforms);
            pass.finish();
            sample(
                &mut targets.bright_texture,
                &targets.bright,
                FilterMode::Linear,
            )
        };
        let quarter = resample(
            context,
            &mut targets.quarter,
            &mut targets.quarter_texture,
            quad,
            &pipes.resample,
            bright,
            Resample::Down,
        );
        let eighth = resample(
            context,
            &mut targets.eighth,
            &mut targets.eighth_texture,
            quad,
            &pipes.resample,
            quarter,
            Resample::Down,
        );
        let bloom = resample(
            context,
            &mut targets.bloom,
            &mut targets.bloom_texture,
            quad,
            &pipes.resample,
            eighth,
            Resample::Up,
        );

        // Composite onto the window, with the captions over it.
        let extent = context.presentation_target().extent();
        let window = (extent.width, extent.height);
        let composite = CompositeUniforms {
            scene: image,
            bloom,
            time: t,
            grade,
            flash: director::flash(t),
            blur: director::blur(t),
            brightness: 1.0,
            tiles: director::tiles(t, chip_screen),
            collapse: director::collapse(t),
        };
        let lines = if self.captions {
            captions(t, window, &self.schedule, word_ink)
        } else {
            Vec::new()
        };
        let captions = pipes.text.set(&lines, window);
        let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
            viewport: Viewport::full(&extent),
            colour_load_op: LoadOp::Clear(Colour::BLACK),
            depth_load_op: None,
        });
        post(&mut pass, quad, &pipes.composite, composite);
        pipes.text.draw(&mut pass, &captions);
        pass.finish()
    }
}

/// The captions on screen at `time`, laid out for a window of `window` pixels.
///
/// Heights and positions are fractions of the window's height. Narration sits about 78% of the way
/// down, a title centred above it with fine print just under the title, an overhead line near the
/// top, and a montage word in the lower third, under the chip, in `word_ink`.
fn captions(
    time: f32,
    window: (usize, usize),
    schedule: &Schedule,
    word_ink: Vec3,
) -> Vec<Line<'static>> {
    let ink = Vec3::new(director::INK, director::INK, director::INK);
    let (width, height) = (window.0 as f32, window.1 as f32);
    let word = schedule.at(time).and_then(|(index, _)| {
        let slot = schedule.slot(index);
        Some(Caption {
            text: director::montage_word(index)?,
            kind: CaptionKind::Word,
            start: slot.start,
            end: slot.end,
        })
    });
    CAPTIONS
        .iter()
        .copied()
        .filter(|c| (c.start..c.end).contains(&time))
        .chain(word)
        .map(|caption| {
            let duration = caption.end - caption.start;
            let age = time - caption.start;
            let (cap_height, baseline, tracking) = match caption.kind {
                // The title's letters drift apart through its hold.
                CaptionKind::Title => (0.12, 0.53, 0.04 + 0.05 * (age / duration).min(1.0)),
                CaptionKind::Narration => (0.045, 0.80, 0.015),
                CaptionKind::Overhead => (0.045, 0.17, 0.015),
                CaptionKind::FinePrint => (0.03, 0.62, 0.015),
                CaptionKind::Word => (0.09, 0.87, 0.02),
            };
            let reveal = caption.reveal(time);
            // The title's light turns slowly about it, and its sweep crosses once, as soon as
            // the title has rendered in (rates set by eye, provisional).
            let finish = match caption.kind {
                CaptionKind::Title => Finish::Metal {
                    sweep: (age - reveal.draw_time()) / 1.4 * 1.6 - 0.3,
                    light: 2.0 + age * 0.5,
                },
                _ => Finish::Flat,
            };
            Line {
                text: caption.text,
                centre: width * 0.5,
                baseline: baseline * height,
                cap_height: cap_height * height,
                tracking,
                colour: if caption.kind == CaptionKind::Word {
                    word_ink
                } else {
                    ink
                },
                reveal,
                finish,
                joints: matches!(caption.kind, CaptionKind::Title | CaptionKind::Word),
            }
        })
        .collect()
}

/// Where a clip-space-to-be point appears in texture coordinates, which may be off screen.
fn project_to_uv(view: &View, point: Vec4) -> Vec2 {
    let clip = view.view_projection * point;
    let w = if clip.w.abs() < 1e-4 { 1e-4 } else { clip.w };
    Vec2::new(clip.x / w * 0.5 + 0.5, clip.y / w * 0.5 + 0.5)
}

fn descriptor(target: &RenderTarget, clear: Colour, depth: bool) -> RenderPassDescriptor {
    RenderPassDescriptor {
        viewport: Viewport::full(&target.extent()),
        colour_load_op: LoadOp::Clear(clear),
        depth_load_op: depth.then_some(LoadOp::Clear(1.0)),
    }
}

/// Copies a finished target into its texture and returns a sampler for the next pass.
fn sample(texture: &mut Texture, target: &RenderTarget, filter: FilterMode) -> Arc<TextureSampler> {
    texture.update_from(target);
    Arc::new(texture.sampler(WrapMode::Clamp, filter))
}

fn post<'a, FS>(
    pass: &mut RenderPass<'_, 'a>,
    quad: &'a Model<ObjVertex>,
    pipeline: &'a SimdPipeline<ScreenVertexShader, FS>,
    uniforms: FS::Uniforms,
) where
    FS: FragmentShaderSimd<shaders::screen::ScreenVaryings> + Send + Sync + 'static,
    FS::Uniforms: Clone,
{
    quad.draw_to_render_pass_simd(pass, pipeline, (), |_| uniforms.clone());
}

fn resample(
    context: &mut RenderContext<'_>,
    target: &mut RenderTarget,
    texture: &mut Texture,
    quad: &Model<ObjVertex>,
    pipeline: &SimdPipeline<ScreenVertexShader, ResampleShader>,
    source: Arc<TextureSampler>,
    mode: Resample,
) -> Arc<TextureSampler> {
    {
        let mut pass = context.begin_render_pass(target, descriptor(target, Colour::BLACK, false));
        post(&mut pass, quad, pipeline, ResampleUniforms { source, mode });
        pass.finish();
    }
    sample(texture, target, FilterMode::Linear)
}

struct Pipelines {
    bright: SimdPipeline<ScreenVertexShader, BrightShader>,
    resample: SimdPipeline<ScreenVertexShader, ResampleShader>,
    composite: SimdPipeline<ScreenVertexShader, CompositeShader>,
    text: Typesetter,
}

impl Pipelines {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        fn screen<FS>(fs: FS) -> SimdPipeline<ScreenVertexShader, FS>
        where
            FS: FragmentShaderSimd<shaders::screen::ScreenVaryings>,
        {
            SimdPipeline::new(ScreenVertexShader, fs)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED)
        }
        Ok(Self {
            bright: screen(BrightShader),
            resample: screen(ResampleShader),
            composite: screen(CompositeShader),
            text: Typesetter::new()?,
        })
    }
}

/// The offscreen targets, each with the texture that the next pass samples it through.
struct Targets {
    scene: RenderTarget,
    scene_texture: Texture,
    bright: RenderTarget,
    bright_texture: Texture,
    quarter: RenderTarget,
    quarter_texture: Texture,
    eighth: RenderTarget,
    eighth_texture: Texture,
    bloom: RenderTarget,
    bloom_texture: Texture,
    /// The scene's resolution relative to the window.
    scale: f32,
}

impl Targets {
    fn new(width: usize, height: usize, scale: f32) -> Self {
        let sizes = Sizes::new(width, height, scale);
        let scene = RenderTarget::new(sizes.scene).with_depth();
        let bright = RenderTarget::new(sizes.half);
        let quarter = RenderTarget::new(sizes.quarter);
        let eighth = RenderTarget::new(sizes.eighth);
        let bloom = RenderTarget::new(sizes.quarter);
        Self {
            scene_texture: Texture::from_render_target(&scene),
            bright_texture: Texture::from_render_target(&bright),
            quarter_texture: Texture::from_render_target(&quarter),
            eighth_texture: Texture::from_render_target(&eighth),
            bloom_texture: Texture::from_render_target(&bloom),
            scale,
            scene,
            bright,
            quarter,
            eighth,
            bloom,
        }
    }

    fn resize(&mut self, width: usize, height: usize) {
        let sizes = Sizes::new(width, height, self.scale);
        self.scene.resize(sizes.scene);
        self.bright.resize(sizes.half);
        self.quarter.resize(sizes.quarter);
        self.eighth.resize(sizes.eighth);
        self.bloom.resize(sizes.quarter);
    }
}

struct Sizes {
    scene: Extent,
    half: Extent,
    quarter: Extent,
    eighth: Extent,
}

impl Sizes {
    fn new(width: usize, height: usize, scale: f32) -> Self {
        let scaled = |divisor: f32| {
            Extent::new(
                ((width as f32 * scale / divisor) as usize).max(1),
                ((height as f32 * scale / divisor) as usize).max(1),
            )
        };
        Self {
            scene: scaled(1.0),
            half: scaled(2.0),
            quarter: scaled(4.0),
            eighth: scaled(8.0),
        }
    }
}
