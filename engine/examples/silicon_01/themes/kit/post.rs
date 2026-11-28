//! Screen-space stylisation: passes that restyle the finished HDR scene for [`Stage::stylise`].
//!
//! A [`Chain`] runs a list of [`Effect`]s in order, each reading the previous one's output.
//! Every effect keeps the scene's alpha (the view distance of opaque surfaces) so later passes
//! can still find edges and surface orientation. Effects that emit light ([`Edges`] ink, a
//! [`Ramp`]'s stops) take linear HDR colours, as in the scene. Effects that quantise or print
//! ([`Effect::Toon`], [`Hatching`], [`Halftone`], [`Cells`]) work in the tones the viewer sees,
//! after the film curve the composite applies, so their bands fall evenly across what is
//! visible; the paper, ink and lead colours they take are as displayed, 0 to 1. Sizes are in
//! pixels of a 720-line frame and scale with the render resolution, so a look doesn't change
//! with `--scale`.
//!
//! ```ignore
//! let mut chain = Chain::new(vec![
//!     Effect::Edges(Edges::default()),
//!     Effect::Toon { levels: 4.0, softness: 0.08 },
//! ]);
//! // In `Stage::stylise`:
//! Some(chain.run(context, scene, image))
//! ```
//!
//! [`Stage::stylise`]: crate::themes::Stage::stylise

use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{ViewRef, view_ray};
use crate::simd::{
    F, aces, clamp01, cross, fract, from_colour, gt, hash2, lt, mix, mix3, mul3, palette, rgb, s,
    smoothstep, splat3, to_colour,
};
use crate::themes::Scene;
use crate::themes::kit::noise;

/// Outlines found in the scene's view distance: silhouettes, and creases where the surface's
/// slope changes. Only surfaces the scene draws have a distance, so a backdrop's own line work
/// is not outlined.
#[derive(Clone, Copy, Debug)]
pub struct Edges {
    pub ink: Vec3,
    /// Half the line's thickness, in pixels.
    pub width: f32,
    /// How readily a crease becomes a line; silhouettes always do. About 1500 draws the edges of
    /// the chip's package and die.
    pub sensitivity: f32,
    /// How much of the line is drawn, 0 to 1.
    pub opacity: f32,
    /// The colour the image fades to where `keep_image` is below 1.
    pub paper: Vec3,
    /// 1 keeps the image under the lines; 0 draws the lines on `paper` alone.
    pub keep_image: f32,
    /// How far, in pixels, the lines wander from where the depth puts them, as if drawn by hand.
    pub wobble: f32,
    /// How many times a second the wobble redraws; 0 holds it still.
    pub boil: f32,
}

impl Default for Edges {
    fn default() -> Self {
        Self {
            ink: Vec3::new(0.01, 0.01, 0.012),
            width: 1.2,
            sensitivity: 1500.0,
            opacity: 1.0,
            paper: Vec3::new(0.0, 0.0, 0.0),
            keep_image: 1.0,
            wobble: 0.0,
            boil: 0.0,
        }
    }
}

/// Strokes whose density follows the image's tone, laid along the surface's orientation where
/// the scene has one: darker tones add layers at turned angles.
#[derive(Clone, Copy, Debug)]
pub struct Hatching {
    pub ink: Vec3,
    pub paper: Vec3,
    /// The distance between strokes, in pixels.
    pub spacing: f32,
    /// A stroke's full width, in pixels, at the darkest tone.
    pub width: f32,
    /// The first layer's angle in radians; the next layers turn by a quarter and an eighth turn.
    pub angle: f32,
    /// 0 lays every stroke at the same angle; 1 turns them with the surface's orientation.
    pub follow: f32,
    /// The number of layers, 1 to 4.
    pub layers: u32,
    /// How far strokes wander from straight, in pixels, to look drawn by hand.
    pub wobble: f32,
    /// 0 draws the strokes on `paper` alone; 1 keeps the image's colour between them.
    pub keep_image: f32,
}

impl Default for Hatching {
    fn default() -> Self {
        Self {
            ink: Vec3::new(0.1, 0.1, 0.12),
            paper: Vec3::new(0.92, 0.9, 0.84),
            spacing: 5.0,
            width: 2.2,
            angle: 0.8,
            follow: 0.7,
            layers: 4,
            wobble: 0.8,
            keep_image: 0.0,
        }
    }
}

/// A print's dot screen: cyan, magenta, yellow and black dots on rotated grids, each sized by
/// that ink's share of the image under the dot's centre.
#[derive(Clone, Copy, Debug)]
pub struct Halftone {
    pub paper: Vec3,
    /// The distance between dot centres, in pixels.
    pub cell: f32,
    /// The inks' colours as they print on paper.
    pub cyan: Vec3,
    pub magenta: Vec3,
    pub yellow: Vec3,
    pub black: Vec3,
    /// How much the dots grow beyond their ink's share, as on wet paper; 1 is none.
    pub gain: f32,
}

impl Default for Halftone {
    fn default() -> Self {
        Self {
            paper: Vec3::new(1.0, 0.97, 0.88),
            cell: 7.0,
            cyan: Vec3::new(0.0, 0.55, 0.9),
            magenta: Vec3::new(0.95, 0.05, 0.5),
            yellow: Vec3::new(1.0, 0.85, 0.0),
            black: Vec3::new(0.03, 0.03, 0.05),
            gain: 1.0,
        }
    }
}

/// A cathode-ray tube's screen: barrel distortion, scanlines, a phosphor mask, colour
/// convergence error and flicker. Black beyond the curved glass.
#[derive(Clone, Copy, Debug)]
pub struct Crt {
    /// How far the glass bulges: 0 is flat, 0.1 a typical monitor.
    pub curvature: f32,
    /// How dark the gaps between scanlines are, 0 to 1.
    pub scanlines: f32,
    /// The distance between scanlines, in pixels.
    pub pitch: f32,
    /// How dark the phosphor stripes' gaps are, 0 to 1.
    pub mask: f32,
    /// How far red and blue land from green, in pixels.
    pub convergence: f32,
    /// How much the brightness wavers, 0 to 1.
    pub flicker: f32,
}

impl Default for Crt {
    fn default() -> Self {
        Self {
            curvature: 0.08,
            scanlines: 0.45,
            pitch: 3.0,
            mask: 0.25,
            convergence: 0.8,
            flicker: 0.03,
        }
    }
}

/// Paper's surface: fibres, grain and relief, multiplied over the image.
#[derive(Clone, Copy, Debug)]
pub struct Paper {
    /// A tint multiplied over the image.
    pub tint: Vec3,
    /// How strongly the grain shows, 0 to 1.
    pub grain: f32,
    /// How strongly the long fibres show, 0 to 1.
    pub fibre: f32,
    /// The grain's size, in pixels.
    pub scale: f32,
}

impl Default for Paper {
    fn default() -> Self {
        Self {
            tint: Vec3::new(1.0, 1.0, 1.0),
            grain: 0.35,
            fibre: 0.25,
            scale: 2.5,
        }
    }
}

/// Sensor or film noise, redrawn 24 times a second.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    /// Multiplies the image, 0 to 1.
    pub amount: f32,
    /// Added in black as well, where the multiplied noise can't show.
    pub floor: f32,
    /// The noise's grain size, in pixels.
    pub cell: f32,
    /// 0 for grey noise, 1 for independent noise in each channel.
    pub chroma: f32,
}

impl Default for Noise {
    fn default() -> Self {
        Self {
            amount: 0.15,
            floor: 0.01,
            cell: 1.0,
            chroma: 0.0,
        }
    }
}

/// A colour ramp from dark to light, evenly spaced stops blended linearly.
#[derive(Clone, Copy, Debug)]
pub struct Ramp {
    stops: [Vec3; Self::STOPS],
    /// Scales the luminance before it is mapped; above 1 brightens, below 1 darkens.
    pub gain: f32,
    /// 0 blends smoothly; above 0 snaps to this many flat colours, as a palette would.
    pub steps: u32,
}

impl Ramp {
    /// The most stops a ramp holds.
    pub const STOPS: usize = 8;

    /// A ramp through `stops`, darkest first.
    ///
    /// # Panics
    ///
    /// Panics if `stops` is empty or holds more than [`Ramp::STOPS`] colours.
    pub fn new(stops: &[Vec3], gain: f32, steps: u32) -> Self {
        assert!(
            (1..=Self::STOPS).contains(&stops.len()),
            "a ramp holds 1 to {} stops, not {}",
            Self::STOPS,
            stops.len()
        );
        let last = stops[stops.len() - 1];
        Self {
            stops: std::array::from_fn(|i| stops.get(i).copied().unwrap_or(last)),
            gain,
            steps,
        }
    }

    /// A thermal camera's iron palette: black, indigo, magenta, red, orange, yellow, white.
    pub fn thermal(gain: f32) -> Self {
        Self::new(
            &[
                Vec3::new(0.0, 0.0, 0.01),
                Vec3::new(0.03, 0.0, 0.2),
                Vec3::new(0.3, 0.0, 0.35),
                Vec3::new(0.75, 0.04, 0.15),
                Vec3::new(1.0, 0.3, 0.0),
                Vec3::new(1.4, 0.9, 0.05),
                Vec3::new(1.8, 1.7, 1.2),
            ],
            gain,
            0,
        )
    }

    /// An image intensifier's green phosphor.
    pub fn night_vision(gain: f32) -> Self {
        Self::new(
            &[
                Vec3::new(0.0, 0.01, 0.0),
                Vec3::new(0.02, 0.18, 0.03),
                Vec3::new(0.12, 0.7, 0.14),
                Vec3::new(0.5, 1.3, 0.5),
                Vec3::new(1.2, 1.7, 1.1),
            ],
            gain,
            0,
        )
    }
}

/// Cells of flat colour divided by lines, as in stained glass or mosaic: each cell shows the
/// image at its seed point, tinted by a hue of its own.
#[derive(Clone, Copy, Debug)]
pub struct Cells {
    /// The distance between seed points, in pixels.
    pub size: f32,
    /// How far seeds stray from a regular grid, 0 to 1.
    pub jitter: f32,
    /// The lines' width, in pixels, and colour.
    pub lead: f32,
    pub lead_colour: Vec3,
    /// How strongly each cell's own hue tints it, 0 to 1.
    pub tint: f32,
    /// Restyles only what the scene draws, leaving the backdrop (view distance beyond the
    /// scene) as it is.
    pub surfaces_only: bool,
}

impl Default for Cells {
    fn default() -> Self {
        Self {
            size: 14.0,
            jitter: 0.85,
            lead: 2.0,
            lead_colour: Vec3::new(0.01, 0.01, 0.012),
            tint: 0.5,
            surfaces_only: false,
        }
    }
}

/// One screen-space pass. See [`Chain`].
#[derive(Clone, Copy, Debug)]
pub enum Effect {
    /// Outlines from the scene's depth.
    Edges(Edges),
    /// Flat bands of tone: the luminance snaps to `levels` values, blending over `softness` of
    /// a band's width, keeping each pixel's hue.
    Toon { levels: f32, softness: f32 },
    /// Oil paint: each pixel takes the average of the calmest of the four quarter-discs around
    /// it, smoothing texture and keeping edges. `radius` is in pixels, up to 6.
    Kuwahara { radius: f32 },
    /// Pencil or engraving strokes by tone: see [`Hatching`].
    Hatching(Hatching),
    /// A print's CMYK dot screen: see [`Halftone`].
    Halftone(Halftone),
    /// Blocks of `cell` pixels, each showing its centre.
    Pixelate { cell: f32 },
    /// A cathode-ray tube's phosphor, scanlines and curvature: see [`Crt`].
    Crt(Crt),
    /// Phosphor persistence: light decays over `half_life` seconds instead of vanishing, leaving
    /// trails behind moving light. Forgets the past at a cut.
    Persistence { half_life: f32 },
    /// Leaded glass: Voronoi cells of flat colour between dark lead lines, see [`Cells`].
    Cells(Cells),
    /// Paper grain and fibre under everything drawn: see [`Paper`].
    Paper(Paper),
    /// Replaces colour with a [`Ramp`] indexed by luminance: heat maps, night vision, palettes.
    Ramp(Ramp),
    /// Film grain or sensor noise: see [`Noise`].
    Noise(Noise),
}

/// A chain of [`Effect`]s with the targets they render into.
pub struct Chain {
    effects: Vec<Effect>,
    pipeline: SimdPipeline<ScreenVertexShader, PostShader>,
    buffers: Option<Buffers>,
    last_time: f32,
}

struct Buffers {
    targets: [(RenderTarget, Texture); 2],
    history: Texture,
}

impl Buffers {
    fn new(extent: Extent) -> Self {
        let target = || {
            let target = RenderTarget::new(extent);
            let texture = Texture::from_render_target(&target);
            (target, texture)
        };
        let history = Texture::from_render_target(&RenderTarget::new(extent));
        Self {
            targets: [target(), target()],
            history,
        }
    }
}

impl Chain {
    /// A chain running `effects` in order.
    pub fn new(effects: Vec<Effect>) -> Self {
        Self {
            effects,
            pipeline: SimdPipeline::new(ScreenVertexShader, PostShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED),
            buffers: None,
            last_time: f32::INFINITY,
        }
    }

    /// Runs the effects over `image`, the HDR scene with view distance in alpha, and returns the
    /// result with the view distance still in alpha. With no effects it returns `image`.
    pub fn run(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Arc<TextureSampler> {
        let extent = scene.extent;
        if self
            .buffers
            .as_ref()
            .is_none_or(|b| b.targets[0].0.extent() != extent)
        {
            self.buffers = Some(Buffers::new(extent));
            self.last_time = f32::INFINITY;
        }
        let buffers = self.buffers.as_mut().expect("buffers were just made");
        let elapsed = scene.local_time - self.last_time;
        // A cut or a loop starts again from an empty history.
        let seconds = if (0.0..0.5).contains(&elapsed) {
            elapsed
        } else {
            f32::INFINITY
        };
        self.last_time = scene.local_time;

        let unit = extent.height as f32 / 720.0;
        let mut source = image;
        for (index, effect) in self.effects.iter().enumerate() {
            let (target, texture) = &mut buffers.targets[index % 2];
            let decay = match effect {
                Effect::Persistence { half_life } => 0.5f32.powf(seconds / half_life),
                _ => 0.0,
            };
            let uniforms = PostUniforms {
                source: source.clone(),
                history: Arc::new(buffers.history.sampler(WrapMode::Clamp, FilterMode::Linear)),
                decay,
                effect: *effect,
                view: scene.view.clone(),
                time: scene.local_time,
                unit,
            };
            source = render_full_screen(context, scene, &self.pipeline, uniforms, target, texture);
            if matches!(effect, Effect::Persistence { .. }) {
                buffers.history.update_from(target);
            }
        }
        source
    }
}

/// A theme's own full-screen pass over the finished scene, for a look the [`Effect`]s don't
/// cover. It renders into a target sized to the frame, made on first use and remade when the
/// resolution changes.
///
/// ```ignore
/// let mut blur = ScreenPass::new(BlurShader);
/// // In `Stage::stylise`:
/// Some(blur.run(context, scene, BlurUniforms { source: image }))
/// ```
pub struct ScreenPass<S: FragmentShaderSimd<ScreenVaryings>> {
    pipeline: SimdPipeline<ScreenVertexShader, S>,
    output: Option<(RenderTarget, Texture)>,
}

impl<S> ScreenPass<S>
where
    S: FragmentShaderSimd<ScreenVaryings> + Send + Sync + 'static,
    S::Uniforms: Clone,
{
    /// A pass shading every pixel with `shader`, without depth testing.
    pub fn new(shader: S) -> Self {
        Self {
            pipeline: SimdPipeline::new(ScreenVertexShader, shader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED),
            output: None,
        }
    }

    /// Shades the frame with `uniforms` and returns the result. The shader writes the scene's
    /// alpha convention itself: pass the source's view distance through to keep it.
    pub fn run(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        uniforms: S::Uniforms,
    ) -> Arc<TextureSampler> {
        let extent = scene.extent;
        if self
            .output
            .as_ref()
            .is_none_or(|(target, _)| target.extent() != extent)
        {
            let target = RenderTarget::new(extent);
            let texture = Texture::from_render_target(&target);
            self.output = Some((target, texture));
        }
        let (target, texture) = self.output.as_mut().expect("the output was just made");
        render_full_screen(context, scene, &self.pipeline, uniforms, target, texture)
    }
}

fn render_full_screen<S>(
    context: &mut RenderContext<'_>,
    scene: &Scene<'_>,
    pipeline: &SimdPipeline<ScreenVertexShader, S>,
    uniforms: S::Uniforms,
    target: &mut RenderTarget,
    texture: &mut Texture,
) -> Arc<TextureSampler>
where
    S: FragmentShaderSimd<ScreenVaryings> + Send + Sync + 'static,
    S::Uniforms: Clone,
{
    let descriptor = RenderPassDescriptor {
        viewport: Viewport::full(&target.extent()),
        colour_load_op: LoadOp::Clear(Colour::BLACK),
        depth_load_op: None,
    };
    let mut pass = context.begin_render_pass(target, descriptor);
    scene
        .quad
        .draw_to_render_pass_simd(&mut pass, pipeline, (), |_| uniforms.clone());
    pass.finish();
    texture.update_from(target);
    Arc::new(texture.sampler(WrapMode::Clamp, FilterMode::Linear))
}

#[derive(Clone)]
pub struct PostUniforms {
    source: Arc<TextureSampler>,
    history: Arc<TextureSampler>,
    decay: f32,
    effect: Effect,
    view: ViewRef,
    time: f32,
    unit: f32,
}

pub struct PostShader;

impl FragmentShaderSimd<ScreenVaryings> for PostShader {
    type Uniforms = PostUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &PostUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let c = Pixel::new(u, nx, ny);
        let (colour, alpha) = match &u.effect {
            Effect::Edges(e) => edges(&c, e),
            Effect::Toon { levels, softness } => toon(&c, *levels, *softness),
            Effect::Kuwahara { radius } => kuwahara(&c, *radius),
            Effect::Hatching(h) => hatching(&c, h),
            Effect::Halftone(h) => halftone(&c, h),
            Effect::Pixelate { cell } => pixelate(&c, *cell),
            Effect::Crt(crt) => crt_screen(&c, crt),
            Effect::Persistence { .. } => persistence(&c),
            Effect::Cells(cells) => cells_pass(&c, cells),
            Effect::Paper(p) => paper(&c, p),
            Effect::Ramp(r) => ramp(&c, r),
            Effect::Noise(n) => noise(&c, n),
        };
        to_colour(colour, alpha)
    }
}

/// The fragments being shaded, with the sampling helpers every effect uses.
struct Pixel<'a> {
    u: &'a PostUniforms,
    /// Texture coordinates, 0 to 1.
    x: F,
    y: F,
    /// Pixel coordinates from the bottom left.
    px: F,
    py: F,
    width: f32,
    height: f32,
}

impl<'a> Pixel<'a> {
    #[inline(always)]
    fn new(u: &'a PostUniforms, nx: F, ny: F) -> Self {
        let (width, height) = (u.source.width() as f32, u.source.height() as f32);
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        Self {
            u,
            x,
            y,
            px: x * s(width),
            py: y * s(height),
            width,
            height,
        }
    }

    /// The source `dx`, `dy` pixels (of a 720-line frame) away.
    #[inline(always)]
    fn tap(&self, dx: f32, dy: f32) -> ColourSimd {
        self.tap_at(self.x, self.y, dx, dy)
    }

    #[inline(always)]
    fn tap_at(&self, x: F, y: F, dx: f32, dy: f32) -> ColourSimd {
        let k = self.u.unit;
        self.u
            .source
            .sample_linear_clamp_simd([x + s(dx * k / self.width), y + s(dy * k / self.height)])
    }

    /// The source at pixel coordinates.
    #[inline(always)]
    fn at_pixel(&self, px: F, py: F) -> ColourSimd {
        self.u
            .source
            .sample_linear_clamp_simd([px * s(1.0 / self.width), py * s(1.0 / self.height)])
    }

    #[inline(always)]
    fn centre(&self) -> ColourSimd {
        self.u.source.sample_linear_clamp_simd([self.x, self.y])
    }

    /// Seconds, stepped to 24 frames a second.
    #[inline(always)]
    fn frame(&self) -> f32 {
        (self.u.time * 24.0).floor()
    }
}

const DISTANCE_CAP: f32 = 60.0;

#[inline(always)]
fn luminance(c: Vec3Simd) -> F {
    c.x * s(0.2126) + c.y * s(0.7152) + c.z * s(0.0722)
}

/// The tones the viewer sees for HDR colour `c`: the composite's film curve.
#[inline(always)]
pub fn compress(c: Vec3Simd) -> Vec3Simd {
    Vec3Simd::new(aces(c.x), aces(c.y), aces(c.z))
}

/// The HDR scene colour the composite's film curve shows as `colour`, for authoring a scene or
/// palette in the colours the viewer should see. Channels above 0.985 clamp, at about 5.
pub fn hdr_of_displayed(colour: Vec3) -> Vec3 {
    let one = |y: f32| {
        let y = y.clamp(0.0, 0.985);
        let b = 0.03 - 0.59 * y;
        let a = 2.51 - 2.43 * y;
        (-b + (b * b + 4.0 * a * 0.14 * y).sqrt()) / (2.0 * a)
    };
    Vec3::new(one(colour.x), one(colour.y), one(colour.z))
}

/// The HDR colour the film curve shows as `t`, up to about 5: the inverse of [`compress`].
#[inline(always)]
pub fn expand(t: Vec3Simd) -> Vec3Simd {
    let one = |y: F| {
        let y = y.fast_max(s(0.0)).fast_min(s(0.985));
        let b = s(0.03) - s(0.59) * y;
        let a = s(2.51) - s(2.43) * y;
        (-b + (b * b + s(4.0) * a * s(0.14) * y).sqrt()) / (s(2.0) * a)
    };
    Vec3Simd::new(one(t.x), one(t.y), one(t.z))
}

fn edges(c: &Pixel, e: &Edges) -> (Vec3Simd, F) {
    let image = c.centre();
    let w = e.width;
    let (ox, oy) = if e.wobble > 0.0 {
        let tick = if e.boil > 0.0 {
            (c.u.time * e.boil).floor() * 13.7
        } else {
            0.0
        };
        let (nx, ny) = (c.px * s(0.045) + s(tick), c.py * s(0.045) - s(tick));
        let amount = s(e.wobble * c.u.unit * 2.0);
        (
            (noise::value(nx, ny) - s(0.5)) * amount,
            (noise::value(ny + s(31.0), nx) - s(0.5)) * amount,
        )
    } else {
        (s(0.0), s(0.0))
    };
    // Kept far enough inside the frame that no tap is clamped: a clamped tap turns the second
    // difference into a first one, which draws a false line along the border.
    let reach = (w + e.wobble) * c.u.unit + 0.5;
    let inside = |p: F, size: f32| p.fast_max(s(reach / size)).fast_min(s(1.0 - reach / size));
    let (x, y) = (inside(c.x, c.width), inside(c.y, c.height));
    let at = |dx: f32, dy: f32| {
        let k = c.u.unit;
        c.u.source
            .sample_linear_clamp_simd([
                x + (ox + s(dx * k)) * s(1.0 / c.width),
                y + (oy + s(dy * k)) * s(1.0 / c.height),
            ])
            .a
            .fast_min(s(DISTANCE_CAP))
    };
    let d = at(0.0, 0.0);
    let (l, r) = (at(-w, 0.0), at(w, 0.0));
    let (u, b) = (at(0.0, w), at(0.0, -w));
    let curvature = ((l + r - d - d).abs() + (u + b - d - d).abs()) / d;
    let line = smoothstep(0.5, 1.0, curvature * s(e.sensitivity)) * s(e.opacity);
    let base = mix3(splat3(e.paper), from_colour(image), s(e.keep_image));
    (mix3(base, splat3(e.ink), line), image.a)
}

/// The surface's normal in camera space (x right, y up, z towards the viewer), rebuilt from the
/// view distance of the pixel and its neighbours, using whichever side of each axis lies on the
/// same surface.
#[inline(always)]
fn view_normal(c: &Pixel) -> Vec3Simd {
    let view = &c.u.view;
    let position = |dx: f32, dy: f32| {
        let k = c.u.unit;
        let x = c.x + s(dx * k / c.width);
        let y = c.y + s(dy * k / c.height);
        let d =
            c.u.source
                .sample_linear_clamp_simd([x, y])
                .a
                .fast_min(s(DISTANCE_CAP));
        (
            view_ray(view, x * s(2.0) - s(1.0), y * s(2.0) - s(1.0)) * d,
            d,
        )
    };
    let (centre, dc) = position(0.0, 0.0);
    let (left, dl) = position(-1.0, 0.0);
    let (right, dr) = position(1.0, 0.0);
    let (up, du) = position(0.0, 1.0);
    let (down, dd) = position(0.0, -1.0);
    let along_x = crate::simd::select3(
        lt((dr - dc).abs(), (dc - dl).abs()),
        right - centre,
        centre - left,
    );
    let along_y = crate::simd::select3(
        lt((du - dc).abs(), (dc - dd).abs()),
        up - centre,
        centre - down,
    );
    let n = cross(along_x, along_y).normalise();
    Vec3Simd::new(
        n.dot(splat3(view.right)),
        n.dot(splat3(view.up)),
        -n.dot(splat3(view.forward)),
    )
}

fn toon(c: &Pixel, levels: f32, softness: f32) -> (Vec3Simd, F) {
    let image = c.centre();
    let colour = from_colour(image);
    let tone = luminance(compress(colour));
    let bands = tone * s(levels);
    let step = smoothstep(0.5 - softness, 0.5 + softness, fract(bands));
    let snapped = ((bands.floor() + step) * s(1.0 / levels)).fast_max(s(0.002));
    let wanted = compress(colour) * (snapped / tone.fast_max(s(0.002)));
    (expand(wanted), image.a)
}

fn kuwahara(c: &Pixel, radius: f32) -> (Vec3Simd, F) {
    let image = c.centre();
    let reach = radius.min(6.0);
    let mut total = Vec3Simd::new(s(0.0), s(0.0), s(0.0));
    let mut weights = s(0.0);
    for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let mut mean = Vec3Simd::new(s(0.0), s(0.0), s(0.0));
        let mut mean_luma = s(0.0);
        let mut square = s(0.0);
        for i in 0..3 {
            for j in 0..3 {
                let (dx, dy) = (sx * reach * i as f32 * 0.5, sy * reach * j as f32 * 0.5);
                let sample = compress(from_colour(c.tap(dx, dy)));
                let luma = luminance(sample);
                mean = mean + sample;
                mean_luma += luma;
                square += luma * luma;
            }
        }
        let n = s(1.0 / 9.0);
        let variance = (square * n - mean_luma * n * mean_luma * n).fast_max(s(0.0));
        let scaled = variance * s(1000.0);
        let weight = (s(1.0) + scaled * scaled * scaled * scaled).recip();
        total = total + mean * (n * weight);
        weights += weight;
    }
    (expand(total * weights.recip()), image.a)
}

fn hatching(c: &Pixel, h: &Hatching) -> (Vec3Simd, F) {
    let image = c.centre();
    let tone = luminance(compress(from_colour(image)));
    let darkness = s(1.0) - tone;
    let n = view_normal(c);
    let facing = (n.x * n.x + n.y * n.y).sqrt();
    let turn = n.y.atan2(n.x) * (smoothstep(0.05, 0.5, facing) * s(h.follow));
    let unit = c.u.unit;
    let spacing = h.spacing * unit;
    let mut ink = s(0.0);
    let layers = h.layers.clamp(1, 4);
    for layer in 0..layers {
        let angle = turn + s(h.angle + [0.0, PI * 0.5, PI * 0.25, PI * 0.75][layer as usize]);
        let (sin, cos) = (angle.sin(), angle.cos());
        let wobble = (noise::value(c.px * s(0.02), c.py * s(0.02) + s(layer as f32 * 7.0))
            - s(0.5))
            * s(h.wobble * 2.0 * unit);
        let across = (c.px * cos + c.py * sin + wobble) * s(1.0 / spacing);
        let line = across.floor();
        let jitter = hash2(line + s(layer as f32 * 31.0), s(1.7));
        let distance = (across - line - s(0.5)).abs() * s(spacing);
        // Layer `k` begins at a darker tone than the one before and thickens as the tone drops.
        let start = 0.18 + 0.2 * layer as f32;
        let depth = clamp01((darkness - s(start)) * s(3.0)) * (s(0.75) + jitter * s(0.25));
        let half = depth * s(h.width * 0.5 * unit);
        ink = ink + clamp01(half - distance + s(0.5)) * (s(1.0) - ink);
    }
    let base = mix3(expand(splat3(h.paper)), from_colour(image), s(h.keep_image));
    (mix3(base, expand(splat3(h.ink)), ink), image.a)
}

fn halftone(c: &Pixel, h: &Halftone) -> (Vec3Simd, F) {
    let alpha = c.centre().a;
    let cell = h.cell * c.u.unit;
    let mut print = splat3(h.paper);
    // Cyan, magenta, yellow and black at the angles that avoid moiré.
    for (channel, angle) in [(0, 15.0f32), (1, 75.0), (2, 0.0), (3, 45.0)] {
        let (sin, cos) = (angle.to_radians().sin(), angle.to_radians().cos());
        let (u, v) = (c.px * s(cos) + c.py * s(sin), c.py * s(cos) - c.px * s(sin));
        let (cu, cv) = (
            (u * s(1.0 / cell)).floor() * s(cell) + s(cell * 0.5),
            (v * s(1.0 / cell)).floor() * s(cell) + s(cell * 0.5),
        );
        let centre = c.at_pixel(cu * s(cos) - cv * s(sin), cu * s(sin) + cv * s(cos));
        let rgb = compress(from_colour(centre));
        let black = s(1.0) - rgb.x.fast_max(rgb.y).fast_max(rgb.z);
        let ink_of = |channel_value: F| {
            let k = s(1.0) - black;
            clamp01((s(1.0) - channel_value - black) / k.fast_max(s(0.001)))
        };
        let amount = match channel {
            0 => ink_of(rgb.x),
            1 => ink_of(rgb.y),
            2 => ink_of(rgb.z),
            _ => black,
        };
        let radius = (amount * s(h.gain)).sqrt() * s(cell * 0.72);
        let distance = ((u - cu) * (u - cu) + (v - cv) * (v - cv)).sqrt();
        let coverage = clamp01(radius - distance + s(0.5));
        let ink = [h.cyan, h.magenta, h.yellow, h.black][channel];
        let transmit = Vec3Simd::new(
            s(1.0) - coverage * s(1.0 - ink.x),
            s(1.0) - coverage * s(1.0 - ink.y),
            s(1.0) - coverage * s(1.0 - ink.z),
        );
        print = mul3(print, transmit);
    }
    (expand(print), alpha)
}

fn pixelate(c: &Pixel, cell: f32) -> (Vec3Simd, F) {
    let cell = cell * c.u.unit;
    let block = |p: F| ((p * s(1.0 / cell)).floor() + s(0.5)) * s(cell);
    let image = c.at_pixel(block(c.px), block(c.py));
    (from_colour(image), image.a)
}

fn crt_screen(c: &Pixel, crt: &Crt) -> (Vec3Simd, F) {
    let unit = c.u.unit;
    let (cx, cy) = (c.x * s(2.0) - s(1.0), c.y * s(2.0) - s(1.0));
    let k = s(crt.curvature);
    let (wx, wy) = (cx * (s(1.0) + k * cy * cy), cy * (s(1.0) + k * cx * cx));
    let (x, y) = (wx * s(0.5) + s(0.5), wy * s(0.5) + s(0.5));
    let shift = crt.convergence * unit;
    let green = c.tap_at(x, y, 0.0, 0.0);
    let red = c.tap_at(x, y, -shift, 0.0).r;
    let blue = c.tap_at(x, y, shift, 0.0).b;
    let mut colour = Vec3Simd::new(red, green.g, blue);

    let pitch = crt.pitch * unit;
    let beam = s(0.5) + s(0.5) * (c.py * s(TAU / pitch)).cos();
    let bright = clamp01(luminance(compress(colour)) * s(1.5));
    let gap = (s(1.0) - beam) * (s(1.0) - bright * s(0.6));
    colour = colour * (s(1.0) - gap * s(crt.scanlines));

    let stripe = (c.px * s(1.0 / unit.max(1.0))).floor();
    let phase = stripe - (stripe * s(1.0 / 3.0)).floor() * s(3.0);
    let lit = |channel: f32| s(1.0) - s(crt.mask) * (phase - s(channel)).abs().fast_min(s(1.0));
    colour = mul3(colour, Vec3Simd::new(lit(0.0), lit(1.0), lit(2.0)));

    let t = c.u.time;
    let wave = ((c.y * s(2.0) + s(t * 0.15)) * s(TAU)).sin();
    let flicker =
        1.0 + crt.flicker * ((hash2(s((t * 24.0).floor()), s(3.0)) - s(0.5)).to_array()[0] * 2.0);
    colour = colour * (s(flicker) * (s(1.0) + wave * s(crt.flicker * 1.5)));

    let edge = s(1.0) - gt(wx.abs(), s(1.0)).select(s(1.0), s(0.0));
    let edge_y = s(1.0) - gt(wy.abs(), s(1.0)).select(s(1.0), s(0.0));
    let glass = edge * edge_y;
    (colour * glass, green.a)
}

fn persistence(c: &Pixel) -> (Vec3Simd, F) {
    let image = c.centre();
    let past = from_colour(c.u.history.sample_linear_clamp_simd([c.x, c.y])) * s(c.u.decay);
    let now = from_colour(image);
    let kept = Vec3Simd::new(
        now.x.fast_max(past.x),
        now.y.fast_max(past.y),
        now.z.fast_max(past.z),
    );
    (kept, image.a)
}

/// The Voronoi cell a point falls in, for seeds jittered about a unit grid.
pub struct Voronoi {
    /// Distances to the nearest and second-nearest seed, in grid units.
    pub nearest: F,
    pub second: F,
    /// The nearest seed's position, and a random number in [0, 1) that identifies its cell.
    pub seed: [F; 2],
    pub id: F,
}

/// The cell of the Voronoi diagram of one seed per unit grid square, strayed from the square's
/// centre by up to `jitter / 2`, that holds (`x`, `y`).
#[inline(always)]
pub fn voronoi(x: F, y: F, jitter: f32) -> Voronoi {
    let (cx, cy) = (x.floor(), y.floor());
    let mut cell = Voronoi {
        nearest: s(1e9),
        second: s(1e9),
        seed: [s(0.0), s(0.0)],
        id: s(0.0),
    };
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (ix, iy) = (cx + s(dx as f32), cy + s(dy as f32));
            let jx = hash2(ix, iy) - s(0.5);
            let jy = hash2(ix + s(57.0), iy + s(13.0)) - s(0.5);
            let (sx, sy) = (ix + s(0.5) + jx * s(jitter), iy + s(0.5) + jy * s(jitter));
            let d = ((x - sx) * (x - sx) + (y - sy) * (y - sy)).sqrt();
            let closer = lt(d, cell.nearest);
            cell.second = closer.select(cell.nearest, cell.second.fast_min(d));
            cell.seed = [
                closer.select(sx, cell.seed[0]),
                closer.select(sy, cell.seed[1]),
            ];
            cell.id = closer.select(hash2(ix + s(3.0), iy + s(91.0)), cell.id);
            cell.nearest = cell.nearest.fast_min(d);
        }
    }
    cell
}

fn cells_pass(c: &Pixel, cells: &Cells) -> (Vec3Simd, F) {
    let image = c.centre();
    let size = cells.size * c.u.unit;
    let cell = voronoi(c.px * s(1.0 / size), c.py * s(1.0 / size), cells.jitter);
    let seed = c.at_pixel(cell.seed[0] * s(size), cell.seed[1] * s(size));
    let hue = palette(cell.id, Vec3::new(0.0, 0.33, 0.67)) * s(0.9) + rgb(0.1, 0.1, 0.1);
    let glass = mix3(
        from_colour(seed),
        mul3(from_colour(seed), hue * s(2.2)),
        s(cells.tint),
    );
    let line = clamp01(
        s(cells.lead * c.u.unit * 0.5) - (cell.second - cell.nearest) * s(0.5 * size) + s(0.5),
    );
    let cell = mix3(glass, expand(splat3(cells.lead_colour)), line);
    let shown = if cells.surfaces_only {
        let covered = lt(image.a, s(DISTANCE_CAP));
        crate::simd::select3(covered, cell, from_colour(image))
    } else {
        cell
    };
    (shown, image.a)
}

fn paper(c: &Pixel, p: &Paper) -> (Vec3Simd, F) {
    let image = c.centre();
    let scale = 1.0 / (p.scale * c.u.unit);
    let (x, y) = (c.px * s(scale), c.py * s(scale));
    let grain = noise::fbm(x, y, 4) - s(0.5);
    let fibres = noise::value(x * s(0.08) + y * s(0.03), y * s(1.6)) - s(0.5);
    let sheet = noise::fbm(c.px * s(0.01), c.py * s(0.01), 4) - s(0.5);
    let light = grain * s(p.grain * 1.6) + fibres * s(p.fibre) + sheet * s(0.12);
    let colour = mul3(from_colour(image), splat3(p.tint)) * (s(1.0) + light);
    (colour, image.a)
}

fn ramp(c: &Pixel, r: &Ramp) -> (Vec3Simd, F) {
    let image = c.centre();
    let heat = s(1.0) - (-luminance(from_colour(image)) * s(r.gain)).exp();
    let mut t = heat;
    if r.steps > 0 {
        let steps = r.steps as f32;
        t = (t * s(steps)).floor().fast_min(s(steps - 1.0)) * s(1.0 / (steps - 1.0).max(1.0));
    }
    let position = t * s((Ramp::STOPS - 1) as f32);
    let mut colour = Vec3Simd::new(s(0.0), s(0.0), s(0.0));
    for (i, stop) in r.stops.iter().enumerate() {
        let weight = clamp01(s(1.0) - (position - s(i as f32)).abs());
        colour = colour + splat3(*stop) * weight;
    }
    (colour, image.a)
}

fn noise(c: &Pixel, n: &Noise) -> (Vec3Simd, F) {
    let image = c.centre();
    let cell = n.cell * c.u.unit;
    let (gx, gy) = (
        (c.px * s(1.0 / cell)).floor(),
        (c.py * s(1.0 / cell)).floor(),
    );
    let seed = s(c.frame() * 7.13 % 977.0);
    let grey = hash2(gx + seed, gy) - s(0.5);
    let channel = |salt: f32| {
        let independent = hash2(gx + seed + s(salt), gy + s(salt * 3.1)) - s(0.5);
        mix(grey, independent, s(n.chroma))
    };
    let colour = from_colour(image);
    let amount = s(n.amount * 2.0);
    let floor = s(n.floor * 2.0);
    let (r, g, b) = (channel(0.0), channel(53.0), channel(97.0));
    (
        Vec3Simd::new(
            colour.x * (s(1.0) + r * amount) + r * floor,
            colour.y * (s(1.0) + g * amount) + g * floor,
            colour.z * (s(1.0) + b * amount) + b * floor,
        ),
        image.a,
    )
}
