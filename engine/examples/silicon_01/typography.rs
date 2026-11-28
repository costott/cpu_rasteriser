//! Captions set in Inter Display and drawn as anti-aliased textured quads.
//!
//! The typeface is Inter Display Regular (The Inter Project, SIL Open Font License 1.1; see
//! `assets/silicon_01/fonts/OFL.txt`), from the rsms/inter 4.1 release. Its printable ASCII characters and a few typographic ones were baked into
//! `atlas.png`, a single-channel signed distance field, at 96 pixels to the em with a spread of
//! 12 pixels: the grey level is 0.5 on a glyph's outline and rises to 1 twelve pixels inside it.
//! `metrics.txt` holds each glyph's advance, quad and atlas rectangle, and the typeface's `kern`
//! pairs, one record per line:
//!
//! ```text
//! font <em> <spread> <atlas width> <atlas height> <cap height> <x height> <ascent> <descent>
//! glyph <code point> <advance> <left> <bottom> <right> <top> <x> <y> <width> <height>
//! kern <left code point> <right code point> <adjustment>
//! ```
//!
//! Lengths in `font`, `glyph` and `kern` records are in ems, except `<em>`, `<spread>` and the
//! atlas rectangles, which are in atlas pixels. A glyph's quad runs from `<left>` to `<right>`
//! and from `<bottom>` to `<top>`, upwards from the baseline, and includes the spread.
//!
//! A [`Typesetter`] lays [`Line`]s out in window pixels and draws them over a finished frame.
//! The fragment shader reconstructs each glyph's edge from the distance field and fades it over
//! one screen pixel, so text stays sharp at any size the atlas can be scaled to.
//!
//! A line renders in the way the screen's tiles do, letter by letter in reading order: a pen
//! traces each letter's outline in straight lines, a sweep down the letter shades it flat, and a
//! second sweep brings its ink as the lines go out. It cuts away by running the stages backwards
//! on every letter at once (see [`Reveal`]).

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::simd::{F, clamp01, mix3, rgb, s, smoothstep, to_colour, v3};
use crate::typography::outlines::{Loop, trace};
use crate::typography::strokes::{Stroke, StrokeShader, StrokeVertex, StrokeVertexShader};

mod outlines;
mod strokes;

const ATLAS: &str = "assets/silicon_01/fonts/atlas.png";
const METRICS: &str = "assets/silicon_01/fonts/metrics.txt";

/// The glyph drawn for a character the typeface lacks.
const FALLBACK: char = '?';

/// How far, in atlas pixels, a letter's traced lines may stray from its true outline: enough for
/// large letters to show slight facets (set by eye, provisional).
const TOLERANCE: f32 = 1.5;

/// How a line's time is spent (set by eye, provisional): rendering in takes this share of its
/// time on screen, up to the line's own limit; cutting away takes this many seconds, but no more
/// than this share.
const DRAW_SHARE: f32 = 0.3;
const CUT: f32 = 0.15;
const CUT_SHARE: f32 = 0.15;
/// The share of a line's draw time each letter takes; the letters' starts are spread over the
/// rest.
const LETTER_SHARE: f32 = 0.4;

/// The traced lines' half width and the half side of the squares at their joints, in pixels of a
/// 720-line window, and the length of the pen's bright tip in ems (set by eye, provisional).
const LINE: f32 = 0.7;
const JOINT: f32 = 1.7;
const TIP: f32 = 0.5;

#[derive(Clone, Copy)]
struct Glyph {
    advance: f32,
    /// Left, bottom, right and top of the quad in ems, upwards from the baseline.
    quad: [f32; 4],
    /// Left, top, right and bottom of the glyph's atlas rectangle, in atlas pixels.
    atlas: [f32; 4],
}

/// The typeface's metrics: how to space characters and where each glyph is in the atlas.
pub struct Typeface {
    glyphs: HashMap<char, Glyph>,
    kerns: HashMap<(char, char), f32>,
    cap_height: f32,
    em_pixels: f32,
    spread: f32,
    atlas_size: (f32, f32),
}

impl Typeface {
    fn parse(metrics: &str) -> Result<Self, Box<dyn Error>> {
        let mut header = None;
        let mut glyphs = HashMap::new();
        let mut kerns = HashMap::new();
        for line in metrics.lines().filter(|l| !l.starts_with('#')) {
            let mut fields = line.split_whitespace();
            let kind = fields.next();
            let numbers = fields.map(str::parse).collect::<Result<Vec<f32>, _>>()?;
            let character = |code: f32| char::from_u32(code as u32).ok_or("invalid code point");
            match (kind, numbers.as_slice()) {
                (Some("font"), &[em, spread, width, height, cap, ..]) => {
                    header = Some((em, spread, (width, height), cap));
                }
                (Some("glyph"), &[code, advance, l, b, r, t, x, y, w, h]) => {
                    let glyph = Glyph {
                        advance,
                        quad: [l, b, r, t],
                        atlas: [x, y, x + w, y + h],
                    };
                    glyphs.insert(character(code)?, glyph);
                }
                (Some("kern"), &[left, right, adjustment]) => {
                    kerns.insert((character(left)?, character(right)?), adjustment);
                }
                (None, _) => {}
                _ => return Err(format!("malformed metrics record {line:?}").into()),
            }
        }
        let (em_pixels, spread, atlas_size, cap_height) =
            header.ok_or("metrics lack a font record")?;
        if !glyphs.contains_key(&FALLBACK) {
            return Err("the typeface has no fallback glyph".into());
        }
        Ok(Self {
            glyphs,
            kerns,
            cap_height,
            em_pixels,
            spread,
            atlas_size,
        })
    }

    fn glyph(&self, c: char) -> &Glyph {
        self.glyphs.get(&c).unwrap_or(&self.glyphs[&FALLBACK])
    }

    /// A glyph's outline traced from the atlas, in ems from the pen's position on the baseline,
    /// upwards.
    fn outline(&self, glyph: &Glyph, atlas: &TextureSampler) -> Vec<Loop> {
        let [u0, v0, u1, v1] = glyph.atlas;
        let [left, bottom, right, top] = glyph.quad;
        // The level of the atlas pixel `x` across and `y` down the glyph's rectangle.
        let level = |x: usize, y: usize| {
            let uv = Vec2::new(
                (u0 + x as f32) / (self.atlas_size.0 - 1.0),
                1.0 - (v0 + y as f32) / (self.atlas_size.1 - 1.0),
            );
            atlas.sample_linear_clamp(uv).r
        };
        let in_ems = |p: Vec2| {
            Vec2::new(
                left + (p.x + 0.5) / (u1 - u0) * (right - left),
                top - (p.y + 0.5) / (v1 - v0) * (top - bottom),
            )
        };
        trace((u1 - u0) as usize, (v1 - v0) as usize, level, TOLERANCE)
            .into_iter()
            .map(|points| points.into_iter().map(in_ems).collect())
            .collect()
    }

    fn kern(&self, left: char, right: char) -> f32 {
        self.kerns.get(&(left, right)).copied().unwrap_or(0.0)
    }

    /// The width of `text` in ems, with kerning.
    fn width(&self, text: &str) -> f32 {
        let mut pairs = text.chars().peekable();
        let mut width = 0.0;
        while let Some(c) = pairs.next() {
            width += self.glyph(c).advance + pairs.peek().map_or(0.0, |&next| self.kern(c, next));
        }
        width
    }
}

/// When a line renders in and cuts away.
///
/// Its letters start rendering in reading order, overlapping, and have all finished after
/// [`draw_time`](Self::draw_time). They cut away together over the last
/// [`cut_time`](Self::cut_time) of the line's time on screen.
#[derive(Clone, Copy, Debug)]
pub struct Reveal {
    /// Seconds since the line came on screen.
    pub age: f32,
    /// Seconds the line is on screen for.
    pub duration: f32,
    /// The longest the line may take to render in, however long it is on screen.
    pub longest_draw: f32,
}

impl Reveal {
    /// How long the whole line takes to render in, in seconds.
    pub fn draw_time(&self) -> f32 {
        (self.duration * DRAW_SHARE).min(self.longest_draw)
    }

    /// How long the line takes to cut away, in seconds.
    pub fn cut_time(&self) -> f32 {
        (self.duration * CUT_SHARE).min(CUT)
    }

    /// Where the letter at `index` of `count`, in reading order, has got to.
    fn letter(&self, index: usize, count: usize) -> Letter {
        let start = index as f32 / count.saturating_sub(1).max(1) as f32 * (1.0 - LETTER_SHARE);
        let progress = (self.age / self.draw_time() - start) / LETTER_SHARE;
        let cut = (self.age - self.duration) / self.cut_time() + 1.0;
        let span = |value: f32, from: f32, to: f32| ((value - from) / (to - from)).clamp(0.0, 1.0);

        // The stages overlap, both rendering in and cutting away.
        let finished = span(progress, 0.65, 1.0);
        Letter {
            traced: span(progress, 0.0, 0.5),
            shaded: span(progress, 0.35, 0.7).min(1.0 - span(cut, 0.2, 0.55)),
            finished: finished.min(1.0 - span(cut, 0.0, 0.35)),
            lines: (1.0 - finished * finished * (3.0 - 2.0 * finished)).max(span(cut, 0.0, 0.2)),
            shrunk: span(cut, 0.5, 1.0),
        }
    }
}

/// How far a letter has got through rendering in and cutting away, each from 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Letter {
    /// How much of its outline the pen has drawn.
    traced: f32,
    /// How far down it the flat shading has swept.
    shaded: f32,
    /// How far down it the ink has swept.
    finished: f32,
    /// The traced lines' opacity.
    lines: f32,
    /// How much of each traced line has shrunk away.
    shrunk: f32,
}

/// The colours a letter is drawn in before its ink arrives: its traced lines, the pen's tip and
/// the flat shading.
#[derive(Clone, Copy)]
struct Sketch {
    line: Vec3,
    tip: Vec3,
    shading: Vec3,
}

impl Sketch {
    /// The renderer's accent green and silver, as the screen's tiles use. A dark ink, which is
    /// chosen for a light ground where those would be lost, is sketched in tints of itself.
    fn of(ink: Vec3) -> Self {
        let luma = ink.x * 0.2126 + ink.y * 0.7152 + ink.z * 0.0722;
        if luma < 0.5 {
            let lighter = |by: f32| ink + (Vec3::ONE - ink) * by;
            Self {
                line: ink,
                tip: lighter(0.35),
                shading: lighter(0.3),
            }
        } else {
            Self {
                line: Vec3::new(0.43, 0.91, 0.72),
                tip: Vec3::new(0.94, 1.0, 0.97),
                shading: Vec3::new(0.55, 0.57, 0.6),
            }
        }
    }
}

/// How a line's letters are finished.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Finish {
    /// Flat ink in the line's colour.
    Flat,
    /// Solid letters for titles: a brushed silver face with bevelled edges that catch a light
    /// turning about the line, and a band of light sweeping across.
    Metal {
        /// Where the sweep is, 0 at the line's left edge and 1 at its right; outside that range
        /// it is off the line.
        sweep: f32,
        /// The direction the light comes from, in radians, anticlockwise from the right.
        light: f32,
    },
}

/// A line of text to draw, in window pixels measured from the top left.
#[derive(Clone, Copy, Debug)]
pub struct Line<'a> {
    pub text: &'a str,
    /// The horizontal centre of the line.
    pub centre: f32,
    pub baseline: f32,
    pub cap_height: f32,
    /// Extra space between characters, in ems.
    pub tracking: f32,
    pub colour: Vec3,
    pub reveal: Reveal,
    pub finish: Finish,
    /// Whether the joints of the traced lines are marked with squares, as suits large text.
    pub joints: bool,
}

/// One corner of a glyph's quad.
#[derive(Clone, Copy, Debug)]
pub struct GlyphVertex {
    position: Vec2,
    uv: Vec2,
    ink: Vec3,
    /// The flat shading's colour.
    shading: Vec3,
    /// Window pixels per atlas pixel.
    scale: f32,
    /// The finish: 1 for metal and 0 for flat; the corner's place across the line (0 at its left
    /// edge, 1 at its right) and up it (0 at the baseline, 1 at the cap height); and its place
    /// down the glyph (0 at its top, 1 at its bottom).
    finish: Vec4,
    /// How far down the glyph the shading and the ink have swept, and the glyph's height in
    /// window pixels.
    sweeps: Vec3,
    /// The metal finish's sweep and light direction (see [`Finish::Metal`]).
    light: Vec2,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct GlyphVaryings {
    uv: Vec2,
    ink: Vec3,
    shading: Vec3,
    scale: f32,
    finish: Vec4,
    sweeps: Vec3,
    light: Vec2,
}

pub struct GlyphVertexShader;

impl VertexShader for GlyphVertexShader {
    type Vertex = GlyphVertex;
    type Uniforms = ();
    type Varyings = GlyphVaryings;

    fn shade(&self, vertex: GlyphVertex, _: &()) -> (Vec4, GlyphVaryings) {
        (
            Vec4::new(vertex.position.x, vertex.position.y, 0.0, 1.0),
            GlyphVaryings {
                uv: vertex.uv,
                ink: vertex.ink,
                shading: vertex.shading,
                scale: vertex.scale,
                finish: vertex.finish,
                sweeps: vertex.sweeps,
                light: vertex.light,
            },
        )
    }
}

/// Covers the pixels inside each glyph's outline that its sweeps have reached, fading over one
/// pixel at the edge, and gives inked metal letters their bevel and light.
pub struct GlyphShader {
    atlas: TextureSampler,
    /// One atlas pixel in texture coordinates.
    texel: Vec2,
    spread: f32,
}

/// The metal finish, in atlas pixels: how much thicker its letters are than the typeface's
/// strokes, and how wide their bevel is (set by eye, provisional).
const EMBOLDEN: f32 = 0.8;
const BEVEL: f32 = 2.5;

impl FragmentShaderSimd<GlyphVaryings> for GlyphShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: GlyphVaryingsSimd, _: &()) -> ColourSimd {
        let level = |u: F, w: F| self.atlas.sample_linear_clamp_simd([u, w]).r;
        let (u, w) = (v.uv[0], v.uv[1]);
        let metal = v.finish[0];
        // Signed distance to the outline in atlas pixels, positive inside.
        let distance = (level(u, w) - s(0.5)) * s(2.0 * self.spread) + metal * s(EMBOLDEN);
        let coverage = clamp01(distance * v.scale + s(0.5));

        let down = v.finish[3];
        let [shaded, inked, height] = v.sweeps;
        // A sweep starts just above the glyph and ends just below it, so that at 0 it covers
        // nothing and at 1 all of the letter's anti-aliased edge.
        let front = |progress: F| progress * s(1.2) - s(0.1);
        let reached = |progress: F| clamp01((front(progress) - down) * height + s(0.5));
        let (shade_cover, ink_cover) = (reached(shaded), reached(inked));
        // A thin bright scanline leads the shading down.
        let sweeping = (shaded.simd_gt(s(0.0)) & shaded.simd_lt(s(1.0))).select(s(1.0), s(0.0));
        let scanline =
            (s(1.0) - smoothstep(0.5, 2.0, (front(shaded) - down).abs() * height)) * sweeping;
        let alpha = coverage * shade_cover.fast_max(ink_cover).fast_max(scanline);
        let unfinished = mix3(v3(v.shading), rgb(1.0, 1.0, 1.0), scanline * s(0.8));

        let ink = v3(v.ink);
        if !(metal.simd_gt(s(0.0)) & ink_cover.simd_gt(s(0.0))).any() {
            return to_colour(mix3(unfinished, ink, ink_cover), alpha);
        }

        // The bevel's slope follows the distance field's gradient, steepest at the outline.
        let (du, dw) = (s(self.texel.x), s(self.texel.y));
        let gx = (level(u + du, w) - level(u - du, w)) * s(self.spread);
        let gy = (level(u, w + dw) - level(u, w - dw)) * s(self.spread);
        let slope = s(1.0) - clamp01(distance / s(BEVEL));
        let normal = Vec3Simd::new(-gx * slope, -gy * slope, s(1.0)).normalise();
        let (sin, cos) = (v.light[1].sin(), v.light[1].cos());
        let light = Vec3Simd::new(cos * s(0.75), sin * s(0.75), s(0.66));
        let half = Vec3Simd::new(light.x, light.y, light.z + s(1.0)).normalise();
        let diffuse = normal.dot(light).fast_max(s(0.0));
        let specular = {
            let n = normal.dot(half).fast_max(s(0.0));
            let n2 = n * n;
            let n8 = n2 * n2 * n2 * n2;
            n8 * n8 * n8
        };

        // A brushed face, darker at the baseline, with the sweep leaning forward as it crosses.
        let (along, up) = (v.finish[1], v.finish[2]);
        let face = mix3(
            rgb(0.66, 0.69, 0.73),
            rgb(1.0, 1.0, 1.0),
            clamp01(up * s(1.1)),
        );
        let offset = along - v.light[0] - up * s(0.12);
        let sweep = (-(offset * offset) * s(180.0)).exp();
        let lit = face * (s(0.7) + diffuse * s(0.45))
            + rgb(1.0, 0.98, 0.94) * (specular * slope * s(1.3) + sweep * s(0.8));
        let metallic = mix3(lit, ink, s(0.15));
        to_colour(mix3(unfinished, metallic, ink_cover), alpha)
    }
}

/// Lines of text laid out by [`Typesetter::set`], ready to draw.
pub struct Lettering {
    fills: Model<GlyphVertex>,
    strokes: Model<StrokeVertex>,
}

/// Lays out [`Line`]s and draws them, alpha blended, in the presentation pass.
pub struct Typesetter {
    typeface: Typeface,
    /// Each glyph's outline as loops of straight lines, in ems from the pen's position on the
    /// baseline, upwards.
    outlines: HashMap<char, Vec<Loop>>,
    fills: SimdPipeline<GlyphVertexShader, GlyphShader>,
    strokes: SimdPipeline<StrokeVertexShader, StrokeShader>,
}

impl Typesetter {
    /// Loads the typeface's atlas and metrics from `assets/silicon_01/fonts`, relative to the
    /// working directory, and traces its glyphs' outlines.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let typeface = Typeface::parse(&std::fs::read_to_string(METRICS)?)?;
        let atlas =
            Texture::from_image(Path::new(ATLAS))?.sampler(WrapMode::Clamp, FilterMode::Linear);
        let outlines = typeface
            .glyphs
            .iter()
            .map(|(&c, glyph)| (c, typeface.outline(glyph, &atlas)))
            .collect();
        let shader = GlyphShader {
            texel: Vec2::new(
                1.0 / (typeface.atlas_size.0 - 1.0),
                1.0 / (typeface.atlas_size.1 - 1.0),
            ),
            atlas,
            spread: typeface.spread,
        };
        let fills = SimdPipeline::new(GlyphVertexShader, shader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED)
            .with_blend_state(BlendState::ALPHA_BLEND);
        let strokes = SimdPipeline::new(StrokeVertexShader, StrokeShader)
            .with_culling_mode(CullingMode::None)
            .with_depth_state(DepthState::DISABLED)
            .with_blend_state(BlendState::ALPHA_BLEND);
        Ok(Self {
            typeface,
            outlines,
            fills,
            strokes,
        })
    }

    /// Lays out `lines` for a window of `window` pixels, to [`draw`](Self::draw).
    pub fn set(&self, lines: &[Line], window: (usize, usize)) -> Lettering {
        let (mut fills, mut strokes) = (Batch::default(), Batch::default());
        for line in lines {
            self.set_line(line, window, &mut fills, &mut strokes);
        }
        Lettering {
            fills: fills.into_model(),
            strokes: strokes.into_model(),
        }
    }

    fn set_line(
        &self,
        line: &Line,
        window: (usize, usize),
        fills: &mut Batch<GlyphVertex>,
        strokes: &mut Batch<StrokeVertex>,
    ) {
        let face = &self.typeface;
        let em = line.cap_height / face.cap_height;
        let gaps = line.text.chars().count().saturating_sub(1) as f32;
        let text_width = (face.width(line.text) + gaps * line.tracking) * em;
        let left_edge = line.centre - text_width * 0.5;
        let mut pen = left_edge;
        let count = line.text.chars().filter(|&c| c != ' ').count();
        let mut index = 0;
        let mut characters = line.text.chars().peekable();
        while let Some(c) = characters.next() {
            let glyph = face.glyph(c);
            if c != ' ' {
                let letter = line.reveal.letter(index, count);
                index += 1;
                let origin = Vec2::new(pen, line.baseline);
                if letter.shaded > 0.0 || letter.finished > 0.0 {
                    let across = |x: f32| (x - left_edge) / text_width.max(1.0);
                    self.fill_glyph(line, glyph, letter, origin, across, window, fills);
                }
                if letter.traced > 0.0 && letter.lines > 0.0 && letter.shrunk < 1.0 {
                    let loops = self.outlines.get(&c).unwrap_or(&self.outlines[&FALLBACK]);
                    trace_glyph(line, loops, letter, origin, window, em, strokes);
                }
            }
            let kern = characters.peek().map_or(0.0, |&next| face.kern(c, next));
            pen += (glyph.advance + kern + line.tracking) * em;
        }
    }

    /// Adds the quad that shades and inks a glyph whose pen position is `origin`; `across` gives
    /// a window position's place across the line.
    #[expect(clippy::too_many_arguments)]
    fn fill_glyph(
        &self,
        line: &Line,
        glyph: &Glyph,
        letter: Letter,
        origin: Vec2,
        across: impl Fn(f32) -> f32,
        window: (usize, usize),
        fills: &mut Batch<GlyphVertex>,
    ) {
        let face = &self.typeface;
        let em = line.cap_height / face.cap_height;
        let (metal, light) = match line.finish {
            Finish::Flat => (0.0, Vec2::new(0.0, 0.0)),
            Finish::Metal { sweep, light } => (1.0, Vec2::new(sweep, light)),
        };
        let [left, bottom, right, top] = glyph.quad;
        let [u0, v0, u1, v1] = glyph.atlas;
        let (atlas_width, atlas_height) = face.atlas_size;
        // The quad reaches the field's spread beyond the glyph itself.
        let padding = face.spread / face.em_pixels;
        let (ink_top, ink_bottom) = (top - padding, bottom + padding);
        let ink_height = (ink_top - ink_bottom).max(1.0 / face.em_pixels);
        let base = fills.vertices.len() as u32;
        let corners = [
            (left, bottom, u0, v1),
            (right, bottom, u1, v1),
            (right, top, u1, v0),
            (left, top, u0, v0),
        ];
        for (x, y, u, v) in corners {
            let point = Vec2::new(origin.x + x * em, origin.y - y * em);
            fills.vertices.push(GlyphVertex {
                position: ndc(point, window),
                uv: Vec2::new(
                    (u - 0.5) / (atlas_width - 1.0),
                    1.0 - (v - 0.5) / (atlas_height - 1.0),
                ),
                ink: line.colour,
                shading: Sketch::of(line.colour).shading,
                scale: em / face.em_pixels,
                finish: Vec4::new(
                    metal,
                    across(point.x),
                    y / face.cap_height,
                    (ink_top - y) / ink_height,
                ),
                sweeps: Vec3::new(letter.shaded, letter.finished, ink_height * em),
                light,
            });
        }
        fills
            .indices
            .extend([base, base + 2, base + 1, base, base + 3, base + 2]);
    }

    /// Draws text laid out by [`set`](Self::set): the letters' fills, then their lines over them.
    pub fn draw<'pass>(&'pass self, pass: &mut RenderPass<'_, 'pass>, text: &'pass Lettering) {
        text.fills
            .draw_to_render_pass_simd(pass, &self.fills, (), |_| ());
        text.strokes
            .draw_to_render_pass_simd(pass, &self.strokes, (), |_| ());
    }
}

/// Adds the lines the pen has traced round a glyph's `loops`, for a glyph whose pen position is
/// `origin` and whose em is `em` window pixels. Every loop has its own pen, and they all finish
/// together.
fn trace_glyph(
    line: &Line,
    loops: &[Loop],
    letter: Letter,
    origin: Vec2,
    window: (usize, usize),
    em: f32,
    strokes: &mut Batch<StrokeVertex>,
) {
    let sketch = Sketch::of(line.colour);
    let size = window.1 as f32 / 720.0;
    let tip = TIP * em;
    let place = |p: Vec2| Vec2::new(origin.x + p.x * em, origin.y - p.y * em);
    for outline in loops {
        let segments = || {
            (0..outline.len()).map(|i| (place(outline[i]), place(outline[(i + 1) % outline.len()])))
        };
        let length: f32 = segments().map(|(from, to)| (to - from).length()).sum();
        // The pen runs on past the end, so that its bright tip runs off the finished loop.
        let reach = letter.traced * (length + tip);
        let mut start = 0.0;
        for (from, to) in segments() {
            let span = (to - from).length();
            if reach <= start {
                break;
            }
            let drawn = ((reach - start) / span.max(1e-4)).min(1.0);
            let end = from + (to - from) * drawn;
            let heat = |at: f32| (1.0 - (reach - at) / tip).clamp(0.0, 1.0);
            let colour = |at: f32| sketch.line + (sketch.tip - sketch.line) * heat(at);
            // A shrinking line closes in on its middle.
            let middle = (from + end) * 0.5;
            let shrink = |p: Vec2| p + (middle - p) * letter.shrunk;
            let stroke = Stroke {
                from: shrink(from),
                to: shrink(end),
                from_colour: colour(start),
                to_colour: colour(start + span * drawn),
                half_width: LINE * size,
                cap: LINE * size * 0.5,
                opacity: letter.lines,
            };
            stroke.push(window, &mut strokes.vertices, &mut strokes.indices);
            if line.joints {
                let half_side = JOINT * size * (1.0 - letter.shrunk);
                Stroke::square(from, half_side, colour(start), letter.lines).push(
                    window,
                    &mut strokes.vertices,
                    &mut strokes.indices,
                );
            }
            start += span;
        }
    }
}

/// The vertices and indices of the quads collected for one draw call.
struct Batch<V> {
    vertices: Vec<V>,
    indices: Vec<u32>,
}

impl<V: Copy> Batch<V> {
    fn into_model(self) -> Model<V> {
        Model::new(
            vec![Mesh::new(self.vertices, self.indices, None)],
            Vec::new(),
            ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
        )
        .expect("a text mesh has no materials to index")
    }
}

impl<V> Default for Batch<V> {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }
}

/// A point in window pixels from the top left, in normalised device coordinates.
fn ndc(point: Vec2, window: (usize, usize)) -> Vec2 {
    Vec2::new(
        2.0 * point.x / window.0 as f32 - 1.0,
        1.0 - 2.0 * point.y / window.1 as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typeface() -> Typeface {
        Typeface::parse(include_str!("../../../assets/silicon_01/fonts/metrics.txt")).unwrap()
    }

    fn reveal(age: f32, duration: f32) -> Reveal {
        Reveal {
            age,
            duration,
            longest_draw: 0.8,
        }
    }

    #[test]
    fn a_line_renders_in_faster_the_less_time_it_has_on_screen() {
        assert_eq!(reveal(0.0, 5.0).draw_time(), 0.8);
        assert!(reveal(0.0, 0.8).draw_time() < reveal(0.0, 1.6).draw_time());
        let brief = reveal(0.0, 0.8);
        assert!(brief.draw_time() + brief.cut_time() < brief.duration);
    }

    #[test]
    fn letters_render_in_reading_order_and_have_all_finished_by_the_draw_time() {
        let early = reveal(0.2, 3.0);
        let letters: Vec<Letter> = (0..12).map(|i| early.letter(i, 12)).collect();
        assert!(
            letters
                .windows(2)
                .all(|pair| pair[1].traced <= pair[0].traced)
        );
        assert!(letters[0].traced > 0.0 && letters[11].traced == 0.0);

        let drawn = reveal(early.draw_time() + 1e-3, 3.0);
        for i in 0..12 {
            let letter = drawn.letter(i, 12);
            assert_eq!(
                (letter.finished, letter.lines, letter.shrunk),
                (1.0, 0.0, 0.0)
            );
        }
    }

    #[test]
    fn every_letter_cuts_away_together_leaving_nothing_at_the_end() {
        let cut_time = reveal(0.0, 3.0).cut_time();
        let cutting = reveal(3.0 - cut_time * 0.6, 3.0);
        let first = cutting.letter(0, 12);
        assert!((1..12).all(|i| cutting.letter(i, 12) == first));
        assert!(first.finished < 1.0 && first.lines == 1.0);

        let gone = reveal(3.0, 3.0).letter(5, 12);
        assert_eq!((gone.shaded, gone.finished, gone.shrunk), (0.0, 0.0, 1.0));
    }

    #[test]
    fn a_glyph_traces_its_outline_and_each_counter_as_separate_loops() {
        let face = typeface();
        let atlas = Texture::from_image(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../assets/silicon_01/fonts/atlas.png"
        )))
        .unwrap()
        .sampler(WrapMode::Clamp, FilterMode::Linear);
        let loops = |c| face.outline(face.glyph(c), &atlas).len();
        assert_eq!(['l', 'O', 'B'].map(loops), [1, 2, 3]);
    }

    #[test]
    fn kerning_tightens_pairs_that_the_typeface_kerns() {
        let face = typeface();
        let unkerned = face.glyph('A').advance + face.glyph('V').advance;
        assert!(face.width("AV") < unkerned);
    }

    #[test]
    fn every_character_of_the_captions_has_a_glyph() {
        let face = typeface();
        let captions = crate::director::CAPTIONS.map(|caption| caption.text);
        for text in captions.iter().chain(&crate::director::MONTAGE) {
            for c in text.chars() {
                assert!(face.glyphs.contains_key(&c), "no glyph for {c:?}");
            }
        }
    }
}
