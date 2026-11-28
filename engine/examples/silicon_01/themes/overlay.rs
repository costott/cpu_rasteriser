//! Glowing bitmap text drawn into a stage's scene pass, so a theme's post effects restyle it
//! with the rest of the picture: a CRT's readouts, a thermal camera's HUD, a drawing's
//! annotations.
//!
//! Each character is a quad whose fragment shader reads a 5Ã—7 glyph from the font table and
//! lights each glyph pixel as a soft square with a glow round it. The text is drawn additively
//! and leaves the scene's view distance untouched.

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::meshes::model;
use crate::simd::{F, clamp01, s, to_colour};

/// The font table: 5Ã—7 glyphs, one row per byte from the top, the leftmost pixel in bit 4.
/// Glyphs of 5×7 pixels, one row per byte from the top, the leftmost pixel in bit 4.
const GLYPHS: [(char, [u8; 7]); 54] = [
    (' ', [0, 0, 0, 0, 0, 0, 0]),
    (
        'A',
        [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'B',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        'C',
        [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
    ),
    (
        'D',
        [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        'E',
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        'F',
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        'G',
        [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
    ),
    (
        'H',
        [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'I',
        [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        'J',
        [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
    ),
    (
        'K',
        [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        'L',
        [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        'M',
        [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'N',
        [
            0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001,
        ],
    ),
    (
        'O',
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'P',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        'Q',
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
    ),
    (
        'R',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        'S',
        [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    ),
    (
        'T',
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        'U',
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'V',
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
    ),
    (
        'W',
        [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
    ),
    (
        'X',
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
    ),
    (
        'Y',
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        'Z',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
    ),
    (
        '0',
        [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
    ),
    (
        '1',
        [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        '2',
        [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
    ),
    (
        '3',
        [
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        '4',
        [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
    ),
    (
        '5',
        [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        '6',
        [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '7',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
    ),
    (
        '8',
        [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '9',
        [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
    ),
    ('.', [0, 0, 0, 0, 0, 0b01100, 0b01100]),
    (',', [0, 0, 0, 0, 0b01100, 0b00100, 0b01000]),
    (':', [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0]),
    ('-', [0, 0, 0, 0b11111, 0, 0, 0]),
    (
        '>',
        [
            0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000,
        ],
    ),
    (
        '/',
        [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
    ),
    (
        '!',
        [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0b00100],
    ),
    ('\'', [0b00100, 0b00100, 0b01000, 0, 0, 0, 0]),
    ('_', [0, 0, 0, 0, 0, 0, 0b11111]),
    (
        '#',
        [
            0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010,
        ],
    ),
    ('=', [0, 0, 0b11111, 0, 0b11111, 0, 0]),
    (
        '(',
        [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
    ),
    (
        ')',
        [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
    ),
    (
        '%',
        [
            0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011,
        ],
    ),
    ('+', [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0]),
    ('*', [0, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0]),
    (
        '\u{2588}',
        [
            0b11111, 0b11111, 0b11111, 0b11111, 0b11111, 0b11111, 0b11111,
        ],
    ),
];

/// The glyph for `c`, or the space for characters the font lacks.
fn glyph_index(c: char) -> usize {
    let c = c.to_ascii_uppercase();
    GLYPHS
        .iter()
        .position(|(glyph, _)| *glyph == c)
        .unwrap_or(0)
}

/// Whether pixel (`column`, `row`) of glyph `index` is lit; false outside the 5Ã—7 grid.
fn glyph_pixel(index: usize, column: i32, row: i32) -> bool {
    if !(0..5).contains(&column) || !(0..7).contains(&row) {
        return false;
    }
    GLYPHS[index].1[row as usize] & (0b10000 >> column) != 0
}

#[derive(Clone, Copy, Debug)]
pub enum Align {
    Left,
    Centre,
    Right,
}

/// A line of text to set.
#[derive(Clone, Copy, Debug)]
pub struct Line<'a> {
    pub text: &'a str,
    /// The anchor in normalised device coordinates.
    pub x: f32,
    pub y: f32,
    pub align: Align,
    pub colour: Vec3,
}

/// The text of one frame.
#[derive(Default)]
pub struct Text {
    /// Each vertex's position holds its place on screen in x and y and its glyph in z; its uv is
    /// the position in glyph pixels.
    vertices: Vec<ObjVertex>,
    indices: Vec<u32>,
}

impl Text {
    /// Adds a line. `pixel` is the size of one glyph pixel in pixels of a `window`-sized target.
    pub fn line(&mut self, line: Line, pixel: f32, window: (usize, usize)) {
        let characters: Vec<char> = line.text.chars().collect();
        let advance = 6.0;
        let width = characters.len() as f32 * advance - 1.0;
        let to_ndc = Vec2::new(2.0 * pixel / window.0 as f32, 2.0 * pixel / window.1 as f32);
        let left = match line.align {
            Align::Left => line.x,
            Align::Centre => line.x - width * 0.5 * to_ndc.x,
            Align::Right => line.x - width * to_ndc.x,
        };
        for (i, &c) in characters.iter().enumerate().filter(|(_, c)| **c != ' ') {
            let x0 = left + i as f32 * advance * to_ndc.x;
            let base = self.vertices.len() as u32;
            for (cx, cy) in [(-1.0, -1.0), (6.0, -1.0), (6.0, 8.0), (-1.0, 8.0)] {
                self.vertices.push(ObjVertex {
                    position: Vec3::new(
                        x0 + cx * to_ndc.x,
                        line.y - cy * to_ndc.y,
                        glyph_index(c) as f32,
                    ),
                    colour: Colour::new(line.colour.x, line.colour.y, line.colour.z, 1.0),
                    normal: Vec3::ZERO,
                    uv: Vec2::new(cx, cy),
                });
            }
            self.indices
                .extend([base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    }

    fn into_model(self) -> Option<Model<ObjVertex>> {
        if self.vertices.is_empty() {
            return None;
        }
        Some(model(Mesh::new(self.vertices, self.indices, None)))
    }
}

/// One frame's text, drawn additively over the scene without touching its view distance.
pub struct Overlay {
    pipeline: SimdPipeline<TextVertexShader, TextShader>,
    model: Option<Model<ObjVertex>>,
}

impl Overlay {
    pub fn new() -> Self {
        Self {
            pipeline: SimdPipeline::new(TextVertexShader, TextShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED)
                .with_blend_state(BlendState::ADDITIVE),
            model: None,
        }
    }

    /// Replaces the text to draw; call it from `Stage::prepare`.
    pub fn set(&mut self, text: Text) {
        self.model = text.into_model();
    }

    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>) {
        if let Some(model) = &self.model {
            model.draw_to_render_pass_simd(pass, &self.pipeline, (), |_| ());
        }
    }
}

#[derive(Interpolate, SimdInterpolate)]
struct TextVaryings {
    cell: Vec2,
    colour: Vec3,
    glyph: f32,
}

struct TextVertexShader;

impl VertexShader for TextVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = ();
    type Varyings = TextVaryings;

    fn shade(&self, vertex: ObjVertex, _: &()) -> (Vec4, TextVaryings) {
        let p = vertex.position;
        (
            Vec4::new(p.x, p.y, 0.0, 1.0),
            TextVaryings {
                cell: vertex.uv,
                colour: Vec3::new(vertex.colour.r, vertex.colour.g, vertex.colour.b),
                glyph: p.z,
            },
        )
    }
}

struct TextShader;

impl FragmentShaderSimd<TextVaryings> for TextShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: TextVaryingsSimd, _: &()) -> ColourSimd {
        let u = v.cell[0].to_array();
        let w = v.cell[1].to_array();
        let glyph = v.glyph.to_array();
        let mut core = [0.0f32; 8];
        let mut glow = [0.0f32; 8];
        for lane in 0..8 {
            let (x, y) = (u[lane], w[lane]);
            let (column, row) = (x.floor() as i32, y.floor() as i32);
            let index = glyph[lane].round() as usize;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (c, r) = (column + dx, row + dy);
                    if !glyph_pixel(index, c, r) {
                        continue;
                    }
                    let ox = (x - (c as f32 + 0.5)).abs() - 0.42;
                    let oy = (y - (r as f32 + 0.5)).abs() - 0.42;
                    let d = ox.max(0.0).hypot(oy.max(0.0));
                    core[lane] = f32::max(core[lane], 1.0 - (d * 6.0).min(1.0));
                    glow[lane] += (-d * 2.2).exp() * 0.12;
                }
            }
        }
        let intensity: F = clamp01(F::from(core) + F::from(glow));
        to_colour(
            Vec3Simd::new(v.colour[0], v.colour[1], v.colour[2]) * intensity,
            s(0.0),
        )
    }
}
