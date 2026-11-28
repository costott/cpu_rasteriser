//! glyph-rain: falling green code. Three depths of glyph columns, big and bright in front and
//! small and dim behind, rain down over a glossy black floor that reflects them. The chip is
//! rewritten in the same glyphs, brightest along its outline.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef};
use crate::simd::{F, clamp01, fract, from_colour, hash2, mix3, rgb, s, select, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::post::{ScreenPass, compress};
use crate::themes::{Scene, Stage};

const GREEN: Vec3 = Vec3 {
    x: 0.06,
    y: 1.0,
    z: 0.22,
};
const DISTANCE_CAP: f32 = 60.0;
/// The glyph cell of the chip's rewriting, in pixels of a 720-line frame.
const CHIP_CELL: (f32, f32) = (14.0, 22.0);
/// The edge strength, as a fraction of the surface's distance, above which a cell is outline.
const EDGE: f32 = 0.08;

/// The three depths of rain: cell width and height in 720-line pixels, fall speed in rows a
/// second, brightness and a seed.
const LAYERS: [(f32, f32, f32, f32, f32); 3] = [
    (7.0, 11.0, 9.0, 0.35, 1.0),
    (11.0, 18.0, 7.0, 0.65, 2.0),
    (17.0, 28.0, 5.0, 1.0, 3.0),
];

/// 5x7 glyphs, one row per byte from the top with the leftmost pixel in bit 4: the digits, then
/// strokes in the manner of half-width katakana.
const GLYPHS: [[u8; 7]; 24] = [
    [
        0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
    ],
    [
        0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
    ],
    [
        0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
    ],
    [
        0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
    ],
    [
        0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
    ],
    [
        0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
    ],
    [
        0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
    ],
    [
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
    ],
    [
        0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
    ],
    [
        0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
    ],
    [
        0b11111, 0b00010, 0b00110, 0b01010, 0b10010, 0b00010, 0b00010,
    ],
    [
        0b00100, 0b11111, 0b00100, 0b01110, 0b10101, 0b00100, 0b00100,
    ],
    [
        0b11110, 0b00010, 0b11110, 0b00010, 0b11110, 0b00010, 0b00010,
    ],
    [
        0b01000, 0b01111, 0b10010, 0b00100, 0b01000, 0b10000, 0b00000,
    ],
    [
        0b00100, 0b11111, 0b00001, 0b00110, 0b01100, 0b00100, 0b00100,
    ],
    [
        0b11111, 0b00001, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000,
    ],
    [
        0b01110, 0b00000, 0b11111, 0b00100, 0b00100, 0b01010, 0b10001,
    ],
    [
        0b00100, 0b00100, 0b11111, 0b00100, 0b01010, 0b10001, 0b00000,
    ],
    [
        0b10001, 0b10001, 0b11111, 0b00001, 0b00010, 0b00100, 0b11000,
    ],
    [
        0b01110, 0b00100, 0b11111, 0b00100, 0b00100, 0b00100, 0b01000,
    ],
    [
        0b00010, 0b11111, 0b00110, 0b01010, 0b10010, 0b00010, 0b00110,
    ],
    [
        0b10000, 0b10000, 0b10000, 0b10000, 0b10001, 0b10010, 0b11100,
    ],
    [
        0b01010, 0b01010, 0b11111, 0b01010, 0b01010, 0b10010, 0b00000,
    ],
    [
        0b00100, 0b01110, 0b10101, 0b00100, 0b00100, 0b00100, 0b00100,
    ],
];

/// 1 where glyph `id` has a lit pixel at (`across`, `down`) in the cell, both 0 to 1 with down
/// measured from the cell's top. The glyph fills the cell's middle four fifths.
#[inline(always)]
fn glyph(id: F, across: F, down: F) -> F {
    let (id, across, down) = (id.to_array(), across.to_array(), down.to_array());
    let mut lit = [0.0f32; 8];
    for lane in 0..8 {
        let column = (across[lane] - 0.1) / 0.8 * 5.0;
        let row = (down[lane] - 0.1) / 0.8 * 7.0;
        if (0.0..5.0).contains(&column) && (0.0..7.0).contains(&row) {
            let bits = GLYPHS[id[lane] as usize % GLYPHS.len()][row as usize];
            lit[lane] = (bits >> (4 - column as usize) & 1) as f32;
        }
    }
    F::from(lit)
}

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Rain;

impl FragmentShaderSimd<ScreenVaryings> for Rain {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let time = u.time;
        let horizon = -view.forward.y / (view.up.y * view.tan_half_fov);
        let reflected = ny.simd_lt(s(horizon));
        let depth_below = (s(horizon) - ny).fast_max(s(0.0));
        let ny_seen = select(reflected, s(2.0 * horizon) - ny, ny);
        let ripple =
            (ny * s(40.0) + s(time * 1.5)).sin() * s(0.002) * select(reflected, s(1.0), s(0.0));

        let x = (nx + ripple) * s(360.0 * view.aspect) + s(360.0 * view.aspect);
        let y = (s(1.0) - ny_seen) * s(360.0);

        let band = (-(ny - s(horizon)).abs() * s(22.0)).exp();
        let mut colour = mix3(rgb(0.0, 0.003, 0.001), rgb(0.0, 0.03, 0.01), band)
            + splat3(GREEN) * (band * s(0.04));
        for layer in LAYERS {
            colour = colour + rain(x, y, layer, time);
        }
        let mirror = select(reflected, (-depth_below * s(3.2)).exp() * s(0.4), s(1.0));
        let lit = colour * mirror;
        to_colour(lit, s(SKY_DISTANCE))
    }
}

/// One depth of rain at (`x`, `y`) in 720-line pixels measured from the top left: columns that
/// each fall at their own speed trailing a fading tail behind a white head.
#[inline(always)]
fn rain(
    x: F,
    y: F,
    (cell_w, cell_h, speed, brightness, seed): (f32, f32, f32, f32, f32),
    time: f32,
) -> Vec3Simd {
    let rows = 720.0 / cell_h;
    let (column, row) = ((x / s(cell_w)).floor(), (y / s(cell_h)).floor());
    let (across, down) = (x / s(cell_w) - column, y / s(cell_h) - row);

    let h = |k: f32| hash2(column * s(1.0) + s(seed * 31.7), s(k));
    let tail = s(10.0) + h(1.0) * s(22.0);
    let period = s(rows * 1.3) + tail;
    let head = fract((s(time * speed) * (s(0.55) + h(2.0)) + h(3.0) * period) / period) * period;
    let behind = head - row;
    let behind = behind - (behind / period).floor() * period;
    let in_tail = select(behind.simd_lt(tail), s(1.0), s(0.0));
    let fade = clamp01(s(1.0) - behind / tail);
    let active = select(h(4.0).simd_lt(s(0.78)), s(1.0), s(0.0));

    let changes = (s(time) * (s(1.5) + h(5.0) * s(3.0)) + hash2(column, row) * s(4.0)).floor();
    let id = (hash2(column + row * s(57.3), changes + s(seed)) * s(GLYPHS.len() as f32)).floor();
    let lit = glyph(id, across, down) * in_tail * active;

    let head_glow = select(behind.simd_lt(s(1.0)), s(1.0), s(0.0));
    let body = splat3(GREEN) * (fade * fade * s(1.4 * brightness));
    let tip = rgb(0.8, 1.0, 0.85) * (s(2.6 * brightness));
    mix3(body, tip, head_glow) * lit
}

#[derive(Clone)]
struct ChipUniforms {
    source: Arc<TextureSampler>,
    unit: f32,
    time: f32,
}

/// Rewrites what the scene drew of the chip (the pixels with a surface distance) as glyphs
/// whose brightness follows the shading, lit hardest on cells that hold an outline.
struct Rewrite;

impl FragmentShaderSimd<ScreenVaryings> for Rewrite {
    type Uniforms = ChipUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &ChipUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (width, height) = (u.source.width() as f32, u.source.height() as f32);
        let image = u.source.sample_linear_clamp_simd([x, y]);
        let (cell_w, cell_h) = (CHIP_CELL.0 * u.unit, CHIP_CELL.1 * u.unit);
        let (px, py) = (x * s(width), y * s(height));
        let (column, row) = ((px / s(cell_w)).floor(), (py / s(cell_h)).floor());
        let across = px / s(cell_w) - column;
        let down = s(1.0) - (py / s(cell_h) - row);
        let (centre_x, centre_y) = (
            (column + s(0.5)) * s(cell_w / width),
            (row + s(0.5)) * s(cell_h / height),
        );
        let tap = |dx: f32, dy: f32| {
            u.source.sample_linear_clamp_simd([
                centre_x + s(dx * cell_w / width),
                centre_y + s(dy * cell_h / height),
            ])
        };

        let mut tone = s(0.0);
        for (ox, oy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
            let sample = tap(ox, oy);
            let c = compress(from_colour(sample));
            let on_chip = select(sample.a.simd_lt(s(DISTANCE_CAP)), s(1.0), s(0.0));
            tone += (c.x * s(0.2126) + c.y * s(0.7152) + c.z * s(0.0722)) * on_chip * s(0.25);
        }
        let distance = |dx: f32, dy: f32| tap(dx, dy).a.fast_min(s(DISTANCE_CAP));
        let here = distance(0.0, 0.0);
        let gx = distance(1.0, 0.0) - distance(-1.0, 0.0);
        let gy = distance(0.0, 1.0) - distance(0.0, -1.0);
        let edge = select(
            ((gx * gx + gy * gy).sqrt() / here).simd_gt(s(EDGE)),
            s(1.0),
            s(0.0),
        );

        let changes = (s(u.time * 7.0) + hash2(column, row) * s(3.0)).floor();
        let id = (hash2(column + row * s(57.3), changes) * s(GLYPHS.len() as f32)).floor();
        let lit = glyph(id, across, down);
        let strength = (s(0.2) + tone * s(1.5) + edge * s(1.2)) * lit + tone * s(0.05);
        let on_chip = select(image.a.simd_lt(s(DISTANCE_CAP)), s(1.0), s(0.0));
        let written = splat3(GREEN) * strength + rgb(0.5, 0.7, 0.55) * (edge * lit * s(0.8));
        let result = mix3(from_colour(image), written, on_chip);
        to_colour(result, image.a)
    }
}

pub struct GlyphRain {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Rain>,
    rewrite: ScreenPass<Rewrite>,
}

impl GlyphRain {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.15, 0.45, 0.22),
                die: Vec3::new(0.02, 0.2, 0.05),
                gold: Vec3::new(0.5, 1.0, 0.6),
                streams: Vec3::new(0.6, 1.0, 0.7),
                key: Vec3::new(-0.55, 0.5, 0.65).normalise(),
                key_colour: Vec3::new(1.0, 3.0, 1.3),
                sky: Vec3::new(0.0, 0.12, 0.04),
                ground: Vec3::new(0.0, 0.05, 0.02),
                softbox: Vec3::ZERO,
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Rain)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            rewrite: ScreenPass::new(Rewrite),
        }
    }
}

impl Stage for GlyphRain {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Chip],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::DigitalTrickle),
                0.0,
                0.45,
                0.064,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        let uniforms = Uniforms {
            view: scene.view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let uniforms = ChipUniforms {
            source: image,
            unit: scene.extent.height as f32 / 720.0,
            time: scene.local_time,
        };
        Some(self.rewrite.run(context, scene, uniforms))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 1.2,
            vignette: 0.4,
            grain: 0.01,
            aberration: 0.002,
            desaturate: 0.0,
        }
    }
}
