//! retro-8bit: a 160x90 side-scrolling level in sixteen colours. Hills with eyes, clouds, bushes
//! and a brick floor scroll past at their own speeds under a bumped block and a spinning coin;
//! the whole frame, chip included, is snapped to 8-pixel cells, dithered and matched to the
//! palette.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef};
use crate::simd::{F, clamp01, fract, from_colour, mix3, s, select, splat3, to_colour};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Treatment, WorldSound};
use crate::themes::kit::post::{Chain, Edges, Effect, ScreenPass, compress, expand};
use crate::themes::{Scene, Stage};

/// The picture's size in cells, and a cell's size in pixels of a 720-line frame.
const COLUMNS: f32 = 160.0;
const ROWS: f32 = 90.0;
const CELL: f32 = 8.0;
/// The row where the brick floor ends and the level begins.
const GROUND: f32 = 24.0;

/// The sixteen colours, as displayed (bytes of red, green and blue).
const PALETTE: [(u8, u8, u8); 16] = [
    (0, 0, 0),
    (29, 43, 83),
    (92, 148, 252),
    (164, 228, 252),
    (255, 255, 255),
    (0, 168, 0),
    (88, 216, 84),
    (0, 88, 0),
    (200, 76, 12),
    (252, 152, 56),
    (248, 216, 120),
    (252, 188, 176),
    (124, 124, 124),
    (188, 188, 188),
    (168, 16, 0),
    (80, 48, 0),
];

const BLACK: usize = 0;
const SKY: usize = 2;
const PALE: usize = 3;
const WHITE: usize = 4;
const GREEN: usize = 5;
const LIGHT_GREEN: usize = 6;
const DARK_GREEN: usize = 7;
const BRICK: usize = 8;
const ORANGE: usize = 9;
const YELLOW: usize = 10;
const BROWN: usize = 15;

#[inline(always)]
fn colour(index: usize) -> Vec3Simd {
    let (r, g, b) = PALETTE[index];
    Vec3Simd::new(
        s(r as f32 / 255.0),
        s(g as f32 / 255.0),
        s(b as f32 / 255.0),
    )
}

/// 1 where `d` is negative.
#[inline(always)]
fn inside(d: F) -> F {
    select(d.simd_lt(s(0.0)), s(1.0), s(0.0))
}

#[inline(always)]
fn rectangle(x: F, y: F, (x0, y0, x1, y1): (f32, f32, f32, f32)) -> F {
    inside(
        (s(x0) - x)
            .fast_max(x - s(x1))
            .fast_max(s(y0) - y)
            .fast_max(y - s(y1)),
    )
}

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Level;

impl FragmentShaderSimd<ScreenVaryings> for Level {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let time = u.time;
        let gx = s(COLUMNS * 0.5) + nx * s(ROWS * 0.5 * u.view.aspect);
        let gy = (ny * s(0.5) + s(0.5)) * s(ROWS);

        let mut result = mix3(colour(PALE), colour(SKY), clamp01((gy - s(28.0)) / s(60.0)));
        for (x, y, size, speed) in [
            (20.0, 66.0, 1.0, 2.0),
            (95.0, 77.0, 0.8, 3.0),
            (165.0, 62.0, 1.2, 2.5),
        ] {
            result = cloud(result, gx, gy, (x, y, size), time * speed);
        }
        result = hills(result, gx, gy, time * 3.0, 0.0, false);
        result = hills(result, gx, gy, time * 7.0, 90.0, true);
        result = bush(result, gx, gy, time * 12.0);
        result = floor(result, gx, gy, time * 12.0);
        result = blocks(result, gx, gy, time);
        to_colour(expand(result), s(SKY_DISTANCE))
    }
}

/// A cloud of three puffs, white over a pale underside, drifting left and wrapping.
#[inline(always)]
fn cloud(base: Vec3Simd, gx: F, gy: F, (x, y, size): (f32, f32, f32), scroll: f32) -> Vec3Simd {
    const PERIOD: f32 = 230.0;
    let offset = gx - s(x - scroll);
    let wrapped = offset - (offset / s(PERIOD) + s(0.5)).floor() * s(PERIOD);
    let (dx, dy) = (wrapped / s(size), (gy - s(y)) / s(size));
    let puff = |cx: f32, cy: f32, r: f32| {
        ((dx - s(cx)) * (dx - s(cx)) + (dy - s(cy)) * (dy - s(cy))).sqrt() - s(r)
    };
    let d = puff(-7.0, 0.0, 5.0)
        .fast_min(puff(0.0, 3.0, 6.5))
        .fast_min(puff(7.0, 0.0, 5.0))
        .fast_max(-dy - s(3.0));
    let body = mix3(colour(WHITE), colour(PALE), inside(dy + s(0.5)));
    mix3(base, body, inside(d))
}

/// A row of round-topped green hills, each with a pair of dark eyes, scrolling by `scroll`.
#[inline(always)]
fn hills(base: Vec3Simd, gx: F, gy: F, scroll: f32, offset: f32, near: bool) -> Vec3Simd {
    const PERIOD: f32 = 250.0;
    let p = fract((gx + s(scroll + offset)) / s(PERIOD)) * s(PERIOD);
    let fill = if near { LIGHT_GREEN } else { GREEN };
    let list: [(f32, f32, f32); 3] = if near {
        [(40.0, 22.0, 14.0), (140.0, 28.0, 18.0), (215.0, 18.0, 12.0)]
    } else {
        [(30.0, 32.0, 24.0), (125.0, 40.0, 34.0), (210.0, 30.0, 22.0)]
    };
    let mut result = base;
    for (cx, rx, ry) in list {
        let (qx, qy) = ((p - s(cx)) / s(rx), (gy - s(GROUND - 2.0)) / s(ry));
        let q = (qx * qx + qy * qy).sqrt();
        let thickness = s(1.4 / rx.min(ry));
        let body = inside(q - s(1.0)) * select(qy.simd_gt(s(0.0)), s(1.0), s(0.0));
        let edge = select((s(1.0) - q).simd_lt(thickness), s(1.0), s(0.0));
        result = mix3(result, mix3(colour(fill), colour(DARK_GREEN), edge), body);
        for side in [-1.0f32, 1.0] {
            let eye = rectangle(
                p - s(cx + side * 3.0),
                gy - s(GROUND - 2.0),
                (-0.9, ry * 0.42, 0.9, ry * 0.42 + 5.0),
            );
            result = mix3(result, colour(BLACK), eye * body);
        }
    }
    result
}

/// A bush: three round green puffs on the floor's edge.
#[inline(always)]
fn bush(base: Vec3Simd, gx: F, gy: F, scroll: f32) -> Vec3Simd {
    const PERIOD: f32 = 170.0;
    let p = fract((gx + s(scroll)) / s(PERIOD)) * s(PERIOD);
    let (dx, dy) = (p - s(60.0), gy - s(GROUND));
    let puff = |cx: f32, cy: f32, r: f32| {
        ((dx - s(cx)) * (dx - s(cx)) + (dy - s(cy)) * (dy - s(cy))).sqrt() - s(r)
    };
    let d = puff(-6.0, 0.0, 4.5)
        .fast_min(puff(0.0, 1.0, 5.5))
        .fast_min(puff(6.0, 0.0, 4.5))
        .fast_max(-dy);
    let edge = select(d.simd_gt(s(-1.2)), s(1.0), s(0.0));
    mix3(
        base,
        mix3(colour(LIGHT_GREEN), colour(DARK_GREEN), edge),
        inside(d),
    )
}

/// The brick floor: staggered bricks with mortar lines, a lit top edge and a shaded underside.
#[inline(always)]
fn floor(base: Vec3Simd, gx: F, gy: F, scroll: f32) -> Vec3Simd {
    let depth = s(GROUND) - gy;
    let row = (depth / s(5.0)).floor();
    let stagger = select(fract(row * s(0.5)).simd_lt(s(0.25)), s(0.0), s(5.0));
    let mx = fract((gx + s(scroll) + stagger) / s(10.0)) * s(10.0);
    let my = depth - row * s(5.0);
    let mortar = select(mx.simd_lt(s(1.0)) | my.simd_lt(s(1.0)), s(1.0), s(0.0));
    let lit = select(my.simd_lt(s(2.0)), s(1.0), s(0.0));
    let face = mix3(colour(BRICK), colour(ORANGE), lit);
    let brick = mix3(face, colour(BROWN), mortar);
    let top = select(depth.simd_lt(s(1.5)), s(1.0), s(0.0));
    let ground = mix3(brick, colour(ORANGE), top);
    mix3(base, ground, inside(-depth))
}

/// A question block that gets bumped every couple of seconds, with a coin spinning out of it.
#[inline(always)]
fn blocks(base: Vec3Simd, gx: F, gy: F, time: f32) -> Vec3Simd {
    let cycle = (time / 2.0).fract();
    let bump = if cycle < 0.15 {
        (cycle / 0.15 * std::f32::consts::PI).sin() * 3.0
    } else {
        0.0
    };
    let (bx, by) = (gx - s(22.0), gy - s(52.0 + bump));
    let square = rectangle(bx, by, (0.0, 0.0, 10.0, 10.0));
    let rim = rectangle(bx, by, (1.0, 1.0, 9.0, 9.0));
    let mark =
        rectangle(bx, by, (4.0, 3.0, 6.0, 7.0)).fast_max(rectangle(bx, by, (4.0, 1.5, 6.0, 2.5)));
    let mut result = mix3(base, colour(BROWN), square);
    result = mix3(result, colour(ORANGE), rim);
    result = mix3(result, colour(YELLOW), mark * rim);

    for k in 0..2 {
        let x0 = 32.0 + 10.0 * k as f32;
        let (cx, cy) = (gx - s(x0), gy - s(52.0));
        let shell = rectangle(cx, cy, (0.0, 0.0, 10.0, 10.0));
        let line = select(
            cx.simd_lt(s(1.0)) | cy.simd_lt(s(1.0)) | cy.simd_gt(s(9.0)),
            s(1.0),
            s(0.0),
        );
        let face = mix3(colour(BRICK), colour(BROWN), line);
        result = mix3(result, face, shell);
    }

    if cycle < 0.5 {
        let q = cycle * 2.0;
        let rise = 10.0 + 14.0 * (q * std::f32::consts::PI).sin();
        let width = 0.8 + 3.2 * (time * 14.0).cos().abs();
        let (px, py) = ((gx - s(27.0)) / s(width), (gy - s(52.0 + rise)) / s(5.0));
        let coin = inside((px * px + py * py).sqrt() - s(1.0));
        let shade = select(px.simd_gt(s(0.2)), s(1.0), s(0.0));
        result = mix3(result, mix3(colour(YELLOW), colour(ORANGE), shade), coin);
    }
    result
}

#[derive(Clone)]
struct PaletteUniforms {
    source: Arc<TextureSampler>,
    unit: f32,
}

/// Snaps the picture to cells, dithers it with a 4x4 Bayer matrix and replaces every cell with
/// the nearest palette colour.
struct Quantise;

impl FragmentShaderSimd<ScreenVaryings> for Quantise {
    type Uniforms = PaletteUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &PaletteUniforms) -> ColourSimd {
        const DITHER: f32 = 0.16;
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (width, height) = (u.source.width() as f32, u.source.height() as f32);
        let cell = CELL * u.unit;
        let (cx, cy) = (
            (x * s(width / cell)).floor(),
            (y * s(height / cell)).floor(),
        );
        let sample = u.source.sample_linear_clamp_simd([
            (cx + s(0.5)) * s(cell / width),
            (cy + s(0.5)) * s(cell / height),
        ]);
        let threshold = bayer(cx, cy) - s(0.5);
        let target = compress(from_colour(sample)) + splat3(Vec3::ONE) * (threshold * s(DITHER));

        let mut best = s(f32::MAX);
        let mut chosen = colour(0);
        for index in 0..PALETTE.len() {
            let c = colour(index);
            let (dr, dg, db) = (target.x - c.x, target.y - c.y, target.z - c.z);
            let distance = dr * dr * s(0.3) + dg * dg * s(0.5) + db * db * s(0.2);
            let closer = distance.simd_lt(best);
            best = select(closer, distance, best);
            chosen = Vec3Simd::new(
                select(closer, c.x, chosen.x),
                select(closer, c.y, chosen.y),
                select(closer, c.z, chosen.z),
            );
        }
        to_colour(expand(chosen), sample.a)
    }
}

/// The 4x4 Bayer matrix's threshold at cell (`x`, `y`), in (0, 1).
#[inline(always)]
fn bayer(x: F, y: F) -> F {
    let (mut px, mut py) = (x, y);
    let mut value = s(0.0);
    let mut weight = 4.0;
    for _ in 0..2 {
        let bit = |p: F| p - (p * s(0.5)).floor() * s(2.0);
        let (xb, yb) = (bit(px), bit(py));
        value += ((xb - yb).abs() * s(2.0) + yb) * s(weight);
        weight *= 0.25;
        px = (px * s(0.5)).floor();
        py = (py * s(0.5)).floor();
    }
    (value + s(0.5)) * s(1.0 / 16.0)
}

pub struct Retro8bit {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Level>,
    outline: Chain,
    quantise: ScreenPass<Quantise>,
}

impl Retro8bit {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.55, 0.55, 0.6),
                die: Vec3::new(0.0, 0.3, 0.0),
                gold: Vec3::new(0.95, 0.7, 0.2),
                streams: Vec3::new(0.4, 0.9, 1.0),
                key: Vec3::new(-0.5, 0.6, 0.65).normalise(),
                key_colour: Vec3::new(2.4, 2.4, 2.2),
                sky: Vec3::new(0.3, 0.45, 0.8),
                ground: Vec3::new(0.35, 0.18, 0.05),
                softbox: Vec3::ZERO,
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Level)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            outline: Chain::new(vec![Effect::Edges(Edges {
                width: 4.5,
                ..Edges::default()
            })]),
            quantise: ScreenPass::new(Quantise),
        }
    }
}

impl Stage for Retro8bit {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Chip],
            treatment: Treatment::EightBit,
            ambience: Vec::new(),
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
        let outlined = self.outline.run(context, scene, image);
        let uniforms = PaletteUniforms {
            source: outlined,
            unit: scene.extent.height as f32 / 720.0,
        };
        Some(self.quantise.run(context, scene, uniforms))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.0,
            vignette: 0.0,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
