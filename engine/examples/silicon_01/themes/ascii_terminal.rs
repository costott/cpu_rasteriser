//! ascii-terminal: a moonlit sea drawn in amber characters on a phosphor terminal. The scene is
//! rendered in grey, each 12x24-pixel cell is averaged and replaced by the glyph of matching
//! density, and silhouettes and creases become line glyphs that follow their direction.

use std::f32::consts::PI;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{
    F, clamp01, from_colour, hash2, mix, pow2n, s, select, smoothstep, splat3, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::noise::fbm;
use crate::themes::kit::post::{Chain, Crt, Effect, Noise, ScreenPass, compress, expand};
use crate::themes::kit::sky::stars;
use crate::themes::{Scene, Stage};

/// The height of the sea below the chip.
const SEA: f32 = -3.0;
/// The moon's centre on the sky plane one unit ahead, and its radius.
const MOON: (f32, f32, f32) = (0.3, 0.17, 0.035);
/// A character cell's size in pixels of a 720-line frame.
const CELL: (f32, f32) = (12.0, 24.0);
/// The edge strength, as a fraction of the surface's distance, above which a cell is an outline.
const EDGE: f32 = 0.1;
const DISTANCE_CAP: f32 = 60.0;
const AMBER: Vec3 = Vec3 {
    x: 1.0,
    y: 0.58,
    z: 0.08,
};

/// 5x7 glyphs, one row per byte from the top with the leftmost pixel in bit 4: ten of rising
/// density from a space to `@`, then the outline glyphs `/`, `|` and `\`.
const GLYPHS: [[u8; 7]; 13] = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0b01100, 0b01100],
    [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0],
    [0, 0, 0, 0b11111, 0, 0, 0],
    [0, 0, 0b11111, 0, 0b11111, 0, 0],
    [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
    [0, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0],
    [
        0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010,
    ],
    [
        0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011,
    ],
    [
        0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110,
    ],
    [
        0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
    ],
    [
        0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
    ],
    [
        0b10000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00010, 0b00001,
    ],
];
/// The glyphs for an outline running along, in order: -, /, | and \.
const OUTLINES: [usize; 4] = [3, 10, 11, 12];

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Moonlight;

impl FragmentShaderSimd<ScreenVaryings> for Moonlight {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let time = u.time;
        let ahead = (-ray.z).fast_max(s(0.05));
        let (sx, sy) = (ray.x / ahead, ray.y / ahead);
        let (mx, my, radius) = MOON;
        let (dx, dy) = (sx - s(mx), sy - s(my));
        let r = (dx * dx + dy * dy).sqrt();

        let cloud = smoothstep(0.52, 0.8, fbm(sx * s(3.0) + s(time * 0.03), sy * s(7.0), 4))
            * smoothstep(0.0, 0.12, sy)
            * (s(1.0) - smoothstep(0.3, 0.5, sy));
        let craters = fbm(dx * s(70.0), dy * s(70.0), 3) - s(0.5);
        let disc = s(1.0) - smoothstep(radius * 0.97, radius * 1.03, r);
        let moon = disc * (s(0.5) + craters * s(0.3) - (r / s(radius)) * s(0.15));
        let glow = (-r * s(9.0)).exp() * s(0.3);
        let night = mix(s(0.14), s(0.02), smoothstep(0.0, 0.5, sy));
        let twinkle = stars(ray.x, ray.y, ray.z, 0.5, s(time)) * smoothstep(0.05, 0.3, ray.y);
        let sky =
            night + glow + moon * (s(1.0) - cloud * s(0.6)) + cloud * s(0.14) + twinkle * s(0.9);

        let t = (s(SEA - view.eye.y) / ray.y.fast_min(s(-0.001))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let (mut slope_x, mut slope_z) = (s(0.0), s(0.0));
        for (fx, fz, amplitude, speed) in [
            (0.9, 0.4, 0.05, 1.3),
            (1.7, -0.9, 0.03, 1.9),
            (0.5, 1.3, 0.04, 1.1),
            (3.1, 1.7, 0.015, 2.7),
        ] {
            let crest = (wx * s(fx) + wz * s(fz) + s(time * speed)).cos() * s(amplitude);
            slope_x += crest * s(fx);
            slope_z += crest * s(fz);
        }
        let calm = (-t * s(0.018)).exp();
        let normal = Vec3Simd::new(-slope_x * calm, s(1.0), -slope_z * calm).normalise();
        let incident = ray - normal * (s(2.0) * ray.dot(normal));
        let towards_moon = Vec3Simd::new(s(mx), s(my), s(-1.0)).normalise();
        let glitter = incident.dot(towards_moon).fast_max(s(0.0));
        let mirror = (s(1.0) - ray.y.abs() * s(4.0)).fast_max(s(0.0));
        let ripple = (slope_x + slope_z).abs() * calm * s(5.0);
        let sea = s(0.03)
            + mirror * mirror * s(0.12)
            + ripple.fast_min(s(0.2)) * (s(0.4) + mirror)
            + pow2n(glitter, 16) * s(0.45) * calm.sqrt();

        let below = ray.y.simd_lt(s(-0.001));
        let tone = select(below, sea, sky).fast_min(s(1.0));
        to_colour(expand(splat3(Vec3::ONE) * tone), s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
struct TextUniforms {
    source: Arc<TextureSampler>,
    unit: f32,
    time: f32,
}

/// Replaces each character cell with the glyph whose density matches the cell's brightness, or
/// the outline glyph along an edge in the surface's distance.
struct Characters;

impl FragmentShaderSimd<ScreenVaryings> for Characters {
    type Uniforms = TextUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &TextUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (width, height) = (u.source.width() as f32, u.source.height() as f32);
        let (cell_w, cell_h) = (CELL.0 * u.unit, CELL.1 * u.unit);
        let (px, py) = (x * s(width), y * s(height));
        let (cx, cy) = ((px / s(cell_w)).floor(), (py / s(cell_h)).floor());
        let across = px / s(cell_w) - cx;
        let down = s(1.0) - (py / s(cell_h) - cy);
        let (centre_x, centre_y) = (
            (cx + s(0.5)) * s(cell_w / width),
            (cy + s(0.5)) * s(cell_h / height),
        );
        let tap = |dx: f32, dy: f32| {
            u.source.sample_linear_clamp_simd([
                centre_x + s(dx * cell_w / width),
                centre_y + s(dy * cell_h / height),
            ])
        };

        let mut brightness = s(0.0);
        for (ox, oy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
            let c = compress(from_colour(tap(ox, oy)));
            brightness += (c.x * s(0.2126) + c.y * s(0.7152) + c.z * s(0.0722)) * s(0.25);
        }
        let flicker = (hash2(cx, cy + s((u.time * 9.0).floor() * 7.0)) - s(0.5)) * s(0.06);
        let tone = clamp01(brightness * s(1.25) + flicker * clamp01(brightness * s(8.0)));

        let distance = |dx: f32, dy: f32| tap(dx, dy).a.fast_min(s(DISTANCE_CAP));
        let here = distance(0.0, 0.0);
        let gradient_x = distance(1.0, 0.0) - distance(-1.0, 0.0);
        let gradient_y = distance(0.0, 1.0) - distance(0.0, -1.0);
        let strength = (gradient_x * gradient_x + gradient_y * gradient_y).sqrt() / here;
        let tangent = gradient_x.atan2(-gradient_y);
        let along = tangent - (tangent / s(PI)).floor() * s(PI);

        let (tone, strength, along) = (tone.to_array(), strength.to_array(), along.to_array());
        let (across, down) = (across.to_array(), down.to_array());
        let mut lit = [0.0f32; 8];
        let mut weight = [0.0f32; 8];
        for lane in 0..8 {
            let outline = strength[lane] > EDGE;
            let glyph = if outline {
                OUTLINES[(along[lane] / (PI * 0.25) + 0.5) as usize % 4]
            } else {
                ((tone[lane].powf(0.8) * 9.99) as usize).min(9)
            };
            let column = (across[lane] - 1.0 / 12.0) / (10.0 / 12.0) * 5.0;
            let row = (down[lane] - 1.5 / 24.0) / (21.0 / 24.0) * 7.0;
            if (0.0..5.0).contains(&column) && (0.0..7.0).contains(&row) {
                let bit = GLYPHS[glyph][row as usize] >> (4 - column as usize) & 1;
                lit[lane] = bit as f32;
            }
            weight[lane] = if outline {
                1.3
            } else {
                0.35 + 0.65 * tone[lane]
            };
        }
        let (lit, weight) = (F::from(lit), F::from(weight));
        let phosphor = splat3(AMBER) * (lit * weight + s(0.012) + F::from(tone) * s(0.02));
        to_colour(expand(phosphor), s(SKY_DISTANCE))
    }
}

pub struct AsciiTerminal {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Moonlight>,
    characters: ScreenPass<Characters>,
    glass: Chain,
}

impl AsciiTerminal {
    pub fn new() -> Self {
        Self {
            look: Look {
                key_colour: Vec3::new(3.4, 3.4, 3.3),
                ..Look::default()
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Moonlight)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            characters: ScreenPass::new(Characters),
            glass: Chain::new(vec![
                Effect::Crt(Crt {
                    curvature: 0.05,
                    scanlines: 0.3,
                    pitch: 3.0,
                    mask: 0.12,
                    convergence: 0.5,
                    flicker: 0.03,
                }),
                Effect::Noise(Noise {
                    amount: 0.06,
                    floor: 0.004,
                    cell: 1.0,
                    chroma: 0.0,
                }),
            ]),
        }
    }
}

impl Stage for AsciiTerminal {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Chip],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Keystrokes),
                0.0,
                0.45,
                0.14,
                0.3,
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
        let uniforms = TextUniforms {
            source: image,
            unit: scene.extent.height as f32 / 720.0,
            time: scene.local_time,
        };
        let text = self.characters.run(context, scene, uniforms);
        Some(self.glass.run(context, scene, text))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.9,
            vignette: 0.35,
            grain: 0.008,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
