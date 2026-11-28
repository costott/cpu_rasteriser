//! halftone: a comic panel. A sunburst and speed lines behind the chip, a skyline and a street
//! in flat colours, all printed as CMYK dots with bold ink outlines.

use std::f32::consts::TAU;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, fract, hash2, mix, mix3, s, select3, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVertexShader, Motion, ToonShader, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::post::{Chain, Edges, Effect, Halftone, hdr_of_displayed};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -3.4;
const INK: Vec3 = Vec3 {
    x: 0.02,
    y: 0.02,
    z: 0.04,
};

/// Comic colours as printed, 0 to 1.
fn flat(r: f32, g: f32, b: f32) -> Vec3 {
    hdr_of_displayed(Vec3::new(r, g, b))
}

pub struct ComicHalftone {
    look: Look,
    skyline: Shape,
    skyline_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    frame: SimdPipeline<ScreenVertexShader, Frame>,
    chain: Chain,
}

impl ComicHalftone {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: flat(0.1, 0.35, 0.95) * 0.6,
                die: flat(0.95, 0.2, 0.15) * 0.6,
                gold: flat(1.0, 0.8, 0.1),
                streams: Vec3::new(1.4, 1.4, 1.2),
                key_colour: Vec3::new(1.8, 1.7, 1.5),
                sky: Vec3::new(0.6, 0.7, 0.9),
                ground: Vec3::new(0.35, 0.25, 0.25),
                ..Look::default()
            },
            skyline: skyline(),
            skyline_pipeline: pipeline(ToonShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            frame: SimdPipeline::new(ScreenVertexShader, Frame)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DISABLED)
                .with_blend_state(BlendState::ALPHA_BLEND),
            chain: Chain::new(vec![
                Effect::Toon {
                    levels: 4.0,
                    softness: 0.08,
                },
                Effect::Halftone(Halftone {
                    cell: 6.5,
                    ..Halftone::default()
                }),
                Effect::Edges(Edges {
                    ink: INK,
                    width: 1.8,
                    sensitivity: 1100.0,
                    ..Edges::default()
                }),
            ]),
        }
    }
}

/// Towers in flat comic colours with a grid of lit windows, a few streets back.
fn skyline() -> Shape {
    let mut rng = Rng::new(35);
    let palette = [
        flat(0.95, 0.25, 0.2),
        flat(1.0, 0.8, 0.15),
        flat(0.2, 0.5, 0.95),
        flat(0.6, 0.3, 0.8),
        flat(0.98, 0.55, 0.15),
    ];
    let window = flat(1.0, 0.95, 0.5);
    let mut city = Shape::new();
    let mut x = -52.0;
    while x < 52.0 {
        let width = rng.range(3.5, 7.0);
        let height = rng.range(7.0, 24.0);
        let z = rng.range(-70.0, -48.0);
        let colour = palette[rng.index(palette.len())];
        let centre = x + width * 0.5;
        city.append(
            &Shape::cuboid(
                Vec3::new(-width * 0.5, 0.0, -width * 0.5),
                Vec3::new(width * 0.5, height, width * 0.5),
                colour,
            )
            .transformed(&Mat4::translation(centre, FLOOR, z)),
        );
        let (columns, rows) = ((width / 1.4) as i32, (height / 2.0) as i32);
        for column in 0..columns {
            for row in 1..rows {
                if rng.unit() < 0.45 {
                    continue;
                }
                let wx = centre - width * 0.5 + (column as f32 + 0.5) * width / columns as f32;
                let wy = FLOOR + row as f32 * 2.0;
                let front = z + width * 0.5 + 0.05;
                let quad = [
                    Vec3::new(wx - 0.4, wy, front),
                    Vec3::new(wx + 0.4, wy, front),
                    Vec3::new(wx + 0.4, wy + 0.9, front),
                    Vec3::new(wx - 0.4, wy + 0.9, front),
                ];
                city.quad(quad, window);
            }
        }
        x += width + rng.range(0.4, 2.5);
    }
    city
}

impl Stage for ComicHalftone {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Trumpet],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::ComicPop),
                0.0,
                0.45,
                0.14,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        scene.chip.draw(pass, view, &self.look);
        self.skyline.draw(
            pass,
            &self.skyline_pipeline,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Still,
            },
            ToonUniforms {
                view: view.clone(),
                lighting: lighting(),
            },
        );
        let burst = view.view_projection * Vec4::new(0.0, 0.0, 0.0, 1.0);
        let uniforms = BackdropUniforms {
            view: view.clone(),
            time: scene.local_time,
            centre: (burst.x / burst.w, burst.y / burst.w),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.frame, (), |_| uniforms.clone());
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        Some(self.chain.run(context, scene, image))
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

fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(-0.4, 0.7, 0.6).normalise(),
        sun_colour: Vec3::new(1.1, 1.05, 0.95),
        sky: Vec3::new(0.55, 0.6, 0.7),
        ground: Vec3::new(0.3, 0.25, 0.3),
        bands: 3.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: flat(0.75, 0.9, 1.0),
            density: 0.004,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
    time: f32,
    /// Where the chip is on screen, in device coordinates: the burst's centre.
    centre: (f32, f32),
}

/// A jagged star's radius at `angle`: `spikes` points reaching `amplitude` beyond `radius`.
#[inline(always)]
fn star(angle: F, radius: f32, amplitude: f32, spikes: f32, turn: f32) -> F {
    let wave = fract((angle + s(turn)) * s(spikes / TAU));
    s(radius) * (s(1.0) + s(amplitude) * (s(1.0) - (wave * s(2.0) - s(1.0)).abs()))
}

/// How much of a pixel lies within `half_width` of the edge where `inside` crosses zero.
#[inline(always)]
fn outline(inside: F, half_width: f32, pixel: F) -> F {
    clamp01((s(half_width) - inside.abs()) / pixel + s(0.5))
}

/// A filled shape with an ink outline over `base`.
#[inline(always)]
fn fill(base: Vec3Simd, inside: F, colour: Vec3, pixel: F, line: f32) -> Vec3Simd {
    let covered = clamp01(inside / pixel + s(0.5));
    let filled = mix3(base, splat3(colour), covered);
    mix3(filled, splat3(INK), outline(inside, line, pixel))
}

struct Backdrop;

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(view.aspect), ny);
        let pixel = s(view.pixel_angle / view.tan_half_fov);
        let (dx, dy) = (x - s(u.centre.0 * view.aspect), y - s(u.centre.1));
        let radius = (dx * dx + dy * dy).sqrt();
        let angle = dy.atan2(dx);
        let turn = u.time * 0.25;

        // Sky, brightening to the horizon, with alternating wedges of light radiating from the
        // burst and speed lines beyond it.
        let height = clamp01((y + s(0.1)) * s(0.85));
        let mut colour = mix3(
            splat3(flat(0.82, 0.95, 1.0)),
            splat3(flat(0.15, 0.6, 0.95)),
            height * height,
        );
        let wedge = fract((angle + s(turn * 0.3)) * s(18.0 / TAU));
        let rays = wedge.simd_lt(s(0.5)).select(s(1.0), s(0.0)) * smoothstep(0.5, 1.2, radius);
        colour = mix3(
            colour,
            colour * s(1.12) + splat3(flat(0.1, 0.1, 0.1)),
            rays * s(0.7),
        );
        let line_id = (angle * s(110.0 / TAU)).floor();
        let across = (fract(angle * s(110.0 / TAU)) - s(0.5)).abs();
        let streak = hash2(line_id, s(7.1))
            .simd_gt(s(0.62))
            .select(s(1.0), s(0.0))
            * (across - s(0.1)).simd_lt(s(0.0)).select(s(1.0), s(0.0))
            * smoothstep(0.95, 1.1, radius + hash2(line_id, s(3.3)) * s(0.35));
        colour = mix3(colour, splat3(flat(0.1, 0.25, 0.6)), streak * s(0.85));

        // Clouds: outlined puffs.
        for (cx, cy, scale) in [(-1.35, 0.62, 1.0f32), (1.25, 0.72, 0.8), (1.55, 0.2, 0.55)] {
            let puff = |ox: f32, oy: f32, r: f32| {
                let (px, py) = (x - s(cx + ox * scale), y - s(cy + oy * scale));
                s(r * scale) - (px * px + py * py).sqrt()
            };
            let inside = puff(0.0, 0.0, 0.17)
                .fast_max(puff(0.18, -0.03, 0.14))
                .fast_max(puff(-0.17, -0.04, 0.13))
                .fast_max(puff(0.06, 0.1, 0.12));
            colour = fill(colour, inside, flat(1.0, 1.0, 1.0), pixel, 0.007);
        }

        // The burst: a yellow star with an orange heart, turning slowly.
        let outer = star(angle, 0.7, 0.2, 14.0, turn) - radius;
        colour = fill(colour, outer, flat(1.0, 0.87, 0.1), pixel, 0.009);
        let inner = star(angle, 0.5, 0.16, 9.0, -turn * 1.4) - radius;
        colour = fill(colour, inner, flat(1.0, 0.4, 0.12), pixel, 0.008);

        // The street: asphalt with a dashed centre line and kerbs, in perspective.
        let ray = view_ray(view, nx, ny);
        let below = ray.y.simd_lt(s(-0.002));
        let t = (s(FLOOR - view.eye.y) / ray.y.fast_min(s(-0.002))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let dash = fract(wz * s(1.0 / 8.0))
            .simd_lt(s(0.5))
            .select(s(1.0), s(0.0))
            * (wx.abs() - s(0.45)).simd_lt(s(0.0)).select(s(1.0), s(0.0));
        let kerb = (wx.abs() - s(12.0))
            .abs()
            .simd_lt(s(0.7))
            .select(s(1.0), s(0.0));
        let pavement = (wx.abs() - s(12.0)).simd_gt(s(0.7)).select(s(1.0), s(0.0));
        let mut street = splat3(flat(0.2, 0.22, 0.38));
        street = mix3(street, splat3(flat(0.95, 0.95, 0.9)), dash);
        street = mix3(street, splat3(flat(1.0, 0.8, 0.15)), kerb);
        street = mix3(street, splat3(flat(0.62, 0.6, 0.72)), pavement);
        let haze = (-t * s(0.012)).exp();
        street = mix3(
            splat3(flat(0.75, 0.9, 1.0)),
            street,
            mix(s(0.4), s(1.0), haze),
        );
        colour = select3(below, street, colour);
        to_colour(colour, s(SKY_DISTANCE))
    }
}

/// The panel's border, drawn over everything: the page's paper outside a thick ink frame.
struct Frame;

impl FragmentShaderSimd<ScreenVaryings> for Frame {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let (x, y) = (nx.abs() * s(view.aspect), ny.abs());
        let pixel = s(view.pixel_angle / view.tan_half_fov);
        let inside = (s(view.aspect - 0.06) - x).fast_min(s(0.93) - y);
        let margin = clamp01(s(0.5) - inside / pixel);
        let frame = outline(inside, 0.012, pixel);
        let paper = splat3(flat(1.0, 0.97, 0.88));
        let colour = mix3(
            splat3(INK),
            paper,
            clamp01(s(0.5) - (inside + s(0.012)) / pixel),
        );
        let coverage = margin.fast_max(frame);
        ColourSimd {
            r: colour.x,
            g: colour.y,
            b: colour.z,
            a: coverage,
        }
    }
}
