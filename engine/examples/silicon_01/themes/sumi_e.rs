//! A sumi-e scroll: ink-wash mountains dissolving into mist, a still water of broken horizontal
//! strokes, one dry-brush branch, and a single red seal. Everything is ink in four greys on rice
//! paper; the finished frame is bled into the paper and its edges are drawn as brush lines that
//! break where the brush runs dry.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, from_colour, hash2, mix, mix3, rgb, s, smoothstep, splat3, step_lt, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonShader,
    ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::post::{ScreenPass, compress, expand, hdr_of_displayed};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The height of the still water where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.4,
    y: 0.7,
    z: 0.3,
};
const INK: Vec3 = Vec3 {
    x: 0.07,
    y: 0.07,
    z: 0.08,
};
const PAPER: Vec3 = Vec3 {
    x: 0.95,
    y: 0.93,
    z: 0.88,
};
const SEAL: Vec3 = Vec3 {
    x: 0.74,
    y: 0.1,
    z: 0.08,
};

/// Still water round the chip and ranges of steep peaks beyond the near shore, each ridge
/// reaching about as high as the camera can see at its distance.
fn ground(x: f32, z: f32) -> f32 {
    let depth = -z - 10.0;
    let reach = (depth / 30.0).clamp(0.0, 1.0);
    let ridged = 1.0 - (2.0 * fbm_scalar(x * 0.022 + 3.0, z * 0.035, 4) - 1.0).abs();
    FLOOR + reach * reach * (3.0 + 0.3 * depth.min(110.0)) * (0.35 + 0.65 * ridged * ridged)
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(0.9, 0.9, 0.9),
        sky: Vec3::new(0.55, 0.55, 0.55),
        ground: Vec3::new(0.35, 0.35, 0.35),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: hdr_of_displayed(Vec3::new(0.96, 0.95, 0.92)),
            density: 0.0022,
            falloff: 0.015,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.2, 0.2, 0.21),
        die: Vec3::new(0.02, 0.02, 0.025),
        gold: Vec3::new(0.55, 0.55, 0.55),
        streams: Vec3::ZERO,
        key: light.sun,
        key_colour: Vec3::new(1.5, 1.5, 1.5),
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.5, 0.5, 0.5),
    }
}

/// A plum branch coming in from the left edge of the frame, close to the camera: a kinked limb
/// with twigs and buds.
fn branch() -> Shape {
    let wood = Vec3::new(0.05, 0.05, 0.06);
    let mut shape = Shape::new();
    let main = [
        Vec3::new(-2.3, -0.05, 8.0),
        Vec3::new(-1.9, 0.18, 8.0),
        Vec3::new(-1.55, 0.0, 8.0),
        Vec3::new(-1.2, 0.28, 8.0),
        Vec3::new(-0.85, 0.15, 8.0),
    ];
    for (i, pair) in main.windows(2).enumerate() {
        let taper = 1.0 - i as f32 * 0.17;
        shape.append(&Shape::limb(
            pair[0],
            pair[1],
            0.035 * taper,
            0.035 * (taper - 0.17),
            5,
            wood,
        ));
    }
    let twigs = [
        (main[1], Vec3::new(-1.75, 0.4, 8.0)),
        (main[2], Vec3::new(-1.4, -0.2, 8.0)),
        (main[3], Vec3::new(-1.0, 0.46, 8.0)),
        (main[4], Vec3::new(-0.55, 0.38, 8.0)),
        (Vec3::new(-1.75, 0.4, 8.0), Vec3::new(-1.55, 0.52, 8.0)),
    ];
    for (from, to) in twigs {
        shape.append(&Shape::limb(from, to, 0.016, 0.006, 4, wood));
        let bud = |at: Vec3, r: f32| {
            Shape::icosphere(r, 0, 0.0, 1.0, wood * 2.0)
                .transformed(&Mat4::translation(at.x, at.y, at.z))
        };
        shape.append(&bud(to, 0.025));
    }
    shape.with_sway(0.5)
}

/// The still water and the mountains: pale paper with ripples as broken horizontal strokes, and
/// peaks washed in grey, darkest where they face away from the light, with vertical brush texture.
struct WashShader;

impl FragmentShaderSimd<KitVaryings> for WashShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let time = view.frame.time;
        let (x, y, z) = (world.x, world.y, world.z);
        let height = clamp01((y - s(FLOOR)) * s(1.0 / 18.0));

        let rock = fbm(x * s(0.1) + s(time * 0.03), y * s(0.09) + z * s(0.02), 3);
        let strata = fbm(x * s(0.25) + s(5.0), y * s(0.3), 2);
        let wash =
            s(0.04) + height * s(0.25) + (rock - s(0.5)) * s(0.45) + (strata - s(0.5)) * s(0.15);

        let distance = crate::simd::length(splat3(view.eye) - world);
        let warp = fbm(x * s(0.1), z * s(0.4), 2) * s(2.0);
        let line =
            ((z * s(0.55) + warp + (x * s(0.15) + s(time * 0.15)).sin() * s(0.4)).sin()).abs();
        let broken = smoothstep(0.38, 0.6, fbm(x * s(0.2) + s(time * 0.05), z * s(0.5), 3));
        let ripple =
            (s(1.0) - smoothstep(0.0, 0.25, line)) * broken * smoothstep(30.0, 8.0, distance);
        let water = s(0.97) - ripple * s(0.7);

        let slope = smoothstep(0.4, 5.0, y - s(FLOOR));
        let tone = mix3(rgb(1.0, 1.0, 1.0) * water, rgb(1.0, 1.0, 1.0) * wash, slope);

        let to_eye = splat3(view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        scene_colour(
            view,
            u.lighting.shade(view, world, normal, tone, s(1.0)),
            world,
        )
    }
}

/// The paper sky: pale, with a low band of mist the far ridges dissolve into and a faint sun disc.
struct MistShader;

impl FragmentShaderSimd<ScreenVaryings> for MistShader {
    type Uniforms = crate::shaders::ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &crate::shaders::ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let up = ray.y.fast_max(s(0.0));
        let drift = fbm(ray.x * s(5.0) + s(view.frame.time * 0.02), up * s(14.0), 3);
        let haze = smoothstep(0.0, 0.25, up + (drift - s(0.5)) * s(0.1));
        let tone = mix3(rgb(0.96, 0.95, 0.92), rgb(0.86, 0.87, 0.88), haze);
        to_colour(tone, s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
struct FinishUniforms {
    source: Arc<TextureSampler>,
    time: f32,
    aspect: f32,
}

/// Brush and paper: the greys bleed outwards where the paper is wet, settle into four tones, and
/// outlines are drawn as dry-brush strokes. A small red seal is stamped in the lower corner.
struct FinishShader;

#[inline(always)]
fn luminance(c: Vec3Simd) -> F {
    c.x * s(0.2126) + c.y * s(0.7152) + c.z * s(0.0722)
}

impl FragmentShaderSimd<ScreenVaryings> for FinishShader {
    type Uniforms = FinishUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &FinishUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (px, py) = (x * s(u.aspect), y);
        let drift = s(u.time * 0.03);
        let time_phase = u.time * 0.3;

        let wet = fbm(px * s(3.0) + drift, py * s(3.0), 3);
        let reach = s(0.003) + wet * s(0.006);
        let warp_x = (fbm(px * s(7.0) + s(11.0), py * s(7.0) + drift, 3) - s(0.5)) * s(0.006);
        let warp_y = (fbm(px * s(7.0) + s(23.0), py * s(7.0) - drift, 3) - s(0.5)) * s(0.006);
        let (ux, uy) = (x + warp_x, y + warp_y);
        let centre = u.source.sample_linear_clamp_simd([ux, uy]);
        let mut total = compress(from_colour(centre));
        let mut count = s(1.0);
        for (dx, dy) in [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (0.7, 0.7),
            (-0.7, 0.7),
            (0.7, -0.7),
            (-0.7, -0.7),
        ] {
            let tap = u
                .source
                .sample_linear_clamp_simd([ux + reach * s(dx), uy + reach * s(dy * u.aspect)]);
            total = total + compress(from_colour(tap));
            count += s(1.0);
        }
        let tone = luminance(total * count.recip());
        let tone = clamp01((tone - s(0.12)) * s(1.0 / 0.62));
        let tone = tone * tone.sqrt();

        let steps = s(4.0);
        let settled = ((tone * steps).floor()
            + smoothstep(0.35, 0.65, tone * steps - (tone * steps).floor()))
            / steps;
        let density = s(1.0) - mix(tone, settled, s(0.75));

        let streak = fbm(px * s(2.0), py * s(40.0) + s(5.0), 3);
        let dry = smoothstep(0.25, 0.7, streak);
        let ink_amount = clamp01(density * (s(0.95) + dry * s(0.1)));

        let (rx, ry) = (0.0016, 0.0016 * u.aspect);
        let depth = |dx: f32, dy: f32| {
            u.source
                .sample_linear_clamp_simd([ux + s(dx), uy + s(dy)])
                .a
                .fast_min(s(60.0))
        };
        let d = depth(0.0, 0.0);
        let curvature = ((depth(rx, 0.0) + depth(-rx, 0.0) - d - d).abs()
            + (depth(0.0, ry) + depth(0.0, -ry) - d - d).abs())
            / d;
        let line = smoothstep(0.4, 1.0, curvature * s(40.0));
        let breaks = smoothstep(0.35, 0.65, fbm(px * s(70.0), py * s(70.0), 2));
        let stroke = line * breaks * (s(0.5) + dry * s(0.5));

        let across = px * s(1.0 / u.aspect);
        let mut strokes = s(0.0);
        for (k, (row, from, to)) in [(0.3, 0.34, 0.78), (0.215, 0.45, 0.9), (0.15, 0.28, 0.6)]
            .into_iter()
            .enumerate()
        {
            let k = k as f32;
            let sway = (across * s(9.0) + s(k * 2.0 + time_phase)).sin() * s(0.004);
            let thick = s(0.004) + fbm(across * s(3.0) + s(k * 7.0), s(k), 2) * s(0.006);
            let band = s(1.0) - smoothstep(0.5, 1.0, (py - s(row) - sway).abs() / thick);
            let reach = smoothstep(from, from + 0.04, across) * smoothstep(to, to - 0.1, across);
            strokes = strokes.fast_max(band * reach);
        }
        let strokes = strokes * smoothstep(0.35, 0.7, fbm(px * s(40.0), py * s(300.0), 2)) * s(0.5);
        let ink = clamp01(ink_amount + stroke * s(0.8) + strokes);
        let sheet = fbm(px * s(480.0), py * s(480.0), 2);
        let fibre = fbm(px * s(40.0), py * s(300.0), 2);
        let paper = splat3(PAPER) * (s(0.98) + sheet * s(0.03) + fibre * s(0.015));
        let mut colour = mix3(paper, splat3(INK) * (s(0.9) + sheet * s(0.2)), ink);

        let (sx, sy) = (px - s(u.aspect - 0.17), py - s(0.15));
        let size = 0.05;
        let rough = (fbm(px * s(60.0), py * s(60.0), 2) - s(0.5)) * s(0.006);
        let (tx, ty) = (sx * s(1.0 / size), sy * s(1.0 / size));
        let reach_t = tx.abs().fast_max(ty.abs());
        let inside = step_lt(reach_t + rough * s(1.0 / size), s(1.0));
        let (cx, cy) = (
            ((tx * s(0.5) + s(0.5)) * s(4.0)).floor(),
            ((ty * s(0.5) + s(0.5)) * s(4.0)).floor(),
        );
        let carved = step_lt(s(0.5), hash2(cx + s(3.0), cy + s(8.0))) * step_lt(reach_t, s(0.82));
        let pad = splat3(SEAL) * (s(0.92) + sheet * s(0.16));
        colour = mix3(colour, pad, inside * (s(1.0) - carved * s(0.9)) * s(0.95));

        to_colour(expand(colour), centre.a)
    }
}

pub struct SumiE {
    land: Shape,
    branch: Shape,
    wash: SimdPipeline<KitVertexShader, WashShader>,
    ink: SimdPipeline<KitVertexShader, ToonShader>,
    sky: SimdPipeline<ScreenVertexShader, MistShader>,
    finish: ScreenPass<FinishShader>,
}

impl SumiE {
    pub fn new() -> Self {
        Self {
            land: terrain(&ring_radii(1.5, 300.0, 56), 112, ground, |_, _| Vec3::ONE),
            branch: branch(),
            wash: pipeline(WashShader, CullingMode::BackFace),
            ink: pipeline(ToonShader, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, MistShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            finish: ScreenPass::new(FinishShader),
        }
    }
}

impl Stage for SumiE {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Harp],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::BrushSwish),
                0.0,
                0.45,
                0.17,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let breeze = Motion::Sway(Sway {
            amplitude: 0.06,
            speed: 0.7,
            wind: Vec2::new(1.0, 0.3),
        });

        self.land
            .draw(pass, &self.wash, motion(Motion::Still), toon.clone());
        self.branch.draw(pass, &self.ink, motion(breeze), toon);
        scene.chip.draw(pass, view, &look());
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| view.clone());
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let aspect = scene.extent.width as f32 / scene.extent.height as f32;
        Some(self.finish.run(
            context,
            scene,
            FinishUniforms {
                source: image,
                time: scene.local_time,
                aspect,
            },
        ))
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
