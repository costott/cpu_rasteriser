//! xray: a radiograph. The world is a scan of something vast: a spine receding across the
//! ground with a cage of ribs arching over it, drawn as translucent density that adds up where
//! bones overlap. The chip sits opaque in the film's centre with its traces glowing through the
//! package. The stylisation maps density to the cool blue of film on a light box, under a
//! scanner's rulers and a sweeping beam.

use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, fract, length, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion,
};
use crate::themes::kit::noise;
use crate::themes::kit::post::{Chain, Edges, Effect, Noise, Ramp};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::overlay::{Align, Line, Overlay, Text};
use crate::themes::{Scene, Stage};

/// The height of the film's ground plane, below the chip.
const GROUND: f32 = -3.4;
/// How quickly bone density thins with distance from the camera.
const ATTENUATION: f32 = 0.018;
/// The distance between vertebrae, the first vertebra's depth and how many there are.
const SPACING: f32 = 1.7;
const FIRST: f32 = -9.0;
const VERTEBRAE: usize = 34;

pub struct Xray {
    look: Look,
    bones: Shape,
    bone_pipeline: SimdPipeline<KitVertexShader, BoneShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    overlay: Overlay,
    chain: Chain,
}

impl Xray {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: density(0.3),
                die: density(0.35),
                gold: density(1.6),
                streams: density(1.4),
                key_colour: density(0.5),
                sky: density(0.2),
                ground: density(0.06),
                softbox: Vec3::ZERO,
                ..Look::default()
            },
            bones: skeleton(),
            bone_pipeline: SimdPipeline::new(KitVertexShader, BoneShader)
                .with_culling_mode(CullingMode::None)
                .with_blend_state(BlendState::ADDITIVE)
                .with_depth_state(DepthState::READ_ONLY),
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            overlay: Overlay::new(),
            chain: Chain::new(vec![
                Effect::Edges(Edges {
                    ink: density(1.0),
                    width: 0.9,
                    opacity: 0.8,
                    ..Edges::default()
                }),
                Effect::Ramp(Ramp::new(
                    &[
                        Vec3::new(0.0, 0.006, 0.03),
                        Vec3::new(0.01, 0.07, 0.3),
                        Vec3::new(0.1, 0.4, 0.95),
                        Vec3::new(0.55, 0.9, 1.4),
                        Vec3::new(1.7, 2.0, 2.3),
                    ],
                    1.5,
                    0,
                )),
                Effect::Noise(Noise {
                    amount: 0.06,
                    floor: 0.004,
                    cell: 1.5,
                    chroma: 0.0,
                }),
            ]),
        }
    }
}

/// A grey of density `d`: the scene is drawn in luminance and the ramp colours it.
fn density(d: f32) -> Vec3 {
    Vec3::new(d, d, d)
}

/// A spine along the ground into the distance, a rib arching out of every other vertebra.
fn skeleton() -> Shape {
    let mut rng = Rng::new(48);
    let mut bones = Shape::new();
    let body = Shape::lathe(
        &[
            (0.0, 0.0),
            (0.9, 0.0),
            (1.15, 0.22),
            (0.95, 0.5),
            (1.15, 0.78),
            (0.9, 1.0),
            (0.0, 1.0),
        ],
        10,
        density(0.5),
    );
    for i in 0..VERTEBRAE {
        let z = FIRST - i as f32 * SPACING;
        let mut vertebra = Shape::new();
        let size = rng.range(0.92, 1.08);
        let lift = Mat4::translation(0.0, GROUND + 1.5, z)
            * Mat4::rotate_x(FRAC_PI_2)
            * Mat4::translation(0.0, -0.5, 0.0);
        bones.append(&body.transformed(&(lift * Mat4::scaling(size, 1.0, size))));
        vertebra.append(&Shape::limb(
            Vec3::new(0.0, GROUND + 2.3, z),
            Vec3::new(0.0, GROUND + 4.4, z - 0.9),
            0.34,
            0.1,
            5,
            density(0.42),
        ));
        for side in [-1.0f32, 1.0] {
            vertebra.append(&Shape::limb(
                Vec3::new(side * 0.9, GROUND + 1.6, z),
                Vec3::new(side * 2.5, GROUND + 1.9, z + 0.25),
                0.3,
                0.13,
                5,
                density(0.4),
            ));
            if i % 3 == 0 {
                vertebra.append(&rib(side, z, rng.range(0.85, 1.15)));
            }
        }
        bones.append(&vertebra.transformed(&Mat4::translation(bend(z), 0.0, 0.0)));
    }
    bones
}

/// How far the spine has swung sideways at depth `z`.
fn bend(z: f32) -> f32 {
    7.0 * ((FIRST - z) * 0.07).sin()
}

/// A rib springing from a vertebra's transverse process, out and up in an arch, narrowing to a tip.
fn rib(side: f32, z: f32, size: f32) -> Shape {
    let steps = 14;
    let point = |i: usize| {
        let a = i as f32 / steps as f32 * 0.78 * PI;
        Vec3::new(
            side * (2.5 + 9.5 * size * a.sin()),
            GROUND + 1.9 + 8.4 * size * (1.0 - a.cos()),
            z + 0.25 - 1.6 * (i as f32 / steps as f32),
        )
    };
    let taper = |i: usize| 0.32 * (1.0 - i as f32 / steps as f32 * 0.8);
    let mut rib = Shape::new();
    for i in 0..steps {
        rib.append(&Shape::limb(
            point(i),
            point(i + 1),
            taper(i),
            taper(i + 1),
            6,
            density(0.36),
        ));
    }
    rib
}

impl Stage for Xray {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Thin],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::BuzzClick),
                0.0,
                0.45,
                0.1,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let window = (scene.extent.width, scene.extent.height);
        let pixel = 2.6 * window.1 as f32 / 720.0;
        let colour = Vec3::new(1.1, 1.3, 1.5);
        let dose = format!(
            "DOSE {:5.2} MGY",
            1.2 + 0.05 * (scene.local_time * 2.3).sin()
        );
        let mut text = Text::default();
        for (words, x, y, align) in [
            ("RADIOGRAPH  PA VIEW", -0.8, 0.84, Align::Left),
            ("120 KVP  200 MA", -0.8, 0.76, Align::Left),
            (dose.as_str(), -0.8, -0.82, Align::Left),
            ("SILICON-01", 0.8, -0.82, Align::Right),
        ] {
            text.line(
                Line {
                    text: words,
                    x,
                    y,
                    align,
                    colour,
                },
                pixel,
                window,
            );
        }
        self.overlay.set(text);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        scene.chip.draw(pass, view, &self.look);
        let uniforms = BackdropUniforms {
            view: view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
        self.bones.draw(
            pass,
            &self.bone_pipeline,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Still,
            },
            BoneUniforms { view: view.clone() },
        );
        self.overlay.draw(pass);
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
            bloom: 0.6,
            vignette: 0.4,
            grain: 0.01,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[derive(Clone)]
struct BoneUniforms {
    view: ViewRef,
}

/// Bone as translucent density: thickest through the middle of a limb, fading with distance,
/// with a faint bright rim. It adds to what is behind it and leaves the scene's distance alone.
struct BoneShader;

impl FragmentShaderSimd<KitVaryings> for BoneShader {
    type Uniforms = BoneUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &BoneUniforms) -> ColourSimd {
        let to_eye = splat3(u.view.eye) - v.world();
        let distance = length(to_eye);
        let facing = (v.normal().normalise().dot(to_eye) / distance).abs();
        let thickness = s(0.3) + s(0.7) * facing.sqrt();
        let rim = (s(1.0) - facing) * (s(1.0) - facing) * s(0.5);
        let fade = (-distance * s(ATTENUATION)).exp();
        let value = v.colour().x * (thickness + rim) * fade * s(0.26);
        to_colour(rgb(1.0, 1.0, 1.0) * value, s(0.0))
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
    time: f32,
}

/// The film: soft tissue haze, a light-box glow round the chip, the scanner's rulers and a beam
/// that sweeps up the frame.
struct Backdrop;

/// Distance from `p` to the nearest whole number.
#[inline(always)]
fn to_whole(p: F) -> F {
    let f = fract(p);
    f.fast_min(s(1.0) - f)
}

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let x = nx * s(view.aspect);
        let ray = view_ray(view, nx, ny);

        let below = ray.y.simd_lt(s(-0.002));
        let t = (s(GROUND - view.eye.y) / ray.y.fast_min(s(-0.002))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let tissue = noise::fbm(wx * s(0.07) + s(u.time * 0.05), wz * s(0.07), 4);
        let fade = (-t * s(0.014)).exp();
        let ground = (s(0.03) + tissue * tissue * s(0.32)) * fade + s(0.075) * (s(1.0) - fade);
        let haze = smoothstep(0.5, -0.1, ny);
        let air = s(0.015)
            + haze * s(0.07)
            + noise::fbm(x * s(2.0), ny * s(2.0) + s(u.time * 0.04), 3) * s(0.03);
        let mut value = below.select(ground, air);

        value += (-(x * x * s(0.9) + ny * ny * s(1.6))).exp() * s(0.1);

        let distance = ny - (fract(s(u.time * 0.22)) * s(2.6) - s(1.3));
        value += (-(distance * distance) * s(9000.0)).exp() * s(0.35)
            + smoothstep(0.0, -0.5, distance) * s(0.04);

        let on_ruler = x.abs().simd_gt(s(view.aspect * 0.93));
        let minor = s(1.0) - smoothstep(0.0, 0.05, to_whole(ny * s(20.0)));
        let major = s(1.0) - smoothstep(0.0, 0.012, to_whole(ny * s(4.0)));
        value = on_ruler.select(value + (minor * s(0.5)).fast_max(major * s(0.9)), value);

        to_colour(rgb(1.0, 1.0, 1.0) * value, s(SKY_DISTANCE))
    }
}
