//! pencil: a graphite sketch on textured paper. A landscape of simple solids is hatched by tone
//! along the surfaces' form and outlined with wandering, doubled lines; smudges darken the
//! corners and clouds are shaded in loosely.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::SKY_DISTANCE;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::simd::{clamp01, rgb, s, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVertexShader, Motion, ToonShader, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::noise;
use crate::themes::kit::post::{
    Chain, Edges, Effect, Hatching, Noise, Paper, expand, hdr_of_displayed,
};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -3.4;
const PAPER: Vec3 = Vec3 {
    x: 0.93,
    y: 0.9,
    z: 0.82,
};

pub struct Pencil {
    look: Look,
    landscape: Shape,
    landscape_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    chain: Chain,
}

impl Pencil {
    pub fn new() -> Self {
        let grey = |tone: f32| Vec3::new(tone, tone, tone);
        Self {
            look: Look {
                substrate: grey(0.38),
                die: grey(0.2),
                gold: grey(0.35),
                streams: grey(0.5),
                key: Vec3::new(-0.6, 0.55, 0.55).normalise(),
                key_colour: grey(1.9),
                sky: grey(0.35),
                ground: grey(0.12),
                ..Look::default()
            },
            landscape: landscape(),
            landscape_pipeline: pipeline(ToonShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            chain: Chain::new(vec![
                Effect::Hatching(Hatching {
                    ink: Vec3::new(0.2, 0.2, 0.22),
                    paper: PAPER,
                    spacing: 4.6,
                    width: 1.9,
                    angle: 0.7,
                    follow: 0.8,
                    layers: 4,
                    wobble: 1.2,
                    keep_image: 0.0,
                }),
                Effect::Edges(Edges {
                    ink: hdr_of_displayed(Vec3::new(0.16, 0.16, 0.18)),
                    width: 1.0,
                    sensitivity: 1200.0,
                    opacity: 0.9,
                    wobble: 1.4,
                    boil: 6.0,
                    ..Edges::default()
                }),
                Effect::Edges(Edges {
                    ink: hdr_of_displayed(Vec3::new(0.25, 0.25, 0.27)),
                    width: 0.8,
                    sensitivity: 1200.0,
                    opacity: 0.5,
                    wobble: 2.4,
                    boil: 5.0,
                    ..Edges::default()
                }),
                Effect::Paper(Paper {
                    tint: Vec3::ONE,
                    grain: 0.4,
                    fibre: 0.35,
                    scale: 2.2,
                }),
                Effect::Noise(Noise {
                    amount: 0.03,
                    floor: 0.0,
                    cell: 1.0,
                    chroma: 0.0,
                }),
            ]),
        }
    }
}

/// The land: ground, far hills, conifers, round trees, rocks and a cottage, in grey albedos
/// whose tones the hatching turns into strokes.
fn landscape() -> Shape {
    let mut rng = Rng::new(42);
    let grey = |tone: f32| Vec3::new(tone, tone, tone);
    let mut land = Shape::new();
    land.append(
        &Shape::cuboid(
            Vec3::new(-400.0, -0.5, -400.0),
            Vec3::new(400.0, 0.0, 8.0),
            grey(0.1),
        )
        .transformed(&Mat4::translation(0.0, FLOOR, 0.0)),
    );
    for (x, z, radius, flat, tone) in [
        (-70.0, -150.0, 70.0, 0.22, 0.62),
        (60.0, -170.0, 85.0, 0.2, 0.55),
        (-10.0, -230.0, 110.0, 0.15, 0.7),
        (120.0, -120.0, 50.0, 0.28, 0.5),
    ] {
        let hill = Shape::icosphere(radius, 2, 0.06, x, grey(tone));
        let place = Mat4::translation(x, FLOOR, z) * Mat4::scaling(1.0, flat, 1.0);
        land.append(&hill.transformed(&place));
    }
    let tree =
        |land: &mut Shape, x: f32, z: f32, scale: f32, round: bool| {
            let trunk = Shape::cylinder(0.3, 0.22, 1.6, 6, grey(0.16));
            let crown = if round {
                Shape::icosphere(2.1, 1, 0.18, x + z, grey(0.28))
                    .transformed(&Mat4::translation(0.0, 3.4, 0.0))
            } else {
                let mut cone = Shape::cone(1.9, 3.6, 7, grey(0.2))
                    .transformed(&Mat4::translation(0.0, 1.3, 0.0));
                cone.append(
                    &Shape::cone(1.4, 3.0, 7, grey(0.24))
                        .transformed(&Mat4::translation(0.0, 3.2, 0.0)),
                );
                cone.append(
                    &Shape::cone(0.9, 2.4, 7, grey(0.28))
                        .transformed(&Mat4::translation(0.0, 5.0, 0.0)),
                );
                cone
            };
            let mut whole = trunk;
            whole.append(&crown);
            land.append(&whole.transformed(
                &(Mat4::translation(x, FLOOR, z) * Mat4::scaling(scale, scale, scale)),
            ));
        };
    for _ in 0..34 {
        let x = rng.range(-55.0, 55.0);
        let z = rng.range(-90.0, -12.0);
        if x.abs() < 6.0 && z > -30.0 {
            continue;
        }
        let round = rng.unit() < 0.4;
        tree(&mut land, x, z, rng.range(0.8, 1.7), round);
    }
    for (x, z, scale, round) in [
        (-9.0, 4.0, 1.5, false),
        (11.0, 2.0, 1.3, true),
        (-15.0, -4.0, 1.9, false),
        (17.0, -6.0, 1.6, false),
    ] {
        tree(&mut land, x, z, scale, round);
    }
    for _ in 0..14 {
        let (x, z) = (rng.range(-30.0, 30.0), rng.range(-40.0, 8.0));
        if x.abs() < 5.0 && z > -10.0 {
            continue;
        }
        let rock = Shape::icosphere(rng.range(0.5, 1.4), 1, 0.3, x, grey(0.4));
        land.append(
            &rock.transformed(
                &(Mat4::translation(x, FLOOR + 0.2, z) * Mat4::scaling(1.4, 0.7, 1.0)),
            ),
        );
    }
    let mut cottage = Shape::cuboid(
        Vec3::new(-2.6, 0.0, -2.0),
        Vec3::new(2.6, 2.6, 2.0),
        grey(0.75),
    );
    cottage.append(&Shape::cone(3.6, 2.4, 4, grey(0.3)).transformed(
        &(Mat4::translation(0.0, 2.6, 0.0)
            * Mat4::rotate_y(std::f32::consts::FRAC_PI_4)
            * Mat4::scaling(1.0, 1.0, 0.8)),
    ));
    cottage.append(&Shape::cuboid(
        Vec3::new(0.9, 0.0, 1.95),
        Vec3::new(1.9, 1.7, 2.1),
        grey(0.15),
    ));
    land.append(
        &cottage.transformed(&(Mat4::translation(20.0, FLOOR, -26.0) * Mat4::rotate_y(-0.5))),
    );
    land
}

fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(-0.6, 0.55, 0.55).normalise(),
        sun_colour: Vec3::new(1.5, 1.5, 1.5),
        sky: Vec3::new(0.4, 0.4, 0.4),
        ground: Vec3::new(0.15, 0.15, 0.15),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: hdr_of_displayed(Vec3::new(0.94, 0.94, 0.94)),
            density: 0.0005,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

impl Stage for Pencil {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Pizzicato],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::PencilScratch),
                0.0,
                0.45,
                0.14,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        scene.chip.draw(pass, view, &self.look);
        self.landscape.draw(
            pass,
            &self.landscape_pipeline,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Still,
            },
            ToonUniforms {
                view: view.clone(),
                lighting: lighting(),
            },
        );
        let aspect = view.aspect;
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| (aspect, scene.local_time));
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
            vignette: 0.25,
            grain: 0.01,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

/// The sky: paper tone, loose clouds shaded from beneath, and graphite smudges that darken the
/// corners, which the hatching renders as dense strokes.
struct Backdrop;

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = (f32, f32);

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, &(aspect, time): &(f32, f32)) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(aspect), ny);
        let mut tone = s(0.95) - smoothstep(0.1, 1.0, y) * s(0.04);
        let body = noise::fbm(x * s(1.1) + s(time * 0.03), y * s(2.6) + s(3.0), 4);
        let cloud = smoothstep(0.52, 0.72, body) * smoothstep(0.1, 0.4, y);
        let underside = smoothstep(
            0.52,
            0.78,
            noise::fbm(x * s(1.1) + s(time * 0.03), y * s(2.6) + s(3.25), 4),
        );
        tone = tone - cloud * s(0.1) - (cloud - underside * cloud).fast_max(s(0.0)) * s(0.12);
        let corner = (x * x * s(0.28) + y * y * s(0.9)).sqrt();
        let smudge = smoothstep(
            0.85,
            1.55,
            corner + noise::fbm(x * s(2.0), y * s(2.0), 3) * s(0.4),
        );
        tone -= smudge * s(0.4);
        let tone = clamp01(tone);
        let colour = expand(rgb(1.0, 1.0, 1.0) * tone);
        to_colour(colour, s(SKY_DISTANCE))
    }
}
