//! crt-vector: a vector display. Green phosphor lines on black glass, a wireframe landscape
//! under an amber vector sun, bloom and persistence trails.

use std::f32::consts::TAU;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, fract, hash2, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVertexShader, Motion, ToonShader, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::post::{Chain, Crt, Edges, Effect, Noise};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::overlay::{Align, Line, Overlay, Text};
use crate::themes::{Scene, Stage};

/// The height of the ground grid in world units, below the chip.
const GROUND: f32 = -3.4;
/// The distance between grid lines.
const GRID: f32 = 2.0;
const GREEN: Vec3 = Vec3 {
    x: 0.15,
    y: 1.0,
    z: 0.35,
};
const AMBER: Vec3 = Vec3 {
    x: 1.0,
    y: 0.55,
    z: 0.06,
};

pub struct CrtVector {
    look: Look,
    city: Shape,
    city_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    overlay: Overlay,
    chain: Chain,
}

impl CrtVector {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.012, 0.03, 0.014),
                die: Vec3::new(0.0, 0.05, 0.02),
                gold: GREEN * 0.4,
                streams: GREEN * 1.2,
                key_colour: GREEN * 0.9,
                sky: GREEN * 0.12,
                ground: Vec3::ZERO,
                ..Look::default()
            },
            city: city(),
            city_pipeline: pipeline(ToonShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            overlay: Overlay::new(),
            chain: Chain::new(vec![
                Effect::Edges(Edges {
                    ink: GREEN * 2.4,
                    width: 1.1,
                    ..Edges::default()
                }),
                Effect::Persistence { half_life: 0.07 },
                Effect::Crt(Crt {
                    curvature: 0.09,
                    scanlines: 0.5,
                    pitch: 3.0,
                    mask: 0.3,
                    convergence: 0.9,
                    flicker: 0.04,
                }),
                Effect::Noise(Noise {
                    amount: 0.1,
                    floor: 0.006,
                    cell: 1.0,
                    chroma: 0.0,
                }),
            ]),
        }
    }
}

/// Towers, pyramids and slabs on the ground, drawn black so the edge pass outlines them in
/// phosphor.
fn city() -> Shape {
    let mut rng = Rng::new(32);
    let mut city = Shape::new();
    while city.triangle_count() < 400 {
        let angle = rng.range(0.0, TAU);
        let distance = rng.range(9.0, 48.0);
        let (x, z) = (
            angle.cos() * distance * 1.5,
            -distance.max(14.0) * angle.sin().abs().max(0.3),
        );
        if x.abs() < 3.5 && z > -22.0 {
            continue;
        }
        let height = rng.range(1.5, 7.0);
        let width = rng.range(0.9, 2.4);
        let solid = match rng.index(3) {
            0 => Shape::cuboid(
                Vec3::new(-width, 0.0, -width),
                Vec3::new(width, height, width),
                Vec3::ZERO,
            ),
            1 => Shape::cone(width * 1.4, height, 4, Vec3::ZERO),
            _ => Shape::cuboid(
                Vec3::new(-width * 1.8, 0.0, -width * 0.6),
                Vec3::new(width * 1.8, height * 0.4, width * 0.6),
                Vec3::ZERO,
            ),
        };
        city.append(&solid.transformed(&Mat4::translation(x, GROUND, z)));
    }
    city
}

impl Stage for CrtVector {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Ringed],
            treatment: Treatment::Ringed,
            ambience: vec![
                Ambience::once(Source::Recording("degauss", 0.0), 0.0, 0.7, 0.2, 0.0),
                Ambience::once(Source::Synth(Accent::TubeHum), 0.0, 0.6, 0.05, 0.0),
            ],
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let window = (scene.extent.width, scene.extent.height);
        let pixel = 2.6 * window.1 as f32 / 720.0;
        let time = scene.local_time;
        let mut text = Text::default();
        let tick = format!("T+{time:06.2}");
        for (words, x, y, align) in [
            ("SILICON-01  VECTOR DISPLAY", -0.78, 0.8, Align::Left),
            ("RASTER: CPU    GPU: NONE", -0.78, 0.7, Align::Left),
            ("SCAN 24 HZ", 0.78, 0.8, Align::Right),
            (tick.as_str(), 0.78, -0.8, Align::Right),
        ] {
            let line = Line {
                text: words,
                x,
                y,
                align,
                colour: GREEN * 1.5,
            };
            text.line(line, pixel, window);
        }
        self.overlay.set(text);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        scene.chip.draw(pass, view, &self.look);
        self.city.draw(
            pass,
            &self.city_pipeline,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Still,
            },
            ToonUniforms {
                view: view.clone(),
                lighting: unlit(),
            },
        );
        let uniforms = BackdropUniforms {
            view: view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
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
            bloom: 1.5,
            vignette: 0.55,
            grain: 0.02,
            aberration: 0.006,
            desaturate: 0.0,
        }
    }
}

/// Shows an albedo as it is: no light, no fog.
fn unlit() -> Lighting {
    Lighting {
        sun: Vec3::new(0.0, 1.0, 0.0),
        sun_colour: Vec3::ZERO,
        sky: Vec3::ONE,
        ground: Vec3::ONE,
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: Vec3::ZERO,
            density: 0.0,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
    time: f32,
}

/// Black glass: a vector grid under a horizon of outlined mountains and an amber sun.
struct Backdrop;

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let footprint = s(view.line_angle);
        let mut colour = rgb(0.0, 0.0, 0.0);

        // The ground: a perspective grid whose lines fade as they approach a pixel's width, and
        // a pulse running out along it from the chip.
        let below = ray.y.simd_lt(s(-0.002)).select(s(1.0), s(0.0));
        let t = (s(GROUND - view.eye.y) / ray.y.fast_min(s(-0.002))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let line = |p: F, size: F| {
            let d = (fract(p * s(1.0 / GRID) + s(0.5)) - s(0.5)).abs() * s(GRID);
            clamp01(s(1.5) - d / size)
        };
        let across = footprint * t;
        let along = across / ray.y.abs().fast_max(s(0.02));
        let detail = clamp01(s(1.0) - along / s(GRID * 0.5));
        let grid = line(wx, across).fast_max(line(wz, along)) * detail * below;
        let range = (-t * s(0.035)).exp();
        let radius = (wx * wx + wz * wz).sqrt();
        let wave = (-(radius - s((u.time * 7.0) % 36.0 + 2.0)).abs() * s(0.5)).exp();
        colour = colour + splat3(GREEN * 2.0) * (grid * range * (s(1.0) + wave * s(2.5)));

        // The horizon and the mountains' outlines, in angle above the horizon.
        let elevation = ray.y;
        let horizon = clamp01(s(1.5) - elevation.abs() / (footprint * s(1.5)));
        colour = colour + splat3(AMBER * 1.4) * horizon;
        for (layer, (scale, height, tint)) in [(0.9f32, 0.05f32, 0.9f32), (2.1, 0.034, 0.5)]
            .into_iter()
            .enumerate()
        {
            let frequency = scale * 6.0;
            let along_ridge = ray.x * s(frequency) + s(layer as f32 * 3.7);
            let peak = s(height) * (s(0.4) + hash2(along_ridge.floor(), s(layer as f32 + 1.0)));
            let outline = peak * (s(1.0) - (fract(along_ridge) * s(2.0) - s(1.0)).abs());
            let slope = peak * s(2.0 * frequency);
            let distance = (elevation - outline).abs() / (s(1.0) + slope * slope).sqrt();
            let above = elevation.simd_gt(s(0.0)).select(s(1.0), s(0.0));
            let mountain = clamp01(s(1.5) - distance / (footprint * s(1.4))) * above;
            colour = colour + splat3(GREEN * (tint * 1.6)) * mountain;
        }

        // The sun: two rings, the lower half cut by slits that drift upwards.
        let pixel = s(view.line_angle / view.tan_half_fov);
        let (sx, sy) = (nx * s(view.aspect), ny - s(0.12));
        let r = (sx * sx + sy * sy).sqrt();
        let ring =
            |radius: f32, width: f32| clamp01(s(1.5) - (r - s(radius)).abs() / (pixel * s(width)));
        let slits = smoothstep(0.0, 0.5, fract(sy * s(14.0) - s(u.time * 0.4)));
        let cut = sy.simd_lt(s(-0.05)).select(slits, s(1.0));
        colour =
            colour + splat3(AMBER * 1.6) * ((ring(0.66, 1.2) + ring(0.56, 0.8) * s(0.6)) * cut);
        to_colour(colour, s(SKY_DISTANCE))
    }
}
