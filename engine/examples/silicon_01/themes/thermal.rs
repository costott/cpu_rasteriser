//! thermal: the view through a thermal camera. The world is drawn as heat, in luminance: a cold
//! sky and ground, a plant with warm tanks and pipes and hot chimneys breathing plumes, and the
//! chip glowing white-hot. The stylisation maps heat to the iron palette through a coarse,
//! noisy sensor, under a minimal HUD.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, rgb, s, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVertexShader, Motion, ToonShader, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::noise;
use crate::themes::kit::post::{Chain, Effect, Noise, Ramp};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::overlay::{Align, Line, Overlay, Text};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -3.4;
const CHIMNEYS: [(f32, f32); 3] = [(-20.0, -40.0), (24.0, -55.0), (-46.0, -80.0)];
const CHIMNEY_HEIGHT: f32 = 18.0;
/// Heat shown as full white on the HUD's text and marks.
const WHITE: f32 = 3.2;

pub struct Thermal {
    look: Look,
    plant: Shape,
    plant_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    overlay: Overlay,
    chain: Chain,
}

impl Thermal {
    pub fn new() -> Self {
        let heat = |h: f32| Vec3::new(h, h, h);
        Self {
            look: Look {
                substrate: heat(1.0),
                die: heat(2.4),
                gold: heat(1.6),
                streams: heat(3.0),
                key: Vec3::new(0.0, 1.0, 0.0),
                key_colour: heat(0.4),
                sky: heat(0.5),
                ground: heat(0.5),
                ..Look::default()
            },
            plant: plant(),
            plant_pipeline: pipeline(ToonShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            overlay: Overlay::new(),
            chain: Chain::new(vec![
                Effect::Pixelate { cell: 2.0 },
                Effect::Ramp(Ramp::thermal(1.3)),
                Effect::Noise(Noise {
                    amount: 0.12,
                    floor: 0.02,
                    cell: 2.0,
                    chroma: 0.0,
                }),
            ]),
        }
    }
}

/// The heat of a surface: its albedo's grey, as the unlit toon shader shows it.
fn heat(h: f32) -> Vec3 {
    Vec3::new(h, h, h)
}

/// Chimneys that glow hotter towards their tops, warm tanks joined by hot pipes and cool sheds
/// with a few warm windows.
fn plant() -> Shape {
    let mut rng = Rng::new(41);
    let mut plant = Shape::new();
    for (x, z) in CHIMNEYS {
        let chimney = Shape::cylinder(1.3, 0.9, CHIMNEY_HEIGHT, 10, heat(0.2)).colour_by(|p, _| {
            let h = p.y / CHIMNEY_HEIGHT;
            heat(0.2 + 1.0 * h * h)
        });
        plant.append(&chimney.transformed(&Mat4::translation(x, FLOOR, z)));
    }
    let mut tanks = Vec::new();
    for _ in 0..9 {
        let (x, z) = (rng.range(-55.0, 55.0), rng.range(-75.0, -28.0));
        if tanks
            .iter()
            .any(|&(tx, tz): &(f32, f32)| (tx - x).hypot(tz - z) < 12.0)
        {
            continue;
        }
        let (radius, height) = (rng.range(3.0, 5.5), rng.range(5.0, 9.0));
        let warmth = rng.range(0.16, 0.5);
        let tank = Shape::cylinder(radius, radius, height, 14, heat(warmth))
            .colour_by(|p, _| heat(warmth + 0.25 * (p.y / height).powi(3)));
        plant.append(&tank.transformed(&Mat4::translation(x, FLOOR, z)));
        tanks.push((x, z));
    }
    for pair in tanks.windows(2) {
        let ((ax, az), (bx, bz)) = (pair[0], pair[1]);
        let spine = [
            Vec3::new(ax, FLOOR + 1.4, az),
            Vec3::new((ax + bx) * 0.5, FLOOR + 2.4, (az + bz) * 0.5),
            Vec3::new(bx, FLOOR + 1.4, bz),
        ];
        plant.append(&Shape::ribbon(
            &spine,
            &[0.7, 0.7, 0.7],
            Vec3::new(0.0, 1.0, 0.0),
            heat(0.55),
        ));
    }
    for _ in 0..7 {
        let (x, z) = (rng.range(-60.0, 60.0), rng.range(-90.0, -22.0));
        let (w, d, h) = (
            rng.range(4.0, 9.0),
            rng.range(4.0, 7.0),
            rng.range(2.5, 5.0),
        );
        plant.append(
            &Shape::cuboid(Vec3::new(-w, 0.0, -d), Vec3::new(w, h, d), heat(0.1))
                .transformed(&Mat4::translation(x, FLOOR, z)),
        );
        for i in 0..4 {
            if rng.unit() < 0.6 {
                let wx = x - w + (i as f32 + 0.5) * w * 0.5;
                let front = z + d + 0.05;
                plant.quad(
                    [
                        Vec3::new(wx - 0.5, FLOOR + 1.2, front),
                        Vec3::new(wx + 0.5, FLOOR + 1.2, front),
                        Vec3::new(wx + 0.5, FLOOR + 2.2, front),
                        Vec3::new(wx - 0.5, FLOOR + 2.2, front),
                    ],
                    heat(rng.range(0.3, 0.6)),
                );
            }
        }
    }
    plant
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
            colour: heat(0.05),
            density: 0.006,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

impl Stage for Thermal {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::SawLead],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::SensorBeep),
                0.0,
                0.45,
                0.07,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let window = (scene.extent.width, scene.extent.height);
        let pixel = 2.6 * window.1 as f32 / 720.0;
        let colour = Vec3::new(WHITE, WHITE, WHITE);
        let wobble = (scene.local_time * 3.0).sin() * 1.3 + (scene.local_time * 7.3).sin() * 0.4;
        let reading = format!("TGT {:5.1} C", 412.0 + wobble);
        let mut text = Text::default();
        for (words, x, y, align) in [
            ("THERMAL  WHT-HOT", -0.85, 0.86, Align::Left),
            ("EMISSIVITY 0.95", -0.85, 0.79, Align::Left),
            (reading.as_str(), -0.85, -0.8, Align::Left),
            ("440", 0.86, 0.66, Align::Right),
            ("20", 0.86, -0.7, Align::Right),
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
        self.plant.draw(
            pass,
            &self.plant_pipeline,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Still,
            },
            ToonUniforms {
                view: view.clone(),
                lighting: unlit(),
            },
        );
        let project = |x: f32, y: f32, z: f32| {
            let clip = view.view_projection * Vec4::new(x, y, z, 1.0);
            (clip.x / clip.w, clip.y / clip.w)
        };
        let uniforms = BackdropUniforms {
            view: view.clone(),
            time: scene.local_time,
            chimneys: CHIMNEYS.map(|(x, z)| project(x, FLOOR + CHIMNEY_HEIGHT, z)),
            chip: project(0.0, 0.0, 0.0),
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
            bloom: 0.35,
            vignette: 0.45,
            grain: 0.015,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
    time: f32,
    /// Where the chimneys' tops are, in device coordinates.
    chimneys: [(f32, f32); 3],
    chip: (f32, f32),
}

/// The cold sky, the ground warmed round the chip, the chimneys' plumes and the HUD's marks.
struct Backdrop;

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(view.aspect), ny);
        let pixel = s(view.pixel_angle / view.tan_half_fov);
        let time = u.time;

        let ray = view_ray(view, nx, ny);
        let below = ray.y.simd_lt(s(-0.002));
        let t = (s(FLOOR - view.eye.y) / ray.y.fast_min(s(-0.002))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let patches = noise::fbm(wx * s(0.06), wz * s(0.06), 4);
        let near_chip = (-(wx * wx + wz * wz) * s(1.0 / 90.0)).exp();
        let ground = s(0.05) + patches * s(0.09) + near_chip * s(0.45);
        let fade = (-t * s(0.012)).exp();
        let ground = ground * fade + s(0.035) * (s(1.0) - fade);
        let sky = s(0.012)
            + smoothstep(0.35, -0.1, ny) * s(0.045)
            + noise::value(x * s(40.0), y * s(40.0)) * s(0.004);
        let mut heat = below.select(ground, sky);

        // The chip's warmth blooming into the air round it.
        let (cx, cy) = ((nx - s(u.chip.0)) * s(view.aspect), ny - s(u.chip.1));
        heat += (-(cx * cx * s(2.2) + cy * cy * s(1.4))).exp() * s(0.3);

        // Chimney plumes: rising, widening, drifting columns of warm air.
        for (tx, ty) in u.chimneys {
            let (dx, dy) = (x - s(tx * view.aspect), y - s(ty));
            let rise = dy.fast_max(s(0.0));
            let width = s(0.015) + rise * s(0.22);
            let column = (-(dx / width) * (dx / width)).exp();
            let turbulence = noise::fbm(x * s(14.0), (y - s(time * 0.5)) * s(7.0), 3);
            let above = dy.simd_gt(s(0.0)).select(s(1.0), s(0.0));
            heat += column * (-rise * s(2.4)).exp() * turbulence * above * s(0.9);
        }

        // The HUD: corner brackets round the target and a heat scale down the right.
        let (half_x, top, bottom, arm, weight) = (0.8f32, 0.9f32, -0.4f32, 0.09f32, 1.2f32);
        let bracket = |sx: F, sy: F, edge_y: f32| {
            let (ox, oy) = (sx.abs() - s(half_x), (sy - s(edge_y)).abs());
            let near_corner_x = (ox.abs() - s(weight) * pixel * s(0.5)).fast_max(s(0.0));
            let along_x = (oy - s(arm)).fast_max(s(0.0));
            let on_vertical = clamp01(s(weight) * pixel * s(0.5) - ox.abs()) / (pixel * s(0.5))
                * clamp01(s(1.0) - along_x / pixel);
            let on_horizontal = clamp01(s(weight) * pixel * s(0.5) - oy) / (pixel * s(0.5))
                * clamp01(s(1.0) - (ox + s(arm)).fast_max(s(0.0)) / pixel)
                * (s(1.0) - clamp01(near_corner_x / pixel))
                    .fast_max(clamp01(s(arm) - ox.abs().fast_max(s(0.0)) + s(0.0)));
            clamp01(on_vertical.fast_max(on_horizontal))
        };
        let marks = bracket(x, y, top).fast_max(bracket(x, y, bottom));
        heat += marks * s(WHITE);
        let (bar_x, bar_low, bar_high) = (1.5f32, -0.7f32, 0.66f32);
        let inside =
            (x - s(bar_x)).abs().simd_lt(s(0.03)) & y.simd_gt(s(bar_low)) & y.simd_lt(s(bar_high));
        let gradient = (y - s(bar_low)) * s(2.6 / (bar_high - bar_low));
        heat = inside.select(gradient, heat);

        to_colour(rgb(1.0, 1.0, 1.0) * heat, s(SKY_DISTANCE))
    }
}
