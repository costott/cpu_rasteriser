//! glitch: a corrupted signal. The world is a plain of data monoliths, sliced and shifted out of
//! true, under lanes of racing packets, with loose fragments tumbling in the air; the chip is
//! dark glass with cyan wiring and magenta light. The stylisation is a feed that keeps breaking:
//! bursts that displace blocks and scan lines, split the colour channels, invert or blank
//! blocks, and calms between them in which the picture is almost clean.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{
    clamp01, fract, from_colour, hash2, hash2_scalar, mix3, rgb, s, select3, smoothstep, splat3,
    step_lt, to_colour, v3,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVertexShader, Motion, ToonShader, ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::ScreenPass;
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::overlay::{Align, Line, Overlay, Text};
use crate::themes::{Scene, Stage};

/// The height of the ground, below the chip.
const GROUND: f32 = -3.4;
/// The spacing of the packet lanes running into the distance.
const LANE: f32 = 7.0;
const CYAN: Vec3 = Vec3 {
    x: 0.0,
    y: 0.95,
    z: 1.3,
};
const MAGENTA: Vec3 = Vec3 {
    x: 1.5,
    y: 0.08,
    z: 0.8,
};
const NAVY: Vec3 = Vec3 {
    x: 0.03,
    y: 0.05,
    z: 0.1,
};

pub struct Glitch {
    look: Look,
    monoliths: Shape,
    fragments: Shape,
    solid: SimdPipeline<KitVertexShader, ToonShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    bits: Particles,
    overlay: Overlay,
    feed: ScreenPass<FeedShader>,
}

impl Glitch {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.06, 0.08, 0.13),
                die: Vec3::new(0.01, 0.03, 0.07),
                gold: CYAN * 0.9,
                streams: MAGENTA * 1.4,
                key: Vec3::new(-0.4, 0.5, 0.75).normalise(),
                key_colour: CYAN * 1.4,
                sky: Vec3::new(0.05, 0.12, 0.2),
                ground: Vec3::new(0.01, 0.01, 0.03),
                softbox: Vec3::new(0.1, 0.0, 0.08),
            },
            monoliths: monoliths(),
            fragments: Shape::new(),
            solid: pipeline(ToonShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            bits: Particles::new(140, 54),
            overlay: Overlay::new(),
            feed: ScreenPass::new(FeedShader),
        }
    }
}

/// How much of the feed is corrupted `time` seconds in: it breaks in bursts that peak as they
/// start and decay, separated by near-clean stretches.
fn severity(time: f32) -> f32 {
    let slot = (time * 3.0).floor();
    let roll = hash2_scalar(slot, 17.0);
    if roll < 0.4 {
        return 0.04;
    }
    let decay = 1.0 - (time * 3.0 - slot);
    (roll - 0.4) / 0.6 * decay * decay + 0.04
}

/// Rows of tall slabs whose upper slices have slipped sideways, in dark navy with a few lit
/// cyan or magenta.
fn monoliths() -> Shape {
    let mut rng = Rng::new(54);
    let mut field = Shape::new();
    for row in 0..11 {
        let z = -28.0 - row as f32 * 7.5;
        let mut x = -75.0 + rng.range(0.0, 4.0);
        while x < 75.0 {
            x += rng.range(5.0, 9.0);
            if x.abs() < 10.0 && row < 6 {
                continue;
            }
            let (width, depth) = (rng.range(1.0, 2.4), rng.range(1.2, 2.6));
            let height = rng.range(8.0, 22.0) * (1.0 + row as f32 * 0.12);
            let tint = match rng.index(12) {
                0 | 1 => CYAN * 0.6,
                2 => MAGENTA * 0.6,
                _ => NAVY * rng.range(0.7, 1.8),
            };
            let slices = 2 + rng.index(2);
            let mut base = GROUND;
            let mut slip = 0.0;
            for i in 0..slices {
                let top = if i + 1 == slices {
                    GROUND + height
                } else {
                    base + height / slices as f32 * rng.range(0.6, 1.3)
                };
                slip += rng.range(-0.9, 0.9);
                let min = Vec3::new(x - width + slip, base, z - depth);
                let max = Vec3::new(x + width + slip, top, z + depth);
                field.append(&Shape::cuboid(min, max, tint * (1.0 - 0.12 * i as f32)));
                base = top + rng.range(0.0, 0.25);
            }
        }
    }
    field
}

/// Loose chunks of the scene tumbling in the air.
fn fragments(time: f32) -> Shape {
    let mut rng = Rng::new(55);
    let mut shape = Shape::new();
    for _ in 0..26 {
        let size = rng.range(0.3, 1.1);
        let home = Vec3::new(
            rng.range(-26.0, 26.0),
            rng.range(-1.0, 9.0),
            rng.range(-12.0, -55.0),
        );
        let (spin, phase) = (rng.range(-0.8, 0.8), rng.range(0.0, std::f32::consts::TAU));
        let tint = if rng.unit() < 0.5 { CYAN } else { MAGENTA };
        let cube = Shape::cuboid(
            Vec3::new(-size, -size * 0.6, -size),
            Vec3::new(size, size * 0.6, size),
            tint,
        );
        let bob = (time * 0.7 + phase).sin() * 0.6;
        let pose = Mat4::translation(home.x, home.y + bob, home.z)
            * Mat4::rotate_y(time * spin + phase)
            * Mat4::rotate_x(time * spin * 0.6);
        shape.append(&cube.transformed(&pose));
    }
    shape
}

/// Flat light and a dark fog, so the monoliths read as silhouettes with lit faces.
fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(-0.4, 0.7, 0.5).normalise(),
        sun_colour: Vec3::new(0.9, 1.0, 1.3),
        sky: Vec3::new(0.18, 0.28, 0.45),
        ground: Vec3::new(0.04, 0.05, 0.1),
        bands: 3.0,
        rim: CYAN * 0.12,
        fog: Fog {
            colour: Vec3::new(0.015, 0.05, 0.08),
            density: 0.012,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

impl Stage for Glitch {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Chip],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Stutter),
                0.0,
                0.45,
                0.057,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.fragments = fragments(scene.local_time);
        let window = (scene.extent.width, scene.extent.height);
        let pixel = 2.6 * window.1 as f32 / 720.0;
        let step = (scene.local_time * 12.0).floor();
        let code = format!("ERR 0x{:04X}", (hash2_scalar(step, 3.0) * 65535.0) as u32);
        let channel = format!("CH {:02}", 1 + (hash2_scalar(step, 9.0) * 3.0) as u32);
        let mut text = Text::default();
        let colour = CYAN * 1.3;
        for (words, x, y, align) in [
            ("SIGNAL UNSTABLE", -0.84, 0.84, Align::Left),
            (code.as_str(), -0.84, 0.76, Align::Left),
            (channel.as_str(), 0.84, -0.82, Align::Right),
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
        for shape in [&self.monoliths, &self.fragments] {
            shape.draw(
                pass,
                &self.solid,
                KitUniforms {
                    view: view.clone(),
                    motion: Motion::Still,
                },
                ToonUniforms {
                    view: view.clone(),
                    lighting: lighting(),
                },
            );
        }
        let uniforms = BackdropUniforms {
            view: view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
        self.bits.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 4.0, -14.0),
                half_extent: Vec3::new(24.0, 9.0, 18.0),
                velocity: Vec3::new(0.0, -2.2, 0.0),
                wander: 0.3,
                size: (0.04, 0.09),
                colour: CYAN * 1.2,
                twinkle: 9.0,
            },
        );
        self.overlay.draw(pass);
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let uniforms = FeedUniforms {
            source: image,
            step: (scene.local_time * 24.0).floor(),
            severity: severity(scene.local_time),
            lines: scene.extent.height as f32,
        };
        Some(self.feed.run(context, scene, uniforms))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.8,
            vignette: 0.35,
            grain: 0.02,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
    time: f32,
}

/// The sky, dark teal at the horizon with a magenta glow and the odd stray bar, over a ground
/// of dark glass crossed by lanes of cyan and magenta packets running into the distance.
struct Backdrop;

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let time = u.time;

        let elevation = ray.y.fast_max(s(0.0));
        let toward_zenith = smoothstep(0.0, 0.45, elevation).sqrt();
        let mut sky = mix3(
            rgb(0.03, 0.11, 0.17),
            rgb(0.003, 0.008, 0.03),
            toward_zenith,
        );
        sky = sky + splat3(MAGENTA * 0.2) * (-elevation * s(16.0)).exp();
        let bar = step_lt(
            s(0.985),
            hash2((ny * s(40.0)).floor(), s((time * 8.0).floor())),
        );
        sky = sky + splat3(CYAN * 0.5) * (bar * toward_zenith * s(0.4) + bar * s(0.1));

        let t = (s(GROUND - view.eye.y) / ray.y.fast_min(s(-0.002))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let cell = wx * s(1.0 / LANE) + s(0.5);
        let lane_id = cell.floor();
        let across = (fract(cell) - s(0.5)).abs() * s(LANE);
        let width = s(view.pixel_angle) * t * s(3.0);
        let line = clamp01(s(1.5) - across / width.fast_max(s(0.02)));
        let speed = hash2(lane_id, s(2.0)) * s(2.0) + s(1.5);
        let dash = step_lt(
            fract(wz * s(0.11) - s(time) * speed + hash2(lane_id, s(5.0))),
            s(0.3),
        );
        let odd = step_lt(fract(lane_id * s(0.5)), s(0.25));
        let colour = mix3(splat3(CYAN), splat3(MAGENTA), odd);
        let far = (-t * s(0.012)).exp();
        let ground = rgb(0.006, 0.01, 0.026) + colour * (line * (dash * s(1.6) + s(0.3)) * far);

        let below = ray.y.simd_lt(s(-0.002));
        let colour = select3(below, ground, sky);
        to_colour(colour, s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
struct FeedUniforms {
    source: Arc<TextureSampler>,
    /// The frame number, so the corruption redraws at 24 frames a second.
    step: f32,
    severity: f32,
    /// The number of scan lines.
    lines: f32,
}

/// The broken feed: blocks and scan lines displaced sideways, the colour channels split apart,
/// some blocks inverted or blanked in cyan or magenta, and faint scan lines throughout.
struct FeedShader;

impl FragmentShaderSimd<ScreenVaryings> for FeedShader {
    type Uniforms = FeedUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &FeedUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (step, severity) = (s(u.step), s(u.severity));

        let (bx, by) = ((x * s(7.0)).floor(), (y * s(18.0)).floor());
        let roll = hash2(bx * s(1.7) + by * s(0.31), step * s(0.37) + s(11.0));
        let hit = step_lt(roll, severity * s(0.4));
        let shove = (hash2(bx + s(3.1), by * s(1.3) + step) - s(0.5)) * s(0.24) * severity * hit;

        let row = (y * s(70.0)).floor();
        let tear = step_lt(
            s(1.0) - severity * s(0.14),
            hash2(row * s(0.91), step + s(5.0)),
        );
        let slide = (hash2(row, step + s(9.0)) - s(0.5)) * s(0.36) * severity * tear;

        let x2 = x + shove + slide;
        let split = s(0.002) + severity * s(0.026) * (s(0.35) + hit * s(0.65));
        let centre = u.source.sample_linear_clamp_simd([x2, y]);
        let red = from_colour(u.source.sample_linear_clamp_simd([x2 + split, y])).x;
        let blue = from_colour(u.source.sample_linear_clamp_simd([x2 - split, y])).z;
        let green = from_colour(centre).y;
        let mut colour = v3([red, green, blue]);

        let flip = hit * step_lt(s(0.78), hash2(bx + s(7.0), by + step));
        let (ir, ig, ib) = (s(0.8) - colour.x, s(0.8) - colour.y, s(0.8) - colour.z);
        let inverted = v3([
            ir.fast_max(s(0.0)),
            ig.fast_max(s(0.0)),
            ib.fast_max(s(0.0)),
        ]);
        colour = mix3(colour, inverted, flip);

        let blank = hit * step_lt(s(0.94), hash2(bx + s(13.0), by * s(0.7) + step));
        let paint = mix3(
            splat3(CYAN * 1.3),
            splat3(MAGENTA * 1.2),
            step_lt(s(0.5), hash2(bx, by)),
        );
        colour = mix3(colour, paint, blank);

        let line = step_lt(fract(y * s(u.lines * 0.5)), s(0.5));
        colour = colour * (s(0.9) + line * s(0.1));
        to_colour(colour, centre.a)
    }
}
