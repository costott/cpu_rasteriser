//! savanna-silhouette: pure black acacias, giraffes and an elephant against a huge low sun, the
//! plain behind them dissolving into the glowing haze. The chip is one more black shape, its
//! edge rim-lit and only the die's seams glowing green.

use std::f32::consts::PI;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{F, clamp01, length, mix3, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, pipeline,
};
use crate::themes::kit::light::facing;
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// The direction towards the sun: huge, low, and just behind and to the right of the chip.
const SUN: Vec3 = Vec3 {
    x: 0.16,
    y: 0.13,
    z: -0.98,
};
/// The sun's angular radius, in radians.
const SUN_SIZE: f32 = 0.13;
const WIND: Vec2 = Vec2 { x: 1.0, y: 0.2 };
const BLACK: Vec3 = Vec3::ZERO;

fn ground(x: f32, z: f32) -> f32 {
    let open = ((x.hypot(z) - 5.0) / 30.0).clamp(0.0, 1.0);
    FLOOR + open * (fbm_scalar(x * 0.025, z * 0.025, 3) - 0.45) * 4.0
}

fn look() -> Look {
    Look {
        substrate: Vec3::new(0.05, 0.03, 0.025),
        die: Vec3::new(0.012, 0.016, 0.014),
        gold: Vec3::new(0.14, 0.08, 0.035),
        streams: Vec3::new(0.25, 0.95, 0.55),
        key: Vec3::new(0.3, 0.3, -0.9).normalise(),
        key_colour: Vec3::new(4.5, 2.0, 0.7),
        sky: Vec3::new(0.14, 0.055, 0.03),
        ground: Vec3::ZERO,
        softbox: Vec3::ZERO,
    }
}

/// The sunset in direction `dir`: gold and orange at the horizon, through crimson and purple to
/// indigo overhead, a wide glow round the sun and thin cloud lit from below.
#[inline(always)]
fn sunset(dir: Vec3Simd, time: F) -> Vec3Simd {
    let up = dir.y.fast_max(s(0.0));
    let sun = SUN.normalise();
    let cos_sun = dir.dot(splat3(sun));
    let angle = cos_sun.fast_min(s(1.0)).acos();
    let horizontal = (dir.x * dir.x + dir.z * dir.z).sqrt().fast_max(s(1e-3));
    let toward = clamp01(
        (dir.x * s(sun.x) + dir.z * s(sun.z))
            / horizontal
            / s((sun.x * sun.x + sun.z * sun.z).sqrt()),
    );
    let near_sun = toward * toward * toward;

    let horizon = mix3(rgb(0.85, 0.24, 0.06), rgb(1.3, 0.62, 0.16), near_sun);
    let mut colour = mix3(horizon, rgb(0.62, 0.14, 0.2), smoothstep(0.0, 0.14, up));
    colour = mix3(colour, rgb(0.26, 0.09, 0.32), smoothstep(0.1, 0.36, up));
    colour = mix3(colour, rgb(0.05, 0.035, 0.17), smoothstep(0.35, 0.95, up));
    colour = colour + rgb(1.5, 0.75, 0.25) * ((-angle * s(3.2)).exp() * s(0.3));

    let azimuth = dir.x.atan2(-dir.z);
    let band = fbm(azimuth * s(6.0) + time * s(0.004), up * s(30.0), 3);
    let streak = smoothstep(0.5, 0.78, band)
        * smoothstep(0.045, 0.1, up)
        * (s(1.0) - smoothstep(0.25, 0.5, up));
    let lit = mix3(
        rgb(0.28, 0.08, 0.26),
        rgb(1.5, 0.55, 0.15),
        clamp01(near_sun * s(1.2) + (s(0.2) - up) * s(2.0)),
    );
    mix3(colour, lit, streak * s(0.7))
}

/// The sky: the sunset and the sun's disc, wavering in the heat above the horizon.
struct SunsetSky;

impl FragmentShaderSimd<ScreenVaryings> for SunsetSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let time = s(view.frame.time);
        let ray = view_ray(view, nx, ny);
        let shimmer = fbm(nx * s(40.0), ny * s(18.0) - time * s(0.6), 2) - s(0.5);
        let haze = (s(1.0) - smoothstep(0.0, 0.12, ray.y.fast_max(s(0.0)))) * shimmer * s(0.012);
        let dir = view_ray(view, nx + haze, ny + haze * s(0.4));
        let angle = dir.dot(splat3(SUN.normalise())).fast_min(s(1.0)).acos();
        let disc = s(1.0) - smoothstep(SUN_SIZE * 0.97, SUN_SIZE * 1.03, angle);
        let limb = s(1.0) - smoothstep(0.0, SUN_SIZE, angle) * s(0.35);
        let core = rgb(3.4, 1.9, 0.6) * limb;
        to_colour(mix3(sunset(dir, time), core, disc), s(SKY_DISTANCE))
    }
}

/// Black ground and black shapes that fade into the haze with distance, and glow faintly at
/// edges turned away from the viewer towards the sun.
struct SilhouetteShader;

impl FragmentShaderSimd<KitVaryings> for SilhouetteShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let normal = facing(v.normal().normalise(), to_eye);
        let outward = splat3(view.eye) - world;
        let away = Vec3Simd::new(-outward.x, s(0.02), -outward.z).normalise();
        let haze = sunset(away, s(view.frame.time));
        let fog = s(1.0) - (-distance * distance * s(0.00006)).exp();

        let facing_eye = clamp01(normal.dot(to_eye * (s(1.0) / distance)));
        let toward_sun = clamp01(normal.dot(splat3(SUN.normalise())) * s(0.7) + s(0.3));
        let rim = (s(1.0) - facing_eye) * (s(1.0) - facing_eye) * toward_sun * s(0.45);
        let dust = fbm(world.x * s(0.4), world.z * s(0.4), 2) * s(0.01);
        let body =
            rgb(0.012, 0.007, 0.008) + rgb(0.02, 0.01, 0.006) * dust + rgb(1.5, 0.55, 0.15) * rim;
        scene_colour(view, mix3(body, haze, fog), world)
    }
}

/// A giraffe walking with its head to +x, `phase` radians into its stride.
fn giraffe(phase: f32) -> Shape {
    let mut shape = Shape::icosphere(1.0, 1, 0.05, 2.0, BLACK)
        .transformed(&(Mat4::translation(0.0, 3.3, 0.0) * Mat4::scaling(1.45, 0.72, 0.58)));
    let bob = (phase * 2.0).sin() * 0.05;
    let sway = phase.sin() * 0.18;
    let base = Vec3::new(1.05, 3.55, 0.0);
    let mid = Vec3::new(1.75 + sway * 0.4, 4.7 + bob, 0.0);
    let top = Vec3::new(2.45 + sway, 5.75 + bob, 0.0);
    shape.append(&Shape::limb(base, mid, 0.34, 0.26, 5, BLACK));
    shape.append(&Shape::limb(mid, top, 0.26, 0.19, 5, BLACK));
    let head = top + Vec3::new(0.32, 0.12, 0.0);
    shape.append(&Shape::limb(
        top,
        head + Vec3::new(0.28, -0.14, 0.0),
        0.19,
        0.12,
        5,
        BLACK,
    ));
    shape.append(
        &Shape::cone(0.04, 0.3, 4, BLACK)
            .transformed(&Mat4::translation_vec(top + Vec3::new(0.05, 0.1, 0.1))),
    );
    shape.append(
        &Shape::cone(0.04, 0.3, 4, BLACK)
            .transformed(&Mat4::translation_vec(top + Vec3::new(0.05, 0.1, -0.1))),
    );
    shape.append(&Shape::limb(
        top,
        top + Vec3::new(-0.3, 0.1, 0.22),
        0.04,
        0.02,
        3,
        BLACK,
    ));
    shape.append(&Shape::limb(
        Vec3::new(-1.35, 3.5, 0.0),
        Vec3::new(-1.7, 2.5 + bob, 0.0),
        0.07,
        0.04,
        4,
        BLACK,
    ));

    for (x, z, offset) in [
        (0.95, 0.28, 0.0),
        (0.95, -0.28, PI),
        (-0.95, 0.28, PI),
        (-0.95, -0.28, 0.0),
    ] {
        let swing = (phase + offset).sin() * 0.42;
        let lift = ((phase + offset).cos().max(0.0)) * 0.25;
        let hip = Vec3::new(x, 2.85, z);
        let knee = hip + Vec3::new(swing.sin() * 1.3, -swing.cos() * 1.3, 0.0);
        let foot_angle = swing * 0.4 - 0.25 * lift;
        let foot = knee
            + Vec3::new(
                foot_angle.sin() * 1.45,
                -foot_angle.cos().min(1.0) * 1.45 + lift,
                0.0,
            );
        shape.append(&Shape::limb(hip, knee, 0.17, 0.11, 5, BLACK));
        shape.append(&Shape::limb(knee, foot, 0.11, 0.08, 5, BLACK));
    }
    shape
}

/// An elephant walking with its head to +x.
fn elephant(phase: f32) -> Shape {
    let mut shape = Shape::icosphere(1.0, 1, 0.05, 5.0, BLACK)
        .transformed(&(Mat4::translation(0.0, 2.5, 0.0) * Mat4::scaling(1.75, 1.2, 0.95)));
    shape.append(
        &Shape::icosphere(1.0, 1, 0.05, 7.0, BLACK)
            .transformed(&(Mat4::translation(1.85, 2.75, 0.0) * Mat4::scaling(0.85, 0.9, 0.7))),
    );
    let sway = phase.sin() * 0.22;
    let mut from = Vec3::new(2.55, 2.5, 0.0);
    let mut width = 0.3;
    for (dx, dy) in [
        (0.35, -0.5),
        (0.2 + sway * 0.5, -0.6),
        (0.15 + sway, -0.55),
        (0.35 + sway, -0.35),
    ] {
        let to = from + Vec3::new(dx, dy, 0.0);
        shape.append(&Shape::limb(from, to, width, width * 0.78, 5, BLACK));
        from = to;
        width *= 0.78;
    }
    for side in [-1.0, 1.0] {
        shape.append(&Shape::icosphere(1.0, 1, 0.05, 9.0, BLACK).transformed(
            &(Mat4::translation(1.45, 3.0, side * 0.75)
                * Mat4::rotate_y(side * 0.35)
                * Mat4::scaling(0.12, 0.85, 0.75)),
        ));
        shape.append(&Shape::limb(
            Vec3::new(2.35, 2.35, side * 0.38),
            Vec3::new(3.0, 2.0, side * 0.42),
            0.07,
            0.02,
            4,
            BLACK,
        ));
    }
    shape.append(&Shape::limb(
        Vec3::new(-1.65, 2.8, 0.0),
        Vec3::new(-1.9, 1.5, 0.0),
        0.08,
        0.05,
        4,
        BLACK,
    ));
    for (x, z, offset) in [
        (1.05, 0.5, 0.0),
        (1.05, -0.5, PI),
        (-1.05, 0.5, PI),
        (-1.05, -0.5, 0.0),
    ] {
        let swing = (phase + offset).sin() * 0.3;
        let hip = Vec3::new(x, 2.2, z);
        let foot = hip
            + Vec3::new(
                swing.sin() * 1.9,
                -swing.cos() * 1.9 + ((phase + offset).cos().max(0.0)) * 0.18,
                0.0,
            );
        shape.append(&Shape::limb(hip, foot, 0.42, 0.34, 6, BLACK));
    }
    shape
}

/// An acacia: a forked trunk under a wide flat canopy.
fn acacia(rng: &mut Rng, height: f32) -> Shape {
    let top = Vec3::new(rng.range(-0.3, 0.3), height, rng.range(-0.3, 0.3));
    let fork = Vec3::new(0.0, height * 0.45, 0.0);
    let mut tree = Shape::limb(Vec3::ZERO, fork, 0.25, 0.2, 5, BLACK);
    tree.append(&Shape::limb(fork, top, 0.2, 0.12, 5, BLACK));
    for side in [-1.0, 1.0] {
        let tip = Vec3::new(side * height * 0.45, height * 0.85, rng.range(-0.8, 0.8));
        tree.append(&Shape::limb(fork, tip, 0.14, 0.08, 4, BLACK));
    }
    let width = height * 1.5;
    for k in 0..3 {
        let offset = Vec3::new(
            (k as f32 - 1.0) * width * 0.55,
            rng.range(-0.2, 0.2),
            rng.range(-0.5, 0.5),
        );
        tree.append(
            &Shape::icosphere(1.0, 1, 0.2, rng.range(0.0, 30.0), BLACK).transformed(
                &(Mat4::translation_vec(top + offset + Vec3::new(0.0, height * 0.08, 0.0))
                    * Mat4::scaling(width * 0.5, height * 0.12, width * 0.34)),
            ),
        );
    }
    tree.sway_by_height(height * 0.7, height * 1.1, 1.0)
}

fn tuft(rng: &mut Rng) -> Shape {
    let mut tuft = Shape::new();
    for _ in 0..5 {
        let angle = rng.range(0.0, 2.0 * PI);
        let (height, lean) = (rng.range(0.5, 1.1), rng.range(0.1, 0.45));
        let base = Vec3::new(angle.cos() * 0.08, 0.0, angle.sin() * 0.08);
        let side = Vec3::new(-angle.sin(), 0.0, angle.cos()) * 0.03;
        tuft.triangle(
            base - side,
            base + side,
            base + Vec3::new(angle.cos() * lean, height, angle.sin() * lean),
            BLACK,
        );
    }
    tuft.sway_by_height(0.0, 1.1, 1.0)
}

/// A walker in the herd: what it is, where it started, how it is posed and how fast it goes.
struct Walker {
    elephant: bool,
    x: f32,
    z: f32,
    scale: f32,
    /// Towards +x (1) or -x (-1).
    heading: f32,
    speed: f32,
    phase: f32,
}

const HERD: [Walker; 9] = [
    Walker {
        elephant: false,
        x: -9.0,
        z: -22.0,
        scale: 1.0,
        heading: 1.0,
        speed: 0.9,
        phase: 0.0,
    },
    Walker {
        elephant: false,
        x: 3.0,
        z: -29.0,
        scale: 0.88,
        heading: 1.0,
        speed: 0.85,
        phase: 1.3,
    },
    Walker {
        elephant: false,
        x: -17.0,
        z: -27.0,
        scale: 0.62,
        heading: 1.0,
        speed: 0.9,
        phase: 2.5,
    },
    Walker {
        elephant: true,
        x: 17.0,
        z: -19.0,
        scale: 1.0,
        heading: -1.0,
        speed: 0.8,
        phase: 0.4,
    },
    Walker {
        elephant: false,
        x: -34.0,
        z: -64.0,
        scale: 1.1,
        heading: 1.0,
        speed: 0.9,
        phase: 0.7,
    },
    Walker {
        elephant: false,
        x: -27.0,
        z: -66.0,
        scale: 1.0,
        heading: 1.0,
        speed: 0.9,
        phase: 2.0,
    },
    Walker {
        elephant: false,
        x: -20.0,
        z: -63.0,
        scale: 0.9,
        heading: 1.0,
        speed: 0.9,
        phase: 3.1,
    },
    Walker {
        elephant: true,
        x: 30.0,
        z: -58.0,
        scale: 0.95,
        heading: -1.0,
        speed: 0.7,
        phase: 1.1,
    },
    Walker {
        elephant: false,
        x: 40.0,
        z: -75.0,
        scale: 1.0,
        heading: -1.0,
        speed: 0.85,
        phase: 4.0,
    },
];

fn herd(time: f32) -> Shape {
    let mut herd = Shape::new();
    for walker in &HERD {
        let x = walker.x + walker.heading * walker.speed * time;
        let phase = walker.phase + time * 2.4;
        let figure = if walker.elephant {
            elephant(phase)
        } else {
            giraffe(phase)
        };
        let turn = if walker.heading > 0.0 { 0.0 } else { PI };
        let at = Mat4::translation(x, ground(x, walker.z) + 0.02, walker.z)
            * Mat4::rotate_y(turn)
            * Mat4::scaling(walker.scale, walker.scale, walker.scale);
        herd.append(&figure.transformed(&at));
    }
    herd
}

pub struct SavannaSilhouette {
    plain: Shape,
    trees: Shape,
    grass: Shape,
    herd: Shape,
    dust: Particles,
    motes: Particles,
    black: SimdPipeline<KitVertexShader, SilhouetteShader>,
    sky: SimdPipeline<ScreenVertexShader, SunsetSky>,
}

impl SavannaSilhouette {
    pub fn new() -> Self {
        let mut rng = Rng::new(1701);
        let plain = terrain(&ring_radii(1.5, 260.0, 44), 80, ground, |_, _| BLACK);

        let acacias: Vec<Shape> = [5.0, 6.2, 7.5]
            .iter()
            .map(|&h| acacia(&mut rng, h))
            .collect();
        let sites = [
            (-24.0, -44.0, 1.0),
            (21.0, -38.0, 0.9),
            (-2.0, -56.0, 1.2),
            (-48.0, -72.0, 1.4),
            (46.0, -80.0, 1.5),
            (-12.0, -95.0, 1.6),
            (62.0, -60.0, 1.2),
            (-70.0, -50.0, 1.1),
        ];
        let placements: Vec<_> = sites
            .iter()
            .enumerate()
            .map(|(i, &(x, z, scale))| Placement {
                position: Vec3::new(x, ground(x, z), z),
                yaw: i as f32 * 2.1,
                scale,
                seed: (i as f32 * 0.29 + 0.1).fract(),
            })
            .collect();
        let mut trees = Batch::new();
        trees.add(&acacias, &placements, |p| (p.seed * 3.0) as usize % 3);

        let tufts: Vec<Shape> = (0..4).map(|_| tuft(&mut rng)).collect();
        let spots = scatter(
            &mut rng,
            30.0,
            0.8,
            9000,
            ground,
            |x, z| {
                (z < 14.0
                    && z > -34.0
                    && !(x.abs() < 3.4 && z > -2.0 && z < 15.0)
                    && x.hypot(z) > 3.8)
                    .then_some(1.0 + x.hypot(z) * 0.03)
            },
            (0.7, 1.5),
        );
        let mut grass = Batch::new();
        grass.add(&tufts, &spots[..spots.len().min(520)], |p| {
            (p.seed * 4.0) as usize % 4
        });

        Self {
            plain,
            trees: trees.finish(),
            grass: grass.finish(),
            herd: Shape::new(),
            dust: Particles::new(60, 17),
            motes: Particles::new(160, 18),
            black: pipeline(SilhouetteShader, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, SunsetSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for SavannaSilhouette {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Kalimba],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Insects),
                0.0,
                0.45,
                0.1,
                -0.35,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.herd = herd(scene.local_time);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let acacia_sway = Motion::Sway(Sway {
            amplitude: 0.08,
            speed: 0.6,
            wind: WIND,
        });
        let grass_sway = Motion::Sway(Sway {
            amplitude: 0.3,
            speed: 1.6,
            wind: WIND,
        });

        self.plain
            .draw(pass, &self.black, motion(Motion::Still), view.clone());
        self.trees
            .draw(pass, &self.black, motion(acacia_sway), view.clone());
        self.grass
            .draw(pass, &self.black, motion(grass_sway), view.clone());
        self.herd
            .draw(pass, &self.black, motion(Motion::Still), view.clone());
        scene.chip.draw(pass, view, &look());
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| view.clone());

        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, -0.4, -10.0),
                half_extent: Vec3::new(24.0, 2.4, 22.0),
                velocity: Vec3::new(1.2, 0.1, 0.2),
                wander: 1.5,
                size: (0.7, 1.6),
                colour: Vec3::new(0.18, 0.06, 0.012),
                twinkle: 0.0,
            },
        );
        self.motes.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 0.2, -4.0),
                half_extent: Vec3::new(16.0, 3.0, 14.0),
                velocity: Vec3::new(0.8, 0.25, 0.1),
                wander: 0.6,
                size: (0.02, 0.05),
                colour: Vec3::new(1.6, 0.8, 0.3),
                twinkle: 2.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.5,
            vignette: 0.4,
            grain: 0.015,
            aberration: 0.0015,
            desaturate: 0.0,
        }
    }
}
