//! flipbook-bw: black ink on white paper, redrawn twelve times a second. A sparse forest is
//! outlined and filled in flat tones, the sky is ink clouds and a rayed sun, and everything
//! (sway, the chip's pose, the wobble of the lines) steps at the flipbook's frame rate.

use std::f32::consts::TAU;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::director;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, View, ViewRef, scene_colour, view_ray};
use crate::simd::{F, clamp01, fract, length, rgb, s, smoothstep, splat3, to_colour};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonShader,
    ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{self, fbm_scalar};
use crate::themes::kit::post::{Chain, Edges, Effect, Noise, Paper, expand, hdr_of_displayed};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, scatter};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// The flipbook's frame rate.
const FPS: f32 = 12.0;
const SUN: Vec3 = Vec3 {
    x: -0.55,
    y: 0.6,
    z: 0.5,
};
const WIND: Vec2 = Vec2 { x: 0.9, y: 0.4 };

fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let flat = (r / 9.0).clamp(0.0, 1.0);
    let roll = (fbm_scalar(x * 0.05 + 2.0, z * 0.06, 3) - 0.5) * 2.4;
    let hills_edge = ((r - 46.0) / 60.0).clamp(0.0, 1.0);
    let hills = hills_edge * hills_edge * (6.0 + 20.0 * fbm_scalar(x * 0.02 + 5.0, z * 0.02, 3));
    FLOOR + flat * flat * roll + hills
}

fn grey(tone: f32) -> Vec3 {
    Vec3::new(tone, tone, tone)
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: grey(1.7),
        sky: grey(0.55),
        ground: grey(0.35),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: hdr_of_displayed(grey(0.95)),
            density: 0.0006,
            falloff: 0.0,
            base: FLOOR,
        },
    }
}

/// The chip in ink: a white package outlined in black, the die solid black.
fn look() -> Look {
    Look {
        substrate: grey(0.7),
        die: grey(0.0),
        gold: grey(0.9),
        streams: Vec3::ZERO,
        key: SUN.normalise(),
        key_colour: grey(1.4),
        sky: grey(0.4),
        ground: grey(0.2),
        softbox: Vec3::ZERO,
    }
}

fn conifer(rng: &mut Rng, solid: bool) -> Shape {
    let tone = if solid { 0.05 } else { 0.8 };
    let mut tree = Shape::cylinder(0.25, 0.15, 1.8, 5, grey(0.1));
    for k in 0..3 {
        let t = k as f32 / 2.0;
        let cone = Shape::cone(2.3 * (1.0 - 0.5 * t), 3.2, 6, grey(tone)).transformed(
            &(Mat4::translation(0.0, 1.3 + k as f32 * 1.7, 0.0)
                * Mat4::rotate_y(rng.range(0.0, TAU))),
        );
        tree.append(&cone);
    }
    tree.sway_by_height(0.8, 7.0, 1.5)
}

fn round_tree(rng: &mut Rng, solid: bool) -> Shape {
    let mut tree = Shape::cylinder(0.3, 0.2, 2.8, 5, grey(0.1));
    for (centre, radius) in [
        (Vec3::new(0.0, 4.2, 0.0), 2.1),
        (Vec3::new(1.2, 3.6, 0.3), 1.5),
        (Vec3::new(-1.0, 3.8, -0.4), 1.5),
    ] {
        let crown = Shape::icosphere(
            radius,
            1,
            0.15,
            rng.range(0.0, 50.0),
            grey(if solid { 0.08 } else { 0.85 }),
        )
        .transformed(&Mat4::translation_vec(centre));
        tree.append(&crown);
    }
    tree.sway_by_height(1.0, 6.0, 1.4)
}

fn boulder(rng: &mut Rng) -> Shape {
    Shape::icosphere(1.0, 1, 0.25, rng.range(0.0, 99.0), grey(0.7)).transformed(&Mat4::scaling(
        rng.range(0.9, 1.4),
        rng.range(0.55, 0.8),
        rng.range(0.8, 1.2),
    ))
}

/// A few ink strokes of grass: dark blades that give the white ground a texture.
fn tuft(rng: &mut Rng) -> Shape {
    let mut tuft = Shape::new();
    for _ in 0..4 {
        let angle = rng.range(0.0, TAU);
        let (height, lean) = (rng.range(0.35, 0.7), rng.range(0.05, 0.3));
        let base = Vec3::new(angle.cos() * 0.08, 0.0, angle.sin() * 0.08);
        let side = Vec3::new(-angle.sin(), 0.0, angle.cos()) * 0.04;
        let tip = base + Vec3::new(angle.cos() * lean, height, angle.sin() * lean);
        tuft.triangle(base - side, base + side, tip, grey(0.05));
    }
    tuft.sway_by_height(0.0, 0.7, 1.0)
}

/// The ground in white with rows of ink strokes running away from the camera, fading out where
/// they would crowd into noise.
struct InkGround;

impl FragmentShaderSimd<KitVaryings> for InkGround {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let world = v.world();
        let to_eye = splat3(u.view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        let wander = noise::fbm(world.x * s(0.2), world.z * s(0.2), 2) * s(1.5);
        let row = fract(world.z * s(0.8) + wander);
        let dash = smoothstep(0.45, 0.6, noise::fbm(world.x * s(0.5), world.z * s(2.0), 2));
        let stroke = smoothstep(0.0, 0.1, row) * (s(1.0) - smoothstep(0.1, 0.2, row)) * dash;
        let near = s(1.0) - smoothstep(14.0, 34.0, length(to_eye));
        let albedo = v.colour() * (s(1.0) - stroke * near * s(0.9));
        scene_colour(
            &u.view,
            u.lighting.shade(&u.view, world, normal, albedo, s(1.0)),
            world,
        )
    }
}

pub struct FlipbookBw {
    land: Shape,
    trees: Shape,
    grass: Shape,
    chain: Chain,
    plain: SimdPipeline<KitVertexShader, ToonShader>,
    ground_ink: SimdPipeline<KitVertexShader, InkGround>,
    sky: SimdPipeline<ScreenVertexShader, InkSky>,
}

impl FlipbookBw {
    pub fn new() -> Self {
        let mut rng = Rng::new(1912);
        let mut land = terrain(&ring_radii(1.5, 230.0, 40), 48, ground, |_, _| grey(0.95));
        let rocks: Vec<Shape> = (0..3).map(|_| boulder(&mut rng)).collect();
        let mut stones = scatter(
            &mut rng,
            26.0,
            6.0,
            600,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 5.0 || x.abs() < 3.2 && z > 0.0 {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.5, 1.2),
        );
        stones.truncate(6);
        stones.iter_mut().for_each(|p| p.position.y -= 0.15);
        let mut rock_batch = Batch::new();
        rock_batch.add(&rocks, &stones, |p| (p.seed * 7.0) as usize % 3);
        land.append(&rock_batch.finish());

        let conifers: Vec<Shape> = (0..4).map(|i| conifer(&mut rng, i == 3)).collect();
        let rounds: Vec<Shape> = (0..4).map(|i| round_tree(&mut rng, i == 3)).collect();
        let tree_spots = scatter(
            &mut rng,
            34.0,
            5.0,
            6000,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 7.0 || x.abs() < 3.4 && z > -2.0 || z > 4.0 {
                    None
                } else {
                    Some(1.0)
                }
            },
            (0.9, 1.5),
        );
        let mut trees = Batch::new();
        let (round, cone): (Vec<_>, Vec<_>) =
            tree_spots.into_iter().take(30).partition(|p| p.seed > 0.65);
        trees.add(&conifers, &cone, |p| (p.seed * 13.0) as usize % 4);
        trees.add(&rounds, &round, |p| (p.seed * 13.0) as usize % 4);

        let tufts: Vec<Shape> = (0..3).map(|_| tuft(&mut rng)).collect();
        let blades = scatter(
            &mut rng,
            24.0,
            2.0,
            2500,
            ground,
            |x, z| {
                if (x * x + z * z).sqrt() < 3.4 || !(-16.0..=8.0).contains(&z) {
                    None
                } else {
                    Some(1.0 + (-z).max(0.0) * 0.15)
                }
            },
            (0.9, 1.5),
        );
        let mut grass = Batch::new();
        grass.add(&tufts, &blades, |p| (p.seed * 11.0) as usize % 3);

        Self {
            land,
            trees: trees.finish(),
            grass: grass.finish(),
            chain: Chain::new(vec![
                Effect::Toon {
                    levels: 3.0,
                    softness: 0.06,
                },
                Effect::Edges(Edges {
                    ink: hdr_of_displayed(grey(0.04)),
                    width: 1.5,
                    sensitivity: 900.0,
                    opacity: 1.0,
                    wobble: 1.8,
                    boil: FPS,
                    ..Edges::default()
                }),
                Effect::Paper(Paper {
                    tint: Vec3::new(1.0, 0.99, 0.96),
                    grain: 0.28,
                    fibre: 0.3,
                    scale: 2.0,
                }),
                Effect::Noise(Noise {
                    amount: 0.03,
                    floor: 0.0,
                    cell: 1.0,
                    chroma: 0.0,
                }),
            ]),
            plain: pipeline(ToonShader, CullingMode::None),
            ground_ink: pipeline(InkGround, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, InkSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

/// The view as it was at the last page turn: time held at the flipbook's frame rate.
fn page(view: &View) -> ViewRef {
    let turned = (view.frame.time * FPS).floor() / FPS;
    Arc::new(View {
        frame: Arc::new(director::frame(turned)),
        ..view.clone()
    })
}

impl Stage for FlipbookBw {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Xylophone],
            ambience: vec![Ambience::once(
                Source::Recording("page", 0.0),
                0.0,
                0.4,
                0.17,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = page(&scene.view);
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let sway = |amplitude, speed| {
            Motion::Sway(Sway {
                amplitude,
                speed,
                wind: WIND,
            })
        };

        self.land
            .draw(pass, &self.ground_ink, motion(Motion::Still), toon.clone());
        self.trees
            .draw(pass, &self.plain, motion(sway(0.3, 1.2)), toon.clone());
        self.grass
            .draw(pass, &self.plain, motion(sway(0.2, 2.4)), toon);
        scene.chip.draw(pass, &view, &look());
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
        Some(self.chain.run(context, scene, image))
    }

    fn caption_ink(&self) -> Vec3 {
        Vec3::new(0.07, 0.07, 0.07)
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.0,
            vignette: 0.3,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

/// The distance from (`u`, `v`) to the outline of a cloud with its flat base on the line
/// `v = 0`, of overlapping circles; negative inside.
#[inline(always)]
fn cloud_distance(u: F, v: F) -> F {
    const PUFFS: [(f32, f32, f32); 5] = [
        (-1.0, 0.1, 0.45),
        (-0.45, 0.4, 0.6),
        (0.25, 0.5, 0.55),
        (0.85, 0.2, 0.45),
        (0.1, 0.1, 0.5),
    ];
    let mut nearest = s(9.0);
    for (cx, cy, radius) in PUFFS {
        let (dx, dy) = (u - s(cx), v - s(cy));
        nearest = nearest.fast_min((dx * dx + dy * dy).sqrt() - s(radius));
    }
    nearest.fast_max(-v)
}

/// The sky in ink on paper: three puffy outlined clouds with a few strokes under each, drifting
/// across, and a sun of two circles and short rays.
struct InkSky;

impl FragmentShaderSimd<ScreenVaryings> for InkSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let time = view.frame.time;
        let (u, height) = (ray.x * s(6.0), ray.y.fast_max(s(0.0)) * s(6.0));

        let mut ink = s(0.0);
        for (home, base, size, speed) in [
            (-1.9, 0.5, 0.9, 0.07),
            (0.5, 0.95, 0.7, 0.05),
            (2.3, 0.35, 1.0, 0.09),
        ] {
            let span = 7.0;
            let along = ((u - s(home + time * speed)) / s(span) + s(0.5)).floor();
            let x = (u - s(home + time * speed)) - along * s(span);
            let (x, y) = (x / s(size), (height - s(base)) / s(size));
            let distance = cloud_distance(x, y) * s(size);
            let outline = s(1.0) - smoothstep(0.0, 0.016, distance.abs());
            let strokes = ((x * s(9.0) + y * s(5.0)).sin() * s(0.5) + s(0.5)).floor()
                * (s(1.0) - smoothstep(0.0, 0.3, (y - s(0.12)).abs() * s(1.0)))
                * (s(1.0) - smoothstep(-0.05, 0.0, distance));
            ink = ink.fast_max(outline).fast_max(strokes * s(0.7));
        }

        let (du, dv) = (u - s(1.2), height - s(1.35));
        let radius = (du * du + dv * dv).sqrt();
        let ring = |r: f32| s(1.0) - smoothstep(0.0, 0.012, (radius - s(r)).abs());
        let spokes = (dv.atan2(du) * s(14.0) + s(time * 0.5)).sin();
        let rays = smoothstep(0.3, 0.6, spokes)
            * smoothstep(0.34, 0.36, radius)
            * (s(1.0) - smoothstep(0.5, 0.52, radius));
        ink = ink.fast_max(ring(0.2)).fast_max(ring(0.27)).fast_max(rays);

        let paper = rgb(0.96, 0.96, 0.95);
        let colour = paper * (s(1.0) - clamp01(ink)) + rgb(0.04, 0.04, 0.05) * clamp01(ink);
        to_colour(expand(colour), s(SKY_DISTANCE))
    }
}
