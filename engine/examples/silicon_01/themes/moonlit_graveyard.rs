//! moonlit-graveyard: blue-black silhouettes under one huge pale moon. Crooked dead trees,
//! tilted headstones and iron fences stand in drifting ground mist while bats cross the moon,
//! and a single stroke of lightning lights the whole scene for an instant.

use std::f32::consts::{FRAC_PI_2, TAU};

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{clamp01, length, mix3, s, smoothstep, splat3, to_colour};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonUniforms,
    pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{cellular, fbm, fbm_scalar, value};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shape::{KitVertex, Shape, lerp3};
use crate::themes::kit::sky::stars;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// The direction towards the moon.
const MOON: Vec3 = Vec3 {
    x: 0.26,
    y: 0.14,
    z: -0.95,
};
const MOON_RADIUS: f32 = 0.085;
/// When, in seconds after the world appears, the lightning strikes: almost at once, as the world
/// is only on screen for about a third of a second.
const STRIKE: f32 = 0.06;
const WIND: Vec2 = Vec2 { x: 0.8, y: 0.6 };
const MIST: Vec3 = Vec3 {
    x: 0.085,
    y: 0.14,
    z: 0.25,
};

/// The lightning's brightness `time` seconds into the world: a main stroke and a weaker
/// return stroke a fraction of a second later.
fn flash(time: f32) -> f32 {
    let spike = |at: f32| {
        let dt = time - at;
        if dt < 0.0 { 0.0 } else { (-dt * 11.0).exp() }
    };
    spike(STRIKE) + 0.6 * spike(STRIKE + 0.16)
}

fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let flat = (r / 6.0).clamp(0.0, 1.0);
    let swell = (fbm_scalar(x * 0.07 + 3.0, z * 0.07, 3) - 0.5) * 2.4
        + (fbm_scalar(x * 0.4, z * 0.4, 2) - 0.5) * 0.3;
    let ridge_edge = ((r - 60.0) / 60.0).clamp(0.0, 1.0);
    let ridge = ridge_edge * ridge_edge * (3.0 + 16.0 * fbm_scalar(x * 0.025 + 8.0, z * 0.025, 3));
    FLOOR + flat * flat * swell + ridge
}

fn ground_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let patch = fbm_scalar(centroid.x * 0.15 + 5.0, centroid.z * 0.15, 3);
    let turf = lerp3(
        Vec3::new(0.03, 0.05, 0.05),
        Vec3::new(0.06, 0.09, 0.08),
        patch,
    );
    let r = (centroid.x * centroid.x + centroid.z * centroid.z).sqrt();
    let earth = Vec3::new(0.07, 0.06, 0.06);
    let c = lerp3(
        turf,
        earth,
        ((patch - 0.62) * 4.0).clamp(0.0, 0.8) + (1.0 - r / 3.5).clamp(0.0, 0.5),
    );
    lerp3(
        c,
        Vec3::new(0.1, 0.1, 0.12),
        ((0.9 - normal.y) * 5.0).clamp(0.0, 0.6),
    )
}

fn lighting(flash: f32) -> Lighting {
    Lighting {
        sun: MOON.normalise(),
        sun_colour: Vec3::new(0.5, 0.65, 1.0) * (1.0 + flash * 5.0),
        sky: Vec3::new(0.07, 0.11, 0.2) * (1.0 + flash * 14.0),
        ground: Vec3::new(0.02, 0.03, 0.05) * (1.0 + flash * 6.0),
        bands: 2.0,
        rim: Vec3::new(0.08, 0.12, 0.24) * (1.0 + flash * 4.0),
        fog: Fog {
            colour: Vec3::ZERO,
            density: 0.0,
            falloff: 0.0,
            base: FLOOR,
        },
    }
}

/// The chip as a weathered stone sarcophagus lit by cold light from the front and the moon.
fn look(flash: f32) -> Look {
    Look {
        substrate: Vec3::new(0.2, 0.22, 0.27),
        die: Vec3::new(0.04, 0.05, 0.09),
        gold: Vec3::new(0.38, 0.4, 0.46),
        streams: Vec3::new(0.35, 0.55, 1.0),
        key: Vec3::new(-0.45, 0.35, 0.8).normalise(),
        key_colour: Vec3::new(0.4, 0.5, 0.8) * (1.0 + flash * 4.0),
        sky: Vec3::new(0.06, 0.09, 0.17) * (1.0 + flash * 8.0),
        ground: Vec3::new(0.02, 0.03, 0.05),
        softbox: Vec3::new(0.05, 0.08, 0.15),
    }
}

fn stone() -> Vec3 {
    Vec3::new(0.2, 0.22, 0.27)
}

/// A headstone of one of four kinds, leaning a little off true.
fn headstone(kind: usize, rng: &mut Rng) -> Shape {
    let stone = stone();
    let mut shape = match kind {
        0 => {
            let mut slab = Shape::cuboid(
                Vec3::new(-0.45, 0.0, -0.1),
                Vec3::new(0.45, 1.2, 0.1),
                stone,
            );
            slab.append(
                &Shape::cylinder(0.45, 0.45, 0.2, 10, stone)
                    .transformed(&(Mat4::translation(0.0, 1.2, -0.1) * Mat4::rotate_x(FRAC_PI_2))),
            );
            slab
        }
        1 => {
            let mut cross = Shape::cuboid(
                Vec3::new(-0.12, 0.0, -0.1),
                Vec3::new(0.12, 1.6, 0.1),
                stone,
            );
            cross.append(&Shape::cuboid(
                Vec3::new(-0.45, 1.0, -0.1),
                Vec3::new(0.45, 1.28, 0.1),
                stone,
            ));
            cross
        }
        2 => {
            let mut obelisk =
                Shape::cuboid(Vec3::new(-0.4, 0.0, -0.4), Vec3::new(0.4, 0.4, 0.4), stone);
            obelisk.append(
                &Shape::cylinder(0.3, 0.08, 1.9, 4, stone)
                    .transformed(&(Mat4::translation(0.0, 0.4, 0.0) * Mat4::rotate_y(0.785))),
            );
            obelisk
        }
        _ => Shape::cuboid(
            Vec3::new(-0.65, 0.0, -0.4),
            Vec3::new(0.65, 0.4, 0.4),
            stone,
        ),
    };
    let lean = Mat4::rotate_z(rng.range(-0.22, 0.22)) * Mat4::rotate_x(rng.range(-0.12, 0.12));
    shape = shape.transformed(&(Mat4::translation(0.0, -0.25, 0.0) * lean));
    shape
        .colour_by(|p, _| {
            let moss = fbm_scalar(p.x * 5.0 + 2.0, p.y * 5.0 + p.z * 5.0, 2);
            lerp3(
                stone,
                Vec3::new(0.09, 0.15, 0.12),
                ((moss - 0.5) * 4.0).clamp(0.0, 0.8) * (1.0 - p.y).clamp(0.2, 1.0),
            )
        })
        .facet_variation(rng, 0.15)
}

/// A bare tree: a crooked trunk that forks three times into twigs.
fn dead_tree(rng: &mut Rng) -> Shape {
    let mut tree = Shape::new();
    let bark = Vec3::new(0.02, 0.025, 0.035);
    let lean = Vec3::new(rng.range(-0.15, 0.15), 1.0, rng.range(-0.15, 0.15));
    grow(
        &mut tree,
        rng,
        bark,
        Vec3::ZERO,
        lean,
        Branch {
            length: 3.6,
            radius: 0.3,
            depth: 3,
        },
    );
    tree.sway_by_height(2.0, 10.0, 1.6)
}

#[derive(Clone, Copy)]
struct Branch {
    length: f32,
    radius: f32,
    /// How many more times it forks.
    depth: u32,
}

fn grow(tree: &mut Shape, rng: &mut Rng, bark: Vec3, from: Vec3, direction: Vec3, branch: Branch) {
    let Branch {
        length,
        radius,
        depth,
    } = branch;
    let bend = Vec3::new(
        rng.range(-0.35, 0.35),
        rng.range(-0.1, 0.2),
        rng.range(-0.35, 0.35),
    );
    let direction = (direction + bend).normalise();
    let to = from + direction * length;
    tree.append(&Shape::limb(from, to, radius, radius * 0.62, 4, bark));
    if depth == 0 {
        return;
    }
    let forks = if depth == 3 { 3 } else { 2 };
    for _ in 0..forks {
        let spread = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-0.2, 0.7),
            rng.range(-1.0, 1.0),
        );
        let child = (direction + spread * 0.9).normalise();
        let next = Branch {
            length: length * rng.range(0.6, 0.78),
            radius: radius * 0.6,
            depth: depth - 1,
        };
        grow(tree, rng, bark, to, child, next);
    }
}

/// A few blades of dead grass, which give the bare ground something to stand in.
fn dead_grass(rng: &mut Rng) -> Shape {
    let mut tuft = Shape::new();
    for _ in 0..5 {
        let angle = rng.range(0.0, TAU);
        let (height, lean) = (rng.range(0.4, 0.9), rng.range(0.1, 0.4));
        let base = Vec3::new(angle.cos() * 0.08, 0.0, angle.sin() * 0.08);
        let side = Vec3::new(-angle.sin(), 0.0, angle.cos()) * 0.04;
        let tip = base + Vec3::new(angle.cos() * lean, height, angle.sin() * lean);
        tuft.triangle(base - side, base + side, tip, Vec3::new(0.05, 0.07, 0.07));
    }
    tuft.sway_by_height(0.0, 0.9, 1.0)
}

/// A run of iron railings along x, `pickets` spears between two rails.
fn fence(pickets: usize, rng: &mut Rng) -> Shape {
    let iron = Vec3::new(0.012, 0.014, 0.022);
    let mut fence = Shape::new();
    let gap = 0.34;
    let length = gap * pickets as f32;
    for rail in [0.35, 1.15] {
        fence.append(&Shape::cuboid(
            Vec3::new(0.0, rail, -0.03),
            Vec3::new(length, rail + 0.07, 0.03),
            iron,
        ));
    }
    for i in 0..pickets {
        let x = gap * (i as f32 + 0.5);
        let lean = if rng.unit() < 0.12 {
            rng.range(-0.3, 0.3)
        } else {
            0.0
        };
        let mut picket = Shape::cuboid(
            Vec3::new(-0.025, 0.0, -0.025),
            Vec3::new(0.025, 1.5, 0.025),
            iron,
        );
        picket.append(
            &Shape::cone(0.06, 0.22, 4, iron).transformed(&Mat4::translation(0.0, 1.5, 0.0)),
        );
        fence.append(&picket.transformed(&(Mat4::translation(x, 0.0, 0.0) * Mat4::rotate_z(lean))));
    }
    fence
}

/// A bat seen from the front, nose along +x and wings spanning z; the vertex sway weight is how
/// far a vertex is along its wing.
fn bat() -> Shape {
    let black = Vec3::new(0.008, 0.01, 0.018);
    let mut bat =
        Shape::icosphere(0.13, 0, 0.0, 0.0, black).transformed(&Mat4::scaling(1.4, 0.8, 0.8));
    for side in [-1.0, 1.0] {
        let shoulder = Vec3::new(0.05, 0.0, side * 0.1);
        let tips = [
            Vec3::new(0.2, 0.0, side * 0.55),
            Vec3::new(-0.1, 0.0, side * 1.0),
            Vec3::new(-0.3, 0.0, side * 0.7),
            Vec3::new(-0.28, 0.0, side * 0.4),
            Vec3::new(-0.18, 0.0, side * 0.12),
        ];
        for pair in tips.windows(2) {
            let (a, b) = if side > 0.0 {
                (pair[1], pair[0])
            } else {
                (pair[0], pair[1])
            };
            bat.triangle(shoulder, a, b, black);
            bat.triangle(shoulder, b, a, black);
        }
    }
    bat.sway_by(|p| (p.z.abs() / 1.0).clamp(0.0, 1.0))
}

#[derive(Clone)]
struct BatUniforms {
    view: ViewRef,
}

#[derive(Interpolate, SimdInterpolate)]
struct BatVaryings {
    world: Vec3,
    colour: Vec3,
}

/// Flies each bat round a wide circle in front of the moon, beating its wings.
struct BatVertexShader;

impl VertexShader for BatVertexShader {
    type Vertex = KitVertex;
    type Uniforms = BatUniforms;
    type Varyings = BatVaryings;

    fn shade(&self, vertex: KitVertex, u: &BatUniforms) -> (Vec4, BatVaryings) {
        let time = u.view.frame.time;
        let seed = vertex.seed;
        let pick = |k: f32| (seed * k).fract();
        let radius = 14.0 + 16.0 * pick(7.31);
        let direction = if pick(29.3) < 0.5 { -1.0 } else { 1.0 };
        let angle = seed * TAU * 5.0 + time * (5.5 + 3.0 * pick(13.17)) / radius * direction;
        let flap = (time * (9.0 + 4.0 * pick(3.7)) + seed * 40.0).sin();
        let height = 8.0 + 7.0 * pick(3.91) + (time * 0.8 + seed * 30.0).sin() * 0.8;
        let heading = (-angle.sin() * direction, angle.cos() * direction);
        let local = Vec3::new(
            vertex.position.x,
            vertex.position.y + vertex.sway * flap * 0.55,
            vertex.position.z,
        );
        let scale = 2.2;
        let (x, z) = (local.x * scale, local.z * scale);
        let world = Vec3::new(
            radius * angle.cos() + x * heading.0 - z * heading.1,
            height + local.y * scale,
            -32.0 + radius * angle.sin() + x * heading.1 + z * heading.0,
        );
        (
            u.view.view_projection * world.to_point4(),
            BatVaryings {
                world,
                colour: vertex.colour,
            },
        )
    }
}

/// Bats are black: a hint of moonlight on the wings and nothing else.
struct BatShader;

impl FragmentShaderSimd<BatVaryings> for BatShader {
    type Uniforms = BatUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: BatVaryingsSimd, u: &BatUniforms) -> ColourSimd {
        let world = Vec3Simd::new(v.world[0], v.world[1], v.world[2]);
        let colour = Vec3Simd::new(v.colour[0], v.colour[1], v.colour[2]);
        scene_colour(&u.view, colour, world)
    }
}

struct GloomUniforms {
    toon: ToonUniforms,
    flash: f32,
}

/// Everything standing on the ground: two-tone moonlight, blotched by moss and weather, sunk in
/// mist that thickens towards the ground and drifts.
struct GloomShader;

impl FragmentShaderSimd<KitVaryings> for GloomShader {
    type Uniforms = GloomUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GloomUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        let weathering = fbm(world.x * s(0.6), world.z * s(0.6) + world.y * s(1.1), 2);
        let albedo = v.colour() * (s(0.7) + weathering * s(0.6));
        let lit = u.toon.lighting.shade(view, world, normal, albedo, s(1.0));

        let t = view.frame.time;
        let drift = fbm(
            world.x * s(0.05) + s(t * 0.04),
            world.z * s(0.05) - s(t * 0.025),
            3,
        );
        let above = (world.y - s(FLOOR)).fast_max(s(0.0));
        let density = s(0.02) * (-above * s(0.3)).exp() * (s(0.35) + drift * s(1.4));
        let amount = clamp01(s(1.0) - (-density * length(to_eye)).exp());
        let mist = splat3(MIST * (1.0 + u.flash * 6.0));
        scene_colour(view, mix3(lit, mist, amount), world)
    }
}

pub struct MoonlitGraveyard {
    land: Shape,
    trees: Shape,
    stones: Shape,
    grass: Shape,
    fences: Shape,
    bats: Shape,
    gloom: SimdPipeline<KitVertexShader, GloomShader>,
    flier: SimdPipeline<BatVertexShader, BatShader>,
    sky: SimdPipeline<ScreenVertexShader, NightSky>,
}

impl MoonlitGraveyard {
    pub fn new() -> Self {
        let mut rng = Rng::new(1818);
        let land = terrain(&ring_radii(1.5, 230.0, 44), 60, ground, ground_colour);

        let clear = |x: f32, z: f32| !(x.abs() < 3.4 && z > -3.0) && (x * x + z * z).sqrt() > 4.5;
        let trees: Vec<Shape> = (0..3).map(|_| dead_tree(&mut rng)).collect();
        let tree_spots = scatter(
            &mut rng,
            70.0,
            12.0,
            3000,
            ground,
            |x, z| {
                if clear(x, z) && z < 2.0 {
                    Some(1.0)
                } else {
                    None
                }
            },
            (0.9, 1.6),
        );
        let mut forest = Batch::new();
        forest.add(&trees, &tree_spots[..tree_spots.len().min(16)], |p| {
            (p.seed * 13.0) as usize % 3
        });

        let kinds: Vec<Shape> = (0..8).map(|i| headstone(i % 4, &mut rng)).collect();
        let grave_spots = scatter(
            &mut rng,
            30.0,
            2.7,
            5000,
            ground,
            |x, z| {
                if clear(x, z) && z < 5.0 && z > -34.0 {
                    Some(1.0)
                } else {
                    None
                }
            },
            (0.8, 1.5),
        );
        let mut stones = Batch::new();
        stones.add(&kinds, &grave_spots[..grave_spots.len().min(44)], |p| {
            (p.seed * 17.0) as usize % 8
        });

        let tufts: Vec<Shape> = (0..3).map(|_| dead_grass(&mut rng)).collect();
        let blades = scatter(
            &mut rng,
            26.0,
            2.2,
            6000,
            ground,
            |x, z| {
                if clear(x, z) && z < 9.0 {
                    Some(1.0)
                } else {
                    None
                }
            },
            (0.6, 1.1),
        );
        let mut grass = Batch::new();
        grass.add(&tufts, &blades, |p| (p.seed * 11.0) as usize % 3);

        let runs: Vec<Shape> = (0..3).map(|i| fence(14 + i * 4, &mut rng)).collect();
        let at = |x: f32, z: f32, yaw: f32| Placement {
            position: Vec3::new(x, ground(x, z) - 0.05, z),
            yaw,
            scale: 1.0,
            seed: 0.5,
        };
        let spots = [
            at(-34.0, -14.0, 0.05),
            at(-23.0, -13.0, -0.04),
            at(8.0, -15.0, 0.1),
            at(20.0, -14.2, -0.06),
            at(-16.0, -3.0, 0.5),
        ];
        let mut fences = Batch::new();
        fences.add(&runs, &spots, |p| {
            (p.position.x as i32).unsigned_abs() as usize % 3
        });

        let flock: Vec<Placement> = (0..14)
            .map(|i| Placement {
                position: Vec3::ZERO,
                yaw: 0.0,
                scale: 1.0,
                seed: (i as f32 * 0.0713 + 0.031).fract(),
            })
            .collect();
        let mut bats = Batch::new();
        bats.add(&[bat()], &flock, |_| 0);

        Self {
            land,
            trees: forest.finish(),
            stones: stones.finish(),
            grass: grass.finish(),
            fences: fences.finish(),
            bats: bats.finish(),
            gloom: pipeline(GloomShader, CullingMode::None),
            flier: SimdPipeline::new(BatVertexShader, BatShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DEFAULT),
            sky: SimdPipeline::new(ScreenVertexShader, NightSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for MoonlitGraveyard {
    fn sound(&self) -> WorldSound {
        let thunder = Source::Recording("thunder", 0.0);
        WorldSound {
            leads: &[Instrument::Organ],
            ambience: vec![
                Ambience::once(thunder, STRIKE, 0.6, 0.3, -0.2),
                Ambience::once(thunder, STRIKE + 0.16, 0.45, 0.16, 0.3),
            ],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let flash = flash(scene.local_time);
        let uniforms = || GloomUniforms {
            toon: ToonUniforms {
                view: view.clone(),
                lighting: lighting(flash),
            },
            flash,
        };
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let sway = Motion::Sway(Sway {
            amplitude: 0.25,
            speed: 0.8,
            wind: WIND,
        });

        self.land
            .draw(pass, &self.gloom, motion(Motion::Still), uniforms());
        self.trees.draw(pass, &self.gloom, motion(sway), uniforms());
        self.stones
            .draw(pass, &self.gloom, motion(Motion::Still), uniforms());
        self.fences
            .draw(pass, &self.gloom, motion(Motion::Still), uniforms());
        self.grass.draw(
            pass,
            &self.gloom,
            motion(Motion::Sway(Sway {
                amplitude: 0.2,
                speed: 1.6,
                wind: WIND,
            })),
            uniforms(),
        );
        scene.chip.draw(pass, view, &look(flash));
        self.bats.draw(
            pass,
            &self.flier,
            BatUniforms { view: view.clone() },
            BatUniforms { view: view.clone() },
        );
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| (view.clone(), flash));
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.6,
            vignette: 0.55,
            grain: 0.035,
            aberration: 0.002,
            desaturate: 0.25,
        }
    }
}

/// The night sky: faint stars, streaky cloud, a cratered moon with a halo, a low band of mist
/// and, during the strike, a jagged bolt and a sky lit from behind.
struct NightSky;

impl FragmentShaderSimd<ScreenVaryings> for NightSky {
    type Uniforms = (ViewRef, f32);

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, (view, flash): &(ViewRef, f32)) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let time = s(view.frame.time);
        let up = ray.y.fast_max(s(0.0));
        let flash = *flash;

        let zenith = Vec3::new(0.004, 0.01, 0.035);
        let horizon = Vec3::new(0.05, 0.1, 0.19);
        let mut colour = mix3(
            splat3(horizon),
            splat3(zenith),
            smoothstep(0.0, 0.55, up).sqrt(),
        );
        colour = colour + splat3(horizon) * ((-up * s(9.0)).exp() * s(0.7));

        let moon = MOON.normalise();
        let cos_moon = ray.dot(splat3(moon));
        let angle = cos_moon.fast_min(s(1.0)).acos();
        let dark = clamp01(s(1.0) - (colour.x + colour.y + colour.z) * s(5.0));
        colour = colour
            + splat3(Vec3::new(0.8, 0.9, 1.0))
                * (stars(ray.x, ray.y, ray.z, 0.55, time) * dark * smoothstep(0.03, 0.15, up));

        let streak = fbm(
            ray.x * s(4.0) / (up + s(0.3)) + time * s(0.02),
            up * s(14.0),
            4,
        );
        let cloud = smoothstep(0.55, 0.85, streak) * smoothstep(0.04, 0.3, up);
        let near_moon = (-angle * s(3.0)).exp();
        colour =
            colour + splat3(Vec3::new(0.35, 0.45, 0.7)) * (cloud * (s(0.05) + near_moon * s(0.5)));

        let right = Vec3::new(0.0, 1.0, 0.0).cross(moon).normalise();
        let top = moon.cross(right);
        let (px, py) = (
            ray.dot(splat3(right)) / s(MOON_RADIUS),
            ray.dot(splat3(top)) / s(MOON_RADIUS),
        );
        let rim = (s(1.0) - px * px - py * py).fast_max(s(0.0));
        let maria = fbm(px * s(1.5) + s(3.0), py * s(1.5), 4);
        let crater = smoothstep(0.1, 0.3, cellular(px * s(5.0), py * s(5.0), 0.0));
        let surface = (s(0.62) + smoothstep(0.35, 0.65, maria) * s(0.38))
            * (s(0.82) + crater * s(0.18))
            * (s(0.55) + rim.sqrt() * s(0.45));
        let disc = s(1.0) - smoothstep(MOON_RADIUS * 0.985, MOON_RADIUS, angle);
        let halo = (-angle * s(9.0)).exp() * s(0.55) + near_moon * s(0.14);
        colour = colour + splat3(Vec3::new(0.55, 0.7, 1.0)) * halo;
        colour = mix3(colour, splat3(Vec3::new(1.15, 1.25, 1.5)) * surface, disc);

        if flash > 0.0 {
            let tx = ray.x / (-ray.z).fast_max(s(0.05));
            let ty = ray.y / (-ray.z).fast_max(s(0.05));
            let jag = (value(ty * s(38.0) + s(7.0), s(1.0)) - s(0.5)) * s(0.05)
                + (value(ty * s(11.0) + s(2.0), s(5.0)) - s(0.5)) * s(0.12);
            let distance = (tx - s(-0.3) - ty * s(0.25) - jag).abs();
            let reach = smoothstep(0.0, 0.03, ty) * (s(1.0) - smoothstep(0.34, 0.4, ty));
            let bolt = (-distance * distance * s(30000.0)).exp() * s(2.5)
                + (-distance * distance * s(120.0)).exp() * s(0.35);
            colour = colour
                + splat3(Vec3::new(0.7, 0.8, 1.0)) * (bolt * reach * s(flash))
                + splat3(Vec3::new(0.2, 0.28, 0.5))
                    * (s(flash * 1.4) * (s(0.35) + streak * s(0.9)));
        }
        to_colour(colour, s(SKY_DISTANCE))
    }
}
