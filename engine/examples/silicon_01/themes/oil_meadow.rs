//! oil-meadow: an impasto landscape. Sunflowers and cypresses on rolling hills under a sky of
//! swirling brushwork, the finished frame smoothed into patches of paint and laid on canvas.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, ge, hash2, mix3, mul3, rgb, s, select3, smoothstep, splat3, to_colour,
};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonUniforms,
    VariedToonShader, VariedToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::post::{Chain, Effect, Noise, Paper, expand, hdr_of_displayed};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement, scatter};
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: 0.5,
    y: 0.34,
    z: -0.8,
};
const WIND: Vec2 = Vec2 { x: 0.9, y: 0.4 };

/// Gentle swells round the chip, with hills that grow into a ridge beyond the flowers.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let clearing = (r / 8.0).clamp(0.0, 1.0);
    let swell = (fbm_scalar(x * 0.035 + 4.0, z * 0.05, 3) - 0.5) * 4.5;
    let ridge_edge = ((r - 48.0) / 70.0).clamp(0.0, 1.0);
    let ridge =
        ridge_edge * ridge_edge * (6.0 + 26.0 * fbm_scalar(x * 0.016 + 9.0, z * 0.02 + 2.0, 3));
    FLOOR + clearing * clearing * swell + ridge
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(2.7, 2.2, 1.3),
        sky: Vec3::new(0.34, 0.4, 0.8),
        ground: Vec3::new(0.3, 0.22, 0.1),
        bands: 3.0,
        rim: Vec3::new(0.25, 0.2, 0.08),
        fog: Fog {
            colour: hdr_of_displayed(Vec3::new(0.62, 0.7, 0.8)),
            density: 0.0035,
            falloff: 0.04,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.62, 0.42, 0.16),
        die: Vec3::new(0.06, 0.14, 0.42),
        gold: Vec3::new(0.95, 0.72, 0.18),
        streams: Vec3::new(1.0, 0.8, 0.25),
        key: light.sun,
        key_colour: light.sun_colour * 0.6,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.3, 0.26, 0.15),
    }
}

/// A sunflower facing +z, a stalk 1.8 tall with leaves and a head of two rings of petals round a
/// domed seed disc.
fn sunflower(rng: &mut Rng) -> Shape {
    let height = 1.8;
    let leaf_green = Vec3::new(0.14, 0.4, 0.07);
    let mut flower = Shape::limb(
        Vec3::ZERO,
        Vec3::new(0.0, height, -0.06),
        0.065,
        0.045,
        5,
        Vec3::new(0.18, 0.4, 0.08),
    );
    for (y, side, length) in [(0.55, 1.0, 0.7), (0.95, -1.0, 0.62), (1.3, 1.0, 0.5)] {
        let spine: Vec<Vec3> = (0..4)
            .map(|i| {
                let t = i as f32 / 3.0;
                Vec3::new(side * t * length, y + 0.12 * t - 0.3 * t * t, 0.02)
            })
            .collect();
        flower.append(&Shape::ribbon(
            &spine,
            &[0.06, 0.3, 0.26, 0.0],
            Vec3::new(0.0, 0.0, 1.0),
            leaf_green,
        ));
    }
    let mut head = Shape::lathe(
        &[
            (0.0, -0.05),
            (0.2, -0.02),
            (0.2, 0.04),
            (0.12, 0.08),
            (0.0, 0.1),
        ],
        9,
        Vec3::new(0.26, 0.12, 0.04),
    )
    .colour_by(|p, _| {
        lerp3(
            Vec3::new(0.46, 0.22, 0.05),
            Vec3::new(0.14, 0.07, 0.03),
            (p.y / 0.1).clamp(0.0, 1.0),
        )
    })
    .transformed(&Mat4::rotate_x(FRAC_PI_2));
    let petals = 15;
    for (ring, (inner, outer, colour, depth)) in [
        (0.19, 0.52, Vec3::new(0.95, 0.5, 0.04), -0.025),
        (0.19, 0.44, Vec3::new(1.0, 0.78, 0.1), 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        for k in 0..petals {
            let angle =
                TAU * (k as f32 + 0.5 * ring as f32) / petals as f32 + rng.range(-0.05, 0.05);
            let spoke = |radius: f32, turn: f32| {
                let a = angle + turn;
                Vec3::new(a.cos() * radius, a.sin() * radius, depth)
            };
            let reach = outer * rng.range(0.9, 1.08);
            let (mid, half) = (inner + (reach - inner) * 0.4, 0.2);
            head.quad(
                [
                    spoke(inner, -0.14),
                    spoke(inner, 0.14),
                    spoke(mid, half),
                    spoke(mid, -half),
                ],
                colour,
            );
            head.triangle(
                spoke(mid, -half),
                spoke(mid, half),
                spoke(reach, 0.0),
                colour,
            );
        }
    }
    flower
        .append(&head.transformed(&(Mat4::translation(0.0, height, -0.04) * Mat4::rotate_x(0.12))));
    flower.sway_by_height(0.3, height, 1.6)
}

/// Field colours: ochre, gold, green, olive, burnt orange and a deeper green.
const FIELDS: [Vec3; 6] = [
    Vec3 {
        x: 0.86,
        y: 0.62,
        z: 0.1,
    },
    Vec3 {
        x: 0.95,
        y: 0.78,
        z: 0.2,
    },
    Vec3 {
        x: 0.2,
        y: 0.5,
        z: 0.14,
    },
    Vec3 {
        x: 0.4,
        y: 0.5,
        z: 0.1,
    },
    Vec3 {
        x: 0.78,
        y: 0.4,
        z: 0.08,
    },
    Vec3 {
        x: 0.12,
        y: 0.38,
        z: 0.16,
    },
];

/// The colour of field number `index` (0 to 5).
#[inline(always)]
fn field_colour(index: F) -> Vec3Simd {
    let mut colour = splat3(FIELDS[0]);
    for (i, field) in FIELDS.iter().enumerate().skip(1) {
        colour = select3(ge(index, s(i as f32)), splat3(*field), colour);
    }
    colour
}

/// The track that winds from the chip's feet to the horizon.
#[inline(always)]
fn path_offset(z: F) -> F {
    s(5.0) * (z * s(0.045) + s(0.6)).sin() + z * s(0.04)
}

/// The ground as a patchwork of fields with ragged brushed edges, worked over with long strokes
/// of lighter and darker paint and rows running away from the viewer, crossed by a pale track
/// and cooling to violet blue with distance. All of it is computed per pixel, so it stays
/// painterly however close the camera is.
struct MeadowShader;

impl FragmentShaderSimd<KitVaryings> for MeadowShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let world = v.world();
        let to_eye = splat3(u.view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        let (x, z) = (world.x, world.z);

        let warp = fbm(x * s(0.03) + s(3.0), z * s(0.03), 2) * s(1.6)
            + fbm(x * s(0.5), z * s(0.5), 2) * s(0.3);
        let pick = hash2((x * s(0.15) + warp).floor(), (z * s(0.2) + warp).floor());
        let index = (pick * s(FIELDS.len() as f32)).floor();
        let mottle = smoothstep(0.42, 0.62, fbm(x * s(0.7), z * s(1.8), 3));
        let field = mix3(
            field_colour(index),
            field_colour((index + s(2.0)) % s(FIELDS.len() as f32)),
            mottle,
        );

        let stroke_warp = fbm(x * s(0.2), z * s(0.3), 2);
        let stroke = smoothstep(0.3, 0.7, fbm(x * s(0.45), z * s(1.1), 3));
        let rows = (x * s(2.1) + stroke_warp * s(7.0)).sin() * s(0.5) + s(0.5);
        let greener = smoothstep(0.5, 0.75, fbm(x * s(0.16) + s(5.0), z * s(0.2), 2)) * s(0.8);
        let tinted = mix3(field, mul3(field, rgb(0.55, 0.95, 0.55)), greener);
        let worked = tinted * (s(0.35) + stroke * s(0.9) + rows * s(0.45));

        let off_path = ((x - path_offset(z)).abs() / s(1.8) - s(0.55))
            .fast_max(s(0.0))
            .fast_min(s(1.0));
        let near = s(1.0) - clamp01((z + s(60.0)) / s(80.0));
        let track = mix3(worked, rgb(0.82, 0.68, 0.42), (s(1.0) - off_path) * near);

        let haze = fbm(x * s(0.02) + s(6.0), z * s(0.02), 3);
        let distant = mix3(rgb(0.22, 0.3, 0.64), rgb(0.46, 0.38, 0.62), haze);
        let albedo = mix3(
            track,
            distant,
            smoothstep(40.0, 90.0, (x * x + z * z).sqrt()),
        );

        let colour = u.lighting.shade(&u.view, world, normal, albedo, s(1.0));
        scene_colour(&u.view, colour, world)
    }
}

/// A dark, flame-shaped cypress whose foliage twists in a spiral of two greens.
fn cypress(rng: &mut Rng) -> Shape {
    let mut tree = Shape::cylinder(0.3, 0.2, 1.0, 5, Vec3::new(0.24, 0.16, 0.08));
    let body = Shape::lathe(
        &[
            (0.0, 0.8),
            (0.95, 0.8),
            (1.15, 2.6),
            (1.0, 5.2),
            (0.72, 7.6),
            (0.38, 9.8),
            (0.0, 12.2),
        ],
        9,
        Vec3::ONE,
    )
    .colour_by(|p, _| {
        let spiral = (p.z.atan2(p.x) * 2.0 + p.y * 1.1).sin() * 0.5 + 0.5;
        lerp3(
            Vec3::new(0.03, 0.12, 0.08),
            Vec3::new(0.11, 0.3, 0.17),
            spiral,
        )
    });
    tree.append(&body);
    tree.facet_variation(rng, 0.12)
        .sway_by_height(1.5, 12.2, 1.5)
}

/// A tuft of grass or wheat, which the paint filter turns into a dab of colour.
fn tuft(rng: &mut Rng) -> Shape {
    let mut tuft = Shape::new();
    for _ in 0..4 {
        let angle = rng.range(0.0, TAU);
        let (height, lean) = (rng.range(0.4, 0.8), rng.range(0.1, 0.4));
        let base = Vec3::new(angle.cos() * 0.1, 0.0, angle.sin() * 0.1);
        let side = Vec3::new(-angle.sin(), 0.0, angle.cos()) * 0.06;
        let tip = base + Vec3::new(angle.cos() * lean, height, angle.sin() * lean);
        tuft.triangle(base - side, base + side, tip, Vec3::ONE);
    }
    tuft.gradient_by_height(
        0.0,
        0.9,
        Vec3::new(0.3, 0.5, 0.1),
        Vec3::new(1.0, 0.85, 0.3),
    )
    .sway_by_height(0.0, 0.9, 1.0)
}

/// Where the chip stands clear: no props in front of it.
fn clear_of_chip(x: f32, z: f32) -> bool {
    !(x.abs() < 3.4 && z > -2.0 && z < 15.0) && (x * x + z * z).sqrt() > 4.5
}

pub struct OilMeadow {
    land: Shape,
    flowers: Shape,
    trees: Shape,
    grass: Shape,
    chain: Chain,
    meadow: SimdPipeline<KitVertexShader, MeadowShader>,
    solid: SimdPipeline<KitVertexShader, VariedToonShader>,
    foliage: SimdPipeline<KitVertexShader, VariedToonShader>,
    sky: SimdPipeline<ScreenVertexShader, SwirlSky>,
}

impl OilMeadow {
    pub fn new() -> Self {
        let mut rng = Rng::new(1889);
        let land = terrain(&ring_radii(1.5, 230.0, 54), 110, ground, |_, _| Vec3::ONE);

        let blooms: Vec<Shape> = (0..4).map(|_| sunflower(&mut rng)).collect();
        let mut spots = scatter(
            &mut rng,
            34.0,
            3.0,
            4000,
            ground,
            |x, z| {
                if clear_of_chip(x, z) && z < 14.0 {
                    Some(1.0)
                } else {
                    None
                }
            },
            (1.3, 2.1),
        );
        spots.truncate(60);
        spots.iter_mut().for_each(|p| p.yaw = rng.range(-0.5, 0.5));
        let mut flowers = Batch::new();
        flowers.add(&blooms, &spots, |p| (p.seed * 13.0) as usize % blooms.len());

        let tufts: Vec<Shape> = (0..4).map(|_| tuft(&mut rng)).collect();
        let blades = scatter(
            &mut rng,
            30.0,
            0.9,
            6000,
            ground,
            |x, z| {
                if clear_of_chip(x, z) && z < 13.0 {
                    Some(1.0 + (x * x + z * z).sqrt() * 0.04)
                } else {
                    None
                }
            },
            (0.6, 1.2),
        );
        let mut grass = Batch::new();
        grass.add(&tufts, &blades, |p| (p.seed * 11.0) as usize % 4);

        let cypresses: Vec<Shape> = (0..3).map(|_| cypress(&mut rng)).collect();
        let anchors = [
            (-11.0, -14.0, 0.9),
            (12.0, -20.0, 1.0),
            (-24.0, -34.0, 1.3),
            (26.0, -44.0, 1.2),
            (4.0, -70.0, 1.6),
        ];
        let trees: Vec<Placement> = anchors
            .iter()
            .enumerate()
            .map(|(i, &(x, z, scale))| Placement {
                position: Vec3::new(x, ground(x, z), z),
                yaw: i as f32 * 1.9,
                scale,
                seed: (i as f32 * 0.37 + 0.11).fract(),
            })
            .collect();
        let mut forest = Batch::new();
        forest.add(&cypresses, &trees, |p| {
            (p.seed * 7.0) as usize % cypresses.len()
        });

        Self {
            land,
            flowers: flowers.finish(),
            trees: forest.finish(),
            grass: grass.finish(),
            chain: Chain::new(vec![
                Effect::Kuwahara { radius: 5.0 },
                Effect::Paper(Paper {
                    tint: Vec3::new(1.0, 0.98, 0.94),
                    grain: 0.3,
                    fibre: 0.12,
                    scale: 1.8,
                }),
                Effect::Noise(Noise {
                    amount: 0.04,
                    floor: 0.0,
                    cell: 1.0,
                    chroma: 0.0,
                }),
            ]),
            meadow: pipeline(MeadowShader, CullingMode::BackFace),
            solid: pipeline(VariedToonShader, CullingMode::BackFace),
            foliage: pipeline(VariedToonShader, CullingMode::None),
            sky: SimdPipeline::new(ScreenVertexShader, SwirlSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for OilMeadow {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Flute],
            ambience: vec![Ambience::once(
                Source::Recording("birdsong", 0.0),
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
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };
        let varied = |variation| VariedToonUniforms {
            toon: toon.clone(),
            variation,
        };
        let flower_sway = Motion::Sway(Sway {
            amplitude: 0.3,
            speed: 1.4,
            wind: WIND,
        });
        let tree_sway = Motion::Sway(Sway {
            amplitude: 0.35,
            speed: 0.9,
            wind: WIND,
        });

        self.land
            .draw(pass, &self.meadow, motion(Motion::Still), toon.clone());
        self.trees
            .draw(pass, &self.solid, motion(tree_sway), varied(0.12));
        self.grass
            .draw(pass, &self.foliage, motion(flower_sway), varied(0.18));
        self.flowers
            .draw(pass, &self.foliage, motion(flower_sway), varied(0.1));
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
        Some(self.chain.run(context, scene, image))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.25,
            vignette: 0.35,
            grain: 0.0,
            aberration: 0.0,
            desaturate: -0.12,
        }
    }
}

/// Turns the plane (`x`, `y`) about (`cx`, `cy`) by an angle that fades with distance: a vortex.
#[inline(always)]
fn vortex(x: F, y: F, cx: f32, cy: f32, turn: F, size: f32) -> (F, F) {
    let (dx, dy) = (x - s(cx), y - s(cy));
    let angle = turn * (-(dx * dx + dy * dy) * s(1.0 / (size * size))).exp();
    let (sin, cos) = (angle.sin(), angle.cos());
    (s(cx) + dx * cos - dy * sin, s(cy) + dx * sin + dy * cos)
}

/// A sky of brush strokes: bands of blue and cream drawn along a flow that curls into two slowly
/// breathing vortices, a haloed moon, and a warm yellow haze at the horizon.
struct SwirlSky;

impl FragmentShaderSimd<ScreenVaryings> for SwirlSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let t = view.frame.time;
        let height = ray.y.fast_max(s(0.0));
        let (x, y) = (ray.x * s(9.0), height * s(9.0));

        let breathe = |phase: f32| s(2.6) + s(0.7) * (s(t * 0.35 + phase)).sin();
        let (x, y) = vortex(x, y, -1.1, 1.05, breathe(0.0), 0.85);
        let (x, y) = vortex(x, y, 0.9, 1.5, -breathe(2.0), 0.7);

        let flow = fbm(x * s(0.5) + s(t * 0.03), y * s(0.8), 3);
        let stroke = (y * s(6.5) + flow * s(7.0) + s(t * 0.15)).sin() * s(0.5) + s(0.5);
        let detail = fbm(x * s(1.4), y * s(4.0) - s(t * 0.1), 3);
        let tone = clamp01(stroke * s(0.7) + detail * s(0.45) - s(0.1));

        let deep = rgb(0.03, 0.08, 0.4);
        let cobalt = rgb(0.1, 0.27, 0.72);
        let sky = rgb(0.42, 0.74, 0.9);
        let cream = rgb(0.98, 0.92, 0.62);
        let mut colour = mix3(deep, cobalt, smoothstep(0.1, 0.45, tone));
        colour = mix3(colour, sky, smoothstep(0.5, 0.78, tone));
        colour = mix3(
            colour,
            cream,
            smoothstep(0.82, 0.98, tone) * smoothstep(0.1, 0.4, flow),
        );
        let haze = smoothstep(0.0, 0.5, height);
        colour = mix3(rgb(0.98, 0.78, 0.38), colour, haze.sqrt());

        let moon = splat3(Vec3::new(0.5, 0.17, -0.85).normalise());
        let angle = ray.dot(moon).fast_min(s(1.0)).acos();
        let disc = s(1.0) - smoothstep(0.052, 0.06, angle);
        let rings = ((angle * s(120.0)) - s(t * 0.8)).sin() * s(0.5) + s(0.5);
        let halo = (-angle * s(10.0)).exp() * rings * s(0.55) + (-angle * s(5.0)).exp() * s(0.3);
        colour = mix3(colour, rgb(1.0, 0.92, 0.5), clamp01(halo));
        colour = mix3(colour, rgb(1.0, 0.97, 0.7), disc);
        to_colour(expand(colour), s(SKY_DISTANCE))
    }
}
