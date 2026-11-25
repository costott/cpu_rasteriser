//! zen-garden: a dry garden under soft overcast. Gravel raked in rings round mossy stones, a
//! bonsai on its mound, a stone lantern and a tiled wall, misty hills behind and petals drifting
//! down. The chip is a slate stone with a patch of moss.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, hash2, length, lt, mix, mix3, rgb, s, select, smoothstep, splat3, to_colour,
};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonUniforms,
    VariedToonShader, VariedToonUniforms, pipeline,
};
use crate::themes::kit::dress::{chip_shadow, rim_point};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, scatter};
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.45,
    y: 0.5,
    z: -0.55,
};
const WALL_Z: f32 = -26.0;
const WALL_HEIGHT: f32 = 1.7;
const WIND: Vec2 = Vec2 { x: 1.0, y: 0.3 };
/// The distance between rake lines, in world units.
const RAKE: f32 = 0.34;

/// What the rake runs round: where, and how wide, as (x, z, radius). The first is the chip.
const ISLANDS: [(f32, f32, f32); 7] = [
    (0.0, 0.0, 2.5),
    (-7.5, -5.0, 1.6),
    (-4.3, -6.8, 0.9),
    (7.0, -9.0, 2.1),
    (-15.0, -15.0, 2.7),
    (11.0, -3.0, 2.5),
    (4.6, 4.2, 0.75),
];

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.05, 1.0, 0.92),
        sky: Vec3::new(0.46, 0.5, 0.52),
        ground: Vec3::new(0.2, 0.19, 0.16),
        bands: 0.0,
        rim: Vec3::new(0.05, 0.06, 0.06),
        fog: Fog {
            colour: Vec3::new(0.6, 0.64, 0.64),
            density: 0.0045,
            falloff: 0.1,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.2, 0.21, 0.23),
        die: Vec3::new(0.035, 0.08, 0.04),
        gold: Vec3::new(0.42, 0.36, 0.2),
        streams: Vec3::new(0.7, 0.9, 0.7),
        key: light.sun,
        key_colour: light.sun_colour * 0.9,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.2, 0.22, 0.22),
    }
}

/// How far (`x`, `z`) is from the nearest island's edge, and the horizontal direction away from
/// it, for the rake to follow.
#[inline(always)]
fn nearest_island(x: F, z: F) -> (F, F, F) {
    let (mut best, mut gx, mut gz) = (s(1000.0), s(0.0), s(1.0));
    for &(cx, cz, radius) in &ISLANDS {
        let (dx, dz) = (x - s(cx), z - s(cz));
        let len = (dx * dx + dz * dz).sqrt().fast_max(s(1e-3));
        let d = len - s(radius);
        let closer = lt(d, best);
        best = select(closer, d, best);
        gx = select(closer, dx / len, gx);
        gz = select(closer, dz / len, gz);
    }
    (best, gx, gz)
}

/// Gravel raked into rings round every island and into straight lines where nothing is near,
/// the grooves shaded by tilting the surface's normal, with pale pebbles in the grain.
struct GravelShader;

impl FragmentShaderSimd<KitVaryings> for GravelShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let (x, z) = (world.x, world.z);
        let distance = length(splat3(view.eye) - world);
        let (edge, gx, gz) = nearest_island(x, z);

        let waver = fbm(x * s(0.35), z * s(0.35), 2) * s(0.7);
        let rings = (edge * s(TAU / RAKE) + waver).sin();
        let lines = (z * s(TAU / RAKE) + waver * s(1.3)).sin();
        let straight = smoothstep(4.5, 9.0, edge);
        let resolved = clamp01(s(RAKE) / (distance * s(view.pixel_angle)) / s(2.5) - s(0.4));
        let slope = (rings.cos() * (s(1.0) - straight) * s(0.8) + lines.cos() * straight * s(0.8))
            * resolved;
        let tilt_x = gx * slope * (s(1.0) - straight);
        let tilt_z = gz * slope * (s(1.0) - straight) + slope * straight;
        let normal = Vec3Simd::new(-tilt_x * s(1.4), s(1.0), -tilt_z * s(1.4)).normalise();

        let near = clamp01((s(45.0) - distance) * s(0.03));
        let grain = hash2((x * s(55.0)).floor(), (z * s(55.0)).floor());
        let pebble = (grain - s(0.9)).fast_max(s(0.0)) * s(2.2) * near;
        let patch = fbm(x * s(0.5), z * s(0.5), 2);
        let tone = s(0.82) + patch * s(0.25) + (grain - s(0.5)) * s(0.32) * near + pebble * s(0.5);
        let groove = s(0.86)
            + s(0.14) * rings.cos() * (s(1.0) - straight)
            + s(0.14) * lines.cos() * straight;
        let albedo = rgb(0.78, 0.75, 0.68) * tone * mix(s(1.0), groove, resolved);

        let contact = smoothstep(0.0, 1.1, edge.fast_max(s(0.0)));
        let shaded = (s(0.5) + s(0.5) * contact)
            * (s(1.0) - chip_shadow(view, u.lighting.sun, world) * s(0.4));
        let colour = u
            .lighting
            .shade(view, world, normal, albedo * shaded, s(1.0));
        scene_colour(view, colour, world)
    }
}

/// Weathered plaster: a soft cream wash with rain streaks running down and splashed dirt at the
/// foot.
struct PlasterShader;

impl FragmentShaderSimd<KitVaryings> for PlasterShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let normal = facing(v.normal().normalise(), splat3(view.eye) - world);
        let height = world.y - s(FLOOR);
        let streaks = fbm(world.x * s(1.6), world.y * s(0.18), 3);
        let patches = fbm(world.x * s(0.25) + s(4.0), world.y * s(0.4), 3);
        let splash = smoothstep(0.55, 0.0, height).fast_max(s(0.0));
        let mut albedo = mix3(rgb(0.55, 0.5, 0.4), rgb(0.64, 0.6, 0.5), patches);
        albedo = albedo * (s(0.82) + streaks * s(0.34));
        albedo = mix3(albedo, rgb(0.24, 0.2, 0.15), splash * s(0.6));
        let colour = u.lighting.shade(view, world, normal, albedo, s(1.0));
        scene_colour(view, colour, world)
    }
}

/// A flattened boulder of slate, mossy where it faces up.
fn stone(rng: &mut Rng, x: f32, z: f32, size: Vec3) -> Shape {
    let seed = rng.range(0.0, 50.0);
    let slate = Vec3::new(0.27, 0.28, 0.3);
    Shape::icosphere(1.0, 1, 0.2, seed, slate)
        .transformed(
            &(Mat4::translation(x, FLOOR + size.y * 0.5, z)
                * Mat4::rotate_y(seed)
                * Mat4::scaling(size.x, size.y, size.z)),
        )
        .colour_by(|p, n| {
            let moss = ((n.y - 0.45) * 3.0 + (fbm_scalar(p.x * 1.4, p.z * 1.4, 3) - 0.5) * 3.0)
                .clamp(0.0, 1.0);
            let rock = lerp3(
                Vec3::new(0.22, 0.23, 0.26),
                Vec3::new(0.38, 0.39, 0.42),
                fbm_scalar(p.x * 2.0 + 3.0, p.y * 2.0, 2),
            );
            lerp3(rock, Vec3::new(0.1, 0.22, 0.06), moss)
        })
        .facet_variation(rng, 0.08)
}

/// A stone lantern: stepped base, slim column, a lit box and a broad cap.
fn lantern(x: f32, z: f32) -> Shape {
    let grey = Vec3::new(0.3, 0.31, 0.3);
    let mut lantern = Shape::lathe(
        &[
            (0.0, 0.0),
            (0.75, 0.0),
            (0.75, 0.2),
            (0.4, 0.22),
            (0.3, 0.3),
            (0.3, 1.1),
            (0.55, 1.18),
            (0.0, 1.2),
        ],
        8,
        grey,
    );
    lantern.append(&Shape::cuboid(
        Vec3::new(-0.38, 1.2, -0.38),
        Vec3::new(0.38, 1.85, 0.38),
        grey * 0.9,
    ));
    lantern.append(&Shape::cuboid(
        Vec3::new(-0.2, 1.35, -0.4),
        Vec3::new(0.2, 1.7, 0.4),
        Vec3::new(0.03, 0.025, 0.02),
    ));
    lantern.append(&Shape::lathe(
        &[
            (0.0, 1.85),
            (1.0, 1.85),
            (0.85, 2.0),
            (0.2, 2.35),
            (0.0, 2.55),
        ],
        8,
        grey,
    ));
    lantern
        .colour_by(|p, n| {
            let moss = ((0.7 - p.y) * 1.5 + (n.y - 0.3)).clamp(0.0, 1.0)
                * if p.y < 0.5 { 0.8 } else { 0.0 };
            lerp3(
                Vec3::new(0.3, 0.31, 0.3) * (1.0 - 0.2 * (p.y / 2.5).clamp(0.0, 1.0)),
                Vec3::new(0.12, 0.24, 0.07),
                moss,
            )
        })
        .transformed(&Mat4::translation(x, FLOOR, z))
}

/// A cloud-pruned pine or a maple: a leaning trunk and flat pads of foliage, `colours` picked by
/// pad.
fn niwaki(rng: &mut Rng, height: f32, pads: usize, colours: &[Vec3]) -> Shape {
    let bark = Vec3::new(0.15, 0.1, 0.07);
    let mut tree = Shape::new();
    let mut top = Vec3::ZERO;
    for k in 0..4 {
        let next = top + Vec3::new(rng.range(-0.4, 0.4), height * 0.25, rng.range(-0.3, 0.3));
        tree.append(&Shape::limb(
            top,
            next,
            0.28 - 0.05 * k as f32,
            0.25 - 0.05 * k as f32,
            5,
            bark,
        ));
        top = next;
        if k > 0 {
            let reach = rng.range(1.0, 2.0) * height * 0.2;
            let angle = rng.range(0.0, TAU);
            let tip = top + Vec3::new(angle.cos() * reach, height * 0.05, angle.sin() * reach);
            tree.append(&Shape::limb(top, tip, 0.1, 0.05, 4, bark));
            let c = colours[rng.index(colours.len())];
            tree.append(
                &Shape::icosphere(1.0, 1, 0.22, rng.range(0.0, 40.0), c)
                    .transformed(
                        &(Mat4::translation_vec(tip + Vec3::new(0.0, 0.3, 0.0))
                            * Mat4::scaling(height * 0.2, height * 0.1, height * 0.2)),
                    )
                    .facet_variation(rng, 0.12),
            );
        }
    }
    for i in 0..pads {
        let angle = TAU * i as f32 / pads as f32 + rng.range(-0.4, 0.4);
        let reach = rng.range(0.1, 0.45) * height;
        let c = colours[rng.index(colours.len())];
        let centre = top
            + Vec3::new(
                angle.cos() * reach,
                rng.range(-0.15, 0.1) * height,
                angle.sin() * reach,
            );
        let width = rng.range(0.16, 0.26) * height;
        tree.append(
            &Shape::icosphere(1.0, 1, 0.22, rng.range(0.0, 40.0), c)
                .transformed(
                    &(Mat4::translation_vec(centre) * Mat4::scaling(width, width * 0.5, width)),
                )
                .facet_variation(rng, 0.12),
        );
    }
    tree.sway_by_height(height * 0.3, height * 1.1, 1.3)
}

/// The bonsai: a mossy mound with a twisted trunk and three layers of needles.
fn bonsai(rng: &mut Rng, x: f32, z: f32) -> Shape {
    let mut shape = Shape::icosphere(1.0, 1, 0.12, 8.0, Vec3::new(0.1, 0.22, 0.06))
        .transformed(&(Mat4::translation(x, FLOOR, z) * Mat4::scaling(2.4, 0.55, 2.4)))
        .facet_variation(rng, 0.14);
    let pine = niwaki(
        rng,
        2.6,
        4,
        &[Vec3::new(0.05, 0.16, 0.07), Vec3::new(0.08, 0.21, 0.09)],
    );
    shape.append(&pine.transformed(&Mat4::translation(x, FLOOR + 0.45, z)));
    shape
}

/// The wall's tiled cap, as a row of ridged tiles, and the plaster below.
fn wall() -> (Shape, Shape) {
    let top = FLOOR + WALL_HEIGHT;
    let plaster = Shape::cuboid(
        Vec3::new(-40.0, FLOOR - 0.2, WALL_Z - 0.3),
        Vec3::new(40.0, top, WALL_Z + 0.3),
        Vec3::ONE,
    );
    let tile = Vec3::new(0.09, 0.09, 0.1);
    let mut cap = Shape::cuboid(
        Vec3::new(-40.0, top, WALL_Z - 0.55),
        Vec3::new(40.0, top + 0.12, WALL_Z + 0.55),
        tile,
    );
    let count = 160;
    for k in 0..count {
        let x = -40.0 + 80.0 * (k as f32 + 0.5) / count as f32;
        cap.append(&Shape::cuboid(
            Vec3::new(x - 0.1, top + 0.12, WALL_Z - 0.5),
            Vec3::new(x + 0.1, top + 0.3, WALL_Z + 0.5),
            tile * (1.0 + 0.4 * ((k * 7 % 5) as f32 / 5.0)),
        ));
    }
    (plaster, cap)
}

pub struct ZenGarden {
    ground: Shape,
    plaster: Shape,
    solids: Shape,
    foliage: Shape,
    petals: Shape,
    extras: Shape,
    falling: Particles,
    gravel: SimdPipeline<KitVertexShader, GravelShader>,
    wall: SimdPipeline<KitVertexShader, PlasterShader>,
    solid: SimdPipeline<KitVertexShader, VariedToonShader>,
    two_sided: SimdPipeline<KitVertexShader, VariedToonShader>,
    hills: SimdPipeline<ScreenVertexShader, HillSky>,
}

impl ZenGarden {
    pub fn new() -> Self {
        let mut rng = Rng::new(2301);
        let ground = terrain(
            &ring_radii(1.2, 34.0, 40),
            90,
            |_, _| FLOOR,
            |_, _| Vec3::ONE,
        );
        let (plaster, cap) = wall();

        let mut solids = cap;
        for (i, &(x, z, radius)) in ISLANDS.iter().enumerate().skip(1) {
            let size = match i {
                5 => continue,
                6 => Vec3::new(radius, 0.45, radius * 0.9),
                3 => Vec3::new(radius, 1.0, radius * 0.85),
                _ => Vec3::new(radius * 0.7, radius * 1.15, radius * 0.65),
            };
            solids.append(&stone(&mut rng, x, z, size));
        }
        solids.append(&lantern(-10.5, -11.5));

        let mut foliage = bonsai(&mut rng, 11.0, -3.0);
        let green = [
            Vec3::new(0.05, 0.15, 0.07),
            Vec3::new(0.07, 0.19, 0.08),
            Vec3::new(0.1, 0.22, 0.09),
        ];
        let maple = [
            Vec3::new(0.5, 0.14, 0.06),
            Vec3::new(0.62, 0.26, 0.07),
            Vec3::new(0.42, 0.1, 0.05),
        ];
        for &(x, z, height) in &[
            (-19.0, -33.0, 7.0),
            (-7.0, -37.0, 8.5),
            (5.0, -35.0, 6.5),
            (20.0, -33.0, 7.5),
            (-30.0, -36.0, 8.0),
        ] {
            foliage.append(
                &niwaki(&mut rng, height, 7, &green).transformed(&Mat4::translation(x, FLOOR, z)),
            );
        }
        foliage.append(
            &niwaki(&mut rng, 7.0, 9, &maple).transformed(&Mat4::translation(13.0, FLOOR, -31.0)),
        );

        let petal = {
            let mut p = Shape::new();
            p.quad(
                [
                    Vec3::new(0.0, 0.0, -1.0),
                    Vec3::new(-0.6, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.6, 0.0, 0.0),
                ],
                Vec3::new(0.85, 0.58, 0.64),
            );
            p
        };
        let spots = scatter(
            &mut rng,
            13.0,
            1.1,
            1800,
            |_, _| FLOOR + 0.02,
            |x, z| {
                ISLANDS
                    .iter()
                    .all(|&(cx, cz, r)| (x - cx).hypot(z - cz) > r + 0.4)
                    .then_some(1.0)
            },
            (0.1, 0.17),
        );
        let mut petals = Batch::new();
        petals.add(std::slice::from_ref(&petal), &spots, |_| 0);
        let petals = petals.finish().facet_variation(&mut rng, 0.15);

        let mut extras = Shape::new();
        for _ in 0..14 {
            let (spot, _) = rim_point(&mut rng);
            let radius = rng.range(0.2, 0.45);
            extras.append(
                &Shape::icosphere(
                    1.0,
                    1,
                    0.3,
                    rng.range(0.0, 40.0),
                    Vec3::new(0.12, 0.28, 0.07),
                )
                .transformed(
                    &(Mat4::translation_vec(spot) * Mat4::scaling(radius, radius * 0.3, radius)),
                )
                .facet_variation(&mut rng, 0.15),
            );
        }

        Self {
            ground,
            plaster,
            solids,
            foliage,
            petals,
            extras,
            falling: Particles::new(70, 91),
            gravel: pipeline(GravelShader, CullingMode::BackFace),
            wall: pipeline(PlasterShader, CullingMode::BackFace),
            solid: pipeline(VariedToonShader, CullingMode::BackFace),
            two_sided: pipeline(VariedToonShader, CullingMode::None),
            hills: SimdPipeline::new(ScreenVertexShader, HillSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for ZenGarden {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Harp],
            ambience: vec![Ambience::once(
                Source::Recording("chimes", 0.0),
                0.0,
                0.6,
                0.2,
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
        let tree_sway = Motion::Sway(Sway {
            amplitude: 0.18,
            speed: 0.8,
            wind: WIND,
        });

        self.ground
            .draw(pass, &self.gravel, motion(Motion::Still), toon.clone());
        self.plaster
            .draw(pass, &self.wall, motion(Motion::Still), toon.clone());
        self.solids
            .draw(pass, &self.solid, motion(Motion::Still), varied(0.05));
        self.petals
            .draw(pass, &self.two_sided, motion(Motion::Still), varied(0.1));
        self.foliage
            .draw(pass, &self.solid, motion(tree_sway), varied(0.1));

        scene.chip.draw(pass, view, &look());
        self.extras
            .draw(pass, &self.solid, motion(Motion::OnChip), varied(0.1));

        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.hills, (), |_| view.clone());

        self.falling.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 2.0, -3.0),
                half_extent: Vec3::new(14.0, 4.5, 12.0),
                velocity: Vec3::new(0.7, -0.45, 0.2),
                wander: 0.9,
                size: (0.04, 0.09),
                colour: Vec3::new(0.55, 0.34, 0.4),
                twinkle: 0.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.1,
            bloom: 0.12,
            vignette: 0.5,
            grain: 0.02,
            aberration: 0.0,
            desaturate: 0.22,
        }
    }
}

/// An overcast sky with layers of mountains dissolving into mist behind the garden.
struct HillSky;

impl FragmentShaderSimd<ScreenVaryings> for HillSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let t = view.frame.time;
        let up = ray.y.fast_max(s(0.0));
        let azimuth = ray.x.atan2(-ray.z);

        let cloud = fbm(azimuth * s(2.2) + s(t * 0.01), up * s(5.0), 3);
        let mist = rgb(0.6, 0.64, 0.64);
        let mut colour = mix3(mist, rgb(0.46, 0.5, 0.54), smoothstep(0.0, 0.6, up));
        colour = colour * (s(0.9) + cloud * s(0.2));

        for (base, amp, shade, seed) in [
            (0.045, 0.085, 0.8, 1.0),
            (0.02, 0.06, 0.62, 5.0),
            (0.0, 0.035, 0.44, 9.0),
        ] {
            let ridge = s(base)
                + fbm(azimuth * s(2.6), s(seed), 3) * s(amp)
                + fbm(azimuth * s(11.0), s(seed + 3.0), 2) * s(0.006);
            let inside = smoothstep(0.0, 0.003, ridge - up);
            let depth = smoothstep(0.0, 0.07, ridge - up);
            let hill =
                mix3(rgb(0.5, 0.56, 0.56), mist, depth * s(0.8)) * s(shade + 0.15) + mist * s(0.0);
            let hill = mix3(mist, hill, s(shade));
            colour = mix3(colour, hill, inside);
        }
        to_colour(colour, s(SKY_DISTANCE))
    }
}
