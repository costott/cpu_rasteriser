//! A treasure map: the chip on a headland of ink-hatched hills, mountains and tiny pines, a
//! wavy-lined sea with a sailing ship, drawn on stained parchment with a compass rose and a
//! scorched edge that smoulders.

use std::f32::consts::FRAC_1_SQRT_2;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, scene_colour};
use crate::simd::{F, clamp01, from_colour, mix3, rgb, s, smoothstep, splat3, step_lt, to_colour};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonShader,
    ToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::{
    Chain, Edges, Effect, Hatching, Paper, ScreenPass, hdr_of_displayed,
};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, scatter};
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The height of the land where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
const SEA: f32 = FLOOR - 0.45;
const SUN: Vec3 = Vec3 {
    x: -0.6,
    y: 0.5,
    z: 0.45,
};
const INK: Vec3 = Vec3 {
    x: 0.24,
    y: 0.13,
    z: 0.06,
};
const PAPER: Vec3 = Vec3 {
    x: 0.93,
    y: 0.82,
    z: 0.6,
};

/// Where the headland's coast runs: land lies behind (at smaller z than) this line.
fn shore(x: f32) -> f32 {
    let neck = ((x.abs() - 3.0) / 11.0).clamp(0.0, 1.0);
    5.0 - 12.0 * neck * neck * (3.0 - 2.0 * neck) + 1.2 * (x * 0.45).sin()
}

fn smooth(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much of (`x`, `z`) is land: 1 on the headland and the far shore, 0 in open sea.
fn land(x: f32, z: f32) -> f32 {
    smooth(shore(x) + 2.0, shore(x) - 1.0, z).max(smooth(-24.0, -30.0, z))
}

/// The terrain: a flat site round the chip, rolling hills inland and the far shore's ranges.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let open = ((r - 5.0) / 9.0).clamp(0.0, 1.0);
    let hills = fbm_scalar(x * 0.08 + 4.0, z * 0.08, 3) * 4.0 * open;
    let far = smooth(-26.0, -40.0, z);
    let ranges = fbm_scalar(x * 0.05 + 9.0, z * 0.07, 4) * 14.0 * far;
    FLOOR + (land(x, z) - 1.0) * 1.3 + hills * land(x, z) + ranges
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.3, 1.15, 0.9),
        sky: Vec3::new(0.55, 0.5, 0.42),
        ground: Vec3::new(0.35, 0.3, 0.22),
        bands: 4.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: Vec3::new(0.7, 0.58, 0.4),
            density: 0.004,
            falloff: 0.05,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.4, 0.26, 0.14),
        die: Vec3::new(0.8, 0.03, 0.02),
        gold: Vec3::new(0.8, 0.55, 0.2),
        streams: Vec3::new(0.9, 0.7, 0.3),
        key: light.sun,
        key_colour: light.sun_colour,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.7, 0.6, 0.45),
    }
}

fn terrain_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let above = centroid.y - SEA;
    let sand = Vec3::new(0.86, 0.74, 0.5);
    let grass = Vec3::new(0.62, 0.62, 0.38);
    let rock = Vec3::new(0.55, 0.46, 0.36);
    let mut c = lerp3(sand, grass, ((above - 0.5) * 1.5).clamp(0.0, 1.0));
    c = lerp3(
        c,
        rock,
        ((0.85 - normal.y) * 3.0 + (centroid.y - FLOOR - 3.0) * 0.15).clamp(0.0, 0.9),
    );
    c * (0.94 + 0.12 * crate::simd::hash2_scalar(centroid.x * 1.7, centroid.z * 1.3))
}

/// A jagged peak: a cone with a ragged profile.
fn mountain(rng: &mut Rng) -> Shape {
    let (radius, height) = (rng.range(1.0, 1.4), rng.range(1.8, 2.6));
    let profile = [
        (0.0, 0.0),
        (radius, 0.0),
        (radius * rng.range(0.6, 0.75), height * 0.4),
        (radius * rng.range(0.3, 0.45), height * 0.7),
        (radius * rng.range(0.1, 0.2), height * 0.9),
        (0.0, height),
    ];
    Shape::lathe(&profile, 7, Vec3::new(0.6, 0.5, 0.4)).facet_variation(rng, 0.22)
}

fn pine() -> Shape {
    let mut shape = Shape::cylinder(0.04, 0.03, 0.22, 4, Vec3::new(0.35, 0.22, 0.12));
    shape.append(
        &Shape::cone(0.2, 0.5, 5, Vec3::new(0.38, 0.5, 0.28))
            .transformed(&Mat4::translation(0.0, 0.15, 0.0)),
    );
    shape.append(
        &Shape::cone(0.15, 0.38, 5, Vec3::new(0.38, 0.5, 0.28))
            .transformed(&Mat4::translation(0.0, 0.42, 0.0)),
    );
    shape.sway_by_height(0.1, 0.8, 1.0)
}

/// A square-rigger with its nose along +x, standing on y = 0.
fn ship() -> Shape {
    let wood = Vec3::new(0.3, 0.17, 0.08);
    let sail = Vec3::new(0.95, 0.88, 0.7);
    let mut shape = Shape::new();
    shape.append(
        &Shape::icosphere(0.5, 1, 0.0, 1.0, wood)
            .transformed(&(Mat4::translation(0.0, 0.1, 0.0) * Mat4::scaling(1.9, 0.55, 0.75))),
    );
    shape.append(&Shape::cuboid(
        Vec3::new(-0.85, 0.15, -0.34),
        Vec3::new(-0.3, 0.5, 0.34),
        wood * 1.1,
    ));
    shape.append(&Shape::limb(
        Vec3::new(0.0, 0.1, 0.0),
        Vec3::new(0.0, 2.7, 0.0),
        0.05,
        0.03,
        5,
        wood,
    ));
    shape.append(&Shape::limb(
        Vec3::new(0.7, 0.35, 0.0),
        Vec3::new(1.6, 0.55, 0.0),
        0.035,
        0.025,
        4,
        wood,
    ));
    shape.triangle(
        Vec3::new(-0.05, 0.55, 0.06),
        Vec3::new(-1.0, 0.5, 0.06),
        Vec3::new(-0.05, 2.5, 0.06),
        sail,
    );
    shape.triangle(
        Vec3::new(0.05, 0.55, -0.06),
        Vec3::new(0.05, 2.4, -0.06),
        Vec3::new(1.4, 0.55, -0.06),
        sail,
    );
    shape.triangle(
        Vec3::new(0.0, 2.7, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(-0.4, 2.85, 0.0),
        Vec3::new(0.7, 0.1, 0.08),
    );
    shape.transformed(&Mat4::scaling(0.5, 0.5, 0.5))
}

/// Sea: pale wash with wavy ink lines that drift, breaking into a dotted line along the shore.
struct SeaShader;

impl FragmentShaderSimd<KitVaryings> for SeaShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let time = view.frame.time;
        let distance = crate::simd::length(splat3(view.eye) - world);
        let aa = distance * s(view.pixel_angle * 1.5);
        let (x, z) = (world.x, world.z);
        let warp = fbm(x * s(0.12), z * s(0.12) + s(time * 0.03), 2) * s(3.0);
        let phase = z * s(0.6) + warp + (x * s(0.8) + s(time * 0.6)).sin() * s(0.25);
        let d = (phase - phase.floor() - s(0.5)).abs();
        let wave = clamp01((d - s(0.36) - aa) / (s(0.06) + aa)) * smoothstep(60.0, 22.0, distance);

        let neck = clamp01((x.abs() - s(3.0)) * s(1.0 / 11.0));
        let coast = s(5.0) - s(12.0) * neck * neck * (s(3.0) - s(2.0) * neck)
            + (x * s(0.45)).sin() * s(1.2);
        let foam_band = smoothstep(
            0.0,
            1.0,
            clamp01(s(1.0) - (z - coast - s(1.4)).abs() * s(1.2)),
        );
        let along = x * s(2.2) + s((time * 0.4).sin());
        let dots = step_lt(along - along.floor(), s(0.45));
        let foam = foam_band * dots;

        let albedo = mix3(rgb(0.62, 0.72, 0.72), rgb(0.18, 0.26, 0.3), wave);
        let albedo = mix3(albedo, rgb(0.95, 0.95, 0.9), foam * s(0.7));
        let normal = Vec3Simd::new(s(0.0), s(1.0), s(0.0));
        scene_colour(
            view,
            u.lighting.shade(view, world, normal, albedo, s(1.0)),
            world,
        )
    }
}

/// The paper behind everything: a plain warm wash.
struct PaperShader;

impl FragmentShaderSimd<ScreenVaryings> for PaperShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, _: &()) -> ColourSimd {
        let [_, y] = v.ndc();
        let tone = mix3(
            rgb(0.85, 0.72, 0.5),
            rgb(0.95, 0.84, 0.62),
            y * s(0.5) + s(0.5),
        );
        to_colour(tone, s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
struct FinishUniforms {
    source: Arc<TextureSampler>,
    time: f32,
    aspect: f32,
}

/// What ink and fire do to the finished drawing: foxing and folds in the paper, a double-ruled
/// border, a compass rose that sways, and a scorched edge with a flickering ember line.
struct FinishShader;

/// A diamond `half_width` across and `half_length` long along v, positive inside.
#[inline(always)]
fn diamond(u: F, v: F, half_width: f32, half_length: f32) -> F {
    s(1.0) - u.abs() * s(1.0 / half_width) - v.abs() * s(1.0 / half_length)
}

#[inline(always)]
fn rule(distance: F, width: f32) -> F {
    s(1.0) - smoothstep(width, width * 2.0, distance.abs())
}

impl FragmentShaderSimd<ScreenVaryings> for FinishShader {
    type Uniforms = FinishUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &FinishUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let sample = u.source.sample_linear_clamp_simd([x, y]);
        let (px, py) = (x * s(u.aspect), y);
        let ink = splat3(hdr_of_displayed(INK));

        let stain = fbm(px * s(1.3) + s(3.0), py * s(1.3), 4);
        let blotch = smoothstep(0.55, 0.85, fbm(px * s(2.7), py * s(2.7) + s(9.0), 3));
        let crease = |a: F, centre: f32| {
            let d = (a - s(centre)).abs();
            (s(1.0) - smoothstep(0.0, 0.01, d)) * s(0.14)
                + (s(1.0) - smoothstep(0.0, 0.12, d)) * s(0.05)
        };
        let folds = crease(px, u.aspect * 0.5) + crease(py, 0.5);
        let mut colour =
            from_colour(sample) * (s(0.9) + stain * s(0.18) - blotch * s(0.16) - folds);

        let edge = (px.fast_min(s(u.aspect) - px)).fast_min(py.fast_min(s(1.0) - py));
        let border =
            (rule(edge - s(0.075), 0.0018) + rule(edge - s(0.088), 0.0007)).fast_min(s(1.0));
        colour = mix3(colour, ink, border * s(0.85));

        let (cx, cy, radius) = (u.aspect - 0.21, 0.76, 0.1);
        let sway = (u.time * 0.5).sin() * 0.12;
        let (dx0, dy0) = (px - s(cx), py - s(cy));
        let (dx, dy) = (
            dx0 * s(sway.cos()) - dy0 * s(sway.sin()),
            dx0 * s(sway.sin()) + dy0 * s(sway.cos()),
        );
        let rho = (dx0 * dx0 + dy0 * dy0).sqrt();
        let plate = splat3(hdr_of_displayed(PAPER * 0.96));
        colour = mix3(
            colour,
            plate,
            (s(1.0) - smoothstep(radius * 1.05, radius * 1.2, rho)) * s(0.95),
        );
        let rings = (rule(rho - s(radius), 0.0016)
            + rule(rho - s(radius * 0.82), 0.0009)
            + rule(rho - s(radius * 0.64), 0.0007))
        .fast_min(s(1.0));
        let (diag_u, diag_v) = ((dx + dy) * s(FRAC_1_SQRT_2), (dy - dx) * s(FRAC_1_SQRT_2));
        let cardinal_v = diamond(dx, dy, radius * 0.17, radius * 1.0);
        let cardinal_h = diamond(dy, dx, radius * 0.17, radius * 1.0);
        let minor_a = diamond(diag_u, diag_v, radius * 0.12, radius * 0.64);
        let minor_b = diamond(diag_v, diag_u, radius * 0.12, radius * 0.64);
        let inside = |d: F| step_lt(s(0.0), d);
        let outline = |d: F| inside(d) * (s(1.0) - smoothstep(0.0, 0.07, d));
        let shapes =
            outline(cardinal_v) + outline(cardinal_h) + outline(minor_a) + outline(minor_b);
        let dark =
            inside(cardinal_v) * step_lt(dx, s(0.0)) + inside(cardinal_h) * step_lt(dy, s(0.0));
        let star = inside(cardinal_v)
            .fast_max(inside(cardinal_h))
            .fast_max(inside(minor_a))
            .fast_max(inside(minor_b));
        let north = inside(cardinal_v) * step_lt(s(radius * 0.45), dy);
        let compass = (rings + shapes + dark * s(0.6)).fast_min(s(1.0));
        let wax = splat3(hdr_of_displayed(Vec3::new(0.62, 0.1, 0.07)));
        colour = mix3(colour, ink, compass * s(0.85));
        colour = mix3(colour, wax, north * star * s(0.9));

        let wobble = fbm(px * s(9.0), py * s(9.0), 3) - s(0.5);
        let d = edge + wobble * s(0.05);
        let char = smoothstep(0.03, 0.012, d);
        let scorch = smoothstep(0.13, 0.03, d);
        let band = smoothstep(0.012, 0.025, d) * smoothstep(0.055, 0.035, d);
        let flicker = s(0.6) + s(0.4) * (s(u.time * 3.0) + wobble * s(24.0)).sin();
        colour = colour * (s(1.0) - scorch * s(0.6));
        colour = mix3(colour, rgb(0.03, 0.02, 0.015), char);
        colour = colour + rgb(1.6, 0.55, 0.12) * (band * flicker * s(0.9));
        to_colour(colour, sample.a)
    }
}

pub struct ParchmentMap {
    land: Shape,
    trees: Shape,
    peaks: Shape,
    ship: Shape,
    sailing: Shape,
    sea: Shape,
    dust: Particles,
    land_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    sea_pipeline: SimdPipeline<KitVertexShader, SeaShader>,
    paper_pipeline: SimdPipeline<ScreenVertexShader, PaperShader>,
    chain: Chain,
    finish: ScreenPass<FinishShader>,
}

impl ParchmentMap {
    pub fn new() -> Self {
        let mut rng = Rng::new(1492);
        let hills = terrain(&ring_radii(1.5, 160.0, 44), 64, ground, terrain_colour);

        let mut sea = Shape::new();
        let far = 400.0;
        sea.quad(
            [
                Vec3::new(-far, SEA, far),
                Vec3::new(far, SEA, far),
                Vec3::new(far, SEA, -far),
                Vec3::new(-far, SEA, -far),
            ],
            Vec3::ONE,
        );

        let on_land = |x: f32, z: f32| land(x, z) > 0.97;
        let pines = scatter(
            &mut rng,
            60.0,
            1.3,
            9000,
            ground,
            |x, z| {
                (on_land(x, z)
                    && (x * x + z * z).sqrt() > 5.5
                    && z < FLOOR.abs() * 10.0
                    && !(x.abs() < 3.0 && z > 0.0))
                    .then_some(1.0)
            },
            (0.7, 1.3),
        );
        let mut batch = Batch::new();
        batch.add(&[pine()], &pines, |_| 0);
        let trees = batch.finish();

        let ridge = scatter(
            &mut rng,
            60.0,
            4.8,
            2500,
            ground,
            |x, z| {
                ((on_land(x, z) && z < -7.0 && (x * x + z * z).sqrt() > 9.0) || z < -30.0)
                    .then_some(1.0)
            },
            (0.7, 1.6),
        );
        let variants: Vec<Shape> = (0..4).map(|_| mountain(&mut rng)).collect();
        let mut batch = Batch::new();
        batch.add(&variants, &ridge, |p| {
            (p.seed * 13.0) as usize % variants.len()
        });
        let peaks = batch.finish();

        Self {
            land: hills,
            trees,
            peaks,
            ship: ship(),
            sailing: Shape::new(),
            sea,
            dust: Particles::new(120, 77),
            land_pipeline: pipeline(ToonShader, CullingMode::None),
            sea_pipeline: pipeline(SeaShader, CullingMode::None),
            paper_pipeline: SimdPipeline::new(ScreenVertexShader, PaperShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            chain: Chain::new(vec![
                Effect::Hatching(Hatching {
                    ink: INK,
                    paper: PAPER,
                    spacing: 4.2,
                    width: 1.7,
                    angle: 0.9,
                    follow: 0.75,
                    layers: 4,
                    wobble: 1.0,
                    keep_image: 0.5,
                }),
                Effect::Edges(Edges {
                    ink: hdr_of_displayed(INK),
                    width: 0.9,
                    sensitivity: 1500.0,
                    opacity: 0.9,
                    paper: Vec3::ONE,
                    keep_image: 1.0,
                    wobble: 0.8,
                    boil: 0.0,
                }),
                Effect::Paper(Paper {
                    tint: Vec3::ONE,
                    grain: 0.35,
                    fibre: 0.3,
                    scale: 2.2,
                }),
            ]),
            finish: ScreenPass::new(FinishShader),
        }
    }
}

impl Stage for ParchmentMap {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Harp],
            ambience: vec![
                Ambience::once(Source::Recording("page", 0.0), 0.0, 0.9, 0.22, -0.3),
                Ambience::bed(Source::Recording("quill", 0.0), 0.12, 0.25),
            ],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let time = scene.view.frame.time;
        let x = 2.2 * (time * 0.1).sin();
        let z = 8.2 + 0.3 * (time * 0.07 + 1.0).sin();
        let (dx, dz) = (0.22 * (time * 0.1).cos(), 0.021 * (time * 0.07 + 1.0).cos());
        let bob = 0.07 * (time * 1.9).sin();
        let pose = Mat4::translation(x, SEA + 0.05 + bob, z)
            * Mat4::rotate_y((-dz).atan2(dx))
            * Mat4::rotate_x(0.04 * (time * 1.3).sin());
        self.sailing = self.ship.transformed(&pose);
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
        let breeze = Motion::Sway(Sway {
            amplitude: 0.05,
            speed: 1.2,
            wind: Vec2::new(1.0, 0.0),
        });

        self.land.draw(
            pass,
            &self.land_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.sea.draw(
            pass,
            &self.sea_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.peaks.draw(
            pass,
            &self.land_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.trees
            .draw(pass, &self.land_pipeline, motion(breeze), toon.clone());
        self.sailing.draw(
            pass,
            &self.land_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        scene.chip.draw(pass, view, &look());

        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.paper_pipeline, (), |_| ());

        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 0.0),
                half_extent: Vec3::new(14.0, 5.0, 12.0),
                velocity: Vec3::new(0.08, 0.1, 0.0),
                wander: 0.7,
                size: (0.015, 0.04),
                colour: Vec3::new(1.2, 0.9, 0.5),
                twinkle: 1.2,
            },
        );
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let drawn = self.chain.run(context, scene, image);
        let aspect = scene.extent.width as f32 / scene.extent.height as f32;
        let uniforms = FinishUniforms {
            source: drawn,
            time: scene.local_time,
            aspect,
        };
        Some(self.finish.run(context, scene, uniforms))
    }

    fn caption_ink(&self) -> Vec3 {
        INK
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.5,
            vignette: 0.2,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
