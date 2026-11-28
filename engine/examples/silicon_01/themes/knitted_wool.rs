//! A knitted landscape: rolling hills in stripes of wool, lollipop trees, a woolly sky with a
//! round sun, all worked in stockinette. The finished frame is knitted: every stitch is a V of
//! two plump strands in the colour beneath it, lit from above, with stray fibres fuzzing the
//! silhouettes and drifting lint in the air.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, fract, from_colour, hash2, lt, mix3, rgb, s, select3, smoothstep, splat3, step_lt,
    to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Sway, ToonUniforms,
    VariedToonShader, VariedToonUniforms, pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::fbm;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::{ScreenPass, expand, hdr_of_displayed};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The height of the knitted meadow where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.4,
    y: 0.7,
    z: 0.5,
};
const WIND: Vec2 = Vec2 { x: 1.0, y: 0.2 };

/// Rolling hills that swell with distance, in long waves across the view.
fn ground(x: f32, z: f32) -> f32 {
    let reach = ((-z - 6.0) / 30.0).clamp(0.0, 1.0);
    let waves = 4.0 + 3.0 * (x * 0.07 + z * 0.03).sin() + 1.8 * (x * 0.16 - z * 0.05 + 1.0).sin();
    FLOOR + reach * reach * (waves + 0.12 * (-z - 6.0).clamp(0.0, 90.0))
}

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(0.85, 0.8, 0.7),
        sky: Vec3::new(0.38, 0.4, 0.46),
        ground: Vec3::new(0.3, 0.27, 0.24),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: hdr_of_displayed(Vec3::new(0.8, 0.86, 0.9)),
            density: 0.002,
            falloff: 0.03,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.82, 0.72, 0.52),
        die: Vec3::new(0.08, 0.34, 0.4),
        gold: Vec3::new(0.9, 0.6, 0.15),
        streams: Vec3::new(0.95, 0.45, 0.2),
        key: light.sun,
        key_colour: light.sun_colour * 0.6,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.55, 0.5, 0.45),
    }
}

/// The wools the landscape is knitted from: mustard, moss, teal, terracotta, cream and rose.
const WOOLS: [Vec3; 6] = [
    Vec3 {
        x: 0.8,
        y: 0.52,
        z: 0.1,
    },
    Vec3 {
        x: 0.3,
        y: 0.42,
        z: 0.14,
    },
    Vec3 {
        x: 0.1,
        y: 0.38,
        z: 0.4,
    },
    Vec3 {
        x: 0.7,
        y: 0.28,
        z: 0.16,
    },
    Vec3 {
        x: 0.9,
        y: 0.8,
        z: 0.6,
    },
    Vec3 {
        x: 0.75,
        y: 0.38,
        z: 0.4,
    },
];

/// A lollipop tree: a brown trunk under a round crown of green wool.
fn tree(rng: &mut Rng) -> Shape {
    let mut tree = Shape::cylinder(0.14, 0.1, 1.6, 6, Vec3::new(0.45, 0.3, 0.18));
    let green = [Vec3::new(0.32, 0.5, 0.2), Vec3::new(0.2, 0.45, 0.35)][rng.index(2)];
    tree.append(
        &Shape::icosphere(0.95, 1, 0.06, rng.unit() * 9.0, green)
            .transformed(&Mat4::translation(0.0, 2.2, 0.0)),
    );
    tree.sway_by_height(0.3, 3.2, 1.2)
}

/// The meadow: stripes of wool that wander across the hills.
struct StripeShader;

impl FragmentShaderSimd<KitVaryings> for StripeShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let (x, z) = (world.x, world.z);
        let wander = (x * s(0.25) + z * s(0.1)).sin() * s(0.5) + (x * s(0.09)).sin() * s(0.8);
        let row = (-z * s(0.3) + wander + world.y * s(0.25)).floor();
        let pick = (fract(row * s(0.618)) * s(5.999)).floor();
        let mut albedo = splat3(WOOLS[0]);
        for (i, wool) in WOOLS.iter().enumerate().skip(1) {
            albedo = select3(lt(s(i as f32 - 0.5), pick), splat3(*wool), albedo);
        }
        let to_eye = splat3(view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        scene_colour(
            view,
            u.lighting.shade(view, world, normal, albedo, s(1.0)),
            world,
        )
    }
}

/// The sky: bands of blue with cream clouds that drift, and a round orange sun.
struct SkyShader;

impl FragmentShaderSimd<ScreenVaryings> for SkyShader {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let time = s(view.frame.time);
        let up = ray.y.fast_max(s(0.0));
        let band = (up * s(14.0)).floor();
        let blue = mix3(
            rgb(0.62, 0.8, 0.86),
            rgb(0.3, 0.55, 0.7),
            clamp01(band * s(0.18)),
        );
        let along = ray.x * s(6.0) + time * s(0.06);
        let puff = fbm(along, up * s(18.0), 3);
        let cloud = smoothstep(0.55, 0.62, puff) * smoothstep(0.05, 0.12, up);
        let mut colour = mix3(blue, rgb(0.97, 0.94, 0.86), cloud);
        let sun = splat3(Vec3::new(0.35, 0.3, -0.89).normalise());
        let angle = ray.dot(sun).fast_min(s(1.0)).acos();
        colour = mix3(
            colour,
            rgb(0.95, 0.55, 0.15),
            smoothstep(0.085, 0.075, angle),
        );
        to_colour(expand(colour), s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
struct StitchUniforms {
    source: Arc<TextureSampler>,
    time: f32,
    width: f32,
    height: f32,
}

/// Knits the frame: stockinette stitches whose two strands are plump ridges lit from above.
struct KnitShader;

/// The distance from `p` to the segment `a` to `b`.
#[inline(always)]
fn segment(px: F, py: F, a: (f32, f32), b: (f32, f32)) -> F {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let t = clamp01(
        ((px - s(a.0)) * s(abx) + (py - s(a.1)) * s(aby)) * s(1.0 / (abx * abx + aby * aby)),
    );
    let (dx, dy) = (px - s(a.0) - t * s(abx), py - s(a.1) - t * s(aby));
    (dx * dx + dy * dy).sqrt()
}

impl FragmentShaderSimd<ScreenVaryings> for KnitShader {
    type Uniforms = StitchUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &StitchUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let stitch = 11.0 * u.height / 720.0;
        let aspect = 0.85;
        let (px, py) = (x * s(u.width), y * s(u.height));
        let (cx, cy) = (
            (px * s(1.0 / stitch)).floor(),
            (py * s(1.0 / (stitch * aspect))).floor(),
        );
        let (lu, lv) = (
            px * s(1.0 / stitch) - cx,
            (py * s(1.0 / (stitch * aspect)) - cy) * s(aspect),
        );

        let centre = [
            (cx + s(0.5)) * s(stitch),
            (cy + s(0.5)) * s(stitch * aspect),
        ];
        let sample = u.source.sample_linear_clamp_simd([
            centre[0] * s(1.0 / u.width),
            centre[1] * s(1.0 / u.height),
        ]);
        let colour = from_colour(sample);

        let top = aspect + 0.08;
        let left = segment(lu, lv, (0.04, top), (0.5, -0.06));
        let right = segment(lu, lv, (0.96, top), (0.5, -0.06));
        let radius = s(0.27);
        let round = |d: F| clamp01(s(1.0) - (d / radius) * (d / radius)).sqrt();
        let (hl, hr) = (round(left), round(right));
        let height = hl.fast_max(hr);
        let side = (hr - hl) * s(0.5);
        let crevice = s(0.35) + smoothstep(0.0, 0.7, height) * s(0.8);
        let light =
            s(0.75) + height * s(0.25) + side * s(0.2) + (lv * s(1.0 / aspect) - s(0.5)) * s(0.15);

        let (fx, fy) = (px * s(0.7), py * s(0.7));
        let fibre = fbm(fx, fy, 2) - s(0.5);
        let strand = ((lu * s(9.0) + lv * s(14.0)).sin()) * s(0.04);
        let variety = hash2(cx + s(3.0), cy + s(9.0)) - s(0.5);
        let wool =
            colour * crevice * light * (s(1.0) + fibre * s(0.35) + strand + variety * s(0.12));

        let reach = stitch * 1.4;
        let depth = |dx: f32, dy: f32| {
            u.source
                .sample_linear_clamp_simd([
                    x + s(dx * reach / u.width),
                    y + s(dy * reach / u.height),
                ])
                .a
                .fast_min(s(60.0))
        };
        let d = sample.a.fast_min(s(60.0));
        let rim = ((depth(1.0, 0.0) - d).abs()
            + (depth(-1.0, 0.0) - d).abs()
            + (depth(0.0, 1.0) - d).abs()
            + (depth(0.0, -1.0) - d).abs())
            / d.fast_max(s(1.0));
        let edge = smoothstep(0.1, 0.5, rim);
        let flick = hash2(px.floor() + s(u.time * 3.0).floor(), py.floor());
        let halo =
            edge * step_lt(s(0.55), flick) * smoothstep(0.3, 0.7, fbm(fx * s(0.4), fy * s(0.4), 2));
        let fuzz = mix3(wool, colour * s(1.3) + rgb(0.12, 0.1, 0.08), halo * s(0.7));
        to_colour(fuzz, sample.a)
    }
}

pub struct KnittedWool {
    land: Shape,
    trees: Shape,
    lint: Particles,
    land_pipeline: SimdPipeline<KitVertexShader, StripeShader>,
    tree_pipeline: SimdPipeline<KitVertexShader, VariedToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyShader>,
    knit: ScreenPass<KnitShader>,
}

impl KnittedWool {
    pub fn new() -> Self {
        let mut rng = Rng::new(1952);
        let land = terrain(&ring_radii(1.5, 260.0, 52), 96, ground, |_, _| Vec3::ONE);
        let shapes: Vec<Shape> = (0..3).map(|_| tree(&mut rng)).collect();
        let places: Vec<Placement> = [
            (-7.0, -4.0, 1.1),
            (6.5, -9.0, 1.3),
            (-13.0, -18.0, 1.5),
            (16.0, -26.0, 1.8),
            (2.0, -40.0, 2.4),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(x, z, scale))| Placement {
            position: Vec3::new(x, ground(x, z), z),
            yaw: i as f32,
            scale,
            seed: (i as f32 * 0.31 + 0.07).fract(),
        })
        .collect();
        let mut batch = Batch::new();
        batch.add(&shapes, &places, |p| (p.seed * 7.0) as usize % shapes.len());
        Self {
            land,
            trees: batch.finish(),
            lint: Particles::new(100, 61),
            land_pipeline: pipeline(StripeShader, CullingMode::BackFace),
            tree_pipeline: pipeline(VariedToonShader, CullingMode::None),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            knit: ScreenPass::new(KnitShader),
        }
    }
}

impl Stage for KnittedWool {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Pizzicato],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::SoftThump),
                0.0,
                0.45,
                0.22,
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
        let breeze = Motion::Sway(Sway {
            amplitude: 0.25,
            speed: 1.0,
            wind: WIND,
        });

        self.land.draw(
            pass,
            &self.land_pipeline,
            motion(Motion::Still),
            toon.clone(),
        );
        self.trees.draw(
            pass,
            &self.tree_pipeline,
            motion(breeze),
            VariedToonUniforms {
                toon,
                variation: 0.15,
            },
        );
        scene.chip.draw(pass, view, &look());
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| view.clone());
        self.lint.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, 0.0),
                half_extent: Vec3::new(12.0, 5.0, 10.0),
                velocity: Vec3::new(0.25, 0.12, 0.0),
                wander: 0.8,
                size: (0.03, 0.07),
                colour: Vec3::new(1.5, 1.4, 1.2),
                twinkle: 0.0,
            },
        );
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let uniforms = StitchUniforms {
            source: image,
            time: scene.local_time,
            width: scene.extent.width as f32,
            height: scene.extent.height as f32,
        };
        Some(self.knit.run(context, scene, uniforms))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.05,
            bloom: 0.1,
            vignette: 0.2,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
