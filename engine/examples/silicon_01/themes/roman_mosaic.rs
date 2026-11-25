//! A Roman mosaic floor: the chip set in a medallion of ringed borders and scattered rosettes,
//! a colonnade at the far edge of the villa court. The finished frame is cut into tesserae of
//! stone and glass: each one a slightly turned square of a palette colour, separated by grout,
//! with a few that glint as the sun moves across the floor.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, from_colour, gt, hash2, mix3, rgb, s, select, select3, smoothstep, splat3, step_lt,
    to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonShader, ToonUniforms,
    pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::fbm;
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::post::{ScreenPass, compress, expand, hdr_of_displayed};
use crate::themes::kit::shape::Shape;
use crate::themes::{Scene, Stage};

/// The height of the floor where the chip stands: its lowest corner rests here.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.45,
    y: 0.6,
    z: 0.3,
};

const CREAM: Vec3 = Vec3 {
    x: 0.88,
    y: 0.8,
    z: 0.62,
};
const TERRACOTTA: Vec3 = Vec3 {
    x: 0.7,
    y: 0.32,
    z: 0.2,
};
const OCHRE: Vec3 = Vec3 {
    x: 0.8,
    y: 0.58,
    z: 0.22,
};
const UMBER: Vec3 = Vec3 {
    x: 0.16,
    y: 0.11,
    z: 0.09,
};
const OLIVE: Vec3 = Vec3 {
    x: 0.42,
    y: 0.45,
    z: 0.25,
};
const SLATE: Vec3 = Vec3 {
    x: 0.3,
    y: 0.42,
    z: 0.5,
};
const MARBLE: Vec3 = Vec3 {
    x: 0.93,
    y: 0.9,
    z: 0.82,
};
const RUST: Vec3 = Vec3 {
    x: 0.5,
    y: 0.22,
    z: 0.14,
};
const PALETTE: [Vec3; 8] = [CREAM, TERRACOTTA, OCHRE, UMBER, OLIVE, SLATE, MARBLE, RUST];
const GROUT: Vec3 = Vec3 {
    x: 0.62,
    y: 0.56,
    z: 0.46,
};

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(2.2, 1.9, 1.4),
        sky: Vec3::new(0.7, 0.72, 0.8),
        ground: Vec3::new(0.5, 0.4, 0.3),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: hdr_of_displayed(Vec3::new(0.85, 0.72, 0.52)),
            density: 0.004,
            falloff: 0.04,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.42, 0.2, 0.1),
        die: Vec3::new(0.04, 0.07, 0.12),
        gold: Vec3::new(0.9, 0.62, 0.2),
        streams: Vec3::new(0.95, 0.6, 0.2),
        key: light.sun,
        key_colour: light.sun_colour * 0.7,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.6, 0.5, 0.35),
    }
}

/// A row of fluted columns along the far side of the court.
fn colonnade() -> Shape {
    let stone = Vec3::new(0.85, 0.78, 0.65);
    let mut shape = Shape::new();
    for i in -6..=6 {
        let x = i as f32 * 5.5;
        let base = Mat4::translation(x, FLOOR, -26.0);
        shape.append(
            &Shape::cuboid(
                Vec3::new(-0.9, 0.0, -0.9),
                Vec3::new(0.9, 0.35, 0.9),
                stone * 0.9,
            )
            .transformed(&base),
        );
        shape.append(
            &Shape::cylinder(0.65, 0.55, 7.0, 10, stone)
                .transformed(&(base * Mat4::translation(0.0, 0.35, 0.0))),
        );
        shape.append(
            &Shape::cuboid(
                Vec3::new(-0.95, 0.0, -0.95),
                Vec3::new(0.95, 0.45, 0.95),
                stone * 0.95,
            )
            .transformed(&(base * Mat4::translation(0.0, 7.35, 0.0))),
        );
    }
    shape.append(&Shape::cuboid(
        Vec3::new(-40.0, 7.8, -27.0),
        Vec3::new(40.0, 9.0, -25.0),
        Vec3::new(0.7, 0.35, 0.22),
    ));
    shape
}

/// `a` where `flag` is 1 and `b` where it is 0.
#[inline(always)]
fn choose(flag: F, a: F, b: F) -> F {
    select(gt(flag, s(0.5)), a, b)
}

/// [`choose`] for colours.
#[inline(always)]
fn choose3(flag: F, a: Vec3Simd, b: Vec3Simd) -> Vec3Simd {
    select3(gt(flag, s(0.5)), a, b)
}

/// Picks `a` or `b` by `flag` (1 for `a`).
#[inline(always)]
fn pick(flag: F, a: Vec3, b: Vec3) -> Vec3Simd {
    choose3(flag, splat3(a), splat3(b))
}

/// The floor's design in colours of the palette: concentric borders round the chip, a field of
/// scattered rosettes, a sawtooth band and a plain terracotta margin.
struct FloorShader;

impl FragmentShaderSimd<KitVaryings> for FloorShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let (x, z) = (world.x, world.z);
        let radius = (x * x + z * z).sqrt();
        let square = x.abs().fast_max(z.abs());

        let angle = z.atan2(x);
        let rope = ((angle * s(14.0) + radius * s(5.0)).sin()).abs();
        let on = |value: F| step_lt(s(0.5), value);
        let medallion = pick(on(rope), CREAM, TERRACOTTA);
        let ring = |inner: f32, outer: f32| step_lt(s(inner), radius) * step_lt(radius, s(outer));
        let mut albedo = pick(step_lt(radius, s(3.6)), UMBER, CREAM);
        albedo = choose3(ring(3.6, 4.0), splat3(UMBER), albedo);
        albedo = choose3(ring(4.0, 5.4), medallion, albedo);
        albedo = choose3(ring(5.4, 5.8), splat3(UMBER), albedo);
        albedo = choose3(ring(5.8, 7.4), splat3(SLATE), albedo);
        let wave = ((angle * s(12.0)).sin() * s(0.5) + s(6.6) - radius).abs();
        albedo = choose3(
            ring(5.8, 7.4) * step_lt(wave, s(0.35)),
            splat3(MARBLE),
            albedo,
        );
        albedo = choose3(ring(7.4, 7.8), splat3(UMBER), albedo);
        albedo = choose3(ring(7.8, 9.6), splat3(TERRACOTTA), albedo);
        let leaf = ((angle * s(9.0)).sin() * s(0.6) + s(8.7) - radius).abs();
        albedo = choose3(
            ring(7.8, 9.6) * step_lt(leaf, s(0.45)),
            splat3(OLIVE),
            albedo,
        );
        albedo = choose3(ring(9.6, 10.0), splat3(UMBER), albedo);
        let beyond = step_lt(s(10.0), radius);

        let (cx, cz) = ((x * s(0.55)).floor(), (z * s(0.55)).floor());
        let (lx, lz) = (x * s(0.55) - cx - s(0.5), z * s(0.55) - cz - s(0.5));
        let petals = (lx.abs() * lz.abs() * s(18.0)).fast_min(s(1.0));
        let rosette = step_lt(
            (lx * lx + lz * lz).sqrt() + (s(0.5) - petals) * s(0.15),
            s(0.3),
        );
        let scatter = step_lt(s(0.55), hash2(cx, cz)) * rosette;
        let field = choose3(
            scatter,
            pick(step_lt(hash2(cx, cz + s(5.0)), s(0.5)), TERRACOTTA, SLATE),
            splat3(CREAM),
        );
        albedo = choose3(beyond * step_lt(square, s(15.0)), field, albedo);

        let along = (x + z).abs() * s(0.9);
        let tooth = ((along - along.floor()) - s(0.5)).abs() * s(2.0);
        let band_pos = (square - s(15.0)) * s(1.0 / 1.6);
        let in_band = step_lt(s(0.0), band_pos) * step_lt(band_pos, s(1.0));
        let band = choose3(step_lt(tooth, band_pos), splat3(UMBER), splat3(MARBLE));
        albedo = choose3(in_band, band, albedo);
        let margin = pick(on(fbm(x * s(0.8), z * s(0.8), 2)), TERRACOTTA, RUST);
        albedo = choose3(step_lt(s(16.6), square), margin, albedo);

        let to_eye = splat3(view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        let sweep = smoothstep(
            -0.3,
            0.9,
            ((x + z) * s(0.06) + s(view.frame.time * 0.1)).sin(),
        );
        let shadow = s(0.75) + sweep * s(0.25);
        scene_colour(
            view,
            u.lighting.shade(view, world, normal, albedo, shadow),
            world,
        )
    }
}

/// A warm hazy sky above the colonnade.
struct SkyWash;

impl FragmentShaderSimd<ScreenVaryings> for SkyWash {
    type Uniforms = crate::shaders::ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &crate::shaders::ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let up = clamp01(ray.y * s(2.5));
        let colour = mix3(rgb(0.78, 0.84, 0.86), rgb(0.25, 0.48, 0.74), up.sqrt());
        to_colour(expand(colour), s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
struct CutUniforms {
    source: Arc<TextureSampler>,
    time: f32,
    width: f32,
    height: f32,
}

/// Cuts the frame into tesserae: each a turned square of the nearest palette colour, lit as the
/// scene lights it, separated by grout; a few glint.
struct CutShader;

#[inline(always)]
fn luminance(c: Vec3Simd) -> F {
    c.x * s(0.2126) + c.y * s(0.7152) + c.z * s(0.0722)
}

impl FragmentShaderSimd<ScreenVaryings> for CutShader {
    type Uniforms = CutUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &CutUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let unit = u.height / 720.0;
        let size = 9.0 * unit;
        let (px, py) = (x * s(u.width), y * s(u.height));
        let (gx, gy) = ((px * s(1.0 / size)).floor(), (py * s(1.0 / size)).floor());

        let mut covered = s(0.0);
        let mut centre_x = s(0.0);
        let mut centre_y = s(0.0);
        let mut id_x = s(0.0);
        let mut id_y = s(0.0);
        let mut bevel = s(0.0);
        for j in -1..=1 {
            for i in -1..=1 {
                let (cx, cy) = (gx + s(i as f32), gy + s(j as f32));
                let jitter_x = (hash2(cx, cy) - s(0.5)) * s(0.18);
                let jitter_y = (hash2(cx + s(31.0), cy + s(17.0)) - s(0.5)) * s(0.18);
                let turn = (hash2(cx + s(5.0), cy + s(2.0)) - s(0.5)) * s(0.35);
                let (mx, my) = (
                    (cx + s(0.5) + jitter_x) * s(size),
                    (cy + s(0.5) + jitter_y) * s(size),
                );
                let (dx, dy) = (px - mx, py - my);
                let (qx, qy) = (
                    dx * turn.cos() + dy * turn.sin(),
                    dy * turn.cos() - dx * turn.sin(),
                );
                let half = s(size * 0.43);
                let depth = half - qx.abs().fast_max(qy.abs());
                let inside = step_lt(s(0.0), depth);
                let newer = inside * step_lt(covered, s(0.5));
                covered = covered.fast_max(inside);
                centre_x = choose(newer, mx, centre_x);
                centre_y = choose(newer, my, centre_y);
                id_x = choose(newer, cx, id_x);
                id_y = choose(newer, cy, id_y);
                bevel = choose(
                    newer,
                    smoothstep(0.0, 0.14 * size, depth)
                        * (s(0.9) + (qx + qy) * s(-0.5 / size) * s(0.5)),
                    bevel,
                );
            }
        }

        let sample = u
            .source
            .sample_linear_clamp_simd([centre_x * s(1.0 / u.width), centre_y * s(1.0 / u.height)]);
        let shown = compress(from_colour(sample));
        let mut best = s(1.0e9);
        let mut snapped = splat3(PALETTE[0]);
        for entry in PALETTE {
            let delta = shown - splat3(entry);
            let distance = delta.x * delta.x * s(1.0)
                + delta.y * delta.y * s(1.2)
                + delta.z * delta.z * s(0.8);
            let nearer = step_lt(distance, best);
            best = choose(nearer, distance, best);
            snapped = choose3(nearer, splat3(entry), snapped);
        }
        let lift = (luminance(shown) / luminance(snapped).fast_max(s(0.05)))
            .fast_max(s(0.7))
            .fast_min(s(1.3));
        let variety = hash2(id_x + s(9.0), id_y + s(4.0)) - s(0.5);
        let stone = snapped * lift * (s(1.0) + variety * s(0.2));

        let phase = hash2(id_x + s(2.0), id_y + s(13.0)) * s(40.0);
        let glint = smoothstep(0.96, 1.0, (s(u.time * 0.9) + phase).sin() * s(0.5) + s(0.5))
            * step_lt(s(0.8), hash2(id_x, id_y + s(7.0)));
        let tessera = stone * (s(0.88) + bevel * s(0.18)) + rgb(0.5, 0.42, 0.28) * glint;

        let grain = fbm(px * s(0.5), py * s(0.5), 2);
        let grout = splat3(GROUT) * (s(0.85) + grain * s(0.25));
        let outcome = mix3(grout, tessera, covered);
        to_colour(expand(outcome), sample.a)
    }
}

pub struct RomanMosaic {
    floor: Shape,
    columns: Shape,
    dust: Particles,
    floor_pipeline: SimdPipeline<KitVertexShader, FloorShader>,
    stone_pipeline: SimdPipeline<KitVertexShader, ToonShader>,
    sky_pipeline: SimdPipeline<ScreenVertexShader, SkyWash>,
    cut: ScreenPass<CutShader>,
}

impl RomanMosaic {
    pub fn new() -> Self {
        let far = 300.0;
        let mut floor = Shape::new();
        floor.quad(
            [
                Vec3::new(-far, FLOOR, far),
                Vec3::new(far, FLOOR, far),
                Vec3::new(far, FLOOR, -far),
                Vec3::new(-far, FLOOR, -far),
            ],
            Vec3::ONE,
        );
        Self {
            floor,
            columns: colonnade(),
            dust: Particles::new(80, 17),
            floor_pipeline: pipeline(FloorShader, CullingMode::None),
            stone_pipeline: pipeline(ToonShader, CullingMode::None),
            sky_pipeline: SimdPipeline::new(ScreenVertexShader, SkyWash)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            cut: ScreenPass::new(CutShader),
        }
    }
}

impl Stage for RomanMosaic {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Trumpet],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Clack),
                0.0,
                0.45,
                0.17,
                -0.3,
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
        let still = || KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };

        self.floor
            .draw(pass, &self.floor_pipeline, still(), toon.clone());
        self.columns.draw(pass, &self.stone_pipeline, still(), toon);
        scene.chip.draw(pass, view, &look());
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky_pipeline, (), |_| view.clone());
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 2.0, 0.0),
                half_extent: Vec3::new(14.0, 5.0, 12.0),
                velocity: Vec3::new(0.1, 0.08, 0.0),
                wander: 0.6,
                size: (0.02, 0.05),
                colour: Vec3::new(1.5, 1.2, 0.7),
                twinkle: 1.5,
            },
        );
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        let uniforms = CutUniforms {
            source: image,
            time: scene.local_time,
            width: scene.extent.width as f32,
            height: scene.extent.height as f32,
        };
        Some(self.cut.run(context, scene, uniforms))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.2,
            vignette: 0.25,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
