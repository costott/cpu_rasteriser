//! bio-cell: the inside of a cell under a dark-field microscope. Folds of rough endoplasmic
//! reticulum for a floor, mitochondria, vesicles and Golgi stacks drifting in warm fluid, out of
//! focus cells and fibres behind, and the chip as the glowing nucleus.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{
    clamp01, fract, hash2, length, mix3, pow_highlight, rgb, s, smoothstep, splat3, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Orbit, ToonUniforms,
    pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{cellular, fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::scatter::{Batch, Placement};
use crate::themes::kit::shape::{Shape, lerp3, transform_point};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// The warm glow of the fluid at the horizon.
const HORIZON: Vec3 = Vec3 {
    x: 0.55,
    y: 0.15,
    z: 0.22,
};
/// The cold light of a dark-field microscope's edge.
const EDGE: Vec3 = Vec3 {
    x: 0.3,
    y: 0.55,
    z: 1.2,
};

fn lighting() -> Lighting {
    Lighting {
        sun: Vec3::new(-0.4, 0.8, 0.45).normalise(),
        sun_colour: Vec3::new(1.5, 0.8, 0.55),
        sky: Vec3::new(0.2, 0.07, 0.15),
        ground: Vec3::new(0.18, 0.07, 0.06),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: HORIZON * 0.75,
            density: 0.04,
            falloff: 0.0,
            base: FLOOR,
        },
    }
}

/// The chip as the nucleus: a gelatinous pink package round a die that glows from within.
fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.6, 0.2, 0.4),
        die: Vec3::new(0.95, 0.32, 0.1),
        gold: Vec3::new(1.0, 0.72, 0.5),
        streams: Vec3::new(0.5, 0.8, 1.3),
        key: light.sun,
        key_colour: light.sun_colour * 0.9,
        sky: light.sky * 1.4,
        ground: light.ground * 1.5,
        softbox: Vec3::new(0.5, 0.3, 0.35),
    }
}

/// The floor's height: folded sheets of membrane behind the chip, calm near it and low
/// towards the camera so they never rise into the view.
fn floor_height(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let calm = smooth(r, 3.0, 8.0) * (1.0 - smooth(z, -1.0, 4.0));
    let fold = 1.0 - (2.0 * fbm_scalar(x * 0.15 + 3.0, z * 0.13, 3) - 1.0).abs();
    FLOOR + calm * (fold * fold * 3.4 + fbm_scalar(x * 0.45, z * 0.45, 2) * 0.35)
}

fn smooth(x: f32, low: f32, high: f32) -> f32 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn floor() -> Shape {
    let membrane = |centroid: Vec3, _normal: Vec3| {
        let crest = ((centroid.y - FLOOR) / 2.4).clamp(0.0, 1.0);
        lerp3(
            Vec3::new(0.2, 0.03, 0.18),
            Vec3::new(0.85, 0.25, 0.5),
            crest.sqrt(),
        )
    };
    terrain(&ring_radii(1.5, 110.0, 44), 96, floor_height, membrane)
}

/// A mitochondrion: a ribbed capsule that wiggles as it drifts.
fn mitochondrion(rng: &mut Rng) -> Shape {
    let (outer, rib) = (Vec3::new(0.85, 0.25, 0.12), Vec3::new(1.0, 0.5, 0.22));
    Shape::icosphere(1.0, 2, 0.03, rng.range(0.0, 30.0), outer)
        .transformed(&Mat4::scaling(1.0, 0.42, 0.42))
        .colour_by(move |p, _| if (p.x * 5.5).sin() > 0.35 { rib } else { outer })
        .sway_by(|p| (p.x.abs() * 0.8).min(1.0))
}

/// A vesicle: a bead of fluid in a membrane.
fn vesicle(colour: Vec3, rng: &mut Rng) -> Shape {
    Shape::icosphere(1.0, 2, 0.05, rng.range(0.0, 30.0), colour)
}

/// A Golgi stack: flat, slightly tapering sacs, one above the next.
fn golgi() -> Shape {
    let mut stack = Shape::new();
    for i in 0..5 {
        let radius = 0.8 - 0.07 * i as f32;
        let y = (i as f32 - 2.0) * 0.17;
        let sac = Shape::cylinder(radius, radius * 0.88, 0.09, 14, Vec3::new(1.0, 0.7, 0.45))
            .transformed(&Mat4::translation(0.0, y, 0.0));
        stack.append(&sac);
    }
    stack.transformed(&Mat4::rotate_z(0.35))
}

/// Instances of the organelle variants, each at the origin: the orbit motion carries them off.
fn drifters(rng: &mut Rng) -> Shape {
    let mut batch = Batch::new();
    let place = |rng: &mut Rng, count: usize, scale: (f32, f32)| -> Vec<Placement> {
        (0..count)
            .map(|_| Placement {
                position: Vec3::ZERO,
                yaw: 0.0,
                scale: rng.range(scale.0, scale.1),
                seed: rng.unit(),
            })
            .collect()
    };
    let mitochondria: Vec<Shape> = (0..3).map(|_| mitochondrion(rng)).collect();
    let placements = place(rng, 10, (0.5, 0.95));
    batch.add(&mitochondria, &placements, |p| (p.seed * 3.0) as usize % 3);
    let colours = [
        Vec3::new(0.45, 0.2, 0.75),
        Vec3::new(0.9, 0.3, 0.4),
        Vec3::new(0.2, 0.5, 0.85),
        Vec3::new(0.9, 0.6, 0.2),
    ];
    let vesicles: Vec<Shape> = colours.iter().map(|&c| vesicle(c, rng)).collect();
    let placements = place(rng, 22, (0.12, 0.45));
    batch.add(&vesicles, &placements, |p| (p.seed * 4.0) as usize % 4);
    let placements = place(rng, 3, (0.6, 0.9));
    batch.add(&[golgi()], &placements, |_| 0);
    batch.finish()
}

/// The fluid behind everything: a warm glow at the horizon under dark magenta, out-of-focus cells
/// drifting by, glowing fibres, and the cold light of the field's edge.
struct FluidShader;

#[derive(Clone)]
struct FluidUniforms {
    view: ViewRef,
    /// Where the nucleus is on screen, in normalised device coordinates.
    nucleus: (f32, f32),
}

impl FragmentShaderSimd<ScreenVaryings> for FluidShader {
    type Uniforms = FluidUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &FluidUniforms) -> ColourSimd {
        let view = &u.view;
        let [x, y] = v.ndc();
        let ray = view_ray(view, x, y);
        let t = s(view.frame.time);
        let up = ray.y;

        let band = (-(up + s(0.01)) * (up + s(0.01)) * s(110.0)).exp();
        let depth = smoothstep(-0.05, 0.45, up);
        let mut colour = mix3(splat3(HORIZON) * s(0.5), rgb(0.03, 0.007, 0.04), depth)
            * (s(0.5) + band * s(1.0));

        let aspect = s(view.aspect);
        for (k, (scale, speed, brightness)) in [(2.2_f32, 0.012_f32, 0.2_f32), (5.5, 0.03, 0.12)]
            .into_iter()
            .enumerate()
        {
            let px = x * aspect * s(scale) + t * s(speed) + s(k as f32 * 7.3);
            let py = y * s(scale) - t * s(speed * 0.6);
            let (cx, cy) = (px.floor(), py.floor());
            let jitter = hash2(cx + s(k as f32), cy);
            let size = hash2(cx * s(1.7), cy + s(4.1));
            let centre_x = s(0.25) + hash2(cx, cy * s(3.3)) * s(0.5);
            let centre_y = s(0.25) + jitter * s(0.5);
            let (dx, dy) = (fract(px) - centre_x, fract(py) - centre_y);
            let d = (dx * dx + dy * dy).sqrt();
            let radius = s(0.1) + size * s(0.16);
            let disc = s(1.0) - smoothstep(0.0, 0.05, d - radius);
            let ring = (-((d - radius) / s(0.025)) * ((d - radius) / s(0.025))).exp();
            let present = smoothstep(0.35, 0.5, hash2(cx * s(2.9), cy * s(1.3) + s(k as f32)));
            let tint = mix3(
                rgb(1.0, 0.4, 0.35),
                splat3(EDGE),
                smoothstep(0.45, 0.8, size),
            );
            colour = colour + tint * ((disc * s(0.3) + ring * s(0.7)) * present * s(brightness));
        }

        for (k, scale) in [1.1_f32, 1.9].into_iter().enumerate() {
            let flow = fbm(
                x * aspect * s(scale) + t * s(0.02),
                y * s(scale) + s(k as f32 * 3.7),
                3,
            );
            let strand = s(1.0) - smoothstep(0.0, 0.015, (flow - s(0.5)).abs());
            colour = colour + rgb(1.0, 0.55, 0.45) * (strand * s(0.12) * (s(0.4) + depth));
        }

        let edge = smoothstep(
            1.0,
            1.7,
            ((x * aspect * s(0.55)) * (x * aspect * s(0.55)) + y * y).sqrt(),
        );
        colour = colour + splat3(EDGE) * (edge * s(0.55));
        let (dx, dy) = ((x - s(u.nucleus.0)) * aspect, y - s(u.nucleus.1));
        let reach = (dx * dx + dy * dy).sqrt();
        let pulse = s(0.8) + s(0.2) * (t * s(1.4)).sin();
        colour =
            colour + rgb(1.0, 0.38, 0.26) * ((-reach * reach * s(1.6)).exp() * pulse * s(0.55));
        to_colour(colour, s(SKY_DISTANCE))
    }
}

/// Translucent membrane: a warm body lit through, a cold glow along its edges and a scatter of
/// ribosomes, with slow bands of light drifting over it.
struct MembraneShader;

impl FragmentShaderSimd<KitVaryings> for MembraneShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let normal = facing(v.normal().normalise(), to_eye);
        let t = s(view.frame.time);

        let (cx, cz) = (world.x * s(5.0), world.z * s(5.0));
        let (fx, fz) = (fract(cx) - s(0.5), fract(cz) - s(0.5));
        let cell = hash2(cx.floor(), cz.floor());
        let dot = (s(1.0) - smoothstep(0.1, 0.22, (fx * fx + fz * fz).sqrt()))
            * smoothstep(0.55, 0.65, cell);
        let near = s(1.0) - smoothstep(10.0, 30.0, distance);
        let drift = fbm(
            world.x * s(0.35) + t * s(0.07),
            world.z * s(0.35) - t * s(0.05),
            3,
        );
        let bands = smoothstep(0.5, 0.8, drift);

        let glow = (s(1.0)
            - smoothstep(
                0.04,
                0.3,
                cellular(world.x * s(0.8), world.z * s(0.8), view.frame.time * 0.3),
            ))
            * near;
        let albedo = v.colour() * (s(1.0) + dot * near * s(0.5));
        let mut colour = u.lighting.shade(view, world, normal, albedo, s(1.0));
        let steep = smoothstep(0.55, 0.95, s(1.0) - normal.y);
        colour = colour + splat3(EDGE) * (steep * s(0.5));
        colour = colour
            + rgb(1.0, 0.45, 0.3) * (bands * s(0.18) * near)
            + rgb(1.0, 0.5, 0.45) * (glow * s(0.4));
        let hazy = u.lighting.fog.apply(colour, world, distance);
        scene_colour(view, hazy, world)
    }
}

/// A jelly-like organelle: brighter where the light passes through its thickest part, a cold
/// rim, a soft highlight and a little mottling inside.
struct JellyShader;

impl FragmentShaderSimd<KitVaryings> for JellyShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let view_dir = to_eye * (s(1.0) / distance);
        let normal = facing(v.normal().normalise(), to_eye);
        let t = s(view.frame.time);

        let thick = clamp01(normal.dot(view_dir));
        let turned = s(1.0) - thick;
        let mottle = fbm(
            world.x * s(2.2) + t * s(0.1),
            world.y * s(2.2) + world.z * s(1.7),
            3,
        );
        let core = v.colour() * (s(0.15) + thick * thick * s(0.9) + mottle * s(0.35));
        let lit = u.lighting.shade(view, world, normal, v.colour(), s(1.0));
        let half = (splat3(u.lighting.sun) + view_dir).normalise();
        let highlight = pow_highlight(normal.dot(half).fast_max(s(0.0)), 64.0) * s(0.8);
        let rim = splat3(EDGE) * (turned * turned * s(0.8))
            + rgb(1.0, 0.8, 0.7) * (turned * turned * turned * s(0.5));
        let colour = (core + lit * s(0.3)) + rim + splat3(u.lighting.sun_colour) * highlight;
        let hazy = u.lighting.fog.apply(colour, world, distance);
        scene_colour(view, hazy, world)
    }
}

pub struct BioCell {
    floor: Shape,
    drifters: Shape,
    specks: Particles,
    membrane: SimdPipeline<KitVertexShader, MembraneShader>,
    jelly: SimdPipeline<KitVertexShader, JellyShader>,
    fluid: SimdPipeline<ScreenVertexShader, FluidShader>,
}

impl BioCell {
    pub fn new() -> Self {
        let mut rng = Rng::new(1665);
        Self {
            floor: floor(),
            drifters: drifters(&mut rng),
            specks: Particles::new(260, 91),
            membrane: pipeline(MembraneShader, CullingMode::None),
            jelly: pipeline(JellyShader, CullingMode::None),
            fluid: SimdPipeline::new(ScreenVertexShader, FluidShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for BioCell {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Marimba],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Bubble),
                0.0,
                0.45,
                0.16,
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
        let current = Motion::Orbit(Orbit {
            centre: Vec3::new(0.0, 0.4, -1.0),
            radius: (3.4, 9.0),
            height: (-0.8, 3.4),
            speed: (0.25, 0.7),
            wiggle: 0.07,
        });

        self.floor.draw(
            pass,
            &self.membrane,
            KitUniforms {
                view: view.clone(),
                motion: Motion::Still,
            },
            toon.clone(),
        );
        self.drifters.draw(
            pass,
            &self.jelly,
            KitUniforms {
                view: view.clone(),
                motion: current,
            },
            toon,
        );
        scene.chip.draw(pass, view, &look());

        let centre = transform_point(&view.frame.chip, Vec3::ZERO).to_point4();
        let clip = view.view_projection * centre;
        let sky = FluidUniforms {
            view: view.clone(),
            nucleus: (clip.x / clip.w, clip.y / clip.w),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.fluid, (), |_| sky.clone());
        self.specks.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.0, -3.0),
                half_extent: Vec3::new(16.0, 5.0, 14.0),
                velocity: Vec3::new(0.15, 0.1, 0.05),
                wander: 0.8,
                size: (0.015, 0.05),
                colour: Vec3::new(1.5, 0.8, 0.7),
                twinkle: 2.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.1,
            bloom: 0.7,
            vignette: 0.7,
            grain: 0.012,
            aberration: 0.002,
            desaturate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_floor_never_rises_into_the_view_near_the_camera() {
        let eye = -0.6;
        for i in 0..200 {
            let (x, z) = (
                (i as f32 * 0.37).sin() * 14.0,
                3.0 + (i as f32 * 0.13) % 12.0,
            );
            assert!(floor_height(x, z) < eye - 0.8, "at ({x}, {z})");
        }
    }

    #[test]
    fn the_floor_is_flat_under_the_chip() {
        assert!((floor_height(0.0, 0.0) - FLOOR).abs() < 1e-3);
    }
}
