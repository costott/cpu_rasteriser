//! wireframe-grid: white wire on black, the last world before the chip returns clean. A grid of
//! wire laid over rolling hills with hidden lines removed, pylons and tumbling solids drawn only
//! as their edges, and the chip reduced to its own outline and light streams. A pulse runs out
//! across the grid from the chip.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{F, clamp01, fract, length, rgb, s, splat3, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonShader, ToonUniforms,
    pipeline,
};
use crate::themes::kit::light::{Fog, Lighting};
use crate::themes::kit::post::{Chain, Edges, Effect};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::overlay::{Align, Line, Overlay, Text};
use crate::themes::{Scene, Stage};

/// The height of the flat ground round the chip.
const FLOOR: f32 = -3.4;
/// The distance between grid lines.
const GRID: f32 = 2.0;
const WHITE: Vec3 = Vec3 {
    x: 1.0,
    y: 1.0,
    z: 1.0,
};

fn smooth(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Flat round the chip, swells that grow taller with distance.
fn height(x: f32, z: f32) -> f32 {
    let distance = (x * x + z * z).sqrt();
    let swell = 2.6
        + 1.6 * (x * 0.18 + z * 0.07).sin()
        + 1.1 * (z * 0.23 - x * 0.05).sin()
        + 0.5 * ((x + z) * 0.41).sin();
    FLOOR + smooth(6.0, 22.0, distance) * swell * 0.6 * (1.0 + distance * 0.02)
}

pub struct WireframeGrid {
    look: Look,
    land: Shape,
    pylons: Shape,
    solids: Shape,
    wire: SimdPipeline<KitVertexShader, WireShader>,
    solid: SimdPipeline<KitVertexShader, ToonShader>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    overlay: Overlay,
    chain: Chain,
}

impl WireframeGrid {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.012, 0.012, 0.012),
                die: Vec3::new(0.004, 0.004, 0.004),
                gold: WHITE * 0.5,
                streams: WHITE * 2.2,
                key: Vec3::new(-0.4, 0.5, 0.75).normalise(),
                key_colour: WHITE * 0.35,
                sky: Vec3::ZERO,
                ground: Vec3::ZERO,
                softbox: Vec3::ZERO,
            },
            land: terrain(&ring_radii(1.2, 220.0, 64), 96, height, |_, _| Vec3::ZERO),
            pylons: pylons(),
            solids: Shape::new(),
            wire: pipeline(WireShader, CullingMode::None),
            solid: pipeline(ToonShader, CullingMode::None),
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            overlay: Overlay::new(),
            chain: Chain::new(vec![Effect::Edges(Edges {
                ink: WHITE * 2.4,
                width: 0.8,
                ..Edges::default()
            })]),
        }
    }
}

/// Towers and slabs standing in the hills, to be seen only by their edges.
fn pylons() -> Shape {
    let mut rng = Rng::new(56);
    let mut pylons = Shape::new();
    for _ in 0..34 {
        let (x, z) = (rng.range(-70.0, 70.0), rng.range(-90.0, -14.0));
        if x.abs() < 7.0 && z > -40.0 {
            continue;
        }
        let (half, tall) = (rng.range(0.7, 1.8), rng.range(3.0, 13.0));
        let base = height(x, z) - 0.3;
        let tower = match rng.index(3) {
            0 => Shape::cuboid(
                Vec3::new(-half, 0.0, -half),
                Vec3::new(half, tall, half),
                Vec3::ZERO,
            ),
            1 => Shape::cone(half * 1.6, tall, 4, Vec3::ZERO),
            _ => Shape::cylinder(half, half * 0.5, tall, 6, Vec3::ZERO),
        };
        pylons.append(&tower.transformed(&Mat4::translation(x, base, z)));
    }
    pylons
}

/// Solids tumbling slowly in the air, as black volumes whose edges the stylisation draws.
fn solids(time: f32) -> Shape {
    let mut rng = Rng::new(57);
    let mut shape = Shape::new();
    for i in 0..9 {
        let size = rng.range(0.7, 1.7);
        let home = Vec3::new(
            rng.range(-22.0, 22.0),
            rng.range(-0.5, 7.0),
            rng.range(-12.0, -48.0),
        );
        let (spin, phase) = (rng.range(-0.5, 0.5), rng.range(0.0, std::f32::consts::TAU));
        let body = if i % 3 == 0 {
            Shape::icosphere(size, 1, 0.0, 0.0, Vec3::ZERO)
        } else {
            Shape::cuboid(
                Vec3::new(-size, -size, -size) * 0.8,
                Vec3::new(size, size, size) * 0.8,
                Vec3::ZERO,
            )
        };
        let pose = Mat4::translation(home.x, home.y + (time * 0.6 + phase).sin() * 0.5, home.z)
            * Mat4::rotate_y(time * spin + phase)
            * Mat4::rotate_x(time * spin * 0.7);
        shape.append(&body.transformed(&pose));
    }
    shape
}

/// Shows an albedo as it is: no light, no fog.
fn unlit() -> Lighting {
    Lighting {
        sun: Vec3::new(0.0, 1.0, 0.0),
        sun_colour: Vec3::ZERO,
        sky: Vec3::ONE,
        ground: Vec3::ONE,
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: Vec3::ZERO,
            density: 0.0,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

impl Stage for WireframeGrid {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Thin],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::LineZap),
                0.0,
                0.45,
                0.1,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        self.solids = solids(scene.local_time);
        let window = (scene.extent.width, scene.extent.height);
        let pixel = 2.6 * window.1 as f32 / 720.0;
        let mut text = Text::default();
        text.line(
            Line {
                text: "56 / 56",
                x: -0.84,
                y: -0.82,
                align: Align::Left,
                colour: WHITE,
            },
            pixel,
            window,
        );
        self.overlay.set(text);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let motion = || KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };
        scene.chip.draw(pass, view, &self.look);
        self.land.draw(
            pass,
            &self.wire,
            motion(),
            WireUniforms {
                view: view.clone(),
                time: scene.local_time,
            },
        );
        let black = ToonUniforms {
            view: view.clone(),
            lighting: unlit(),
        };
        self.pylons.draw(pass, &self.solid, motion(), black.clone());
        self.solids.draw(pass, &self.solid, motion(), black);
        let uniforms = BackdropUniforms { view: view.clone() };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
        self.overlay.draw(pass);
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
            bloom: 0.9,
            vignette: 0.3,
            grain: 0.0,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[derive(Clone)]
struct WireUniforms {
    view: ViewRef,
    time: f32,
}

/// A black surface with a grid of white wire laid on it in world space, fading as the lines
/// close up towards a pixel's width, and brightened by a pulse running out from the chip.
struct WireShader;

impl FragmentShaderSimd<KitVaryings> for WireShader {
    type Uniforms = WireUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &WireUniforms) -> ColourSimd {
        let world = v.world();
        let distance = length(splat3(u.view.eye) - world);
        let width = distance * s(u.view.line_angle) * s(1.3);
        let line = |p: F| {
            let across = (fract(p * s(1.0 / GRID) + s(0.5)) - s(0.5)).abs() * s(GRID);
            clamp01(s(1.5) - across / width.fast_max(s(0.001)))
        };
        let detail = clamp01(s(1.4) - width * s(1.0 / (GRID * 0.35)));
        let grid = line(world.x).fast_max(line(world.z)) * detail;
        let radius = (world.x * world.x + world.z * world.z).sqrt();
        let pulse = (-(radius - s((u.time * 9.0) % 80.0 + 3.0)).abs() * s(0.45)).exp();
        let range = (-distance * s(0.012)).exp();
        let brightness = grid * range * (s(0.9) + pulse * s(1.6));
        scene_colour(&u.view, rgb(1.0, 1.0, 1.0) * brightness, world)
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
}

/// Black sky with a hairline horizon.
struct Backdrop;

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(&u.view, nx, ny);
        let footprint = s(u.view.line_angle);
        let horizon = clamp01(s(1.5) - ray.y.abs() / (footprint * s(1.5)));
        to_colour(rgb(1.0, 1.0, 1.0) * (horizon * s(0.6)), s(SKY_DISTANCE))
    }
}
