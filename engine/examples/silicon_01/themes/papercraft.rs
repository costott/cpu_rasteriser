//! papercraft: a landscape cut from stacked sheets of pastel paper. Each sheet is a silhouette of
//! hills and trees (or clouds, or the sun) on its own plane behind the chip, shaded with the soft
//! shadow of the sheet in front, a lit cut edge and paper fibre. The camera trucks slowly, so the
//! sheets slide over each other.
//!
//! The whole world is one full-screen shader: a sheet is a signed distance function, so its
//! shadow on the sheet behind is the same function sampled a little towards the light and
//! blurred.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, View, ViewRef, view_ray};
use crate::simd::{
    F, clamp01, fract, hash2, mix3, mul3, rgb, s, select, smoothstep, splat3, to_colour,
};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::noise::value;
use crate::themes::kit::post::expand;
use crate::themes::{Scene, Stage};

/// Where the chip's centre is on screen, as a height in the world: sheets are laid out in
/// fractions of the half-height the camera sees at their depth, measured from here.
const CENTRE_Y: f32 = -0.9;
/// The direction towards the light that casts the chip's shadow onto the sheets.
const LIGHT: Vec3 = Vec3 {
    x: -0.3,
    y: 0.45,
    z: 0.84,
};
/// The screen-plane direction towards the light, which the sheets' shadows fall away from.
const LIGHT_SCREEN: (f32, f32) = (-0.6, 0.8);

/// The shape of a sheet's upper edge: a sum of three sines about `base`, in fractions of the
/// screen's half-height.
#[derive(Clone, Copy)]
struct Hills {
    base: f32,
    amplitude: [f32; 3],
    frequency: [f32; 3],
}

impl Hills {
    #[inline(always)]
    fn top(&self, a: F, seed: f32) -> F {
        let mut y = s(self.base);
        for i in 0..3 {
            y += s(self.amplitude[i])
                * (a * s(self.frequency[i]) + s(seed * (1.3 + 2.1 * i as f32))).sin();
        }
        y
    }
}

/// What is cut into a sheet besides its hills, repeated along it in cells of `spacing`.
#[derive(Clone, Copy)]
enum Decor {
    None,
    /// Tiered firs standing on the hills.
    Firs {
        spacing: f32,
        height: f32,
        width: f32,
        density: f32,
    },
    /// Round trees on thin stalks.
    Orbs {
        spacing: f32,
        height: f32,
        radius: f32,
        density: f32,
    },
    /// Flat-bottomed clouds of three puffs, drifting at `drift` fractions per second.
    Clouds {
        spacing: f32,
        level: f32,
        radius: f32,
        density: f32,
        drift: f32,
    },
    /// A scalloped disc that turns slowly.
    Sun {
        x: f32,
        y: f32,
        radius: f32,
    },
}

#[derive(Clone, Copy)]
struct Layer {
    z: f32,
    /// As displayed, 0 to 1.
    colour: Vec3,
    seed: f32,
    hills: Option<Hills>,
    decor: Decor,
}

const FAR: f32 = 10.0;

impl Layer {
    /// The signed distance to the sheet's edge at layer coordinates (`a`, `b`), negative on the
    /// paper, in fractions of the half-height.
    #[inline(always)]
    fn distance(&self, a: F, b: F, time: f32) -> F {
        let mut d = s(FAR);
        if let Some(hills) = &self.hills {
            d = (b - hills.top(a, self.seed)) * s(0.85);
        }
        let seed = self.seed;
        match self.decor {
            Decor::None => d,
            Decor::Firs {
                spacing,
                height,
                width,
                density,
            } => {
                let hills = self.hills.expect("firs stand on hills");
                let (_, size, centre) = cell_of(a, spacing, seed);
                let base = hills.top(centre, seed) - s(0.01);
                let scale = s(0.6) + size * s(0.8);
                let tree_height = s(height) * scale;
                let q = clamp01((b - base) / tree_height);
                let lean =
                    (s(time * 1.3 + seed) + size * s(TAU)).sin() * q * q * tree_height * s(0.05);
                let tier = fract(q * s(4.0));
                let half = s(width) * scale * (s(1.0) - q * s(0.78)) * (s(1.0) - tier * s(0.5));
                let across = (a - centre - lean).abs() - half;
                let along = (base - b).fast_max(b - base - tree_height);
                let tree = across.fast_max(along);
                let present = select(smoothstep(density - 0.001, density, size), s(FAR), tree);
                d.fast_min(present)
            }
            Decor::Orbs {
                spacing,
                height,
                radius,
                density,
            } => {
                let hills = self.hills.expect("orbs stand on hills");
                let (_, size, centre) = cell_of(a, spacing, seed);
                let base = hills.top(centre, seed) - s(0.01);
                let scale = s(0.7) + size * s(0.6);
                let lean = (s(time * 0.9 + seed) + size * s(TAU)).sin() * s(0.004);
                let (dx, top) = (a - centre - lean, base + s(height) * scale);
                let r = s(radius) * scale;
                let crown = ((dx * dx + (b - top) * (b - top)).sqrt() - r).fast_min(
                    ((dx + r * s(0.7)) * (dx + r * s(0.7))
                        + (b - top + r * s(0.55)) * (b - top + r * s(0.55)))
                    .sqrt()
                        - r * s(0.6),
                );
                let stalk = (dx.abs() - s(0.004)).fast_max((base - b).fast_max(b - top));
                let present = select(
                    smoothstep(density - 0.001, density, size),
                    s(FAR),
                    crown.fast_min(stalk),
                );
                d.fast_min(present)
            }
            Decor::Clouds {
                spacing,
                level,
                radius,
                density,
                drift,
            } => {
                let moved = a - s(time * drift);
                let (cell, size, centre) = cell_of(moved, spacing, seed);
                let jitter = hash2(cell * s(3.7), s(seed + 11.0));
                let r = s(radius) * (s(0.7) + jitter * s(0.6));
                let floor = s(level) + (jitter - s(0.5)) * s(0.25);
                let (dx, dy) = (moved - centre, b - floor);
                let puff = |ox: f32, oy: f32, k: f32| {
                    let (px, py) = (dx - r * s(ox), dy - r * s(oy));
                    (px * px + py * py).sqrt() - r * s(k)
                };
                let cloud = puff(-0.62, 0.35, 0.62)
                    .fast_min(puff(0.0, 0.55, 0.82))
                    .fast_min(puff(0.66, 0.3, 0.58));
                let cloud = cloud.fast_max(-dy);
                let present = select(smoothstep(density - 0.001, density, size), s(FAR), cloud);
                d.fast_min(present)
            }
            Decor::Sun { x, y, radius } => {
                let (dx, dy) = (a - s(x), b - s(y));
                let angle = dy.atan2(dx) * s(16.0) + s(time * 0.25 + seed);
                let scallop = s(1.0) + angle.cos() * s(0.035);
                ((dx * dx + dy * dy).sqrt() - s(radius) * scallop).fast_min(d)
            }
        }
    }
}

/// The cell of width `spacing` that `a` falls in, a random value for it, and the centre of the
/// tree or cloud standing in it, jittered but kept clear of the cell's edges.
#[inline(always)]
fn cell_of(a: F, spacing: f32, seed: f32) -> (F, F, F) {
    let cell = (a / s(spacing)).floor();
    let size = hash2(cell, s(seed * 7.0 + 3.0));
    let jitter = hash2(cell, s(seed * 3.0 + 41.0)) - s(0.5);
    (cell, size, (cell + s(0.5) + jitter * s(0.3)) * s(spacing))
}

fn layers() -> Vec<Layer> {
    let hills = |base: f32, a: [f32; 3], f: [f32; 3]| {
        Some(Hills {
            base,
            amplitude: a,
            frequency: f,
        })
    };
    let tint = |r: f32, g: f32, b: f32| Vec3::new(r, g, b);
    // Far to near.
    vec![
        Layer {
            z: -62.0,
            colour: tint(1.0, 0.93, 0.74),
            seed: 1.0,
            hills: None,
            decor: Decor::Sun {
                x: 1.05,
                y: 0.34,
                radius: 0.34,
            },
        },
        Layer {
            z: -60.0,
            colour: tint(0.99, 0.80, 0.58),
            seed: 2.0,
            hills: None,
            decor: Decor::Sun {
                x: 1.05,
                y: 0.34,
                radius: 0.22,
            },
        },
        Layer {
            z: -52.0,
            colour: tint(0.99, 0.95, 0.92),
            seed: 3.0,
            hills: None,
            decor: Decor::Clouds {
                spacing: 1.5,
                level: 0.5,
                radius: 0.34,
                density: 0.7,
                drift: 0.012,
            },
        },
        Layer {
            z: -44.0,
            colour: tint(0.76, 0.72, 0.86),
            seed: 4.0,
            hills: hills(0.02, [0.13, 0.07, 0.03], [1.1, 2.9, 7.0]),
            decor: Decor::None,
        },
        Layer {
            z: -36.0,
            colour: tint(0.98, 0.91, 0.91),
            seed: 5.0,
            hills: None,
            decor: Decor::Clouds {
                spacing: 1.1,
                level: 0.12,
                radius: 0.2,
                density: 0.6,
                drift: 0.02,
            },
        },
        Layer {
            z: -29.0,
            colour: tint(0.91, 0.69, 0.73),
            seed: 6.0,
            hills: hills(-0.1, [0.1, 0.05, 0.025], [1.6, 3.7, 8.3]),
            decor: Decor::Firs {
                spacing: 0.1,
                height: 0.1,
                width: 0.026,
                density: 0.55,
            },
        },
        Layer {
            z: -22.0,
            colour: tint(0.96, 0.78, 0.64),
            seed: 7.0,
            hills: hills(-0.22, [0.09, 0.05, 0.02], [2.1, 4.3, 9.1]),
            decor: Decor::Orbs {
                spacing: 0.15,
                height: 0.1,
                radius: 0.04,
                density: 0.5,
            },
        },
        Layer {
            z: -16.0,
            colour: tint(0.62, 0.77, 0.68),
            seed: 8.0,
            hills: hills(-0.35, [0.09, 0.045, 0.02], [1.9, 3.9, 7.7]),
            decor: Decor::Firs {
                spacing: 0.14,
                height: 0.17,
                width: 0.04,
                density: 0.6,
            },
        },
        Layer {
            z: -10.5,
            colour: tint(0.36, 0.62, 0.64),
            seed: 9.0,
            hills: hills(-0.5, [0.08, 0.04, 0.02], [2.3, 4.7, 8.9]),
            decor: Decor::Orbs {
                spacing: 0.22,
                height: 0.17,
                radius: 0.065,
                density: 0.5,
            },
        },
        Layer {
            z: -6.5,
            colour: tint(0.22, 0.44, 0.54),
            seed: 10.0,
            hills: hills(-0.66, [0.07, 0.035, 0.015], [1.7, 3.3, 7.3]),
            decor: Decor::Firs {
                spacing: 0.3,
                height: 0.36,
                width: 0.085,
                density: 0.5,
            },
        },
        Layer {
            z: -3.4,
            colour: tint(0.13, 0.27, 0.39),
            seed: 11.0,
            hills: hills(-0.86, [0.05, 0.025, 0.01], [1.3, 3.1, 6.1]),
            decor: Decor::Firs {
                spacing: 0.55,
                height: 0.5,
                width: 0.13,
                density: 0.35,
            },
        },
    ]
}

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
    layers: std::sync::Arc<Vec<Layer>>,
}

/// Where a view ray meets a sheet's plane, in the sheet's own coordinates.
struct Plane {
    t: F,
    x: F,
    y: F,
    a: F,
    b: F,
    scale: f32,
}

impl Plane {
    #[inline(always)]
    fn new(view: &View, ray: Vec3Simd, layer: &Layer, time: f32) -> Self {
        let t = s(layer.z - view.eye.z) / ray.z;
        let x = s(view.eye.x) + ray.x * t;
        let y = s(view.eye.y) + ray.y * t;
        let scale = view.tan_half_fov * (view.eye.z - layer.z);
        let truck = 0.9 * (time * 0.33).sin();
        let bob = 0.003 * (time * 0.7 + layer.seed * 5.0).sin();
        Self {
            t,
            x,
            y,
            a: (x + s(truck)) / s(scale),
            b: (y - s(CENTRE_Y)) / s(scale) + s(bob),
            scale,
        }
    }
}

struct Sheets;

impl FragmentShaderSimd<ScreenVaryings> for Sheets {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let time = u.time;

        let mut colour = rgb(0.0, 0.0, 0.0);
        let mut remaining = s(1.0);
        let mut distance = s(SKY_DISTANCE);
        let mut nearer: Option<(Plane, &Layer)> = None;
        for layer in u.layers.iter().rev() {
            if !remaining.simd_gt(s(0.002)).any() {
                break;
            }
            let plane = Plane::new(view, ray, layer, time);
            let d = layer.distance(plane.a, plane.b, time) * s(plane.scale);
            let pixel = plane.t * s(view.pixel_angle);
            let cover = smoothstep(0.0, 1.0, clamp01(s(0.5) - d / (pixel * s(2.0))));
            if cover.simd_gt(s(0.0)).any() {
                let paper = paper(layer, &plane, d, pixel, time, view, nearer.as_ref());
                let weight = cover * remaining;
                colour = colour + paper * weight;
                distance = select(
                    remaining.simd_gt(s(0.5)) & cover.simd_gt(s(0.5)),
                    plane.t,
                    distance,
                );
                remaining *= s(1.0) - cover;
            }
            nearer = Some((plane, layer));
        }
        let sky = sky(ny, view, ray);
        colour = colour + sky * remaining;
        to_colour(expand(colour), distance)
    }
}

/// The sky behind every sheet: a pale gradient from blue to peach at the horizon, as paper.
#[inline(always)]
fn sky(ny: F, view: &View, ray: Vec3Simd) -> Vec3Simd {
    let height = clamp01(ny * s(0.5) + s(0.55));
    let top = rgb(0.74, 0.82, 0.9);
    let horizon = rgb(1.0, 0.9, 0.78);
    let base = mix3(horizon, top, smoothstep(0.35, 1.0, height));
    let t = s(-80.0 - view.eye.z) / ray.z;
    let (x, y) = (view.eye.x + ray.x * t, view.eye.y + ray.y * t);
    base * (s(1.0) + grain(x, y) * s(0.8))
}

/// Paper fibre and speckle, in sheet coordinates, as a multiplier's offset from 1.
#[inline(always)]
fn grain(x: F, y: F) -> F {
    let fibres =
        value(x * s(3.0) + s(2.0), y * s(70.0)) + value(x * s(60.0), y * s(2.5) + s(7.0)) - s(1.0);
    let cloud = value(x * s(0.9), y * s(0.9)) - s(0.5);
    let fine = hash2((x * s(220.0)).floor(), (y * s(220.0)).floor()) - s(0.5);
    fibres * s(0.022) + cloud * s(0.03) + fine * s(0.02)
}

/// A sheet's colour at a pixel `d` from its edge: fibre, the shadow of the sheet in front and of
/// the chip, and the lit or shaded rim of the cut.
#[inline(always)]
fn paper(
    layer: &Layer,
    plane: &Plane,
    d: F,
    pixel: F,
    time: f32,
    view: &View,
    nearer: Option<&(Plane, &Layer)>,
) -> Vec3Simd {
    let mut colour = splat3(layer.colour) * (s(1.0) + grain(plane.x, plane.y));
    if let Some(hills) = &layer.hills {
        let below = (plane.b - s(hills.base)).fast_max(s(-0.7)).fast_min(s(0.1));
        colour = colour * (s(1.0) + below * s(0.2));
    }

    let mut shade = s(0.0);
    if let Some((front, caster)) = nearer {
        let gap = caster.z - layer.z;
        let blur = 0.007 + 0.0012 * gap;
        let offset = 0.012 + 0.0015 * gap;
        let da = s(LIGHT_SCREEN.0 * offset);
        let db = s(LIGHT_SCREEN.1 * offset);
        let ds = caster.distance(front.a + da, front.b + db, time);
        let near = s(1.0) - smoothstep(-blur * 0.4, blur, ds);
        let wide = s(1.0) - smoothstep(-0.02, 0.08, ds);
        shade = near * s(0.34) + wide * s(0.16);
    }
    if layer.z > -30.0 {
        let world = Vec3Simd::new(plane.x, plane.y, s(layer.z));
        shade += chip_shadow(view, LIGHT.normalise(), world) * s(0.4) * smoothstep(-0.8, 0.0, -d);
    }
    let shadow = rgb(0.58, 0.52, 0.66);
    colour = mix3(colour, mul3(colour, shadow), clamp01(shade));

    let band = s(0.012) * s(plane.scale);
    if (d + band).simd_gt(s(0.0)).any() {
        let step = s(0.5) * pixel + s(0.01);
        let toward = layer.distance(
            plane.a + s(LIGHT_SCREEN.0) * step / s(plane.scale),
            plane.b + s(LIGHT_SCREEN.1) * step / s(plane.scale),
            time,
        ) * s(plane.scale);
        let facing = ((toward - d) / step).fast_max(s(-1.0)).fast_min(s(1.0));
        let rim = s(1.0) - smoothstep(0.0, 1.0, clamp01(-d / band));
        colour = colour * (s(1.0) + facing * rim * s(0.2));
    }
    colour
}

pub struct Papercraft {
    look: Look,
    layers: std::sync::Arc<Vec<Layer>>,
    pipeline: SimdPipeline<ScreenVertexShader, Sheets>,
}

impl Papercraft {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.78, 0.68, 0.55),
                die: Vec3::new(0.35, 0.18, 0.15),
                gold: Vec3::new(0.95, 0.68, 0.3),
                streams: Vec3::new(1.0, 0.8, 0.55),
                key: LIGHT.normalise(),
                key_colour: Vec3::new(1.6, 1.5, 1.4),
                sky: Vec3::new(0.7, 0.68, 0.7),
                ground: Vec3::new(0.3, 0.26, 0.24),
                softbox: Vec3::ZERO,
            },
            layers: std::sync::Arc::new(layers()),
            pipeline: SimdPipeline::new(ScreenVertexShader, Sheets)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for Papercraft {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Pizzicato],
            ambience: vec![Ambience::once(
                Source::Recording("page", 0.0),
                0.0,
                0.45,
                0.2,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        let uniforms = Uniforms {
            view: scene.view.clone(),
            time: scene.local_time,
            layers: self.layers.clone(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.pipeline, (), |_| uniforms.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.0,
            vignette: 0.18,
            grain: 0.008,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
