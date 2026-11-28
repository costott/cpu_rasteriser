//! The hologram: a mandelbulb ray-marched inside a proxy cube, drawn as translucent light in the
//! frustum, forming from the die upwards and closing back into it. Its power drifts and its space
//! twists while it tumbles on three axes. It flickers with the rest of the projection.

use cpu_rasteriser::prelude::*;

use crate::director::{Phase, ramp};
use crate::effects::frustum::{Flicker, HEIGHT};
use crate::effects::streak::torn;
use crate::effects::{DIE_TOP, GREEN_BRIGHT};
use crate::shaders::{ViewRef, transform_point};
use crate::simd::{F, clamp01, fract, mix3, rgb, s, smoothstep, splat3, to_colour, v3};

/// Half the width of the cube the bulb is marched in, in the chip's units.
const HALF_SIZE: f32 = 2.07;
/// How far the cube's inscribed sphere reaches in the bulb's own space. The mandelbulb stays
/// within about 1.2 of its centre for every power the hologram uses.
const EXTENT: f32 = 1.3;
/// How high in the frustum the bulb floats, as a fraction of its height.
const LEVEL: f32 = 0.56;
const MAX_STEPS: usize = 48;
const ITERATIONS: usize = 6;
/// The fraction of the distance estimate a step covers; the twist makes the estimate optimistic.
const STEP_SCALE: f32 = 0.8;
const HIT_DISTANCE: f32 = 0.003;

/// When the bulb starts to form and is whole, and when it starts to close and is gone.
const FORM: (f32, f32) = (Phase::Hologram.start() + 2.0, Phase::Hologram.start() + 4.0);
const CLOSE: (f32, f32) = (
    Phase::Hologram.start() + 10.4,
    Phase::Hologram.start() + 12.1,
);

/// How much of the bulb has formed, 0 to 1: it forms and closes from the bottom up.
pub fn formed(time: f32) -> f32 {
    ramp(time, FORM.0, FORM.1) * (1.0 - ramp(time, CLOSE.0, CLOSE.1))
}

#[derive(Clone)]
pub struct HologramUniforms {
    view: ViewRef,
    /// Takes world space to the bulb's own, where the cube's inscribed sphere has radius 1.
    to_bulb: Mat4,
    /// Takes the unit cube to the world.
    cube: Mat4,
    /// The world's vertical axis in the bulb's own space, so the reveal and bands stay level
    /// while it tumbles.
    level: Vec3,
    formed: f32,
    power: f32,
    twist: f32,
    flicker: Flicker,
    /// The height of the bulb's centre above the tear's, and the height of one of the bulb's
    /// units, both in heights of the frustum.
    above_tear: f32,
    unit_height: f32,
}

impl HologramUniforms {
    /// The bulb's placement in `view`'s frame, or `None` while it isn't there.
    pub fn new(view: &ViewRef) -> Option<Self> {
        let frame = &view.frame;
        let formed = formed(frame.time);
        if formed < 0.005 {
            return None;
        }
        let t = frame.time - FORM.0;
        let closing = ramp(frame.time, CLOSE.0, CLOSE.1);
        let height = DIE_TOP + HEIGHT * LEVEL * (1.0 - closing) + 0.4 * closing;
        let size = HALF_SIZE * (0.3 + 0.7 * formed);
        let placement = frame.chip * Mat4::translation(0.0, height, 0.0);
        let tumble = Mat4::rotate_y(0.45 * t + 0.6)
            * Mat4::rotate_x(0.35 + 0.3 * (0.31 * t).sin())
            * Mat4::rotate_z(0.25 * (0.23 * t + 1.0).sin());
        let flicker = Flicker::new(view);
        let up = tumble.data[1];
        Some(Self {
            view: view.clone(),
            to_bulb: Mat4::scaling(1.0 / size, 1.0 / size, 1.0 / size)
                * tumble.inverse()
                * placement.inverse(),
            cube: placement * Mat4::scaling(2.0 * size, 2.0 * size, 2.0 * size),
            level: Vec3::new(up[0], up[1], up[2]),
            formed,
            power: 8.0 + 2.0 * (0.6 * t).sin(),
            twist: 0.45 * (0.37 * t + 0.8).sin(),
            flicker,
            above_tear: (height - DIE_TOP) / HEIGHT - flicker.tear,
            unit_height: size / HEIGHT,
        })
    }
}

#[derive(Interpolate, SimdInterpolate)]
pub struct HologramVaryings {
    world: Vec3,
}

/// Draws the proxy cube, a unit cube scaled and moved to the bulb's bounds.
pub struct HologramVertexShader;

impl VertexShader for HologramVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = HologramUniforms;
    type Varyings = HologramVaryings;

    fn shade(&self, vertex: ObjVertex, u: &HologramUniforms) -> (Vec4, HologramVaryings) {
        let world = (u.cube * vertex.position.to_point4()).xyz();
        let clip = u.flicker.jolted(u.view.view_projection * world.to_point4());
        (clip, HologramVaryings { world })
    }
}

pub struct HologramShader;

impl FragmentShaderSimd<HologramVaryings> for HologramShader {
    type Uniforms = HologramUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: HologramVaryingsSimd, u: &HologramUniforms) -> ColourSimd {
        let view = &u.view;
        let time = view.frame.time;
        let m = &u.to_bulb;
        let entry = v3(v.world);
        let direction = {
            let d = (entry - splat3(view.eye)).normalise();
            let row = |r: [f32; 4]| s(r[0]) * d.x + s(r[1]) * d.y + s(r[2]) * d.z;
            Vec3Simd::new(row(m.data[0]), row(m.data[1]), row(m.data[2])).normalise()
        };
        let start = transform_point(m, entry);
        let start = Vec3Simd::new(start[0], start[1], start[2]);
        let level = splat3(u.level);

        let b = start.dot(direction);
        let discriminant = b * b - start.dot(start) + s(1.0);
        let mut active = discriminant.simd_ge(s(0.0));
        let to_sphere = (-b - discriminant.fast_max(s(0.0)).sqrt()).fast_max(s(0.0));
        let mut position = start + direction * to_sphere;
        let mut travelled = s(0.0);
        let mut glow = s(0.0);
        let mut hit = s(0.0);
        let mut steps = s(0.0);
        let front = s(-1.0 + 2.2 * u.formed);
        let twist = s(u.twist);
        let power = s(u.power);

        for _ in 0..MAX_STEPS {
            let d = bulb(position, power, twist);
            glow +=
                (-(d.fast_max(s(0.0))) * s(30.0)).exp() * s(0.02) * active.select(s(1.0), s(0.0));
            let surface = d.simd_lt(s(HIT_DISTANCE)) & active;
            hit = surface.select(s(1.0), hit);
            let outside = position.dot(position).simd_gt(s(1.02 * 1.02));
            active = active & !surface & !outside;
            if !active.any() {
                break;
            }
            let step = active.select((d * s(STEP_SCALE)).fast_max(s(0.004)), s(0.0));
            position = position + direction * step;
            travelled += step;
            steps += active.select(s(1.0), s(0.0));
        }

        let height = position.dot(level);
        let revealed = height.simd_lt(front).select(s(1.0), s(0.0));
        let hit = hit * revealed;
        let work = steps * s(1.0 / MAX_STEPS as f32);
        let occlusion = s(1.0) - work;
        let depth = clamp01(s(1.0) - travelled * s(0.35));
        let shift = clamp01(height * s(0.5) + s(0.5));
        let colour = mix3(rgb(0.25, 0.92, 0.7), rgb(0.6, 1.0, 0.92), shift);
        let rim = smoothstep(0.2, 0.6, work) * hit;
        let bands = smoothstep(0.88, 1.0, fract(height * s(7.0) - s(time * 1.2)));
        let edge = (-(height - front) * (height - front) * s(260.0)).exp() * hit;
        let tear = torn(s(u.above_tear) + height * s(u.unit_height));
        let surface = colour * (hit * occlusion * depth * s(0.13) + rim * s(1.4));
        let halo = colour * (glow * s(0.22) * revealed);
        let lines = rgb(0.7, 1.0, 0.9) * (bands * hit * s(0.25));
        let scan = splat3(GREEN_BRIGHT) * (edge * s(1.6));
        let lit = (surface + halo + lines + scan) * (tear * s(u.flicker.brightness));
        to_colour(lit, s(0.0))
    }
}

/// The estimated distance to a mandelbulb of the given power, twisted about the vertical by
/// `twist` radians per unit of height (after Inigo Quilez). `p` is in the cube's inscribed
/// sphere's units, which `EXTENT` maps to the bulb's own.
#[inline(always)]
fn bulb(p: Vec3Simd, power: F, twist: F) -> F {
    let p = p * s(EXTENT);
    let (sin, cos) = (p.y * twist).sin_cos();
    let c = Vec3Simd::new(cos * p.x + sin * p.z, p.y, cos * p.z - sin * p.x);
    let mut z = c;
    let mut derivative = s(1.0);
    for _ in 0..ITERATIONS {
        let r2 = z.dot(z);
        let alive = r2.simd_lt(s(4.0));
        if !alive.any() {
            break;
        }
        let r = r2.sqrt().fast_max(s(1e-6));
        let theta = (z.x * z.x + z.y * z.y).sqrt().atan2(z.z) * power;
        let phi = z.y.atan2(z.x) * power;
        let radius_power = ((power - s(1.0)) * r.ln()).exp();
        let next_derivative = power * radius_power * derivative + s(1.0);
        let (sin_theta, cos_theta) = theta.sin_cos();
        let (sin_phi, cos_phi) = phi.sin_cos();
        let scale = radius_power * r;
        z = Vec3Simd::new(
            alive.select(sin_theta * cos_phi * scale + c.x, z.x),
            alive.select(sin_theta * sin_phi * scale + c.y, z.y),
            alive.select(cos_theta * scale + c.z, z.z),
        );
        derivative = alive.select(next_derivative, derivative);
    }
    let r = z.dot(z).sqrt().fast_max(s(1e-6));
    s(0.5) * r.ln() * r / derivative / s(EXTENT)
}
