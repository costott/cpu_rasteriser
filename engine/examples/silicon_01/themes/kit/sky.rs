//! Skies: a gradient with a sun disc and glow, drifting clouds and stars, drawn as a full-screen
//! quad behind the world.
//!
//! Draw the quad after opaque geometry with a pipeline of [`SkyShader`] on
//! [`ScreenVertexShader`](crate::shaders::screen::ScreenVertexShader) and
//! `DepthState::READ_ONLY`, so it shades only uncovered pixels. [`stars`] and [`sun_glow`] are
//! also free functions, for backdrops that aren't a sky.

use cpu_rasteriser::prelude::*;

use super::noise::{fbm, value};
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{F, clamp01, hash2, mix3, rgb, s, smoothstep, splat3, to_colour};

/// A layer of clouds on a flat plane above the world.
#[derive(Clone, Copy, Debug)]
pub struct Clouds {
    pub height: f32,
    /// How much of the sky they cover, 0 to 1.
    pub coverage: f32,
    /// Lattice cells per world unit.
    pub scale: f32,
    /// How fast they drift, in world units per second (x, z).
    pub wind: Vec2,
    /// The colour of sunlit edges and of shaded bellies.
    pub lit: Vec3,
    pub shade: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Sky {
    pub zenith: Vec3,
    pub horizon: Vec3,
    /// The direction towards the sun, unit length, and its colour.
    pub sun: Vec3,
    pub sun_colour: Vec3,
    /// The sun disc's angular radius, in radians.
    pub sun_size: f32,
    /// How strongly the sun's glow tints the horizon haze.
    pub glow: f32,
    pub clouds: Option<Clouds>,
    /// How many stars there are, 0 (none) to 1 (a dense field).
    pub stars: f32,
}

#[derive(Clone)]
pub struct SkyUniforms {
    pub view: ViewRef,
    pub sky: Sky,
}

/// Shades uncovered pixels with the [`Sky`] in [`SkyUniforms`]; see the [module documentation](self).
pub struct SkyShader;

impl FragmentShaderSimd<ScreenVaryings> for SkyShader {
    type Uniforms = SkyUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &SkyUniforms) -> ColourSimd {
        let [x, y] = v.ndc();
        let ray = view_ray(&u.view, x, y);
        let sky = &u.sky;
        let time = s(u.view.frame.time);
        let up = ray.y.fast_max(s(0.0));
        let cos_sun = ray.dot(splat3(sky.sun));

        let toward_sun = clamp01(cos_sun * s(0.5) + s(0.5));
        let haze = mix3(
            splat3(sky.horizon),
            splat3(sky.sun_colour) * s(0.6) + splat3(sky.horizon) * s(0.4),
            toward_sun.sqrt().sqrt() * toward_sun * s(sky.glow),
        );
        let mut colour = mix3(haze, splat3(sky.zenith), smoothstep(0.0, 0.7, up).sqrt());
        colour = colour + splat3(sky.sun_colour) * (sun_glow(cos_sun, sky.sun_size) * s(0.5));

        if sky.stars > 0.0 {
            let dark = clamp01(s(1.0) - (colour.x + colour.y + colour.z) * s(2.0));
            colour =
                colour + rgb(1.0, 0.95, 0.9) * (stars(ray.x, ray.y, ray.z, sky.stars, time) * dark);
        }

        if let Some(clouds) = &sky.clouds {
            colour = cloud_layer(colour, ray, cos_sun, up, time, sky, clouds);
        }
        to_colour(colour, s(SKY_DISTANCE))
    }
}

/// The sun's disc and the glow around it for the cosine of the angle from the sun, as a
/// brightness to scale the sun's colour by.
#[inline(always)]
pub fn sun_glow(cos_sun: F, size: f32) -> F {
    let angle = cos_sun.fast_min(s(1.0)).acos();
    let disc = s(1.0) - smoothstep(size * 0.92, size * 1.08, angle);
    let halo = (-angle * s(5.0)).exp() * s(0.55);
    let bloom = (-angle * s(1.4)).exp() * s(0.25);
    disc * s(14.0) + halo + bloom
}

/// Points of light on the sky, `density` from 0 to 1, with a slow twinkle. Pass the unit view
/// direction's components.
#[inline(always)]
pub fn stars(x: F, y: F, z: F, density: f32, time: F) -> F {
    const GRID: f32 = 150.0;
    let (gx, gy, gz) = (x * s(GRID), y * s(GRID), z * s(GRID));
    let (cx, cy, cz) = (gx.floor(), gy.floor(), gz.floor());
    let h = hash2(cx * s(1.7) + cz * s(5.3), cy * s(2.9) + cz * s(3.1));
    let present = (h - s(1.0 - 0.03 * density)).fast_max(s(0.0)) * s(1.0 / (0.03 * density));
    let frac = |k: f32| (h * s(k)) - (h * s(k)).floor();
    let centre = (
        frac(17.3) * s(0.5) + s(0.25),
        frac(91.7) * s(0.5) + s(0.25),
        frac(53.1) * s(0.5) + s(0.25),
    );
    let (dx, dy, dz) = (gx - cx - centre.0, gy - cy - centre.1, gz - cz - centre.2);
    let point = (-(dx * dx + dy * dy + dz * dz) * s(70.0)).exp();
    let twinkle = s(0.75) + s(0.25) * (time * s(1.5) + h * s(40.0)).sin();
    present * point * (s(0.4) + frac(7.7) * s(1.6)) * twinkle
}

#[inline(always)]
fn cloud_layer(
    colour: Vec3Simd,
    ray: Vec3Simd,
    cos_sun: F,
    up: F,
    time: F,
    sky: &Sky,
    clouds: &Clouds,
) -> Vec3Simd {
    let reach = s(clouds.height) / up.fast_max(s(0.02));
    let wind = Vec2::new(clouds.wind.x, clouds.wind.y);
    let px = ray.x * reach + time * s(wind.x);
    let pz = ray.z * reach + time * s(wind.y);
    let (u, w) = (px * s(clouds.scale), pz * s(clouds.scale));
    let density = fbm(u, w, 4);
    let cover = smoothstep(1.0 - clouds.coverage, 1.0 - clouds.coverage + 0.22, density);
    let towards = s(0.35);
    let thinner = fbm(u + towards * s(sky.sun.x), w + towards * s(sky.sun.z), 3);
    let edge = clamp01((density - thinner) * s(4.0) + s(0.5));
    let sun_side = clamp01(cos_sun * s(0.5) + s(0.55));
    let body = mix3(
        splat3(clouds.shade),
        splat3(clouds.lit),
        edge * sun_side * s(0.8) + s(0.2) * sun_side,
    );
    let fade = smoothstep(0.02, 0.22, up) * (s(0.9) + value(u * s(0.3), w * s(0.3)) * s(0.1));
    mix3(colour, body, cover * fade)
}
