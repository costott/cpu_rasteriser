//! The lagoon's water: a flat sea whose colour comes from a baked map of how deep it is.
//!
//! The surface is a disc at sea level shaded in one fragment shader: sand and seagrass seen
//! through water that absorbs red first, caustics, ripples, a Fresnel mix with the sky and
//! foam where the depth map says the water is shallow or something stands in it.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::shaders::scene_colour;
use crate::simd::{F, clamp01, length, mix3, pow2n, reflect, rgb, s, smoothstep, splat3};
use crate::themes::kit::draw::{KitVaryings, KitVaryingsSimd, ToonUniforms};
use crate::themes::kit::noise::{cellular, fbm, value};
use crate::themes::kit::sky::Sky;

/// How many foam rings round things standing in the water a [`WaterUniforms`] holds.
pub const FOAM_SPOTS: usize = 24;

/// How deep the water is over a square of the world, baked once.
pub struct Bathymetry {
    texture: TextureSampler,
    min: Vec2,
    size: f32,
}

impl Bathymetry {
    /// Samples `depth(x, z)` (negative on land) over the square from `min` (x, z), `size` on a
    /// side, into a `resolution` square texture.
    pub fn bake(min: Vec2, size: f32, resolution: usize, depth: impl Fn(f32, f32) -> f32) -> Self {
        let mut pixels = vec![Colour::new(0.0, 0.0, 0.0, 1.0); resolution * resolution];
        for j in 0..resolution {
            for i in 0..resolution {
                let x = min.x + (i as f32 + 0.5) / resolution as f32 * size;
                let z = min.y + (j as f32 + 0.5) / resolution as f32 * size;
                pixels[(resolution - 1 - j) * resolution + i] =
                    Colour::new(depth(x, z), 0.0, 0.0, 1.0);
            }
        }
        let pixels: Arc<[Colour]> = pixels.into();
        let texture = TextureSampler::new(
            pixels,
            resolution as u32,
            resolution as u32,
            WrapMode::Clamp,
            FilterMode::Linear,
        );
        Self { texture, min, size }
    }

    /// The water depth at (`x`, `z`), negative on land; the map's edge value beyond it.
    #[inline(always)]
    pub fn depth(&self, x: F, z: F) -> F {
        let u = (x - s(self.min.x)) * s(1.0 / self.size);
        let v = (z - s(self.min.y)) * s(1.0 / self.size);
        self.texture.sample_linear_clamp_simd([u, v]).r
    }
}

#[derive(Clone)]
pub struct WaterUniforms {
    pub toon: ToonUniforms,
    pub sky: Sky,
    pub bathymetry: Arc<Bathymetry>,
    /// Where things stand in the water: x, z and the radius of the foam ring round each.
    pub foam: Arc<[Vec3; FOAM_SPOTS]>,
}

/// Bright web-like light on the sea floor: two drifting cellular layers multiplied.
#[inline(always)]
fn caustics(x: F, z: F, t: f32) -> F {
    let a = s(1.0) - cellular(x * s(0.8) + s(t * 0.1), z * s(0.8) - s(t * 0.08), t * 0.8);
    let b = s(1.0)
        - cellular(
            x * s(1.2) - s(t * 0.09) + s(9.0),
            z * s(1.2) + s(t * 0.11),
            t * 0.6 + 3.0,
        );
    (pow2n(a * b, 2) * s(4.5)).fast_min(s(1.7))
}

/// Shades the sea; see the [module documentation](self).
pub struct WaterShader;

impl FragmentShaderSimd<KitVaryings> for WaterShader {
    type Uniforms = WaterUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &WaterUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let world = v.world();
        let t = view.frame.time;
        let (x, z) = (world.x, world.z);
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let incoming = to_eye * (s(1.0) / distance);
        let depth = u.bathymetry.depth(x, z);

        let fade = clamp01(s(1.0) - distance * s(0.017));
        let surface = |x: F, z: F| {
            value(x * s(0.8) + s(t * 0.35), z * s(1.0) - s(t * 0.25)) * s(0.8)
                + value(x * s(2.4) - s(t * 0.55), z * s(2.2) + s(t * 0.4)) * s(0.4) * fade
        };
        let e = s(0.1);
        let h0 = surface(x, z);
        let slope = s(0.13) * (s(0.35) + fade * s(0.65));
        let gx = (surface(x + e, z) - h0) / e * slope;
        let gz = (surface(x, z + e) - h0) / e * slope;
        let normal = Vec3Simd::new(-gx, s(1.0), -gz).normalise();

        let cosine = clamp01(normal.dot(incoming));
        let grazing = s(1.0) - cosine;
        let grazing2 = grazing * grazing;
        let fresnel = s(0.025) + s(0.975) * grazing2 * grazing2 * grazing;

        let light = u.toon.lighting;
        let sun = splat3(light.sun);
        let bounce = reflect(incoming * s(-1.0), normal);
        let up = bounce.y.fast_max(s(0.01));
        let haze = splat3(u.sky.horizon);
        let mut sky = mix3(haze, splat3(u.sky.zenith), smoothstep(0.0, 0.7, up).sqrt());
        let cloud = smoothstep(
            0.55,
            0.78,
            fbm(
                bounce.x / (up + s(0.3)) * s(0.9) + s(t * 0.01),
                bounce.z / (up + s(0.3)) * s(0.9),
                3,
            ),
        );
        sky = mix3(
            sky,
            rgb(1.5, 1.5, 1.45),
            cloud * smoothstep(0.02, 0.2, up) * s(0.8),
        );
        let glint = pow2n(clamp01(bounce.dot(sun)), 8) * s(40.0) * smoothstep(0.0, 0.02, up);
        sky = sky + splat3(u.sky.sun_colour) * glint;

        let depth_c = depth.fast_max(s(0.03));
        let patch = fbm(x * s(0.15) + s(4.0), z * s(0.15), 3);
        let bed = mix3(
            rgb(0.9, 0.82, 0.6),
            rgb(0.1, 0.27, 0.2),
            smoothstep(0.58, 0.7, patch),
        );
        let sand = s(0.88) + value(x * s(2.5), z * s(4.5)) * s(0.24);
        let web = s(0.55) + caustics(x, z, t) * s(0.9) * (s(-0.25) * depth_c).exp();
        let absorb = |k: f32| (s(-k) * depth_c).exp();
        let transmit = Vec3Simd::new(
            bed.x * absorb(0.5),
            bed.y * absorb(0.14),
            bed.z * absorb(0.1),
        ) * (sand * web);
        let deepness = smoothstep(2.5, 9.0, depth);
        let body_colour = mix3(rgb(0.0, 0.5, 0.52), rgb(0.0, 0.14, 0.32), deepness);
        let body = body_colour * (s(1.0) - (s(-0.55) * depth_c).exp());
        let mut colour = (transmit * (s(1.0) - deepness) + body) * s(1.85);
        colour = mix3(colour, sky, fresnel * s(0.7));

        let lace = cellular(x * s(1.5) + s(t * 0.1), z * s(1.5), t * 0.7);
        let bubbles = smoothstep(0.2, 0.5, lace);
        let lap = (t * 0.9 + value(x * s(0.4), z * s(0.4)) * s(6.0)).sin() * s(0.16);
        let edge = depth + lap;
        let mut foam = (s(1.0) - smoothstep(0.0, 0.9, edge)) * (s(0.3) + bubbles * s(0.7))
            + (s(1.0) - smoothstep(0.0, 0.1, edge));
        for spot in u.foam.iter() {
            let dx = x - s(spot.x);
            let dz = z - s(spot.y);
            let ring = (dx * dx + dz * dz).sqrt() - s(spot.z);
            let wobble = s(0.12) * (s(t * 1.3) + x * s(2.0)).sin();
            foam += (s(1.0) - smoothstep(0.0, 0.7, ring + wobble))
                * (s(0.35) + bubbles * s(0.65))
                * s(if spot.z > 0.0 { 1.0 } else { 0.0 });
        }
        colour = mix3(colour, rgb(1.7, 1.75, 1.7), clamp01(foam) * s(0.88));

        let colour = light.fog.apply(colour, world, distance);
        scene_colour(view, colour, world)
    }
}
