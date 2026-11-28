//! Lighting and fog for toon-lit worlds: one sun, a sky-and-ground ambient, optional banding
//! and rim light, and distance fog that thickens towards the ground.

use cpu_rasteriser::prelude::*;

use crate::shaders::View;
use crate::simd::{F, clamp01, length, mix3, mul3, s, smoothstep, splat3};

/// Distance fog towards a colour, the haze a world's far terrain and props fade into.
#[derive(Clone, Copy, Debug)]
pub struct Fog {
    pub colour: Vec3,
    /// How thick the fog is per world unit at `base` height.
    pub density: f32,
    /// How fast it thins with height above `base`; 0 for uniform fog.
    pub falloff: f32,
    pub base: f32,
}

impl Fog {
    /// Fades `colour` into the fog for a surface at `world`, `distance` from the eye.
    #[inline(always)]
    pub fn apply(&self, colour: Vec3Simd, world: Vec3Simd, distance: F) -> Vec3Simd {
        let above = (world.y - s(self.base)).fast_max(s(0.0));
        let density = s(self.density) * (s(-self.falloff) * above).exp();
        mix3(
            colour,
            splat3(self.colour),
            clamp01(s(1.0) - (-density * distance).exp()),
        )
    }
}

/// How a world is lit.
#[derive(Clone, Copy, Debug)]
pub struct Lighting {
    /// The direction towards the sun, unit length.
    pub sun: Vec3,
    pub sun_colour: Vec3,
    /// Ambient light from straight up and from straight down.
    pub sky: Vec3,
    pub ground: Vec3,
    /// How many steps the sun's diffuse term is quantised to, softly; 0 for smooth shading.
    pub bands: f32,
    /// Light added at grazing angles, for a glow round silhouettes.
    pub rim: Vec3,
    pub fog: Fog,
}

impl Lighting {
    /// The colour of a surface of `albedo` at `world` with `normal` (unit, facing the eye), with
    /// `shadow` the fraction of the sun that reaches it (1 when lit).
    #[inline(always)]
    pub fn shade(
        &self,
        view: &View,
        world: Vec3Simd,
        normal: Vec3Simd,
        albedo: Vec3Simd,
        shadow: F,
    ) -> Vec3Simd {
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let mut diffuse = normal.dot(splat3(self.sun)).fast_max(s(0.0));
        if self.bands > 0.0 {
            let stepped = (diffuse * s(self.bands) + s(0.5)).floor() / s(self.bands);
            diffuse = diffuse * s(0.4) + stepped * s(0.6);
        }
        let ambient = mix3(
            splat3(self.ground),
            splat3(self.sky),
            normal.y * s(0.5) + s(0.5),
        );
        let lit = ambient + splat3(self.sun_colour) * (diffuse * shadow);
        let facing = clamp01(normal.dot(to_eye * (s(1.0) / distance)));
        let rim = splat3(self.rim)
            * ((s(1.0) - facing) * (s(1.0) - facing) * smoothstep(0.0, 0.4, normal.y + s(0.3)));
        self.fog.apply(mul3(albedo, lit) + rim, world, distance)
    }
}

/// `normal` turned to face the eye, so one-sided surfaces light correctly from both sides.
#[inline(always)]
pub fn facing(normal: Vec3Simd, to_eye: Vec3Simd) -> Vec3Simd {
    let flip = normal.dot(to_eye).simd_lt(s(0.0)).select(s(-1.0), s(1.0));
    normal * flip
}
