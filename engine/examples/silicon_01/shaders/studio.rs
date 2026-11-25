//! The studio's backdrop, and its depth of field for the macro shots.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::shaders::SKY_DISTANCE;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd};
use crate::simd::{F, clamp01, from_colour, hash2, mix3, rgb, s, splat3, to_colour};

/// Where the backdrop's glows sit this frame.
#[derive(Clone, Copy, Debug)]
pub struct BackdropUniforms {
    /// The chip's centre in normalised device coordinates, which the glow behind it follows.
    pub centre: Vec2,
    /// The target's width over its height.
    pub aspect: f32,
    /// The colour of the chip's light, which tints the floor glow under it as it powers up.
    pub light: Vec3,
}

/// Near black with a faint cool glow behind the chip, a soft floor glow under it that anchors
/// it, and a vignette into the corners.
pub struct BackdropShader;

impl FragmentShaderSimd<ScreenVaryings> for BackdropShader {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let [x, y] = v.ndc();
        let (dx, dy) = (
            (x - s(u.centre.x)) * s(u.aspect),
            y - s(u.centre.y) - s(0.05),
        );
        let behind = (-(dx * dx + dy * dy) * s(0.9)).exp();
        let (fx, fy) = (
            (x - s(u.centre.x)) * s(u.aspect) * s(0.75),
            (y - s(u.centre.y) + s(0.78)) * s(5.0),
        );
        let floor = (-(fx * fx + fy * fy)).exp();
        let corners = clamp01((x * x * s(0.7) + y * y) * s(0.5));
        let base = rgb(0.02, 0.023, 0.028) * (s(1.0) - corners * s(0.55));
        let glow = rgb(0.045, 0.058, 0.08) * behind;
        let floor_light = mix3(rgb(0.05, 0.058, 0.066), splat3(u.light) * s(0.09), s(0.35)) * floor;
        to_colour(base + glow + floor_light, s(SKY_DISTANCE))
    }
}

#[derive(Clone)]
pub struct DepthOfFieldUniforms {
    /// The sharp scene, with view distance in alpha.
    pub source: Arc<TextureSampler>,
    /// The distance in focus.
    pub focus: f32,
    /// The circle of confusion's radius, in pixels of the target, for a surface infinitely far
    /// behind the focus; nearer surfaces blur in proportion to how far out of focus they are.
    pub aperture: f32,
    /// The largest radius any blur reaches, in pixels of the target.
    pub reach: f32,
}

/// Blurs the scene by each pixel's distance from the focus, gathering a disc of samples that
/// each spread as far as their own blur reaches. A sample behind the pixel spreads no further
/// than the pixel's own blur, so in-focus surfaces stay sharp against a blurred background.
/// The disc turns from pixel to pixel, trading banding for a fine grain.
pub struct DepthOfFieldShader;

impl FragmentShaderSimd<ScreenVaryings> for DepthOfFieldShader {
    type Uniforms = DepthOfFieldUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &DepthOfFieldUniforms) -> ColourSimd {
        const TAPS: usize = 24;
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (width, height) = (u.source.width() as f32, u.source.height() as f32);
        let blur = |distance: F| {
            ((s(1.0) - s(u.focus) / distance.fast_max(s(1e-3))).abs() * s(u.aperture))
                .fast_min(s(u.reach))
        };
        let centre = u.source.sample_linear_clamp_simd([x, y]);
        let centre_blur = blur(centre.a);
        let weight_of = |spread: F| s(1.0) / (spread * spread).fast_max(s(1.0));
        let centre_weight = weight_of(centre_blur);
        let mut sum = from_colour(centre) * centre_weight;
        let mut total = centre_weight;
        let turn = hash2(x * s(width), y * s(height)) * s(std::f32::consts::TAU);
        let (sin_turn, cos_turn) = turn.sin_cos();
        for tap in 0..TAPS {
            let radius = ((tap as f32 + 0.5) / TAPS as f32).sqrt();
            let angle = tap as f32 * 2.399_963;
            let (ax, ay) = (angle.cos() * radius, angle.sin() * radius);
            let ox = s(ax) * cos_turn - s(ay) * sin_turn;
            let oy = s(ax) * sin_turn + s(ay) * cos_turn;
            let sample = u.source.sample_linear_clamp_simd([
                x + ox * s(u.reach / width),
                y + oy * s(u.reach / height),
            ]);
            let sample_blur = blur(sample.a);
            let spread = sample
                .a
                .simd_lt(centre.a)
                .select(sample_blur, sample_blur.fast_min(centre_blur));
            let covers = clamp01(spread - s(radius * u.reach) + s(0.5));
            let weight = covers * weight_of(spread);
            sum = sum + from_colour(sample) * weight;
            total += weight;
        }
        to_colour(sum * (s(1.0) / total), centre.a)
    }
}

#[derive(Clone)]
pub struct EdgeSmoothingUniforms {
    /// The scene, with view distance in alpha.
    pub source: Arc<TextureSampler>,
}

/// Smooths jagged silhouettes and highlights (Timothy Lottes' FXAA, in its small "console"
/// form): where brightness changes sharply, it blends along the edge rather than across it.
/// Brightness is judged after a simple tone map, so HDR highlights don't dominate. The view
/// distance in alpha passes through untouched.
pub struct EdgeSmoothingShader;

impl FragmentShaderSimd<ScreenVaryings> for EdgeSmoothingShader {
    type Uniforms = EdgeSmoothingUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &EdgeSmoothingUniforms) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(0.5) + s(0.5), ny * s(0.5) + s(0.5));
        let (tx, ty) = (
            1.0 / u.source.width() as f32,
            1.0 / u.source.height() as f32,
        );
        let at = |dx: F, dy: F| {
            u.source
                .sample_linear_clamp_simd([x + dx * s(tx), y + dy * s(ty)])
        };
        let luma = |c: Vec3Simd| {
            let l = c.x * s(0.299) + c.y * s(0.587) + c.z * s(0.114);
            l / (s(1.0) + l)
        };
        let centre = at(s(0.0), s(0.0));
        let corner = |dx: f32, dy: f32| luma(from_colour(at(s(dx), s(dy))));
        let (nw, ne) = (corner(-0.5, 0.5), corner(0.5, 0.5));
        let (sw, se) = (corner(-0.5, -0.5), corner(0.5, -0.5));
        let middle = luma(from_colour(centre));
        let low = middle.fast_min(nw.fast_min(ne).fast_min(sw.fast_min(se)));
        let high = middle.fast_max(nw.fast_max(ne).fast_max(sw.fast_max(se)));
        let dx = -((nw + ne) - (sw + se));
        let dy = (nw + sw) - (ne + se);
        let reduce = ((nw + ne + sw + se) * s(0.25 / 8.0)).fast_max(s(1.0 / 128.0));
        let scale = s(1.0) / (dx.abs().fast_min(dy.abs()) + reduce);
        let dx = (dx * scale).fast_max(s(-8.0)).fast_min(s(8.0));
        let dy = (dy * scale).fast_max(s(-8.0)).fast_min(s(8.0));
        let tap = |t: f32| from_colour(at(dx * s(t), dy * s(t)));
        let near = (tap(1.0 / 3.0 - 0.5) + tap(2.0 / 3.0 - 0.5)) * s(0.5);
        let far = near * s(0.5) + (tap(-0.5) + tap(0.5)) * s(0.25);
        let far_luma = luma(far);
        let outside = far_luma.simd_lt(low) | far_luma.simd_gt(high);
        let colour = Vec3Simd::new(
            outside.select(near.x, far.x),
            outside.select(near.y, far.y),
            outside.select(near.z, far.z),
        );
        to_colour(colour, centre.a)
    }
}
