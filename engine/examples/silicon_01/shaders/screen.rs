//! Full-screen passes: the quad they draw, and bloom over the finished HDR image.
//! The final composite is in `composite`.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::simd::{F, clamp01, from_colour, s, to_colour};

#[derive(Interpolate, SimdInterpolate)]
pub struct ScreenVaryings {
    pub ndc: Vec2,
}

/// The depth the full-screen quad is drawn at: behind anything the camera can see, up to its
/// far plane, so a backdrop can be drawn after opaque geometry and shaded only where nothing
/// covers it.
const SKY_DEPTH: f32 = 0.99999;

/// Passes a full-screen quad through, at [`SKY_DEPTH`].
pub struct ScreenVertexShader;

impl VertexShader for ScreenVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = ();
    type Varyings = ScreenVaryings;

    fn shade(&self, vertex: ObjVertex, _: &()) -> (Vec4, ScreenVaryings) {
        let p = vertex.position;
        (
            Vec4::new(p.x, p.y, SKY_DEPTH, 1.0),
            ScreenVaryings {
                ndc: Vec2::new(p.x, p.y),
            },
        )
    }
}

#[inline(always)]
fn uv(v: &ScreenVaryingsSimd) -> [F; 2] {
    [v.ndc[0] * s(0.5) + s(0.5), v.ndc[1] * s(0.5) + s(0.5)]
}

#[derive(Clone)]
pub struct BrightUniforms {
    pub source: Arc<TextureSampler>,
    pub threshold: f32,
}

/// Keeps what is brighter than the threshold, with a soft knee, averaging four source texels
/// so it can write to a half-resolution target.
pub struct BrightShader;

impl FragmentShaderSimd<ScreenVaryings> for BrightShader {
    type Uniforms = BrightUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BrightUniforms) -> ColourSimd {
        let [x, y] = uv(&v);
        let (tx, ty) = (
            0.5 / u.source.width() as f32,
            0.5 / u.source.height() as f32,
        );
        let mut sum = Vec3Simd::new(s(0.0), s(0.0), s(0.0));
        for (dx, dy) in [(-tx, -ty), (tx, -ty), (-tx, ty), (tx, ty)] {
            sum = sum + from_colour(u.source.sample_linear_clamp_simd([x + s(dx), y + s(dy)]));
        }
        let colour = sum * s(0.25);
        let luminance = colour.x * s(0.2126) + colour.y * s(0.7152) + colour.z * s(0.0722);
        let knee = clamp01((luminance - s(u.threshold)) * s(1.5));
        to_colour(colour * (knee * knee), s(1.0))
    }
}

#[derive(Clone, Copy)]
pub enum Resample {
    /// A four-tap box filter, for halving the resolution.
    Down,
    /// A nine-tap tent filter, for doubling it smoothly.
    Up,
}

#[derive(Clone)]
pub struct ResampleUniforms {
    pub source: Arc<TextureSampler>,
    pub mode: Resample,
}

pub struct ResampleShader;

impl FragmentShaderSimd<ScreenVaryings> for ResampleShader {
    type Uniforms = ResampleUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &ResampleUniforms) -> ColourSimd {
        let [x, y] = uv(&v);
        let (tx, ty) = (
            1.0 / u.source.width() as f32,
            1.0 / u.source.height() as f32,
        );
        let tap = |dx: f32, dy: f32| {
            from_colour(
                u.source
                    .sample_linear_clamp_simd([x + s(dx * tx), y + s(dy * ty)]),
            )
        };
        let colour = match u.mode {
            Resample::Down => {
                (tap(-1.0, -1.0) + tap(1.0, -1.0) + tap(-1.0, 1.0) + tap(1.0, 1.0)) * s(0.25)
            }
            Resample::Up => {
                let edges = tap(-1.0, 0.0) + tap(1.0, 0.0) + tap(0.0, -1.0) + tap(0.0, 1.0);
                let corners = tap(-1.0, -1.0) + tap(1.0, -1.0) + tap(-1.0, 1.0) + tap(1.0, 1.0);
                (tap(0.0, 0.0) * s(4.0) + edges * s(2.0) + corners) * s(1.0 / 16.0)
            }
        };
        to_colour(colour, s(1.0))
    }
}

impl ScreenVaryingsSimd {
    /// The fragments' normalised device coordinates, for full-screen passes in other modules.
    #[inline(always)]
    pub fn ndc(&self) -> [F; 2] {
        self.ndc
    }
}
