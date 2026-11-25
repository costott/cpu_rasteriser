//! The straight lines letters are drawn with: anti-aliased quads, alpha blended over the frame.

use cpu_rasteriser::prelude::*;

use crate::simd::{clamp01, s, to_colour, v3};
use crate::typography::ndc;

/// One corner of a stroke's quad.
#[derive(Clone, Copy, Debug)]
pub struct StrokeVertex {
    position: Vec2,
    /// In window pixels: the corner's distance along the stroke from its start and across it from
    /// its axis, the stroke's length, and how far its ink reaches past each end.
    local: Vec4,
    /// Half the stroke's width, in window pixels.
    half_width: f32,
    colour: Vec3,
    opacity: f32,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct StrokeVaryings {
    local: Vec4,
    half_width: f32,
    colour: Vec3,
    opacity: f32,
}

/// A straight stroke between two points in window pixels, shading from one colour to the other.
#[derive(Clone, Copy)]
pub struct Stroke {
    pub from: Vec2,
    pub to: Vec2,
    pub from_colour: Vec3,
    pub to_colour: Vec3,
    pub half_width: f32,
    /// How far the ink reaches past each end.
    pub cap: f32,
    pub opacity: f32,
}

impl Stroke {
    /// A square of ink centred on `centre`, `half_side` from it to each side.
    pub fn square(centre: Vec2, half_side: f32, colour: Vec3, opacity: f32) -> Self {
        Self {
            from: centre,
            to: centre,
            from_colour: colour,
            to_colour: colour,
            half_width: half_side,
            cap: half_side,
            opacity,
        }
    }

    /// Adds the stroke's quad, for a window of `window` pixels.
    pub fn push(
        &self,
        window: (usize, usize),
        vertices: &mut Vec<StrokeVertex>,
        indices: &mut Vec<u32>,
    ) {
        let axis = self.to - self.from;
        let length = axis.length();
        let direction = if length > 1e-4 {
            axis * (1.0 / length)
        } else {
            Vec2::new(1.0, 0.0)
        };
        let normal = Vec2::new(-direction.y, direction.x);
        // A pixel beyond the ink, for its anti-aliased edge.
        let (past, aside) = (self.cap + 1.0, self.half_width + 1.0);
        let base = vertices.len() as u32;
        let corners = [
            (-past, -aside, self.from_colour),
            (length + past, -aside, self.to_colour),
            (length + past, aside, self.to_colour),
            (-past, aside, self.from_colour),
        ];
        for (along, across, colour) in corners {
            let point = self.from + direction * along + normal * across;
            vertices.push(StrokeVertex {
                position: ndc(point, window),
                local: Vec4::new(along, across, length, self.cap),
                half_width: self.half_width,
                colour,
                opacity: self.opacity,
            });
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

pub struct StrokeVertexShader;

impl VertexShader for StrokeVertexShader {
    type Vertex = StrokeVertex;
    type Uniforms = ();
    type Varyings = StrokeVaryings;

    fn shade(&self, vertex: StrokeVertex, _: &()) -> (Vec4, StrokeVaryings) {
        (
            Vec4::new(vertex.position.x, vertex.position.y, 0.0, 1.0),
            StrokeVaryings {
                local: vertex.local,
                half_width: vertex.half_width,
                colour: vertex.colour,
                opacity: vertex.opacity,
            },
        )
    }
}

pub struct StrokeShader;

impl FragmentShaderSimd<StrokeVaryings> for StrokeShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: StrokeVaryingsSimd, _: &()) -> ColourSimd {
        let [along, across, length, cap] = v.local;
        let past_end = (-along).fast_max(along - length) - cap;
        let outside = past_end.fast_max(across.abs() - v.half_width);
        let coverage = clamp01(s(0.5) - outside);
        to_colour(v3(v.colour), coverage * v.opacity)
    }
}
