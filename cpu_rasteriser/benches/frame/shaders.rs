use cpu_rasteriser::prelude::*;

#[derive(Clone)]
pub(crate) struct Vertex {
    pub(crate) position: Vec4,
    pub(crate) colour: Vec4,
}

#[derive(Interpolate, SimdInterpolate)]
pub(crate) struct Varyings {
    colour: Vec4,
}

pub(crate) struct PassThrough;
impl VertexShader for PassThrough {
    type Vertex = Vertex;
    type Uniforms = ();
    type Varyings = Varyings;

    fn shade(&self, vertex: Vertex, _: &()) -> (Vec4, Varyings) {
        (
            vertex.position,
            Varyings {
                colour: vertex.colour,
            },
        )
    }
}

/// Outputs the interpolated vertex colour. Both implementations live here because the derive
/// makes the fields of `VaryingsSimd` private to this module.
pub(crate) struct VertexColour;
impl FragmentShader<Varyings> for VertexColour {
    type Uniforms = ();

    fn shade(&self, varyings: Varyings, _: &()) -> Colour {
        Colour::from(varyings.colour)
    }
}
impl FragmentShaderSimd<Varyings> for VertexColour {
    type Uniforms = ();

    fn shade_simd(&self, varyings: VaryingsSimd, _: &()) -> ColourSimd {
        let [r, g, b, a] = varyings.colour;
        ColourSimd { r, g, b, a }
    }
}
