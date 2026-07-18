use crate::prelude::*;

use crate::graphics::lighting::DirectionalLight;

pub trait VertexShader {
    /// Processes a world-space vertex, before projection and clipping.
    fn shade(&self, vertex: Vertex3D, uniforms: &VertexUniforms) -> Vertex3D;
}

pub struct VertexUniforms<'a> {
    pub lights: &'a [DirectionalLight],
}

pub struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    fn shade(&self, vertex: Vertex3D, _uniforms: &VertexUniforms) -> Vertex3D {
        vertex
    }
}

pub struct GouraudVertexShader;
impl VertexShader for GouraudVertexShader {
    fn shade(&self, mut vertex: Vertex3D, uniforms: &VertexUniforms) -> Vertex3D {
        let normal = vertex.normal.normalise();
        let mut colour = vertex.colour;

        for light in uniforms.lights {
            let light_dir = light.direction.normalise();
            let diffuse_intensity = normal.dot(&-light_dir).max(0.0);
            let diffuse_colour = light.colour * diffuse_intensity;

            colour = colour * diffuse_colour;
        }

        vertex.colour = colour;
        vertex
    }
}
