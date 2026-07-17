use crate::prelude::*;

use crate::graphics::lighting::DirectionalLight;

pub trait VertexShader {
    fn shade(&self, vertex: Vertex3D) -> Vertex3D;
}

pub struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    fn shade(&self, vertex: Vertex3D) -> Vertex3D {
        vertex
    }
}

pub struct GouraudVertexShader {
    pub light: DirectionalLight,
}
impl GouraudVertexShader {
    pub fn new(light: DirectionalLight) -> Self {
        Self { light }
    }
}
impl VertexShader for GouraudVertexShader {
    fn shade(&self, mut vertex: Vertex3D) -> Vertex3D {
        let normal = vertex.normal.normalise();
        let light_dir = self.light.direction.normalise();
        let diffuse_intensity = normal.dot(&-light_dir).max(0.0);
        let diffuse_colour = self.light.colour * diffuse_intensity;

        vertex.colour = vertex.colour * diffuse_colour;
        vertex
    }
}
