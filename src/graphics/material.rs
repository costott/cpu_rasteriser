use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub ambient: Colour,
    pub diffuse: Colour,
    pub specular: Colour,
    pub shininess: f32,
}
impl Material {
    pub fn new(ambient: Colour, diffuse: Colour, specular: Colour, shininess: f32) -> Self {
        Self {
            ambient,
            diffuse,
            specular,
            shininess,
        }
    }
}
