use crate::prelude::*;

#[derive(Debug)]
pub struct Material {
    pub name: String,
    pub ambient: Colour,
    pub diffuse: Colour,
    pub specular: Colour,
    pub shininess: f32,
}
impl Material {
    pub fn new(
        name: String,
        ambient: Colour,
        diffuse: Colour,
        specular: Colour,
        shininess: f32,
    ) -> Self {
        Self {
            name,
            ambient,
            diffuse,
            specular,
            shininess,
        }
    }
}
