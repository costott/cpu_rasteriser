use crate::prelude::*;

use cpu_rasteriser::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub name: String,

    pub ambient: Colour,
    pub diffuse: Colour,
    pub specular: Colour,
    pub shininess: f32,

    pub ambient_texture: Option<TextureSampler>,
    pub diffuse_texture: Option<TextureSampler>,
    pub specular_texture: Option<TextureSampler>,
    pub normal_texture: Option<TextureSampler>,
}
impl Material {
    pub fn new(
        name: String,
        ambient: Colour,
        diffuse: Colour,
        specular: Colour,
        shininess: f32,
        ambient_texture: Option<TextureSampler>,
        diffuse_texture: Option<TextureSampler>,
        specular_texture: Option<TextureSampler>,
        normal_texture: Option<TextureSampler>,
    ) -> Self {
        Self {
            name,
            ambient,
            diffuse,
            specular,
            shininess,
            ambient_texture,
            diffuse_texture,
            specular_texture,
            normal_texture,
        }
    }

    pub fn new_simple(
        name: String,
        ambient: Colour,
        diffuse: Colour,
        specular: Colour,
        shininess: f32,
    ) -> Self {
        Self::new(
            name, ambient, diffuse, specular, shininess, None, None, None, None,
        )
    }

    pub fn default(name: String) -> Self {
        Self::new(
            name,
            Colour::WHITE,
            Colour::WHITE,
            Colour::WHITE,
            0.0,
            None,
            None,
            None,
            None,
        )
    }
}
