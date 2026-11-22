//! Surface materials, in the Phong-style form of MTL material libraries.

use cpu_rasteriser::prelude::*;

use crate::prelude::*;

/// How a surface reflects light: Phong-style colours and shininess, with optional texture maps.
///
/// The fields mirror the MTL statements [`load_mtl`] reads (`Ka`, `Kd`, `Ks`, `Ns`, and the `map_`
/// statements). The engine doesn't interpret a material: shaders do, typically by passing the
/// material of each mesh as fragment uniforms, as in [`Model::draw_to_render_pass`]. A texture
/// usually modulates or replaces its matching colour.
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let gold = Material::new_simple(
///     "gold".to_owned(),
///     Colour::new(0.25, 0.2, 0.07, 1.0),
///     Colour::new(0.75, 0.6, 0.23, 1.0),
///     Colour::new(0.63, 0.56, 0.37, 1.0),
///     51.2,
/// );
///
/// // A Blinn-Phong specular term, where `n_dot_h` is the cosine of the angle between the normal
/// // and the half vector.
/// let n_dot_h: f32 = 0.99;
/// let specular = n_dot_h.powf(gold.shininess);
/// assert!((0.5..0.7).contains(&specular));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    /// The material's name, which `usemtl` statements in OBJ files refer to.
    pub name: String,

    /// The colour reflected from ambient light (`Ka`).
    pub ambient: Colour,
    /// The colour reflected diffusely (`Kd`).
    pub diffuse: Colour,
    /// The colour of specular highlights (`Ks`).
    pub specular: Colour,
    /// The specular exponent (`Ns`): higher values give smaller, sharper highlights.
    pub shininess: f32,

    /// The ambient colour texture (`map_Ka`).
    pub ambient_texture: Option<TextureSampler>,
    /// The diffuse colour texture (`map_Kd`).
    pub diffuse_texture: Option<TextureSampler>,
    /// The specular colour texture (`map_Ks`).
    pub specular_texture: Option<TextureSampler>,
    /// The normal or bump map (`map_Bump`, `bump`, or `norm`).
    pub normal_texture: Option<TextureSampler>,
}
impl Material {
    /// Creates a material with the given colours, shininess, and textures.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let checker = TextureSampler::new(
    ///     Arc::from([Colour::BLACK, Colour::WHITE, Colour::WHITE, Colour::BLACK]),
    ///     2,
    ///     2,
    ///     WrapMode::Repeat,
    ///     FilterMode::Nearest,
    /// );
    ///
    /// let floor = Material::new(
    ///     "floor".to_owned(),
    ///     Colour::BLACK,
    ///     Colour::WHITE,
    ///     Colour::BLACK,
    ///     1.0,
    ///     None,
    ///     Some(checker),
    ///     None,
    ///     None,
    /// );
    /// assert!(floor.diffuse_texture.is_some());
    /// ```
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

    /// Creates an untextured material with the given colours and shininess.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let matte = Material::new_simple(
    ///     "matte".to_owned(),
    ///     Colour::BLACK,
    ///     Colour::new(0.5, 0.5, 0.5, 1.0),
    ///     Colour::BLACK,
    ///     0.0,
    /// );
    /// assert!(matte.diffuse_texture.is_none() && matte.normal_texture.is_none());
    /// ```
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

    /// Creates a material with white ambient, diffuse, and specular colours, zero shininess, and
    /// no textures.
    ///
    /// These are the values [`load_mtl`] starts each material with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let mut red = Material::named("red".to_owned());
    /// red.diffuse = Colour::RED;
    /// assert_eq!(red.ambient, Colour::WHITE);
    /// ```
    pub fn named(name: String) -> Self {
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
