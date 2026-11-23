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
/// let gold = Material::new("gold".to_owned())
///     .with_ambient(Colour::new(0.25, 0.2, 0.07, 1.0))
///     .with_diffuse(Colour::new(0.75, 0.6, 0.23, 1.0))
///     .with_specular(Colour::new(0.63, 0.56, 0.37, 1.0))
///     .with_shininess(51.2);
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
    /// Creates a material with white ambient, diffuse, and specular colours, zero shininess, and
    /// no textures.
    ///
    /// These are the values [`load_mtl`] starts each material with. Set the others with the
    /// `with_` methods, such as [`with_diffuse`](Material::with_diffuse).
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let red = Material::new("red".to_owned());
    /// assert_eq!(red.ambient, Colour::WHITE);
    /// assert_eq!(red.shininess, 0.0);
    /// assert!(red.diffuse_texture.is_none());
    /// ```
    pub fn new(name: String) -> Self {
        Self {
            name,
            ambient: Colour::WHITE,
            diffuse: Colour::WHITE,
            specular: Colour::WHITE,
            shininess: 0.0,
            ambient_texture: None,
            diffuse_texture: None,
            specular_texture: None,
            normal_texture: None,
        }
    }

    /// Sets the ambient colour.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let dim = Material::new("dim".to_owned()).with_ambient(Colour::new(0.1, 0.1, 0.1, 1.0));
    /// assert_eq!(dim.ambient, Colour::new(0.1, 0.1, 0.1, 1.0));
    /// ```
    pub fn with_ambient(mut self, ambient: Colour) -> Self {
        self.ambient = ambient;
        self
    }

    /// Sets the diffuse colour.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let red = Material::new("red".to_owned()).with_diffuse(Colour::RED);
    /// assert_eq!(red.diffuse, Colour::RED);
    /// ```
    pub fn with_diffuse(mut self, diffuse: Colour) -> Self {
        self.diffuse = diffuse;
        self
    }

    /// Sets the specular colour.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let matte = Material::new("matte".to_owned()).with_specular(Colour::BLACK);
    /// assert_eq!(matte.specular, Colour::BLACK);
    /// ```
    pub fn with_specular(mut self, specular: Colour) -> Self {
        self.specular = specular;
        self
    }

    /// Sets the shininess.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let glossy = Material::new("glossy".to_owned()).with_shininess(64.0);
    /// assert_eq!(glossy.shininess, 64.0);
    /// ```
    pub fn with_shininess(mut self, shininess: f32) -> Self {
        self.shininess = shininess;
        self
    }

    /// Sets the ambient colour texture.
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
    /// let floor = Material::new("floor".to_owned()).with_ambient_texture(checker);
    /// assert!(floor.ambient_texture.is_some());
    /// ```
    pub fn with_ambient_texture(mut self, texture: TextureSampler) -> Self {
        self.ambient_texture = Some(texture);
        self
    }

    /// Sets the diffuse colour texture.
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
    /// let floor = Material::new("floor".to_owned()).with_diffuse_texture(checker);
    /// assert!(floor.diffuse_texture.is_some());
    /// ```
    pub fn with_diffuse_texture(mut self, texture: TextureSampler) -> Self {
        self.diffuse_texture = Some(texture);
        self
    }

    /// Sets the specular colour texture.
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
    /// let floor = Material::new("floor".to_owned()).with_specular_texture(checker);
    /// assert!(floor.specular_texture.is_some());
    /// ```
    pub fn with_specular_texture(mut self, texture: TextureSampler) -> Self {
        self.specular_texture = Some(texture);
        self
    }

    /// Sets the normal or bump texture.
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
    /// let floor = Material::new("floor".to_owned()).with_normal_texture(checker);
    /// assert!(floor.normal_texture.is_some());
    /// ```
    pub fn with_normal_texture(mut self, texture: TextureSampler) -> Self {
        self.normal_texture = Some(texture);
        self
    }
}
