//! Light sources.

use cpu_rasteriser::prelude::*;

/// A light infinitely far away, such as the sun, that shines in the same direction everywhere.
///
/// The engine doesn't interpret the light: shaders do. The bundled examples treat
/// [`direction`](DirectionalLight::direction) as the direction the light travels, so the
/// direction towards the light is its negation.
///
/// # Examples
///
/// Lambertian diffuse lighting in a fragment shader:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let light = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Colour::WHITE);
///
/// let normal = Vec3::new(0.0, 1.0, 0.0);
/// let diffuse = normal.dot(-light.direction.normalise()).max(0.0);
/// assert_eq!(diffuse, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight {
    /// The direction the light shines in, in world space. It needn't be normalised.
    pub direction: Vec3,
    /// The light's colour and intensity.
    pub colour: Colour,
}
impl DirectionalLight {
    /// Creates a light shining along `direction` with the given colour.
    ///
    /// # Examples
    ///
    /// See [`DirectionalLight`].
    pub fn new(direction: Vec3, colour: Colour) -> Self {
        Self { direction, colour }
    }
}
