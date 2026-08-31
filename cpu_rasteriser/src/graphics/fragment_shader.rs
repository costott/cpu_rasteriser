use crate::prelude::*;

/// A trait for fragment shaders, which process fragments after rasterisation.
///
/// # Example
/// ```ignore
/// #[derive(Interpolate)]
/// struct Varyings {
///     pub world_position: Vec3,
///     pub colour: Colour,
///     pub normal: Vec3,
/// }
///
/// struct FragmentUniforms {
///    camera: Camera,
///    lights: Vec<DirectionalLight>,
///    material: Option<Material>,
/// }
///
/// /// Reflects a vector around a normal, using the formula: R = V - 2 * (V . N) * N
/// fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
///     vector - normal * 2.0 * vector.dot(&normal)
/// }
///
/// struct PhongFragmentShader;
/// impl FragmentShader<Varyings> for PhongFragmentShader {
///     type Uniforms = FragmentUniforms;
///
///     fn shade(&self, varyings: Varyings, uniforms: &Self::Uniforms) -> Colour {
///         let normal = varyings.normal.normalise();
///
///         let mut colour = Colour::BLACK;
///
///         let view_dir = (uniforms.camera.eye - varyings.world_position).normalise();
///
///         if let Some(material) = &uniforms.material {
///             colour = material.ambient;
///
///             for light in &uniforms.lights {
///                 let light_dir = (-light.direction).normalise();
///
///                 // Diffuse
///                 let diffuse_strength = normal.dot(&light_dir).max(0.0);
///
///                 let diffuse = material.diffuse * light.colour * diffuse_strength;
///
///                 // Specular
///                 let reflect_dir = reflect(-light_dir, normal);
///
///                 let specular_strength =
///                     view_dir.dot(&reflect_dir).max(0.0).powf(material.shininess);
///
///                 let specular = material.specular * light.colour * specular_strength;
///
///                 colour = colour + diffuse + specular;
///             }
///         }
///
///         colour
///     }
/// }
/// ```
pub trait FragmentShader<V>: Send + Sync + 'static
where
    V: Interpolate,
{
    type Uniforms: Send + Sync + 'static;

    fn shade(&self, varyings: V, uniforms: &Self::Uniforms) -> Colour;
}

pub trait FragmentShaderSimd<V>: FragmentShader<V>
where
    V: SimdInterpolate,
{
    fn shade_simd(&self, varyings: V::Simd, uniforms: &Self::Uniforms) -> ColourSimd;
}
