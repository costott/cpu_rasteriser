use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::graphics::fragment::Fragment;
use crate::graphics::lighting::DirectionalLight;

pub trait FragmentShader {
    fn shade(&self, fragment: Fragment, uniforms: &FragmentUniforms) -> Option<Fragment>;
}

pub struct FragmentUniforms<'a> {
    pub camera: &'a Camera,
    pub lights: &'a [DirectionalLight],
    pub material: Option<&'a Material>,
}

pub struct BasicFragmentShader;
impl FragmentShader for BasicFragmentShader {
    fn shade(&self, fragment: Fragment, _uniforms: &FragmentUniforms) -> Option<Fragment> {
        Some(fragment)
    }
}

pub struct PhongFragmentShader;
impl PhongFragmentShader {
    fn phong_shade(
        &self,
        fragment: Fragment,
        uniforms: &FragmentUniforms,
        material: &Material,
    ) -> Option<Fragment> {
        let normal = fragment.normal.normalise();

        let mut colour = material.ambient;

        let view_dir = (uniforms.camera.eye - fragment.world_position).normalise();

        for light in uniforms.lights {
            let light_dir = (-light.direction).normalise();

            // Diffuse
            let diffuse_strength = normal.dot(&light_dir).max(0.0);

            let diffuse = material.diffuse * light.colour * diffuse_strength;

            // Specular
            let reflect_dir = reflect(-light_dir, normal);

            let specular_strength = view_dir.dot(&reflect_dir).max(0.0).powf(material.shininess);

            let specular = material.specular * light.colour * specular_strength;

            colour = colour + diffuse + specular;
        }

        Some(Fragment { colour, ..fragment })
    }
}
impl FragmentShader for PhongFragmentShader {
    fn shade(&self, fragment: Fragment, uniforms: &FragmentUniforms) -> Option<Fragment> {
        if let Some(material) = uniforms.material {
            self.phong_shade(fragment, uniforms, material)
        } else {
            Some(fragment)
        }
    }
}

/// Reflects a vector around a normal, using the formula: R = V - 2 * (V . N) * N
pub fn reflect(vector: Vec3, normal: Vec3) -> Vec3 {
    vector - normal * 2.0 * vector.dot(&normal)
}
