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
    pub material: &'a Material,
}

pub struct BasicFragmentShader;
impl FragmentShader for BasicFragmentShader {
    fn shade(&self, fragment: Fragment, _uniforms: &FragmentUniforms) -> Option<Fragment> {
        Some(fragment)
    }
}

pub struct PhongFragmentShader;
impl FragmentShader for PhongFragmentShader {
    fn shade(&self, fragment: Fragment, uniforms: &FragmentUniforms) -> Option<Fragment> {
        let normal = fragment.normal.normalise();
        let mut colour = fragment.colour;

        for light in uniforms.lights {
            let light_dir = light.direction.normalise();
            let diffuse_intensity = normal.dot(&-light_dir).max(0.0);
            let diffuse_colour = light.colour * diffuse_intensity;

            colour = colour * diffuse_colour;
        }

        Some(Fragment {
            position: fragment.position,
            colour,
            depth: fragment.depth,
            normal: fragment.normal,
        })
    }
}
