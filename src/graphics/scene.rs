use crate::prelude::*;

use crate::graphics::camera::Camera;
use crate::graphics::lighting::DirectionalLight;

pub struct Scene {
    pub camera: Camera,
    models: Vec<Model>,
    lights: Vec<DirectionalLight>,
}
impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            models: Vec::new(),
            lights: Vec::new(),
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn add_model(&mut self, model: Model) {
        self.models.push(model);
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    pub fn add_light(&mut self, light: DirectionalLight) {
        self.lights.push(light);
    }

    pub fn lights(&self) -> &[DirectionalLight] {
        &self.lights
    }
}
