use crate::prelude::*;

pub struct DirectionalLight {
    pub direction: Vec3,
    pub colour: Colour,
}
impl DirectionalLight {
    pub fn new(direction: Vec3, colour: Colour) -> Self {
        Self { direction, colour }
    }
}
