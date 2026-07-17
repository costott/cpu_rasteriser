use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex3D {
    pub position: Vec3,
    pub colour: Colour,
}
impl Vertex3D {
    pub fn new(position: Vec3, colour: Colour) -> Self {
        Self { position, colour }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            colour: self.colour.lerp(&other.colour, t),
        }
    }
}

impl From<(f32, f32, f32, Colour)> for Vertex3D {
    fn from((x, y, z, colour): (f32, f32, f32, Colour)) -> Self {
        Self {
            position: Vec3::new(x, y, z),
            colour,
        }
    }
}
