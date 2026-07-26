use crate::prelude::*;

pub struct Fragment {
    pub position: Vec2,
    pub world_position: Vec3,

    pub depth: f32,

    pub colour: Colour,
    pub normal: Vec3,
}
impl Fragment {
    pub fn new(
        position: Vec2,
        world_position: Vec3,
        colour: Colour,
        normal: Vec3,
        depth: f32,
    ) -> Self {
        Self {
            position,
            world_position,
            colour,
            normal,
            depth,
        }
    }
}
