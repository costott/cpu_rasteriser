use crate::prelude::*;

pub struct Fragment<V> {
    pub position: Vec2,
    pub depth: f32,
    pub varyings: V,
}
impl<V> Fragment<V> {
    pub fn new(position: Vec2, depth: f32, varyings: V) -> Self {
        Self {
            position,
            depth,
            varyings,
        }
    }
}
