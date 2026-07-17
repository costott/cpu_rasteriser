use crate::prelude::*;

pub struct Triangle3D {
    pub a: Vertex3D,
    pub b: Vertex3D,
    pub c: Vertex3D,
}
impl Triangle3D {
    pub fn new(a: Vertex3D, b: Vertex3D, c: Vertex3D) -> Self {
        Self { a, b, c }
    }
}
