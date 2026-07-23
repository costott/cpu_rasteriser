use crate::prelude::*;

pub struct Camera {
    pub eye: Vec3,
    pub lookat: Vec3,
    pub up: Vec3,
    pub projection: Projection,
}
impl Camera {
    pub fn new(eye: Vec3, lookat: Vec3, up: Vec3, projection: Projection) -> Self {
        Self {
            eye,
            lookat,
            up,
            projection,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let n = (self.eye - self.lookat).normalise();
        let u = self.up.cross(&n).normalise();
        let v = n.cross(&u);

        Mat4::new([
            [u.x, u.y, u.z, -u.dot(&self.eye)],
            [v.x, v.y, v.z, -v.dot(&self.eye)],
            [n.x, n.y, n.z, -n.dot(&self.eye)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn projection_matrix(&self) -> Mat4 {
        self.projection.matrix()
    }
}

pub enum Projection {
    Perspective(PerspectiveProjection),
    Orthographic(OrthographicProjection),
}
impl Projection {
    pub fn matrix(&self) -> Mat4 {
        match self {
            Projection::Perspective(perspective) => perspective.matrix(),
            Projection::Orthographic(orthographic) => orthographic.matrix(),
        }
    }
}

pub struct OrthographicProjection {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
    pub near: f32,
    pub far: f32,
}
impl OrthographicProjection {
    pub fn new(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Self {
            left,
            right,
            bottom,
            top,
            near,
            far,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::new([
            [
                2.0 / (self.right - self.left),
                0.0,
                0.0,
                -(self.right + self.left) / (self.right - self.left),
            ],
            [
                0.0,
                2.0 / (self.top - self.bottom),
                0.0,
                -(self.top + self.bottom) / (self.top - self.bottom),
            ],
            [
                0.0,
                0.0,
                -2.0 / (self.far - self.near),
                -(self.far + self.near) / (self.far - self.near),
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}

pub struct PerspectiveProjection {
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}
impl PerspectiveProjection {
    pub fn new(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        Self {
            fov,
            aspect_ratio,
            near,
            far,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        let top = self.near * (self.fov / 2.0).tan();
        let bottom = -top;

        let right = top * self.aspect_ratio;
        let left = -right;

        Mat4::new([
            [
                (2.0 * self.near) / (right - left),
                0.0,
                (right + left) / (right - left),
                0.0,
            ],
            [
                0.0,
                (2.0 * self.near) / (top - bottom),
                (top + bottom) / (top - bottom),
                0.0,
            ],
            [
                0.0,
                0.0,
                -(self.far + self.near) / (self.far - self.near),
                -(2.0 * self.far * self.near) / (self.far - self.near),
            ],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }
}
