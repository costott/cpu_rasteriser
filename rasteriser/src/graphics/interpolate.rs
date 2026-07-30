pub trait Interpolate: Clone {
    fn interpolate(&self, other: &Self, t: f32) -> Self;
    fn difference(&self, other: &Self) -> Self;
    fn scale(&self, factor: f32) -> Self;
    fn add_scaled(&self, other: &Self, factor: f32) -> Self;
}

impl Interpolate for f32 {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
    fn difference(&self, other: &Self) -> Self {
        *self - *other
    }
    fn scale(&self, factor: f32) -> Self {
        self * factor
    }
    fn add_scaled(&self, other: &Self, factor: f32) -> Self {
        self + other * factor
    }
}
