pub trait Interpolate {
    fn interpolate(&self, other: &Self, t: f32) -> Self;
}
