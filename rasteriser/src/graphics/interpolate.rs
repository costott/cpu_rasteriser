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

impl Interpolate for u8 {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        let self_f32 = *self as f32;
        let other_f32 = *other as f32;
        let result = self_f32 + (other_f32 - self_f32) * t;
        result.clamp(0.0, 255.0) as u8
    }
    fn difference(&self, other: &Self) -> Self {
        (self - other).clamp(0, 255)
    }
    fn scale(&self, factor: f32) -> Self {
        let self_f32 = *self as f32;
        let result = self_f32 * factor;
        result.clamp(0.0, 255.0) as u8
    }
    fn add_scaled(&self, other: &Self, factor: f32) -> Self {
        let self_f32 = *self as f32;
        let other_f32 = *other as f32;
        let result = self_f32 + other_f32 * factor;
        result.clamp(0.0, 255.0) as u8
    }
}
