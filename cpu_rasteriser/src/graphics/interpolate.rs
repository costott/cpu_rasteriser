use wide::f32x8;

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

/// Provides SIMD interpolation support for fragment varyings.
///
/// [`SimdPipeline`] uses this trait to interpolate varyings across multiple
/// fragments simultaneously during rasterisation.
///
/// Implementations should produce the same values as the corresponding scalar
/// [`Interpolate`] operations, represented in SIMD form.
pub trait SimdInterpolate: Interpolate {
    type Simd: Copy;

    /// Construct the eight interpolated lanes:
    ///
    /// value[i] = self + step * i
    fn simd_step(value: &Self, step: &Self, lanes: f32x8) -> Self::Simd;

    /// Add a scaled step to the SIMD value:
    ///
    /// value[i] = value[i] + step * scale
    fn simd_add_scaled(value: &Self::Simd, step: &Self, scale: f32x8) -> Self::Simd;

    /// Apply perspective correction to every component.
    fn simd_perspective(value: Self::Simd, perspective: f32x8) -> Self::Simd;

    // /// Extract one SIMD lane back into the scalar varying type.
    // fn simd_extract(value: &Self::Simd, lane: usize) -> Self;

    /// Extract all SIMD lanes back into an array of scalar varying types.
    fn simd_extract_all(value: &Self::Simd) -> [Self; 8];
}
