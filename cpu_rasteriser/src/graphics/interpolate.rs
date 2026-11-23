use wide::f32x8;

/// A value the rasteriser can interpolate across a triangle: the bound on a vertex shader's
/// [`Varyings`](crate::VertexShader::Varyings).
///
/// The vertex shader outputs one value per triangle corner, and the rasteriser blends them into the
/// value each fragment receives, with perspective correction: the blend follows positions on the
/// triangle's surface in the scene, not on the screen. The rasteriser works only through the four
/// operations of this trait, which treat the value as a vector of `f32` components.
///
/// Implementations must therefore be component-wise linear and agree with each other:
/// `a.interpolate(&b, t)` must equal `a.add_scaled(&b.difference(&a), t)` up to rounding. To keep a
/// component constant across a triangle, give it the same value at every corner.
///
/// The trait is implemented for `f32`, [`Vec2`](crate::Vec2), [`Vec3`](crate::Vec3), and
/// [`Vec4`](crate::Vec4). [`Colour`](crate::Colour) doesn't implement it: pass colours as a `Vec4`
/// varying and convert it with `Colour::from` in the fragment shader.
///
/// # Deriving
///
/// `#[derive(Interpolate)]` implements every operation field by field. See
/// [the derive's documentation](derive@crate::Interpolate) for its requirements.
///
/// Implement the trait by hand for a type the derive doesn't support, such as a tuple struct, a
/// generic struct, or a struct with an array field.
///
/// # Examples
///
/// Deriving the trait for a vertex shader's varyings, and finding the value the rasteriser gives a
/// fragment halfway between two corners:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// #[derive(Debug, PartialEq, Interpolate)]
/// struct Varyings {
///     colour: Vec4,
///     uv: Vec2,
/// }
///
/// let a = Varyings { colour: Vec4::new(1.0, 0.0, 0.0, 1.0), uv: Vec2::new(0.0, 0.0) };
/// let b = Varyings { colour: Vec4::new(0.0, 0.0, 1.0, 1.0), uv: Vec2::new(1.0, 0.0) };
///
/// assert_eq!(
///     a.interpolate(&b, 0.5),
///     Varyings { colour: Vec4::new(0.5, 0.0, 0.5, 1.0), uv: Vec2::new(0.5, 0.0) },
/// );
/// ```
///
/// Implementing it by hand for blend weights stored in an array, which the derive doesn't support:
///
/// ```
/// use cpu_rasteriser::prelude::*;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct Weights([f32; 3]);
///
/// impl Interpolate for Weights {
///     fn interpolate(&self, other: &Self, t: f32) -> Self {
///         Weights(std::array::from_fn(|i| self.0[i].interpolate(&other.0[i], t)))
///     }
///
///     fn difference(&self, other: &Self) -> Self {
///         Weights(std::array::from_fn(|i| self.0[i] - other.0[i]))
///     }
///
///     fn scale(&self, factor: f32) -> Self {
///         Weights(self.0.map(|w| w * factor))
///     }
///
///     fn add_scaled(&self, other: &Self, factor: f32) -> Self {
///         Weights(std::array::from_fn(|i| self.0[i] + other.0[i] * factor))
///     }
/// }
///
/// let a = Weights([1.0, 0.0, 0.0]);
/// let b = Weights([0.0, 1.0, 0.0]);
///
/// assert_eq!(a.interpolate(&b, 0.25), Weights([0.75, 0.25, 0.0]));
/// assert_eq!(a.interpolate(&b, 0.25), a.add_scaled(&b.difference(&a), 0.25));
/// ```
pub trait Interpolate: Clone {
    /// Returns the value a fraction `t` of the way from `self` to `other`:
    /// `self + (other - self) * t`.
    ///
    /// `t` is 0 at `self` and 1 at `other`.
    fn interpolate(&self, other: &Self, t: f32) -> Self;

    /// Returns `self - other`, component by component.
    fn difference(&self, other: &Self) -> Self;

    /// Returns `self * factor`, component by component.
    fn scale(&self, factor: f32) -> Self;

    /// Returns `self + other * factor`, component by component.
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

/// The eight-lane form of [`Interpolate`]: the bound on the varyings of a draw through a
/// [`SimdPipeline`](crate::SimdPipeline).
///
/// The SIMD rasteriser shades eight horizontally adjacent pixels at once. The associated
/// [`Simd`](Self::Simd) type holds the varyings of all eight and is passed to
/// [`FragmentShaderSimd::shade_simd`](crate::FragmentShaderSimd::shade_simd), which defines how
/// lanes map to pixels. `Simd` is laid out as a structure of arrays: one
/// [`f32x8`](crate::wide::f32x8) per scalar component, holding that component for all eight
/// pixels. Every operation must give, in each lane, the value the scalar [`Interpolate`]
/// operations would give for that pixel, up to rounding.
///
/// Lanes that are masked off in [`shade_simd`](crate::FragmentShaderSimd::shade_simd) still hold
/// values, extrapolated beyond the triangle.
///
/// # Deriving
///
/// `#[derive(SimdInterpolate)]` generates a struct `<Name>Simd` and the implementation. See
/// [the derive's documentation](derive@crate::SimdInterpolate) for its requirements and the layout
/// of `<Name>Simd`.
///
/// Implement the trait by hand for a type the derive doesn't support, such as a tuple struct, a
/// field of another type, or a `Simd` type that shaders in other modules can read.
///
/// # Examples
///
/// Deriving the trait for varyings that a SIMD fragment shader will read:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use cpu_rasteriser::wide::f32x8;
///
/// #[derive(Debug, PartialEq, Interpolate, SimdInterpolate)]
/// struct Varyings {
///     brightness: f32,
///     uv: Vec2,
/// }
///
/// // Eight pixels starting at `start`, each `step` further along the span.
/// let start = Varyings { brightness: 0.0, uv: Vec2::new(0.0, 0.5) };
/// let step = Varyings { brightness: 0.125, uv: Vec2::new(0.0, 0.0) };
/// let lanes = f32x8::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
///
/// let group: VaryingsSimd = Varyings::simd_step(&start, &step, lanes);
///
/// // A SIMD shader in this module reads one `f32x8` per component.
/// assert_eq!(group.brightness.to_array()[4], 0.5);
/// assert_eq!(group.uv[1], f32x8::splat(0.5));
///
/// assert_eq!(
///     Varyings::simd_extract_all(&group)[4],
///     Varyings { brightness: 0.5, uv: Vec2::new(0.0, 0.5) },
/// );
/// ```
///
/// Implementing it by hand for a tuple struct, which the derive doesn't support:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use cpu_rasteriser::wide::f32x8;
///
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// struct Fog(f32);
///
/// impl Interpolate for Fog {
///     fn interpolate(&self, other: &Self, t: f32) -> Self {
///         Fog(self.0.interpolate(&other.0, t))
///     }
///
///     fn difference(&self, other: &Self) -> Self {
///         Fog(self.0 - other.0)
///     }
///
///     fn scale(&self, factor: f32) -> Self {
///         Fog(self.0 * factor)
///     }
///
///     fn add_scaled(&self, other: &Self, factor: f32) -> Self {
///         Fog(self.0 + other.0 * factor)
///     }
/// }
///
/// impl SimdInterpolate for Fog {
///     type Simd = f32x8;
///
///     fn simd_step(value: &Self, step: &Self, lanes: f32x8) -> f32x8 {
///         f32x8::splat(value.0) + lanes * f32x8::splat(step.0)
///     }
///
///     fn simd_add_scaled(value: &f32x8, step: &Self, scale: f32x8) -> f32x8 {
///         *value + f32x8::splat(step.0) * scale
///     }
///
///     fn simd_perspective(value: f32x8, perspective: f32x8) -> f32x8 {
///         value * perspective
///     }
///
///     fn simd_extract_all(value: &f32x8) -> [Self; 8] {
///         value.to_array().map(Fog)
///     }
/// }
///
/// let lanes = f32x8::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
/// let group = Fog::simd_step(&Fog(1.0), &Fog(-0.125), lanes);
/// let next = Fog::simd_add_scaled(&group, &Fog(-0.125), f32x8::splat(8.0));
///
/// assert_eq!(Fog::simd_extract_all(&group)[2], Fog(0.75));
/// assert_eq!(Fog::simd_extract_all(&next)[0], Fog(0.0));
/// ```
pub trait SimdInterpolate: Interpolate {
    /// The values of eight fragments, with one [`f32x8`](crate::wide::f32x8) per scalar component.
    type Simd: Copy;

    /// Returns eight lanes stepping from `value`: lane `i` is `value + step * lanes[i]`.
    ///
    /// The rasteriser passes `lanes = [0, 1, …, 7]`, so lane `i` is `i` steps from `value`.
    fn simd_step(value: &Self, step: &Self, lanes: f32x8) -> Self::Simd;

    /// Returns `value` with `step` scaled by `scale` added to each lane: lane `i` is
    /// `value[i] + step * scale[i]`.
    ///
    /// The rasteriser passes `scale = 8` in every lane to move a group on to the next eight
    /// pixels.
    fn simd_add_scaled(value: &Self::Simd, step: &Self, scale: f32x8) -> Self::Simd;

    /// Returns `value` with every component of lane `i` multiplied by `perspective[i]`.
    ///
    /// It applies perspective correction: the rasteriser passes the reciprocal of each fragment's
    /// interpolated `1/w`, which undoes the division by `w` applied to the varyings before
    /// interpolation.
    fn simd_perspective(value: Self::Simd, perspective: f32x8) -> Self::Simd;

    // /// Extract one SIMD lane back into the scalar varying type.
    // fn simd_extract(value: &Self::Simd, lane: usize) -> Self;

    /// Returns the eight lanes of `value` as scalar values, in lane order.
    ///
    /// The rasteriser doesn't call this; it is for checking or debugging a SIMD shader against its
    /// scalar form.
    fn simd_extract_all(value: &Self::Simd) -> [Self; 8];
}

#[expect(
    dead_code,
    reason = "unused internally; kept rather than deleted for now"
)]
pub trait SimdInterpolateField {
    type Simd;

    fn simd_step(value: &Self, step: &Self, lanes: f32x8) -> Self::Simd;
    fn simd_add_scaled(value: &Self::Simd, step: &Self, scale: f32x8) -> Self::Simd;
    fn simd_perspective(value: Self::Simd, perspective: f32x8) -> Self::Simd;
    fn simd_extract(value: &Self::Simd, lane: usize) -> Self;
}

impl SimdInterpolateField for f32 {
    type Simd = f32x8;

    #[inline(always)]
    fn simd_step(value: &Self, step: &Self, lanes: f32x8) -> Self::Simd {
        f32x8::splat(*value) + lanes * f32x8::splat(*step)
    }

    #[inline(always)]
    fn simd_add_scaled(value: &Self::Simd, step: &Self, scale: f32x8) -> Self::Simd {
        *value + f32x8::splat(*step) * scale
    }

    #[inline(always)]
    fn simd_perspective(value: Self::Simd, perspective: f32x8) -> Self::Simd {
        value * perspective
    }

    #[inline(always)]
    fn simd_extract(value: &Self::Simd, lane: usize) -> Self {
        value.to_array()[lane]
    }
}
