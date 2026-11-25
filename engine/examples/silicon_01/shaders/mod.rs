//! The shaders the film shares, and the view and frame state every scene shader receives.
//!
//! Scene passes render HDR colour with the view distance of opaque surfaces in alpha (see
//! [`scene_colour`]), so screen-space passes can find depth, edges and normals without a
//! separate geometry pass.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;

use crate::director::Phase;
use crate::simd::{F, length, s, splat3, to_colour};

pub mod chip;
pub mod composite;
pub mod screen;
pub mod studio;

/// The distance written to alpha where nothing was drawn: further than any scene reaches.
pub const SKY_DISTANCE: f32 = 1000.0;

/// What every shader in a scene pass needs: the camera it renders from and the frame's state.
#[derive(Clone)]
pub struct View {
    pub view_projection: Mat4,
    pub eye: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub tan_half_fov: f32,
    pub aspect: f32,
    /// The angle one pixel of the target covers, for fading detail smaller than a pixel.
    pub pixel_angle: f32,
    /// The angle to size thin line art by: a pixel's, but never less than that of a pixel in a
    /// 720-line frame, so lines keep their share of the picture at higher resolutions.
    pub line_angle: f32,
    pub frame: Arc<Frame>,
}

pub type ViewRef = Arc<View>;

/// The state of the film in one frame, as the director sets it.
pub struct Frame {
    /// Seconds since the start of the loop.
    pub time: f32,
    pub phase: Phase,
    /// Seconds since the current phase started.
    pub phase_time: f32,
    /// The chip's model matrix: its pose, shake included.
    pub chip: Mat4,
    /// The inverse of [`Frame::chip`], for shaders that work in the chip's own space.
    pub chip_inverse: Mat4,
    /// How busy the chip is, 0 (dark) to 1: drives its light streams and glows.
    pub activity: f32,
    /// Where the specular sweep across the chip is, from -1 (not yet arrived) through 0 (the
    /// chip's centre) to 1 (gone).
    pub sweep: f32,
}

/// Transforms eight points by `m`, returning homogeneous coordinates.
#[inline(always)]
pub fn transform_point(m: &Mat4, p: Vec3Simd) -> [F; 4] {
    let row = |r: [f32; 4]| s(r[0]) * p.x + s(r[1]) * p.y + s(r[2]) * p.z + s(r[3]);
    [
        row(m.data[0]),
        row(m.data[1]),
        row(m.data[2]),
        row(m.data[3]),
    ]
}

/// The view ray through normalised device coordinates (`ndc_x`, `ndc_y`).
#[inline(always)]
pub fn view_ray(view: &View, ndc_x: F, ndc_y: F) -> Vec3Simd {
    let x = ndc_x * s(view.tan_half_fov * view.aspect);
    let y = ndc_y * s(view.tan_half_fov);
    (splat3(view.forward) + splat3(view.right) * x + splat3(view.up) * y).normalise()
}

/// A scene pass's output for an opaque surface at `world`: its colour, with its distance from
/// the eye in alpha.
#[inline(always)]
pub fn scene_colour(view: &View, colour: Vec3Simd, world: Vec3Simd) -> ColourSimd {
    to_colour(colour, length(world - splat3(view.eye)))
}
