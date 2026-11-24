use cpu_rasteriser::prelude::*;

use crate::frame::Implementation;
use crate::workloads::{CLEAR, FAR, Workload};

/// The largest difference between a colour channel of the mean colour and its expected value.
/// Provisional: above the error of the approximate reciprocal (`rcpps`, relative error at most
/// 1.5 × 2⁻¹²) that the SIMD path's perspective correction uses, which scales every colour channel
/// of at most 1, and far below the differences between the colours the layers contribute. Issue
/// #35 tunes it.
const TOLERANCE: f32 = 1e-3;

/// Panics unless `target`, rendered from `workload` by `implementation`, covers the pixels and has
/// the mean colour that `workload` is built to produce.
pub(crate) fn check(
    case: &str,
    implementation: Implementation,
    workload: &Workload,
    target: &RenderTarget,
) {
    let name = format!("frame/{case}/{}", implementation.name());
    let pixels = target.pixels();

    let covered = pixels.iter().filter(|&&pixel| pixel != CLEAR).count();
    let expected_covered = workload.pixel_count();
    assert!(
        covered == expected_covered,
        "{name}: {covered} pixels covered, expected {expected_covered}"
    );

    let mean = mean_colour(&pixels);
    let expected_mean = expected_mean_colour(workload);
    let error = mean - expected_mean;
    assert!(
        [error.r, error.g, error.b]
            .iter()
            .all(|channel| channel.abs() <= TOLERANCE),
        "{name}: mean colour {mean:?}, expected {expected_mean:?} within {TOLERANCE}"
    );
}

/// Sums in `f64`, because an `f32` running sum over a frame's hundreds of thousands to millions of
/// pixels can lose more precision than `TOLERANCE` allows.
fn mean_colour(pixels: &[Colour]) -> Colour {
    let mut sum = [0.0; 4];
    for pixel in pixels {
        for (total, channel) in sum.iter_mut().zip([pixel.r, pixel.g, pixel.b, pixel.a]) {
            *total += f64::from(channel);
        }
    }
    let [r, g, b, a] = sum.map(|total| (total / pixels.len() as f64) as f32);
    Colour::new(r, g, b, a)
}

/// Returns the mean colour of the frame the workload draws, derived from how it is built. Only
/// the colour channels are derived, because only they are compared: the scalar and SIMD blends
/// compute alpha differently ([#18](https://github.com/costott/cpu_rasteriser/issues/18)).
///
/// Every layer tiles the whole target and has one depth, so a layer is drawn at every pixel or at
/// none. The first layer's depth is nearer than the cleared depth, so every pixel is drawn at
/// least once. The layers run in drawing order across the passes, the later ones loading what the
/// earlier ones left. With depth testing a layer is drawn if it is nearer than every layer drawn
/// before it (the depth test is strictly less, and a drawn layer writes its depth); without, every
/// layer is drawn. An opaque layer replaces the colour, and a blended one is blended onto it with
/// `src * src.a + dst * (1 - src.a)`.
///
/// All cells of a layer have the same alpha, so blending is affine in a cell's colour, and the
/// mean of the blended frame is the blend of the layers' mean colours. A layer's mean is over its
/// cells because they have equal area.
fn expected_mean_colour(workload: &Workload) -> Colour {
    let depth = workload.variant.depth();
    let blended = workload.variant.blended();

    let mut colour = CLEAR;
    let mut nearest = FAR;
    for layer in workload.layers() {
        if depth {
            if layer.depth >= nearest {
                continue;
            }
            nearest = layer.depth;
        }

        let mean = mean_colour(&layer.colours);

        colour = if blended {
            mean * mean.a + colour * (1.0 - mean.a)
        } else {
            mean
        };
    }
    colour
}
