use std::f32::consts::TAU;
use std::hint::black_box;

use cpu_rasteriser::internals::{ClipVertex, GeometryProcessor, TriangleClip, clip_triangle};
use cpu_rasteriser::prelude::*;
use criterion::{Criterion, Throughput, criterion_group};

use crate::fixtures::{BATCH, PassThrough, Varyings, unit, varyings};

const INSIDE: Region = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5));
const NEAR_PLANE: Region = ((-0.5, 0.5), (-0.5, 0.5), (-1.0, -0.95));
const TOP_RIGHT_CORNER: Region = ((0.9, 1.0), (0.9, 1.0), (-0.5, 0.5));
const BEYOND_RIGHT_EDGE: Region = ((1.5, 2.0), (-0.5, 0.5), (-0.5, 0.5));

/// How far a triangle's corners lie from its centre in NDC depth, at most.
const DEPTH_RADIUS: f32 = 0.2;

/// `GeometryProcessor::process_triangle` has a case for each way a triangle can leave it: kept
/// whole with and without the culling test (`process_front_facing_*`), dropped by the culling test
/// or kept without it (`process_back_facing_*`, the same triangles with their winding reversed),
/// and split by clipping (`process_clipped_back_face_culling`, which crosses the near plane and
/// keeps the culling test, so it differs from `process_front_facing_back_face_culling` only by the
/// clipping). Each iteration clones its triangles because the function consumes them.
/// `clip_triangle` has a case for each clipping outcome: untouched (`clip_inside`), one plane
/// (`clip_one_plane`, the near plane, which every view that has geometry behind the camera clips),
/// two planes at a corner (`clip_two_planes`, the right and top ones), and rejected at the right
/// plane (`clip_rejected`). Both are scalar-only, as the scalar and SIMD pipelines share them.
fn benchmark_geometry(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometry");
    group.throughput(Throughput::Elements(BATCH as u64));

    let viewport = Viewport::new(0, 0, 1280, 720);
    let front_facing = triangles(INSIDE);
    let clipped = triangles(NEAR_PLANE);
    let cases = [
        (
            "process_front_facing_no_culling",
            front_facing.clone(),
            CullingMode::None,
        ),
        (
            "process_front_facing_back_face_culling",
            front_facing.clone(),
            CullingMode::BackFace,
        ),
        (
            "process_back_facing_back_face_culling",
            reversed(&front_facing),
            CullingMode::BackFace,
        ),
        (
            "process_back_facing_no_culling",
            reversed(&front_facing),
            CullingMode::None,
        ),
        (
            "process_clipped_back_face_culling",
            clipped.clone(),
            CullingMode::BackFace,
        ),
    ];
    for (name, triangles, culling_mode) in cases {
        group.bench_function(name, |bench| {
            bench.iter(|| {
                let viewport = black_box(&viewport);
                let culling_mode = black_box(culling_mode);
                let mut total = 0.0;
                for triangle in black_box(&triangles) {
                    let screen = GeometryProcessor::process_triangle(
                        Triangle3D::new(triangle.a.clone(), triangle.b.clone(), triangle.c.clone()),
                        &PassThrough,
                        &(),
                        viewport,
                        culling_mode,
                    );
                    total += screen.len() as f32;
                    for triangle in &screen {
                        total += triangle.a.position.x
                            + triangle.b.position.y
                            + triangle.c.inv_w
                            + triangle.c.varyings.sum();
                    }
                }
                black_box(total)
            })
        });
    }

    let cases = [
        ("clip_inside", front_facing),
        ("clip_one_plane", clipped),
        ("clip_two_planes", triangles(TOP_RIGHT_CORNER)),
        ("clip_rejected", triangles(BEYOND_RIGHT_EDGE)),
    ];
    for (name, triangles) in cases {
        group.bench_function(name, |bench| {
            bench.iter(|| {
                let mut total = 0.0;
                for triangle in black_box(&triangles) {
                    let pieces = clip_triangle(triangle.clone());
                    total += pieces.len() as f32;
                    for piece in &pieces {
                        total += piece.a.position.x
                            + piece.b.position.y
                            + piece.c.position.w
                            + piece.c.varyings.sum();
                    }
                }
                black_box(total)
            })
        });
    }

    group.finish();
}

/// Returns `BATCH` counter-clockwise (front-facing) triangles in clip space with a different
/// shape, size, orientation, `w`, depth and varyings each, centred in `region`.
///
/// Each triangle's corners lie on a circle of radius `[0.25, 0.4)` around its centre, and their
/// NDC depths are `DEPTH_RADIUS` times the cosine of a random phase plus the corner's angle, around
/// the centre's depth. Of three values of that form spaced 120° apart, one is always at least half
/// the radius above the centre and one at least half below. So a triangle centred in `[0.9, 1.0)`
/// on an axis always crosses that plane and keeps a corner inside it, and one centred in
/// `[-0.5, 0.5)` stays within NDC `[-0.9, 0.9]` on the x and y axes, clear of every plane. Depth
/// works alike: with a centre in `[-1.0, -0.95)` and `DEPTH_RADIUS` 0.2, a triangle always has a
/// corner on each side of the near plane (NDC depth -1), and with a centre in `[-0.5, 0.5)` it
/// stays within `[-0.7, 0.7]`, clear of the near and far planes. `w` is in `[0.5, 1.5)`, so never
/// zero, and clip-space coordinates are the NDC ones times `w`, which keeps each plane test's sign.
fn triangles(region: Region) -> Vec<TriangleClip<Varyings>> {
    let ((min_x, max_x), (min_y, max_y), (min_z, max_z)) = region;
    (0..BATCH)
        .map(|i| {
            let centre_x = min_x + (max_x - min_x) * unit(2 * i);
            let centre_y = min_y + (max_y - min_y) * unit(3 * i + 1);
            let centre_z = min_z + (max_z - min_z) * unit(17 * i + 5);
            let radius = 0.25 + 0.15 * unit(5 * i + 2);
            let phase = TAU * unit(7 * i + 3);
            let depth_phase = TAU * unit(19 * i + 6);
            let corner = |c: usize| {
                let corner_angle = TAU / 3.0 * c as f32;
                let angle = phase + corner_angle;
                let w = 0.5 + unit(11 * i + c);
                let z = centre_z + DEPTH_RADIUS * (depth_phase + corner_angle).cos();
                ClipVertex::new(
                    Vec4::new(
                        (centre_x + radius * angle.cos()) * w,
                        (centre_y + radius * angle.sin()) * w,
                        z * w,
                        w,
                    ),
                    varyings(3 * i + c),
                )
            };
            TriangleClip::new(corner(0), corner(1), corner(2))
        })
        .collect()
}

/// Where the centres of a batch's triangles lie in NDC, as `x`, `y` and depth ranges.
type Region = ((f32, f32), (f32, f32), (f32, f32));

fn reversed(triangles: &[TriangleClip<Varyings>]) -> Vec<TriangleClip<Varyings>> {
    triangles
        .iter()
        .map(|triangle| {
            TriangleClip::new(triangle.a.clone(), triangle.c.clone(), triangle.b.clone())
        })
        .collect()
}

criterion_group!(benches, benchmark_geometry);
