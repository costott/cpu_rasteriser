use std::hint::black_box;

use cpu_rasteriser::prelude::*;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use crate::check::check;
use crate::frame::{Frame, Implementation};
use crate::workloads::{EXTENT, Variant, Workload};

mod check;
mod frame;
mod shaders;
mod workloads;

fn benchmark_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame");

    for case in cases() {
        let workload = &case.workload;
        group.throughput(Throughput::Elements(workload.pixel_count() as u64));

        for implementation in Implementation::ALL {
            let mut frame = None;

            group.bench_function(
                BenchmarkId::new(case.name, implementation.name()),
                |bench| {
                    // Criterion runs this closure once per sample, so the set-up happens once.
                    let frame = frame.get_or_insert_with(|| {
                        let mut frame = Frame::new(workload, implementation);
                        frame.render();
                        check(case.name, implementation, workload, frame.target());
                        frame
                    });

                    bench.iter(|| {
                        black_box(&mut *frame).render();
                        black_box(frame.target());
                    })
                },
            );
        }
    }

    group.finish();
}

struct Case {
    name: &'static str,
    workload: Workload,
}

/// Each case isolates a cost that shows only across a whole frame. `small_triangles` stresses
/// per-triangle costs (geometry processing, binning, tile set-up) and `large_triangles`
/// per-pixel costs (fragment shading, depth testing, blending), each with and without depth and
/// opaque and blended. `large_triangles_depth_opaque` at the two other resolutions shows how
/// cost scales with pixel count. `multipass_depth_blended` splits the layers of
/// `large_triangles_depth_blended` over two passes, which times the load path of tile set-up.
/// `many_draws_opaque` draws `small_triangles_opaque`'s frame one cell per draw call, so the
/// difference between the two is the per-draw cost.
///
/// Building the workloads is cheap; rendering and checking a frame waits for a benchmark the
/// filter selects.
fn cases() -> Vec<Case> {
    let large = |name, extent, variant| Case {
        name,
        workload: Workload::large_triangles(extent, variant),
    };
    let small = |name, variant| Case {
        name,
        workload: Workload::small_triangles(variant),
    };

    vec![
        small("small_triangles_opaque", Variant::Opaque),
        small("small_triangles_blended", Variant::Blended),
        small("small_triangles_depth_opaque", Variant::DepthOpaque),
        small("small_triangles_depth_blended", Variant::DepthBlended),
        large("large_triangles_opaque", EXTENT, Variant::Opaque),
        large("large_triangles_blended", EXTENT, Variant::Blended),
        large("large_triangles_depth_opaque", EXTENT, Variant::DepthOpaque),
        large(
            "large_triangles_depth_blended",
            EXTENT,
            Variant::DepthBlended,
        ),
        large(
            "large_triangles_depth_opaque_640x360",
            Extent::new(640, 360),
            Variant::DepthOpaque,
        ),
        large(
            "large_triangles_depth_opaque_1920x1080",
            Extent::new(1920, 1080),
            Variant::DepthOpaque,
        ),
        Case {
            name: "many_draws_opaque",
            workload: Workload::many_draws(),
        },
        Case {
            name: "multipass_depth_blended",
            workload: Workload::multipass(),
        },
    ]
}

criterion_group!(benches, benchmark_frame);
criterion_main!(benches);
