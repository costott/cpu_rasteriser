use std::hint::black_box;

use cpu_rasteriser::internals::ClipVertex;
use cpu_rasteriser::prelude::*;
use criterion::{Criterion, Throughput, criterion_group};

use crate::fixtures::{PassThrough, Varyings, unit, varyings};

const EXTENT: Extent = Extent {
    width: 1280,
    height: 720,
};

/// Each of the four ways to read a frame back has a case, on a 1280×720 target with varied
/// content, so a change to any of them shows. `pixels` and `pixels_u32` allocate the result on
/// every call, and `write_pixels` and `write_pixels_u32` fill a buffer made once. The two `u32`
/// forms pack with SIMD in the one implementation that exists, and nothing else in the crate
/// reads a target back as `u32`s by another route, so they are not paired as scalar and SIMD:
/// the per-pixel scalar packing is `Colour::to_u32`, in the `colour` group. `RenderTarget::resize`
/// is not benchmarked because it doesn't run every frame.
fn benchmark_output(c: &mut Criterion) {
    let mut group = c.benchmark_group("output");
    let pixel_count = EXTENT.width * EXTENT.height;
    group.throughput(Throughput::Elements(pixel_count as u64));

    let target = rendered_target();

    group.bench_function("pixels", |bench| {
        bench.iter(|| black_box(black_box(&target).pixels()))
    });

    let mut colours = vec![Colour::BLACK; pixel_count];
    group.bench_function("write_pixels", |bench| {
        bench.iter(|| {
            black_box(&target).write_pixels(&mut colours);
            black_box(&colours);
        })
    });

    group.bench_function("pixels_u32", |bench| {
        bench.iter(|| black_box(black_box(&target).pixels_u32()))
    });

    let mut packed = vec![0; pixel_count];
    group.bench_function("write_pixels_u32", |bench| {
        bench.iter(|| {
            black_box(&target).write_pixels_u32(&mut packed);
            black_box(&packed);
        })
    });

    group.finish();
}

/// Draws overlapping triangles with varying vertex colours over a dark grey background, so the
/// pixels read back are not all one value.
fn rendered_target() -> RenderTarget {
    const TRIANGLES: usize = 16;

    let vertices: Vec<ClipVertex<Varyings>> = (0..3 * TRIANGLES)
        .map(|i| {
            let position = Vec4::new(
                unit(5 * i) * 2.4 - 1.2,
                unit(7 * i + 1) * 2.4 - 1.2,
                unit(11 * i + 2) * 2.0 - 1.0,
                1.0,
            );
            ClipVertex::new(position, varyings(i))
        })
        .collect();
    let indices: Vec<u32> = (0..3 * TRIANGLES as u32).collect();

    let mut renderer = Renderer::new().expect("the available parallelism is known");
    let mut target = RenderTarget::new(EXTENT);
    let pipeline = Pipeline::new(PassThrough, VertexColour).with_culling_mode(CullingMode::None);

    let mut pass = renderer.begin_render_pass(
        &mut target,
        RenderPassDescriptor {
            viewport: Viewport::full(&EXTENT),
            colour_load_op: LoadOp::Clear(Colour::new(0.1, 0.1, 0.1, 1.0)),
            depth_load_op: None,
        },
    );
    pass.draw(
        &pipeline,
        DrawCall::new(&vertices, &indices, PrimitiveMode::TRIANGLES, ()),
        (),
    );
    pass.finish();

    target
}

struct VertexColour;
impl FragmentShader<Varyings> for VertexColour {
    type Uniforms = ();

    fn shade(&self, varyings: Varyings, _: &()) -> Colour {
        Colour::from(varyings.colour)
    }
}

criterion_group!(benches, benchmark_output);
