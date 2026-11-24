use cpu_rasteriser::prelude::*;

use crate::shaders::{PassThrough, VertexColour};
use crate::workloads::{CLEAR, FAR, Workload};

/// Which pipeline a frame is drawn with; it decides the benchmark's id.
#[derive(Clone, Copy)]
pub(crate) enum Implementation {
    Scalar,
    Simd,
}
impl Implementation {
    pub(crate) const ALL: [Self; 2] = [Self::Scalar, Self::Simd];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Simd => "simd",
        }
    }
}

/// A workload with everything needed to render it repeatedly: the renderer, the target and the
/// pipeline of one implementation.
pub(crate) struct Frame<'a> {
    workload: &'a Workload,
    renderer: Renderer,
    target: RenderTarget,
    pipeline: AnyPipeline,
}
impl<'a> Frame<'a> {
    pub(crate) fn new(workload: &'a Workload, implementation: Implementation) -> Self {
        let target = RenderTarget::new(workload.extent);

        Self {
            workload,
            renderer: Renderer::new().expect("the available parallelism is known"),
            target: if workload.variant.depth() {
                target.with_depth()
            } else {
                target
            },
            pipeline: AnyPipeline::new(workload, implementation),
        }
    }

    pub(crate) fn target(&self) -> &RenderTarget {
        &self.target
    }

    /// Renders every pass of the workload into the target. The first pass clears it and the
    /// others load what the one before left, so each call starts from the same contents.
    pub(crate) fn render(&mut self) {
        let depth = self.workload.variant.depth();

        for (index, layers) in self.workload.passes.iter().enumerate() {
            let first = index == 0;
            let mut pass = self.renderer.begin_render_pass(
                &mut self.target,
                RenderPassDescriptor {
                    viewport: Viewport::full(&self.workload.extent),
                    colour_load_op: load_op(first, CLEAR),
                    depth_load_op: depth.then(|| load_op(first, FAR)),
                },
            );

            for layer in layers {
                for indices in layer.indices.chunks(layer.indices_per_draw) {
                    let draw_call =
                        DrawCall::new(&layer.vertices, indices, PrimitiveMode::TRIANGLES, ());
                    match &self.pipeline {
                        AnyPipeline::Scalar(pipeline) => pass.draw(pipeline, draw_call, ()),
                        AnyPipeline::Simd(pipeline) => pass.draw_simd(pipeline, draw_call, ()),
                    }
                }
            }

            pass.finish();
        }
    }
}

enum AnyPipeline {
    Scalar(Pipeline<PassThrough, VertexColour>),
    Simd(SimdPipeline<PassThrough, VertexColour>),
}
impl AnyPipeline {
    fn new(workload: &Workload, implementation: Implementation) -> Self {
        let depth_state = if workload.variant.depth() {
            DepthState::DEFAULT
        } else {
            DepthState::DISABLED
        };

        match implementation {
            Implementation::Scalar => {
                let pipeline =
                    Pipeline::new(PassThrough, VertexColour).with_depth_state(depth_state);
                Self::Scalar(if workload.variant.blended() {
                    pipeline.with_blend_state(BlendState::ALPHA_BLEND)
                } else {
                    pipeline
                })
            }
            Implementation::Simd => {
                let pipeline =
                    SimdPipeline::new(PassThrough, VertexColour).with_depth_state(depth_state);
                Self::Simd(if workload.variant.blended() {
                    pipeline.with_blend_state(BlendState::ALPHA_BLEND)
                } else {
                    pipeline
                })
            }
        }
    }
}

fn load_op<T>(first: bool, clear: T) -> LoadOp<T> {
    if first {
        LoadOp::Clear(clear)
    } else {
        LoadOp::Load
    }
}
