use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cpu_rasteriser::{depthbuffer::DepthBuffer, framebuffer::FrameBuffer, prelude::*};

// -----------------------------------------------------------------------------
// Benchmark shader
// -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct TestUniforms;

struct TestShader;

impl FragmentShader<TestVaryings> for TestShader {
    type Uniforms = TestUniforms;

    #[inline(always)]
    fn shade(&self, varyings: TestVaryings, _uniforms: &Self::Uniforms) -> Colour {
        let mut r = varyings.colour.x;
        let mut g = varyings.colour.y;
        let mut b = varyings.colour.z;

        for _ in 0..16 {
            r = (r * 1.37 + g * 0.21).max(0.0);
            g = (g * 0.91 + b * 0.34).max(0.0);
            b = (b * 1.13 + r * 0.17).max(0.0);
        }

        Colour::new(r, g, b, 1.0)
    }
}

impl SimdInterpolate for TestVaryings {
    type Simd = TestVaryingsSimd;

    fn simd_step(value: &Self, step: &Self, lanes: wide::f32x8) -> Self::Simd {
        Self::Simd {
            colour: [
                wide::f32x8::splat(value.colour.x) + lanes * wide::f32x8::splat(step.colour.x),
                wide::f32x8::splat(value.colour.y) + lanes * wide::f32x8::splat(step.colour.y),
                wide::f32x8::splat(value.colour.z) + lanes * wide::f32x8::splat(step.colour.z),
            ],
        }
    }

    fn simd_add_scaled(value: &Self::Simd, step: &Self, scale: wide::f32x8) -> Self::Simd {
        Self::Simd {
            colour: [
                value.colour[0] + wide::f32x8::splat(step.colour.x) * scale,
                value.colour[1] + wide::f32x8::splat(step.colour.y) * scale,
                value.colour[2] + wide::f32x8::splat(step.colour.z) * scale,
            ],
        }
    }

    fn simd_perspective(value: Self::Simd, perspective: wide::f32x8) -> Self::Simd {
        Self::Simd {
            colour: [
                value.colour[0] * perspective,
                value.colour[1] * perspective,
                value.colour[2] * perspective,
            ],
        }
    }

    fn simd_extract_all(value: &Self::Simd) -> [Self; 8] {
        let r = value.colour[0].to_array();
        let g = value.colour[1].to_array();
        let b = value.colour[2].to_array();

        [
            Self {
                colour: Vec3::new(r[0], g[0], b[0]),
            },
            Self {
                colour: Vec3::new(r[1], g[1], b[1]),
            },
            Self {
                colour: Vec3::new(r[2], g[2], b[2]),
            },
            Self {
                colour: Vec3::new(r[3], g[3], b[3]),
            },
            Self {
                colour: Vec3::new(r[4], g[4], b[4]),
            },
            Self {
                colour: Vec3::new(r[5], g[5], b[5]),
            },
            Self {
                colour: Vec3::new(r[6], g[6], b[6]),
            },
            Self {
                colour: Vec3::new(r[7], g[7], b[7]),
            },
        ]
    }
}

#[derive(Clone, Copy)]
struct TestVaryingsSimd {
    colour: [wide::f32x8; 3],
}

impl FragmentShaderSimd<TestVaryings> for TestShader {
    fn shade_simd(&self, varyings: TestVaryingsSimd, _uniforms: &TestUniforms) -> ColourSimd {
        let mut r = varyings.colour[0];
        let mut g = varyings.colour[1];
        let mut b = varyings.colour[2];

        let c137 = wide::f32x8::splat(1.37);
        let c021 = wide::f32x8::splat(0.21);
        let c091 = wide::f32x8::splat(0.91);
        let c034 = wide::f32x8::splat(0.34);
        let c113 = wide::f32x8::splat(1.13);
        let c017 = wide::f32x8::splat(0.17);
        let zero = wide::f32x8::splat(0.0);

        for _ in 0..16 {
            r = (r * c137 + g * c021).fast_max(zero);
            g = (g * c091 + b * c034).fast_max(zero);
            b = (b * c113 + r * c017).fast_max(zero);
        }

        ColourSimd {
            r,
            g,
            b,
            a: wide::f32x8::splat(1.0),
        }
    }
}

// -----------------------------------------------------------------------------
// Geometry
// -----------------------------------------------------------------------------

#[derive(Debug, Interpolate)]
struct TestVaryings {
    colour: Vec3,
}

fn vertex(x: f32, y: f32, colour: Vec3) -> RasterVertex<TestVaryings> {
    RasterVertex {
        position: Vec2::new(x, y),
        depth: 0.5,
        inv_w: 1.0,
        varyings: TestVaryings { colour },
    }
}

fn large_triangle() -> Triangle2D<TestVaryings> {
    Triangle2D::new(
        vertex(64.0, 32.0, Vec3::new(1.0, 0.0, 0.0)),
        vertex(576.0, 96.0, Vec3::new(0.0, 1.0, 0.0)),
        vertex(320.0, 328.0, Vec3::new(0.0, 0.0, 1.0)),
    )
}

fn full_bounds() -> Rect {
    Rect {
        min_x: 0,
        min_y: 0,
        max_x: 640,
        max_y: 360,
    }
}

// -----------------------------------------------------------------------------
// Mock Implementation of RasterCommand
// -----------------------------------------------------------------------------

struct PreSimdTriangleRasterCommand<V, FS>
where
    V: Interpolate,
    FS: FragmentShader<V>,
{
    triangle: Triangle2D<V>,
    uniforms: Arc<FS::Uniforms>,
    shader: Arc<FS>,
    blend_state: Option<BlendState>,
    depth_state: DepthState,
}
impl<V, FS> PreSimdTriangleRasterCommand<V, FS>
where
    V: Interpolate + Send + Sync + 'static,
    FS: FragmentShader<V> + Send + Sync + 'static,
{
    fn rasterise(
        &self,
        framebuffer: &mut FrameBuffer,
        mut depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        self.triangle.rasterise_segment(bounds, |mut fragment| {
            fragment.position.x -= bounds.min_x as f32;
            fragment.position.y -= bounds.min_y as f32;

            if self.depth_state.test_enabled {
                let depthbuffer = depthbuffer
                    .as_deref_mut()
                    .expect("depth testing enabled but render target has no depth buffer");

                if fragment.depth >= depthbuffer.get(fragment.position) {
                    return;
                }
            }

            let src = self.shader.shade(fragment.varyings, self.uniforms.as_ref());
            let dst = framebuffer
                .get_pixel(fragment.position)
                .unwrap_or(Colour::BLACK);

            let colour = match self.blend_state {
                Some(blend_state) => blend_state.apply(src, dst),
                None => src,
            };
            framebuffer.set_pixel(fragment.position, colour);

            if self.depth_state.write_enabled {
                let depthbuffer = depthbuffer
                    .as_deref_mut()
                    .expect("depth writing enabled but render target has no depth buffer");

                depthbuffer.set_depth(fragment.position, fragment.depth);
            }
        });
    }
}

struct TriangleRasterCommand<V, FS>
where
    V: SimdInterpolate,
    FS: FragmentShader<V>,
{
    triangle: Triangle2D<V>,

    uniforms: Arc<FS::Uniforms>,

    shader: Arc<FS>,

    blend_state: Option<BlendState>,
    depth_state: DepthState,
}
impl<V, FS> TriangleRasterCommand<V, FS>
where
    V: SimdInterpolate + Send + Sync + 'static,
    FS: FragmentShader<V> + Send + Sync + 'static,
{
    // consts are used to avoid runtime branching in the inner rasterisation loop
    fn rasterise_impl<const TEST_DEPTH: bool, const WRITE_DEPTH: bool>(
        &self,
        framebuffer: &mut FrameBuffer,
        mut depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        self.triangle
            .rasterise_segment_simd(bounds, |fragment_simd| {
                let pass = if TEST_DEPTH {
                    let depthbuffer = depthbuffer
                        .as_deref_mut()
                        .expect("depth testing enabled but no depth buffer");

                    let index = fragment_simd.y as usize * depthbuffer.width()
                        + fragment_simd.x_start as usize;

                    let stored = unsafe { depthbuffer.get8_unchecked(index) };

                    // Only covered lanes can pass.
                    fragment_simd.mask & fragment_simd.depth.simd_lt(stored)
                } else {
                    // No depth test: the rasteriser's coverage mask is the
                    // complete set of valid lanes.
                    fragment_simd.mask
                };

                if !pass.any() {
                    return;
                }

                let mask = pass.to_bitmask();

                let base = (fragment_simd.y - bounds.min_y) as usize * framebuffer.width()
                    + (fragment_simd.x_start - bounds.min_x) as usize;

                let varyings = V::simd_extract_all(&fragment_simd.varyings);

                for lane in 0..8 {
                    if mask & (1 << lane) == 0 {
                        continue;
                    }

                    let src = self
                        .shader
                        .shade(varyings[lane].clone(), self.uniforms.as_ref());

                    let colour = match self.blend_state {
                        Some(blend_state) => {
                            let dst = unsafe { framebuffer.get_pixel_index_unchecked(base + lane) };

                            blend_state.apply(src, dst)
                        }

                        None => src,
                    };

                    unsafe {
                        framebuffer.set_pixel_index_unchecked(base + lane, colour);
                    }
                }

                if WRITE_DEPTH {
                    let depthbuffer = depthbuffer
                        .as_deref_mut()
                        .expect("depth writing enabled but no depth buffer");

                    let index = fragment_simd.y as usize * depthbuffer.width()
                        + fragment_simd.x_start as usize;

                    unsafe {
                        depthbuffer.set8_unchecked_with_mask(index, fragment_simd.depth, pass);
                    }
                }
            });
    }
}
impl<V, FS> TriangleRasterCommand<V, FS>
where
    V: SimdInterpolate + Send + Sync + 'static,
    FS: FragmentShader<V> + Send + Sync + 'static,
{
    fn rasterise(
        &self,
        framebuffer: &mut FrameBuffer,
        depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        match (
            self.depth_state.test_enabled,
            self.depth_state.write_enabled,
        ) {
            (false, false) => {
                self.rasterise_impl::<false, false>(framebuffer, depthbuffer, bounds);
            }

            (false, true) => {
                self.rasterise_impl::<false, true>(framebuffer, depthbuffer, bounds);
            }

            (true, false) => {
                self.rasterise_impl::<true, false>(framebuffer, depthbuffer, bounds);
            }

            (true, true) => {
                self.rasterise_impl::<true, true>(framebuffer, depthbuffer, bounds);
            }
        }
    }
}

struct TriangleRasterCommandSimd<V, FS>
where
    V: SimdInterpolate,
    FS: FragmentShaderSimd<V>,
{
    triangle: Triangle2D<V>,
    uniforms: Arc<FS::Uniforms>,
    shader: Arc<FS>,
    blend_state: Option<BlendState>,
    depth_state: DepthState,
}

impl<V, FS> TriangleRasterCommandSimd<V, FS>
where
    V: SimdInterpolate + Send + Sync + 'static,
    FS: FragmentShaderSimd<V> + Send + Sync + 'static,
{
    fn rasterise_impl<const TEST_DEPTH: bool, const WRITE_DEPTH: bool>(
        &self,
        framebuffer: &mut FrameBuffer,
        mut depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        self.triangle
            .rasterise_segment_simd(bounds, |fragment_simd| {
                let pass = if TEST_DEPTH {
                    let depthbuffer = depthbuffer
                        .as_deref_mut()
                        .expect("depth testing enabled but no depth buffer");

                    let index = fragment_simd.y as usize * depthbuffer.width()
                        + fragment_simd.x_start as usize;

                    let stored = unsafe { depthbuffer.get8_unchecked(index) };

                    fragment_simd.mask & fragment_simd.depth.simd_lt(stored)
                } else {
                    fragment_simd.mask
                };

                if !pass.any() {
                    return;
                }

                let mask = pass.to_bitmask();

                let base = (fragment_simd.y - bounds.min_y) as usize * framebuffer.width()
                    + (fragment_simd.x_start - bounds.min_x) as usize;

                let src = self
                    .shader
                    .shade_simd(fragment_simd.varyings, self.uniforms.as_ref());

                let colour = if let Some(blend_state) = self.blend_state {
                    let dst = unsafe { framebuffer.get8_unchecked(base) };

                    blend_state.apply_simd(src, dst)
                } else {
                    src
                };

                let r = colour.r.to_array();
                let g = colour.g.to_array();
                let b = colour.b.to_array();
                let a = colour.a.to_array();

                for lane in 0..8 {
                    if mask & (1 << lane) == 0 {
                        continue;
                    }

                    let frag_colour = Colour::new(r[lane], g[lane], b[lane], a[lane]);

                    unsafe {
                        framebuffer.set_pixel_index_unchecked(base + lane, frag_colour);
                    }
                }

                if WRITE_DEPTH {
                    let depthbuffer = depthbuffer
                        .as_deref_mut()
                        .expect("depth writing enabled but no depth buffer");

                    let index = fragment_simd.y as usize * depthbuffer.width()
                        + fragment_simd.x_start as usize;

                    unsafe {
                        depthbuffer.set8_unchecked_with_mask(index, fragment_simd.depth, pass);
                    }
                }
            });
    }
}

impl<V, FS> TriangleRasterCommandSimd<V, FS>
where
    V: SimdInterpolate + Send + Sync + 'static,
    FS: FragmentShaderSimd<V> + Send + Sync + 'static,
{
    fn rasterise(
        &self,
        framebuffer: &mut FrameBuffer,
        depthbuffer: Option<&mut DepthBuffer>,
        bounds: Rect,
    ) {
        match (
            self.depth_state.test_enabled,
            self.depth_state.write_enabled,
        ) {
            (false, false) => {
                self.rasterise_impl::<false, false>(framebuffer, depthbuffer, bounds);
            }

            (false, true) => {
                self.rasterise_impl::<false, true>(framebuffer, depthbuffer, bounds);
            }

            (true, false) => {
                self.rasterise_impl::<true, false>(framebuffer, depthbuffer, bounds);
            }

            (true, true) => {
                self.rasterise_impl::<true, true>(framebuffer, depthbuffer, bounds);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Benchmark
// -----------------------------------------------------------------------------

fn bench_pre_simd_command(c: &mut Criterion) {
    let mut group = c.benchmark_group("raster_command/pre_simd");

    let triangle = large_triangle();
    let bounds = full_bounds();

    let shader = Arc::new(TestShader);
    let uniforms = Arc::new(TestUniforms);

    let cases = [
        (
            "no_depth_no_blend",
            DepthState {
                test_enabled: false,
                write_enabled: false,
            },
            None,
        ),
        (
            "depth_no_blend",
            DepthState {
                test_enabled: true,
                write_enabled: true,
            },
            None,
        ),
        (
            "depth_blend",
            DepthState {
                test_enabled: true,
                write_enabled: true,
            },
            Some(BlendState::ALPHA_BLEND),
        ),
    ];

    for (name, depth_state, blend_state) in cases {
        let command = PreSimdTriangleRasterCommand {
            triangle: triangle.clone(),
            uniforms: uniforms.clone(),
            shader: shader.clone(),
            blend_state,
            depth_state,
        };

        group.bench_with_input(BenchmarkId::from_parameter(name), &command, |b, command| {
            let mut framebuffer = FrameBuffer::new(640, 360);
            let mut depthbuffer = DepthBuffer::new(640, 360);

            b.iter(|| {
                depthbuffer.clear(1.0);

                command.rasterise(&mut framebuffer, Some(&mut depthbuffer), bounds);

                black_box(framebuffer.pixels());
                black_box(&depthbuffer);
            });
        });
    }

    group.finish();
}

fn bench_simd_scalar_shader_command(c: &mut Criterion) {
    let mut group = c.benchmark_group("raster_command/simd_scalar_shader");

    let triangle = large_triangle();
    let bounds = full_bounds();

    let shader = Arc::new(TestShader);
    let uniforms = Arc::new(TestUniforms);

    let cases = [
        (
            "no_depth_no_blend",
            DepthState {
                test_enabled: false,
                write_enabled: false,
            },
            None,
        ),
        (
            "depth_no_blend",
            DepthState {
                test_enabled: true,
                write_enabled: true,
            },
            None,
        ),
        (
            "depth_blend",
            DepthState {
                test_enabled: true,
                write_enabled: true,
            },
            Some(BlendState::ALPHA_BLEND),
        ),
    ];

    for (name, depth_state, blend_state) in cases {
        let command = TriangleRasterCommand {
            triangle: triangle.clone(),
            uniforms: uniforms.clone(),
            shader: shader.clone(),
            blend_state,
            depth_state,
        };

        group.bench_with_input(BenchmarkId::from_parameter(name), &command, |b, command| {
            let mut framebuffer = FrameBuffer::new(640, 360);
            let mut depthbuffer = DepthBuffer::new(640, 360);

            b.iter(|| {
                depthbuffer.clear(1.0);

                command.rasterise(&mut framebuffer, Some(&mut depthbuffer), bounds);

                black_box(framebuffer.pixels());
                black_box(&depthbuffer);
            });
        });
    }

    group.finish();
}

fn bench_simd_shader_command(c: &mut Criterion) {
    let mut group = c.benchmark_group("raster_command/simd_shader");

    let triangle = large_triangle();
    let bounds = full_bounds();

    let shader = Arc::new(TestShader);
    let uniforms = Arc::new(TestUniforms);

    let cases = [
        (
            "no_depth_no_blend",
            DepthState {
                test_enabled: false,
                write_enabled: false,
            },
            None,
        ),
        (
            "depth_no_blend",
            DepthState {
                test_enabled: true,
                write_enabled: true,
            },
            None,
        ),
        (
            "depth_blend",
            DepthState {
                test_enabled: true,
                write_enabled: true,
            },
            Some(BlendState::ALPHA_BLEND),
        ),
    ];

    for (name, depth_state, blend_state) in cases {
        let command = TriangleRasterCommandSimd {
            triangle: triangle.clone(),
            uniforms: uniforms.clone(),
            shader: shader.clone(),
            blend_state,
            depth_state,
        };

        group.bench_with_input(BenchmarkId::from_parameter(name), &command, |b, command| {
            let mut framebuffer = FrameBuffer::new(640, 360);
            let mut depthbuffer = DepthBuffer::new(640, 360);

            b.iter(|| {
                depthbuffer.clear(1.0);

                command.rasterise(&mut framebuffer, Some(&mut depthbuffer), bounds);

                black_box(framebuffer.pixels());
                black_box(&depthbuffer);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pre_simd_command,
    bench_simd_scalar_shader_command,
    bench_simd_shader_command
);

criterion_main!(benches);
