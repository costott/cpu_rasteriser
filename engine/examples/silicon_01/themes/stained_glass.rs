//! stained-glass: the chip before a cathedral window of leaded glass, light streaming through it
//! onto a polished floor. The window, its masonry and the floor are a backdrop shader; the chip
//! is cut into glass cells by the screen-space stylisation.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, view_ray};
use crate::simd::{
    F, clamp01, fract, hash2, lt, mix3, rgb, s, select3, smoothstep, splat3, to_colour,
};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::noise;
use crate::themes::kit::post::{Cells, Chain, Edges, Effect, voronoi};
use crate::themes::{Scene, Stage};

/// Where the floor and the window's wall stand in the world, giving the line where they meet.
const FLOOR: f32 = -3.4;
const WALL: f32 = -60.0;
/// The window: half its width at the sill, the radius of the arcs of its pointed arch and the
/// height where the arch springs, in drawing units (x scaled by the aspect ratio, y in device
/// coordinates).
const HALF_WIDTH: f32 = 1.0;
const ARCH_RADIUS: f32 = 1.25;
const SPRING: f32 = 0.0;
/// Glass cells per drawing unit.
const CELLS: f32 = 5.0;
const LEAD: Vec3 = Vec3 {
    x: 0.008,
    y: 0.008,
    z: 0.01,
};
/// The glass: cobalt blues dominate, with ruby, amber and a little emerald, as in a medieval window.
const JEWELS: [Vec3; 6] = [
    Vec3 {
        x: 0.03,
        y: 0.12,
        z: 0.75,
    },
    Vec3 {
        x: 0.05,
        y: 0.22,
        z: 0.85,
    },
    Vec3 {
        x: 0.02,
        y: 0.07,
        z: 0.5,
    },
    Vec3 {
        x: 0.8,
        y: 0.05,
        z: 0.1,
    },
    Vec3 {
        x: 0.85,
        y: 0.42,
        z: 0.04,
    },
    Vec3 {
        x: 0.03,
        y: 0.3,
        z: 0.2,
    },
];
/// The light the window throws on the wall, floor and air, with the colours of its glass.
const SPILL_COOL: Vec3 = Vec3 {
    x: 0.1,
    y: 0.25,
    z: 1.0,
};
const SPILL_WARM: Vec3 = Vec3 {
    x: 1.0,
    y: 0.45,
    z: 0.15,
};

pub struct StainedGlass {
    look: Look,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    chain: Chain,
}

impl StainedGlass {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.55, 0.06, 0.1),
                die: Vec3::new(0.1, 0.35, 1.0),
                gold: Vec3::new(1.0, 0.65, 0.1),
                streams: Vec3::new(1.0, 0.95, 0.7),
                key_colour: Vec3::new(2.2, 1.9, 1.5),
                sky: Vec3::new(0.7, 0.55, 0.7),
                ground: Vec3::new(0.25, 0.1, 0.15),
                ..Look::default()
            },
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            chain: Chain::new(vec![
                Effect::Cells(Cells {
                    size: 15.0,
                    jitter: 0.9,
                    lead: 2.6,
                    lead_colour: LEAD,
                    tint: 0.65,
                    surfaces_only: true,
                }),
                Effect::Edges(Edges {
                    ink: LEAD,
                    width: 1.6,
                    sensitivity: 900.0,
                    ..Edges::default()
                }),
            ]),
        }
    }
}

impl Stage for StainedGlass {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Organ, Instrument::Choir],
            treatment: Treatment::Vast,
            ambience: vec![Ambience::once(
                Source::Recording("bell", 0.0),
                0.05,
                1.4,
                0.2,
                0.25,
            )],
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        let view = &scene.view;
        let sill = view.view_projection * Vec4::new(0.0, FLOOR, WALL, 1.0);
        let uniforms = BackdropUniforms {
            view: view.clone(),
            time: scene.local_time,
            sill: sill.y / sill.w,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
    }

    fn stylise(
        &mut self,
        context: &mut RenderContext<'_>,
        scene: &Scene<'_>,
        image: Arc<TextureSampler>,
    ) -> Option<Arc<TextureSampler>> {
        Some(self.chain.run(context, scene, image))
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.9,
            vignette: 0.55,
            grain: 0.012,
            aberration: 0.002,
            desaturate: -0.1,
        }
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
    time: f32,
    /// The device y where the floor meets the window's wall.
    sill: f32,
}

struct Backdrop;

/// A jewel colour picked by `id` in [0, 1).
#[inline(always)]
fn jewel(id: F) -> Vec3Simd {
    let k = (id * s(JEWELS.len() as f32)).floor();
    let mut colour = splat3(JEWELS[JEWELS.len() - 1]);
    for i in (0..JEWELS.len() - 1).rev() {
        colour = select3(lt(k, s(i as f32 + 0.5)), splat3(JEWELS[i]), colour);
    }
    colour
}

/// How far (`x`, `y`) lies inside the window's opening, negative outside it.
#[inline(always)]
fn depth_in_window(x: F, y: F, sill: f32) -> F {
    let centre = ARCH_RADIUS - HALF_WIDTH;
    let above = y - s(SPRING);
    let (left, right) = (x - s(centre), x + s(centre));
    let arch = (s(ARCH_RADIUS) - (left * left + above * above).sqrt())
        .fast_min(s(ARCH_RADIUS) - (right * right + above * above).sqrt());
    let straight = s(HALF_WIDTH) - x.abs();
    let side = above.simd_lt(s(0.0)).select(straight, arch);
    side.fast_min(y - s(sill))
}

/// The glass at (`x`, `y`) in the window, with the lead's coverage: backlit jewel cells with a
/// shimmer and streaked glass, crossed by saddle bars.
#[inline(always)]
fn glass(x: F, y: F, time: f32, pixel: F) -> (Vec3Simd, F) {
    let cell = voronoi(x * s(CELLS), y * s(CELLS), 0.9);
    let streak = noise::fbm(x * s(3.0) + cell.id * s(5.0), y * s(9.0), 3);
    let centre = s(1.0) - smoothstep(0.0, 0.7, cell.nearest);
    let shimmer = s(1.0) + s(0.14) * (s(time * 0.8) + cell.id * s(40.0)).sin();
    let glow = (s(0.7) + centre * s(0.7) + streak * s(0.6)) * shimmer * s(1.2);
    let colour = jewel(cell.id) * glow;
    let edge = (cell.second - cell.nearest) * s(0.5 / CELLS);
    let mut lead = clamp01((s(0.011) - edge) / pixel + s(0.5));
    for bar in [0.28f32, 0.74] {
        lead = lead.fast_max(clamp01((s(0.008) - (y - s(bar)).abs()) / pixel + s(0.5)));
    }
    lead = lead.fast_max(clamp01((s(0.016) - x.abs()) / pixel + s(0.5)));
    (colour, lead)
}

/// Masonry: staggered blocks, each a little different, with mortar between.
#[inline(always)]
fn masonry(x: F, y: F, pixel: F) -> Vec3Simd {
    let (height, width) = (0.13, 0.34);
    let course = (y * s(1.0 / height)).floor();
    let stagger = (course * s(0.5) - (course * s(0.5)).floor()) * s(width);
    let column = ((x + stagger) * s(1.0 / width)).floor();
    let block = hash2(column, course);
    let (fx, fy) = (
        fract((x + stagger) * s(1.0 / width)),
        fract(y * s(1.0 / height)),
    );
    let mortar_x = fx.fast_min(s(1.0) - fx) * s(width);
    let mortar_y = fy.fast_min(s(1.0) - fy) * s(height);
    let mortar = clamp01((s(0.006) - mortar_x.fast_min(mortar_y)) / pixel + s(0.5));
    let weathering = noise::fbm(x * s(6.0), y * s(6.0), 3);
    let stone = rgb(0.05, 0.052, 0.07) * (s(0.7) + block * s(0.6) + weathering * s(0.5));
    mix3(stone, rgb(0.008, 0.008, 0.012), mortar)
}

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(view.aspect), ny);
        let pixel = s(view.pixel_angle / view.tan_half_fov);
        let time = u.time;
        let on_floor = ny.simd_lt(s(u.sill));

        // The window seen directly above the sill, and mirrored in the floor below it.
        let sample_y = on_floor.select(s(2.0 * u.sill) - y, y);
        let depth = depth_in_window(x, sample_y, u.sill);
        let (glass, lead) = glass(x, sample_y, time, pixel);
        let in_glass = clamp01(depth / pixel + s(0.5));
        let window = mix3(glass, splat3(LEAD), lead);

        // Dressed stone round the opening, then the wall, lit by the colours spilling from it.
        let outside = (-depth).fast_max(s(0.0));
        let spill = (-outside * s(2.6)).exp();
        let hue = mix3(
            splat3(SPILL_COOL),
            splat3(SPILL_WARM),
            smoothstep(
                -0.8,
                0.8,
                x + noise::fbm(x * s(2.0), sample_y * s(2.0), 2) - s(0.5),
            ),
        ) * s(1.1);
        let wall = crate::simd::mul3(
            masonry(x, sample_y, pixel),
            splat3(Vec3::ONE) * s(0.6) + hue * (spill * s(4.0)),
        );
        let frame =
            clamp01((s(0.08) - outside) / pixel * s(0.5)) * s(0.3) * smoothstep(0.0, 0.01, outside);
        let wall = wall + rgb(0.2, 0.17, 0.14) * (frame * spill);
        let mut colour = mix3(wall, window, in_glass);

        // Light in the air: shafts streaming from the window's heart, in the colours of its glass.
        let (dx, dy) = (x, y - s(0.35));
        let radius = (dx * dx + dy * dy).sqrt();
        let angle = dy.atan2(dx);
        let rays = smoothstep(
            0.42,
            0.85,
            noise::fbm(angle * s(5.0) + s(time * 0.04), s(3.3), 3),
        );
        let shaft_hue = mix3(
            splat3(SPILL_COOL),
            splat3(SPILL_WARM),
            smoothstep(
                -1.0,
                1.0,
                angle.sin() * s(1.0) + noise::fbm(angle * s(2.0), s(7.7), 2) - s(0.5),
            ),
        ) + rgb(0.25, 0.22, 0.15);
        let shafts = rays * (-radius * s(0.8)).exp();
        colour = colour + shaft_hue * (shafts * s(0.9));

        // Dust turning in the light.
        let grid = s(18.0);
        let (mx, my) = (x * grid, y * grid - s(time * 1.2));
        let (gx, gy) = (mx.floor(), my.floor());
        let (fx, fy) = (mx - gx - s(0.5), my - gy - s(0.5));
        let mote = (s(1.0) - (fx * fx + fy * fy).sqrt() * s(5.0)).fast_max(s(0.0));
        let twinkle = hash2(gx, gy).simd_gt(s(0.93)).select(s(1.0), s(0.0))
            * (s(0.6) + s(0.4) * (s(time * 3.0) + hash2(gy, gx) * s(30.0)).sin());
        colour = colour + rgb(1.0, 0.9, 0.7) * (mote * twinkle * (shafts + s(0.15)) * s(5.0));

        // The floor: polished dark flagstones that hold the window's mirror image.
        let ray = view_ray(view, nx, ny);
        let t = (s(FLOOR - view.eye.y) / ray.y.fast_min(s(-0.002))).fast_max(s(0.0));
        let (wx, wz) = (s(view.eye.x) + ray.x * t, s(view.eye.z) + ray.z * t);
        let footprint = s(view.pixel_angle) * t;
        let joint = |p: F, size: F| {
            let d = (fract(p * s(0.4) + s(0.5)) - s(0.5)).abs() * s(2.5);
            clamp01(s(1.0) - d / size.fast_max(s(0.02)))
        };
        let joints =
            joint(wx, footprint).fast_max(joint(wz, footprint * s(4.0))) * (-t * s(0.03)).exp();
        let slab = hash2((wx * s(0.4)).floor(), (wz * s(0.4)).floor());
        let polish = (s(1.0) - (s(u.sill) - ny).fast_max(s(0.0)) * s(1.1)).fast_max(s(0.0));
        let mirrored = window * (in_glass * s(0.55) + (s(1.0) - in_glass) * s(0.1)) * polish;
        let floor = rgb(0.012, 0.013, 0.02) * (s(0.7) + slab * s(0.6)) * (s(1.0) - joints * s(0.8))
            + mirrored * (s(1.0) - joints * s(0.6));
        colour = select3(on_floor, floor + shaft_hue * (shafts * s(0.35)), colour);
        to_colour(colour, s(SKY_DISTANCE))
    }
}
