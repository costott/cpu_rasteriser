//! blueprint: a technical drawing. White line work on cyanotype paper with a drafting grid, the
//! chip outlined, an exploded view and a plan of it beside, dimension lines, leaders, a datum
//! and a title block.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, View, ViewRef};
use crate::simd::{F, clamp01, fract, mix3, rgb, s, smoothstep, splat3, step_lt, to_colour};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::post::{Chain, Edges, Effect, Noise, Paper};
use crate::themes::overlay::{Align, Line, Overlay, Text};
use crate::themes::{Scene, Stage};

const LINE: Vec3 = Vec3 {
    x: 0.62,
    y: 0.86,
    z: 1.0,
};
const HALF_PAGE: (f32, f32) = (1.68, 0.92);
const EXPLODED_CENTRE: (f32, f32) = (-1.1, 0.42);
const PLAN_CENTRE: (f32, f32) = (1.2, 0.46);
const PLAN_SCALE: f32 = 0.62;

/// One line of the drawing, in drawing units: x scaled by the aspect ratio, y as normalised
/// device coordinates.
#[derive(Clone, Copy, Debug)]
struct Segment {
    from: (f32, f32),
    to: (f32, f32),
    /// The length of a dash and of the gap after it; 0 for a solid line.
    dash: f32,
    /// Half the line's thickness in pixels.
    weight: f32,
    brightness: f32,
}

/// Everything the drawing holds beyond the chip.
struct Drawing {
    segments: Vec<Segment>,
    text: Vec<(String, (f32, f32), Align)>,
    /// The centre and factor that shrink the view being drawn about its own origin.
    view_scale: ((f32, f32), f32),
}

impl Drawing {
    fn new() -> Self {
        let mut drawing = Self {
            segments: Vec::new(),
            text: Vec::new(),
            view_scale: ((0.0, 0.0), 1.0),
        };
        drawing.sheet();
        drawing.view_scale = (EXPLODED_CENTRE, 0.68);
        drawing.exploded_view(EXPLODED_CENTRE);
        drawing.view_scale = (PLAN_CENTRE, PLAN_SCALE);
        drawing.plan_view(PLAN_CENTRE);
        drawing.view_scale = ((0.0, 0.0), 1.0);
        drawing.callouts();
        drawing.notes();
        drawing
    }

    /// `point` shrunk about the view being drawn's centre.
    fn place(&self, point: (f32, f32)) -> (f32, f32) {
        let ((cx, cy), k) = self.view_scale;
        (cx + (point.0 - cx) * k, cy + (point.1 - cy) * k)
    }

    fn line(&mut self, from: (f32, f32), to: (f32, f32), weight: f32, brightness: f32) {
        self.segments.push(Segment {
            from: self.place(from),
            to: self.place(to),
            dash: 0.0,
            weight,
            brightness,
        });
    }

    fn dashed(&mut self, from: (f32, f32), to: (f32, f32)) {
        self.segments.push(Segment {
            from: self.place(from),
            to: self.place(to),
            dash: 0.035,
            weight: 0.7,
            brightness: 0.8,
        });
    }

    fn rectangle(&mut self, (x0, y0): (f32, f32), (x1, y1): (f32, f32), weight: f32) {
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        for i in 0..4 {
            self.line(corners[i], corners[(i + 1) % 4], weight, 1.0);
        }
    }

    fn label(&mut self, text: &str, at: (f32, f32), align: Align) {
        self.label_beside(text, at, (0.0, 0.0), align);
    }

    /// A label `offset` away from `at`, the offset kept at full size inside a shrunken view.
    fn label_beside(&mut self, text: &str, at: (f32, f32), offset: (f32, f32), align: Align) {
        let (x, y) = self.place(at);
        self.text
            .push((text.to_string(), (x + offset.0, y + offset.1), align));
    }

    fn sheet(&mut self) {
        let (w, h) = HALF_PAGE;
        self.rectangle((-w - 0.05, -h - 0.04), (w + 0.05, h + 0.04), 0.6);
        self.rectangle((-w, -h), (w, h), 1.3);
        let (left, bottom) = (0.8, -h);
        self.rectangle((left, bottom), (w, -0.7), 1.0);
        self.line((left, -0.805), (w, -0.805), 0.8, 1.0);
        self.line((1.2, -0.805), (1.2, -0.7), 0.8, 1.0);
        self.label("SILICON-01  CPU", (left + 0.03, -0.745), Align::Left);
        self.label("SCALE 1:1", (left + 0.03, -0.855), Align::Left);
        self.label("SHEET 1 OF 1", (1.23, -0.855), Align::Left);
    }

    /// Plates drawn as isometric outlines, pulled apart along a dashed axis.
    fn exploded_view(&mut self, (cx, cy): (f32, f32)) {
        let plate = |drawing: &mut Self, y: f32, half_width: f32, thickness: f32| {
            let h = half_width * 0.577;
            let (left, top, right, bottom) = (
                (cx - half_width, y),
                (cx, y + h),
                (cx + half_width, y),
                (cx, y - h),
            );
            for (a, b) in [(left, top), (top, right), (right, bottom), (bottom, left)] {
                drawing.line(a, b, 1.2, 1.0);
            }
            let drop = |p: (f32, f32)| (p.0, p.1 - thickness);
            for p in [left, bottom, right] {
                drawing.line(p, drop(p), 1.2, 1.0);
            }
            drawing.line(drop(left), drop(bottom), 1.2, 1.0);
            drawing.line(drop(bottom), drop(right), 1.2, 1.0);
        };
        let (base, die, lid) = (cy - 0.2, cy, cy + 0.26);
        plate(self, base, 0.5, 0.035);
        plate(self, die, 0.24, 0.03);
        plate(self, lid, 0.4, 0.03);
        // Pins: a row of stubs below the substrate's front edges.
        for i in 0..=6 {
            let t = i as f32 / 6.0;
            let x = cx - 0.5 + t * 0.5;
            let y = base - 0.289 * t - 0.035;
            self.line((x, y), (x, y - 0.055), 1.0, 1.0);
            let x = cx + t * 0.5;
            let y = base - 0.289 * (1.0 - t) - 0.035;
            self.line((x, y), (x, y - 0.055), 1.0, 1.0);
        }
        self.dashed((cx, base - 0.4), (cx, lid + 0.35));
        for (dx, from, to) in [
            (-0.24, die, base),
            (0.24, die, base),
            (-0.4, lid, die),
            (0.4, lid, die),
        ] {
            self.dashed((cx + dx, from - 0.03), (cx + dx, to));
        }
        for (name, y) in [
            ("1 LID", lid + 0.12),
            ("2 DIE", die + 0.1),
            ("3 BASE", base + 0.12),
        ] {
            self.line((cx + 0.48, y - 0.04), (cx + 0.62, y - 0.04), 0.8, 0.9);
            self.label(name, (cx + 0.5, y + 0.02), Align::Left);
        }
        self.label("EXPLODED VIEW", (cx, cy + 0.68), Align::Centre);
        self.line((cx - 0.3, cy + 0.65), (cx + 0.3, cy + 0.65), 0.8, 1.0);
    }

    /// The chip from above with its pads, centre lines and dimensions.
    fn plan_view(&mut self, (cx, cy): (f32, f32)) {
        let (outer, inner) = (0.4, 0.2);
        self.rectangle((cx - outer, cy - outer), (cx + outer, cy + outer), 1.3);
        self.rectangle((cx - inner, cy - inner), (cx + inner, cy + inner), 1.2);
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            self.dashed(
                (cx + dx * 0.345, cy + dy * 0.345 - 0.02),
                (cx + dx * 0.345, cy + dy * 0.345 + 0.02),
            );
        }
        self.dashed((cx - outer - 0.1, cy), (cx + outer + 0.1, cy));
        self.dashed((cx, cy - outer - 0.1), (cx, cy + outer + 0.1));
        self.dimension(
            (cx - outer, cy - outer),
            (cx + outer, cy - outer),
            -0.1,
            "4.00",
        );
        self.dimension_vertical(
            (cx - outer, cy - inner),
            (cx - outer, cy + inner),
            -0.1,
            "1.80",
        );
        self.label("PLAN", (cx, cy + outer + 0.14), Align::Centre);
        self.line(
            (cx - 0.12, cy + outer + 0.11),
            (cx + 0.12, cy + outer + 0.11),
            0.8,
            1.0,
        );
    }

    /// A dimension between `a` and `b`, which share a y, drawn `offset` below them with arrows
    /// at both ends and its value above the line.
    fn dimension(&mut self, a: (f32, f32), b: (f32, f32), offset: f32, value: &str) {
        let y = a.1 + offset;
        for p in [a, b] {
            self.line(
                (p.0, p.1 + offset.signum() * 0.015),
                (p.0, y + offset.signum() * 0.03),
                0.8,
                1.0,
            );
        }
        self.line((a.0, y), (b.0, y), 1.0, 1.0);
        for (p, direction) in [(a, 1.0), (b, -1.0)] {
            self.line((p.0, y), (p.0 + direction * 0.04, y + 0.012), 1.0, 1.0);
            self.line((p.0, y), (p.0 + direction * 0.04, y - 0.012), 1.0, 1.0);
        }
        self.label_beside(value, ((a.0 + b.0) * 0.5, y), (0.0, 0.055), Align::Centre);
    }

    /// A dimension between `a` and `b`, which share an x, drawn `offset` to the right of them.
    fn dimension_vertical(&mut self, a: (f32, f32), b: (f32, f32), offset: f32, value: &str) {
        let x = a.0 + offset;
        for p in [a, b] {
            self.line((p.0 + 0.015, p.1), (x + 0.03, p.1), 0.8, 1.0);
        }
        self.line((x, a.1), (x, b.1), 1.0, 1.0);
        for (p, direction) in [(a, 1.0), (b, -1.0)] {
            self.line((x, p.1), (x + 0.012, p.1 + direction * 0.04), 1.0, 1.0);
            self.line((x, p.1), (x - 0.012, p.1 + direction * 0.04), 1.0, 1.0);
        }
        self.label_beside(value, (x, (a.1 + b.1) * 0.5), (-0.03, 0.02), Align::Right);
    }

    /// General notes and a scale bar, below the datum.
    fn notes(&mut self) {
        for (i, note) in [
            "NOTES",
            "1  DIMENSIONS IN MM",
            "2  BREAK ALL SHARP EDGES",
            "3  DRAWN BY A CPU",
        ]
        .iter()
        .enumerate()
        {
            self.label(note, (-1.62, -0.42 - 0.075 * i as f32), Align::Left);
        }
        self.line((-1.62, -0.37), (-1.2, -0.37), 0.8, 1.0);
        let bar = (-0.4, -0.78);
        self.line(bar, (bar.0 + 0.8, bar.1), 1.2, 1.0);
        for i in 0..=4 {
            let x = bar.0 + 0.2 * i as f32;
            self.line((x, bar.1), (x, bar.1 + 0.04), 1.0, 1.0);
            self.label(&format!("{i}"), (x, bar.1 - 0.075), Align::Centre);
        }
    }

    /// Notes with leaders to where the chip stands.
    fn callouts(&mut self) {
        for (note, from, to) in [
            ("A  PKG 4.00", (0.62, 0.72), (0.42, 0.55)),
            ("B  DIE 1.80", (0.62, -0.12), (0.3, 0.1)),
        ] {
            self.line(from, to, 0.9, 1.0);
            self.line(to, (to.0 + 0.05, to.1 + 0.005), 1.0, 1.0);
            self.line(to, (to.0 + 0.04, to.1 + 0.03), 1.0, 1.0);
            self.line(from, (from.0 + 0.5, from.1), 0.9, 1.0);
            self.label(note, (from.0 + 0.02, from.1 + 0.05), Align::Left);
        }
    }
}

pub struct Blueprint {
    look: Look,
    segments: Arc<[Segment]>,
    text: Vec<(String, (f32, f32), Align)>,
    backdrop: SimdPipeline<ScreenVertexShader, Backdrop>,
    overlay: Overlay,
    chain: Chain,
}

impl Blueprint {
    pub fn new() -> Self {
        let drawing = Drawing::new();
        Self {
            look: Look {
                substrate: Vec3::new(0.05, 0.2, 0.52),
                die: Vec3::new(0.07, 0.3, 0.75),
                gold: Vec3::new(0.25, 0.55, 1.0),
                streams: Vec3::new(0.55, 0.85, 1.0),
                key_colour: Vec3::new(0.7, 0.85, 1.0),
                sky: Vec3::new(0.2, 0.3, 0.45),
                ground: Vec3::new(0.05, 0.1, 0.2),
                ..Look::default()
            },
            segments: drawing.segments.into(),
            text: drawing.text,
            backdrop: SimdPipeline::new(ScreenVertexShader, Backdrop)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
            overlay: Overlay::new(),
            chain: Chain::new(vec![
                Effect::Edges(Edges {
                    ink: LINE * 2.2,
                    width: 1.2,
                    sensitivity: 1200.0,
                    wobble: 0.0,
                    ..Edges::default()
                }),
                Effect::Noise(Noise {
                    amount: 0.04,
                    floor: 0.004,
                    cell: 1.0,
                    chroma: 0.0,
                }),
                Effect::Paper(Paper {
                    tint: Vec3::new(0.95, 1.0, 1.08),
                    grain: 0.12,
                    fibre: 0.3,
                    scale: 2.0,
                }),
            ]),
        }
    }
}

/// The ndc height of the horizon, where the drawing's ground datum lies.
fn horizon(view: &View) -> f32 {
    -view.forward.y / (view.up.y * view.tan_half_fov)
}

impl Stage for Blueprint {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Marimba],
            ambience: vec![Ambience::once(
                Source::Recording("quill", 0.0),
                0.0,
                0.45,
                0.2,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let window = (scene.extent.width, scene.extent.height);
        let pixel = 2.0 * window.1 as f32 / 720.0;
        let aspect = scene.view.aspect;
        let mut text = Text::default();
        for (words, (x, y), align) in &self.text {
            let line = Line {
                text: words,
                x: x / aspect,
                y: *y,
                align: *align,
                colour: LINE * 1.8,
            };
            text.line(line, pixel, window);
        }
        let datum = horizon(&scene.view);
        let line = Line {
            text: "GL +0.00",
            x: -HALF_PAGE.0 / aspect + 0.02,
            y: datum + 0.06,
            align: Align::Left,
            colour: LINE * 1.8,
        };
        text.line(line, pixel, window);
        self.overlay.set(text);
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        let uniforms = BackdropUniforms {
            view: scene.view.clone(),
            segments: self.segments.clone(),
            horizon: horizon(&scene.view),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.backdrop, (), |_| uniforms.clone());
        self.overlay.draw(pass);
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
            bloom: 0.25,
            vignette: 0.3,
            grain: 0.008,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[derive(Clone)]
struct BackdropUniforms {
    view: ViewRef,
    segments: Arc<[Segment]>,
    horizon: f32,
    time: f32,
}

/// The sheet: blue paper, the drafting grid and every line of the drawing.
struct Backdrop;

impl FragmentShaderSimd<ScreenVaryings> for Backdrop {
    type Uniforms = BackdropUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &BackdropUniforms) -> ColourSimd {
        let view = &u.view;
        let [nx, ny] = v.ndc();
        let (x, y) = (nx * s(view.aspect), ny);
        let pixel = s(view.line_angle / view.tan_half_fov);

        let glow = smoothstep(0.0, 2.2, x * x * s(0.4) + y * y);
        let mut colour = mix3(rgb(0.05, 0.22, 0.55), rgb(0.025, 0.12, 0.36), glow);

        let grid = |spacing: f32, weight: f32| {
            let near = |p: F| (fract(p * s(1.0 / spacing) + s(0.5)) - s(0.5)).abs() * s(spacing);
            clamp01(s(weight) - near(x).fast_min(near(y)) / pixel + s(0.5))
        };
        let lines = grid(0.05, 0.5) * s(0.1) + grid(0.25, 0.8) * s(0.22);
        colour = colour + splat3(LINE) * lines;

        let mut ink = s(0.0);
        for segment in u.segments.iter() {
            ink = ink.fast_max(cover(segment, x, y, pixel, u.time));
        }
        // The ground datum, a dash-dot line with section hatching beneath it.
        let datum = Segment {
            from: (-HALF_PAGE.0, u.horizon),
            to: (HALF_PAGE.0, u.horizon),
            dash: 0.09,
            weight: 1.0,
            brightness: 1.0,
        };
        ink = ink.fast_max(cover(&datum, x, y, pixel, u.time));
        let below = u.horizon - 0.03;
        let in_band = (y - s(below)).simd_gt(s(0.0))
            & (y - s(u.horizon)).simd_lt(s(0.0))
            & x.abs().simd_lt(s(HALF_PAGE.0));
        let diagonal = (fract((x + y) * s(1.0 / 0.02)) - s(0.5)).abs() * s(0.02);
        let hatch = clamp01(s(0.7) - diagonal / pixel + s(0.5)) * in_band.select(s(0.7), s(0.0));
        ink = ink.fast_max(hatch);
        // The plan's pads: small squares in a band round the die.
        ink = ink.fast_max(pads(x, y, pixel));
        colour = mix3(colour, splat3(LINE * 1.7), ink);
        to_colour(colour, s(SKY_DISTANCE))
    }
}

/// How much of a pixel at (`x`, `y`) `segment` covers.
#[inline(always)]
fn cover(segment: &Segment, x: F, y: F, pixel: F, time: f32) -> F {
    let (ax, ay) = (segment.from.0, segment.from.1);
    let (bx, by) = (segment.to.0 - ax, segment.to.1 - ay);
    let length2 = bx * bx + by * by;
    let (px, py) = (x - s(ax), y - s(ay));
    let along = clamp01((px * s(bx) + py * s(by)) * s(1.0 / length2));
    let (dx, dy) = (px - along * s(bx), py - along * s(by));
    let distance = (dx * dx + dy * dy).sqrt();
    let mut coverage =
        clamp01(s(segment.weight) - distance / pixel + s(0.5)) * s(segment.brightness);
    if segment.dash > 0.0 {
        let phase = along * s(length2.sqrt() / (segment.dash * 2.0)) + s(time * 0.05);
        coverage *= step_lt(fract(phase), s(0.6));
    }
    coverage
}

/// Outlined pad squares on the plan view's perimeter band.
#[inline(always)]
fn pads(x: F, y: F, pixel: F) -> F {
    const CELL: f32 = 0.06 * PLAN_SCALE;
    let (qx, qy) = (x - s(PLAN_CENTRE.0), y - s(PLAN_CENTRE.1));
    let radius = qx.abs().fast_max(qy.abs());
    let band = (radius - s(0.3 * PLAN_SCALE))
        .simd_gt(s(0.0))
        .select(s(1.0), s(0.0))
        * (radius - s(0.37 * PLAN_SCALE))
            .simd_lt(s(0.0))
            .select(s(1.0), s(0.0));
    let (lx, ly) = (
        (fract(qx * s(1.0 / CELL) + s(0.5)) - s(0.5)) * s(CELL),
        (fract(qy * s(1.0 / CELL) + s(0.5)) - s(0.5)) * s(CELL),
    );
    let box_distance = (lx.abs().fast_max(ly.abs()) - s(0.016 * PLAN_SCALE)).abs();
    clamp01(s(0.6) - box_distance / pixel + s(0.5)) * band * s(0.7)
}
