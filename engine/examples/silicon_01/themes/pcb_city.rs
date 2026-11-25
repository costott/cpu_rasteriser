//! pcb-city: a circuit board seen as a city at night. A skyline of capacitor barrels, banded
//! resistors, black IC slabs, heatsinks and blinking LEDs stands along the board's far edge, three
//! rows deep and soft with distance, in front of a haze of out-of-focus lights. Green solder
//! mask with copper traces, pads, vias and silkscreen runs towards it from the front.
//!
//! One full-screen shader in screen units (the screen is 2 high, so a pixel is `2 / height`).
//! The skyline is laid out per cell from a hash, so a row needs no loop over its components.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::ViewRef;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::simd::{F, clamp01, fract, hash2, mix3, rgb, s, select, select3, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::noise::{fbm, value};
use crate::themes::kit::post::expand;
use crate::themes::{Scene, Stage};

/// Where the board's far edge meets the skyline: the world's horizon line.
const HORIZON: f32 = -0.5;
/// The distance the whole scene reports as its view distance.
const SCENE_DISTANCE: f32 = 30.0;

/// One row of the skyline, front rows drawn last.
struct Row {
    /// The cell width, which sets the component spacing.
    pitch: f32,
    /// How much the row's components are scaled and how far below the horizon they stand.
    scale: f32,
    drop: f32,
    /// The edge softness in screen units (depth of field) and how far the row fades into the haze.
    blur: f32,
    haze: f32,
    seed: f32,
}

const ROWS: [Row; 3] = [
    Row {
        pitch: 0.17,
        scale: 0.55,
        drop: 0.0,
        blur: 0.012,
        haze: 0.62,
        seed: 3.0,
    },
    Row {
        pitch: 0.25,
        scale: 0.8,
        drop: -0.04,
        blur: 0.005,
        haze: 0.32,
        seed: 11.0,
    },
    Row {
        pitch: 0.36,
        scale: 1.1,
        drop: -0.1,
        blur: 0.0,
        haze: 0.0,
        seed: 23.0,
    },
];

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Board;

impl FragmentShaderSimd<ScreenVaryings> for Board {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let (a, b) = (nx * s(view.aspect), ny);
        let pixel = s(view.pixel_angle / view.tan_half_fov);
        let time = u.time;

        let mut colour = backdrop(a, b, time);
        let floor = board(a, b, pixel, time);
        colour = mix3(colour, floor, clamp01((s(HORIZON) - b) / pixel + s(0.5)));
        for row in &ROWS {
            colour = skyline(colour, a, b, row, pixel, time);
        }
        to_colour(expand(colour), s(SCENE_DISTANCE))
    }
}

/// 1 inside a shape whose signed distance is `d`, fading across `edge`.
#[inline(always)]
fn coverage(d: F, edge: F) -> F {
    clamp01(s(0.5) - d / edge)
}

/// The signed distance to a box centred at (`cx`, `cy`) with half-size (`hx`, `hy`).
#[inline(always)]
fn rectangle(a: F, b: F, cx: F, cy: F, hx: F, hy: F) -> F {
    let (dx, dy) = ((a - cx).abs() - hx, (b - cy).abs() - hy);
    let (ox, oy) = (dx.fast_max(s(0.0)), dy.fast_max(s(0.0)));
    (ox * ox + oy * oy).sqrt() + dx.fast_max(dy).fast_min(s(0.0))
}

/// The lamp colour of the LED with index `n`: amber, red, green or cyan.
#[inline(always)]
fn led_colour(n: F) -> Vec3Simd {
    let amber = rgb(1.0, 0.62, 0.15);
    let red = rgb(1.0, 0.16, 0.1);
    let green = rgb(0.25, 1.0, 0.35);
    let cyan = rgb(0.2, 0.8, 1.0);
    let lower = mix3(amber, red, select(n.simd_gt(s(0.25)), s(1.0), s(0.0)));
    let upper = mix3(green, cyan, select(n.simd_gt(s(0.75)), s(1.0), s(0.0)));
    mix3(lower, upper, select(n.simd_gt(s(0.5)), s(1.0), s(0.0)))
}

/// The dark space behind the board: a teal haze at the horizon and out-of-focus LEDs that
/// breathe at their own rates.
#[inline(always)]
fn backdrop(a: F, b: F, time: f32) -> Vec3Simd {
    let height = smoothstep(HORIZON, 1.0, b);
    let mut colour = mix3(
        rgb(0.05, 0.19, 0.15),
        rgb(0.006, 0.02, 0.022),
        height.sqrt(),
    );
    for (layer, (size, drift)) in [(0.42, 0.012), (0.24, 0.02), (0.14, 0.03)]
        .into_iter()
        .enumerate()
    {
        let layer = layer as f32;
        let (u, w) = (
            a / s(size) + s(time * drift + layer * 5.0),
            b / s(size) + s(layer * 3.0),
        );
        let (cu, cv) = (u.floor(), w.floor());
        let present = hash2(cu, cv + s(11.0 * layer));
        let (jx, jy) = (
            hash2(cu, cv * s(1.3)) - s(0.5),
            hash2(cv, cu * s(1.7)) - s(0.5),
        );
        let (dx, dy) = (
            fract(u) - s(0.5) - jx * s(0.4),
            fract(w) - s(0.5) - jy * s(0.4),
        );
        let disc = s(1.0) - smoothstep(0.2, 0.3, (dx * dx + dy * dy).sqrt());
        let breathe = s(0.65) + s(0.35) * (s(time * (0.6 + 0.4 * layer)) + present * s(40.0)).sin();
        let shown = select(present.simd_gt(s(0.6)), disc * breathe, s(0.0));
        colour = colour
            + led_colour(hash2(cu * s(2.3), cv + s(layer))) * shown * s(0.07) * (s(0.4) + height);
    }
    colour
}

/// The board's surface running towards the horizon: solder mask with copper traces, pads, vias and
/// silkscreen, in perspective.
#[inline(always)]
fn board(a: F, b: F, pixel: F, time: f32) -> Vec3Simd {
    let v = (s(HORIZON) - b).fast_max(s(0.0));
    let depth = s(0.3) / (v + s(0.02));
    let (x, y) = (a * depth, depth);
    let blur = (pixel * depth * depth / s(0.3)).fast_max(s(0.002));

    let tone = fbm(x * s(0.8), y * s(0.8), 4) - s(0.5);
    let grain = value(x * s(40.0), y * s(40.0)) - s(0.5);
    let mut colour = rgb(0.03, 0.27, 0.15) * (s(0.9) + tone * s(0.5) + grain * s(0.06));

    let (rows, columns) = (y * s(8.0), x * s(1.6));
    let (track_row, track) = (rows.floor(), fract(rows));
    let shift = hash2(track_row, s(4.0)) * s(9.0);
    let (segment, along) = ((columns + shift).floor(), fract(columns + shift));
    let live = select(hash2(segment, track_row).simd_gt(s(0.3)), s(1.0), s(0.0));
    let wander = (hash2(segment, track_row + s(7.0)) - s(0.5)) * s(0.3);
    let half_width = s(0.05) + hash2(track_row, s(1.0)) * s(0.05);
    let centre = track - s(0.5) - wander;
    let trace = coverage((centre.abs() - half_width) / s(8.0), blur) * live;
    let (px, py) = ((along - s(0.04)) / s(1.6), centre / s(8.0));
    let pad_radius = (px * px + py * py).sqrt();
    let pad = coverage(pad_radius - s(0.045), blur) * live;
    let copper = rgb(0.72, 0.43, 0.2) * (s(0.8) + grain * s(0.5));
    let tinned = rgb(0.85, 0.75, 0.55) * (s(0.85) + grain * s(0.3));
    let hole = rgb(0.01, 0.02, 0.02);
    colour = mix3(colour, copper, trace);
    colour = mix3(colour, tinned, pad);
    colour = mix3(colour, hole, coverage(pad_radius - s(0.016), blur) * live);

    let (vu, vv) = (x * s(3.0), y * s(3.0));
    let via = select(
        hash2(vu.floor(), vv.floor() + s(2.0)).simd_gt(s(0.82)),
        s(1.0),
        s(0.0),
    );
    let (dx, dy) = (fract(vu) - s(0.5), fract(vv) - s(0.5));
    let radius = (dx * dx + dy * dy).sqrt();
    colour = mix3(
        colour,
        tinned,
        coverage((radius - s(0.14)) / s(3.0), blur) * via,
    );
    colour = mix3(
        colour,
        hole,
        coverage((radius - s(0.06)) / s(3.0), blur) * via,
    );

    let (su, sv) = (x * s(1.2), y * s(1.2));
    let outline = select(
        hash2(su.floor() + s(5.0), sv.floor()).simd_gt(s(0.72)),
        s(1.0),
        s(0.0),
    );
    let outline_d = rectangle(fract(su), fract(sv), s(0.5), s(0.5), s(0.3), s(0.2)).abs() / s(1.2);
    colour = mix3(
        colour,
        rgb(0.82, 0.86, 0.8),
        coverage(outline_d - s(0.002), blur) * outline * s(0.7),
    );

    let (ka, kb) = ((a + s(0.35)) * s(0.8), (b + s(0.75)) * s(1.4));
    let key = s(1.0) - smoothstep(0.0, 1.3, (ka * ka + kb * kb).sqrt());
    let flicker = s(0.92 + 0.08 * (time * 1.3).sin());
    colour = colour * (s(0.5) + key * s(0.95) * flicker);
    let sheen = (s(1.0) - smoothstep(0.0, 0.5, (a * s(0.5) + (b + s(0.78)) * s(2.2)).abs()))
        * (s(1.0) - smoothstep(0.2, 1.2, a.abs()));
    colour = colour + rgb(0.6, 0.85, 0.7) * sheen * s(0.2);

    mix3(haze_colour(), colour, smoothstep(0.0, 0.3, v))
}

#[inline(always)]
fn haze_colour() -> Vec3Simd {
    rgb(0.04, 0.2, 0.15)
}

/// The number of kinds of skyline component: IC, capacitor can, resistor, ceramic capacitor,
/// heatsink and LED.
const KINDS: usize = 6;

/// A skyline component's colour, coverage and the light it gives off.
struct Part {
    colour: Vec3Simd,
    mask: F,
    glow: Vec3Simd,
}

/// The value for `kind` from `values`, one per component kind.
#[inline(always)]
fn per_kind(kind: F, values: [f32; KINDS]) -> F {
    let mut result = s(values[0]);
    for (index, &value) in values.iter().enumerate().skip(1) {
        result = select(kind.simd_ge(s(index as f32)), s(value), result);
    }
    result
}

/// The colour for `kind` from `colours`, one per component kind.
#[inline(always)]
fn per_kind3(kind: F, colours: [Vec3Simd; KINDS]) -> Vec3Simd {
    let mut result = colours[0];
    for (index, &colour) in colours.iter().enumerate().skip(1) {
        result = select3(kind.simd_ge(s(index as f32)), colour, result);
    }
    result
}

/// A resistor's band colour from `n` in [0, 1): brown, red, orange, green or blue.
#[inline(always)]
fn band_colour(n: F) -> Vec3Simd {
    let mut colour = rgb(0.32, 0.15, 0.06);
    for (edge, next) in [
        (0.2, rgb(0.7, 0.1, 0.07)),
        (0.4, rgb(0.9, 0.4, 0.05)),
        (0.6, rgb(0.1, 0.5, 0.2)),
        (0.8, rgb(0.1, 0.25, 0.7)),
    ] {
        colour = mix3(colour, next, select(n.simd_gt(s(edge)), s(1.0), s(0.0)));
    }
    colour
}

/// A component standing on the board, at (`lx`, `ly`) relative to the middle of its foot.
struct Shape {
    kind: F,
    half: F,
    height: F,
    random: F,
}

#[inline(always)]
fn component(lx: F, ly: F, shape: &Shape, edge: F, time: f32) -> Part {
    let Shape {
        kind,
        half,
        height,
        random,
    } = *shape;
    let across = lx / half;
    let up = ly / height;
    let round = (per_kind(kind, [0.12, 0.3, 1.0, 0.2, 0.0, 1.0]) * half).fast_min(height * s(0.5));
    let d = rectangle(
        lx,
        ly,
        s(0.0),
        height * s(0.5),
        half - round,
        height * s(0.5) - round,
    ) - round;
    let mask = coverage(d, edge);

    let cylinder = (s(1.0) - across * across).fast_max(s(0.0)).sqrt();
    let lit = s(0.3) + s(0.7) * clamp01(s(0.5) - across * s(0.35) + cylinder * s(0.45));
    let specular = (s(1.0) - smoothstep(0.0, 0.18, (across + s(0.4)).abs())) * cylinder;
    let top_light = smoothstep(0.8, 1.0, up);

    let pin_zone = s(1.0) - smoothstep(0.1, 0.13, up);
    let pins = select(fract(across * s(4.0)).simd_lt(s(0.55)), s(1.0), s(0.0));
    let marking = select(fract(up * s(7.0)).simd_lt(s(0.4)), s(1.0), s(0.0))
        * (smoothstep(0.3, 0.34, up) - smoothstep(0.78, 0.82, up))
        * (s(1.0) - smoothstep(0.55, 0.6, across.abs()));
    let (ox, oy) = ((across + s(0.7)) * half, (up - s(0.85)) * height);
    let pin_one = s(1.0) - smoothstep(0.0, 0.004, (ox * ox + oy * oy).sqrt() - s(0.005));
    let chip_body =
        rgb(0.03, 0.033, 0.04) * (s(0.8) + top_light * s(0.9) + (s(1.0) - across.abs()) * s(0.3));
    let ic = mix3(chip_body, rgb(0.28, 0.3, 0.3), marking * s(0.7));
    let ic = mix3(ic, rgb(0.5, 0.5, 0.48), pin_one * s(0.6));
    let ic = mix3(
        ic,
        mix3(rgb(0.02, 0.02, 0.025), rgb(0.62, 0.62, 0.6), pins),
        pin_zone,
    );
    let (dx, dy) = ((across - s(0.72)) * half, (up - s(0.78)) * height);
    let dot = s(1.0) - smoothstep(0.0, 0.003, (dx * dx + dy * dy).sqrt() - s(0.005));

    let stripe = smoothstep(0.35, 0.4, across) * (s(1.0) - smoothstep(0.78, 0.83, across));
    let rim = smoothstep(0.9, 0.93, up);
    let can_body = mix3(rgb(0.015, 0.03, 0.07), rgb(0.31, 0.32, 0.33), stripe);
    let can = mix3(
        can_body * lit,
        rgb(0.6, 0.62, 0.64) * (s(0.5) + lit * s(0.6)),
        rim,
    );
    let can = can + rgb(0.5, 0.55, 0.6) * specular * s(0.35);

    let band = |lo: f32, hi: f32| smoothstep(lo, lo + 0.02, up) - smoothstep(hi, hi + 0.02, up);
    let mut resistor = rgb(0.78, 0.66, 0.44) * lit;
    resistor = mix3(resistor, band_colour(random) * lit, band(0.26, 0.34));
    resistor = mix3(
        resistor,
        band_colour(hash2(random * s(7.0), s(1.0))) * lit,
        band(0.42, 0.5),
    );
    resistor = mix3(resistor, rgb(0.9, 0.4, 0.05) * lit, band(0.58, 0.66));
    resistor = mix3(resistor, rgb(0.8, 0.62, 0.2) * lit, band(0.78, 0.86));
    resistor = resistor + rgb(0.7, 0.65, 0.5) * specular * s(0.25);

    let cap_ends = clamp01(smoothstep(0.78, 0.82, up) + s(1.0) - smoothstep(0.18, 0.22, up));
    let ceramic = mix3(
        rgb(0.82, 0.5, 0.1) * (s(0.7) + (s(1.0) - up) * s(0.3) + lit * s(0.3)),
        rgb(0.7, 0.7, 0.66),
        cap_ends,
    );

    let fins = (s(0.5) + s(0.5) * (across * s(11.0)).cos())
        * (s(0.35) + s(0.65) * smoothstep(0.0, 0.3, up));
    let plate = s(1.0) - smoothstep(0.1, 0.14, up);
    let sink = rgb(0.5, 0.54, 0.56) * (s(0.25) + fins * s(0.75) + top_light * s(0.3));
    let sink = mix3(sink, rgb(0.3, 0.32, 0.33), plate);

    let rate = s(1.0) + random * s(2.5);
    let blink = smoothstep(-0.2, 0.2, (rate * s(time) + random * s(30.0)).sin());
    let lamp = led_colour(hash2(random * s(3.1), s(2.0)));
    let led = lamp * (s(0.25) + blink * s(1.0) + cylinder * s(0.3))
        + rgb(1.0, 1.0, 1.0) * specular * s(0.3);

    let ic = mix3(ic, lamp * (s(0.5) + blink * s(1.5)), dot);
    let colour = per_kind3(kind, [ic, can, resistor, ceramic, sink, led]);
    let is_led = select(kind.simd_ge(s(5.0)), s(1.0), s(0.0));
    let is_ic = select(kind.simd_lt(s(1.0)), s(1.0), s(0.0));
    let centre_x = select(is_led.simd_gt(s(0.0)), s(0.0), s(0.72) * half);
    let centre_y = select(is_led.simd_gt(s(0.0)), height * s(0.5), s(0.78) * height);
    let (gx, gy) = (lx - centre_x, ly - centre_y);
    let halo = s(1.0) / (s(1.0) + (gx * gx + gy * gy) * s(2500.0));
    let glow = lamp * halo * blink * (is_led + is_ic * s(0.6)) * s(0.4);
    Part { colour, mask, glow }
}

/// One row of the skyline: a component in each cell, drawn over `colour`, and its glossy
/// reflection in the board.
#[inline(always)]
fn skyline(colour: Vec3Simd, a: F, b: F, row: &Row, pixel: F, time: f32) -> Vec3Simd {
    let base = HORIZON + row.drop;
    let ly = b - s(base);
    let tallest = 0.7 * row.scale;
    let in_strip = ly.simd_gt(s(-tallest * 0.5)) & ly.simd_lt(s(tallest));
    if !in_strip.any() {
        return colour;
    }

    let shift = row.seed * 0.37;
    let cell = ((a + s(shift)) / s(row.pitch)).floor();
    let present = select(
        hash2(cell + s(9.0), s(row.seed)).simd_gt(s(0.1)),
        s(1.0),
        s(0.0),
    );
    let kind = (hash2(cell, s(row.seed)) * s(KINDS as f32)).floor();
    let tall = hash2(cell, s(row.seed + 4.0));
    let random = hash2(cell + s(2.0), s(row.seed + 8.0));
    let narrow = s(row.pitch * 0.5) * (s(0.55) + s(0.45) * hash2(cell, s(row.seed + 2.0)));
    let heights = per_kind(kind, [0.1, 0.32, 0.2, 0.08, 0.32, 0.06]);
    let spans = per_kind(kind, [0.07, 0.2, 0.14, 0.05, 0.2, 0.04]);
    let shape = Shape {
        kind,
        half: narrow * per_kind(kind, [1.5, 0.8, 0.42, 0.55, 1.2, 0.2]),
        height: (heights + spans * tall) * s(row.scale),
        random,
    };
    let lx = a + s(shift) - (cell + s(0.5)) * s(row.pitch);
    let edge = s(row.blur) + pixel;

    let reflected = component(lx, -ly, &shape, edge * s(3.0), time);
    let under =
        smoothstep(0.0, 0.01, -ly) * (s(1.0) - smoothstep(0.0, 0.5, -ly / shape.height)) * present;
    let mut result = mix3(
        colour,
        reflected.colour * s(0.5),
        reflected.mask * under * s(0.22),
    );

    let part = component(lx, ly, &shape, edge, time);
    let ambient = s(0.55) + s(0.45) * clamp01(ly / shape.height);
    let faded = mix3(part.colour * ambient, haze_colour(), s(row.haze));
    result = mix3(result, faded, part.mask * present);
    result + part.glow * present
}

pub struct PcbCity {
    look: Look,
    pipeline: SimdPipeline<ScreenVertexShader, Board>,
}

impl PcbCity {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.05, 0.3, 0.16),
                die: Vec3::new(0.02, 0.025, 0.03),
                gold: Vec3::new(0.95, 0.72, 0.35),
                streams: Vec3::new(0.5, 1.0, 0.7),
                key: Vec3::new(-0.35, 0.5, 0.8).normalise(),
                key_colour: Vec3::new(0.9, 1.0, 0.95),
                sky: Vec3::new(0.1, 0.3, 0.22),
                ground: Vec3::new(0.03, 0.15, 0.09),
                softbox: Vec3::ZERO,
            },
            pipeline: SimdPipeline::new(ScreenVertexShader, Board)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for PcbCity {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Chip],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::RelayClick),
                0.0,
                0.45,
                0.16,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        scene.chip.draw(pass, &scene.view, &self.look);
        let uniforms = Uniforms {
            view: scene.view.clone(),
            time: scene.local_time,
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.pipeline, (), |_| uniforms.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.55,
            vignette: 0.4,
            grain: 0.012,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
