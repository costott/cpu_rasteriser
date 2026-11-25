//! snow-globe: a glass dome on a walnut plinth, in a dark room of warm bokeh. Inside, a snowy
//! village under a moon, lit windows glowing, and snow that never stops falling; the glass
//! magnifies and bends it, and catches the room's lights along its rim.
//!
//! One full-screen shader in screen units (the screen is 2 high, so a pixel is `2 / height`):
//! the room, the village inside the dome, the dome's lens and reflections, then the plinth and
//! table in front of the dome's foot.

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::ViewRef;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::simd::{F, clamp01, fract, hash2, mix3, mul3, rgb, s, select, smoothstep, to_colour};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::noise::{fbm, value};
use crate::themes::kit::post::expand;
use crate::themes::{Scene, Stage};

/// The dome: its centre and radius, in screen units.
const DOME: (f32, f32, f32) = (0.0, 0.08, 0.82);
/// Where the snow inside the dome and the plinth's top edge meet: the world's horizon line.
const GROUND: f32 = -0.5;
/// The distance the whole scene reports as its view distance.
const SCENE_DISTANCE: f32 = 30.0;

/// A house in the village: where it stands, its size and the roof's colour.
struct House {
    x: f32,
    width: f32,
    height: f32,
    roof: (f32, f32, f32),
}

const HOUSES: [House; 5] = [
    House {
        x: -0.52,
        width: 0.26,
        height: 0.18,
        roof: (0.62, 0.2, 0.16),
    },
    House {
        x: -0.26,
        width: 0.17,
        height: 0.13,
        roof: (0.22, 0.32, 0.5),
    },
    House {
        x: 0.22,
        width: 0.19,
        height: 0.15,
        roof: (0.5, 0.3, 0.16),
    },
    House {
        x: 0.5,
        width: 0.27,
        height: 0.19,
        roof: (0.2, 0.36, 0.34),
    },
    House {
        x: 0.0,
        width: 0.12,
        height: 0.09,
        roof: (0.55, 0.22, 0.2),
    },
];

/// A snow-laden fir: where it stands (its base height comes from the ground), its height and width.
const FIRS: [(f32, f32, f32); 9] = [
    (-0.68, 0.42, 0.11),
    (-0.38, 0.26, 0.075),
    (-0.13, 0.17, 0.055),
    (0.1, 0.2, 0.06),
    (0.35, 0.3, 0.085),
    (0.68, 0.46, 0.12),
    (-0.58, 0.2, 0.06),
    (0.6, 0.18, 0.055),
    (0.04, 0.12, 0.04),
];

#[derive(Clone)]
struct Uniforms {
    view: ViewRef,
    time: f32,
}

struct Globe;

impl FragmentShaderSimd<ScreenVaryings> for Globe {
    type Uniforms = Uniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &Uniforms) -> ColourSimd {
        let view = &*u.view;
        let [nx, ny] = v.ndc();
        let (a, b) = (nx * s(view.aspect), ny);
        let pixel = s(view.pixel_angle / view.tan_half_fov);
        let time = u.time;

        let mut colour = room(a, b, time);
        let (glass, cover) = dome(a, b, pixel, time);
        colour = mix3(colour, glass, cover);
        colour = stand(colour, a, b, pixel);
        to_colour(expand(colour), s(SCENE_DISTANCE))
    }
}

/// 1 inside a shape whose signed distance is `d`, fading across one pixel.
#[inline(always)]
fn coverage(d: F, pixel: F) -> F {
    clamp01(s(0.5) - d / pixel)
}

/// The dark room behind the globe: a warm-to-cool gradient and out-of-focus lights.
#[inline(always)]
fn room(a: F, b: F, time: f32) -> Vec3Simd {
    let base = mix3(
        rgb(0.16, 0.09, 0.06),
        rgb(0.03, 0.05, 0.09),
        smoothstep(-0.8, 0.9, b),
    );
    let mut colour = base;
    for (layer, (size, drift)) in [(0.34, 0.01), (0.2, 0.02)].into_iter().enumerate() {
        let (u, w) = (
            a / s(size) + s(time * drift + layer as f32 * 5.0),
            b / s(size) + s(layer as f32 * 3.0),
        );
        let (cu, cv) = (u.floor(), w.floor());
        let present = hash2(cu, cv + s(11.0 * layer as f32));
        let (jx, jy) = (
            hash2(cu, cv * s(1.3)) - s(0.5),
            hash2(cv, cu * s(1.7)) - s(0.5),
        );
        let (dx, dy) = (
            fract(u) - s(0.5) - jx * s(0.4),
            fract(w) - s(0.5) - jy * s(0.4),
        );
        let disc = (s(1.0) - smoothstep(0.18, 0.3, (dx * dx + dy * dy).sqrt()))
            * select(present.simd_gt(s(0.55)), s(1.0), s(0.0));
        let warm = hash2(cu * s(2.1), cv).simd_gt(s(0.35));
        let lamp = crate::simd::select3(warm, rgb(1.0, 0.62, 0.28), rgb(0.35, 0.55, 1.0));
        colour = colour + lamp * disc * s(0.13);
    }
    colour
}

/// The dome: what shows through it and how opaque the glass is there, in the dome's own disc.
#[inline(always)]
fn dome(a: F, b: F, pixel: F, time: f32) -> (Vec3Simd, F) {
    let (cx, cy, radius) = DOME;
    let (px, py) = (a - s(cx), b - s(cy));
    let r = (px * px + py * py).sqrt() / s(radius);
    let cover = coverage((r - s(1.0)) * s(radius), pixel);

    let lens = s(0.82) + s(0.34) * r * r;
    let (x, y) = (s(cx) + px * lens, s(cy) + py * lens);
    let mut colour = interior(x, y, pixel, time);

    colour = colour * (s(1.0) - r * r * r * r * s(0.45));
    let fresnel = r * r * r * r * r;
    colour = colour + rgb(0.45, 0.6, 0.85) * fresnel * s(0.28);

    let angle = py.atan2(px);
    let arc = (s(1.0) - smoothstep(0.0, 0.045, (r - s(0.9)).abs()))
        * smoothstep(1.7, 2.1, angle)
        * (s(1.0) - smoothstep(2.6, 3.0, angle));
    colour = colour + rgb(0.9, 0.95, 1.0) * arc * s(0.5);
    let (wx, wy) = ((px + s(0.36)) / s(0.1), (py - s(0.4)) / s(0.2));
    let window = s(1.0) - smoothstep(0.75, 1.0, (wx * wx + wy * wy).sqrt());
    colour = colour + rgb(0.85, 0.92, 1.0) * window * s(0.07);
    let rim = (s(1.0) - smoothstep(0.0, 0.012, (r - s(1.0)).abs() * s(radius))) * s(0.5);
    colour = colour + rgb(0.7, 0.8, 1.0) * rim;
    (colour, cover)
}

/// The village inside the dome, in the lens's sampling coordinates.
#[inline(always)]
fn interior(x: F, y: F, pixel: F, time: f32) -> Vec3Simd {
    let mut colour = sky(x, y, pixel, time);

    let ridge = |scale: f32, offset: f32, base: f32, amplitude: f32| {
        s(base)
            + s(amplitude) * (x * s(scale) + s(offset)).sin()
            + s(amplitude * 0.45) * (x * s(scale * 2.7) + s(offset * 2.0)).sin()
    };
    let far = ridge(1.6, 0.4, GROUND + 0.2, 0.05);
    colour = mix3(
        colour,
        rgb(0.34, 0.42, 0.62) * (s(0.8) + (y - far) * s(0.8)),
        coverage(y - far, pixel),
    );
    let near = ridge(2.3, 2.1, GROUND + 0.06, 0.025);
    colour = mix3(colour, rgb(0.6, 0.68, 0.86), coverage(y - near, pixel));

    for &tree in &FIRS {
        colour = fir(colour, x, y, tree, pixel);
    }

    let ground =
        s(GROUND + 0.02) + s(0.02) * (x * s(3.1) + s(1.0)).sin() + s(0.012) * (x * s(8.7)).sin();
    let sparkle = {
        let (u, w) = (x * s(160.0), y * s(160.0));
        let cell = hash2(u.floor(), w.floor());
        let twinkle = (s(time * 3.0) + cell * s(40.0)).sin() * s(0.5) + s(0.5);
        select(cell.simd_gt(s(0.985)), twinkle, s(0.0))
    };
    let depth = clamp01((ground - y) / s(0.15));
    let snow = mix3(rgb(0.8, 0.86, 0.98), rgb(0.5, 0.6, 0.82), depth)
        * (s(0.95) + (fbm(x * s(9.0), y * s(9.0), 3) - s(0.5)) * s(0.3));
    let snow = snow + rgb(1.0, 1.0, 1.0) * sparkle * s(0.6);
    let mut glow = rgb(0.0, 0.0, 0.0);
    for house in &HOUSES {
        let (dx, dy) = ((x - s(house.x)) / s(0.2), (y - s(GROUND + 0.06)) / s(0.16));
        glow =
            glow + rgb(1.0, 0.62, 0.25) * (s(1.0) / (s(1.0) + dx * dx + dy * dy * s(1.5))) * s(0.3);
    }
    let lit_snow = mul3(snow, rgb(1.0, 1.0, 1.0) + glow);
    colour = mix3(colour, lit_snow, coverage(y - ground, pixel));

    for house in &HOUSES {
        colour = cottage(colour, x, y, house, pixel, time);
    }

    colour + snowfall(x, y, time)
}

/// The sky: a deep blue fading to lavender at the horizon, stars, and the moon with its halo.
#[inline(always)]
fn sky(x: F, y: F, pixel: F, time: f32) -> Vec3Simd {
    let height = smoothstep(GROUND, 0.9, y);
    let mut colour = mix3(rgb(0.6, 0.5, 0.66), rgb(0.06, 0.09, 0.24), height.sqrt());
    let (u, w) = (x * s(90.0), y * s(90.0));
    let cell = hash2(u.floor(), w.floor());
    let blink = (s(time * 2.0) + cell * s(30.0)).sin() * s(0.25) + s(0.75);
    colour = colour + rgb(1.0, 0.95, 0.85) * select(cell.simd_gt(s(0.975)), blink, s(0.0)) * height;
    let (mx, my) = (x - s(0.38), y - s(0.5));
    let moon_distance = (mx * mx + my * my).sqrt();
    let halo = s(1.0) / (s(1.0) + moon_distance * moon_distance * s(180.0));
    colour = colour + rgb(0.5, 0.55, 0.75) * halo * s(0.35);
    let moon = coverage(moon_distance - s(0.06), pixel);
    let crater = fbm(mx * s(40.0), my * s(40.0), 3);
    mix3(
        colour,
        rgb(1.0, 0.97, 0.86) * (s(0.88) + crater * s(0.15)),
        moon,
    )
}

/// A snowy fir at (`cx`, `base`): dark needles with snow settled on each tier.
#[inline(always)]
fn fir(colour: Vec3Simd, x: F, y: F, (cx, height, width): (f32, f32, f32), pixel: F) -> Vec3Simd {
    let base = if height < 0.22 {
        GROUND + 0.04
    } else {
        GROUND + 0.01
    };
    let q = clamp01((y - s(base)) / s(height));
    let tier = fract(q * s(4.0));
    let half = s(width) * (s(1.0) - q * s(0.8)) * (s(1.0) - tier * s(0.55));
    let across = (x - s(cx)).abs() - half;
    let along = (s(base) - y).fast_max(y - s(base + height));
    let d = across.fast_max(along);
    let needles = rgb(0.03, 0.13, 0.12) * (s(0.7) + q * s(0.5));
    let settled = smoothstep(0.35, 0.8, tier) * smoothstep(0.0, 0.12, s(1.0) - q);
    let tree = mix3(needles, rgb(0.82, 0.88, 1.0), settled * s(0.85));
    mix3(colour, tree, coverage(d, pixel))
}

/// A cottage: timber walls, a snow-capped roof, lit windows with their glow and a door.
#[inline(always)]
fn cottage(colour: Vec3Simd, x: F, y: F, house: &House, pixel: F, time: f32) -> Vec3Simd {
    let base = GROUND + 0.015;
    let (dx, dy) = (x - s(house.x), y - s(base + house.height * 0.5));
    let body = (dx.abs() - s(house.width * 0.5)).fast_max(dy.abs() - s(house.height * 0.5));
    let roof_height = house.width * 0.38;
    let top = base + house.height;
    let up = clamp01((y - s(top)) / s(roof_height));
    let roof_half = s(house.width * 0.5 + 0.012) * (s(1.0) - up);
    let roof = (dx.abs() - roof_half)
        .fast_max(s(top) - y)
        .fast_max(y - s(top + roof_height));

    let timber =
        rgb(0.34, 0.2, 0.13) * (s(0.8) + (fbm(x * s(120.0), y * s(30.0), 2) - s(0.5)) * s(0.6));
    let mut walls = timber;
    let (r, g, bl) = house.roof;
    let mut roofing = rgb(r, g, bl) * (s(0.8) + value(x * s(300.0), y * s(80.0)) * s(0.3));
    let snow_line = up + (value(x * s(40.0), s(house.x * 9.0)) - s(0.5)) * s(0.25);
    roofing = mix3(
        roofing,
        rgb(0.9, 0.93, 1.0),
        smoothstep(0.1, 0.35, snow_line),
    );

    let flicker = s(0.85 + 0.15 * (time * 6.0 + house.x * 20.0).sin());
    let window_width = house.width * 0.16;
    for side in [-1.0f32, 1.0] {
        let (wx, wy) = (
            dx - s(side * house.width * 0.22),
            dy - s(house.height * 0.08),
        );
        let pane = (wx.abs() - s(window_width)).fast_max(wy.abs() - s(window_width));
        let halo = (s(1.0) - smoothstep(0.0, 0.05, pane)) * s(0.45);
        walls = walls + rgb(1.0, 0.6, 0.2) * halo * flicker;
        walls = mix3(
            walls,
            rgb(1.0, 0.78, 0.38) * flicker * s(1.15),
            coverage(pane, pixel),
        );
    }
    let door = (dx.abs() - s(house.width * 0.07))
        .fast_max((dy + s(house.height * 0.5 - house.height * 0.2)).abs() - s(house.height * 0.2));
    walls = mix3(walls, rgb(0.14, 0.07, 0.05), coverage(door, pixel));

    let with_walls = mix3(colour, walls, coverage(body, pixel));
    mix3(with_walls, roofing, coverage(roof, pixel))
}

/// Three depths of falling flakes, swirling as if just shaken: added light.
#[inline(always)]
fn snowfall(x: F, y: F, time: f32) -> Vec3Simd {
    let mut total = s(0.0);
    for (layer, (size, speed, radius, brightness)) in [
        (0.16, 0.12, 0.2, 0.55),
        (0.09, 0.2, 0.17, 0.8),
        (0.05, 0.3, 0.15, 1.0),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = layer as f32 * 7.31;
        let swirl = (y * s(5.0) + s(time * 0.8 + seed)).sin() * s(0.02)
            + (y * s(2.3) - s(time * 0.5)).sin() * s(0.015);
        let (u, w) = (
            (x + swirl) / s(size) + s(seed),
            y / s(size) + s(time * speed / size),
        );
        let (cu, cv) = (u.floor(), w.floor());
        let present = hash2(cu, cv + s(seed));
        let (jx, jy) = (
            (hash2(cv, cu + s(seed)) - s(0.5)) * s(0.5),
            (hash2(cu + s(3.0), cv) - s(0.5)) * s(0.5),
        );
        let (dx, dy) = (fract(u) - s(0.5) - jx, fract(w) - s(0.5) - jy);
        let flake = s(1.0) - smoothstep(radius * 0.4, radius, (dx * dx + dy * dy).sqrt());
        total += flake * select(present.simd_gt(s(0.5)), s(brightness), s(0.0));
    }
    rgb(0.9, 0.95, 1.0) * clamp01(total) * s(0.9)
}

/// The walnut plinth the dome stands on and the table under it, drawn over the dome's foot.
#[inline(always)]
fn stand(colour: Vec3Simd, a: F, b: F, pixel: F) -> Vec3Simd {
    let grain = |scale: F| {
        let streaks = fbm(a * s(2.0), b * scale, 4);
        let fine = value(a * s(3.0), b * s(180.0));
        (streaks - s(0.5)) * s(0.6) + (fine - s(0.5)) * s(0.2)
    };
    let lamp = s(1.0)
        - smoothstep(
            0.0,
            1.8,
            ((a + s(0.5)) * s(0.7)).abs() + (b + s(0.7)).abs() * s(0.8),
        );

    let table_top = -0.82;
    let table = mul3(
        rgb(0.3, 0.17, 0.09) * (s(1.0) + grain(s(14.0))),
        rgb(0.5, 0.5, 0.5) + rgb(0.9, 0.8, 0.65) * lamp,
    );
    let mut result = mix3(colour, table, coverage(b - s(table_top), pixel));

    let flare = s(GROUND - table_top);
    let spread = s(0.7) + clamp01((s(GROUND) - b) / flare) * s(0.17);
    let plinth_distance = (a.abs() - spread)
        .fast_max(b - s(GROUND))
        .fast_max(s(table_top) - b);
    let shade = s(0.55) + (s(1.0) - (a / spread).abs()) * s(0.5) + lamp * s(0.3);
    let moulding = (smoothstep(0.0, 1.0, fract((b + s(0.5)) * s(14.0))) - s(0.5)) * s(0.12);
    let wood = rgb(0.22, 0.11, 0.065) * (s(1.0) + grain(s(25.0)) + moulding) * shade;
    let brass_band = s(1.0) - smoothstep(0.0, 0.004, (b - s(GROUND + 0.015)).abs() - s(0.012));
    let band =
        rgb(0.8, 0.58, 0.26) * (s(0.6) + (s(1.0) - clamp01((s(GROUND) - b) * s(60.0))) * s(0.7));
    let plinth = mix3(
        wood,
        band,
        brass_band * coverage((b - s(GROUND + 0.027)).abs() - s(0.027), pixel).fast_max(s(0.0)),
    );
    result = mix3(result, plinth, coverage(plinth_distance, pixel));
    result
}

pub struct SnowGlobe {
    look: Look,
    pipeline: SimdPipeline<ScreenVertexShader, Globe>,
}

impl SnowGlobe {
    pub fn new() -> Self {
        Self {
            look: Look {
                substrate: Vec3::new(0.3, 0.34, 0.45),
                die: Vec3::new(0.05, 0.1, 0.22),
                gold: Vec3::new(0.95, 0.72, 0.38),
                streams: Vec3::new(1.0, 0.7, 0.35),
                key: Vec3::new(-0.35, 0.5, 0.8).normalise(),
                key_colour: Vec3::new(0.8, 0.88, 1.1),
                sky: Vec3::new(0.2, 0.26, 0.45),
                ground: Vec3::new(0.2, 0.24, 0.32),
                softbox: Vec3::ZERO,
            },
            pipeline: SimdPipeline::new(ScreenVertexShader, Globe)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for SnowGlobe {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Glockenspiel],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Chime),
                0.0,
                0.8,
                0.12,
                0.2,
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
            bloom: 0.4,
            vignette: 0.35,
            grain: 0.012,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}
