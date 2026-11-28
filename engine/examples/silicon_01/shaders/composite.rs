//! The final pass onto the window: tone mapping and finishing, flashes, and the screen's tiles
//! powering on and off.

use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd};
use crate::simd::{
    F, aces, clamp01, fract, from_colour, ge, gt, hash2, hex_digit, lt, mix3, palette, rgb, s,
    select, select3, smoothstep, to_colour,
};

/// The renderer's tile size, in scene pixels, for a scene up to 720 pixels high: the grid the
/// power-on and shutdown draw (see [`tile_size`]).
pub const TILE: f32 = 64.0;
/// Worker threads the tile effects assign tiles to, in turn.
pub const THREADS: usize = 16;
/// How long one tile takes to power on or off, in seconds.
pub const TILE_LIFE: f32 = 0.8;
/// Seconds between waves of tiles during power-on, one wave per `THREADS` tiles.
const WAVE: f32 = 0.3;
/// How far through its life a tile shrinking at shutdown leaves only a dot.
const DOT_AT: f32 = 0.83;
/// Hex digit cells across and down one tile of the rain.
const RAIN_CELLS: f32 = 4.0;

#[derive(Clone, Copy, PartialEq)]
pub enum Tiles {
    /// Every tile shows the finished frame.
    Done,
    /// Tiles light up wave by wave; `elapsed` is the time since the first started.
    PowerOn { elapsed: f32 },
    /// Tiles switch off from `centre` (texture coordinates) outwards.
    Shutdown { elapsed: f32, centre: Vec2 },
}

/// How the finished image is graded: a theme may choose its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    pub exposure: f32,
    pub bloom: f32,
    /// How much the corners darken, 0 to 1.
    pub vignette: f32,
    /// Film grain amplitude.
    pub grain: f32,
    /// Chromatic aberration at the edge of the screen, in texture coordinates.
    pub aberration: f32,
    /// 0 keeps the colour, 1 is greyscale, negative values boost saturation.
    pub desaturate: f32,
}

impl Default for Grade {
    /// The studio grade: restrained bloom, a light vignette and barely visible grain.
    fn default() -> Self {
        Self {
            exposure: 1.0,
            bloom: 0.35,
            vignette: 0.35,
            grain: 0.012,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct CompositeUniforms {
    /// The HDR scene, with view distance in alpha.
    pub scene: Arc<TextureSampler>,
    pub bloom: Arc<TextureSampler>,
    pub time: f32,
    pub grade: Grade,
    /// White added on top, for impacts.
    pub flash: f32,
    /// A zoom blur towards the centre of the screen: how far the furthest sample is pulled in,
    /// as a fraction of the distance to the centre. 0 for none.
    pub blur: f32,
    /// 0 for black, 1 for the full image.
    pub brightness: f32,
    pub tiles: Tiles,
    /// The CRT power-off: 0 for none, 1 for a dark screen.
    pub collapse: f32,
}

pub struct CompositeShader;

impl FragmentShaderSimd<ScreenVaryings> for CompositeShader {
    type Uniforms = CompositeUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &CompositeUniforms) -> ColourSimd {
        let mut x = v.ndc()[0] * s(0.5) + s(0.5);
        let mut y = v.ndc()[1] * s(0.5) + s(0.5);

        // CRT power-off: the picture squashes to a line, then the line to a dot.
        let mut crt = s(1.0);
        if u.collapse > 0.0 {
            let squash_y = (1.0 - u.collapse * 2.0).max(0.004);
            let squash_x = (1.0 - (u.collapse - 0.5).max(0.0) * 2.0).max(0.003);
            x = (x - s(0.5)) / s(squash_x) + s(0.5);
            y = (y - s(0.5)) / s(squash_y) + s(0.5);
            let inside = lt((x - s(0.5)).abs(), s(0.5)) & lt((y - s(0.5)).abs(), s(0.5));
            crt = inside.select(s(1.0 + u.collapse * 3.0), s(0.0));
        }

        let mut colour = match u.tiles {
            Tiles::Done => graded(u, x, y),
            Tiles::PowerOn { elapsed } => power_on(u, x, y, elapsed),
            Tiles::Shutdown { elapsed, centre } => shutdown(u, x, y, elapsed, centre),
        };

        colour = mix3(colour, rgb(1.0, 1.0, 1.0), s(u.flash.min(1.0)));
        to_colour(colour * s(u.brightness) * crt, s(1.0))
    }
}

/// The scene with bloom, tone mapped and finished with the grade.
#[inline(always)]
pub fn graded(u: &CompositeUniforms, x: F, y: F) -> Vec3Simd {
    let g = &u.grade;
    let (cx, cy) = (x - s(0.5), y - s(0.5));
    let radius2 = cx * cx + cy * cy;
    let centre = if u.blur > 0.0 {
        zoom_blurred(u, x, y)
    } else {
        u.scene.sample_linear_clamp_simd([x, y])
    };
    let (red, blue) = if g.aberration > 0.0 {
        let shift = s(g.aberration) * radius2 * s(4.0);
        (
            u.scene
                .sample_linear_clamp_simd([x + cx * shift, y + cy * shift])
                .r,
            u.scene
                .sample_linear_clamp_simd([x - cx * shift, y - cy * shift])
                .b,
        )
    } else {
        (centre.r, centre.b)
    };
    let bloom = from_colour(u.bloom.sample_linear_clamp_simd([x, y]));
    let hdr = (Vec3Simd::new(red, centre.g, blue) + bloom * s(g.bloom)) * s(g.exposure);
    let mut mapped = Vec3Simd::new(aces(hdr.x), aces(hdr.y), aces(hdr.z));
    if g.desaturate != 0.0 {
        let luma = mapped.x * s(0.2126) + mapped.y * s(0.7152) + mapped.z * s(0.0722);
        mapped = mix3(mapped, Vec3Simd::new(luma, luma, luma), s(g.desaturate));
    }
    let vignette = s(1.0) - smoothstep(0.15, 0.7, radius2) * s(g.vignette);
    let frame = (u.time * 24.0).floor();
    let grain =
        (hash2(x * s(1931.0) + s(frame * 7.13 % 977.0), y * s(1373.0)) - s(0.5)) * s(g.grain);
    mapped * vignette + Vec3Simd::new(grain, grain, grain)
}

/// A tile of the grid: its position, index and top left corner in texture coordinates.
struct Tile {
    column: F,
    row: F,
    index: F,
    x0: F,
    y0: F,
}

/// The scene averaged along the line from `(x, y)` towards the centre of the screen.
#[inline(always)]
fn zoom_blurred(u: &CompositeUniforms, x: F, y: F) -> ColourSimd {
    const SAMPLES: usize = 10;
    let mut sum = Vec3Simd::new(s(0.0), s(0.0), s(0.0));
    let mut alpha = s(0.0);
    for i in 0..SAMPLES {
        let pull = s(1.0 - u.blur * i as f32 / (SAMPLES - 1) as f32);
        let c = u
            .scene
            .sample_linear_clamp_simd([(x - s(0.5)) * pull + s(0.5), (y - s(0.5)) * pull + s(0.5)]);
        sum = sum + from_colour(c);
        alpha += c.a;
    }
    let n = s(1.0 / SAMPLES as f32);
    to_colour(sum * n, alpha * n)
}

/// The side of a drawn tile, in scene pixels. Scenes taller than 720 pixels scale it up, keeping
/// the number of tiles, which the power-on's waves are timed for.
#[inline(always)]
fn tile_size(u: &CompositeUniforms) -> f32 {
    TILE * (u.scene.height() as f32 / 720.0).max(1.0)
}

/// Which tile (`x`, `y`) falls in, and the position within it from its top left in [0, 1]².
#[inline(always)]
fn tile_at(u: &CompositeUniforms, x: F, y: F) -> (Tile, F, F) {
    let (w, h) = (u.scene.width() as f32, u.scene.height() as f32);
    let side = tile_size(u);
    let (px, py) = (x * s(w), (s(1.0) - y) * s(h));
    let (column, row) = ((px / s(side)).floor(), (py / s(side)).floor());
    let columns = (w / side).ceil();
    let tile = Tile {
        column,
        row,
        index: row * s(columns) + column,
        x0: column * s(side / w),
        y0: s(1.0) - row * s(side / h),
    };
    (tile, fract(px / s(side)), fract(py / s(side)))
}

/// The colour of the worker thread that rasterises tile `index`: a shade of the accent green, or
/// gold for about one thread in five.
#[inline(always)]
fn thread_colour(index: F) -> Vec3Simd {
    let threads = s(THREADS as f32);
    let thread = index - threads * (index / threads).floor();
    // Golden-ratio spacing keeps neighbouring threads' shades apart.
    let shade = fract(thread * s(0.618_034));
    let green = mix3(rgb(0.18, 0.71, 0.54), rgb(0.43, 0.91, 0.72), shade);
    select3(gt(shade, s(0.8)), rgb(0.94, 0.84, 0.54), green) * s(1.4)
}

/// How brightly the border of a tile is drawn at `(lx, ly)` in it.
#[inline(always)]
fn border(lx: F, ly: F) -> F {
    let d = lx.fast_min(s(1.0) - lx).fast_min(ly).fast_min(s(1.0) - ly);
    s(1.0) - smoothstep(0.0, 0.035, d)
}

/// When the tile `index` is taken by its worker thread, in seconds into the power-on: a wave of
/// [`THREADS`] tiles at a time.
#[inline(always)]
fn power_on_start(index: F) -> F {
    (index / s(THREADS as f32)).floor() * s(WAVE) + hash2(index, s(3.7)) * s(0.25)
}

/// [`power_on_start`] for one tile, for timing its sound.
pub fn power_on_start_of(index: usize) -> f32 {
    power_on_start(s(index as f32)).to_array()[0]
}

/// When the tile `index`, whose middle is at `middle` in texture coordinates, starts to switch
/// off, in seconds into the shutdown: a wave spreading from `centre`.
#[inline(always)]
fn shutdown_start(index: F, middle: (F, F), centre: Vec2, aspect: f32) -> F {
    let dx = (middle.0 - s(centre.x)) * s(aspect);
    let dy = middle.1 - s(centre.y);
    (dx * dx + dy * dy).sqrt() * s(3.2) + hash2(index, s(9.1)) * s(0.35)
}

/// When the tile `index` of a `width` by `height` window, no more than 720 pixels high, shrinks
/// to its last dot, in seconds into the shutdown spreading from `centre`, for timing its sound.
pub fn shutdown_dot_of(index: usize, (width, height): (f32, f32), centre: Vec2) -> f32 {
    let columns = (width / TILE).ceil() as usize;
    let along = |cell: usize, size: f32| (cell as f32 + 0.5) * TILE / size;
    let middle = (
        s(along(index % columns, width)),
        s(1.0 - along(index / columns, height)),
    );
    let start = shutdown_start(s(index as f32), middle, centre, width / height);
    start.to_array()[0] + DOT_AT * TILE_LIFE
}

/// Tiles rendering wave by wave, as the renderer's worker threads would take them, black until
/// then. Once its thread takes it, each tile shows its border in its thread's colour and a rain
/// of hex digits, then a raster sweep steps it from the scene's edges to its shading to the
/// finished frame, glitching and pixelating less as it resolves. A few finished tiles glitch
/// again for a moment.
#[inline(always)]
fn power_on(u: &CompositeUniforms, x: F, y: F, elapsed: f32) -> Vec3Simd {
    let (tile, lx, ly) = tile_at(u, x, y);
    let age = (s(elapsed) - power_on_start(tile.index)) / s(TILE_LIFE);
    let tick = glitch_tick(u);
    let thread = thread_colour(tile.index);

    // Three stages, each swept in from the top of the tile; glitching flickers back to an
    // earlier one.
    let stage = (age * s(3.0)).floor().fast_min(s(3.0));
    let sweep = fract(age * s(3.0));
    let ahead = lt(ly, sweep);
    let mut shown = stage + ahead.select(s(1.0), s(0.0));

    let burst = lt(hash2(tile.index * s(5.3), tick), s(0.6)).select(s(1.0), s(0.0));
    let running = gt(age, s(0.0)) & lt(age, s(1.0));
    let replay = lt(hash2(tile.index, s(41.0)), s(0.3))
        & lt(
            (age - s(1.0) - hash2(tile.index, s(5.0)) * s(0.5)).abs(),
            s(0.09),
        );
    let strength = running.select(clamp01(s(1.0) - age) * (s(0.35) + burst * s(0.65)), s(0.0))
        + replay.select(s(0.8), s(0.0));
    let back = lt(hash2(tile.index + s(0.5), tick * s(1.7)), strength * s(0.6));
    let steps = select(gt(hash2(tile.index, tick), s(0.5)), s(2.0), s(1.0));
    shown = gt(shown, s(0.5)).select(back.select((shown - steps).fast_max(s(0.0)), shown), shown);

    let picture = |sx: F, sy: F| {
        let mut colour = rgb(0.0, 0.0, 0.0);
        if lt(shown, s(2.5)).any() && gt(shown, s(0.5)).any() {
            let (edges, shading) = raster_stages(u, sx, sy);
            colour = select3(lt(shown, s(1.5)), edges, shading);
        }
        if gt(shown, s(2.5)).any() {
            colour = select3(gt(shown, s(2.5)), graded(u, sx, sy), colour);
        }
        select3(lt(shown, s(0.5)), rgb(0.0, 0.0, 0.0), colour)
    };
    let glitching = gt(strength, s(0.0));
    let mut colour = picture(x, y);
    if glitching.any() {
        let glitched = glitch(u, &tile, lx, ly, strength, thread, picture);
        colour = select3(glitching, glitched, colour);
    }

    let scanline = (s(1.0) - smoothstep(0.0, 0.04, (ly - sweep).abs()))
        * lt(age, s(1.0)).select(s(1.0), s(0.0));
    let edge = border(lx, ly);
    let active = clamp01(s(1.0) - (age - s(1.0)) * s(3.0)) * gt(age, s(0.0)).select(s(1.0), s(0.0));
    colour = colour + thread * (edge * active * s(1.2) + scanline * active * s(0.8));

    // Nothing shows on a tile until it starts loading, so the loop opens on black.
    let rain_gain = gt(age, s(0.0)).select(
        clamp01(age * s(12.0)) * clamp01(s(1.0) - age) * s(1.3),
        s(0.0),
    );
    if gt(rain_gain, s(0.0)).any() {
        colour = colour + rain(u, &tile, lx, ly, thread) * rain_gain;
    }
    colour
}

/// What the raster stages show of the scene before it is finished: its edges in the accent green
/// on black, then its shading in silver-grey. Both come from the scene's colour, so they keep
/// their detail where its depth is flat, as across the die's face.
#[inline(always)]
fn raster_stages(u: &CompositeUniforms, x: F, y: F) -> (Vec3Simd, Vec3Simd) {
    let (dx, dy) = (1.0 / u.scene.width() as f32, 1.0 / u.scene.height() as f32);
    // Rec. 709 luma, compressed into [0, 1) with Reinhard's curve.
    let luma = |x: F, y: F| {
        let c = u.scene.sample_linear_clamp_simd([x, y]);
        let l = c.r * s(0.2126) + c.g * s(0.7152) + c.b * s(0.0722);
        l / (l + s(1.0))
    };
    let gx = luma(x + s(dx), y) - luma(x - s(dx), y);
    let gy = luma(x, y + s(dy)) - luma(x, y - s(dy));
    // Gain set by eye so the die's floorplan reads as fine line work (provisional).
    let edge = clamp01((gx * gx + gy * gy).sqrt() * s(4.0));
    (
        rgb(0.43, 0.91, 0.72) * edge,
        rgb(0.79, 0.80, 0.82) * luma(x, y),
    )
}

/// Tiles switching off in a wave from `centre`: each flares in its thread's colour, rains hex
/// digits, glitches hard, collapses to a dot and goes dark.
#[inline(always)]
fn shutdown(u: &CompositeUniforms, x: F, y: F, elapsed: f32, centre: Vec2) -> Vec3Simd {
    let (tile, lx, ly) = tile_at(u, x, y);
    let (w, h) = (u.scene.width() as f32, u.scene.height() as f32);
    let side = tile_size(u);
    let middle = (tile.x0 + s(0.5 * side / w), tile.y0 - s(0.5 * side / h));
    let age = (s(elapsed) - shutdown_start(tile.index, middle, centre, w / h)) / s(TILE_LIFE);
    let thread = thread_colour(tile.index);
    let tick = glitch_tick(u);

    // Collapse: the tile's picture shrinks towards its centre.
    let shrink = clamp01((age - s(0.5)) * s(3.0));
    let scale = (s(1.0) - shrink).fast_max(s(0.001));
    let (ox, oy) = (
        (lx - s(0.5)) / scale + s(0.5),
        (ly - s(0.5)) / scale + s(0.5),
    );
    let inside = lt((ox - s(0.5)).abs(), s(0.5)) & lt((oy - s(0.5)).abs(), s(0.5));
    let picture = |sx: F, sy: F| graded(u, sx, sy);
    let mut colour = picture(tile.x0 + ox * s(side / w), tile.y0 - oy * s(side / h));

    let burst = lt(hash2(tile.index * s(5.3), tick), s(0.7)).select(s(1.0), s(0.0));
    let window = smoothstep(0.0, 0.12, age) * (s(1.0) - smoothstep(0.5, 0.6, age));
    let strength = window * (s(0.9) + burst * s(0.5));
    let glitching = gt(strength, s(0.0));
    if glitching.any() {
        let glitched = glitch(u, &tile, ox, oy, strength, thread, picture);
        colour = select3(glitching, glitched, colour);
        colour = colour + rain(u, &tile, ox, oy, thread) * (window * s(1.6));
    }

    // Flare: the border lights up as the tile's thread lets it go.
    let flare = (s(1.0) - smoothstep(0.0, 0.6, age)) * gt(age, s(0.0)).select(s(1.0), s(0.0));
    let edge = border(lx, ly);
    colour = colour + thread * (edge * flare * s(1.6));

    // The last of the tile: a dot that fades out.
    let dot_d = ((lx - s(0.5)) * (lx - s(0.5)) + (ly - s(0.5)) * (ly - s(0.5))).sqrt();
    let dot = (s(1.0) - smoothstep(0.02, 0.08, dot_d))
        * clamp01(s(1.0) - (age - s(DOT_AT)) * s(6.0))
        * gt(age, s(DOT_AT)).select(s(1.0), s(0.0));
    colour = select3(inside, colour, rgb(0.0, 0.0, 0.0)) + thread * (dot * s(2.0));
    select3(gt(age, s(1.0)), thread * (dot * s(2.0)), colour)
}

/// The glitch clock: the frame number glitches change on, 12 a second.
#[inline(always)]
fn glitch_tick(u: &CompositeUniforms) -> F {
    s((u.time * 12.0).floor())
}

/// A tile's picture, corrupted in proportion to `strength` (about 0 to 1.5) at `(lx, ly)` in the
/// tile: rows slip sideways, the picture pixelates into chunky blocks, and random rows split
/// their colour channels, swap them, invert them, go monochrome in `tint` or flood with a
/// saturated hue; short blocks turn to one-bit noise. The corruption changes on every `glitch_tick`. `picture` gives the colour at a
/// texture coordinate; power-on and shutdown share this so they read as one family.
#[inline(always)]
fn glitch(
    u: &CompositeUniforms,
    tile: &Tile,
    lx: F,
    ly: F,
    strength: F,
    tint: Vec3Simd,
    picture: impl Fn(F, F) -> Vec3Simd,
) -> Vec3Simd {
    let (w, h) = (u.scene.width() as f32, u.scene.height() as f32);
    let tick = glitch_tick(u);
    let random = |salt: f32, a: F, b: F| {
        hash2(
            tile.index * s(13.7) + a + s(salt * 5.1),
            b + tick * s(3.1) + s(salt * 0.37),
        )
    };
    let zero = s(0.0);

    let slip_row = (ly * s(12.0)).floor();
    let slip = lt(
        random(1.0, slip_row, zero),
        (strength * s(0.5)).fast_min(s(0.9)),
    )
    .select(
        (random(2.0, slip_row, zero) - s(0.5)) * strength * s(0.7),
        zero,
    );
    let (mut sx, mut sy) = (lx + slip, ly);

    let blocky = lt(random(3.0, zero, zero), strength * s(1.5));
    let size = select(
        gt(strength, s(0.66)),
        s(8.0),
        select(gt(strength, s(0.33)), s(4.0), s(2.0)),
    );
    let snap = |l: F| ((l * s(TILE) / size).floor() + s(0.5)) * size / s(TILE);
    sx = blocky.select(snap(sx), sx);
    sy = blocky.select(snap(sy), sy);
    let side = tile_size(u);
    let (gx, gy) = (tile.x0 + sx * s(side / w), tile.y0 - sy * s(side / h));

    let chaos_row = (ly * s(16.0)).floor();
    let chaos = lt(random(4.0, chaos_row, zero), strength * s(0.9));
    let kind = random(5.0, chaos_row, zero);
    let split = chaos & lt(kind, s(0.3));
    let offset = split.select((random(6.0, chaos_row, zero) - s(0.5)) * s(0.1), zero);
    let mut colour = picture(gx, gy);
    if split.any() {
        colour = Vec3Simd::new(
            picture(gx + offset, gy).x,
            colour.y,
            picture(gx - offset, gy).z,
        );
    }
    let swapped = Vec3Simd::new(colour.y, colour.z, colour.x);
    let inverted = Vec3Simd::new(
        s(1.0) - clamp01(colour.x),
        s(1.0) - clamp01(colour.y),
        s(1.0) - clamp01(colour.z),
    ) * s(0.4)
        + tint * s(0.25);
    let luma = colour.x * s(0.2126) + colour.y * s(0.7152) + colour.z * s(0.0722);
    let mono = tint * (luma * s(1.5));
    // Saturated bars of any hue, the picture showing through them.
    let hue = random(8.0, chaos_row, zero);
    let vivid = palette(hue, Vec3::new(0.0, 0.33, 0.67)) * (s(0.35) + luma * s(2.0));
    let range = |from: f32, to: f32| chaos & ge(kind, s(from)) & lt(kind, s(to));
    colour = select3(range(0.3, 0.45), swapped, colour);
    colour = select3(range(0.45, 0.55), inverted, colour);
    colour = select3(range(0.55, 0.7), mono, colour);
    colour = select3(range(0.7, 1.0), vivid, colour);

    let noise_block = lt(
        random(7.0, (lx * s(8.0)).floor(), (ly * s(8.0)).floor()),
        strength * s(0.12),
    );
    let noise = random(8.0, (lx * s(32.0)).floor(), (ly * s(32.0)).floor());
    let bit = |c: F| gt(c + (noise - s(0.5)) * s(0.5), s(0.35)).select(s(1.0), s(0.0));
    let one_bit = Vec3Simd::new(bit(colour.x), bit(colour.y), bit(colour.z));
    select3(noise_block, mix3(one_bit, tint, s(0.3)), colour)
}

/// Columns of hex digits falling through the tile grid at `(lx, ly)` in `tile`, in `tint`: a
/// near-white head and a fading trail, the digits changing as they fall. Columns continue from
/// tile to tile.
#[inline(always)]
fn rain(u: &CompositeUniforms, tile: &Tile, lx: F, ly: F, tint: Vec3Simd) -> Vec3Simd {
    let (ux, uy) = (
        (tile.column + lx) * s(RAIN_CELLS),
        (tile.row + ly) * s(RAIN_CELLS),
    );
    let (cx, cy) = (ux.floor(), uy.floor());
    let speed = s(6.0) + hash2(cx, s(1.0)) * s(10.0);
    let period = s(18.0) + hash2(cx, s(2.0)) * s(22.0);
    let head = s(u.time) * speed + hash2(cx, s(3.0)) * period;
    let behind = head - cy;
    let behind = behind - period * (behind / period).floor();
    let trail = (-behind * s(0.35)).exp();
    if !gt(trail, s(0.03)).any() {
        return rgb(0.0, 0.0, 0.0);
    }
    let seed = hash2(
        cx * s(3.7) + cy * s(17.3),
        (s(u.time) * s(4.0) + hash2(cx, cy) * s(8.0)).floor(),
    );
    let digit = hex_digit(
        fract(ux) * s(2.0) - s(1.0),
        s(1.0) - fract(uy) * s(2.0),
        s(0.4),
        seed,
    );
    let heat = clamp01(s(1.0) - behind);
    mix3(tint, rgb(1.0, 1.0, 1.0), heat * s(0.8)) * (digit * trail * s(1.8))
}
