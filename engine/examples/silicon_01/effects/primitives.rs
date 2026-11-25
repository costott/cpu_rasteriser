//! The renderer's own primitives drawn as light: filled and wireframe triangles with barycentric
//! colours, 64×64 tile outlines, depth bands, normal arrows and hex digits, plus the glows and
//! rings that go with them. Each is a screen-facing quad whose fragment shader draws the shape,
//! so any number of them cost one draw.

use cpu_rasteriser::prelude::*;

use crate::shaders::View;
use crate::simd::{F, clamp01, cover, fract, hex_digit, mix3, rgb, s, smoothstep, to_colour, v3};

/// How big a sprite may look, as a half-width in pixels of a 720p frame: sprites flying past the
/// camera stop growing here.
const MAX_HALF_PIXELS: f32 = 40.0;
/// Below this half-width in pixels a sprite has nothing to show.
const MIN_HALF_PIXELS: f32 = 1.5;
/// Past this half-width in pixels of a 720p frame a ring's band is outside the frame, beyond its
/// half-diagonal of 734 pixels with room for the band's soft edge, so the ring is no longer drawn.
const MAX_RING_HALF_PIXELS: f32 = 800.0;

/// What a sprite draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A triangle filled with the three barycentric colours.
    Triangle,
    /// A triangle's edges.
    Wireframe,
    /// A tile's outline, with the grid of pixels inside faintly.
    Tile,
    /// A patch of depth, shown as bands.
    DepthBands,
    /// A surface normal: a shaft with a head.
    Normal,
    /// A hex digit in seven-segment strokes; the sprite's seed picks the digit.
    Hex,
    /// A soft radial glow.
    Glow,
    /// A thin ring at the quad's edge.
    Ring,
}

/// The kinds a stream of renderer primitives is drawn from, repeated in proportion.
pub const PRIMITIVES: [Kind; 16] = [
    Kind::Triangle,
    Kind::Triangle,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Wireframe,
    Kind::Tile,
    Kind::DepthBands,
    Kind::Tile,
    Kind::Normal,
    Kind::Normal,
    Kind::Hex,
    Kind::Hex,
    Kind::Hex,
    Kind::Hex,
];

/// One corner of a sprite's quad, `corner` running from (-1, -1) to (1, 1); `data` is the
/// effect's own per-sprite data.
#[derive(Clone, Copy)]
pub struct SpriteVertex<D> {
    pub corner: Vec2,
    pub data: D,
}

/// The four corners of a sprite's quad, with `data` on each.
pub fn sprite_corners<D: Copy>(data: D) -> [SpriteVertex<D>; 4] {
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(x, y)| SpriteVertex {
        corner: Vec2::new(x, y),
        data,
    })
}

/// A sprite placed in the world.
#[derive(Clone, Copy)]
pub struct Sprite {
    pub centre: Vec3,
    /// Half its width, in world units.
    pub size: f32,
    /// Its turn about the view direction, in radians.
    pub angle: f32,
    pub kind: Kind,
    /// Picks the hex digit and varies the bands.
    pub seed: f32,
    /// The light's colour and strength: the first of the three barycentric colours, and the
    /// colour of everything else.
    pub colour: Vec3,
    /// 0 to 1.
    pub fade: f32,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct SpriteVaryings {
    local: Vec2,
    /// The kind, the seed, and the fade.
    info: Vec3,
    colour: Vec3,
    /// How much of the quad one pixel covers, along one axis.
    pixel: f32,
}

/// Expands `sprite` into the screen-facing quad's `corner`. A sprite that is invisible or too
/// small to see goes off screen.
pub fn expand(view: &View, sprite: &Sprite, corner: Vec2) -> (Vec4, SpriteVaryings) {
    let to_centre = sprite.centre - view.eye;
    let distance = to_centre.length();
    let pixel = view.tan_half_fov * 2.0 / 720.0 * distance;
    let size = if matches!(sprite.kind, Kind::Glow | Kind::Ring) {
        sprite.size
    } else {
        sprite.size.min(MAX_HALF_PIXELS * pixel)
    };
    let past_frame = sprite.kind == Kind::Ring && size > MAX_RING_HALF_PIXELS * pixel;
    if sprite.fade < 0.004
        || size < MIN_HALF_PIXELS * pixel
        || past_frame
        || to_centre.dot(view.forward) < 0.1
    {
        return hidden();
    }
    let (sin, cos) = sprite.angle.sin_cos();
    let right = view.right * cos + view.up * sin;
    let up = view.up * cos - view.right * sin;
    let world = sprite.centre + right * (corner.x * size) + up * (corner.y * size);
    (
        view.view_projection * world.to_point4(),
        SpriteVaryings {
            local: corner,
            info: Vec3::new(sprite.kind as u8 as f32, sprite.seed, sprite.fade),
            colour: sprite.colour,
            pixel: pixel / size,
        },
    )
}

/// A sprite that isn't there: a point outside the view.
pub fn hidden() -> (Vec4, SpriteVaryings) {
    (
        Vec4::new(3.0, 3.0, 0.5, 1.0),
        SpriteVaryings {
            local: Vec2::new(0.0, 0.0),
            info: Vec3::ZERO,
            colour: Vec3::ZERO,
            pixel: 1.0,
        },
    )
}

pub struct SpriteShader;

impl FragmentShaderSimd<SpriteVaryings> for SpriteShader {
    type Uniforms = ();

    #[inline(always)]
    fn shade_simd(&self, v: SpriteVaryingsSimd, _: &()) -> ColourSimd {
        let [x, y] = v.local;
        let aa = v.pixel;
        let tint = v3(v.colour);
        // Rounded, because interpolation across a large quad can leave the constant kind a hair
        // below its value, which truncation would turn into the previous kind.
        let kind = v.info[0].to_array()[0].round() as u8;
        let seed = v.info[1];
        let light = match kind {
            k if k == Kind::Triangle as u8 => triangle(x, y, aa, tint, false),
            k if k == Kind::Wireframe as u8 => triangle(x, y, aa, tint, true),
            k if k == Kind::Tile as u8 => tile(x, y, aa, tint),
            k if k == Kind::DepthBands as u8 => depth_bands(x, y, aa, seed, tint),
            k if k == Kind::Normal as u8 => normal(x, y, aa, tint),
            k if k == Kind::Hex as u8 => hex(x, y, aa, seed, tint),
            k if k == Kind::Glow as u8 => glow(x, y, tint),
            _ => ring(x, y, aa, tint),
        };
        to_colour(light * v.info[2], s(0.0))
    }
}

#[inline(always)]
fn white() -> Vec3Simd {
    rgb(1.0, 1.0, 1.0)
}

/// The three barycentric coordinates of (`x`, `y`) in the equilateral triangle with corners
/// (0, 0.9), (-0.78, -0.45) and (0.78, -0.45).
#[inline(always)]
fn barycentric(x: F, y: F) -> [F; 3] {
    let dy = y - s(0.9);
    let d20 = s(-0.78) * x - s(1.35) * dy;
    let d21 = s(0.78) * x - s(1.35) * dy;
    let b = (s(2.4309) * d20 - s(1.2141) * d21) / s(4.4352);
    let c = (s(2.4309) * d21 - s(1.2141) * d20) / s(4.4352);
    [s(1.0) - b - c, b, c]
}

/// The altitude of that triangle: a barycentric coordinate times it is the distance to its edge.
const ALTITUDE: f32 = 1.35;

#[inline(always)]
fn triangle(x: F, y: F, aa: F, tint: Vec3Simd, wire_only: bool) -> Vec3Simd {
    let [a, b, c] = barycentric(x, y);
    let inside = a.fast_min(b).fast_min(c) * s(ALTITUDE);
    let edge = cover(-inside, aa);
    let outline = (s(1.0) - smoothstep_px(inside, aa, 2.2)) * edge;
    if wire_only {
        return tint * outline;
    }
    let green = rgb(0.2, 0.8, 0.6);
    let gold = rgb(0.94, 0.84, 0.54);
    let fill = tint * a + green * (b * s(1.3)) + gold * (c * s(1.2));
    fill * (edge * s(0.32)) + mix3(tint, white(), s(0.3)) * (outline * s(0.95))
}

/// 0 up to `pixels` pixels (of size `aa`), 1 beyond.
#[inline(always)]
fn smoothstep_px(d: F, aa: F, pixels: f32) -> F {
    clamp01(d / (aa * s(pixels)))
}

#[inline(always)]
fn tile(x: F, y: F, aa: F, tint: Vec3Simd) -> Vec3Simd {
    let inside = s(0.9) - x.abs().fast_max(y.abs());
    let outline = (s(1.0) - smoothstep_px(inside.abs(), aa, 1.6)) * cover(-inside, aa * s(2.0));
    let cell = |c: F| {
        let f = fract(c * s(4.0));
        f.fast_min(s(1.0) - f)
    };
    let grid =
        (s(1.0) - smoothstep_px(cell(x).fast_min(cell(y)) * s(0.25), aa, 1.2)) * cover(-inside, aa);
    tint * (outline * s(1.0) + grid * s(0.1))
}

#[inline(always)]
fn depth_bands(x: F, y: F, aa: F, seed: F, tint: Vec3Simd) -> Vec3Simd {
    let inside = s(0.9) - x.abs().fast_max(y.abs());
    let edge = cover(-inside, aa * s(2.0));
    let t = (x * s(0.6) + y * s(0.8)) * s(0.5) + s(0.5);
    let bands = t * s(5.0) + seed * s(3.0);
    let level = bands.floor();
    let within = fract(bands);
    let rise = smoothstep_px(within, aa * s(0.2), 1.0);
    let step = (level * s(0.2)).fast_min(s(1.0));
    let band = rise * (s(0.15) + step * s(0.45));
    mix3(tint * band, white() * band, step * s(0.25)) * edge
}

#[inline(always)]
fn normal(x: F, y: F, aa: F, tint: Vec3Simd) -> Vec3Simd {
    let shaft = (x.abs() - s(0.05))
        .fast_max(s(-0.85) - y)
        .fast_max(y - s(0.25));
    let head_edge = (x.abs() * s(0.7) - (s(0.9) - y) * s(0.3)) * s(0.919);
    let head = head_edge.fast_max(s(0.2) - y);
    let ring = ((x * x + (y + s(0.85)) * (y + s(0.85))).sqrt() - s(0.12)).abs() - s(0.02);
    let d = shaft.fast_min(head).fast_min(ring);
    tint * cover(d, aa)
}

#[inline(always)]
fn hex(x: F, y: F, aa: F, seed: F, tint: Vec3Simd) -> Vec3Simd {
    tint * hex_digit(x, y, aa, seed)
}

#[inline(always)]
fn glow(x: F, y: F, tint: Vec3Simd) -> Vec3Simd {
    let r2 = x * x + y * y;
    let falloff = (-r2 * s(7.0)).exp() * (s(1.0) - smoothstep(0.6, 1.0, r2.sqrt()));
    let core = (-r2 * s(60.0)).exp();
    tint * falloff + white() * (core * s(0.6))
}

#[inline(always)]
fn ring(x: F, y: F, aa: F, tint: Vec3Simd) -> Vec3Simd {
    let r = (x * x + y * y).sqrt();
    let d = (r - s(0.92)).abs();
    let line = (-(d / (aa * s(1.4))) * (d / (aa * s(1.4)))).exp();
    let halo = (-d * s(30.0)).exp() * s(0.25) * (s(1.0) - smoothstep(0.92, 1.0, r));
    mix3(tint, white(), s(0.5)) * line + tint * halo
}
