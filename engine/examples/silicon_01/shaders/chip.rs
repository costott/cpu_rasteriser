//! The chip's surfaces: the die's floorplan, the package's metal, gold and markings, the light
//! streams, and the studio light they are seen in.
//!
//! Every pattern is filtered by the area a pixel covers on the surface ([`View::pixel_angle`]
//! times the distance, widened at grazing angles): stripes and grids are box filtered exactly,
//! and random per-cell detail fades to its average before it is small enough to shimmer. So the
//! die holds up from a full-screen macro shot down to a few hundred pixels across.

use std::f32::consts::TAU;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;

use crate::chip::{
    CAP, CAP_ROW, DIE, DIE_CORNER, DIE_TOP, HALF, LAND_CAPS, LANE_PITCH, Look, SEAL,
    SUBSTRATE_BOTTOM,
};
use crate::shaders::{Frame, View, ViewRef, scene_colour};
use crate::simd::{
    F, clamp01, fract, gt, hash2, lt, mix, mix3, mul3, reflect, rgb, s, select, smoothstep, splat3,
    v3,
};

/// The die's horizontal routing channels (the ring bus either side of the cache and the cache's
/// spine), as z in die units (the die spans -1 to 1).
pub const ROUTING_ROWS: [f32; 3] = [-0.26, 0.0, 0.26];
/// The die's vertical routing channels (between the cores, and beside the uncore), as x in die
/// units.
pub const ROUTING_COLUMNS: [f32; 4] = [-0.529, -0.17, 0.189, 0.555];

/// Which surface of the chip a triangle belongs to, carried in its vertices' red channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The substrate's flat top.
    Top,
    /// The substrate's polished bevels.
    Bevel,
    /// The substrate's side, where its layers show.
    Edge,
    /// The substrate's underside, with its contact pads.
    Bottom,
    /// The epoxy fillet the die sits in.
    Underfill,
    DieSide,
    DieChamfer,
    DieTop,
    /// The gold ring around the die.
    Seal,
    Capacitor,
    /// The capacitors on the underside.
    LandCapacitor,
}

impl Part {
    const ALL: [Part; 11] = [
        Part::Top,
        Part::Bevel,
        Part::Edge,
        Part::Bottom,
        Part::Underfill,
        Part::DieSide,
        Part::DieChamfer,
        Part::DieTop,
        Part::Seal,
        Part::Capacitor,
        Part::LandCapacitor,
    ];

    /// The vertex colour that marks a triangle as this part.
    pub fn colour(self) -> Colour {
        let index = Self::ALL
            .iter()
            .position(|&part| part == self)
            .expect("every part is listed");
        Colour::new(index as f32, 0.0, 0.0, 1.0)
    }

    fn from_code(code: f32) -> Part {
        Self::ALL[(code.round().max(0.0) as usize).min(Self::ALL.len() - 1)]
    }
}

/// One straight run of a light stream's trail, as it is lit this frame.
#[derive(Clone, Copy, Debug)]
pub struct TrailSegment {
    start: Vec2,
    direction: Vec2,
    length: f32,
    /// How far the stream's head is from `start`, at or beyond `length`.
    head: f32,
    /// One over the distance behind the head at which the trail has faded to 1/e.
    fade: f32,
    brightness: f32,
    /// The core's width, in pixels.
    width: f32,
    min: Vec2,
    max: Vec2,
}

impl TrailSegment {
    pub fn new(
        start: Vec2,
        direction: Vec2,
        length: f32,
        head: f32,
        fade: f32,
        brightness: f32,
        width: f32,
    ) -> Self {
        let end = start + direction * length;
        Self {
            start,
            direction,
            length,
            head,
            fade,
            brightness,
            width,
            min: Vec2::new(start.x.min(end.x), start.y.min(end.y)),
            max: Vec2::new(start.x.max(end.x), start.y.max(end.y)),
        }
    }
}

/// The bright head of a light stream.
#[derive(Clone, Copy, Debug)]
pub struct StreamHead {
    pub position: Vec2,
    pub brightness: f32,
    /// The trail's core width, in pixels.
    pub width: f32,
}

/// The light streams this frame, in the chip's top plane (x, z).
#[derive(Clone, Debug, Default)]
pub struct Streams {
    pub segments: Vec<TrailSegment>,
    pub heads: Vec<StreamHead>,
}

#[derive(Clone)]
pub struct ChipUniforms {
    pub view: ViewRef,
    pub look: Look,
    pub streams: Arc<Streams>,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct ChipVaryings {
    world: Vec3,
    /// The position in the chip's own space.
    local: Vec3,
    normal: Vec3,
    part: f32,
}

pub struct ChipVertexShader;

impl VertexShader for ChipVertexShader {
    type Vertex = ObjVertex;
    type Uniforms = ChipUniforms;
    type Varyings = ChipVaryings;

    fn shade(&self, vertex: ObjVertex, u: &ChipUniforms) -> (Vec4, ChipVaryings) {
        let chip = u.view.frame.chip;
        let world = chip * vertex.position.to_point4();
        let n = vertex.normal;
        let normal = chip * Vec4::new(n.x, n.y, n.z, 0.0);
        (
            u.view.view_projection * world,
            ChipVaryings {
                world: Vec3::new(world.x, world.y, world.z),
                local: vertex.position,
                normal: Vec3::new(normal.x, normal.y, normal.z),
                part: vertex.colour.r,
            },
        )
    }
}

pub struct ChipShader;

impl FragmentShaderSimd<ChipVaryings> for ChipShader {
    type Uniforms = ChipUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ChipVaryingsSimd, u: &ChipUniforms) -> ColourSimd {
        let view = &u.view;
        // A triangle is one part, and the eight lanes of a group come from one triangle.
        let part = Part::from_code(v.part.to_array()[0]);
        let world = v3(v.world);
        let local = v3(v.local);
        let normal = v3(v.normal).normalise();
        let to_eye = splat3(view.eye) - world;
        let distance = to_eye.dot(to_eye).sqrt();
        let eye = to_eye * (s(1.0) / distance);
        let facing = normal.dot(eye).abs().fast_max(s(0.12));
        let pixel = distance * s(view.pixel_angle);
        let filter = pixel * s(1.3) / facing.sqrt();
        let look = &u.look;
        let surface = match part {
            Part::Top => substrate_top(look, local, filter),
            Part::Bevel => Surface::new(look.substrate * 1.15, 1.0, 0.2),
            Part::Edge => substrate_edge(look, local, filter),
            Part::Bottom => underside(look, local, filter),
            Part::Underfill => Surface::new(Vec3::new(0.035, 0.036, 0.04), 0.0, 0.45),
            Part::DieSide => die_side(look, local, filter),
            Part::DieChamfer => Surface::new(look.die * 1.4, 0.0, 0.08),
            Part::DieTop => die_top(look, local, filter, facing),
            Part::Seal => Surface::new(look.gold, 1.0, 0.24),
            Part::Capacitor => capacitor(
                look,
                local,
                (local.x.abs().fast_max(local.z.abs()) - s(CAP_ROW)).abs(),
            ),
            Part::LandCapacitor => {
                let offset = (fract(local.x * s(1.0 / LAND_CAPS.2) + s(0.5)) - s(0.5)).abs();
                capacitor(look, local, offset * s(LAND_CAPS.2))
            }
        };
        let mut colour = light(view, look, &surface, normal, eye, local);
        if part.carries_streams() && look.streams != Vec3::ZERO {
            // Light is sized in pixels of a 720-line frame, as the style brief measures it, so
            // it looks the same at any render scale.
            let reference = distance * s(2.0 * view.tan_half_fov / 720.0);
            colour = colour + streams(&view.frame, look, &u.streams, local, reference, pixel);
        }
        scene_colour(view, colour, world)
    }
}

impl Part {
    /// Whether the light streams and the perimeter glow show on this part.
    fn carries_streams(self) -> bool {
        matches!(
            self,
            Part::Top
                | Part::Bevel
                | Part::Underfill
                | Part::DieSide
                | Part::DieChamfer
                | Part::DieTop
                | Part::Seal
        )
    }
}

/// A surface's material at eight fragments.
struct Surface {
    albedo: Vec3Simd,
    /// 0 for a dielectric, 1 for a metal.
    metal: F,
    roughness: F,
    /// How much of the surroundings' light reaches it, 0 to 1.
    occlusion: F,
}

impl Surface {
    fn new(albedo: Vec3, metal: f32, roughness: f32) -> Self {
        Self {
            albedo: splat3(albedo),
            metal: s(metal),
            roughness: s(roughness),
            occlusion: s(1.0),
        }
    }
}

/// Lights a surface: the studio's environment reflected with Fresnel, the key light rimming
/// the bevels, and the specular sweep.
#[inline(always)]
fn light(
    view: &View,
    look: &Look,
    m: &Surface,
    normal: Vec3Simd,
    eye: Vec3Simd,
    local: Vec3Simd,
) -> Vec3Simd {
    let n_dot_v = normal.dot(eye).fast_max(s(1e-3));
    let f0 = mix3(rgb(0.04, 0.04, 0.04), m.albedo, m.metal);
    let grazing = s(1.0) - n_dot_v;
    let grazing2 = grazing * grazing;
    let smooth = s(1.0) - m.roughness;
    let schlick = grazing2 * grazing2 * grazing * smooth * smooth;
    let fresnel = f0 + (rgb(1.0, 1.0, 1.0) - f0) * schlick;
    let reflected = reflect(eye * s(-1.0), normal);
    let reflection = environment(look, reflected, m.roughness);
    let irradiance = environment(look, normal, s(1.0)) * s(2.5);

    let key = splat3(look.key);
    let n_dot_l = normal.dot(key).fast_max(s(0.0));
    let half = (key + eye).normalise();
    let n_dot_h = normal.dot(half).fast_max(s(1e-4));
    let exponent = s(2.0) / (m.roughness * m.roughness).fast_max(s(1e-3));
    let highlight = (n_dot_h.ln() * exponent).exp() * (exponent + s(2.0)) * s(0.125) * n_dot_l;

    let diffuse = mul3(
        m.albedo * (s(1.0) - m.metal),
        irradiance + splat3(look.key_colour) * (n_dot_l * s(0.3)),
    );
    let specular = mul3(
        fresnel,
        reflection * m.occlusion.sqrt() + splat3(look.key_colour) * highlight,
    );
    let sweep = sweep(view.frame.sweep, local) * (smooth * smooth);
    let sweep_light = mul3(
        fresnel * s(0.8) + rgb(0.03, 0.03, 0.03),
        splat3(look.key_colour) * sweep,
    );
    diffuse * m.occlusion + specular + sweep_light
}

/// The specular sweep: a soft band of light at 25° to the chip's edges that crosses it as
/// `progress` goes from -1 to 1.
#[inline(always)]
fn sweep(progress: f32, local: Vec3Simd) -> F {
    if progress.abs() > 1.2 {
        return s(0.0);
    }
    let (sin, cos) = 25f32.to_radians().sin_cos();
    let across = (local.x * s(cos) + local.z * s(sin)) * s(1.0 / (HALF * 1.45));
    let offset = (across - s(progress * 1.1)) * s(1.0 / 0.075);
    (-(offset * offset)).exp()
}

/// The studio around the chip, as its surfaces reflect it: dark, with softbox cards above, to
/// each side and behind. `roughness` blurs the cards' edges.
#[inline(always)]
fn environment(look: &Look, direction: Vec3Simd, roughness: F) -> Vec3Simd {
    let horizon = smoothstep(-0.3, 0.6, direction.y);
    let mut colour = mix3(splat3(look.ground), splat3(look.sky), horizon);
    if look.softbox == Vec3::ZERO {
        return colour;
    }
    let soft = s(0.02) + roughness * s(0.45);
    let mut cards = s(0.0);
    for card in &CARDS {
        let facing = direction.dot(splat3(card.centre));
        let inverse = s(1.0) / facing.fast_max(s(0.05));
        let x = direction.dot(splat3(card.across)) * inverse;
        let y = direction.dot(splat3(card.up)) * inverse;
        let edge = |coordinate: F, half: f32| {
            let t = clamp01((s(half) - coordinate.abs()) / soft + s(0.5));
            t * t * (s(3.0) - s(2.0) * t)
        };
        let spread =
            s(card.half.0 * card.half.1) / ((s(card.half.0) + soft) * (s(card.half.1) + soft));
        let (u, v) = (x * s(1.0 / card.half.0), y * s(1.0 / card.half.1));
        let falloff = s(1.0) - (u * u + v * v).fast_min(s(2.0)) * s(card.falloff * 0.5);
        let lit =
            edge(x, card.half.0) * edge(y, card.half.1) * spread * falloff * s(card.intensity);
        cards += select(gt(facing, s(0.05)), lit, s(0.0));
    }
    colour = colour + splat3(look.softbox) * cards;
    colour
}

/// A rectangular softbox in the studio, seen from the chip: the direction to its centre, its
/// axes, its half-size in tangent units, and its brightness.
struct Card {
    centre: Vec3,
    across: Vec3,
    up: Vec3,
    half: (f32, f32),
    intensity: f32,
    /// How much dimmer the card is at its edges than its centre, 0 to 1.
    falloff: f32,
}

/// A large softbox above and in front, a tall strip behind to the left (the rim), a dimmer strip
/// to the right, a broad, faint fill behind the camera and a low bounce in front. The directions are unit vectors and
/// each card's axes are perpendicular.
const CARDS: [Card; 5] = [
    Card {
        centre: Vec3 {
            x: 0.0,
            y: 0.72,
            z: 0.694,
        },
        across: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        up: Vec3 {
            x: 0.0,
            y: 0.694,
            z: -0.72,
        },
        half: (0.95, 0.2),
        intensity: 1.5,
        falloff: 0.5,
    },
    Card {
        centre: Vec3 {
            x: -0.83,
            y: 0.2,
            z: -0.52,
        },
        across: Vec3 {
            x: 0.531,
            y: 0.0,
            z: -0.847,
        },
        up: Vec3 {
            x: -0.104,
            y: 0.979,
            z: -0.065,
        },
        half: (0.08, 0.9),
        intensity: 2.6,
        falloff: 0.3,
    },
    Card {
        centre: Vec3 {
            x: 0.87,
            y: 0.12,
            z: 0.478,
        },
        across: Vec3 {
            x: -0.482,
            y: 0.0,
            z: 0.876,
        },
        up: Vec3 {
            x: -0.058,
            y: 0.993,
            z: 0.105,
        },
        half: (0.06, 0.8),
        intensity: 0.9,
        falloff: 0.3,
    },
    Card {
        centre: Vec3 {
            x: 0.0,
            y: 0.196,
            z: 0.981,
        },
        across: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        up: Vec3 {
            x: 0.0,
            y: 0.981,
            z: -0.196,
        },
        half: (1.6, 0.7),
        intensity: 0.5,
        falloff: 0.9,
    },
    Card {
        centre: Vec3 {
            x: 0.0,
            y: -0.55,
            z: 0.835,
        },
        across: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        up: Vec3 {
            x: 0.0,
            y: 0.835,
            z: 0.55,
        },
        half: (1.8, 0.45),
        intensity: 0.55,
        falloff: 0.8,
    },
];

/// The fraction of `[x - w/2, x + w/2]` covered by a pulse train that is 1 on `[0, duty)` of
/// each unit period: an exactly box-filtered stripe.
#[inline(always)]
fn pulse(x: F, duty: f32, w: F) -> F {
    let integral = |x: F| x.floor() * s(duty) + fract(x).fast_min(s(duty));
    let w = w.fast_max(s(1e-4));
    clamp01((integral(x + w * s(0.5)) - integral(x - w * s(0.5))) / w)
}

/// A pulse train of period `period` in `x`, box filtered over `w`.
#[inline(always)]
fn stripes(x: F, period: f32, duty: f32, w: F) -> F {
    pulse(x * s(1.0 / period), duty, w * s(1.0 / period))
}

/// How much of `[x - w/2, x + w/2]` lies between `low` and `high`, approximately.
#[inline(always)]
fn inside(x: F, low: f32, high: f32, w: F) -> F {
    let inverse = s(1.0) / w.fast_max(s(1e-5));
    clamp01((x - s(low)) * inverse + s(0.5)) * clamp01((s(high) - x) * inverse + s(0.5))
}

/// Coverage of the region where the signed distance `d` is negative.
#[inline(always)]
fn coverage(d: F, w: F) -> F {
    clamp01(s(0.5) - d / w.fast_max(s(1e-5)))
}

/// How much of a random per-cell detail of size `size` to keep: it fades to its average before
/// cells shrink to a pixel.
#[inline(always)]
fn detail(w: F, size: f32) -> F {
    s(1.0) - smoothstep(0.3, 0.9, w * s(1.0 / size))
}

/// The signed distance from (x, z) to a square of half-width `half` with corners of `radius`.
#[inline(always)]
fn rounded_square(x: F, z: F, half: f32, radius: f32) -> F {
    let qx = x.abs() - s(half - radius);
    let qz = z.abs() - s(half - radius);
    let outside = (qx.fast_max(s(0.0)) * qx.fast_max(s(0.0))
        + qz.fast_max(s(0.0)) * qz.fast_max(s(0.0)))
    .sqrt();
    outside + qx.fast_max(qz).fast_min(s(0.0)) - s(radius)
}

/// A shade of the die's green, how much gold covers it, and how warm the block is.
struct Floor {
    shade: F,
    gold: F,
    warm: F,
}

/// The die's face: its floorplan, under a passivation layer whose thin film shifts the green
/// from teal seen square on to yellow at a glancing angle (`facing` is the cosine of the angle).
#[inline(always)]
fn die_top(look: &Look, local: Vec3Simd, filter: F, facing: F) -> Surface {
    let floor = floorplan(
        local.x * s(1.0 / DIE),
        local.z * s(1.0 / DIE),
        filter * s(1.0 / DIE),
    );
    let warm = floor.warm + (s(0.55) - facing) * s(0.9);
    let tint = Vec3Simd::new(
        s(1.0) + warm * s(0.35),
        s(1.0) + warm * s(0.05),
        s(1.0) - warm * s(0.3),
    );
    let green = mul3(splat3(look.die), tint) * (floor.shade * s(3.0));
    Surface {
        albedo: mix3(green, splat3(look.gold), floor.gold),
        metal: floor.gold * s(0.75),
        roughness: mix(s(0.16), s(0.5), floor.gold),
        occlusion: s(1.0),
    }
}

/// The die's floorplan in die units (the die spans -1 to 1): a ring of I/O pads inside the seal
/// ring; two rows of four cores either side of the shared cache, with the ring bus between them;
/// and a column of memory interface, system agent and accelerator array beside them.
#[inline(always)]
fn floorplan(u: F, v: F, w: F) -> Floor {
    const CORE_EDGE: f32 = 0.888;
    const CORE_PITCH: f32 = 0.359;
    let (au, av) = (u.abs(), v.abs());
    let r = au.fast_max(av);
    let mut shade = s(0.24);
    let mut gold = s(0.0);
    let mut warm = s(0.0);

    // The seal ring, the I/O pad ring and the gap inside it.
    let seal = inside(r, 0.972, 0.986, w) + inside(r, 0.961, 0.965, w) * s(0.7);
    let along = select(gt(au, av), v, u);
    let pads = stripes(along + s(0.008), 0.032, 0.55, w) * inside(r, 0.915, 0.95, w);
    let io = inside(r, 0.905, 0.958, w);
    shade = mix(shade, s(0.42) + stripes(along, 0.008, 0.5, w) * s(0.2), io);
    gold = (gold + pads * s(0.95) + seal).fast_min(s(1.0));

    let core_region = inside(r, -1.0, CORE_EDGE, w);
    if !gt(core_region, s(0.0)).any() {
        return Floor { shade, gold, warm };
    }

    // Where each kind of block is.
    let main = inside(u, -2.0, 0.548, w) * core_region;
    let uncore = inside(u, 0.562, 2.0, w) * core_region;
    let column = (u + s(CORE_EDGE)) * s(1.0 / CORE_PITCH);
    let index = column.floor();
    let a = fract(column);
    let between_cores = s(1.0) - inside(a, 0.025, 0.975, w * s(1.0 / CORE_PITCH));
    let mirrored = select(gt(fract(index * s(0.5)), s(0.25)), s(1.0) - a, a);
    let core_row = inside(av, 0.305, 2.0, w) * main;
    let bus = inside(av, 0.225, 0.295, w) * main;
    let cache = inside(av, 0.022, 0.215, w) * main;
    let spine = inside(av, -1.0, 0.012, w) * main;
    let b = (av - s(0.305)) * s(1.0 / (CORE_EDGE - 0.305));
    let (wa, wb) = (w * s(1.0 / CORE_PITCH), w * s(1.0 / (CORE_EDGE - 0.305)));
    let l2 = inside(b, 0.03, 0.3, wb) * inside(mirrored, 0.04, 0.96, wa);
    let l1 = inside(b, 0.36, 0.62, wb) * inside(mirrored, 0.05, 0.3, wa);
    let core_logic = inside(b, 0.34, 0.985, wb) * inside(mirrored, 0.03, 0.97, wa) * (s(1.0) - l1);
    let tags = inside(a, 0.46, 0.54, wa);
    let phy = inside(v, 0.3, 2.0, w) * uncore;
    let agent = inside(v, -0.25, 0.285, w) * uncore;
    let accelerator = inside(v, -2.0, -0.265, w) * uncore;
    let unit = inside(
        fract((u - s(0.562)) * s(1.0 / 0.1087)),
        0.07,
        0.93,
        w * s(1.0 / 0.1087),
    ) * inside(
        fract((v + s(CORE_EDGE)) * s(1.0 / 0.1246)),
        0.07,
        0.93,
        w * s(1.0 / 0.1246),
    );
    let inner = s(1.0) - between_cores;

    let sram_weight =
        core_row * inner * (l2 + l1) + cache * inner * (s(1.0) - tags) + accelerator * unit;
    let logic_weight = core_row * inner * core_logic + cache * inner * tags + agent;
    let vertical = (core_row + cache) * between_cores
        + inside(u, 0.548, 0.562, w) * core_region
        + accelerator * (s(1.0) - unit);
    let horizontal = bus + spine;

    // A slightly different shade for each block, so they read apart at a distance.
    let block = hash2(
        index + s(11.0),
        (v * s(2.0)).floor() + select(gt(u, s(0.555)), s(7.0), s(0.0)),
    );
    let block_shade = s(0.92) + block * s(0.16);

    if gt(sram_weight, s(0.0)).any() {
        let (cells, wiring) = sram(u, v, w);
        shade = mix(shade, cells * block_shade, sram_weight);
        gold = mix(gold, wiring, sram_weight);
        warm -= sram_weight * s(0.5);
    }
    if gt(logic_weight, s(0.0)).any() {
        let (cells, wiring) = logic(u, v, w);
        shade = mix(shade, cells * block_shade, logic_weight);
        gold = mix(gold, wiring, logic_weight);
        warm += logic_weight * s(0.45);
    }
    let channels = vertical.fast_min(s(1.0)) + horizontal;
    if gt(channels, s(0.0)).any() {
        let across = select(gt(vertical, horizontal), u, v);
        let wires = stripes(across, 0.0042, 0.45, w) * stripes(across + s(0.002), 0.03, 0.82, w);
        shade = mix(shade, s(0.22), channels);
        gold = mix(gold, wires * s(0.85), channels);
    }
    if gt(phy, s(0.0)).any() {
        let lanes = stripes(u, 0.04, 0.5, w);
        let hatching = stripes(v, 0.004, 0.5, w);
        shade = mix(shade, s(0.35) + hatching * s(0.25), phy);
        gold = mix(gold, lanes * (s(0.55) + hatching * s(0.35)), phy);
        warm += phy * s(0.2);
    }

    // The top metal's power straps, a coarse grid over the whole core region.
    let straps = (stripes(u + s(0.03), 0.11, 0.03, w) + stripes(v + s(0.03), 0.11, 0.03, w))
        .fast_min(s(1.0));
    gold = mix(gold, s(0.8), straps * core_region * s(0.45));
    Floor { shade, gold, warm }
}

/// An SRAM array: sub-arrays of fine bit cells with gold contacts, separated by decoder strips.
/// Returns the shade and the gold coverage.
#[inline(always)]
fn sram(x: F, y: F, w: F) -> (F, F) {
    let (sub_x, sub_y) = (0.052, 0.046);
    let sub = stripes(x, sub_x, 0.86, w) * stripes(y, sub_y, 0.82, w);
    let (cell_x, cell_y) = (0.0052, 0.0034);
    let cells = stripes(x, cell_x, 0.62, w) * stripes(y, cell_y, 0.55, w);
    let contacts = stripes(x + s(cell_x * 0.7), cell_x, 0.18, w)
        * stripes(y + s(cell_y * 0.6), cell_y, 0.3, w);
    let tone = (hash2((x * s(1.0 / sub_x)).floor(), (y * s(1.0 / sub_y)).floor()) - s(0.5))
        * detail(w, sub_x)
        * s(0.12);
    let array_shade = s(0.62) - cells * s(0.3) + tone;
    let periphery = stripes(x, 0.0026, 0.5, w);
    (
        mix(s(0.6) + periphery * s(0.1), array_shade, sub),
        mix(periphery * s(0.35), contacts * s(0.75), sub),
    )
}

/// Standard-cell logic: rows of cells of random widths between power rails, some of them gold,
/// with gate fingers inside and gold wires routed over them. Returns the shade and the gold
/// coverage.
#[inline(always)]
fn logic(x: F, y: F, w: F) -> (F, F) {
    const ROW: f32 = 0.0078;
    const CELL: f32 = 0.0062;
    let row = (y * s(1.0 / ROW)).floor();
    let rails = stripes(y + s(ROW * 0.06), ROW, 0.12, w);
    let cell = x * s(1.0 / CELL) + hash2(row, s(3.3)) * s(7.0);
    let random = hash2(cell.floor(), row + s(17.0));
    let visible = detail(w, CELL);
    let fingers = stripes(x, 0.0011, 0.4, w) - s(0.4);
    let gap = pulse(cell, 0.88, w * s(1.0 / CELL));
    let shade = s(0.5) + (random - s(0.5)) * s(0.45) * visible + fingers * s(0.2)
        - (s(1.0) - gap) * s(0.12);
    let gold_cells = select(gt(random, s(0.9)), s(1.0), s(0.0)) * gap;
    let gold_cells = mix(s(0.1 * 0.88), gold_cells, visible);

    // Wires on two metal layers, in random lengths along their tracks.
    let track = (y * s(1.0 / 0.0039)).floor();
    let run = (x * s(1.0 / 0.045) + hash2(track, s(5.1)) * s(5.0)).floor();
    let present = select(gt(hash2(run, track), s(0.55)), s(1.0), s(0.0));
    let wires = stripes(y, 0.0039, 0.3, w) * mix(s(0.45), present, detail(w, 0.0039 * 2.0));
    let track_v = (x * s(1.0 / 0.0055)).floor();
    let run_v = (y * s(1.0 / 0.06) + hash2(track_v, s(8.7)) * s(5.0)).floor();
    let present_v = select(gt(hash2(track_v, run_v), s(0.7)), s(1.0), s(0.0));
    let wires_v = stripes(x, 0.0055, 0.25, w) * mix(s(0.3), present_v, detail(w, 0.0055 * 2.0));

    let tone = (hash2(
        (x * s(1.0 / 0.09)).floor(),
        (y * s(1.0 / 0.07)).floor() + s(31.0),
    ) - s(0.5))
        * detail(w, 0.07)
        * s(0.14);
    let gold =
        (gold_cells * s(0.55) + rails * s(0.35) + (wires + wires_v) * s(0.35)).fast_min(s(1.0));
    (shade + tone, gold)
}

/// The die's sides: dark silicon, with the metal stack's layers in gold near the top.
#[inline(always)]
fn die_side(look: &Look, local: Vec3Simd, filter: F) -> Surface {
    let layers =
        stripes(local.y, 0.0024, 0.35, filter) * inside(local.y, DIE_TOP - 0.03, DIE_TOP, filter);
    Surface {
        albedo: mix3(splat3(look.die * 0.45), splat3(look.gold), layers * s(0.7)),
        metal: layers * s(0.7),
        roughness: s(0.2),
        occlusion: s(0.7) + clamp01((local.y - s(0.03)) * s(30.0)) * s(0.3),
    }
}

/// The substrate's top: a dark field around the die, crossed by gold traces fanning out to the
/// package's edge, inside a satin metal frame with the maker's markings.
#[inline(always)]
fn substrate_top(look: &Look, local: Vec3Simd, w: F) -> Surface {
    let (x, z) = (local.x, local.z);
    let (ax, az) = (x.abs(), z.abs());
    let radial = ax.fast_max(az);
    let along = select(gt(ax, az), z, x);

    // The dark field and the frame, with a polished step between them.
    let frame = inside(radial, 1.565, 3.0, w);
    let step = inside(radial, 1.535, 1.565, w);
    let mut albedo = mix3(rgb(0.07, 0.085, 0.08), splat3(look.substrate), frame);
    let mut metal = mix(s(0.15), s(0.9), frame);
    let mut roughness = mix(s(0.34), s(0.4), frame);

    // Fine brushing on the frame and a faint mottling at a larger scale.
    let brushing = (hash2(
        (radial * s(1.0 / 0.0022)).floor(),
        (along * s(1.0 / 0.3) + hash2(radial.floor(), s(2.0))).floor(),
    ) - s(0.5))
        * detail(w, 0.0022)
        * s(0.09);
    let mottling = (hash2((x * s(1.0 / 0.012)).floor(), (z * s(1.0 / 0.012)).floor()) - s(0.5))
        * detail(w, 0.012)
        * s(0.05);
    albedo = albedo * (s(1.0) + brushing * frame + mottling);

    // Traces from the seal ring to the edge, each with a via part way out, and a field of vias
    // in the corners.
    let lanes = inside(along, -0.76, 0.76, w) * inside(radial, DIE + SEAL.1 + 0.025, HALF, w);
    let trace = stripes(along + s(LANE_PITCH * 0.07), LANE_PITCH, 0.14, w) * lanes;
    let lane = (along * s(1.0 / LANE_PITCH)).round();
    let via_at = s(1.12)
        + hash2(
            lane,
            select(gt(ax, az), s(1.0), s(2.0)) + select(gt(x + z, s(0.0)), s(3.0), s(0.0)),
        ) * s(0.35);
    let via_d = ((along - lane * s(LANE_PITCH)) * (along - lane * s(LANE_PITCH))
        + (radial - via_at) * (radial - via_at))
        .sqrt();
    let via = coverage(via_d - s(0.017), w) * lanes;
    let corner_field = inside(along.abs(), 0.86, 1.45, w) * inside(radial, 1.05, 1.5, w);
    let (gx, gz) = ((x * s(1.0 / 0.05)).round(), (z * s(1.0 / 0.05)).round());
    let dot_d = (((x - gx * s(0.05)) * (x - gx * s(0.05)))
        + (z - gz * s(0.05)) * (z - gz * s(0.05)))
    .sqrt();
    let dots = coverage(dot_d - s(0.009), w)
        * corner_field
        * mix(
            s(0.5),
            select(gt(hash2(gx, gz), s(0.5)), s(1.0), s(0.0)),
            detail(w, 0.05),
        );
    let copper = (trace * mix(s(0.75), s(0.35), frame) + via + dots).fast_min(s(1.0));
    albedo = mix3(
        albedo,
        splat3(look.gold) * mix(s(1.0), s(0.8), frame),
        copper,
    );
    metal = mix(metal, s(1.0), copper);
    roughness = mix(roughness, s(0.28), copper);
    albedo = mix3(albedo, splat3(look.substrate * 1.3), step);
    metal = mix(metal, s(1.0), step);
    roughness = mix(roughness, s(0.12), step);

    // Markings: a gold pin-one triangle and an etched data matrix and line of type in the frame.
    let triangle = coverage(
        (x + z + s(3.34))
            .fast_max(s(-1.83) - x)
            .fast_max(s(-1.83) - z),
        w,
    );
    albedo = mix3(albedo, splat3(look.gold), triangle);
    metal = mix(metal, s(1.0), triangle);
    let etch = marking(x, z, w);
    albedo = albedo * (s(1.0) - etch * s(0.6));
    roughness = mix(roughness, s(0.6), etch);

    // Shadowing where the die and the capacitors meet the substrate.
    let die_d = rounded_square(x, z, DIE, DIE_CORNER);
    let near_die = s(0.4) + smoothstep(0.04, 0.2, die_d) * s(0.6);
    let cap_band =
        s(1.0) - smoothstep(CAP.0 * 0.5, CAP.0 * 0.5 + 0.06, (radial - s(CAP_ROW)).abs());
    let near_caps = s(1.0) - cap_band * inside(along, -0.8, 0.8, s(0.05)) * s(0.35);
    Surface {
        albedo,
        metal,
        roughness,
        occlusion: near_die * near_caps,
    }
}

/// How much laser etching marks the frame at (x, z): a data matrix in one corner and a line of
/// dot-matrix type along the near edge.
#[inline(always)]
fn marking(x: F, z: F, w: F) -> F {
    const MODULE: f32 = 0.016;
    let (mx, mz) = (
        (x - s(1.56)) * s(1.0 / MODULE),
        (z + s(1.82)) * s(1.0 / MODULE),
    );
    let in_matrix =
        inside(mx, 0.0, 14.0, w * s(1.0 / MODULE)) * inside(mz, 0.0, 14.0, w * s(1.0 / MODULE));
    let (cx, cz) = (mx.floor(), mz.floor());
    let finder = select(lt(cx, s(1.0)) | gt(cz, s(12.0)), s(1.0), s(0.0));
    let timing = select(
        (lt(cz, s(1.0)) | gt(cx, s(12.0))) & lt(fract((cx + cz) * s(0.5)), s(0.25)),
        s(1.0),
        s(0.0),
    );
    let data = select(gt(hash2(cx + s(3.0), cz + s(5.0)), s(0.5)), s(1.0), s(0.0));
    let module = (finder + timing).fast_min(s(1.0));
    let module = select(gt(finder + timing, s(0.0)), module, data);
    let matrix = in_matrix * mix(s(0.5), module, detail(w, MODULE));

    const DOT: f32 = 0.0085;
    let (tx, tz) = ((x + s(0.62)) * s(1.0 / DOT), (z - s(1.66)) * s(1.0 / DOT));
    let in_type = inside(tx, 0.0, 146.0, w * s(1.0 / DOT)) * inside(tz, 0.0, 5.0, w * s(1.0 / DOT));
    let glyph = (tx * s(1.0 / 4.0)).floor();
    let column = tx - glyph * s(4.0);
    let blank = select(
        gt(column, s(3.0)) | gt(hash2(glyph, s(9.0)), s(0.86)),
        s(0.0),
        s(1.0),
    );
    let lit = select(
        gt(
            hash2(glyph * s(4.0) + column.floor(), tz.floor() + s(1.0)),
            s(0.45),
        ),
        s(1.0),
        s(0.0),
    );
    let type_line = in_type * mix(s(0.3), lit * blank, detail(w, DOT));
    (matrix + type_line).fast_min(s(1.0))
}

/// The substrate's side: dark laminate between layers of copper, under the solder mask.
#[inline(always)]
fn substrate_edge(look: &Look, local: Vec3Simd, w: F) -> Surface {
    let height = (local.y - s(SUBSTRATE_BOTTOM + 0.025)) * s(1.0 / 0.107);
    let wh = w * s(1.0 / 0.107);
    let copper = [0.1, 0.22, 0.34, 0.66, 0.78, 0.9]
        .into_iter()
        .fold(s(0.0), |sum, at| {
            sum + inside(height, at - 0.018, at + 0.018, wh)
        });
    let core = inside(height, 0.4, 0.6, wh);
    let weave = stripes(local.x + local.z, 0.006, 0.5, w) * detail(w, 0.006) * s(0.25);
    let laminate = mix3(
        rgb(0.05, 0.055, 0.055),
        rgb(0.13, 0.13, 0.12) * (s(1.0) + weave),
        core,
    );
    Surface {
        albedo: mix3(laminate, splat3(look.gold), copper),
        metal: copper,
        roughness: mix(s(0.5), s(0.25), copper),
        occlusion: s(1.0),
    }
}

/// The underside: a grid of gold contact pads on dark solder mask around the capacitor cluster.
#[inline(always)]
fn underside(look: &Look, local: Vec3Simd, w: F) -> Surface {
    const PITCH: f32 = 0.075;
    let (x, z) = (local.x, local.z);
    let (ax, az) = (x.abs(), z.abs());
    let field = inside(ax.fast_max(az), -1.0, 1.86, w)
        * (s(1.0) - inside(ax, -1.0, 0.44, w) * inside(az, -1.0, 0.52, w));
    let pads = stripes(x + s(PITCH * 0.3), PITCH, 0.6, w)
        * stripes(z + s(PITCH * 0.3), PITCH, 0.6, w)
        * field;
    let triangle = coverage(
        (x + z + s(3.3)).fast_max(s(-1.8) - x).fast_max(s(-1.8) - z),
        w,
    );
    let gold = (pads + triangle).fast_min(s(1.0));
    let mask = rgb(0.035, 0.05, 0.042)
        * (s(1.0)
            + (hash2((x * s(80.0)).floor(), (z * s(80.0)).floor()) - s(0.5))
                * detail(w, 0.0125)
                * s(0.2));
    Surface {
        albedo: mix3(mask, splat3(look.gold), gold),
        metal: gold,
        roughness: mix(s(0.4), s(0.22), gold),
        occlusion: s(1.0),
    }
}

/// A ceramic capacitor, with silver terminals where it is more than its body's half-length from
/// its centre (`offset`).
#[inline(always)]
fn capacitor(look: &Look, local: Vec3Simd, offset: F) -> Surface {
    let terminal = select(gt(offset, s(CAP.0 * 0.3)), s(1.0), s(0.0));
    let height = select(
        gt(local.y, s(SUBSTRATE_BOTTOM * 0.5)),
        local.y,
        s(SUBSTRATE_BOTTOM) - local.y,
    );
    Surface {
        albedo: mix3(
            rgb(0.16, 0.14, 0.12),
            splat3(look.substrate * 1.2),
            terminal,
        ),
        metal: terminal,
        roughness: mix(s(0.5), s(0.22), terminal),
        occlusion: s(0.6) + clamp01(height * s(25.0)) * s(0.4),
    }
}

/// The light streams and the perimeter glow at a point of the chip's top. Their sizes are in
/// `reference` pixels (world units across a pixel of a 720-line frame, as the style brief
/// measures them), anti-aliased over the target's own `pixel`.
#[inline(always)]
fn streams(
    frame: &Frame,
    look: &Look,
    streams: &Streams,
    local: Vec3Simd,
    reference: F,
    pixel: F,
) -> Vec3Simd {
    let (x, z) = (local.x, local.z);
    let colour = splat3(look.streams);
    let deep = mul3(colour, rgb(0.45, 0.8, 0.75));
    let white = rgb(0.93, 1.0, 0.97);
    let mut light = Vec3Simd::new(s(0.0), s(0.0), s(0.0));
    let sharpness = reference / pixel;

    let xs = x.to_array();
    let zs = z.to_array();
    let reach = reference.to_array().into_iter().fold(0.0f32, f32::max) * 14.0;
    let low = Vec2::new(
        xs.into_iter().fold(f32::MAX, f32::min) - reach,
        zs.into_iter().fold(f32::MAX, f32::min) - reach,
    );
    let high = Vec2::new(
        xs.into_iter().fold(f32::MIN, f32::max) + reach,
        zs.into_iter().fold(f32::MIN, f32::max) + reach,
    );
    let near = |min: Vec2, max: Vec2| {
        max.x >= low.x && min.x <= high.x && max.y >= low.y && min.y <= high.y
    };

    for segment in streams
        .segments
        .iter()
        .filter(|segment| near(segment.min, segment.max))
    {
        let (dx, dz) = (x - s(segment.start.x), z - s(segment.start.y));
        let t = (dx * s(segment.direction.x) + dz * s(segment.direction.y))
            .fast_max(s(0.0))
            .fast_min(s(segment.length));
        let (ox, oz) = (
            dx - s(segment.direction.x) * t,
            dz - s(segment.direction.y) * t,
        );
        let d = (ox * ox + oz * oz).sqrt() / reference;
        let trail = (-(s(segment.head) - t) * s(segment.fade)).exp();
        let core = clamp01((s(segment.width * 0.5) - d) * sharpness + s(0.5))
            .fast_min(sharpness * s(segment.width));
        let glow = (-(d * d) * s(1.0 / 8.0)).exp();
        let hot = trail * trail;
        let tint = mix3(mix3(deep, colour, trail), white, hot * hot);
        let lit = tint
            * (trail * s(segment.brightness) * (core * (s(0.9) + hot * s(1.8)) + glow * s(0.25)));
        light = Vec3Simd::new(
            light.x.fast_max(lit.x),
            light.y.fast_max(lit.y),
            light.z.fast_max(lit.z),
        );
    }
    for head in &streams.heads {
        let r = Vec2::new(reach, reach);
        if !near(head.position - r, head.position + r) {
            continue;
        }
        let (dx, dz) = (x - s(head.position.x), z - s(head.position.y));
        let d2 = (dx * dx + dz * dz) / (reference * reference);
        let core = (-d2 * s(1.0 / (head.width * head.width * 0.6))).exp();
        let halo = (-d2 * s(1.0 / 30.0)).exp();
        light = light + (white * (core * s(3.5)) + colour * (halo * s(0.35))) * s(head.brightness);
    }

    light + perimeter(frame, look, x, z, reference, sharpness)
}

/// The glow that traces the die's perimeter along the seal ring while the sweep passes: a thin
/// core with a soft falloff, spreading both ways round from the front left corner.
#[inline(always)]
fn perimeter(frame: &Frame, look: &Look, x: F, z: F, reference: F, sharpness: F) -> Vec3Simd {
    let sweep = frame.sweep;
    if sweep.abs() > 1.0 || frame.activity <= 0.0 {
        return Vec3Simd::new(s(0.0), s(0.0), s(0.0));
    }
    let t = ((sweep + 1.0) / 1.2).clamp(0.0, 1.0);
    let progress = t * t * (3.0 - 2.0 * t);
    let strength = (1.0 - ((sweep - 0.2) / 0.8).clamp(0.0, 1.0)) * frame.activity;
    let ring =
        (rounded_square(x, z, DIE, DIE_CORNER) - s((SEAL.0 + SEAL.1) * 0.5)).abs() / reference;
    let core = clamp01((s(1.4) - ring) * sharpness + s(0.5));
    let falloff = (-ring * s(1.0 / 7.0)).exp();
    let angle = fract(z.atan2(x) * s(1.0 / TAU) + s(0.5) - s(0.625));
    let from_start = angle.fast_min(s(1.0) - angle);
    let reach = s(progress * 0.5);
    let lit = clamp01((reach - from_start) * s(40.0));
    let ahead = (reach - from_start) * s(60.0);
    let front = (-(ahead * ahead)).exp() * s(1.0 - progress * 0.6);
    let white = rgb(0.95, 1.0, 0.97);
    let colour = splat3(look.streams);
    (colour * (lit * s(0.7)) + white * (front * s(1.5)))
        * ((core * s(1.4) + falloff * s(0.22)) * s(strength))
}
