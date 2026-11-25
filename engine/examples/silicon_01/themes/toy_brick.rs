//! toy-brick: snap-together plastic. A studded baseplate with a ring road, a house, pines and a
//! car built of bricks that drop in one by one, a skyline of giant towers in the haze, and the
//! chip as a yellow plate with studs.

use std::collections::HashSet;
use std::f32::consts::TAU;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;

use crate::chip::{HALF, Look};
use crate::shaders::composite::Grade;
use crate::shaders::screen::ScreenVertexShader;
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{
    F, clamp01, fract, hash2, length, mix3, pow_highlight, reflect, rgb, s, smoothstep, splat3, v3,
};
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{KitVaryings, KitVaryingsSimd, ToonUniforms, pipeline};
use crate::themes::kit::dress::chip_shadow;
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shadow::GroundShadow;
use crate::themes::kit::shape::{KitVertex, Shape, transform_point};
use crate::themes::kit::sky::{Clouds, Sky, SkyShader, SkyUniforms};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

/// The distance between studs, in world units; bricks are built on this grid.
const PITCH: f32 = 0.4;
const BRICK_HEIGHT: f32 = 0.48;
const STUD_RADIUS: f32 = 0.12;
const STUD_HEIGHT: f32 = 0.07;
/// The gap left round each brick so neighbours show their seams.
const SEAM: f32 = 0.006;
/// The baseplate's top, the ground line shared with the other worlds.
const FLOOR: f32 = -1.95;
const SUN: Vec3 = Vec3 {
    x: -0.55,
    y: 0.75,
    z: 0.5,
};

/// A baseplate is 32 studs across.
const PLATE: f32 = 32.0 * PITCH;
/// The ring road's centre (x, z) and its inner and outer radii.
const ROAD_CENTRE: (f32, f32) = (0.5, -0.5);
const ROAD_INNER: f32 = 3.5;
const ROAD_OUTER: f32 = 5.1;
/// The house's path from its door to the road, as (x, z) end points, and the pond's centre and
/// radii (x, z).
const PATH: ((f32, f32), (f32, f32)) = ((-7.4, -2.3), (-4.6, -0.7));
const POND: (f32, f32, f32, f32) = (4.2, 3.0, 2.2, 0.9);
/// How fast the car drives round the ring road, in radians per second.
const CAR_SPEED: f32 = 0.4;

/// Bricks drop in over this many seconds, each falling for [`DROP`] from [`DROP_HEIGHT`] up. The
/// build starts [`LEAD`] seconds in, so the first frame already shows bricks landing.
const BUILD: f32 = 1.5;
const DROP: f32 = 0.4;
const DROP_HEIGHT: f32 = 14.0;
const LEAD: f32 = 0.9;

const RED: Vec3 = Vec3 {
    x: 0.62,
    y: 0.05,
    z: 0.04,
};
const YELLOW: Vec3 = Vec3 {
    x: 0.9,
    y: 0.6,
    z: 0.03,
};
const BLUE: Vec3 = Vec3 {
    x: 0.04,
    y: 0.2,
    z: 0.55,
};
const WHITE: Vec3 = Vec3 {
    x: 0.78,
    y: 0.76,
    z: 0.7,
};
const SKY_BLUE: Vec3 = Vec3 {
    x: 0.3,
    y: 0.62,
    z: 0.85,
};
const GREEN: Vec3 = Vec3 {
    x: 0.05,
    y: 0.36,
    z: 0.1,
};
const LIME: Vec3 = Vec3 {
    x: 0.22,
    y: 0.5,
    z: 0.08,
};
const BROWN: Vec3 = Vec3 {
    x: 0.24,
    y: 0.11,
    z: 0.05,
};
const ORANGE: Vec3 = Vec3 {
    x: 0.85,
    y: 0.3,
    z: 0.04,
};
const BLACK: Vec3 = Vec3 {
    x: 0.025,
    y: 0.025,
    z: 0.03,
};
const GREY: Vec3 = Vec3 {
    x: 0.3,
    y: 0.31,
    z: 0.33,
};

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.55, 1.4, 1.2),
        sky: Vec3::new(0.38, 0.5, 0.72),
        ground: Vec3::new(0.3, 0.27, 0.2),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: Vec3::new(0.78, 0.85, 0.95),
            density: 0.0016,
            falloff: 0.0,
            base: FLOOR,
        },
    }
}

/// The haze the giant towers stand in, thicker than the foreground's.
fn tower_lighting() -> Lighting {
    Lighting {
        fog: Fog {
            density: 0.005,
            ..lighting().fog
        },
        ..lighting()
    }
}

fn sky() -> Sky {
    Sky {
        zenith: Vec3::new(0.2, 0.42, 0.9),
        horizon: Vec3::new(0.85, 0.92, 1.05),
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.8, 1.55, 1.2),
        sun_size: 0.035,
        glow: 0.5,
        clouds: Some(Clouds {
            height: 380.0,
            coverage: 0.42,
            scale: 0.004,
            wind: Vec2::new(5.0, 1.5),
            lit: Vec3::new(1.6, 1.55, 1.5),
            shade: Vec3::new(0.82, 0.86, 0.96),
        }),
        stars: 0.0,
    }
}

/// The chip as a plate: yellow plastic, a blue die with red wiring.
fn look() -> Look {
    let light = lighting();
    Look {
        substrate: YELLOW * 1.05,
        die: BLUE * 0.9,
        gold: Vec3::new(0.85, 0.12, 0.06),
        streams: Vec3::new(1.0, 0.95, 0.8),
        key: light.sun,
        key_colour: light.sun_colour * 0.8,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.35, 0.35, 0.32),
    }
}

/// One brick on the stud grid of its [`Site`]: `width` studs along x and `depth` along z from the
/// stud at (`x`, `z`), standing on course `layer`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Brick {
    x: i32,
    z: i32,
    width: i32,
    depth: i32,
    layer: i32,
    colour: Vec3,
}

/// Something built of bricks: its stud grid's corner in the world, the height it floats above
/// the baseplate, when in the build it starts and the bricks.
struct Site {
    origin: Vec2,
    lift: f32,
    start: f32,
    bricks: Vec<Brick>,
}

/// Splits a run of `length` studs into brick lengths of at most four, opening with a short
/// brick when `stagger` so that neighbouring courses break their joints.
fn chunks(length: i32, stagger: bool) -> Vec<i32> {
    let mut left = length;
    let mut lengths = Vec::new();
    if stagger && left > 3 {
        lengths.push(2);
        left -= 2;
    }
    while left > 0 {
        let long = left.min(4);
        lengths.push(long);
        left -= long;
    }
    lengths
}

/// Fills the rectangle of studs from (`x0`, `z0`) to (`x1`, `z1`), exclusive, with bricks in one
/// colour.
fn tile(
    bricks: &mut Vec<Brick>,
    layer: i32,
    (x0, z0): (i32, i32),
    (x1, z1): (i32, i32),
    colour: Vec3,
) {
    for (row, z) in (z0..z1).step_by(2).enumerate() {
        let mut x = x0;
        for width in chunks(x1 - x0, (layer + row as i32) % 2 == 1) {
            bricks.push(Brick {
                x,
                z,
                width,
                depth: (z1 - z).min(2),
                layer,
                colour,
            });
            x += width;
        }
    }
}

/// Lays the coloured `cells` of one straight wall course (each a stud position and colour,
/// ordered along the wall) as bricks one stud thick, joining neighbours of one colour.
fn wall(bricks: &mut Vec<Brick>, layer: i32, along_x: bool, cells: &[((i32, i32), Vec3)]) {
    let mut start = 0;
    while start < cells.len() {
        let colour = cells[start].1;
        let mut end = start;
        while end < cells.len() && cells[end].1 == colour {
            end += 1;
        }
        let mut at = start;
        for length in chunks((end - start) as i32, layer % 2 == 1) {
            let (x, z) = cells[at].0;
            let (width, depth) = if along_x { (length, 1) } else { (1, length) };
            bricks.push(Brick {
                x,
                z,
                width,
                depth,
                layer,
                colour,
            });
            at += length as usize;
        }
        start = end;
    }
}

fn house(origin: Vec2, start: f32) -> Site {
    let (w, d) = (8, 6);
    let colour_at = |layer: i32, x: i32, z: i32| {
        let front = z == d - 1;
        if front && (3..5).contains(&x) && layer < 3 {
            RED
        } else if (1..3).contains(&layer)
            && ((front && (x == 1 || x == w - 2))
                || ((x == 0 || x == w - 1) && (2..4).contains(&z)))
        {
            SKY_BLUE
        } else {
            WHITE
        }
    };
    let mut bricks = Vec::new();
    for layer in 0..5 {
        let even = layer % 2 == 0;
        let (across, down) = if even {
            (0..w, 1..d - 1)
        } else {
            (1..w - 1, 0..d)
        };
        for z in [0, d - 1] {
            let cells: Vec<_> = across
                .clone()
                .map(|x| ((x, z), colour_at(layer, x, z)))
                .collect();
            wall(&mut bricks, layer, true, &cells);
        }
        for x in [0, w - 1] {
            let cells: Vec<_> = down
                .clone()
                .map(|z| ((x, z), colour_at(layer, x, z)))
                .collect();
            wall(&mut bricks, layer, false, &cells);
        }
    }
    tile(&mut bricks, 5, (0, 0), (w, d), RED);
    tile(&mut bricks, 6, (1, 1), (w - 1, d - 1), RED);
    tile(&mut bricks, 7, (2, 2), (w - 2, d - 2), RED);
    tile(&mut bricks, 8, (3, 2), (w - 3, d - 2), RED);
    for layer in 7..10 {
        bricks.push(Brick {
            x: w - 2,
            z: 1,
            width: 1,
            depth: 1,
            layer,
            colour: GREY,
        });
    }
    Site {
        origin,
        lift: 0.0,
        start,
        bricks,
    }
}

/// A pine: a brown trunk and canopies of green plates, narrowing to a tip.
fn pine(origin: Vec2, start: f32, green: Vec3) -> Site {
    let mut bricks = Vec::new();
    for layer in 0..3 {
        bricks.push(Brick {
            x: 2,
            z: 2,
            width: 1,
            depth: 1,
            layer,
            colour: BROWN,
        });
    }
    tile(&mut bricks, 3, (0, 0), (5, 5), green);
    tile(&mut bricks, 4, (1, 1), (4, 4), green);
    tile(&mut bricks, 5, (1, 1), (4, 4), green * 1.25);
    bricks.push(Brick {
        x: 2,
        z: 2,
        width: 1,
        depth: 1,
        layer: 6,
        colour: green * 1.25,
    });
    Site {
        origin,
        lift: 0.0,
        start,
        bricks,
    }
}

/// A flower: a green stem and a bright one-stud head.
fn flower(origin: Vec2, start: f32, head: Vec3) -> Site {
    let bricks = vec![
        Brick {
            x: 0,
            z: 0,
            width: 1,
            depth: 1,
            layer: 0,
            colour: GREEN,
        },
        Brick {
            x: 0,
            z: 0,
            width: 1,
            depth: 1,
            layer: 1,
            colour: head,
        },
    ];
    Site {
        origin,
        lift: 0.0,
        start,
        bricks,
    }
}

/// A red car on its own stud grid centred on the origin, which the vertex shader drives round the
/// road; its wheels are built separately by [`wheels`].
fn car(start: f32) -> Site {
    let mut bricks = Vec::new();
    tile(&mut bricks, 0, (0, 0), (6, 3), RED);
    for (x, width, colour) in [(0, 2, RED), (2, 3, SKY_BLUE), (5, 1, RED)] {
        bricks.push(Brick {
            x,
            z: 0,
            width,
            depth: 3,
            layer: 1,
            colour,
        });
    }
    tile(&mut bricks, 2, (2, 0), (5, 3), RED);
    Site {
        origin: Vec2::new(-3.0 * PITCH, -1.5 * PITCH),
        lift: 0.12,
        start,
        bricks,
    }
}

/// The car's four wheels, in the car's own space.
fn wheels(start: f32) -> Shape {
    let mut shape = Shape::new();
    let centre = Vec3::new(0.0, FLOOR + 0.2, 0.0);
    for x in [-0.62, 0.62] {
        for side in [-1.0, 1.0] {
            let inner = Vec3::new(x, 0.0, side * 0.5) + centre;
            let outer = Vec3::new(x, 0.0, side * 0.68) + centre;
            shape.append(&Shape::limb(inner, outer, 0.2, 0.2, 12, BLACK));
            shape.append(&Shape::limb(
                outer,
                outer + Vec3::new(0.0, 0.0, side * 0.01),
                0.11,
                0.11,
                10,
                GREY,
            ));
        }
    }
    shape.with_seed(start)
}

/// The mesh of a site's bricks, studs only where nothing sits on top, each brick tagged with the
/// time it drops in as a fraction of [`BUILD`].
fn bake(site: &Site, rng: &mut Rng) -> Shape {
    let covered: HashSet<(i32, i32, i32)> = site
        .bricks
        .iter()
        .flat_map(|b| {
            (0..b.width).flat_map(move |i| (0..b.depth).map(move |j| (b.x + i, b.z + j, b.layer)))
        })
        .collect();
    let mut shape = Shape::new();
    for (index, brick) in site.bricks.iter().enumerate() {
        let min = Vec3::new(
            site.origin.x + brick.x as f32 * PITCH,
            FLOOR + site.lift + brick.layer as f32 * BRICK_HEIGHT,
            site.origin.y + brick.z as f32 * PITCH,
        );
        let max = min
            + Vec3::new(
                brick.width as f32 * PITCH,
                BRICK_HEIGHT,
                brick.depth as f32 * PITCH,
            );
        let inset = Vec3::new(SEAM, 0.0, SEAM);
        let mut piece = Shape::cuboid(min + inset, max - inset, brick.colour);
        for i in 0..brick.width {
            for j in 0..brick.depth {
                let (x, z) = (brick.x + i, brick.z + j);
                if !covered.contains(&(x, z, brick.layer + 1)) {
                    let top = Vec3::new(
                        min.x + (i as f32 + 0.5) * PITCH,
                        max.y,
                        min.z + (j as f32 + 0.5) * PITCH,
                    );
                    piece.append(
                        &Shape::cylinder(STUD_RADIUS, STUD_RADIUS, STUD_HEIGHT, 10, brick.colour)
                            .transformed(&Mat4::translation_vec(top)),
                    );
                }
            }
        }
        let seed =
            site.start + brick.layer as f32 * 0.05 + index as f32 * 0.0006 + rng.range(0.0, 0.01);
        shape.append(&piece.with_seed(seed.min(0.999)));
    }
    shape
}

/// A skyline of giant stacked blocks far behind the baseplate, for the haze to swallow.
fn towers(rng: &mut Rng) -> Shape {
    let palette = [RED, YELLOW, BLUE, WHITE, SKY_BLUE, ORANGE, GREEN];
    let mut shape = Shape::new();
    let mut x = -170.0;
    while x < 170.0 {
        let z = rng.range(-120.0, -75.0);
        let width = rng.range(9.0, 18.0);
        let depth = rng.range(9.0, 16.0);
        let mut top = FLOOR;
        for _ in 0..rng.index(3) + 1 {
            let colour = palette[rng.index(palette.len())];
            let shrink = rng.range(0.7, 1.0);
            let (w, d) = (width * shrink, depth * shrink);
            let height = rng.range(8.0, 22.0);
            let block = Shape::cuboid(
                Vec3::new(x - w * 0.5, top, z - d * 0.5),
                Vec3::new(x + w * 0.5, top + height, z + d * 0.5),
                colour,
            );
            shape.append(&block);
            top += height;
            let studs = (w / 3.0).floor().max(1.0) as i32;
            for i in 0..studs {
                for j in 0..((d / 3.0).floor().max(1.0) as i32) {
                    let at = Vec3::new(
                        x - w * 0.5 + (i as f32 + 0.5) * w / studs as f32,
                        top,
                        z - d * 0.5 + (j as f32 + 0.5) * 3.0,
                    );
                    shape.append(
                        &Shape::cylinder(1.0, 1.0, 0.6, 8, colour)
                            .transformed(&Mat4::translation_vec(at)),
                    );
                }
            }
        }
        x += width + rng.range(2.0, 12.0);
    }
    shape
}

/// Studs along the chip's edge, in its own space, as on a plate.
fn chip_studs() -> Shape {
    let count = 11;
    let step = (2.0 * HALF - 0.6) / (count - 1) as f32;
    let edge = HALF - 0.2;
    let mut shape = Shape::new();
    for k in 0..count {
        let along = -HALF + 0.3 + k as f32 * step;
        let mut spots = vec![Vec3::new(along, 0.0, -edge), Vec3::new(along, 0.0, edge)];
        if k > 0 && k < count - 1 {
            spots.extend([Vec3::new(-edge, 0.0, along), Vec3::new(edge, 0.0, along)]);
        }
        for spot in spots {
            let stud = Shape::cylinder(0.1, 0.1, 0.08, 10, YELLOW * 1.05)
                .transformed(&Mat4::translation_vec(spot));
            shape.append(&stud);
        }
    }
    shape
}

/// A plate of the baseplate's ground: a polar mesh flat at [`FLOOR`] out to the haze.
fn ground() -> Shape {
    terrain(
        &ring_radii(1.5, 400.0, 36),
        48,
        |_, _| FLOOR,
        |_, _| Vec3::ONE,
    )
}

#[derive(Clone, Copy)]
enum Pose {
    /// Turned about the vertical through the shape's origin, then moved to `position`.
    World { position: Vec3, yaw: f32 },
    /// Posed with the chip.
    OnChip,
}

#[derive(Clone)]
struct BrickUniforms {
    view: ViewRef,
    lighting: Lighting,
    pose: Pose,
    /// Whether bricks drop in (tagged with their start by seed) or are always in place.
    dropping: bool,
    /// Seconds since the world came on screen.
    time: f32,
}

#[derive(Interpolate, SimdInterpolate)]
struct BrickVaryings {
    world: Vec3,
    normal: Vec3,
    colour: Vec3,
}

/// Poses a brick shape and lowers each brick from above at the time its seed gives.
struct BrickVertexShader;

impl VertexShader for BrickVertexShader {
    type Vertex = KitVertex;
    type Uniforms = BrickUniforms;
    type Varyings = BrickVaryings;

    fn shade(&self, vertex: KitVertex, u: &BrickUniforms) -> (Vec4, BrickVaryings) {
        let (mut position, normal) = match u.pose {
            Pose::World { position, yaw } => {
                let (sin, cos) = yaw.sin_cos();
                let turn = |p: Vec3| Vec3::new(p.x * cos + p.z * sin, p.y, -p.x * sin + p.z * cos);
                (turn(vertex.position) + position, turn(vertex.normal))
            }
            Pose::OnChip => {
                let chip = u.view.frame.chip;
                (
                    transform_point(&chip, vertex.position),
                    (chip * vertex.normal.to_direction4()).xyz(),
                )
            }
        };
        if u.dropping {
            let fall = ((u.time + LEAD - vertex.seed * BUILD) / DROP).clamp(0.0, 1.0);
            position.y += DROP_HEIGHT * (1.0 - fall * fall);
        }
        (
            u.view.view_projection * position.to_point4(),
            BrickVaryings {
                world: position,
                normal,
                colour: vertex.colour,
            },
        )
    }
}

/// Glossy plastic: the sun's hard highlight and a reflection of the sky, stronger at grazing
/// angles, with the shade deepening where a brick meets the baseplate.
struct PlasticShader;

impl FragmentShaderSimd<BrickVaryings> for PlasticShader {
    type Uniforms = BrickUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: BrickVaryingsSimd, u: &BrickUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v3(v.world);
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let view_dir = to_eye * (s(1.0) / distance);
        let normal = facing(v3(v.normal).normalise(), to_eye);
        let height = (world.y - s(FLOOR)).fast_max(s(0.0));
        let occlusion = s(0.55) + s(0.45) * smoothstep(0.0, 0.5, height);
        let mut colour = u
            .lighting
            .shade(view, world, normal, v3(v.colour) * occlusion, s(1.0));
        let half = (splat3(u.lighting.sun) + view_dir).normalise();
        let highlight = pow_highlight(normal.dot(half).fast_max(s(0.0)), 128.0) * s(1.3);
        let turned = s(1.0) - clamp01(normal.dot(view_dir));
        let fresnel = s(0.05) + turned * turned * turned * s(0.5);
        let mirrored = reflect(-view_dir, normal);
        let environment = mix3(
            rgb(0.3, 0.28, 0.22),
            rgb(0.55, 0.72, 1.0),
            smoothstep(-0.2, 0.5, mirrored.y),
        );
        colour = colour + splat3(u.lighting.sun_colour) * highlight + environment * fresnel;
        scene_colour(view, colour, world)
    }
}

#[derive(Clone)]
struct GroundUniforms {
    toon: ToonUniforms,
    shadow: Arc<GroundShadow>,
    /// How much of the baked shadow shows, as the build casts it.
    built: f32,
}

/// The baseplate: green with studs, plate seams and a ring road of smooth grey, lit by the sun
/// with the build's shadow and the chip's. Detail fades into plain colour with distance.
struct GroundShader;

impl FragmentShaderSimd<KitVaryings> for GroundShader {
    type Uniforms = GroundUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GroundUniforms) -> ColourSimd {
        let view = &u.toon.view;
        let lighting = &u.toon.lighting;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let view_dir = to_eye * (s(1.0) / distance);
        let detail = s(1.0) - smoothstep(9.0, 34.0, distance);

        let (dx, dz) = (world.x - s(ROAD_CENTRE.0), world.z - s(ROAD_CENTRE.1));
        let radius = (dx * dx + dz * dz).sqrt();
        let road = smoothstep(ROAD_INNER - 0.04, ROAD_INNER + 0.04, radius)
            * (s(1.0) - smoothstep(ROAD_OUTER - 0.04, ROAD_OUTER + 0.04, radius));

        let (cx, cz) = (world.x * s(1.0 / PITCH), world.z * s(1.0 / PITCH));
        let (fx, fz) = (fract(cx) - s(0.5), fract(cz) - s(0.5));
        let off = (fx * fx + fz * fz).sqrt().fast_max(s(1e-4));
        let r = off * s(PITCH);
        let rim = smoothstep(STUD_RADIUS * 0.75, STUD_RADIUS, r)
            * (s(1.0) - smoothstep(STUD_RADIUS, STUD_RADIUS + 0.01, r));
        let lean = rim * s(1.8) * detail;
        let normal = Vec3Simd::new(fx / off * lean, s(1.0), fz / off * lean).normalise();
        let foot = (s(1.0) - smoothstep(STUD_RADIUS, STUD_RADIUS + 0.07, r))
            * smoothstep(STUD_RADIUS - 0.01, STUD_RADIUS + 0.01, r);
        let occlusion = s(1.0) - foot * s(0.35) * detail * (s(1.0) - road);

        let (sx, sz) = (world.x * s(1.0 / PLATE), world.z * s(1.0 / PLATE));
        let seam_x = (fract(sx) - s(0.5)).abs() * s(2.0);
        let seam_z = (fract(sz) - s(0.5)).abs() * s(2.0);
        let seam = smoothstep(0.993, 0.998, seam_x.fast_max(seam_z)) * detail;

        let (ix, iz) = ((sx).floor(), (sz).floor());
        let plate = hash2(ix * s(3.7), iz * s(5.1));
        let stud_tone = hash2(cx.floor(), cz.floor());
        let grass = mix3(
            rgb(0.1, 0.33, 0.07),
            rgb(0.15, 0.4, 0.08),
            plate * s(0.7) + stud_tone * s(0.3) * detail,
        );
        let top = s(1.0) + smoothstep(0.0, STUD_RADIUS, STUD_RADIUS - r) * s(0.08) * detail;
        let dash = smoothstep(
            0.45,
            0.5,
            (fract(dz.atan2(dx) * s(14.0 / TAU)) - s(0.5)).abs(),
        ) * (s(1.0)
            - smoothstep(
                0.03,
                0.06,
                (radius - s(0.5 * (ROAD_INNER + ROAD_OUTER))).abs(),
            ));
        let edge_line = (s(1.0) - smoothstep(0.02, 0.05, (radius - s(ROAD_INNER + 0.12)).abs()))
            + (s(1.0) - smoothstep(0.02, 0.05, (radius - s(ROAD_OUTER - 0.12)).abs()));
        let tarmac = mix3(
            rgb(0.1, 0.1, 0.11),
            rgb(0.7, 0.68, 0.6),
            clamp01(dash * s(0.9) + edge_line * s(0.6)),
        );
        let along = Vec3::new(PATH.1.0 - PATH.0.0, 0.0, PATH.1.1 - PATH.0.1);
        let path_length = (along.x * along.x + along.z * along.z).sqrt();
        let (ax, az) = (along.x / path_length, along.z / path_length);
        let (px, pz) = (world.x - s(PATH.0.0), world.z - s(PATH.0.1));
        let down = px * s(ax) + pz * s(az);
        let across = (pz * s(ax) - px * s(az)).abs();
        let on_path = smoothstep(0.38, 0.34, across.fast_min(s(1.0)))
            * smoothstep(-0.05, 0.05, down)
            * smoothstep(path_length + 0.05, path_length - 0.05, down);
        let paving = mix3(
            rgb(0.55, 0.45, 0.3),
            rgb(0.62, 0.52, 0.36),
            hash2(
                (down * s(1.0 / PITCH)).floor(),
                (across * s(1.0 / PITCH)).floor(),
            ),
        );
        let (qx, qz) = (
            (world.x - s(POND.0)) / s(POND.2),
            (world.z - s(POND.1)) / s(POND.3),
        );
        let pond = smoothstep(1.0, 0.92, (qx * qx + qz * qz).sqrt());
        let albedo = mix3(grass * top, tarmac, road);
        let albedo = mix3(albedo, paving, on_path * (s(1.0) - road));
        let albedo = mix3(albedo, rgb(0.05, 0.28, 0.55), pond);
        let flat = clamp01(road + on_path + pond);
        let albedo = albedo * (s(1.0) - seam * s(0.18));

        let lit = u.shadow.lit(world.x, world.z) * u.built + s(1.0 - u.built);
        let blocked = chip_shadow(view, lighting.sun, world);
        let shadow = lit * (s(1.0) - blocked * s(0.85));
        let normal = mix_normal(normal, flat);
        let mut colour = lighting.shade(view, world, normal, albedo * occlusion, shadow);
        let half = (splat3(lighting.sun) + view_dir).normalise();
        let gloss = pow_highlight(normal.dot(half).fast_max(s(0.0)), 64.0)
            * (s(0.25) + flat * s(0.35))
            * shadow;
        colour = colour
            + splat3(lighting.sun_colour) * gloss * (s(1.0) - smoothstep(20.0, 60.0, distance));
        scene_colour(view, colour, world)
    }
}

/// The ground's normal, flattened on the road where there are no studs.
#[inline(always)]
fn mix_normal(normal: Vec3Simd, flat: F) -> Vec3Simd {
    mix3(normal, Vec3Simd::new(s(0.0), s(1.0), s(0.0)), flat)
}

pub struct ToyBrick {
    ground: Shape,
    bricks: Shape,
    car: Shape,
    wheels: Shape,
    towers: Shape,
    studs: Shape,
    shadow: Arc<GroundShadow>,
    dust: Particles,
    ground_pipeline: SimdPipeline<crate::themes::kit::draw::KitVertexShader, GroundShader>,
    brick_pipeline: SimdPipeline<BrickVertexShader, PlasticShader>,
    sky: SimdPipeline<ScreenVertexShader, SkyShader>,
}

impl ToyBrick {
    pub fn new() -> Self {
        let mut rng = Rng::new(1958);
        let sites = [
            house(Vec2::new(-8.8, -4.8), 0.0),
            pine(Vec2::new(5.9, -5.6), 0.12, GREEN),
            pine(Vec2::new(8.2, -1.8), 0.2, LIME),
            pine(Vec2::new(-6.6, 0.4), 0.28, GREEN),
            pine(Vec2::new(-9.0, 2.6), 0.34, LIME),
            pine(Vec2::new(2.4, -10.0), 0.1, GREEN),
        ];
        let heads = [RED, YELLOW, ORANGE, WHITE, SKY_BLUE];
        let flowers: Vec<Site> = [
            (-3.8, 2.0),
            (4.6, 1.8),
            (6.8, 0.2),
            (-5.6, -2.4),
            (-2.2, -6.6),
            (3.6, -7.2),
            (0.8, -8.0),
            (7.2, -3.6),
            (-4.2, -7.8),
            (9.4, -5.8),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(x, z))| {
            flower(
                Vec2::new(x, z),
                0.1 + 0.05 * (i % 6) as f32,
                heads[i % heads.len()],
            )
        })
        .collect();
        let mut bricks = Shape::new();
        for site in sites.iter().chain(&flowers) {
            bricks.append(&bake(site, &mut rng));
        }
        let shadow = GroundShadow::bake(
            Vec2::new(-30.0, -30.0),
            60.0,
            640,
            SUN.normalise(),
            |_, _| FLOOR,
            &[&bricks],
            2,
        );
        let car_site = car(0.62);
        Self {
            ground: ground(),
            car: bake(&car_site, &mut rng),
            wheels: wheels(0.62),
            bricks,
            towers: towers(&mut rng),
            studs: chip_studs(),
            shadow: Arc::new(shadow),
            dust: Particles::new(90, 58),
            ground_pipeline: pipeline(GroundShader, CullingMode::None),
            brick_pipeline: SimdPipeline::new(BrickVertexShader, PlasticShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DEFAULT),
            sky: SimdPipeline::new(ScreenVertexShader, SkyShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }

    /// Where the car is and which way it faces at `time`.
    fn car_pose(time: f32) -> Pose {
        let angle = time * CAR_SPEED + 1.2;
        let radius = 0.5 * (ROAD_INNER + ROAD_OUTER);
        Pose::World {
            position: Vec3::new(
                ROAD_CENTRE.0 + radius * angle.cos(),
                0.0,
                ROAD_CENTRE.1 + radius * angle.sin(),
            ),
            yaw: (-angle.cos()).atan2(-angle.sin()),
        }
    }

    /// When, in seconds after the world appears, bricks land, to the nearest thirtieth: many land
    /// together.
    fn landings(&self) -> Vec<f32> {
        let mut frames: Vec<i32> = self
            .bricks
            .vertices()
            .iter()
            .chain(self.car.vertices())
            .map(|vertex| ((vertex.seed * BUILD + DROP - LEAD) * 30.0).round() as i32)
            .filter(|&frame| frame >= 0)
            .collect();
        frames.sort_unstable();
        frames.dedup();
        frames
            .into_iter()
            .map(|frame| frame as f32 / 30.0)
            .collect()
    }
}

impl Stage for ToyBrick {
    fn sound(&self) -> WorldSound {
        let clicks = self
            .landings()
            .into_iter()
            .map(|at| Ambience::once(Source::Recording("click", 0.0), at, 0.12, 0.08, 0.0));
        WorldSound {
            leads: &[Instrument::Xylophone],
            ambience: clicks.collect(),
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let time = scene.local_time;
        let built = smoothstep(0.0, 1.0, s((time + LEAD) / (BUILD + DROP))).to_array()[0];
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let bricks = |lighting, pose, dropping| BrickUniforms {
            view: view.clone(),
            lighting,
            pose,
            dropping,
            time,
        };
        let origin = Pose::World {
            position: Vec3::ZERO,
            yaw: 0.0,
        };

        self.ground.draw(
            pass,
            &self.ground_pipeline,
            crate::themes::kit::draw::KitUniforms {
                view: view.clone(),
                motion: crate::themes::kit::draw::Motion::Still,
            },
            GroundUniforms {
                toon,
                shadow: self.shadow.clone(),
                built,
            },
        );
        let uniforms = bricks(lighting(), origin, true);
        self.bricks
            .draw(pass, &self.brick_pipeline, uniforms.clone(), uniforms);
        let driving = bricks(lighting(), Self::car_pose(time), true);
        self.car
            .draw(pass, &self.brick_pipeline, driving.clone(), driving);
        let rolling = bricks(lighting(), Self::car_pose(time), true);
        self.wheels
            .draw(pass, &self.brick_pipeline, rolling.clone(), rolling);
        let backdrop = bricks(tower_lighting(), origin, false);
        self.towers
            .draw(pass, &self.brick_pipeline, backdrop.clone(), backdrop);

        scene.chip.draw(pass, view, &look());
        let on_chip = bricks(lighting(), Pose::OnChip, false);
        self.studs
            .draw(pass, &self.brick_pipeline, on_chip.clone(), on_chip);

        let sky = SkyUniforms {
            view: view.clone(),
            sky: sky(),
        };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| sky.clone());
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 1.5, -2.0),
                half_extent: Vec3::new(15.0, 4.5, 12.0),
                velocity: Vec3::new(0.25, 0.1, 0.05),
                wander: 0.5,
                size: (0.015, 0.04),
                colour: Vec3::new(1.5, 1.4, 1.2),
                twinkle: 1.5,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.25,
            vignette: 0.25,
            grain: 0.008,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studs_show_only_where_no_brick_sits_on_top() {
        let bricks = vec![
            Brick {
                x: 0,
                z: 0,
                width: 2,
                depth: 1,
                layer: 0,
                colour: RED,
            },
            Brick {
                x: 0,
                z: 0,
                width: 1,
                depth: 1,
                layer: 1,
                colour: RED,
            },
        ];
        let site = Site {
            origin: Vec2::new(0.0, 0.0),
            lift: 0.0,
            start: 0.0,
            bricks,
        };
        let with_top = bake(&site, &mut Rng::new(1)).triangle_count();
        let alone = bake(
            &Site {
                bricks: vec![site.bricks[0]],
                ..site
            },
            &mut Rng::new(1),
        )
        .triangle_count();
        let one_stud =
            Shape::cylinder(STUD_RADIUS, STUD_RADIUS, STUD_HEIGHT, 10, RED).triangle_count();
        let brick = Shape::cuboid(Vec3::ZERO, Vec3::ONE, RED).triangle_count();
        assert_eq!(alone, brick + 2 * one_stud);
        assert_eq!(with_top, 2 * brick + 2 * one_stud);
    }

    #[test]
    fn every_course_of_a_wall_is_covered_without_overlap() {
        let site = house(Vec2::new(0.0, 0.0), 0.0);
        let mut seen = HashSet::new();
        for b in &site.bricks {
            for i in 0..b.width {
                for j in 0..b.depth {
                    assert!(seen.insert((b.x + i, b.z + j, b.layer)), "overlap at {b:?}");
                }
            }
        }
    }

    #[test]
    fn brick_lengths_add_up() {
        for length in 1..12 {
            for stagger in [false, true] {
                assert_eq!(chunks(length, stagger).iter().sum::<i32>(), length);
            }
        }
    }
}
