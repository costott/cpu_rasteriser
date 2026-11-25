//! The hero: the chip, and how a stage restyles it.
//!
//! The chip is a bare-die package modelled lying flat, centred on the origin, with its die facing
//! +y; the director's pose in [`Frame::chip`](crate::shaders::Frame::chip) places it. Stages draw
//! it with [`Chip::draw`] in a [`Look`] of their own.
//!
//! The model is a few thousand triangles: the substrate with bevelled edges, the raised die on
//! its underfill, a gold seal ring, a ring of capacitors on top and a cluster underneath. Every
//! surface's detail (the die's floorplan, traces, pads, markings) is drawn by its fragment
//! shader, as are the light streams that run across the die and out to the package's edge.

use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::meshes::{MeshBuilder, model};
use crate::shaders::ViewRef;
use crate::shaders::chip::{
    ChipShader, ChipUniforms, ChipVertexShader, Part, ROUTING_COLUMNS, ROUTING_ROWS, StreamHead,
    Streams, TrailSegment,
};
use crate::simd::hash2_scalar;

/// The package's width and depth, in world units.
pub const SIZE: f32 = 4.0;
/// Half the package's width.
pub const HALF: f32 = SIZE * 0.5;
/// The package's corner radius, seen from above.
pub const CORNER: f32 = 0.12;
/// The height of the substrate's top and bottom faces.
pub const SUBSTRATE_TOP: f32 = 0.0;
pub const SUBSTRATE_BOTTOM: f32 = -0.16;
/// Where the substrate's bevelled edge meets its flat top, in from the package's outline.
pub const TOP_BEVEL: f32 = 0.035;
/// Half the die's width, its height above the substrate and its corner radius.
pub const DIE: f32 = 0.95;
pub const DIE_TOP: f32 = 0.075;
pub const DIE_CORNER: f32 = 0.02;
/// How far beyond the die's edge the gold seal ring runs, inside and outside.
pub const SEAL: (f32, f32) = (0.07, 0.1);
/// The spacing of the substrate's traces, which fan out from the die to the package's edge.
pub const LANE_PITCH: f32 = 0.08;
/// The distance from the chip's centre to the middle of each side's row of capacitors.
pub const CAP_ROW: f32 = 1.34;
/// A capacitor's length (out from the die), width and height.
pub const CAP: (f32, f32, f32) = (0.11, 0.046, 0.05);
/// The underside's capacitor cluster: columns and rows, and their spacing.
pub const LAND_CAPS: (i32, i32, f32, f32) = (4, 6, 0.17, 0.11);

/// How a stage dresses the chip: its materials and the light it sits in.
///
/// Colours are linear HDR values. The surfaces derive their detail from these few colours (the
/// floorplan's blocks are shades of `die`, its wiring is `gold`), so restyling them restyles the
/// whole chip.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    /// The package's metal: the substrate's satin top, its polished bevels and the capacitors'
    /// terminals.
    pub substrate: Vec3,
    /// The die's base colour; its blocks are lighter and darker shades of it.
    pub die: Vec3,
    /// The die's wiring, the seal ring, the traces and the pads underneath.
    pub gold: Vec3,
    /// The colour of the light streams and the perimeter glow; black turns them off.
    pub streams: Vec3,
    /// Direction towards the key light, which rims the bevels, and its colour.
    pub key: Vec3,
    pub key_colour: Vec3,
    /// What the chip's surfaces reflect above and below the horizon.
    pub sky: Vec3,
    pub ground: Vec3,
    /// The studio's softbox cards, as the metal and the die's glass reflect them; black for none.
    pub softbox: Vec3,
}

impl Default for Look {
    /// The studio look: silver package, green die, gold contacts, green data light.
    fn default() -> Self {
        Self {
            substrate: Vec3::new(0.52, 0.54, 0.57),
            die: Vec3::new(0.05, 0.2, 0.1),
            gold: Vec3::new(0.8, 0.58, 0.24),
            streams: Vec3::new(0.43, 0.9, 0.72),
            key: Vec3::new(-0.62, 0.55, -0.56).normalise(),
            key_colour: Vec3::new(3.0, 2.95, 2.85),
            sky: Vec3::new(0.1, 0.11, 0.125),
            ground: Vec3::new(0.012, 0.013, 0.015),
            softbox: Vec3::new(1.0, 1.0, 1.02),
        }
    }
}

pub struct Chip {
    model: Model<ObjVertex>,
    pipeline: SimdPipeline<ChipVertexShader, ChipShader>,
    lanes: Vec<Lane>,
}

impl Chip {
    pub fn new() -> Self {
        let mut mesh = MeshBuilder::default();
        substrate(&mut mesh);
        die(&mut mesh);
        capacitors(&mut mesh);
        Self {
            model: model(mesh.build()),
            pipeline: SimdPipeline::new(ChipVertexShader, ChipShader)
                .with_depth_state(DepthState::DEFAULT),
            lanes: lanes(),
        }
    }

    /// Draws the chip, posed by the view's frame, in `look`.
    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, view: &ViewRef, look: &Look) {
        let uniforms = ChipUniforms {
            view: view.clone(),
            look: *look,
            streams: Arc::new(self.streams(view.frame.time, view.frame.activity)),
        };
        self.model
            .draw_to_render_pass_simd(pass, &self.pipeline, uniforms.clone(), |_| uniforms.clone());
    }

    /// Where the light streams are at `time`. Lanes come on one by one as `activity` rises.
    fn streams(&self, time: f32, activity: f32) -> Streams {
        let mut streams = Streams::default();
        let count = self.lanes.len() as f32;
        for (i, lane) in self.lanes.iter().enumerate() {
            let on = ((activity * count * 1.25 - i as f32) * 0.5).clamp(0.0, 1.0);
            if on > 0.0 {
                lane.light(time, lane.brightness * on * activity, &mut streams);
            }
        }
        streams
    }
}

/// The substrate: a slab with rounded corners, a bevelled top and bottom edge and straight sides,
/// so its layers show from the side.
fn substrate(mesh: &mut MeshBuilder) {
    let bottom_bevel = 0.03;
    loft(
        mesh,
        (HALF, CORNER),
        &[
            (-bottom_bevel, SUBSTRATE_BOTTOM),
            (0.0, SUBSTRATE_BOTTOM + 0.025),
        ],
        Part::Bevel,
    );
    loft(
        mesh,
        (HALF, CORNER),
        &[
            (0.0, SUBSTRATE_BOTTOM + 0.025),
            (0.0, SUBSTRATE_TOP - 0.028),
        ],
        Part::Edge,
    );
    loft(
        mesh,
        (HALF, CORNER),
        &[(0.0, SUBSTRATE_TOP - 0.028), (-TOP_BEVEL, SUBSTRATE_TOP)],
        Part::Bevel,
    );
    cap(
        mesh,
        (HALF, CORNER),
        -TOP_BEVEL,
        SUBSTRATE_TOP,
        true,
        Part::Top,
    );
    cap(
        mesh,
        (HALF, CORNER),
        -bottom_bevel,
        SUBSTRATE_BOTTOM,
        false,
        Part::Bottom,
    );
}

/// The die on its underfill, and the gold seal ring around it.
fn die(mesh: &mut MeshBuilder) {
    let outline = (DIE, DIE_CORNER);
    let chamfer = 0.015;
    loft(
        mesh,
        outline,
        &[(0.045, SUBSTRATE_TOP), (0.0, 0.03)],
        Part::Underfill,
    );
    loft(
        mesh,
        outline,
        &[(0.0, 0.03), (0.0, DIE_TOP - chamfer)],
        Part::DieSide,
    );
    loft(
        mesh,
        outline,
        &[(0.0, DIE_TOP - chamfer), (-chamfer, DIE_TOP)],
        Part::DieChamfer,
    );
    cap(mesh, outline, -chamfer, DIE_TOP, true, Part::DieTop);
    let height = 0.012;
    loft(
        mesh,
        outline,
        &[
            (SEAL.1, SUBSTRATE_TOP),
            (SEAL.1, height),
            (SEAL.0, height),
            (SEAL.0, SUBSTRATE_TOP),
        ],
        Part::Seal,
    );
}

/// A row of capacitors along each side of the die, in groups between the traces, and a cluster
/// on the underside.
fn capacitors(mesh: &mut MeshBuilder) {
    let (length, width, height) = CAP;
    let colour = Part::Capacitor.colour();
    for side in 0..4 {
        let turn = Mat4::rotate_y(side as f32 * FRAC_PI_2);
        for k in -10i32..10 {
            if k.rem_euclid(5) == 2 {
                continue;
            }
            let along = (k as f32 + 0.5) * LANE_PITCH;
            let min = Vec3::new(along - width * 0.5, SUBSTRATE_TOP, CAP_ROW - length * 0.5);
            let max = Vec3::new(
                along + width * 0.5,
                SUBSTRATE_TOP + height,
                CAP_ROW + length * 0.5,
            );
            let [a, b] = [min, max].map(|p| {
                let p = turn * p.to_point4();
                Vec3::new(p.x, p.y, p.z)
            });
            mesh.cuboid(
                Vec3::new(a.x.min(b.x), a.y, a.z.min(b.z)),
                Vec3::new(a.x.max(b.x), b.y, a.z.max(b.z)),
                colour,
            );
        }
    }
    let (columns, rows, dx, dz) = LAND_CAPS;
    let colour = Part::LandCapacitor.colour();
    for column in 0..columns {
        for row in 0..rows {
            let x = (column as f32 - (columns - 1) as f32 * 0.5) * dx;
            let z = (row as f32 - (rows - 1) as f32 * 0.5) * dz;
            mesh.cuboid(
                Vec3::new(
                    x - length * 0.5,
                    SUBSTRATE_BOTTOM - height * 0.8,
                    z - width * 0.5,
                ),
                Vec3::new(x + length * 0.5, SUBSTRATE_BOTTOM, z + width * 0.5),
                colour,
            );
        }
    }
}

/// The outline of a square of half-width `outline.0` with corners rounded to radius `outline.1`,
/// as points with their outward normals, going round anticlockwise seen from above.
fn outline((half, radius): (f32, f32)) -> Vec<(Vec2, Vec2)> {
    const STEPS: usize = 6;
    let mut points = Vec::with_capacity(4 * (STEPS + 1));
    for corner in 0..4 {
        let start = corner as f32 * FRAC_PI_2;
        let centre = Vec2::new(
            if corner == 0 || corner == 3 {
                half - radius
            } else {
                radius - half
            },
            if corner < 2 {
                half - radius
            } else {
                radius - half
            },
        );
        for step in 0..=STEPS {
            let angle = start + FRAC_PI_2 * step as f32 / STEPS as f32;
            let normal = Vec2::new(angle.cos(), angle.sin());
            points.push((centre + normal * radius, normal));
        }
    }
    points
}

/// Sweeps a profile round an outline: each pair of consecutive `(offset, height)` points makes a
/// band, `offset` out from the outline at that height. The profile runs round the solid's
/// cross-section anticlockwise (outward to the right, up), so each band faces away from the solid.
fn loft(mesh: &mut MeshBuilder, shape: (f32, f32), profile: &[(f32, f32)], part: Part) {
    let ring = outline(shape);
    let colour = part.colour();
    for pair in profile.windows(2) {
        let ((o0, y0), (o1, y1)) = (pair[0], pair[1]);
        let slope = Vec2::new(y1 - y0, o0 - o1);
        let slope = slope * (1.0 / slope.length());
        let rows: Vec<[u32; 2]> = ring
            .iter()
            .map(|&(point, out)| {
                let normal = Vec3::new(out.x * slope.x, slope.y, out.y * slope.x);
                let at = |o: f32, y: f32| {
                    let p = point + out * o;
                    Vec3::new(p.x, y, p.y)
                };
                [
                    mesh.vertex(at(o0, y0), normal, colour),
                    mesh.vertex(at(o1, y1), normal, colour),
                ]
            })
            .collect();
        for i in 0..rows.len() {
            let ([a, d], [b, c]) = (rows[i], rows[(i + 1) % rows.len()]);
            mesh.triangle(a, c, b).triangle(a, d, c);
        }
    }
}

/// A flat face filling an outline offset by `offset`, at height `y`, facing up or down.
fn cap(mesh: &mut MeshBuilder, shape: (f32, f32), offset: f32, y: f32, up: bool, part: Part) {
    let colour = part.colour();
    let normal = Vec3::new(0.0, if up { 1.0 } else { -1.0 }, 0.0);
    let centre = mesh.vertex(Vec3::new(0.0, y, 0.0), normal, colour);
    let ring: Vec<u32> = outline(shape)
        .into_iter()
        .map(|(point, out)| {
            let p = point + out * offset;
            mesh.vertex(Vec3::new(p.x, y, p.y), normal, colour)
        })
        .collect();
    for i in 0..ring.len() {
        let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
        if up {
            mesh.triangle(centre, b, a);
        } else {
            mesh.triangle(centre, a, b);
        }
    }
}

/// A route a light stream runs along, in the chip's top plane (x, z), with how it moves.
struct Lane {
    points: Vec<Vec2>,
    /// Arc length at each point.
    distances: Vec<f32>,
    /// Units per second.
    speed: f32,
    /// How far behind the head the trail fades to 1/e.
    trail: f32,
    /// Seconds between one stream starting down the lane and the next.
    period: f32,
    offset: f32,
    brightness: f32,
    /// The core's width, in pixels.
    width: f32,
}

/// How many trail lengths behind its head a trail is still drawn.
const TRAIL_REACH: f32 = 5.0;

impl Lane {
    fn new(points: Vec<Vec2>, seed: f32, packet: bool) -> Self {
        let distances = points
            .iter()
            .scan((0.0, points[0]), |(total, last), &p| {
                *total += (p - *last).length();
                *last = p;
                Some(*total)
            })
            .collect::<Vec<_>>();
        let random = |salt: f32| hash2_scalar(seed * 7.31 + salt, salt * 3.17 + 1.3);
        let length = *distances.last().expect("a lane has points");
        let speed = 2.6 * (0.7 + 0.6 * random(1.0));
        let trail = if packet {
            0.05
        } else {
            // Mostly medium trails, with a few long ones: a 3-5x spread between neighbours.
            0.22 + 1.1 * random(2.0).powf(1.6)
        };
        let travel = (length + trail * TRAIL_REACH) / speed;
        Self {
            points,
            distances,
            speed,
            trail,
            period: travel + 0.3 + 2.4 * random(3.0),
            offset: 40.0 * random(4.0),
            brightness: if packet {
                1.6
            } else {
                0.65 + 0.45 * random(5.0)
            },
            width: if packet { 3.2 } else { 1.6 + 0.9 * random(6.0) },
        }
    }

    /// Adds this lane's trail and head at `time` to `streams`.
    fn light(&self, time: f32, brightness: f32, streams: &mut Streams) {
        let head = (time + self.offset).rem_euclid(self.period) * self.speed;
        let length = *self.distances.last().expect("a lane has points");
        let tail = head - self.trail * TRAIL_REACH;
        if tail >= length {
            return;
        }
        for i in 1..self.points.len() {
            let (s0, s1) = (self.distances[i - 1], self.distances[i]);
            let (from, to) = (tail.max(s0), head.min(s1));
            if from >= to {
                continue;
            }
            let (a, b) = (self.points[i - 1], self.points[i]);
            let direction = (b - a) * (1.0 / (s1 - s0));
            streams.segments.push(TrailSegment::new(
                a + direction * (from - s0),
                direction,
                to - from,
                head - from,
                1.0 / self.trail,
                brightness,
                self.width,
            ));
        }
        if head < length {
            let i = self
                .distances
                .partition_point(|&d| d <= head)
                .clamp(1, self.points.len() - 1);
            let (a, b) = (self.points[i - 1], self.points[i]);
            let t = (head - self.distances[i - 1]) / (self.distances[i] - self.distances[i - 1]);
            streams.heads.push(StreamHead {
                position: a + (b - a) * t,
                brightness,
                width: self.width,
            });
        }
    }
}

/// The lanes the streams run along: in from a routing channel on the die, along it, and out along
/// a substrate trace to the package's edge (or the same in reverse), turning at right angles.
fn lanes() -> Vec<Lane> {
    const COUNT: usize = 30;
    const PACKETS: usize = 4;
    let mut lanes = Vec::with_capacity(COUNT + PACKETS);
    for i in 0..COUNT + PACKETS {
        let seed = i as f32 + 0.5;
        let random = |salt: f32| hash2_scalar(seed * 13.7 + salt * 5.3, salt + 2.9);
        let side = i % 4;
        let exit = ((random(1.0) * 19.0).floor() - 9.0) * LANE_PITCH;
        let packet = i >= COUNT;
        let mut points = if packet {
            vec![Vec2::new(exit, DIE + SEAL.1 + 0.05), Vec2::new(exit, HALF)]
        } else {
            // The run along the die follows a routing channel, which after the turn below is a
            // row for the lanes leaving north and south and a column for east and west.
            let channels = if side % 2 == 0 {
                &ROUTING_ROWS[..]
            } else {
                &ROUTING_COLUMNS[..]
            };
            let channel = channels[(random(2.0) * channels.len() as f32) as usize];
            let row = if side < 2 { channel } else { -channel } * DIE;
            let start = (random(3.0) * 1.7 - 0.85) * DIE;
            let mut points = vec![
                Vec2::new(start, row),
                Vec2::new(exit, row),
                Vec2::new(exit, HALF),
            ];
            if random(4.0) < 0.35 {
                let from = (random(5.0) * 1.6 - 0.8) * DIE;
                points.insert(0, Vec2::new(start, from));
            }
            points
        };
        if random(6.0) < 0.3 && !packet {
            points.reverse();
        }
        let turn = |p: Vec2| match side {
            0 => p,
            1 => Vec2::new(p.y, -p.x),
            2 => Vec2::new(-p.x, -p.y),
            _ => Vec2::new(-p.y, p.x),
        };
        lanes.push(Lane::new(
            points.into_iter().map(turn).collect(),
            seed,
            packet,
        ));
    }
    lanes
}
