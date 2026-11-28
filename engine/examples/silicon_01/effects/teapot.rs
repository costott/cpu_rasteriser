//! The teapot: its triangles stream up out of the die along thin beams of light, assemble the
//! Utah teapot hovering above the chip, and stream back down into it.
//!
//! Sixteen beams fan out from the die to sixteen sectors of the teapot. Each carries the
//! triangles of its sector one after another, lowest first, 40 to 60 ms apart. A triangle rides
//! its beam as a hot green packet with a streak behind it, then settles onto the teapot, cooling
//! to the material as it lands.

use std::f32::consts::TAU;
use std::path::Path;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::director::{Phase, smootherstep, window};
use crate::effects::streak::{
    Streak, StreakShader, StreakVaryings, StreakVertex, expand, hidden, streak_corners,
};
use crate::effects::{DIE_HALF, DIE_TOP, GREEN, GREEN_BRIGHT, in_phase, quad_model, random};
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{clamp01, mix3, mul3, pow2n, reflect, rgb, s, smoothstep, splat3, v3};

/// The teapot's height, in the chip's units, and where its base hovers above the chip.
const SCALE: f32 = 1.0;
const BASE: f32 = 1.5;
const LANES: usize = 16;
/// How long a triangle takes to fly up to the teapot, and back down.
const FLIGHT: f32 = 1.25;
const LEAVE_FLIGHT: f32 = 0.9;
/// When the first triangle leaves the die, and when the first triangle starts back down.
const ASSEMBLY_START: f32 = 0.5;
const LEAVE_START: f32 = 7.05;
/// The gap between a beam's triangles as the teapot is built: 40 to 52 ms per beam.
const SPACING: (f32, f32) = (0.040, 0.012);
const LEAVE_SPACING: f32 = 0.034;
/// The teapot's turn, in radians per second.
const TURN_RATE: f32 = 0.5;
/// How far behind its head a triangle's streak reaches, as a fraction of the flight.
const TRAIL: f32 = 0.2;

/// What a triangle needs to fly from the die to its place and back.
#[derive(Clone, Copy)]
pub struct Flight {
    /// Where on the die its beam starts, and where the beam ends.
    source: Vec3,
    beam_end: Vec3,
    /// The triangle's centre in the finished teapot, relative to the teapot's base.
    centroid: Vec3,
    /// When it leaves the die and when it starts back, in seconds since the teapot phase began.
    launch: f32,
    leave: f32,
    seed: f32,
}

/// Where a triangle is at `time`.
pub struct FlightState {
    /// 0 at the die, 1 landed.
    progress: f32,
    descending: bool,
    turn: Mat4,
    target: Vec3,
}

impl Flight {
    fn state(&self, time: f32) -> FlightState {
        let local = time - Phase::Teapot.start();
        let rise = ((local - self.launch) / FLIGHT).clamp(0.0, 1.0);
        let fall = ((local - self.leave) / LEAVE_FLIGHT).clamp(0.0, 1.0);
        let turn = Mat4::rotate_y(TURN_RATE * local.max(0.0));
        let bob = 0.05 * (time * 1.1).sin();
        let target = (turn * self.centroid.to_point4()).xyz() + Vec3::new(0.0, BASE + bob, 0.0);
        FlightState {
            progress: (1.0 - (1.0 - rise).powi(2)) * (1.0 - fall * fall),
            descending: fall > 0.0,
            turn,
            target,
        }
    }

    /// The point `progress` of the way along the flight: up the beam, then onto the target.
    fn path(&self, target: Vec3, progress: f32) -> Vec3 {
        let leave = self.source + (self.beam_end - self.source) * 0.7;
        let arrive = self.beam_end + (target - self.beam_end) * 0.45;
        let q = 1.0 - progress;
        self.source * (q * q * q)
            + leave * (3.0 * q * q * progress)
            + arrive * (3.0 * q * progress * progress)
            + target * (progress * progress * progress)
    }
}

#[derive(Clone, Copy)]
pub struct TeapotVertex {
    position: Vec3,
    normal: Vec3,
    barycentric: Vec3,
    flight: Flight,
}

/// A beam of light from the die, with when it is on: while the teapot builds, and while it is
/// unbuilt.
#[derive(Clone, Copy)]
pub struct Beam {
    source: Vec3,
    end: Vec3,
    building: (f32, f32),
    unbuilding: (f32, f32),
}

#[derive(Clone)]
pub struct TeapotUniforms {
    view: ViewRef,
    look: Look,
}

#[derive(Interpolate, SimdInterpolate)]
pub struct TeapotVaryings {
    world: Vec3,
    normal: Vec3,
    barycentric: Vec3,
    /// 1 for a hot packet in flight, 0 for a cooled surface.
    heat: f32,
    /// A flash as the triangle lands, fading.
    landed: f32,
}

pub struct TeapotVertexShader;

impl VertexShader for TeapotVertexShader {
    type Vertex = TeapotVertex;
    type Uniforms = TeapotUniforms;
    type Varyings = TeapotVaryings;

    fn shade(&self, vertex: TeapotVertex, u: &TeapotUniforms) -> (Vec4, TeapotVaryings) {
        let frame = &u.view.frame;
        let flight = &vertex.flight;
        let state = flight.state(frame.time);
        if state.progress <= 0.0 {
            return (
                Vec4::new(3.0, 3.0, 0.5, 1.0),
                TeapotVaryings {
                    world: Vec3::ZERO,
                    normal: Vec3::ZERO,
                    barycentric: Vec3::ZERO,
                    heat: 0.0,
                    landed: 0.0,
                },
            );
        }
        let progress = state.progress;
        let settle = smootherstep(((progress - 0.45) / 0.55).clamp(0.0, 1.0));
        let tumble = (1.0 - settle) * TAU * (0.6 + flight.seed);
        let spin =
            Mat4::rotate_x(tumble * 0.7) * Mat4::rotate_y(tumble) * Mat4::rotate_z(tumble * 0.4);
        let scale = 0.14 + 0.86 * smootherstep(((progress - 0.72) / 0.28).clamp(0.0, 1.0));
        let centre = flight.path(state.target, progress);
        let local = vertex.position - flight.centroid;
        let offset = (state.turn * (spin * local.to_direction4())).xyz() * scale;
        let normal = (state.turn * (spin * vertex.normal.to_direction4())).xyz();
        let world = frame.chip * (centre + offset).to_point4();
        let normal = frame.chip * normal.to_direction4();
        let since_landing = frame.time - Phase::Teapot.start() - flight.launch - FLIGHT;
        let landed = if since_landing >= 0.0 && !state.descending {
            (-since_landing * 9.0).exp()
        } else {
            0.0
        };
        (
            u.view.view_projection * world,
            TeapotVaryings {
                world: world.xyz(),
                normal: normal.xyz(),
                barycentric: vertex.barycentric,
                heat: 1.0 - smoothstep_scalar(0.45, 0.98, progress),
                landed,
            },
        )
    }
}

fn smoothstep_scalar(edge0: f32, edge1: f32, x: f32) -> f32 {
    smootherstep(((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0))
}

pub struct TeapotShader;

impl FragmentShaderSimd<TeapotVaryings> for TeapotShader {
    type Uniforms = TeapotUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: TeapotVaryingsSimd, u: &TeapotUniforms) -> ColourSimd {
        let look = &u.look;
        let world = v3(v.world);
        let to_eye = (splat3(u.view.eye) - world).normalise();
        let facing = v3(v.normal).normalise();
        let flip = facing.dot(to_eye).simd_lt(s(0.0)).select(s(-1.0), s(1.0));
        let normal = facing * flip;
        let n_dot_v = clamp01(normal.dot(to_eye));
        let key = splat3(look.key);

        // Polished silver: it shows the studio it reflects, with a trace of diffuse light, and
        // the cool rim light that outlines the chip.
        let ambient = mix3(
            splat3(look.ground),
            splat3(look.sky),
            normal.y * s(0.5) + s(0.5),
        );
        let diffuse = normal.dot(key).fast_max(s(0.0));
        let base = mul3(
            rgb(0.5, 0.52, 0.55),
            ambient + splat3(look.key_colour) * (diffuse * s(0.12)),
        );
        let grazing = s(1.0) - n_dot_v;
        let silver = rgb(0.9, 0.91, 0.93);
        let fresnel = pow2n(grazing, 2) * grazing;
        let reflected = reflect(-to_eye, normal);
        let reflectance = silver + (rgb(1.0, 1.0, 1.0) - silver) * fresnel;
        let mirror = mul3(reflectance, environment(reflected));
        let rim = rgb(0.45, 0.7, 0.62) * (pow2n(grazing, 1) * grazing * s(0.35));
        let surface = base + mirror + rim;

        let b = v3(v.barycentric);
        let edge = b.x.fast_min(b.y).fast_min(b.z);
        let wire = s(1.0) - smoothstep(0.0, 0.08, edge);
        let packet = splat3(GREEN_BRIGHT) * s(1.5) + rgb(0.8, 1.0, 0.9) * wire * s(0.8);
        let cooled = mix3(surface, packet, v.heat);
        let flash = splat3(GREEN_BRIGHT) * (v.landed * s(1.4));
        scene_colour(&u.view, cooled + flash, world)
    }
}

/// What the silver reflects: a dark studio lit by a large softbox overhead, a key softbox and a
/// strip light to the sides, and the green glow of the die below.
#[inline(always)]
fn environment(direction: Vec3Simd) -> Vec3Simd {
    let up = direction.y;
    let below = smoothstep(0.0, 0.9, -up);
    let backdrop = mix3(
        rgb(0.012, 0.014, 0.018),
        rgb(0.05, 0.065, 0.085),
        smoothstep(-0.2, 0.8, up),
    );
    let die_glow = splat3(GREEN) * (below * s(0.2));
    let panel = smoothstep(0.62, 0.8, up)
        * (s(1.0) - smoothstep(0.45, 0.75, direction.x.abs()))
        * (s(1.0) - smoothstep(0.5, 0.8, direction.z.abs()));
    let key = smoothstep(0.78, 0.9, direction.dot(rgb(-0.55, 0.35, 0.76).normalise()));
    let strip = smoothstep(0.93, 0.97, direction.dot(rgb(0.85, 0.2, -0.48).normalise()));
    let horizon = (-(up - s(0.02)) * (up - s(0.02)) * s(120.0)).exp();
    backdrop
        + die_glow
        + rgb(1.0, 1.0, 1.04) * (panel * s(1.8))
        + rgb(1.0, 0.97, 0.92) * (key * s(1.8))
        + rgb(0.7, 0.95, 0.9) * (strip * s(2.8))
        + rgb(0.5, 0.65, 0.7) * (horizon * s(0.25))
}

pub struct TrailVertexShader;

impl VertexShader for TrailVertexShader {
    type Vertex = StreakVertex<Flight>;
    type Uniforms = ViewRef;
    type Varyings = StreakVaryings;

    fn shade(&self, vertex: StreakVertex<Flight>, view: &ViewRef) -> (Vec4, StreakVaryings) {
        let frame = &view.frame;
        let flight = &vertex.data;
        let state = flight.state(frame.time);
        let progress = state.progress;
        if progress <= 0.002 || progress >= 0.998 {
            return hidden();
        }
        let behind = if state.descending {
            (progress + TRAIL).min(1.0)
        } else {
            (progress - TRAIL).max(0.0)
        };
        let point = |p: f32| frame.chip * flight.path(state.target, p).to_point4();
        let fade =
            smoothstep_scalar(0.0, 0.08, progress) * (1.0 - smoothstep_scalar(0.8, 1.0, progress));
        let streak = Streak {
            tail: point(behind).xyz(),
            head: point(progress).xyz(),
            width: 1.0,
            colour: GREEN_BRIGHT * (1.6 * fade),
            falloff: 1.4,
        };
        expand(view, &streak, vertex.corner)
    }
}

pub struct BeamVertexShader;

impl VertexShader for BeamVertexShader {
    type Vertex = StreakVertex<Beam>;
    type Uniforms = ViewRef;
    type Varyings = StreakVaryings;

    fn shade(&self, vertex: StreakVertex<Beam>, view: &ViewRef) -> (Vec4, StreakVaryings) {
        let frame = &view.frame;
        let beam = &vertex.data;
        let level = |(start, end): (f32, f32)| window(frame.time, start, end, 0.35);
        let on = level(beam.building).max(level(beam.unbuilding));
        let top = beam.source + (beam.end - beam.source) * on;
        let streak = Streak {
            tail: (frame.chip * top.to_point4()).xyz(),
            head: (frame.chip * beam.source.to_point4()).xyz(),
            width: 0.8,
            colour: GREEN * (0.9 * on),
            falloff: 0.6,
        };
        expand(view, &streak, vertex.corner)
    }
}

pub struct Teapot {
    triangles: Model<TeapotVertex>,
    trails: Model<StreakVertex<Flight>>,
    beams: Model<StreakVertex<Beam>>,
    material: SimdPipeline<TeapotVertexShader, TeapotShader>,
    trail: SimdPipeline<TrailVertexShader, StreakShader>,
    beam: SimdPipeline<BeamVertexShader, StreakShader>,
    look: Look,
}

impl Teapot {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut obj = load_obj(Path::new("assets/utah_teapot.obj"))?;
        obj.calculate_vertex_normals();
        let (flights, corners, beams) = flights(&obj);
        Ok(Self {
            triangles: Model::new(
                vec![Mesh::new(
                    corners.iter().flatten().copied().collect(),
                    (0..corners.len() as u32 * 3).collect(),
                    None,
                )],
                Vec::new(),
                ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
            )?,
            trails: quad_model(flights.iter().map(|f| streak_corners(*f))),
            beams: quad_model(beams.iter().map(|b| streak_corners(*b))),
            material: SimdPipeline::new(TeapotVertexShader, TeapotShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::DEFAULT),
            trail: SimdPipeline::new(TrailVertexShader, StreakShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY)
                .with_blend_state(BlendState::ADDITIVE),
            beam: SimdPipeline::new(BeamVertexShader, StreakShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY)
                .with_blend_state(BlendState::ADDITIVE),
            look: Look::default(),
        })
    }

    pub fn draw_opaque<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, view: &ViewRef) {
        if !self.active(view) {
            return;
        }
        let uniforms = TeapotUniforms {
            view: view.clone(),
            look: self.look,
        };
        self.triangles
            .draw_to_render_pass_simd(pass, &self.material, uniforms.clone(), |_| uniforms.clone());
    }

    pub fn draw_light<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, view: &ViewRef) {
        if !self.active(view) {
            return;
        }
        self.trails
            .draw_to_render_pass_simd(pass, &self.trail, view.clone(), |_| ());
        self.beams
            .draw_to_render_pass_simd(pass, &self.beam, view.clone(), |_| ());
    }

    fn active(&self, view: &ViewRef) -> bool {
        in_phase(view, Phase::Teapot)
            || (in_phase(view, Phase::Hologram) && view.frame.phase_time < 0.3)
    }
}

/// When and where one of the teapot's triangles lands and leaves, for timing its sound.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    /// When it leaves the die, in seconds into the film.
    pub launches: f32,
    /// When it lands on the teapot, and when it starts back to the die, in seconds into the film.
    pub lands: f32,
    pub leaves: f32,
    /// How high up the teapot it sits, 0 to 1.
    pub height: f32,
    /// Where it lands across the picture, from -1 (left) to 1 (right).
    pub side: f32,
}

/// Every triangle of the teapot the picture builds.
///
/// # Errors
///
/// Fails if the teapot's model can't be loaded.
pub fn triangles() -> Result<Vec<Triangle>, Box<dyn std::error::Error>> {
    let obj = load_obj(Path::new("assets/utah_teapot.obj"))?;
    let (flights, ..) = flights(&obj);
    let top = flights.iter().fold(0.0_f32, |top, f| top.max(f.centroid.y));
    Ok(flights
        .iter()
        .map(|flight| {
            let lands = Phase::Teapot.start() + flight.launch + FLIGHT;
            Triangle {
                launches: Phase::Teapot.start() + flight.launch,
                lands,
                leaves: Phase::Teapot.start() + flight.leave,
                height: flight.centroid.y / top.max(1e-3),
                // The teapot reaches about two units to each side of its middle.
                side: (flight.state(lands).target.x / 2.0).clamp(-1.0, 1.0),
            }
        })
        .collect())
}

/// Every triangle's flight, its three corners, and every beam, for the teapot in `obj`.
fn flights(obj: &Model<ObjVertex>) -> (Vec<Flight>, Vec<[TeapotVertex; 3]>, Vec<Beam>) {
    let mesh = &obj.meshes[0];
    let (mut low, mut high) = (
        Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        Vec3::new(f32::MIN, f32::MIN, f32::MIN),
    );
    for vertex in &mesh.vertices {
        let p = vertex.position;
        low = Vec3::new(low.x.min(p.x), low.y.min(p.y), low.z.min(p.z));
        high = Vec3::new(high.x.max(p.x), high.y.max(p.y), high.z.max(p.z));
    }
    let origin = Vec3::new((low.x + high.x) * 0.5, low.y, (low.z + high.z) * 0.5);
    let place = |p: Vec3| (p - origin) * SCALE;

    struct Corners {
        corners: [ObjVertex; 3],
        centroid: Vec3,
        azimuth: f32,
    }
    let mut triangles: Vec<Corners> = mesh
        .indices
        .chunks_exact(3)
        .map(|i| {
            let corners = [0, 1, 2].map(|k| mesh.vertices[i[k] as usize]);
            let centroid =
                place((corners[0].position + corners[1].position + corners[2].position) / 3.0);
            Corners {
                corners,
                centroid,
                azimuth: centroid.z.atan2(centroid.x),
            }
        })
        .collect();
    triangles.sort_by(|a, b| a.azimuth.total_cmp(&b.azimuth));

    let per_lane = triangles.len().div_ceil(LANES);
    let mut flights = Vec::new();
    let mut corners = Vec::new();
    let mut beams = Vec::new();
    let height = (high.y - low.y) * SCALE;
    for (lane, sector) in triangles.chunks_mut(per_lane).enumerate() {
        let lane_seed = lane as f32 * 7.31;
        let azimuth = sector.iter().map(|t| t.azimuth).sum::<f32>() / sector.len() as f32;
        let mean = sector.iter().fold(Vec3::ZERO, |sum, t| sum + t.centroid) / sector.len() as f32;
        let source = Vec3::new(
            azimuth.cos() * DIE_HALF * 0.7,
            DIE_TOP + 0.02,
            azimuth.sin() * DIE_HALF * 0.7,
        );
        let beam_end = Vec3::new(mean.x * 1.1, BASE + height * 0.55, mean.z * 1.1);
        sector.sort_by(|a, b| a.centroid.y.total_cmp(&b.centroid.y));
        let spacing = SPACING.0 + SPACING.1 * random(lane_seed, 1.0);
        let offset = ASSEMBLY_START + 0.12 * random(lane_seed, 2.0);
        let count = sector.len();
        let (mut first, mut last_leave) = (f32::MAX, 0.0_f32);
        let (mut last, mut first_leave) = (0.0_f32, f32::MAX);
        for (slot, triangle) in sector.iter().enumerate() {
            let seed = random(lane_seed, slot as f32);
            let jitter = (random(lane_seed, slot as f32 + 100.0) - 0.5) * 0.012;
            let launch = offset + slot as f32 * spacing + jitter;
            let leave = LEAVE_START + (count - 1 - slot) as f32 * LEAVE_SPACING + jitter;
            first = first.min(launch);
            last = last.max(launch + FLIGHT);
            first_leave = first_leave.min(leave);
            last_leave = last_leave.max(leave + LEAVE_FLIGHT);
            let flight = Flight {
                source,
                beam_end,
                centroid: triangle.centroid,
                launch,
                leave,
                seed,
            };
            flights.push(flight);
            let barycentric = [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ];
            corners.push([0, 1, 2].map(|k| TeapotVertex {
                position: place(triangle.corners[k].position),
                normal: triangle.corners[k].normal,
                barycentric: barycentric[k],
                flight,
            }));
        }
        let start = Phase::Teapot.start();
        beams.push(Beam {
            source,
            end: beam_end,
            building: (start + first - 0.3, start + last - 0.4),
            unbuilding: (start + first_leave - 0.2, start + last_leave - 0.3),
        });
    }
    (flights, corners, beams)
}
