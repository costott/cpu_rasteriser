//! clay-stopmotion: a plasticine landscape shot frame by frame at 8 fps. Hills, trees, bushes,
//! flowers and cloud puffs are modelled in clay that is nudged and thumbed between frames, so the
//! whole world and the chip boil slightly from one frame to the next.

use std::f32::consts::TAU;
use std::sync::Arc;

use cpu_rasteriser::prelude::*;
use engine::prelude::*;

use crate::chip::Look;
use crate::director;
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, View, ViewRef, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, ge, hash2, length, mix3, rgb, s, select, select3, smoothstep, splat3, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonUniforms, pipeline,
};
use crate::themes::kit::dress::rim_point;
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm_scalar, value_scalar};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::kit::terrain::{ring_radii, terrain};
use crate::themes::{Scene, Stage};

const FLOOR: f32 = -1.95;
/// How many frames a second the animation is shot at.
const FPS: f32 = 8.0;
const SUN: Vec3 = Vec3 {
    x: -0.45,
    y: 0.75,
    z: 0.45,
};

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(1.9, 1.7, 1.4),
        sky: Vec3::new(0.5, 0.58, 0.8),
        ground: Vec3::new(0.34, 0.28, 0.2),
        bands: 0.0,
        rim: Vec3::new(0.06, 0.06, 0.05),
        fog: Fog {
            colour: Vec3::new(0.62, 0.78, 0.95),
            density: 0.002,
            falloff: 0.05,
            base: FLOOR,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.62, 0.34, 0.2),
        die: Vec3::new(0.1, 0.34, 0.2),
        gold: Vec3::new(0.95, 0.74, 0.18),
        streams: Vec3::new(1.0, 0.9, 0.6),
        key: light.sun,
        key_colour: light.sun_colour * 0.7,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.35, 0.3, 0.24),
    }
}

/// The hills: a clearing round the chip, swells that rise into rounded ridges further back, and a
/// slow lumpiness that is redrawn every frame.
fn ground(x: f32, z: f32, step: f32) -> f32 {
    let r = x.hypot(z);
    let clearing = ((r - 3.0) / 10.0).clamp(0.0, 1.0);
    let swell = (fbm_scalar(x * 0.045 + 4.0, z * 0.06, 3) - 0.5) * 9.0;
    let back = ((-z - 40.0) / 70.0).clamp(0.0, 1.0);
    let ridge = back * back * (7.0 + 24.0 * fbm_scalar(x * 0.02 + 9.0, z * 0.03 + 2.0, 3));
    let thumbed = (value_scalar(x * 0.22 + step * 2.9, z * 0.22 - step * 1.7) - 0.5) * 0.5;
    FLOOR + clearing * clearing * swell + ridge + thumbed * clearing
}

fn ground_colour(centroid: Vec3, normal: Vec3) -> Vec3 {
    let patch = fbm_scalar(centroid.x * 0.05 + 7.0, centroid.z * 0.05, 3);
    let green = lerp3(
        Vec3::new(0.16, 0.38, 0.12),
        Vec3::new(0.36, 0.5, 0.14),
        patch,
    );
    let ochre = Vec3::new(0.62, 0.4, 0.14);
    let dry =
        ((fbm_scalar(centroid.x * 0.03, centroid.z * 0.03 + 5.0, 2) - 0.6) * 6.0).clamp(0.0, 1.0);
    let far = ((-centroid.z - 45.0) / 90.0).clamp(0.0, 1.0);
    let hill = lerp3(lerp3(green, ochre, dry), Vec3::new(0.3, 0.55, 0.45), far);
    hill * (0.9 + 0.12 * normal.y)
}

/// The clays the fields of the patchwork hills are modelled from.
const FIELDS: [Vec3; 5] = [
    Vec3 {
        x: 0.2,
        y: 0.46,
        z: 0.13,
    },
    Vec3 {
        x: 0.42,
        y: 0.56,
        z: 0.15,
    },
    Vec3 {
        x: 0.66,
        y: 0.44,
        z: 0.14,
    },
    Vec3 {
        x: 0.3,
        y: 0.4,
        z: 0.12,
    },
    Vec3 {
        x: 0.16,
        y: 0.42,
        z: 0.3,
    },
];

#[inline(always)]
fn field_colour(index: F) -> Vec3Simd {
    let mut colour = splat3(FIELDS[0]);
    for (i, field) in FIELDS.iter().enumerate().skip(1) {
        colour = select3(ge(index, s(i as f32)), splat3(*field), colour);
    }
    colour
}

#[derive(Clone)]
pub struct ClayUniforms {
    toon: ToonUniforms,
    /// How much the surface takes its colour from the hills' patchwork instead of the vertex.
    patchwork: f32,
}

/// A bump pattern of thumb prints: each cell of a jittered grid holds a disc of concentric
/// ridges, over a lumpy undertone.
struct ClayShader;

impl FragmentShaderSimd<KitVaryings> for ClayShader {
    type Uniforms = ClayUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, c: &ClayUniforms) -> ColourSimd {
        let u = &c.toon;
        let view = &u.view;
        let world = v.world();
        let distance = length(splat3(view.eye) - world);
        let mut normal = facing(v.normal().normalise(), splat3(view.eye) - world);

        let flat = smoothstep(0.55, 0.65, normal.y.abs());
        let a = select(flat, world.x, world.x + world.z);
        let b = select(flat, world.z, world.y * s(1.2));
        let (ca, cb) = ((a * s(1.5)).floor(), (b * s(1.5)).floor());
        let id = hash2(ca, cb);
        let (fa, fb) = (a * s(1.5) - ca, b * s(1.5) - cb);
        let (da, db) = (
            fa - s(0.3) - id * s(0.4),
            fb - s(0.3) - hash2(cb, ca) * s(0.4),
        );
        let radius = (da * da + db * db).sqrt().fast_max(s(1e-3));
        let wavelength = s(TAU / 24.0);
        let resolved = clamp01((wavelength / (distance * s(view.pixel_angle)) - s(2.0)) * s(0.25));
        let ridge = (radius * s(24.0) + id * s(6.0)).cos();
        let strength = smoothstep(0.5, 0.12, radius) * ridge * resolved * s(0.3);
        let bump = fbm(
            world.x * s(3.0) + world.z * s(1.7),
            world.y * s(3.0) + world.z * s(2.3),
            2,
        ) - s(0.5);
        let lumps = bump * s(0.5) * clamp01(s(1.0) - distance * s(0.02));
        normal = Vec3Simd::new(
            normal.x + (da / radius) * strength + lumps,
            normal.y + (db / radius) * strength * s(0.6),
            normal.z - (da / radius) * strength * s(0.5) + lumps * s(0.6),
        )
        .normalise();

        let mottle = fbm(world.x * s(0.9), world.y * s(0.9) + world.z * s(0.9), 2);
        let warp = fbm(world.x * s(0.05), world.z * s(0.05), 2) * s(2.5);
        let pick = hash2(
            (world.x * s(0.045) + warp).floor(),
            (world.z * s(0.06) + warp).floor(),
        );
        let field = field_colour((pick * s(FIELDS.len() as f32)).floor());
        let far = smoothstep(40.0, 150.0, -world.z);
        let patch = mix3(mix3(v.colour(), field, s(0.85)), v.colour(), far);
        let base = mix3(v.colour(), patch, s(c.patchwork));
        let albedo = base * (s(0.9) + mottle * s(0.2) + (id - s(0.5)) * s(0.05));
        let colour = u.lighting.shade(view, world, normal, albedo, s(1.0));
        scene_colour(view, colour, world)
    }
}

/// A flat disc on the ground, for a soft shadow under a prop.
fn blob_shadow(radius: f32) -> Shape {
    let ring = |i: usize| {
        let angle = TAU * i as f32 / 12.0;
        Vec3::new(radius * angle.cos(), 0.0, -radius * angle.sin())
    };
    let mut disc = Shape::new();
    for i in 0..12 {
        disc.triangle(Vec3::ZERO, ring(i), ring(i + 1), Vec3::new(0.2, 0.28, 0.1));
    }
    disc
}

/// A round tree: a trunk and two or three lumpy crowns.
fn round_tree(rng: &mut Rng, crown: Vec3, detail: u32) -> Shape {
    let mut tree = Shape::limb(
        Vec3::ZERO,
        Vec3::new(0.1, 2.6, 0.0),
        0.3,
        0.2,
        7,
        Vec3::new(0.5, 0.3, 0.14),
    );
    for (i, (x, y, size)) in [(0.0, 3.4, 1.5), (-0.7, 2.9, 1.0), (0.8, 3.0, 1.1)]
        .into_iter()
        .enumerate()
    {
        let tone = 1.0 + (i as f32 - 1.0) * 0.1;
        tree.append(
            &Shape::icosphere(size, detail, 0.05, rng.range(0.0, 30.0), crown * tone)
                .transformed(&Mat4::translation(x, y, rng.range(-0.3, 0.3))),
        );
    }
    tree
}

/// A cone-shaped fir of three stacked layers.
fn fir(rng: &mut Rng, detail: u32) -> Shape {
    let mut tree = Shape::limb(
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        0.25,
        0.2,
        7,
        Vec3::new(0.45, 0.28, 0.14),
    );
    for (k, (y, size)) in [(1.0, 1.4), (2.2, 1.1), (3.2, 0.75)]
        .into_iter()
        .enumerate()
    {
        let green = lerp3(
            Vec3::new(0.1, 0.36, 0.2),
            Vec3::new(0.16, 0.45, 0.22),
            k as f32 / 2.0,
        );
        tree.append(
            &Shape::icosphere(1.0, detail, 0.05, rng.range(0.0, 30.0), green).transformed(
                &(Mat4::translation(0.0, y + 0.6, 0.0) * Mat4::scaling(size, size * 1.15, size)),
            ),
        );
    }
    tree
}

fn bush(rng: &mut Rng, colour: Vec3, detail: u32) -> Shape {
    let mut bush = Shape::new();
    for (x, size) in [(-0.5, 0.6), (0.0, 0.8), (0.55, 0.55)] {
        bush.append(
            &Shape::icosphere(size, detail, 0.05, rng.range(0.0, 30.0), colour)
                .transformed(&Mat4::translation(x, size * 0.7, rng.range(-0.2, 0.2))),
        );
    }
    bush
}

fn flower(rng: &mut Rng, petals: Vec3) -> Shape {
    let mut flower = Shape::limb(
        Vec3::ZERO,
        Vec3::new(0.0, 0.9, 0.0),
        0.05,
        0.04,
        5,
        Vec3::new(0.2, 0.5, 0.15),
    );
    flower.append(
        &Shape::icosphere(
            0.14,
            1,
            0.1,
            rng.range(0.0, 20.0),
            Vec3::new(0.95, 0.78, 0.15),
        )
        .transformed(&Mat4::translation(0.0, 1.0, 0.0)),
    );
    for k in 0..6 {
        let angle = TAU * k as f32 / 6.0;
        flower.append(
            &Shape::icosphere(0.14, 1, 0.1, rng.range(0.0, 20.0), petals).transformed(
                &Mat4::translation(angle.cos() * 0.24, 1.0, angle.sin() * 0.24),
            ),
        );
    }
    flower
}

/// A puff of cloud: a cluster of fat blobs, flat underneath.
fn cloud(rng: &mut Rng) -> Shape {
    let mut cloud = Shape::new();
    for k in 0..5 {
        let x = (k as f32 - 2.0) * 1.5;
        let size = 1.9 - (k as f32 - 2.0).abs() * 0.35 + rng.range(-0.1, 0.2);
        cloud.append(
            &Shape::icosphere(
                1.0,
                2,
                0.06,
                rng.range(0.0, 30.0),
                Vec3::new(0.97, 0.97, 0.96),
            )
            .transformed(
                &(Mat4::translation(x, size * 0.45 + rng.range(0.0, 0.4), rng.range(-0.4, 0.4))
                    * Mat4::scaling(size, size * 0.8, size)),
            ),
        );
    }
    cloud
}

/// A sun with clay rays, facing the camera.
fn sun(rng: &mut Rng) -> Shape {
    let mut sun = Shape::icosphere(2.6, 2, 0.04, 3.0, Vec3::new(1.0, 0.82, 0.18));
    for k in 0..12 {
        let angle = TAU * k as f32 / 12.0;
        let (c, s) = (angle.cos(), angle.sin());
        let at =
            |r: f32, turn: f32| Vec3::new((angle + turn).cos() * r, (angle + turn).sin() * r, 0.0);
        sun.triangle(
            at(3.2, -0.11),
            at(3.2, 0.11),
            Vec3::new(c * 4.6, s * 4.6, 0.0),
            Vec3::new(1.0, 0.62, 0.15) * rng.range(0.92, 1.05),
        );
        sun.triangle(
            at(3.2, 0.11),
            at(3.2, -0.11),
            Vec3::new(c * 4.6, s * 4.6, 0.0),
            Vec3::new(1.0, 0.62, 0.15),
        );
    }
    sun
}

/// Nudges every vertex by a smooth, frame-dependent offset (the clay shifting under the
/// animator's fingers) with a little roughness of its own. Corners shared between triangles
/// move together, so the clay stays whole.
fn thumb(shape: &Shape, step: f32, amount: f32) -> Shape {
    let displace = |p: Vec3| {
        let offset = |k: f32| {
            let smooth = value_scalar(
                p.x * 0.8 + k * 7.3 + step * 3.3,
                p.y * 0.8 + p.z * 0.7 + step * 1.9 - k * 3.1,
            ) - 0.5;
            let rough = crate::simd::hash2_scalar(
                p.x * 41.0 + k * 3.0 + step,
                p.y * 37.0 + p.z * 29.0 - step,
            ) - 0.5;
            smooth * 2.0 * amount + rough * 0.4 * amount
        };
        p + Vec3::new(offset(0.0), offset(1.0), offset(2.0))
    };
    let mut out = Shape::new();
    for t in shape.vertices().chunks_exact(3) {
        out.triangle(
            displace(t[0].position),
            displace(t[1].position),
            displace(t[2].position),
            t[0].colour,
        );
    }
    out
}

/// A prop that stands on the ground and is nudged between frames.
struct Prop {
    shape: Shape,
    x: f32,
    z: f32,
    scale: f32,
    yaw: f32,
    /// How far the animator moves it between frames: the whole prop and its surface.
    shift: f32,
    wobble: f32,
    /// How high above the ground it floats: 0 for things that stand, more for clouds.
    lift: f32,
    drift: f32,
    shadow: Option<f32>,
}

impl Prop {
    fn pose(&self, step: f32, time: f32) -> Mat4 {
        let jitter = |k: f32| {
            crate::simd::hash2_scalar(
                self.x * 3.1 + step * 1.7 + k,
                self.z * 2.3 - step * 0.9 - k * 5.0,
            ) - 0.5
        };
        let x = if self.drift == 0.0 {
            self.x
        } else {
            (self.x + time * self.drift + 90.0).rem_euclid(180.0) - 90.0
        };
        let y = if self.lift == 0.0 {
            ground(x, self.z, step)
        } else {
            self.lift
        };
        Mat4::translation(
            x + jitter(0.0) * self.shift,
            y + jitter(1.0) * self.shift * 0.4,
            self.z + jitter(2.0) * self.shift,
        ) * Mat4::rotate_y(self.yaw + jitter(3.0) * 0.12)
            * Mat4::rotate_z(jitter(4.0) * 0.08)
            * Mat4::scaling(self.scale, self.scale, self.scale)
    }
}

pub struct ClayStopmotion {
    props: Vec<Prop>,
    hills: Shape,
    clay: Shape,
    shadows: Shape,
    chip_extras: Shape,
    step: f32,
    view: Option<ViewRef>,
    clay_pipeline: SimdPipeline<KitVertexShader, ClayShader>,
    sky: SimdPipeline<ScreenVertexShader, CardboardSky>,
}

impl ClayStopmotion {
    pub fn new() -> Self {
        let mut rng = Rng::new(808);
        let mut props = Vec::new();
        let mut place =
            |shape: Shape, x: f32, z: f32, scale: f32, shadow: Option<f32>, wobble: f32| {
                props.push(Prop {
                    shape,
                    x,
                    z,
                    scale,
                    yaw: x * 1.7 + z,
                    shift: 0.06 * scale,
                    wobble,
                    lift: 0.0,
                    drift: 0.0,
                    shadow,
                });
            };
        let crowns = [
            Vec3::new(0.3, 0.6, 0.16),
            Vec3::new(0.85, 0.45, 0.1),
            Vec3::new(0.2, 0.52, 0.28),
        ];
        for (i, &(x, z, scale)) in [
            (-8.0, -6.0, 1.1),
            (-13.0, -12.0, 1.4),
            (9.0, -9.0, 1.2),
            (14.0, -4.0, 1.0),
            (-4.0, -18.0, 1.5),
            (6.0, -22.0, 1.6),
            (-20.0, -26.0, 1.9),
            (22.0, -20.0, 1.7),
            (-30.0, -38.0, 2.3),
            (30.0, -36.0, 2.2),
        ]
        .iter()
        .enumerate()
        {
            let detail = if z > -15.0 { 3 } else { 2 };
            let tree = if i % 3 == 1 {
                fir(&mut rng, detail)
            } else {
                round_tree(&mut rng, crowns[i % crowns.len()], detail)
            };
            place(tree, x, z, scale, Some(1.9 * scale), 0.05);
        }
        for &(x, z, scale) in &[
            (-5.0, 4.0, 0.9),
            (-11.0, -2.0, 1.1),
            (7.5, 5.0, 0.8),
            (12.0, 1.0, 1.0),
            (-2.0, -10.0, 1.0),
            (4.0, -13.0, 1.3),
            (-16.0, -8.0, 1.2),
        ] {
            let colour = lerp3(
                Vec3::new(0.18, 0.46, 0.16),
                Vec3::new(0.35, 0.6, 0.2),
                rng.unit(),
            );
            place(
                bush(&mut rng, colour, if z > -5.0 { 3 } else { 2 }),
                x,
                z,
                scale,
                Some(1.2 * scale),
                0.04,
            );
        }
        let petals = [
            Vec3::new(0.95, 0.35, 0.4),
            Vec3::new(0.97, 0.97, 0.9),
            Vec3::new(0.55, 0.5, 0.9),
            Vec3::new(1.0, 0.6, 0.15),
        ];
        for k in 0..34 {
            let (x, z) = (rng.range(-16.0, 16.0), rng.range(-14.0, 7.0));
            if x.hypot(z) < 4.6 || (x.abs() < 3.6 && z > -2.0) {
                continue;
            }
            place(
                flower(&mut rng, petals[k % 4]),
                x,
                z,
                rng.range(0.7, 1.2),
                None,
                0.02,
            );
        }
        for &(x, z, lift, speed) in &[
            (-30.0, -60.0, 19.0, 0.35),
            (-4.0, -80.0, 24.0, 0.28),
            (26.0, -55.0, 15.0, 0.4),
            (50.0, -90.0, 27.0, 0.3),
            (-60.0, -75.0, 22.0, 0.32),
        ] {
            props.push(Prop {
                shape: cloud(&mut rng),
                x,
                z,
                scale: 2.4,
                yaw: 0.0,
                shift: 0.15,
                wobble: 0.08,
                lift,
                drift: speed,
                shadow: None,
            });
        }
        props.push(Prop {
            shape: sun(&mut rng),
            x: 26.0,
            z: -105.0,
            scale: 2.0,
            yaw: 0.0,
            shift: 0.1,
            wobble: 0.06,
            lift: 24.0,
            drift: 0.0,
            shadow: None,
        });

        Self {
            props,
            hills: Shape::new(),
            clay: Shape::new(),
            shadows: Shape::new(),
            chip_extras: Shape::new(),
            step: f32::NAN,
            view: None,
            clay_pipeline: pipeline(ClayShader, CullingMode::BackFace),
            sky: SimdPipeline::new(ScreenVertexShader, CardboardSky)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }

    /// Reshoots the scene for frame `step`.
    fn pose_all(&mut self, step: f32, time: f32) {
        self.hills = terrain(
            &ring_radii(1.4, 190.0, 38),
            84,
            |x, z| ground(x, z, step),
            ground_colour,
        );
        let mut clay = Shape::new();
        let mut shadows = Shape::new();
        for (id, prop) in self.props.iter().enumerate() {
            let at = prop.pose(step, time);
            clay.append(&thumb(
                &prop.shape.transformed(&at),
                step + id as f32 * 0.37,
                prop.wobble,
            ));
            if let Some(radius) = prop.shadow {
                let y = ground(prop.x, prop.z, step) + 0.05;
                shadows.append(&blob_shadow(radius).transformed(&Mat4::translation(
                    prop.x + 0.35,
                    y,
                    prop.z + 0.3,
                )));
            }
        }
        self.clay = clay;
        self.shadows = shadows;
        let mut rng = Rng::new(31);
        let mut extras = Shape::new();
        for k in 0..12 {
            let (spot, _) = rim_point(&mut rng);
            let radius = rng.range(0.2, 0.4);
            let colours = [
                Vec3::new(0.95, 0.74, 0.18),
                Vec3::new(0.3, 0.6, 0.25),
                Vec3::new(0.85, 0.35, 0.3),
                Vec3::new(0.4, 0.55, 0.85),
            ];
            extras.append(
                &Shape::icosphere(radius, 2, 0.1, k as f32, colours[k % 4]).transformed(
                    &Mat4::translation_vec(spot + Vec3::new(0.0, radius * 0.5, 0.0)),
                ),
            );
        }
        self.chip_extras = thumb(&extras, step, 0.03);
    }
}

impl Stage for ClayStopmotion {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Marimba],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Squelch),
                0.0,
                0.45,
                0.16,
                0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn prepare(&mut self, _context: &mut RenderContext<'_>, scene: &Scene<'_>) {
        let step = (scene.view.frame.time * FPS).floor();
        let time = step / FPS;
        if step != self.step {
            self.step = step;
            self.view = Some(Arc::new(View {
                frame: Arc::new(director::frame(time)),
                ..(*scene.view).clone()
            }));
            self.pose_all(step, time);
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = self.view.as_ref().unwrap_or(&scene.view);
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let clay = |patchwork| ClayUniforms {
            toon: toon.clone(),
            patchwork,
        };
        let motion = |motion| KitUniforms {
            view: view.clone(),
            motion,
        };

        self.hills
            .draw(pass, &self.clay_pipeline, motion(Motion::Still), clay(1.0));
        self.shadows
            .draw(pass, &self.clay_pipeline, motion(Motion::Still), clay(0.0));
        self.clay
            .draw(pass, &self.clay_pipeline, motion(Motion::Still), clay(0.0));
        scene.chip.draw(pass, view, &look());
        self.chip_extras
            .draw(pass, &self.clay_pipeline, motion(Motion::OnChip), clay(0.0));
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.sky, (), |_| view.clone());
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.15,
            vignette: 0.3,
            grain: 0.0,
            aberration: 0.0,
            desaturate: -0.1,
        }
    }
}

/// A painted cardboard backdrop: flat blue shading to cream at the horizon, with paper fibres.
struct CardboardSky;

impl FragmentShaderSimd<ScreenVaryings> for CardboardSky {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [nx, ny] = v.ndc();
        let ray = view_ray(view, nx, ny);
        let up = ray.y.fast_max(s(0.0));
        let fibre = fbm(nx * s(60.0), ny * s(60.0), 2);
        let wash = fbm(nx * s(2.0), ny * s(3.0), 3);
        let mut colour = mix3(
            rgb(0.78, 0.88, 0.95),
            rgb(0.3, 0.55, 0.85),
            smoothstep(0.0, 0.35, up).sqrt(),
        );
        colour = colour * (s(0.94) + wash * s(0.1) + fibre * s(0.06));
        to_colour(colour, s(SKY_DISTANCE))
    }
}
