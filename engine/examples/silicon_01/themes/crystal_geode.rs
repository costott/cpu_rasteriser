//! The inside of an amethyst geode: the chip a cut topaz on a rock pedestal, hexagonal crystals
//! growing inward from banded agate walls, a glowing pool at its feet and sparkling motes, lit
//! by lamps whose glints sweep across the facets.

use std::f32::consts::{PI, TAU};

use cpu_rasteriser::prelude::*;

use crate::chip::Look;
use crate::shaders::composite::Grade;
use crate::shaders::{ViewRef, scene_colour};
use crate::simd::{clamp01, hash2, length, mix3, pow2n, reflect, rgb, s, smoothstep, splat3};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, pipeline,
};
use crate::themes::kit::dress::{chip_shadow, rim_point};
use crate::themes::kit::light::facing;
use crate::themes::kit::noise::{fbm, fbm_scalar};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::{Shape, lerp3};
use crate::themes::{Scene, Stage};

/// The floor of the geode: the chip's lowest corner rests on the pedestal's top at this height.
const FLOOR: f32 = -1.95;
/// The cavity: an ellipsoid's centre and radii.
const CENTRE: Vec3 = Vec3 {
    x: 0.0,
    y: 3.0,
    z: -2.0,
};
const RADII: Vec3 = Vec3 {
    x: 17.0,
    y: 9.5,
    z: 23.0,
};
const POOL: (f32, f32, f32) = (0.0, 6.5, 5.0);
const HAZE: Vec3 = Vec3 {
    x: 0.05,
    y: 0.02,
    z: 0.11,
};

/// The cavity wall's point in direction (`theta`, `phi`) from the centre, roughened into rock.
fn wall(theta: f32, phi: f32) -> Vec3 {
    let direction = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
    let rough = 1.0
        + (fbm_scalar(
            direction.x * 2.5 + 3.0,
            direction.z * 2.5 + direction.y * 1.5,
            3,
        ) - 0.5)
            * 0.12;
    CENTRE
        + Vec3::new(
            direction.x * RADII.x,
            direction.y * RADII.y,
            direction.z * RADII.z,
        ) * rough
}

/// A hexagonal prism of `radius` and `length` standing on y = 0 with a pyramid tip, shaded from
/// a deep base to a pale tip in the colour of `tint`.
fn crystal(radius: f32, length: f32, tint: Vec3) -> Shape {
    let tip = radius * 1.1;
    Shape::lathe(
        &[
            (0.0, 0.0),
            (radius, 0.0),
            (radius, length),
            (0.0, length + tip),
        ],
        6,
        tint,
    )
    .gradient_by_height(0.0, length + tip, tint * 0.35, lerp3(tint, Vec3::ONE, 0.45))
}

/// A rotation taking +y to the unit vector `direction`.
fn point_along(direction: Vec3) -> Mat4 {
    Mat4::rotate_y(direction.x.atan2(direction.z))
        * Mat4::rotate_x(direction.y.clamp(-1.0, 1.0).acos())
}

fn amethyst(rng: &mut Rng) -> Vec3 {
    let pale = Vec3::new(0.75, 0.55, 0.95);
    let deep = Vec3::new(0.4, 0.13, 0.7);
    lerp3(deep, pale, rng.unit().powi(2))
}

/// A cluster of crystals fanning out of one root on `base`, rising along `up` (unit).
fn cluster(rng: &mut Rng, base: Vec3, up: Vec3, size: f32, count: usize) -> Shape {
    let mut shape = Shape::new();
    for k in 0..count {
        let lean = if k == 0 { 0.0 } else { rng.range(0.25, 0.7) };
        let around = rng.range(0.0, TAU);
        let side = Vec3::new(around.cos(), 0.0, around.sin());
        let direction = (up + side * lean).normalise();
        let radius = size * rng.range(0.12, 0.2) * if k == 0 { 1.4 } else { 1.0 };
        let length = size * rng.range(0.7, 1.0) * if k == 0 { 1.0 } else { rng.range(0.5, 0.9) };
        let tint = amethyst(rng);
        shape.append(
            &crystal(radius, length, tint)
                .facet_variation(rng, 0.1)
                .with_seed(rng.unit())
                .transformed(&(Mat4::translation_vec(base) * point_along(direction))),
        );
    }
    shape
}

/// Crystals growing inward from the cavity wall, in clumps of a big one and a few satellites.
fn walls(rng: &mut Rng) -> Shape {
    let mut shape = Shape::new();
    let mut placed: Vec<Vec3> = Vec::new();
    for _ in 0..30_000 {
        if placed.len() >= 230 {
            break;
        }
        let (theta, phi) = (rng.range(0.0, TAU), (1.0 - 2.0 * rng.unit()).acos());
        let at = wall(theta, phi);
        if at.y < FLOOR + 1.0 || at.z > 14.0 || placed.iter().any(|p| (*p - at).length() < 2.4) {
            continue;
        }
        placed.push(at);
        let inward = (CENTRE - at).normalise();
        let size = rng.range(1.6, 3.6) * (1.0 + 0.3 * (at.z.abs() / 20.0));
        let tilt = Vec3::new(rng.range(-0.2, 0.2), 0.0, rng.range(-0.2, 0.2));
        shape.append(&cluster(rng, at, (inward + tilt).normalise(), size, 3));
    }
    shape
}

/// Larger clusters on the floor either side of the chip, framing it.
fn floor_clusters(rng: &mut Rng) -> Shape {
    let mut shape = Shape::new();
    for (x, z, size) in [
        (-6.5f32, 4.5f32, 2.6f32),
        (-8.5, 0.5, 3.4),
        (-7.0, -5.0, 4.2),
        (7.2, 3.0, 2.8),
        (9.0, -2.0, 3.6),
        (7.5, -7.0, 4.4),
        (-13.0, -12.0, 5.5),
        (12.5, -13.0, 5.5),
        (1.5, -16.0, 5.0),
    ] {
        let outward = Vec3::new(x.signum() * 0.35, 1.0, 0.0).normalise();
        shape.append(&cluster(rng, Vec3::new(x, FLOOR, z), outward, size, 6));
    }
    shape
}

/// The cavity's rock: a roughened ellipsoid seen from inside.
fn cavity() -> Shape {
    let (columns, rows) = (56, 30);
    let point =
        |i: usize, j: usize| wall(TAU * i as f32 / columns as f32, PI * j as f32 / rows as f32);
    let mut shape = Shape::new();
    for i in 0..columns {
        for j in 0..rows {
            let shade = 0.85 + 0.3 * crate::simd::hash2_scalar(i as f32, j as f32);
            shape.quad(
                [
                    point(i, j),
                    point(i + 1, j),
                    point(i + 1, j + 1),
                    point(i, j + 1),
                ],
                Vec3::new(0.16, 0.08, 0.24) * shade,
            );
        }
    }
    shape
}

/// The pedestal: a stack of faceted slabs the chip stands on, with its own small crystals.
fn pedestal() -> Shape {
    let rock = Vec3::new(0.32, 0.26, 0.34);
    let mut shape = Shape::lathe(
        &[
            (0.0, FLOOR - 0.6),
            (3.6, FLOOR - 0.6),
            (3.3, FLOOR - 0.25),
            (3.0, FLOOR),
            (0.0, FLOOR),
        ],
        7,
        rock,
    );
    shape.append(&Shape::lathe(
        &[
            (0.0, FLOOR - 1.3),
            (4.6, FLOOR - 1.3),
            (4.2, FLOOR - 0.6),
            (0.0, FLOOR - 0.6),
        ],
        9,
        rock * 0.8,
    ));
    shape
}

fn lamps(time: f32) -> [(Vec3, Vec3); 3] {
    let sweep = Vec3::new(
        13.0 * (time * 0.35).cos(),
        6.0,
        -4.0 + 12.0 * (time * 0.35).sin(),
    );
    [
        (
            Vec3::new(0.0, FLOOR + 1.5, POOL.2 + 1.0),
            Vec3::new(0.2, 1.0, 1.3) * (1.0 + 0.12 * (time * 1.7).sin()),
        ),
        (Vec3::new(2.0, 9.0, -4.0), Vec3::new(0.9, 0.7, 1.6)),
        (sweep, Vec3::new(2.2, 2.0, 2.6)),
    ]
}

#[derive(Clone)]
struct GemUniforms {
    view: ViewRef,
    lamps: [(Vec3, Vec3); 3],
}

/// What the lamps give a surface at `world` facing `normal`: its diffuse light and the mirror
/// glints, each lamp falling off with distance.
#[inline(always)]
fn lamp_light(
    u: &GemUniforms,
    world: Vec3Simd,
    normal: Vec3Simd,
    eye_dir: Vec3Simd,
) -> (Vec3Simd, Vec3Simd) {
    let mirror = reflect(-eye_dir, normal);
    let mut diffuse = rgb(0.0, 0.0, 0.0);
    let mut glints = rgb(0.0, 0.0, 0.0);
    for (position, colour) in u.lamps {
        let to_lamp = splat3(position) - world;
        let distance = length(to_lamp);
        let direction = to_lamp * (s(1.0) / distance);
        let falloff = s(1.0) / (s(1.0) + distance * distance * s(0.012));
        diffuse = diffuse + splat3(colour) * (normal.dot(direction).fast_max(s(0.0)) * falloff);
        let along = mirror.dot(direction).fast_max(s(0.0));
        glints = glints + splat3(colour) * (pow2n(along, 6) * falloff * s(1.6));
    }
    (diffuse, glints)
}

/// Amethyst: a deep glow under the facets, bright edges where the surface turns away, mirror
/// glints from the lamps and the odd flare of a facet catching the light.
struct CrystalShader;

impl FragmentShaderSimd<KitVaryings> for CrystalShader {
    type Uniforms = GemUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GemUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let eye_dir = to_eye * (s(1.0) / distance);
        let normal = facing(v.normal().normalise(), to_eye);
        let (diffuse, glints) = lamp_light(u, world, normal, eye_dir);
        let base = v.colour();

        let rim = s(1.0) - clamp01(normal.dot(eye_dir));
        let rim = rim * rim;
        let flare = pow2n(
            s(0.5) + s(0.5) * (s(view.frame.time * 1.3) + v.seed() * s(40.0)).sin(),
            4,
        );
        let body = crate::simd::mul3(base, rgb(0.28, 0.2, 0.38) + diffuse * s(0.9));
        let colour =
            body + rgb(0.5, 0.3, 1.0) * (rim * s(0.5)) + glints * s(0.8) + base * (flare * s(0.5));
        let haze = s(1.0) - (-distance * s(0.02)).exp();
        scene_colour(view, mix3(colour, splat3(HAZE), haze), world)
    }
}

/// Banded rock: agate stripes of violet and cream following the wall, lit by the lamps and
/// fading into haze.
struct AgateShader;

impl FragmentShaderSimd<KitVaryings> for AgateShader {
    type Uniforms = GemUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GemUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let eye_dir = to_eye * (s(1.0) / distance);
        let normal = facing(v.normal().normalise(), to_eye);
        let (diffuse, _) = lamp_light(u, world, normal, eye_dir);

        let radial = ((world.x - s(CENTRE.x)) * s(1.0 / RADII.x)).abs()
            + ((world.y - s(CENTRE.y)) * s(1.0 / RADII.y)).abs();
        let waves = fbm(
            world.x * s(0.18) + s(2.0),
            world.z * s(0.18) + world.y * s(0.1),
            3,
        ) * s(5.0);
        let phase = radial * s(9.0) + waves;
        let band = smoothstep(0.3, 0.5, phase - phase.floor())
            - smoothstep(0.7, 0.9, phase - phase.floor());
        let cream = rgb(0.75, 0.62, 0.72);
        let albedo = mix3(v.colour(), cream * s(0.5), band * s(0.55));
        let body = crate::simd::mul3(albedo, rgb(0.3, 0.22, 0.4) + diffuse);
        let haze = s(1.0) - (-distance * s(0.02)).exp();
        scene_colour(view, mix3(body, splat3(HAZE), haze), world)
    }
}

/// The floor's dark rock and the glowing pool: concentric ripples, a shimmer of glints and an
/// emission that brightens from the edge to the middle, pulsing slowly.
struct FloorShader;

impl FragmentShaderSimd<KitVaryings> for FloorShader {
    type Uniforms = GemUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &GemUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let time = view.frame.time;
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let eye_dir = to_eye * (s(1.0) / distance);
        let grain = fbm(world.x * s(1.1), world.z * s(1.1), 3);
        let normal = Vec3Simd::new(
            (grain - s(0.5)) * s(0.4),
            s(1.0),
            (fbm(world.z * s(1.1), world.x * s(1.1), 2) - s(0.5)) * s(0.4),
        )
        .normalise();
        let (diffuse, glints) = lamp_light(u, world, normal, eye_dir);
        let shadow = s(1.0) - chip_shadow(view, Vec3::new(0.0, 1.0, 0.0), world) * s(0.6);
        let rock = crate::simd::mul3(
            rgb(0.16, 0.1, 0.2) * (s(0.7) + grain * s(0.6)),
            rgb(0.3, 0.22, 0.4) + diffuse * shadow,
        );

        let (dx, dz) = (world.x - s(POOL.0), world.z - s(POOL.1));
        let radius = (dx * dx + dz * dz).sqrt();
        let inside = s(1.0) - smoothstep(POOL.2 - 0.5, POOL.2, radius);
        let ripple = (radius * s(3.2) - s(time * 1.1)).sin() * s(0.5) + s(0.5);
        let depth = s(1.0) - radius * s(1.0 / POOL.2);
        let pulse = s(0.85) + s(0.15) * (s(time * 0.8)).sin();
        let (cell_x, cell_z) = ((world.x * s(9.0)).floor(), (world.z * s(9.0)).floor());
        let (spot_x, spot_z) = (
            cell_x + s(0.3) + hash2(cell_z, cell_x) * s(0.4),
            cell_z + s(0.3) + hash2(cell_x + s(4.0), cell_z) * s(0.4),
        );
        let dot = ((world.x * s(9.0) - spot_x) * (world.x * s(9.0) - spot_x)
            + (world.z * s(9.0) - spot_z) * (world.z * s(9.0) - spot_z))
            .sqrt();
        let twinkle = s(0.5) + s(0.5) * (s(time * 4.0) + hash2(cell_z, cell_x) * s(40.0)).sin();
        let sparkle = (s(1.0) - smoothstep(0.05, 0.22, dot))
            * smoothstep(0.9, 0.96, hash2(cell_x, cell_z))
            * twinkle;
        let water = mix3(rgb(0.08, 0.2, 0.45), rgb(0.35, 0.9, 1.5), depth * depth)
            * (pulse * (s(0.55) + ripple * s(0.45)))
            + glints * s(0.6)
            + rgb(1.0, 1.1, 1.4) * (sparkle * s(1.4));
        let colour = mix3(rock + glints * s(0.15), water, inside);
        let haze = s(1.0) - (-distance * s(0.02)).exp();
        scene_colour(view, mix3(colour, splat3(HAZE), haze), world)
    }
}

/// Crystals grown on the chip, in its own space: amethyst clusters standing off the package's
/// edge.
fn chip_crystals(rng: &mut Rng) -> Shape {
    let mut shape = Shape::new();
    for _ in 0..9 {
        let (at, out) = rim_point(rng);
        let direction = (out * 0.8 + Vec3::new(0.0, 0.7, 0.0)).normalise();
        let size = rng.range(0.6, 1.2);
        shape.append(&cluster(rng, at, direction, size, 3));
    }
    shape
}

fn look() -> Look {
    Look {
        substrate: Vec3::new(0.95, 0.55, 0.12),
        die: Vec3::new(0.18, 0.26, 0.5),
        gold: Vec3::new(1.0, 0.8, 0.45),
        streams: Vec3::new(0.45, 0.6, 0.95),
        key: Vec3::new(0.3, 0.8, 0.5).normalise(),
        key_colour: Vec3::new(1.0, 0.85, 1.25),
        sky: Vec3::new(0.25, 0.15, 0.4),
        ground: Vec3::new(0.2, 0.5, 0.7),
        softbox: Vec3::new(0.22, 0.15, 0.35),
    }
}

pub struct CrystalGeode {
    cavity: Shape,
    floor: Shape,
    pedestal: Shape,
    crystals: Shape,
    chip_crystals: Shape,
    motes: Particles,
    crystal_pipeline: SimdPipeline<KitVertexShader, CrystalShader>,
    agate_pipeline: SimdPipeline<KitVertexShader, AgateShader>,
    floor_pipeline: SimdPipeline<KitVertexShader, FloorShader>,
}

impl CrystalGeode {
    pub fn new() -> Self {
        let mut rng = Rng::new(9001);
        let mut floor = Shape::new();
        let reach = 60.0;
        floor.quad(
            [
                Vec3::new(-reach, FLOOR, reach),
                Vec3::new(reach, FLOOR, reach),
                Vec3::new(reach, FLOOR, -reach),
                Vec3::new(-reach, FLOOR, -reach),
            ],
            Vec3::ONE,
        );
        let mut crystals = walls(&mut rng);
        crystals.append(&floor_clusters(&mut rng));
        Self {
            cavity: cavity(),
            floor,
            pedestal: pedestal(),
            crystals,
            chip_crystals: chip_crystals(&mut rng),
            motes: Particles::new(160, 311),
            crystal_pipeline: pipeline(CrystalShader, CullingMode::None),
            agate_pipeline: pipeline(AgateShader, CullingMode::None),
            floor_pipeline: pipeline(FloorShader, CullingMode::None),
        }
    }
}

impl Stage for CrystalGeode {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Glockenspiel],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::CrystalTing),
                0.0,
                0.45,
                0.1,
                -0.3,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let gem = GemUniforms {
            view: view.clone(),
            lamps: lamps(view.frame.time),
        };
        let still = KitUniforms {
            view: view.clone(),
            motion: Motion::Still,
        };
        let on_chip = KitUniforms {
            view: view.clone(),
            motion: Motion::OnChip,
        };

        self.cavity
            .draw(pass, &self.agate_pipeline, still.clone(), gem.clone());
        self.floor
            .draw(pass, &self.floor_pipeline, still.clone(), gem.clone());
        self.pedestal
            .draw(pass, &self.agate_pipeline, still.clone(), gem.clone());
        self.crystals
            .draw(pass, &self.crystal_pipeline, still.clone(), gem.clone());

        scene.chip.draw(pass, view, &look());
        self.chip_crystals
            .draw(pass, &self.crystal_pipeline, on_chip, gem);

        self.motes.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 2.0, 0.0),
                half_extent: Vec3::new(13.0, 6.0, 14.0),
                velocity: Vec3::new(0.05, 0.12, 0.0),
                wander: 0.6,
                size: (0.015, 0.045),
                colour: Vec3::new(1.4, 1.0, 2.0),
                twinkle: 2.4,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.1,
            bloom: 1.1,
            vignette: 0.45,
            grain: 0.01,
            aberration: 0.002,
            desaturate: 0.0,
        }
    }
}
