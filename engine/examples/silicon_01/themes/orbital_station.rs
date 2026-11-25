//! orbital-station: a clean grey station in raw sunlight, with Earth's curved horizon below. The
//! chip is the core module, with solar arrays on its flanks, laboratory modules above and below
//! and a dish; a visiting vehicle circles the station and sparkling ice drifts past.

use std::f32::consts::FRAC_PI_2;

use cpu_rasteriser::prelude::*;

use crate::chip::{HALF, Look};
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{
    clamp01, length, mix3, pow_highlight, reflect, rgb, s, select3, smoothstep, splat3, to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, Orbit, ToonUniforms,
    pipeline,
};
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm3, value3};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::shape::Shape;
use crate::themes::kit::sky::stars;
use crate::themes::{Scene, Stage};

/// Earth's centre and radius, placed so its limb crosses the frame in an arc below the chip.
const EARTH_CENTRE: Vec3 = Vec3 {
    x: 0.0,
    y: -112.0,
    z: -62.0,
};
const EARTH_RADIUS: f32 = 104.0;
/// The direction towards the sun: raw light from above, to the left and behind the camera.
const SUN: Vec3 = Vec3 {
    x: -0.6,
    y: 0.5,
    z: 0.62,
};

/// Materials, carried in a shape's seed: how the station's shader treats its surface.
const PAINT: f32 = 0.0;
const FOIL: f32 = 0.25;
const PANEL: f32 = 0.5;
const BEACON: f32 = 1.0;

const WHITE: Vec3 = Vec3 {
    x: 0.62,
    y: 0.63,
    z: 0.65,
};
const GREY: Vec3 = Vec3 {
    x: 0.32,
    y: 0.34,
    z: 0.37,
};
const GOLD: Vec3 = Vec3 {
    x: 0.75,
    y: 0.5,
    z: 0.18,
};
const CELL: Vec3 = Vec3 {
    x: 0.012,
    y: 0.03,
    z: 0.13,
};

fn lighting() -> Lighting {
    Lighting {
        sun: SUN.normalise(),
        sun_colour: Vec3::new(2.1, 2.0, 1.9),
        sky: Vec3::new(0.01, 0.012, 0.025),
        ground: Vec3::new(0.05, 0.1, 0.24),
        bands: 0.0,
        rim: Vec3::ZERO,
        fog: Fog {
            colour: Vec3::ZERO,
            density: 0.0,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

/// The chip as the core module: pale painted shell, a dark glass die, gold foil contacts and
/// blue docking lights.
fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.62, 0.64, 0.67),
        die: Vec3::new(0.02, 0.05, 0.12),
        gold: Vec3::new(0.85, 0.6, 0.22),
        streams: Vec3::new(0.5, 0.8, 1.3),
        key: light.sun,
        key_colour: light.sun_colour * 0.6,
        sky: Vec3::new(0.03, 0.05, 0.1),
        ground: light.ground,
        softbox: Vec3::ZERO,
    }
}

/// A flat quad in the plane `y`, from corner (`x0`, `z0`) to (`x1`, `z1`).
fn plate(shape: &mut Shape, y: f32, (x0, z0): (f32, f32), (x1, z1): (f32, f32), colour: Vec3) {
    shape.quad(
        [
            Vec3::new(x0, y, z0),
            Vec3::new(x0, y, z1),
            Vec3::new(x1, y, z1),
            Vec3::new(x1, y, z0),
        ],
        colour,
    );
}

/// A solar array reaching along x from `root` towards `tip`: a gold-backed blanket of cells
/// between two rails with cross struts, `width` across.
fn array(root: f32, tip: f32, width: f32) -> Shape {
    let (y, half) = (0.12, width * 0.5);
    let mut shape = Shape::new();
    let columns = 8;
    let rows = 3;
    let (low, high) = (root.min(tip), root.max(tip));
    let step = (high - low) / columns as f32;
    for i in 0..columns {
        for j in 0..rows {
            let z0 = -half + width * j as f32 / rows as f32;
            let z1 = -half + width * (j + 1) as f32 / rows as f32;
            let x0 = low + step * i as f32;
            let tone = 0.85 + 0.3 * (((i * 7 + j * 3) % 5) as f32 / 4.0);
            plate(
                &mut shape,
                y,
                (x0 + 0.02, z0 + 0.02),
                (x0 + step - 0.02, z1 - 0.02),
                CELL * tone,
            );
        }
    }
    let mut cells = shape.with_seed(PANEL);
    let mut backing = Shape::new();
    plate(
        &mut backing,
        y - 0.02,
        (low, -half),
        (high, half),
        GOLD * 0.8,
    );
    cells.append(&backing.with_seed(FOIL));
    let mut frame = Shape::new();
    for z in [-half, half] {
        frame.append(&Shape::limb(
            Vec3::new(low, y, z),
            Vec3::new(high, y, z),
            0.035,
            0.035,
            6,
            GREY,
        ));
    }
    for i in 0..=columns {
        let x = low + step * i as f32;
        frame.append(&Shape::limb(
            Vec3::new(x, y, -half),
            Vec3::new(x, y, half),
            0.025,
            0.025,
            6,
            GREY,
        ));
    }
    cells.append(&frame.with_seed(PAINT));
    cells
}

/// A cylindrical laboratory along x centred at (`0`, `y`, `z`): a painted hull with bands of gold
/// foil, rounded ends and a port on its outer side.
fn laboratory(z: f32, outward: f32) -> Shape {
    let (length, radius) = (2.6, 0.38);
    let centre = Vec3::new(0.0, 0.25, z);
    let mut hull = Shape::limb(
        centre - Vec3::new(length * 0.5, 0.0, 0.0),
        centre + Vec3::new(length * 0.5, 0.0, 0.0),
        radius,
        radius,
        18,
        WHITE,
    )
    .with_seed(PAINT);
    for k in 0..3 {
        let x = (k as f32 - 1.0) * 0.9;
        let band = Shape::limb(
            centre + Vec3::new(x - 0.2, 0.0, 0.0),
            centre + Vec3::new(x + 0.2, 0.0, 0.0),
            radius * 1.04,
            radius * 1.04,
            18,
            GOLD,
        );
        hull.append(&band.with_seed(FOIL));
    }
    for end in [-1.0, 1.0] {
        let dome = Shape::icosphere(radius, 2, 0.0, 0.0, WHITE).transformed(
            &(Mat4::translation_vec(centre + Vec3::new(end * length * 0.5, 0.0, 0.0))
                * Mat4::scaling(0.55, 1.0, 1.0)),
        );
        hull.append(&dome.with_seed(PAINT));
    }
    let port = Shape::limb(
        centre + Vec3::new(0.0, 0.0, outward * radius * 0.9),
        centre + Vec3::new(0.0, 0.0, outward * (radius + 0.3)),
        0.17,
        0.2,
        12,
        GREY,
    );
    hull.append(&port.with_seed(PAINT));
    hull
}

/// A dish on a mast above the top laboratory.
fn dish(z: f32, outward: f32) -> Shape {
    let base = Vec3::new(-1.0, 0.25, z + outward * 0.4);
    let mut shape = Shape::limb(
        base,
        base + Vec3::new(0.0, 0.0, outward * 0.55),
        0.04,
        0.04,
        6,
        GREY,
    )
    .with_seed(PAINT);
    let bowl = Shape::lathe(
        &[
            (0.0, 0.0),
            (0.12, 0.02),
            (0.3, 0.1),
            (0.34, 0.16),
            (0.0, 0.0),
        ],
        14,
        WHITE,
    )
    .transformed(
        &(Mat4::translation_vec(base + Vec3::new(0.0, 0.0, outward * 0.55))
            * Mat4::rotate_x(outward * FRAC_PI_2)),
    );
    shape.append(&bowl.with_seed(PAINT));
    shape
}

/// The station's fittings in the chip's own space, attached to the package.
fn fittings() -> Shape {
    let mut station = Shape::new();
    for side in [-1.0, 1.0] {
        let boom = Shape::limb(
            Vec3::new(side * (HALF - 0.1), 0.12, 0.0),
            Vec3::new(side * (HALF + 0.55), 0.12, 0.0),
            0.12,
            0.1,
            10,
            GREY,
        );
        station.append(&boom.with_seed(PAINT));
        let joint = Shape::cuboid(
            Vec3::new(side * (HALF + 0.5) - 0.12, 0.0, -0.22),
            Vec3::new(side * (HALF + 0.5) + 0.12, 0.24, 0.22),
            GOLD,
        );
        station.append(&joint.with_seed(FOIL));
        station.append(&array(side * (HALF + 0.6), side * (HALF + 4.6), 1.8));
    }
    for outward in [-1.0, 1.0] {
        let z = outward * (HALF + 0.72);
        let strut = Shape::limb(
            Vec3::new(0.0, 0.1, outward * (HALF - 0.1)),
            Vec3::new(0.0, 0.2, z),
            0.1,
            0.1,
            8,
            GREY,
        );
        station.append(&strut.with_seed(PAINT));
        station.append(&laboratory(z, outward));
    }
    station.append(&dish(-(HALF + 0.72), -1.0));
    station
}

/// The visiting vehicle, nose along +x: a capsule on a gold service module with two small arrays
/// and a blinking beacon.
fn vehicle() -> Shape {
    let body = Shape::lathe(
        &[
            (0.0, 0.0),
            (0.26, 0.0),
            (0.42, 0.32),
            (0.55, 0.36),
            (0.55, 1.2),
            (0.62, 1.22),
            (0.62, 1.34),
            (0.42, 1.8),
            (0.16, 2.1),
            (0.0, 2.14),
        ],
        14,
        WHITE,
    );
    let mut shape = body.with_seed(PAINT);
    let service = Shape::limb(
        Vec3::new(0.0, 0.36, 0.0),
        Vec3::new(0.0, 1.2, 0.0),
        0.58,
        0.58,
        14,
        GOLD,
    );
    shape.append(&service.with_seed(FOIL));
    for side in [-1.0, 1.0] {
        let mut wing = Shape::new();
        wing.quad(
            [
                Vec3::new(side * 0.6, 0.45, 0.0),
                Vec3::new(side * 1.7, 0.45, 0.0),
                Vec3::new(side * 1.7, 1.1, 0.0),
                Vec3::new(side * 0.6, 1.1, 0.0),
            ],
            CELL,
        );
        shape.append(&wing.with_seed(PANEL));
    }
    let beacon = Shape::icosphere(0.07, 1, 0.0, 0.0, Vec3::new(1.0, 0.25, 0.2))
        .transformed(&Mat4::translation(0.0, 2.17, 0.0));
    shape.append(&beacon.with_seed(BEACON));
    shape.transformed(
        &(Mat4::scaling(0.85, 0.85, 0.85)
            * Mat4::rotate_z(-FRAC_PI_2)
            * Mat4::translation(0.0, -1.0, 0.0)),
    )
}

/// A material's response to the sun: painted shells lit and glossy, gold foil crinkled and
/// bright, solar cells mirror-dark with a sheen of earthshine, beacons glowing.
struct StationShader;

impl FragmentShaderSimd<KitVaryings> for StationShader {
    type Uniforms = ToonUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &ToonUniforms) -> ColourSimd {
        let view = &u.view;
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let distance = length(to_eye);
        let view_dir = to_eye * (s(1.0) / distance);
        let normal = facing(v.normal().normalise(), to_eye);
        let id = v.seed();
        let is = |material: f32| s(1.0) - smoothstep(0.0, 0.1, (id - s(material)).abs());
        let (panel, foil, beacon) = (is(PANEL), is(FOIL), is(BEACON));

        let crinkle = value3(world.x * s(9.0), world.y * s(9.0), world.z * s(9.0)) - s(0.5);
        let rough = normal + Vec3Simd::new(crinkle, crinkle * s(0.7), -crinkle) * (foil * s(0.35));
        let rough = rough.normalise();
        let mut colour = u.lighting.shade(view, world, rough, v.colour(), s(1.0));
        let half = (splat3(u.lighting.sun) + view_dir).normalise();
        let facing_half = rough.dot(half).fast_max(s(0.0));
        let sharp = pow_highlight(facing_half, 128.0) * (s(0.25) + panel * s(0.8));
        let broad = pow_highlight(facing_half, 16.0) * (foil * s(0.7) + s(0.05));
        colour = colour + splat3(u.lighting.sun_colour) * (sharp + broad);
        let turned = s(1.0) - clamp01(normal.dot(view_dir));
        colour = colour
            + rgb(0.1, 0.25, 0.7) * (turned * turned * (panel * s(1.2) + foil * s(0.4) + s(0.08)));
        let blink = (s(view.frame.time * 5.0).sin() * s(0.5) + s(0.5)).fast_max(s(0.15));
        colour = colour + v.colour() * (beacon * blink * s(4.0));
        scene_colour(view, colour, world)
    }
}

/// Space: a field of stars, a faint band of the galaxy and, below, Earth with its clouds and a
/// thin glowing atmosphere, lit by the same sun as the station.
struct EarthShader;

#[derive(Clone)]
struct EarthUniforms {
    view: ViewRef,
}

impl FragmentShaderSimd<ScreenVaryings> for EarthShader {
    type Uniforms = EarthUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, u: &EarthUniforms) -> ColourSimd {
        let view = &u.view;
        let [x, y] = v.ndc();
        let ray = view_ray(view, x, y);
        let time = view.frame.time;
        let sun = splat3(SUN.normalise());

        let from_centre = splat3(view.eye - EARTH_CENTRE);
        let b = from_centre.dot(ray);
        let c = from_centre.dot(from_centre) - s(EARTH_RADIUS * EARTH_RADIUS);
        let disc = b * b - c;
        let reach = -b - disc.fast_max(s(0.0)).sqrt();
        let hit = disc.simd_gt(s(0.0)) & reach.simd_gt(s(0.0));

        let axis = ray.x * s(0.35) + ray.y * s(0.9) - ray.z * s(0.2);
        let band = (-axis * axis * s(9.0)).exp();
        let dust = fbm3(ray.x * s(4.0), ray.y * s(4.0), ray.z * s(4.0), 4);
        let mut space = rgb(0.012, 0.018, 0.04) * (s(0.4) + band * dust * s(2.2));
        space = space + rgb(1.0, 0.95, 0.9) * (stars(ray.x, ray.y, ray.z, 1.0, s(time)) * s(0.9));

        let closest = (from_centre.dot(from_centre) - b * b)
            .fast_max(s(0.0))
            .sqrt();
        let altitude = (closest - s(EARTH_RADIUS)).fast_max(s(0.0));
        let limb_point = (from_centre + ray * (-b)).normalise();
        let sunward = clamp01(limb_point.dot(sun) * s(1.2) + s(0.35));
        let glow = (-altitude * s(1.0 / 1.5)).exp()
            * (s(0.25) + sunward * s(0.9))
            * (s(1.0) - hit.select(s(1.0), s(0.0)));
        let haze = rgb(0.25, 0.5, 1.15) * glow
            + rgb(0.3, 0.6, 1.0) * ((-altitude * s(0.12)).exp() * sunward * s(0.12));

        let point = splat3(view.eye) + ray * reach;
        let normal = (point - splat3(EARTH_CENTRE)) * s(1.0 / EARTH_RADIUS);
        let spin = time * 0.012;
        let (sin, cos) = spin.sin_cos();
        let rotated = Vec3Simd::new(
            normal.x * s(cos) + normal.z * s(sin),
            normal.y,
            -normal.x * s(sin) + normal.z * s(cos),
        );
        let (kx, ky, kz) = (rotated.x * s(3.2), rotated.y * s(3.2), rotated.z * s(3.2));
        let height = fbm3(kx, ky, kz, 5);
        let land = smoothstep(0.52, 0.56, height);
        let dryness = fbm3(kx * s(1.7) + s(8.0), ky * s(1.7), kz * s(1.7) - s(3.0), 3);
        let depth = smoothstep(0.3, 0.52, height);
        let ocean = mix3(rgb(0.004, 0.02, 0.1), rgb(0.01, 0.07, 0.26), depth);
        let ground = mix3(
            rgb(0.05, 0.16, 0.05),
            rgb(0.4, 0.31, 0.17),
            smoothstep(0.4, 0.65, dryness),
        );
        let surface = mix3(ocean, ground, land);
        let drift = time * 0.02;
        let cloud_noise = fbm3(
            rotated.x * s(5.5) + s(drift),
            rotated.y * s(5.5),
            rotated.z * s(5.5) - s(drift),
            5,
        );
        let cloud = smoothstep(0.5, 0.72, cloud_noise);
        let albedo = mix3(surface, rgb(0.9, 0.92, 0.96), cloud * s(0.85));

        let lambert = normal.dot(sun).fast_max(s(0.0));
        let day = smoothstep(-0.05, 0.3, normal.dot(sun));
        let mirrored = reflect(ray, normal);
        let glint = pow_highlight(mirrored.dot(sun).fast_max(s(0.0)), 256.0)
            * s(2.5)
            * (s(1.0) - land)
            * (s(1.0) - cloud);
        let limb = s(1.0) - clamp01(normal.dot(-ray));
        let atmosphere = rgb(0.25, 0.5, 1.1) * (limb * limb * limb * s(1.1));
        let lit = mix3(
            albedo * s(0.015),
            albedo * (lambert * s(1.5) + s(0.03)),
            day,
        );
        let planet =
            lit + atmosphere * (day * s(0.9) + s(0.05)) + rgb(1.0, 0.95, 0.85) * (glint * day);

        let colour = select3(hit, planet, space + haze);
        to_colour(colour, s(SKY_DISTANCE))
    }
}

pub struct OrbitalStation {
    fittings: Shape,
    vehicle: Shape,
    ice: Particles,
    station: SimdPipeline<KitVertexShader, StationShader>,
    earth: SimdPipeline<ScreenVertexShader, EarthShader>,
}

impl OrbitalStation {
    pub fn new() -> Self {
        Self {
            fittings: fittings(),
            vehicle: vehicle().with_sway(0.0),
            ice: Particles::new(110, 12),
            station: pipeline(StationShader, CullingMode::None),
            earth: SimdPipeline::new(ScreenVertexShader, EarthShader)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for OrbitalStation {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Organ],
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Beep),
                0.0,
                0.45,
                0.06,
                -0.4,
            )],
            ..WorldSound::default()
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        let toon = ToonUniforms {
            view: view.clone(),
            lighting: lighting(),
        };
        let circling = Motion::Orbit(Orbit {
            centre: Vec3::new(0.0, 1.6, -3.0),
            radius: (8.5, 8.5),
            height: (1.0, 1.0),
            speed: (1.8, 1.8),
            wiggle: 0.0,
        });

        scene.chip.draw(pass, view, &look());
        self.fittings.draw(
            pass,
            &self.station,
            KitUniforms {
                view: view.clone(),
                motion: Motion::OnChip,
            },
            toon.clone(),
        );
        self.vehicle.draw(
            pass,
            &self.station,
            KitUniforms {
                view: view.clone(),
                motion: circling,
            },
            toon,
        );

        let earth = EarthUniforms { view: view.clone() };
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.earth, (), |_| earth.clone());
        self.ice.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 0.5, -1.0),
                half_extent: Vec3::new(14.0, 6.0, 12.0),
                velocity: Vec3::new(-0.35, 0.05, 0.1),
                wander: 0.3,
                size: (0.01, 0.03),
                colour: Vec3::new(2.2, 2.2, 2.4),
                twinkle: 3.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.55,
            vignette: 0.4,
            grain: 0.01,
            aberration: 0.0,
            desaturate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_rises_below_the_chip_not_above_it() {
        let eye = Vec3::new(0.0, -0.6, 15.0);
        let top = EARTH_CENTRE.y + EARTH_RADIUS;
        assert!(top < eye.y, "the horizon is below the eye");
    }
}
