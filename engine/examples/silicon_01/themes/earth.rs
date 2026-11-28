//! Earth orbit: the chip a gold-foiled spacecraft payload with solar wings, floating in black
//! space above a procedural Earth (land, ocean, clouds, night lights, an atmosphere rim), with a
//! starfield and the moon.
//!
//! The planet and moon are ray-traced as spheres in one full-screen shader, so their silhouettes
//! and the atmosphere's glow are exact and cost nothing in triangles.

use cpu_rasteriser::prelude::*;

use crate::chip::{HALF, Look};
use crate::shaders::composite::Grade;
use crate::shaders::screen::{ScreenVaryings, ScreenVaryingsSimd, ScreenVertexShader};
use crate::shaders::{SKY_DISTANCE, ViewRef, scene_colour, view_ray};
use crate::simd::{
    F, clamp01, hash2, mix, mix3, mul3, pow_highlight, rgb, s, select3, smoothstep, splat3,
    to_colour,
};
use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;
use crate::sound::world::{Ambience, Source, Treatment, WorldSound};
use crate::themes::kit::draw::{
    KitUniforms, KitVaryings, KitVaryingsSimd, KitVertexShader, Motion, ToonUniforms, pipeline,
};
use crate::themes::kit::dress::rim_point;
use crate::themes::kit::light::{Fog, Lighting, facing};
use crate::themes::kit::noise::{fbm, fbm3};
use crate::themes::kit::particles::{Drift, Particles};
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::Shape;
use crate::themes::kit::sky::stars;
use crate::themes::{Scene, Stage};

/// The planet: its centre and radius, placed so its limb crosses the frame just under the chip.
const EARTH_CENTRE: Vec3 = Vec3 {
    x: 0.0,
    y: -40.0,
    z: -14.0,
};
const EARTH_RADIUS: f32 = 38.0;
const MOON_CENTRE: Vec3 = Vec3 {
    x: 27.0,
    y: 13.0,
    z: -90.0,
};
const MOON_RADIUS: f32 = 3.2;
/// The direction towards the sun.
const SUN: Vec3 = Vec3 {
    x: -0.95,
    y: 0.12,
    z: 0.25,
};
/// How fast the planet turns, in radians per second: far faster than the real one, so the
/// continents visibly drift in the few seconds the world is on screen.
const AXIAL_TILT: f32 = 1.15;
const SPIN: f32 = 0.035;

fn sun() -> Vec3 {
    SUN.normalise()
}

/// The key light on the spacecraft: swung round towards the camera so the foil is lit face-on while
/// the planet's terminator stays dramatic.
fn craft_sun() -> Vec3 {
    Vec3::new(-0.75, 0.35, 0.45).normalise()
}

fn lighting() -> Lighting {
    Lighting {
        sun: craft_sun(),
        sun_colour: Vec3::new(1.7, 1.62, 1.5),
        sky: Vec3::new(0.012, 0.015, 0.03),
        ground: Vec3::new(0.1, 0.17, 0.3),
        bands: 0.0,
        rim: Vec3::new(0.0, 0.0, 0.0),
        fog: Fog {
            colour: Vec3::ZERO,
            density: 0.0,
            falloff: 0.0,
            base: 0.0,
        },
    }
}

fn look() -> Look {
    let light = lighting();
    Look {
        substrate: Vec3::new(0.62, 0.46, 0.2),
        die: Vec3::new(0.02, 0.05, 0.16),
        gold: Vec3::new(1.0, 0.76, 0.28),
        streams: Vec3::new(0.35, 0.75, 1.0),
        key: light.sun,
        key_colour: light.sun_colour,
        sky: light.sky,
        ground: light.ground,
        softbox: Vec3::new(0.04, 0.07, 0.14),
    }
}

#[inline(always)]
fn spun(v: Vec3Simd, angle: F) -> Vec3Simd {
    let (sin, cos) = (angle.sin(), angle.cos());
    Vec3Simd::new(v.x * cos + v.z * sin, v.y, v.z * cos - v.x * sin)
}

/// Tips the planet's axis away from the camera: the visible cap faces the camera at a mid latitude
/// rather than the pole, which would render as ice.
#[inline(always)]
fn tilted(v: Vec3Simd) -> Vec3Simd {
    let (sin, cos) = (AXIAL_TILT.sin(), AXIAL_TILT.cos());
    Vec3Simd::new(v.x, v.y * cos - v.z * sin, v.y * sin + v.z * cos)
}

struct Planet;

impl FragmentShaderSimd<ScreenVaryings> for Planet {
    type Uniforms = ViewRef;

    #[inline(always)]
    fn shade_simd(&self, v: ScreenVaryingsSimd, view: &ViewRef) -> ColourSimd {
        let [x, y] = v.ndc();
        let ray = view_ray(view, x, y);
        let time = s(view.frame.time);
        let eye = splat3(view.eye);
        let sun = splat3(sun());

        let mut colour = space(ray, time);

        let moon_hit = sphere_hit(eye, ray, MOON_CENTRE, MOON_RADIUS);
        if moon_hit.mask.to_bitmask() != 0 {
            let n = (eye + ray * moon_hit.t - splat3(MOON_CENTRE)) * s(1.0 / MOON_RADIUS);
            colour = select3(moon_hit.mask, moon(n, sun), colour);
        }

        let hit = sphere_hit(eye, ray, EARTH_CENTRE, EARTH_RADIUS);
        let limb = atmosphere_glow(eye, ray, &hit, sun);
        colour = colour + limb;
        if hit.mask.to_bitmask() != 0 {
            let p = eye + ray * hit.t;
            let n = (p - splat3(EARTH_CENTRE)) * s(1.0 / EARTH_RADIUS);
            colour = select3(hit.mask, surface(n, -ray, sun, time), colour);
        }
        to_colour(colour, s(SKY_DISTANCE))
    }
}

struct Hit {
    mask: F,
    t: F,
    /// The ray's distance of closest approach to the sphere's centre, and the direction there.
    closest: F,
    along: F,
}

#[inline(always)]
fn sphere_hit(eye: Vec3Simd, ray: Vec3Simd, centre: Vec3, radius: f32) -> Hit {
    let oc = eye - splat3(centre);
    let b = oc.dot(ray);
    let c = oc.dot(oc) - s(radius * radius);
    let disc = b * b - c;
    let mask = disc.simd_gt(s(0.0)) & b.simd_lt(s(0.0));
    Hit {
        mask,
        t: -b - disc.fast_max(s(0.0)).sqrt(),
        closest: (oc.dot(oc) - b * b).fast_max(s(0.0)).sqrt(),
        along: -b,
    }
}

/// Black sky: a faint galactic band, and the stars.
#[inline(always)]
fn space(ray: Vec3Simd, time: F) -> Vec3Simd {
    let band_normal = Vec3::new(0.35, 0.8, -0.4).normalise();
    let across = ray.dot(splat3(band_normal));
    let band = (-across * across * s(14.0)).exp();
    let cloud = fbm(
        ray.x * s(5.0) + ray.z * s(3.0),
        ray.y * s(6.0) + ray.z * s(2.0),
        4,
    );
    let haze = rgb(0.16, 0.2, 0.34) * (band * smoothstep(0.35, 0.9, cloud) * s(0.12))
        + rgb(0.04, 0.05, 0.09) * (band * s(0.1));
    haze + rgb(1.0, 0.96, 0.92) * stars(ray.x, ray.y, ray.z, 1.0, time)
}

#[inline(always)]
fn moon(n: Vec3Simd, sun: Vec3Simd) -> Vec3Simd {
    let craters = fbm(n.x * s(9.0) + n.z * s(5.0), n.y * s(9.0) + n.z * s(7.0), 4);
    let maria = smoothstep(
        0.45,
        0.62,
        fbm(n.x * s(2.4) + s(3.0), n.y * s(2.4) + n.z * s(1.7), 3),
    );
    let albedo = mix(s(0.38), s(0.2), maria) * (s(0.7) + craters * s(0.6));
    let diffuse = smoothstep(-0.05, 0.3, n.dot(sun));
    let shade = diffuse * s(0.9) + s(0.01);
    rgb(1.0, 0.98, 0.94) * (albedo * shade)
}

/// The atmosphere seen past the limb: a thin blue glow that turns orange along the terminator.
#[inline(always)]
fn atmosphere_glow(eye: Vec3Simd, ray: Vec3Simd, hit: &Hit, sun: Vec3Simd) -> Vec3Simd {
    let closest_point = eye + ray * hit.along;
    let n = (closest_point - splat3(EARTH_CENTRE)).normalise();
    let altitude = (hit.closest - s(EARTH_RADIUS)).fast_max(s(0.0));
    let lit = smoothstep(-0.3, 0.45, n.dot(sun));
    let dusk = clamp01(s(1.0) - n.dot(sun).abs() * s(3.5));
    let tint = mix3(rgb(0.25, 0.55, 1.3), rgb(1.2, 0.5, 0.2), dusk * s(0.7));
    let falloff = (-altitude * s(1.1)).exp() * s(0.9) + (-altitude * s(0.25)).exp() * s(0.07);
    let miss = hit.mask.select(s(0.0), s(1.0)) * hit.along.simd_gt(s(0.0)).select(s(1.0), s(0.0));
    tint * (falloff * lit * miss)
}

#[inline(always)]
fn surface(n: Vec3Simd, to_eye: Vec3Simd, sun: Vec3Simd, time: F) -> Vec3Simd {
    let q = spun(tilted(n), time * s(SPIN));
    let height = fbm3(q.x * s(3.2) + s(3.7), q.y * s(3.2), q.z * s(3.2), 5);
    let moisture = fbm3(
        q.x * s(4.5) + s(9.1),
        q.y * s(4.5) + s(2.0),
        q.z * s(4.5),
        3,
    );
    let latitude = q.y.abs();
    let land = smoothstep(0.5, 0.515, height);

    let desert = smoothstep(0.42, 0.3, moisture) * smoothstep(0.6, 0.35, latitude);
    let forest = smoothstep(0.5, 0.65, moisture);
    let mut ground = mix3(rgb(0.14, 0.27, 0.06), rgb(0.03, 0.13, 0.05), forest);
    ground = mix3(ground, rgb(0.5, 0.36, 0.18), desert);
    ground = mix3(ground, rgb(0.34, 0.29, 0.24), smoothstep(0.6, 0.72, height));
    let snow = smoothstep(0.78, 0.86, height + latitude * s(0.1))
        .fast_max(smoothstep(0.88, 0.96, latitude));
    ground = mix3(ground, rgb(0.9, 0.93, 0.96), snow);

    let depth = clamp01((s(0.5) - height) * s(7.0));
    let ocean = mix3(rgb(0.07, 0.38, 0.5), rgb(0.005, 0.03, 0.15), depth.sqrt());
    let ocean = mix3(ocean, rgb(0.9, 0.95, 1.0), smoothstep(0.88, 0.95, latitude));
    let albedo = mix3(ocean, ground, land);

    let facing_sun = n.dot(sun);
    let day = smoothstep(-0.06, 0.28, facing_sun);
    let mut colour = mul3(albedo, rgb(1.0, 0.97, 0.92) * day + rgb(0.01, 0.015, 0.03));

    let half = (sun + to_eye).normalise();
    let glint = pow_highlight(clamp01(n.dot(half)), 256.0) * (s(1.0) - land) * day;
    colour = colour + rgb(1.0, 0.95, 0.85) * (glint * s(3.0));

    let cloud_dir = spun(tilted(n), time * s(SPIN * 1.35) + s(0.6));
    let density = fbm3(
        cloud_dir.x * s(4.5) + s(5.0),
        cloud_dir.y * s(4.5),
        cloud_dir.z * s(4.5),
        5,
    );
    let cover = smoothstep(0.52, 0.72, density) * s(0.85);
    let dusk = clamp01(s(1.0) - facing_sun.abs() * s(4.0));
    let cloud_tint = mix3(rgb(1.0, 1.0, 1.0), rgb(1.2, 0.65, 0.4), dusk * s(0.6));
    let cloud = cloud_tint * (day * s(1.05) + s(0.012));
    colour = mix3(colour, cloud, cover);

    let night = s(1.0) - day;
    let (cell_x, cell_y) = (
        q.x * s(220.0) + q.z * s(31.0),
        q.y * s(220.0) + q.z * s(53.0),
    );
    let (grid_x, grid_y) = (cell_x.floor(), cell_y.floor());
    let (off_x, off_y) = (cell_x - grid_x - s(0.5), cell_y - grid_y - s(0.5));
    let dot = smoothstep(0.32, 0.0, (off_x * off_x + off_y * off_y).sqrt());
    let sprinkle = smoothstep(0.55, 0.9, hash2(grid_x, grid_y)) * dot;
    let population = smoothstep(
        0.55,
        0.78,
        fbm3(q.x * s(6.0) + s(1.0), q.y * s(6.0), q.z * s(6.0), 3),
    );
    let lights = sprinkle * population * land * night * (s(1.0) - cover);
    colour = colour + rgb(1.0, 0.68, 0.3) * (lights * s(1.8));

    let rim = (s(1.0) - n.dot(to_eye).fast_max(s(0.0))).fast_min(s(1.0));
    let scatter = rim * rim * rim * smoothstep(-0.25, 0.4, facing_sun);
    colour + rgb(0.25, 0.5, 1.2) * (scatter * s(0.8))
}

#[derive(Clone)]
struct SpaceUniforms {
    toon: ToonUniforms,
    /// How hard the surface glints in the sun.
    shine: f32,
}

/// Lights the spacecraft's extras in hard sunlight with earthshine, adding a sharp glint.
struct SpacecraftShader;

impl FragmentShaderSimd<KitVaryings> for SpacecraftShader {
    type Uniforms = SpaceUniforms;

    #[inline(always)]
    fn shade_simd(&self, v: KitVaryingsSimd, u: &SpaceUniforms) -> ColourSimd {
        let (view, light) = (&u.toon.view, &u.toon.lighting);
        let world = v.world();
        let to_eye = splat3(view.eye) - world;
        let normal = facing(v.normal().normalise(), to_eye);
        let colour = light.shade(view, world, normal, v.colour(), s(1.0));
        let half = (splat3(light.sun) + to_eye.normalise()).normalise();
        let glint = pow_highlight(clamp01(normal.dot(half)), 64.0) * s(u.shine);
        scene_colour(
            view,
            colour + splat3(light.sun_colour) * (glint * s(0.35)),
            world,
        )
    }
}

/// Gold foil, solar wings and struts, on the chip in its own space.
fn spacecraft(rng: &mut Rng) -> Shape {
    let mut craft = Shape::new();
    let foil = Vec3::new(0.95, 0.68, 0.2);
    let strips = [
        (
            Vec3::new(-HALF, 0.0, -HALF),
            Vec3::new(HALF, 0.0, -HALF + 0.42),
        ),
        (
            Vec3::new(-HALF, 0.0, HALF - 0.42),
            Vec3::new(HALF, 0.0, HALF),
        ),
        (
            Vec3::new(-HALF, 0.0, -HALF + 0.42),
            Vec3::new(-HALF + 0.42, 0.0, HALF - 0.42),
        ),
        (
            Vec3::new(HALF - 0.42, 0.0, -HALF + 0.42),
            Vec3::new(HALF, 0.0, HALF - 0.42),
        ),
    ];
    for (min, max) in strips {
        let (nx, nz) = (
            ((max.x - min.x) / 0.14).ceil() as usize,
            ((max.z - min.z) / 0.14).ceil() as usize,
        );
        let mut heights = vec![0.0_f32; (nx + 1) * (nz + 1)];
        for h in &mut heights {
            *h = 0.016 + rng.range(0.0, 0.03);
        }
        let point = |i: usize, j: usize| {
            Vec3::new(
                min.x + (max.x - min.x) * i as f32 / nx as f32,
                heights[j * (nx + 1) + i],
                min.z + (max.z - min.z) * j as f32 / nz as f32,
            )
        };
        for j in 0..nz {
            for i in 0..nx {
                let shade = foil * rng.range(0.82, 1.08);
                craft.quad(
                    [
                        point(i, j + 1),
                        point(i + 1, j + 1),
                        point(i + 1, j),
                        point(i, j),
                    ],
                    shade,
                );
            }
        }
    }

    let frame = Vec3::new(0.55, 0.57, 0.62);
    let cell = Vec3::new(0.025, 0.06, 0.26);
    for side in [-1.0_f32, 1.0] {
        let boom_from = Vec3::new(side * (HALF - 0.1), -0.06, 0.0);
        let boom_to = Vec3::new(side * (HALF + 0.55), -0.06, 0.0);
        craft.append(&Shape::limb(boom_from, boom_to, 0.05, 0.05, 6, frame));
        let width = 1.7;
        for segment in 0..3 {
            let near = HALF + 0.55 + segment as f32 * 1.25;
            let (x0, x1) = (side * near.min(near + 1.17), side * near.max(near + 1.17));
            let (low, high) = (x0.min(x1), x0.max(x1));
            craft.append(&Shape::cuboid(
                Vec3::new(low, -0.1, -width * 0.5),
                Vec3::new(high, -0.07, width * 0.5),
                frame,
            ));
            let (columns, rows) = (6, 4);
            for i in 0..columns {
                for j in 0..rows {
                    let (cx, cz) = ((high - low) / columns as f32, width / rows as f32);
                    let (ax, az) = (
                        low + cx * i as f32 + cx * 0.06,
                        -width * 0.5 + cz * j as f32 + cz * 0.06,
                    );
                    let tone = if (i + j) % 2 == 0 { cell } else { cell * 1.25 };
                    craft.append(&Shape::cuboid(
                        Vec3::new(ax, -0.07, az),
                        Vec3::new(ax + cx * 0.88, -0.062, az + cz * 0.88),
                        tone,
                    ));
                }
            }
        }
        craft.append(&Shape::limb(
            Vec3::new(side * (HALF + 0.55), -0.06, 0.0),
            Vec3::new(side * (HALF + 4.3), -0.06, 0.0),
            0.03,
            0.03,
            5,
            frame,
        ));
    }

    for _ in 0..4 {
        let (spot, _) = rim_point(rng);
        let height = rng.range(0.15, 0.3);
        craft.append(&Shape::limb(
            spot,
            spot + Vec3::new(0.0, height, 0.0),
            0.035,
            0.015,
            5,
            frame,
        ));
        craft.append(
            &Shape::icosphere(0.05, 0, 0.0, 0.0, Vec3::new(1.0, 0.2, 0.15))
                .transformed(&Mat4::translation_vec(spot + Vec3::new(0.0, height, 0.0))),
        );
    }
    craft
}

pub struct Earth {
    craft: Shape,
    dust: Particles,
    spacecraft: SimdPipeline<KitVertexShader, SpacecraftShader>,
    planet: SimdPipeline<ScreenVertexShader, Planet>,
}

impl Earth {
    pub fn new() -> Self {
        let mut rng = Rng::new(404);
        Self {
            craft: spacecraft(&mut rng),
            dust: Particles::new(110, 12),
            spacecraft: pipeline(SpacecraftShader, CullingMode::BackFace),
            planet: SimdPipeline::new(ScreenVertexShader, Planet)
                .with_culling_mode(CullingMode::None)
                .with_depth_state(DepthState::READ_ONLY),
        }
    }
}

impl Stage for Earth {
    fn sound(&self) -> WorldSound {
        WorldSound {
            leads: &[Instrument::Glockenspiel],
            treatment: Treatment::Vast,
            ambience: vec![Ambience::once(
                Source::Synth(Accent::Beep),
                0.3,
                0.26,
                0.07,
                0.45,
            )],
        }
    }

    fn draw<'a>(&'a self, pass: &mut RenderPass<'_, 'a>, scene: &Scene<'a>) {
        let view = &scene.view;
        scene.chip.draw(pass, view, &look());
        self.craft.draw(
            pass,
            &self.spacecraft,
            KitUniforms {
                view: view.clone(),
                motion: Motion::OnChip,
            },
            SpaceUniforms {
                toon: ToonUniforms {
                    view: view.clone(),
                    lighting: lighting(),
                },
                shine: 0.4,
            },
        );
        let backdrop = view.clone();
        scene
            .quad
            .draw_to_render_pass_simd(pass, &self.planet, (), |_| backdrop.clone());
        self.dust.draw(
            pass,
            view,
            Drift {
                centre: Vec3::new(0.0, 0.5, 2.0),
                half_extent: Vec3::new(14.0, 6.0, 12.0),
                velocity: Vec3::new(0.3, 0.04, 0.08),
                wander: 0.0,
                size: (0.012, 0.03),
                colour: Vec3::new(1.2, 1.2, 1.3),
                twinkle: 2.0,
            },
        );
    }

    fn grade(&self) -> Grade {
        Grade {
            exposure: 1.0,
            bloom: 0.7,
            vignette: 0.45,
            grain: 0.012,
            aberration: 0.002,
            desaturate: 0.0,
        }
    }
}
