//! What the lagoon is dressed with: palms, a sailboat, gulls and the pier's timber.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::simd::hash2_scalar;
use crate::themes::kit::random::Rng;
use crate::themes::kit::shape::{Shape, lerp3};

/// A coconut palm standing at the origin, leaning to a random side, with a crown of drooping
/// fronds. Sway weights grow up the trunk and out along the fronds.
pub fn palm(rng: &mut Rng) -> Shape {
    let height = rng.range(5.6, 7.6);
    let heading = rng.range(0.0, TAU);
    let lean = Vec3::new(heading.cos(), 0.0, heading.sin()) * rng.range(0.8, 1.9);
    let spine = |t: f32| Vec3::new(0.0, height * t, 0.0) + lean * (t * t);
    let rings = 8;
    let mut palm = Shape::new();
    for k in 0..rings {
        let (t0, t1) = (k as f32 / rings as f32, (k + 1) as f32 / rings as f32);
        let bark = if k % 2 == 0 {
            Vec3::new(0.50, 0.40, 0.28)
        } else {
            Vec3::new(0.38, 0.30, 0.21)
        };
        palm.append(&Shape::limb(
            spine(t0),
            spine(t1),
            0.34 - 0.17 * t0,
            0.34 - 0.17 * t1,
            6,
            bark,
        ));
    }
    let crown = spine(1.0);
    let fronds = 11;
    for k in 0..fronds + 2 {
        let dead = k >= fronds;
        let angle = TAU * k as f32 / fronds as f32 + rng.range(-0.15, 0.15);
        let length = rng.range(2.5, 3.3);
        let rise = if dead { 0.1 } else { rng.range(0.55, 0.9) };
        let out = Vec3::new(angle.cos(), 0.0, angle.sin());
        let across = Vec3::new(-out.z, 0.0, out.x);
        let points = 8;
        let centre = |i: usize| {
            let t = i as f32 / (points - 1) as f32;
            crown
                + out * (t * length)
                + Vec3::new(0.0, length * (rise * t - 0.95 * t * t) + 0.12, 0.0)
        };
        let widths = [0.1, 0.75, 0.45, 0.72, 0.4, 0.55, 0.24, 0.0];
        let tone = rng.range(0.0, 1.0);
        for i in 0..points - 1 {
            let t = i as f32 / (points - 1) as f32;
            let colour = if dead {
                lerp3(Vec3::new(0.46, 0.38, 0.16), Vec3::new(0.6, 0.5, 0.22), t)
            } else {
                lerp3(
                    lerp3(
                        Vec3::new(0.06, 0.26, 0.07),
                        Vec3::new(0.1, 0.32, 0.08),
                        tone,
                    ),
                    lerp3(
                        Vec3::new(0.3, 0.55, 0.12),
                        Vec3::new(0.42, 0.62, 0.16),
                        tone,
                    ),
                    t,
                )
            };
            for side in [-1.0, 1.0] {
                let edge = |j: usize| {
                    centre(j)
                        + across * (widths[j] * 0.5 * side)
                        + Vec3::new(0.0, -widths[j] * 0.2, 0.0)
                };
                palm.quad([centre(i), centre(i + 1), edge(i + 1), edge(i)], colour);
            }
        }
    }
    for offset in [(0.25, -0.2, 0.1), (-0.15, -0.22, 0.2), (0.05, -0.25, -0.22)] {
        let nut = Shape::icosphere(0.2, 0, 0.1, rng.range(0.0, 20.0), Vec3::new(0.3, 0.4, 0.12))
            .transformed(&Mat4::translation_vec(
                crown + Vec3::new(offset.0, offset.1, offset.2),
            ));
        palm.append(&nut);
    }
    palm.sway_by(|p| {
        let trunk = (p.y / height).clamp(0.0, 1.0).powf(1.6) * 0.5;
        let reach = Vec3::new(p.x - lean.x, 0.0, p.z - lean.z).length();
        (trunk + reach * 0.22).min(1.0)
    })
}

/// A small sloop with its bow along +x, the waterline at y = 0 and the mast 5 units tall. Its
/// sails are one-sided: draw it with culling off.
pub fn sailboat() -> Shape {
    let hull_white = Vec3::new(0.93, 0.93, 0.9);
    let antifoul = Vec3::new(0.5, 0.1, 0.08);
    let (stern, length) = (-1.9, 3.8);
    let stations = 9;
    let section = |i: usize| {
        let u = i as f32 / (stations - 1) as f32;
        let beam = if u < 0.35 {
            0.55 + 0.25 * (u / 0.35)
        } else {
            0.8 * (1.0 - ((u - 0.35) / 0.65).powf(1.6)).max(0.0).powf(0.8)
        };
        let deck = 0.42 + 0.24 * u.powi(3);
        let keel = -0.55 * (1.0 - u.powf(1.5));
        let x = stern + length * u;
        [
            Vec3::new(x, deck, -beam),
            Vec3::new(x, 0.0, -beam * 0.9),
            Vec3::new(x, keel, 0.0),
            Vec3::new(x, 0.0, beam * 0.9),
            Vec3::new(x, deck, beam),
        ]
    };
    let mut boat = Shape::new();
    for i in 0..stations - 1 {
        let (a, b) = (section(i), section(i + 1));
        boat.quad([a[0], b[0], b[1], a[1]], hull_white);
        boat.quad([a[1], b[1], b[2], a[2]], antifoul);
        boat.quad([a[2], b[2], b[3], a[3]], antifoul);
        boat.quad([a[3], b[3], b[4], a[4]], hull_white);
        boat.quad([a[0], b[0], b[4], a[4]], Vec3::new(0.72, 0.56, 0.38));
    }
    let transom = section(0);
    boat.quad([transom[0], transom[1], transom[2], transom[3]], antifoul);
    boat.append(&Shape::cuboid(
        Vec3::new(-0.45, 0.42, -0.36),
        Vec3::new(0.65, 0.78, 0.36),
        Vec3::new(0.95, 0.95, 0.92),
    ));
    for side in [-1.0, 1.0] {
        boat.append(&Shape::cuboid(
            Vec3::new(-0.2, 0.54, 0.355 * side - 0.012),
            Vec3::new(0.45, 0.7, 0.355 * side + 0.012),
            Vec3::new(0.08, 0.14, 0.2),
        ));
    }
    let mast = Vec3::new(0.4, 0.78, 0.0);
    let top = Vec3::new(0.4, 5.0, 0.0);
    boat.append(&Shape::limb(
        mast,
        top,
        0.06,
        0.03,
        5,
        Vec3::new(0.82, 0.82, 0.8),
    ));
    boat.append(&Shape::limb(
        Vec3::new(0.4, 1.1, 0.0),
        Vec3::new(-1.55, 1.05, 0.0),
        0.04,
        0.035,
        5,
        Vec3::new(0.82, 0.82, 0.8),
    ));
    let main = [
        Vec3::new(0.4, 1.2, 0.0),
        Vec3::new(0.4, 4.8, 0.0),
        Vec3::new(-1.5, 1.15, 0.0),
    ];
    let billow = (main[0] + main[1] + main[2]) / 3.0 + Vec3::new(0.0, 0.0, 0.32);
    let panels = [
        Vec3::new(0.98, 0.96, 0.92),
        Vec3::new(0.94, 0.92, 0.88),
        Vec3::new(0.97, 0.95, 0.9),
    ];
    for (k, (a, b)) in [(main[0], main[1]), (main[1], main[2]), (main[2], main[0])]
        .into_iter()
        .enumerate()
    {
        boat.triangle(a, b, billow, panels[k]);
    }
    let jib = [
        Vec3::new(0.46, 4.5, 0.0),
        Vec3::new(1.85, 0.62, 0.0),
        Vec3::new(0.46, 0.75, 0.0),
    ];
    let jib_belly = (jib[0] + jib[1] + jib[2]) / 3.0 + Vec3::new(0.0, 0.0, -0.26);
    for (k, (a, b)) in [(jib[0], jib[1]), (jib[1], jib[2]), (jib[2], jib[0])]
        .into_iter()
        .enumerate()
    {
        boat.triangle(a, b, jib_belly, panels[(k + 1) % 3]);
    }
    boat.triangle(
        top,
        top + Vec3::new(-0.55, -0.12, 0.0),
        top + Vec3::new(0.0, -0.3, 0.0),
        Vec3::new(0.85, 0.15, 0.1),
    );
    boat
}

/// A seagull on the wing at `time`, nose along +x, up +y, wings along z. It flaps in bursts and
/// glides between them; `phase` varies each bird.
pub fn gull(time: f32, phase: f32) -> Shape {
    let white = Vec3::new(0.97, 0.97, 0.95);
    let grey = Vec3::new(0.62, 0.66, 0.7);
    let tip = Vec3::new(0.1, 0.11, 0.14);
    let mut gull =
        Shape::icosphere(1.0, 0, 0.0, 0.0, white).transformed(&Mat4::scaling(0.4, 0.12, 0.15));
    gull.append(
        &Shape::icosphere(0.095, 0, 0.0, 0.0, white)
            .transformed(&Mat4::translation(0.43, 0.06, 0.0)),
    );
    let beak = Vec3::new(0.62, 0.035, 0.0);
    gull.triangle(
        Vec3::new(0.5, 0.075, 0.0),
        Vec3::new(0.5, 0.01, 0.0),
        beak,
        Vec3::new(0.95, 0.65, 0.1),
    );
    gull.quad(
        [
            Vec3::new(-0.3, 0.02, -0.07),
            Vec3::new(-0.3, 0.02, 0.07),
            Vec3::new(-0.62, 0.0, 0.11),
            Vec3::new(-0.62, 0.0, -0.11),
        ],
        grey,
    );
    let glide = (time * 0.37 + phase * 3.0).sin() * 0.5 + 0.5;
    let beat = (time * (5.0 + phase) + phase * 7.0).sin() * (0.15 + 0.6 * glide.powf(0.7)) + 0.12;
    for side in [-1.0, 1.0] {
        let wing =
            |x: f32, reach: f32| Vec3::new(x, reach * beat.sin() * 0.55, side * reach * beat.cos());
        let (root_front, root_back) = (
            Vec3::new(0.16, 0.03, 0.07 * side),
            Vec3::new(-0.24, 0.03, 0.07 * side),
        );
        let (elbow_front, elbow_back) = (wing(0.14, 0.75), wing(-0.3, 0.72));
        let (tip_front, tip_back) = (
            wing(-0.12, 1.5) + Vec3::new(0.0, -0.05, 0.0),
            wing(-0.46, 1.42),
        );
        gull.quad([root_front, elbow_front, elbow_back, root_back], white);
        gull.quad([elbow_front, tip_front, tip_back, elbow_back], grey);
        gull.triangle(tip_front, tip_back, wing(-0.3, 1.66), tip);
    }
    gull
}

/// A gull's pose along a circle about `centre`, as a matrix: it banks into the turn and rides a
/// slow swell in height.
pub fn gull_flight(time: f32, index: usize, centre: Vec3) -> Mat4 {
    let seed = hash2_scalar(index as f32 * 3.7, 1.3);
    let direction = if seed < 0.25 { -1.0 } else { 1.0 };
    let radius = 5.0 + 2.4 * (index % 3) as f32 + seed * 1.5;
    let angular = (0.28 + 0.12 * seed) * direction;
    let angle = seed * TAU + time * angular;
    let height = 1.6 + (0.65 * index as f32) % 3.4 + 0.3 * (time * 0.8 + seed * 20.0).sin();
    let position = centre + Vec3::new(radius * angle.cos(), height, radius * angle.sin() * 0.7);
    let (heading_x, heading_z) = (-angle.sin() * direction, angle.cos() * 0.7 * direction);
    let yaw = (-heading_z).atan2(heading_x);
    Mat4::translation_vec(position)
        * Mat4::rotate_y(yaw)
        * Mat4::rotate_x(0.3 * direction + (time * 0.5 + seed).sin() * 0.05)
}

/// A rounded beach shrub: a lumpy clump of leaves.
pub fn shrub(rng: &mut Rng) -> Shape {
    let leaf = lerp3(
        Vec3::new(0.1, 0.3, 0.08),
        Vec3::new(0.2, 0.42, 0.1),
        rng.unit(),
    );
    Shape::icosphere(1.0, 1, 0.3, rng.range(0.0, 60.0), leaf)
        .transformed(&Mat4::scaling(
            rng.range(0.9, 1.4),
            rng.range(0.55, 0.85),
            rng.range(0.9, 1.3),
        ))
        .colour_by(move |p, n| {
            lerp3(
                leaf * 0.8,
                leaf * 1.5 + Vec3::new(0.04, 0.05, 0.0),
                (n.y * 0.5 + 0.5).powf(1.4) * (p.y / 1.0).clamp(0.5, 1.0),
            )
        })
        .facet_variation(rng, 0.12)
        .sway_by_height(0.0, 1.0, 1.0)
}

/// A tuft of pale beach grass.
pub fn beach_grass(rng: &mut Rng) -> Shape {
    let mut tuft = Shape::new();
    for _ in 0..7 {
        let angle = rng.range(0.0, TAU);
        let height = rng.range(0.45, 0.95);
        let lean = rng.range(0.1, 0.45);
        let base = Vec3::new(angle.cos() * 0.08, 0.0, angle.sin() * 0.08);
        let side = Vec3::new(-angle.sin(), 0.0, angle.cos()) * 0.04;
        let tip = base + Vec3::new(angle.cos() * lean, height, angle.sin() * lean);
        tuft.triangle(base - side, base + side, tip, Vec3::ONE);
    }
    tuft.gradient_by_height(
        0.0,
        0.9,
        Vec3::new(0.25, 0.36, 0.1),
        Vec3::new(0.78, 0.76, 0.34),
    )
    .sway_by_height(0.0, 0.9, 1.0)
}

/// A dark weathered boulder.
pub fn rock(rng: &mut Rng) -> Shape {
    let stone = Vec3::new(0.34, 0.31, 0.29);
    Shape::icosphere(1.0, 1, 0.3, rng.range(0.0, 90.0), stone)
        .transformed(&Mat4::scaling(
            rng.range(0.9, 1.3),
            rng.range(0.55, 0.8),
            rng.range(0.85, 1.2),
        ))
        .colour_by(move |_, n| {
            lerp3(
                stone * 0.8,
                stone * 1.35 + Vec3::new(0.05, 0.05, 0.03),
                (n.y * 0.5 + 0.5).powi(2),
            )
        })
        .facet_variation(rng, 0.1)
}

/// A hibiscus bloom on a short stem.
pub fn hibiscus(rng: &mut Rng) -> Shape {
    let colours = [
        Vec3::new(0.95, 0.25, 0.3),
        Vec3::new(1.0, 0.6, 0.25),
        Vec3::new(0.98, 0.9, 0.85),
    ];
    let colour = colours[rng.index(colours.len())];
    let mut flower = Shape::new();
    flower.append(
        &Shape::icosphere(0.09, 0, 0.0, 0.0, colour).transformed(&Mat4::scaling(1.5, 0.45, 1.5)),
    );
    flower.sway_by_height(0.0, 0.1, 1.0)
}

/// The pier's deck: planks across, on stringers below. The deck's top is at `top`, it is `width`
/// wide (x) and runs from `z_near` back to `z_far`.
pub fn pier_deck(rng: &mut Rng, top: f32, half_width: f32, z_near: f32, z_far: f32) -> Shape {
    let mut deck = Shape::new();
    let pitch = 0.42;
    let planks = ((z_near - z_far) / pitch) as usize;
    for k in 0..planks {
        let z1 = z_near - k as f32 * pitch;
        let z0 = z1 - pitch + 0.045;
        let weathering = rng.range(0.84, 1.12);
        let colour = Vec3::new(0.5, 0.4, 0.3) * weathering;
        let overhang = rng.range(0.0, 0.12);
        let lift = rng.range(-0.012, 0.012);
        let plank = Shape::cuboid(
            Vec3::new(-half_width - overhang, top - 0.14, z0),
            Vec3::new(half_width + overhang, top + lift, z1),
            colour,
        )
        .with_seed(rng.unit());
        deck.append(&plank);
    }
    for x in [-half_width + 0.3, 0.0, half_width - 0.3] {
        deck.append(&Shape::cuboid(
            Vec3::new(x - 0.12, top - 0.38, z_far),
            Vec3::new(x + 0.12, top - 0.14, z_near),
            Vec3::new(0.28, 0.22, 0.17),
        ));
    }
    deck
}

/// A pile standing from `bottom` to `top` at the origin; with `bollard` set it carries a rounded
/// cap above the deck.
pub fn pile(rng: &mut Rng, bottom: f32, top: f32, bollard: bool) -> Shape {
    let wood = Vec3::new(0.38, 0.3, 0.22) * rng.range(0.85, 1.1);
    let mut pile = Shape::cylinder(0.22, 0.19, top - bottom, 6, wood)
        .transformed(&Mat4::translation(0.0, bottom, 0.0));
    if bollard {
        pile.append(
            &Shape::icosphere(0.24, 1, 0.0, 0.0, wood * 1.15)
                .transformed(&(Mat4::translation(0.0, top, 0.0) * Mat4::scaling(1.0, 0.6, 1.0))),
        );
    }
    pile.with_seed(rng.unit())
}

/// A barrel standing on the origin, with two dark hoops.
pub fn barrel(colour: Vec3) -> Shape {
    let hoop = Vec3::new(0.12, 0.1, 0.09);
    let mut barrel = Shape::lathe(
        &[
            (0.0, 0.0),
            (0.34, 0.0),
            (0.4, 0.45),
            (0.34, 0.9),
            (0.0, 0.9),
        ],
        10,
        colour,
    );
    for height in [0.2_f32, 0.7] {
        let radius = 0.34 + 0.06 * (1.0 - ((height - 0.45) / 0.45).powi(2)) + 0.012;
        barrel.append(&Shape::lathe(
            &[(radius, height - 0.035), (radius, height + 0.035)],
            10,
            hoop,
        ));
    }
    barrel
}

/// The crate frame and brass studs that make the chip a piece of cargo, in the chip's space.
pub fn cargo_frame(rng: &mut Rng, half: f32) -> Shape {
    let timber = Vec3::new(0.42, 0.32, 0.23);
    let brass = Vec3::new(0.85, 0.62, 0.22);
    let (rail, height) = (0.2, 0.1);
    let mut frame = Shape::new();
    for (min, max) in [
        (
            Vec3::new(-half, 0.0, -half),
            Vec3::new(half, height, -half + rail),
        ),
        (
            Vec3::new(-half, 0.0, half - rail),
            Vec3::new(half, height, half),
        ),
        (
            Vec3::new(-half, 0.0, -half + rail),
            Vec3::new(-half + rail, height, half - rail),
        ),
        (
            Vec3::new(half - rail, 0.0, -half + rail),
            Vec3::new(half, height, half - rail),
        ),
    ] {
        frame.append(&Shape::cuboid(min, max, timber * rng.range(0.9, 1.1)));
    }
    let studs = 6;
    for k in 0..studs {
        let along = -half + 0.3 + (2.0 * half - 0.6) * k as f32 / (studs - 1) as f32;
        let inset = half - rail * 0.5;
        for (x, z) in [
            (along, -inset),
            (along, inset),
            (-inset, along),
            (inset, along),
        ] {
            frame.append(
                &Shape::icosphere(0.045, 0, 0.0, 0.0, brass)
                    .transformed(&Mat4::translation(x, height, z)),
            );
        }
    }
    frame
}
