//! The forest's fox: a low-poly fox that trots in from the right, sits to look at the chip, and
//! trots off to the left, on a loop of [`PERIOD`] seconds. It is rebuilt each frame from a [`Pose`].

use std::f32::consts::PI;

use cpu_rasteriser::prelude::*;

use crate::themes::kit::shape::{Shape, lerp3};

/// Seconds for one visit: in, sit, out.
pub const PERIOD: f32 = 14.0;
/// How far into its loop the fox is when the forest comes on screen: at the frame's right edge,
/// about a second short of its resting place, so a world on screen for little more than a second
/// still shows it trot in (set by eye, provisional).
pub const ON_CUT: f32 = 2.1;

const FUR: Vec3 = Vec3 {
    x: 0.78,
    y: 0.30,
    z: 0.06,
};
const CREAM: Vec3 = Vec3 {
    x: 0.92,
    y: 0.84,
    z: 0.68,
};
const DARK: Vec3 = Vec3 {
    x: 0.09,
    y: 0.055,
    z: 0.04,
};

/// Where the fox rests, seen from the camera: to the right of the chip, in profile.
const REST: (f32, f32) = (4.5, 1.8);

/// Everything that poses the fox.
pub struct Pose {
    /// On the ground, in world space.
    pub position: Vec3,
    /// Heading as a rotation about y: 0 faces +x, and positive turns towards -z.
    pub heading: f32,
    /// Leg cycle phase, in radians, and how hard the legs swing, 0 to 1.
    pub gait: f32,
    pub stride: f32,
    /// 0 standing, 1 sitting.
    pub sit: f32,
    /// How far the head turns to the fox's right, in radians.
    pub look: f32,
    pub wag: f32,
    pub ear: f32,
}

/// The fox's pose `time` seconds into its loop, standing on `ground(x, z)`.
pub fn pose(time: f32, ground: impl Fn(f32, f32) -> f32) -> Pose {
    let t = time.rem_euclid(PERIOD);
    let smooth = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let (x, z, speed, sit);
    if t < 3.2 {
        let s = t / 3.2;
        let eased = 1.0 - (1.0 - s) * (1.0 - s);
        x = REST.0 + 11.0 * (1.0 - eased);
        z = REST.1 + 1.6 * (1.0 - eased) + 0.35 * (s * 7.0).sin() * (1.0 - s);
        speed = 2.0 * 11.0 * (1.0 - s) / 3.2;
        sit = 0.0;
    } else if t < 11.6 {
        (x, z, speed) = (REST.0, REST.1, 0.0);
        sit = smooth((t - 3.3) / 0.7) - smooth((t - 11.0) / 0.6);
    } else {
        let u = (t - 11.6) / 2.4;
        x = REST.0 - 21.0 * u * u;
        z = REST.1 + 1.4 * u;
        speed = 2.0 * 21.0 * u / 2.4;
        sit = 0.0;
    }
    let travelled = if t < 3.2 {
        let s = t / 3.2;
        11.0 * (1.0 - (1.0 - s) * (1.0 - s))
    } else if t < 11.6 {
        11.0
    } else {
        11.0 + 21.0 * ((t - 11.6) / 2.4).powi(2)
    };
    let heading = if t < 11.6 { PI - 0.2 } else { PI - 0.04 };
    let sitting = sit > 0.5;
    let glance = if sitting {
        (0.5 + 0.5 * (smooth((t - 6.6) / 0.6) - smooth((t - 8.6) / 0.6))) * -0.55 - 0.1
    } else {
        0.0
    };
    let flick = (t * 1.9).sin().max(0.92) - 0.92;
    Pose {
        position: Vec3::new(x, ground(x, z), z),
        heading,
        gait: travelled * 3.1,
        stride: (speed / 4.0).min(1.0),
        sit,
        look: glance + 0.06 * (t * 0.8).sin(),
        wag: if sitting {
            (t * 3.2).sin() * 0.7
        } else {
            (t * 9.0).sin() * 0.35 * (1.0 - sit)
        },
        ear: flick * 9.0,
    }
}

fn blob(centre: Vec3, radii: Vec3, colour: Vec3) -> Shape {
    Shape::icosphere(1.0, 1, 0.0, 0.0, colour)
        .transformed(&(Mat4::translation_vec(centre) * Mat4::scaling(radii.x, radii.y, radii.z)))
}

/// The fox's geometry for `pose`, in world space.
pub fn build(pose: &Pose) -> Shape {
    let sit = pose.sit;
    let mut fox = Shape::new();

    let pivot = Vec3::new(-0.55, 0.62, 0.0);
    let group = Mat4::translation_vec(pivot + Vec3::new(0.0, -0.34 * sit, 0.0))
        * Mat4::rotate_z(0.6 * sit)
        * Mat4::translation_vec(pivot * -1.0);
    let bob = 0.035 * pose.stride * (pose.gait * 2.0).sin().abs();
    let lift = Mat4::translation(0.0, bob, 0.0);

    let mut upper = Shape::new();
    let body = blob(Vec3::new(0.0, 0.68, 0.0), Vec3::new(0.8, 0.33, 0.29), FUR)
        .colour_by(|_, n| if n.y < -0.15 { CREAM } else { FUR });
    upper.append(&body);
    upper.append(
        &blob(
            Vec3::new(0.46, 0.72, 0.0),
            Vec3::new(0.34, 0.34, 0.27),
            CREAM,
        )
        .colour_by(|_, n| if n.y > 0.35 { FUR } else { CREAM }),
    );
    let head = Vec3::new(0.86, 0.9, 0.0);
    let turn = Mat4::translation_vec(head)
        * Mat4::rotate_y(pose.look)
        * Mat4::translation_vec(head * -1.0);
    let mut face = Shape::new();
    face.append(&blob(head, Vec3::new(0.24, 0.2, 0.21), FUR));
    face.append(&blob(
        head + Vec3::new(0.05, -0.1, 0.0),
        Vec3::new(0.2, 0.1, 0.19),
        CREAM,
    ));
    face.append(&Shape::limb(
        head + Vec3::new(0.1, -0.01, 0.0),
        head + Vec3::new(0.5, -0.09, 0.0),
        0.15,
        0.05,
        6,
        FUR,
    ));
    face.append(&blob(
        head + Vec3::new(0.52, -0.085, 0.0),
        Vec3::new(0.045, 0.045, 0.045),
        DARK,
    ));
    for side in [-1.0, 1.0] {
        let ear_base = head + Vec3::new(-0.08, 0.14, side * 0.12);
        let ear_tip =
            ear_base + Vec3::new(-0.05, 0.3 - 0.04 * pose.ear * side.max(0.0), side * 0.06);
        face.append(&Shape::limb(ear_base, ear_tip, 0.1, 0.0, 4, FUR));
        face.append(&Shape::limb(
            ear_base + Vec3::new(0.02, 0.0, 0.0),
            ear_tip + Vec3::new(0.01, -0.06, 0.0),
            0.06,
            0.0,
            4,
            DARK,
        ));
        face.append(&blob(
            head + Vec3::new(0.2, 0.06, side * 0.17),
            Vec3::new(0.03, 0.03, 0.03),
            DARK,
        ));
    }
    upper.append(&face.transformed(&turn));
    fox.append(&upper.transformed(&(lift * group)));

    let shoulder_standing = Vec3::new(0.42, 0.55, 0.0);
    let shoulder = (group * shoulder_standing.to_point4()).xyz();
    let swing = 0.55 * pose.stride;
    for side in [-1.0_f32, 1.0] {
        let front = (
            Vec3::new(shoulder.x, shoulder.y + bob, side * 0.15),
            if side < 0.0 { 0.0 } else { PI },
            Vec3::new(shoulder.x + 0.02, 0.0, side * 0.15),
            0.05,
        );
        let hind = (
            Vec3::new(-0.5, 0.52 + bob, side * 0.16),
            if side < 0.0 { PI } else { 0.0 },
            Vec3::new(-0.05, 0.0, side * 0.2),
            -0.12,
        );
        for (top, leg_phase, sitting_foot, bend) in [front, hind] {
            let angle = swing * (pose.gait + leg_phase).sin();
            let raised = 0.12 * pose.stride * (pose.gait + leg_phase).cos().max(0.0);
            let walking = Vec3::new(top.x + angle.sin() * 0.52, raised, top.z);
            let foot = lerp3(walking, sitting_foot, sit);
            let knee = lerp3(top, foot, 0.5) + Vec3::new(bend * (1.0 - 0.5 * sit), 0.0, 0.0);
            let ankle = lerp3(knee, foot, 0.55);
            fox.append(&Shape::limb(top, knee, 0.1, 0.075, 6, FUR));
            fox.append(&Shape::limb(knee, ankle, 0.075, 0.05, 6, FUR));
            fox.append(&Shape::limb(
                ankle,
                foot + Vec3::new(0.0, 0.03, 0.0),
                0.05,
                0.05,
                6,
                DARK,
            ));
            fox.append(&blob(
                foot + Vec3::new(0.05, 0.03, 0.0),
                Vec3::new(0.09, 0.04, 0.06),
                DARK,
            ));
        }
        if sit > 0.0 {
            fox.append(&blob(
                Vec3::new(-0.42, 0.26, side * 0.2),
                Vec3::new(0.26, 0.2 * sit + 0.01, 0.15 * sit + 0.01),
                FUR,
            ));
        }
    }

    let tail_standing = [
        Vec3::new(-0.7, 0.68, 0.0),
        Vec3::new(-1.0, 0.64, 0.0),
        Vec3::new(-1.32, 0.7, 0.0),
        Vec3::new(-1.58, 0.84, 0.0),
        Vec3::new(-1.74, 1.0, 0.0),
    ];
    let tail_sitting = [
        Vec3::new(-0.72, 0.3, 0.0),
        Vec3::new(-0.98, 0.1, 0.12),
        Vec3::new(-1.25, 0.1, 0.3),
        Vec3::new(-1.45, 0.1, 0.55),
        Vec3::new(-1.52, 0.1, 0.75),
    ];
    let radii = [0.12, 0.2, 0.19, 0.11, 0.0];
    let wag = [0.0, 0.05, 0.14, 0.24, 0.3];
    let tail: Vec<Vec3> = (0..5)
        .map(|i| {
            let p = lerp3(tail_standing[i], tail_sitting[i], sit);
            Vec3::new(p.x, p.y, p.z + wag[i] * pose.wag)
        })
        .collect();
    for i in 0..4 {
        let colour = if i == 3 { CREAM } else { FUR };
        fox.append(&Shape::limb(
            tail[i],
            tail[i + 1],
            radii[i],
            radii[i + 1],
            6,
            colour,
        ));
    }

    fox.transformed(
        &(Mat4::translation_vec(pose.position)
            * Mat4::rotate_y(pose.heading)
            * Mat4::scaling(1.15, 1.15, 1.15)),
    )
}
