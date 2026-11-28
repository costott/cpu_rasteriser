//! The caravan: camels and the people who lead them, as dark figures on a far ridge.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::themes::kit::shape::Shape;

const HIDE: Vec3 = Vec3 {
    x: 0.09,
    y: 0.06,
    z: 0.045,
};

/// A camel walking towards +x at `time`, standing on y = 0. `phase` offsets its gait.
pub fn camel(time: f32, phase: f32) -> Shape {
    let stride = time * 3.4 + phase;
    let mut camel = Shape::icosphere(1.0, 1, 0.0, 0.0, HIDE)
        .transformed(&(Mat4::translation(0.0, 1.35, 0.0) * Mat4::scaling(1.05, 0.46, 0.38)));
    camel.append(
        &Shape::icosphere(0.34, 1, 0.1, 3.0, HIDE)
            .transformed(&(Mat4::translation(0.1, 1.95, 0.0) * Mat4::scaling(1.0, 1.15, 0.9))),
    );
    camel.append(&Shape::limb(
        Vec3::new(0.85, 1.45, 0.0),
        Vec3::new(1.4, 2.25, 0.0),
        0.2,
        0.11,
        5,
        HIDE,
    ));
    camel.append(
        &Shape::icosphere(0.2, 0, 0.0, 0.0, HIDE)
            .transformed(&(Mat4::translation(1.52, 2.32, 0.0) * Mat4::scaling(1.4, 0.8, 0.8))),
    );
    camel.append(&Shape::limb(
        Vec3::new(-0.95, 1.45, 0.0),
        Vec3::new(-1.15, 0.85, 0.0),
        0.05,
        0.02,
        4,
        HIDE,
    ));
    for (hip_x, side, offset) in [
        (0.7, -1.0, 0.0),
        (0.7, 1.0, 0.5),
        (-0.65, -1.0, 0.5),
        (-0.65, 1.0, 0.0),
    ] {
        let swing = (stride + offset * TAU).sin();
        let lift = (stride + offset * TAU).cos().max(0.0);
        let hip = Vec3::new(hip_x, 1.2, side * 0.2);
        let knee = hip + Vec3::new(swing * 0.18, -0.62, 0.0);
        let foot = Vec3::new(hip_x + swing * 0.36, 0.1 * lift, side * 0.2);
        camel.append(&Shape::limb(hip, knee, 0.1, 0.06, 4, HIDE));
        camel.append(&Shape::limb(knee, foot, 0.06, 0.05, 4, HIDE));
    }
    camel
}

/// A person on foot, in a robe, walking towards +x.
pub fn walker(time: f32, phase: f32) -> Shape {
    let stride = time * 3.0 + phase;
    let mut person = Shape::cone(0.34, 1.55, 6, HIDE);
    person.append(
        &Shape::icosphere(0.16, 0, 0.0, 0.0, HIDE).transformed(&Mat4::translation(0.0, 1.68, 0.0)),
    );
    for side in [-1.0, 1.0] {
        let swing = (stride + side * 1.57).sin();
        person.append(&Shape::limb(
            Vec3::new(0.0, 0.5, side * 0.1),
            Vec3::new(swing * 0.3, 0.0, side * 0.1),
            0.07,
            0.05,
            4,
            HIDE,
        ));
    }
    person
}
