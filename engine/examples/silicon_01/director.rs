//! The film's script: its phases, where the chip and the camera are, the screen-wide effects and
//! the captions, all as functions of the time since the loop started.
//!
//! The chip is modelled lying flat with its die facing +y; it is posed by a pitch about x
//! (90° turns the die towards a camera on +z), then a yaw about y. Every move eases in and out
//! between keyframes, with a slow drift on top so no frame is ever still.

use std::f32::consts::TAU;

use cpu_rasteriser::prelude::*;

use crate::shaders::Frame;
use crate::shaders::composite::Tiles;
use crate::typography::Reveal;

pub const LOOP_LENGTH: f32 = 114.75;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The screen's tiles light up over a close-up of the die's face.
    PowerOn,
    /// A slow glide across the die's face.
    Macro,
    /// The camera pulls out to the whole chip, which turns to a new pose for each caption.
    Reveal,
    /// The chip tilts to face up and drifts to the bottom of the frame.
    Rise,
    /// Triangles stream out of the die and assemble the teapot above it.
    Teapot,
    /// The chip shudders and projects a light frustum holding the fractal hologram.
    Hologram,
    /// The chip spins ever faster, shaking.
    BuildUp,
    /// Light floods the frame, the renderer's primitives burst out, and the chip swings back to
    /// the centre.
    Eruption,
    /// The chip passes through the themed worlds, faster and faster.
    Worlds,
    /// The tiles switch off.
    Shutdown,
}

impl Phase {
    pub const ALL: [Phase; 10] = [
        Phase::PowerOn,
        Phase::Macro,
        Phase::Reveal,
        Phase::Rise,
        Phase::Teapot,
        Phase::Hologram,
        Phase::BuildUp,
        Phase::Eruption,
        Phase::Worlds,
        Phase::Shutdown,
    ];

    pub const fn start(self) -> f32 {
        match self {
            Phase::PowerOn => 0.0,
            Phase::Macro => 6.0,
            Phase::Reveal => 12.0,
            Phase::Rise => 34.0,
            Phase::Teapot => 37.0,
            Phase::Hologram => 47.0,
            Phase::BuildUp => 59.0,
            Phase::Eruption => ERUPTION,
            Phase::Worlds => 69.0,
            Phase::Shutdown => SHUTDOWN,
        }
    }

    pub fn end(self) -> f32 {
        self.next().map_or(LOOP_LENGTH, Phase::start)
    }

    pub fn at(time: f32) -> Phase {
        Phase::ALL
            .into_iter()
            .rev()
            .find(|phase| time >= phase.start())
            .unwrap_or(Phase::PowerOn)
    }

    pub fn next(self) -> Option<Phase> {
        Phase::ALL
            .into_iter()
            .find(|phase| phase.start() > self.start())
    }

    pub fn previous(self) -> Option<Phase> {
        Phase::ALL
            .into_iter()
            .rev()
            .find(|phase| phase.start() < self.start())
    }
}

/// The moment the chip erupts: the end of the build-up's spin and the peak of the flash.
const ERUPTION: f32 = 66.5;
/// When the build-up's spin starts, and the turns it makes by the eruption.
const SPIN_START: f32 = Phase::BuildUp.start() + 1.0;
const SPIN_TURNS: f32 = 14.0;
/// The turns the blast carries the chip through, and how long it takes over them: what starting at
/// the build-up's final speed and easing to rest as a cube needs.
const COAST_TURNS: f32 = 5.0;
pub const COAST: f32 = COAST_TURNS * (ERUPTION - SPIN_START) / SPIN_TURNS;
/// When the worlds give way to the shutdown.
const SHUTDOWN: f32 = 106.75;
/// How long before the worlds end the picture starts to blur, in seconds; the soundtrack's final
/// roll starts with it (set by eye, provisional).
pub const FINALE_BLUR: f32 = 1.2;
/// When the tiles start switching off, and when the CRT collapse takes the last of the picture.
pub const SHUTDOWN_TILES: f32 = SHUTDOWN + 1.0;
pub const COLLAPSE: (f32, f32) = (SHUTDOWN + 6.6, SHUTDOWN + 7.4);
/// How long the screen stays black before the first tile can power on.
pub const POWER_ON_DELAY: f32 = 0.4;

/// Where the camera is and what it looks at.
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    pub eye: Vec3,
    pub target: Vec3,
    /// Vertical field of view, in degrees.
    pub fov: f32,
}

/// The chip's placement: its centre, and its pitch and yaw in radians.
#[derive(Clone, Copy, Debug)]
struct Pose {
    position: Vec3,
    pitch: f32,
    yaw: f32,
}

impl Pose {
    fn to_array(self) -> [f32; 5] {
        let p = self.position;
        [p.x, p.y, p.z, self.pitch, self.yaw]
    }
}

/// The chip facing the camera, centred, turned a little so the opening's close-up has depth.
const FACING: Pose = Pose {
    position: Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    },
    pitch: 1.44,
    yaw: -0.08,
};
/// The chip lying face up, low in the frame, for the projections.
const LOW: Pose = Pose {
    position: Vec3 {
        x: 0.0,
        y: -2.3,
        z: 0.0,
    },
    pitch: 0.3,
    yaw: 0.0,
};

/// The chip's poses. Each hold is a pair of keys a little apart, so the chip keeps drifting
/// through it, and each move between holds eases in and out. The reveal shows the die's face, its
/// edge, the contacts underneath and the face again, one pose for each caption; the worlds turn
/// the chip slowly across the frame, about 27° off frontal.
const CHIP_KEYS: [(f32, Pose); 18] = [
    (0.0, FACING),
    (Phase::Macro.start(), pose(1.47, -0.03)),
    (Phase::Reveal.start(), pose(1.5, 0.02)),
    (Phase::Reveal.start() + 3.2, pose(1.12, -0.42)),
    (Phase::Reveal.start() + 5.0, pose(1.17, -0.31)),
    (Phase::Reveal.start() + 7.6, pose(0.98, 0.5)),
    (Phase::Reveal.start() + 9.3, pose(1.02, 0.6)),
    (Phase::Reveal.start() + 11.5, pose(-1.2, 0.12)),
    (Phase::Reveal.start() + 14.4, pose(-1.16, 0.26)),
    (Phase::Reveal.start() + 16.7, pose(1.18, 0.34)),
    (Phase::Reveal.start() + 20.7, pose(1.22, 0.24)),
    (Phase::Rise.start() + 2.5, LOW),
    (ERUPTION, LOW),
    (Phase::Worlds.start() - 0.2, pose(1.22, -0.32)),
    (Phase::Worlds.start() + 18.0, pose(1.24, 0.0)),
    (SHUTDOWN, pose(1.22, 0.32)),
    (SHUTDOWN + 3.0, pose(1.35, 0.12)),
    (LOOP_LENGTH, FACING),
];

const fn pose(pitch: f32, yaw: f32) -> Pose {
    Pose {
        position: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        pitch,
        yaw,
    }
}

impl Shot {
    fn to_array(self) -> [f32; 7] {
        let (e, t) = (self.eye, self.target);
        [e.x, e.y, e.z, t.x, t.y, t.z, self.fov]
    }
}

/// The opening fills the frame with the chip's face and creeps in; the macro shots glide low
/// across the die, close enough for its cells to show.
const OPENING: Shot = Shot {
    eye: Vec3 {
        x: -0.45,
        y: -0.85,
        z: 3.45,
    },
    target: Vec3 {
        x: 0.05,
        y: 0.04,
        z: 0.0,
    },
    fov: 30.0,
};
const OPENING_CLOSE: Shot = Shot {
    eye: Vec3 {
        x: -0.3,
        y: -0.75,
        z: 3.0,
    },
    target: Vec3 {
        x: 0.08,
        y: 0.06,
        z: 0.0,
    },
    fov: 30.0,
};
const MACRO_START: Shot = Shot {
    eye: Vec3 {
        x: -0.85,
        y: -1.3,
        z: 1.3,
    },
    target: Vec3 {
        x: -0.4,
        y: 0.0,
        z: 0.08,
    },
    fov: 28.0,
};
const MACRO_END: Shot = Shot {
    eye: Vec3 {
        x: 0.5,
        y: -1.05,
        z: 1.22,
    },
    target: Vec3 {
        x: 0.3,
        y: 0.15,
        z: 0.08,
    },
    fov: 28.0,
};
/// The reveal frames the whole chip above the captions, which sit about 78% down the frame.
const REVEAL: Shot = Shot {
    eye: Vec3 {
        x: 0.0,
        y: -0.2,
        z: 13.0,
    },
    target: Vec3 {
        x: 0.0,
        y: -0.75,
        z: 0.0,
    },
    fov: 30.0,
};
/// The worlds frame the centred chip, as every theme is composed around it.
const WIDE: Shot = Shot {
    eye: Vec3 {
        x: 0.0,
        y: -0.6,
        z: 15.0,
    },
    target: Vec3 {
        x: 0.0,
        y: -0.9,
        z: 0.0,
    },
    fov: 30.0,
};
/// The high shot looks down on the chip lying near the bottom.
const HIGH: Shot = Shot {
    eye: Vec3 {
        x: 0.0,
        y: 2.6,
        z: 17.5,
    },
    target: Vec3 {
        x: 0.0,
        y: 1.05,
        z: 0.0,
    },
    fov: 32.0,
};
/// The build-up comes down to the chip's level and pushes in slowly, holding the spinning chip
/// about 60% down the frame so its caption sits beneath it.
const BUILD: Shot = Shot {
    eye: Vec3 {
        x: 0.0,
        y: 0.55,
        z: 14.0,
    },
    target: Vec3 {
        x: 0.0,
        y: -1.6,
        z: 0.0,
    },
    fov: 32.0,
};
const BUILD_CLOSE: Shot = Shot {
    eye: Vec3 {
        x: 0.0,
        y: 0.25,
        z: 12.0,
    },
    target: Vec3 {
        x: 0.0,
        y: -1.6,
        z: 0.0,
    },
    fov: 32.0,
};

const SHOT_KEYS: [(f32, Shot); 14] = [
    (0.0, OPENING),
    (5.6, OPENING_CLOSE),
    (7.4, MACRO_START),
    (Phase::Reveal.start(), MACRO_END),
    (Phase::Reveal.start() + 4.5, REVEAL),
    (Phase::Rise.start() - 1.0, REVEAL),
    (Phase::Rise.start() + 3.0, HIGH),
    (Phase::BuildUp.start(), HIGH),
    (Phase::BuildUp.start() + 2.0, BUILD),
    (ERUPTION, BUILD_CLOSE),
    (Phase::Worlds.start(), WIDE),
    (SHUTDOWN, WIDE),
    (SHUTDOWN + 2.0, WIDE),
    (LOOP_LENGTH, OPENING),
];

/// The camera at `time`, with a slow drift that scales with its distance from what it looks at.
pub fn shot(time: f32) -> Shot {
    let [ex, ey, ez, tx, ty, tz, fov] =
        glide(&SHOT_KEYS.map(|(t, shot)| (t, shot.to_array())), time);
    let (eye, target) = (Vec3::new(ex, ey, ez), Vec3::new(tx, ty, tz));
    let reach = (eye - target).length() / 15.0;
    let drift = Vec3::new(
        (time * 0.21).sin() * 0.08 + (time * 0.13).sin() * 0.05,
        (time * 0.17).cos() * 0.05,
        0.0,
    );
    Shot {
        eye: eye + drift * reach,
        target,
        fov,
    }
}

/// The chip's model matrix at `time`: its pose, the build-up's spin and what the blast leaves of
/// it, a slow drift and shake.
pub fn chip_matrix(time: f32) -> Mat4 {
    let [x, y, z, pitch, yaw] = glide(&CHIP_KEYS.map(|(t, pose)| (t, pose.to_array())), time);
    let drift = Vec3::new(
        (time * 0.31).sin() * 0.035 + (time * 0.13).sin() * 0.02,
        (time * 0.23).sin() * 0.06,
        (time * 0.19).cos() * 0.035,
    );
    let (tilt, roll) = wobble(time);
    let pitch = pitch + drift.x + tilt;
    let yaw = yaw + drift.z + spin(time);
    let shaking = shake(time);
    let jitter = Vec3::new(
        (time * 71.0).sin() * (time * 13.0).cos(),
        (time * 83.0).cos() * (time * 11.0).sin(),
        0.0,
    ) * (0.06 * shaking);
    let position = Vec3::new(x, y, z) + Vec3::new(0.0, drift.y, 0.0) + jitter;
    Mat4::translation_vec(position)
        * Mat4::rotate_y(yaw)
        * Mat4::rotate_x(pitch)
        * Mat4::rotate_z(jitter.x * 0.5 + roll)
}

/// How fast the chip is turning at `time`, in radians a second.
pub fn turn_rate(time: f32) -> f32 {
    let step = 0.004;
    let (before, after) = (chip_matrix(time), chip_matrix(time + step));
    let moved: f32 = [
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
    ]
    .into_iter()
    .map(|axis| {
        ((after * axis).xyz() - (before * axis).xyz())
            .length()
            .powi(2)
    })
    .sum();
    (moved * 0.5).sqrt() / step
}

/// Keyframed values at `time`, on a monotone cubic through the keys (Fritsch and Carlson's): the
/// motion's speed changes smoothly through every key, it comes to rest only where it turns back
/// (or between equal keys) and at the first and last keys, and it never overshoots a key.
fn glide<const N: usize>(keys: &[(f32, [f32; N])], time: f32) -> [f32; N] {
    let last = keys.len() - 1;
    let next = keys
        .iter()
        .position(|(t, _)| *t > time)
        .unwrap_or(last)
        .max(1);
    let (i, j) = (next - 1, next);
    let (t0, a) = keys[i];
    let (t1, b) = keys[j];
    let h = t1 - t0;
    let u = ((time - t0) / h).clamp(0.0, 1.0);
    let secant =
        |k: usize, c: usize| (keys[k + 1].1[c] - keys[k].1[c]) / (keys[k + 1].0 - keys[k].0);
    let tangent = |k: usize, c: usize| {
        if k == 0 || k == last {
            return 0.0;
        }
        let (before, after) = (secant(k - 1, c), secant(k, c));
        if before * after <= 0.0 {
            return 0.0;
        }
        let (h0, h1) = (keys[k].0 - keys[k - 1].0, keys[k + 1].0 - keys[k].0);
        3.0 * (h0 + h1) / ((2.0 * h1 + h0) / before + (h1 + 2.0 * h0) / after)
    };
    let (u2, u3) = (u * u, u * u * u);
    std::array::from_fn(|c| {
        (2.0 * u3 - 3.0 * u2 + 1.0) * a[c]
            + (u3 - 2.0 * u2 + u) * h * tangent(i, c)
            + (-2.0 * u3 + 3.0 * u2) * b[c]
            + (u3 - u2) * h * tangent(j, c)
    })
}

/// The chip's spin about its face. Through the build-up it accelerates as the cube of time; the
/// eruption's blast then carries it on at the same speed, winding down to rest. Each part is a
/// whole number of turns, so the poses it starts and ends on are the keyframed ones.
fn spin(time: f32) -> f32 {
    let wound = ((time - SPIN_START) / (ERUPTION - SPIN_START))
        .clamp(0.0, 1.0)
        .powi(3);
    let left = 1.0 - ((time - ERUPTION) / COAST).clamp(0.0, 1.0);
    TAU * (SPIN_TURNS * wound + COAST_TURNS * (1.0 - left.powi(3)))
}

/// The tilt about x and about z that the blast leaves the chip with, in radians: it comes round
/// like a settling coin's as it dies away (size and rates set from frame strips, provisional).
fn wobble(time: f32) -> (f32, f32) {
    let since = time - ERUPTION;
    if !(0.0..COAST).contains(&since) {
        return (0.0, 0.0);
    }
    let size = 0.3 * ramp(since, 0.0, 0.15) * (-since * 1.4).exp() * settling(time);
    let (sin, cos) = (since * 12.0).sin_cos();
    (size * sin, size * cos)
}

/// What is left of the blast's shaking at `time` as the chip comes to rest: all of it until half a
/// second before, and none from then on.
fn settling(time: f32) -> f32 {
    1.0 - ramp(time, ERUPTION + COAST - 0.5, ERUPTION + COAST)
}

/// How hard the chip shakes, 0 to 1: a shudder before the hologram, then a tremor that grows
/// through the build-up and dies away into the blast's wobble after the eruption.
pub fn shake(time: f32) -> f32 {
    let hologram = Phase::Hologram.start();
    let shudder = window(time, hologram + 0.2, hologram + 1.6, 0.3) * 0.6;
    let build = if time < ERUPTION {
        ramp(time, Phase::BuildUp.start() + 2.0, ERUPTION).powi(2)
    } else {
        (-(time - ERUPTION) * 2.2).exp() * settling(time)
    };
    shudder.max(build)
}

/// Everything the shaders need to know about this moment of the film.
pub fn frame(time: f32) -> Frame {
    let chip = chip_matrix(time);
    let phase = Phase::at(time);
    Frame {
        time,
        phase,
        phase_time: time - phase.start(),
        chip,
        chip_inverse: chip.inverse(),
        activity: ramp(time, 0.5, 3.0) * (1.0 - ramp(time, SHUTDOWN_TILES, COLLAPSE.0)),
        sweep: sweep(time),
    }
}

/// When each gleam, the specular sweep across the chip, starts: once for each new pose. Each
/// takes [`GLEAM_LENGTH`] seconds to cross, from left to right.
pub const GLEAMS: [f32; 7] = [
    Phase::Macro.start() + 2.2,
    Phase::Reveal.start() + 2.4,
    Phase::Reveal.start() + 7.2,
    Phase::Reveal.start() + 11.2,
    Phase::Reveal.start() + 16.5,
    Phase::Rise.start() + 1.0,
    Phase::Worlds.start() - 0.4,
];
pub const GLEAM_LENGTH: f32 = 1.6;

/// Where the gleam is across the chip at `time`, from -1 to 1, or 2 when there is none.
fn sweep(time: f32) -> f32 {
    GLEAMS
        .into_iter()
        .rev()
        .find(|start| time >= *start && time < start + GLEAM_LENGTH)
        .map_or(2.0, |start| (time - start) / GLEAM_LENGTH * 2.0 - 1.0)
}

/// The tile effects at `time`, with the chip at `chip_screen` in texture coordinates.
pub fn tiles(time: f32, chip_screen: Vec2) -> Tiles {
    if time < Phase::Macro.start() {
        Tiles::PowerOn {
            elapsed: time - POWER_ON_DELAY,
        }
    } else if time >= SHUTDOWN_TILES {
        Tiles::Shutdown {
            elapsed: time - SHUTDOWN_TILES,
            centre: chip_screen,
        }
    } else {
        Tiles::Done
    }
}

/// White over the picture: the eruption's half-strength burst of light, the white-out the grid of
/// primitives builds into and the first world emerges from, and the white-out the worlds blur
/// into at their end. Each rises steeply into its peak and decays quickly (rates set by eye,
/// provisional).
pub fn flash(time: f32) -> f32 {
    let eruption = 0.5 * peak(time, ERUPTION, 0.35, 3.5);
    let first_world = peak(time, Phase::Worlds.start(), 0.9, 5.0);
    let white_out = peak(time, Phase::Worlds.end(), 0.6, 3.0);
    eruption.max(first_world).max(white_out)
}

/// A flash peaking at 1 at `at`: a cubic rise over `rise` seconds, then exponential decay at
/// `decay` per second.
fn peak(time: f32, at: f32, rise: f32, decay: f32) -> f32 {
    if time < at {
        ramp(time, at - rise, at).powi(3)
    } else {
        (-(time - at) * decay).exp()
    }
}

/// The finale's zoom blur: it builds over the worlds' last [`FINALE_BLUR`] seconds as the cuts
/// reach a frame each, and is gone under the white-out (strength set by eye, provisional).
pub fn blur(time: f32) -> f32 {
    let end = Phase::Worlds.end();
    if time < end {
        0.18 * ramp(time, end - FINALE_BLUR, end)
    } else {
        0.0
    }
}

/// The CRT power-off at the very end: 0 for none, 1 for a dark screen.
pub fn collapse(time: f32) -> f32 {
    ((time - COLLAPSE.0) / (COLLAPSE.1 - COLLAPSE.0)).clamp(0.0, 1.0)
}

/// How a caption is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionKind {
    /// The film's title card.
    Title,
    /// A line of narration under the chip.
    Narration,
    /// A narration-sized line set high in the frame, over a subject that fills the middle.
    Overhead,
    /// A small line on the closing card, under the title.
    FinePrint,
    /// One word of the [`MONTAGE`], on screen for as long as its world.
    Word,
}

impl CaptionKind {
    /// The longest this kind takes to render in, in seconds (set by eye, provisional).
    fn longest_draw(self) -> f32 {
        match self {
            Self::Title => 1.5,
            Self::Narration | Self::Overhead => 0.8,
            Self::FinePrint | Self::Word => 0.5,
        }
    }
}

/// A caption on screen from `start` to `end`.
#[derive(Clone, Copy, Debug)]
pub struct Caption {
    pub text: &'static str,
    pub kind: CaptionKind,
    pub start: f32,
    pub end: f32,
}

impl Caption {
    /// How the caption renders in and cuts away, as it stands at `time`.
    pub fn reveal(&self, time: f32) -> Reveal {
        Reveal {
            age: time - self.start,
            duration: self.end - self.start,
            longest_draw: self.kind.longest_draw(),
        }
    }

    /// When the caption has finished rendering in.
    pub fn drawn(&self) -> f32 {
        self.start + self.reveal(self.start).draw_time()
    }

    /// When the caption starts to cut away.
    pub fn cuts_away(&self) -> f32 {
        self.end - self.reveal(self.start).cut_time()
    }
}

/// The grey level of captions: an off-white, softer than pure white against the dark studio.
pub const INK: f32 = 0xF2 as f32 / 255.0;

const fn caption(text: &'static str, kind: CaptionKind, start: f32, end: f32) -> Caption {
    Caption {
        text,
        kind,
        start,
        end,
    }
}

pub const CAPTIONS: [Caption; 10] = [
    caption(
        "First Light",
        CaptionKind::Title,
        Phase::Macro.start() + 1.0,
        Phase::Macro.start() + 5.0,
    ),
    caption(
        "Meet the graphics card.",
        CaptionKind::Narration,
        Phase::Reveal.start() + 2.5,
        Phase::Reveal.start() + 5.3,
    ),
    caption(
        "It’s a CPU.",
        CaptionKind::Narration,
        Phase::Reveal.start() + 5.9,
        Phase::Reveal.start() + 8.1,
    ),
    caption(
        "(The GPUs were busy training AI)",
        CaptionKind::Narration,
        Phase::Reveal.start() + 8.7,
        Phase::Reveal.start() + 11.5,
    ),
    caption(
        "Drawing every pixel by hand.",
        CaptionKind::Narration,
        Phase::Reveal.start() + 12.1,
        Phase::Reveal.start() + 16.0,
    ),
    caption(
        "On every core it’s got.",
        CaptionKind::Narration,
        Phase::Reveal.start() + 16.6,
        Phase::Reveal.start() + 20.5,
    ),
    caption(
        "Obligatory teapot.",
        CaptionKind::Overhead,
        Phase::Teapot.start() + 4.2,
        Phase::Teapot.start() + 7.2,
    ),
    caption(
        "Now let’s get carried away.",
        CaptionKind::Narration,
        Phase::BuildUp.start() + 1.0,
        Phase::BuildUp.start() + 5.8,
    ),
    // The title waits until the shutdown has darkened the tiles behind all of it, and the wider
    // fine print until its own ends are clear.
    caption("Silicon 01", CaptionKind::Title, SHUTDOWN + 3.3, COLLAPSE.0),
    caption(
        "No GPUs were harmed in the making of this film",
        CaptionKind::FinePrint,
        SHUTDOWN + 4.7,
        COLLAPSE.0,
    ),
];

/// The sentences spelled across the worlds in running order, one word per world, with one world
/// left without a word between sentences as a pause. Worlds are on screen for less than a second
/// after the first thirteen, too briefly to read a word, so the sentences fit within those.
pub const MONTAGE: [&str; 2] = [
    "Any world you can imagine.",
    "Drawn by the chip you already own.",
];

/// The [`MONTAGE`] word shown with the world at `index` in the running order, if it has one.
pub fn montage_word(index: usize) -> Option<&'static str> {
    MONTAGE
        .iter()
        .flat_map(|sentence| sentence.split(' ').map(Some).chain([None]))
        .nth(index)
        .flatten()
}

/// Rises over `fade` seconds from `start`, holds, and falls over `fade` seconds before `end`.
pub fn window(time: f32, start: f32, end: f32, fade: f32) -> f32 {
    let rise = ((time - start) / fade).clamp(0.0, 1.0);
    let fall = ((end - time) / fade).clamp(0.0, 1.0);
    smootherstep(rise.min(fall))
}

/// 0 before `start`, 1 after `end`, eased between.
pub fn ramp(time: f32, start: f32, end: f32) -> f32 {
    smootherstep(((time - start) / (end - start)).clamp(0.0, 1.0))
}

pub fn smootherstep(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chip_carries_its_spin_through_the_eruption_and_rests_facing_the_first_world() {
        // From one sixtieth of a second to the next the turn speed never jumps: stopping the
        // build-up's spin dead would change it by about 40 radians a second.
        let speeds: Vec<f32> = (0..200)
            .map(|frame| turn_rate(ERUPTION - 0.5 + frame as f32 / 60.0))
            .collect();
        assert!(
            speeds
                .windows(2)
                .all(|pair| (pair[1] - pair[0]).abs() < 8.0)
        );
        let first_world = Phase::Worlds.start();
        assert!(turn_rate(first_world) < 1.0);
        let face = chip_matrix(first_world) * Vec4::new(0.0, 1.0, 0.0, 0.0);
        assert!(face.z > 0.85, "the die faces the camera: {}", face.z);
    }

    #[test]
    fn montage_spells_one_word_per_world_with_a_silent_world_between_sentences() {
        let words: Vec<_> = (0..15).map(montage_word).collect();
        assert_eq!(
            words[..5],
            ["Any", "world", "you", "can", "imagine."].map(Some)
        );
        assert_eq!(words[5], None);
        assert_eq!(words[6], Some("Drawn"));
        assert_eq!(words[12], Some("own."));
        assert_eq!(words[13..], [None, None]);
    }
}
