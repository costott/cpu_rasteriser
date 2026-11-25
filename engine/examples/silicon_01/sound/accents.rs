//! The synthesised sounds of the worlds, for what no recording was found for. Each is built from
//! a few shapes (gliding tones, swept noise, struck partials and showers of small grains) and
//! scaled so that its loudest sample is 1, as the recordings' excerpts are. Pitches, rates and
//! lengths are set by ear (provisional).

use std::f32::consts::PI;

use crate::sound::dsp::{self, Band, decay, fade, filter, frequency, noise, oscillator, samples};
use crate::themes::kit::random::Rng;

/// A synthesised sound a world makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accent {
    /// A stream of rising bubbles over a low wash.
    Bubbles,
    /// A spacecraft's two-tone radio beep.
    Beep,
    /// Thick bubbles breaking in lava, over its rumble.
    Lava,
    BlockPops,
    EngineSweep,
    /// A deep tone falling away.
    SubFall,
    Chime,
    /// The buzz of a cathode ray tube.
    TubeHum,
    ComicPop,
    SmallBell,
    WoodClap,
    DigitalTrickle,
    ToyHorn,
    Insects,
    IcyGust,
    SandHiss,
    CrystalTing,
    Bubble,
    RisingPop,
    StoneChime,
    BirdCall,
    SensorBeep,
    Squelch,
    SoftSweep,
    BuzzClick,
    Tinkle,
    SoftThump,
    Shimmer,
    WaterDrop,
    ScopeWhine,
    RelayClick,
    Bloop,
    Clack,
    PencilScratch,
    Ping,
    BrushSwish,
    Keystrokes,
    SoftChime,
    Stutter,
    LineZap,
}

impl Accent {
    /// `seconds` of the sound, its loudest sample 1.
    pub fn sound(self, seconds: f32) -> Vec<f32> {
        let mut sound = self.shape(seconds);
        sound.resize(samples(seconds), 0.0);
        let peak = sound.iter().fold(0.0_f32, |peak, v| peak.max(v.abs()));
        sound.iter_mut().for_each(|v| *v /= peak.max(1e-3));
        sound
    }

    fn shape(self, seconds: f32) -> Vec<f32> {
        match self {
            Self::Bubbles => bubbles(seconds),
            Self::Beep => pulses(&[1180.0, 1570.0], 0.09, 0.04),
            Self::Lava => {
                let mut sound = swept(seconds, Band::Low, 0.8, 5, |_| 110.0);
                sprinkle(&mut sound, 7.0, 41, |rng| {
                    let length = rng.range(0.1, 0.22);
                    let from = rng.range(60.0, 110.0);
                    glide(length, from, from * 2.3, length)
                });
                sound
            }
            Self::BlockPops => {
                let mut sound = vec![0.0; samples(seconds)];
                sprinkle(&mut sound, 8.0, 43, |rng| {
                    let from = rng.range(240.0, 520.0);
                    glide(0.06, from, from * 1.8, 0.06)
                });
                sound
            }
            Self::EngineSweep => {
                let climb = move |t: f32| 52.0 + 90.0 * (t / seconds).min(1.0);
                let mut sound = oscillator(seconds, climb, dsp::saw);
                filter(&mut sound, Band::Low, 1.4, |t| climb(t) * 7.0);
                for (i, value) in sound.iter_mut().enumerate() {
                    *value *= 0.75 + 0.25 * (dsp::time_of(i) * 2.0 * PI * 27.0).sin();
                }
                sound
            }
            Self::SubFall => glide(seconds, 95.0, 28.0, seconds * 1.6),
            Self::Chime => struck(seconds, 1568.0, &[1.0, 2.76, 5.4], 1.2),
            Self::TubeHum => {
                let mut sound = oscillator(seconds, |_| 60.0, dsp::square);
                filter(&mut sound, Band::Low, 1.0, |_| 700.0);
                sound
            }
            Self::ComicPop => {
                let mut sound = glide(0.1, 170.0, 950.0, 0.12);
                add(&mut sound, 0.0, &tap(0.012, 2600.0, 3), 0.6);
                sound
            }
            Self::SmallBell => struck(seconds, 2093.0, &[1.0, 2.4, 3.9], 0.8),
            Self::WoodClap => {
                let mut sound = vec![0.0; samples(0.2)];
                for (at, hz) in [(0.0, 1150.0), (0.07, 1050.0)] {
                    let mut block = swept(0.05, Band::Peak, 7.0, 47, |_| hz);
                    decay(&mut block, 0.04);
                    add(&mut sound, at, &block, 1.0);
                }
                sound
            }
            Self::DigitalTrickle => {
                let mut sound = vec![0.0; samples(seconds)];
                sprinkle(&mut sound, 28.0, 53, |rng| {
                    let hz = frequency(86.0 + [0.0, 3.0, 5.0, 7.0, 10.0, 12.0][rng.index(6)]);
                    let mut blip = oscillator(0.022, |_| hz, dsp::square);
                    decay(&mut blip, 0.022);
                    blip
                });
                filter(&mut sound, Band::Low, 0.7, |_| 5000.0);
                sound
            }
            Self::ToyHorn => {
                let mut sound = vec![0.0; samples(0.3)];
                for (at, hz) in [(0.0, 523.0), (0.14, 659.0)] {
                    let mut beep = oscillator(0.1, |_| hz, dsp::square);
                    filter(&mut beep, Band::Low, 0.8, |_| 2200.0);
                    fade(&mut beep, 0.008, 0.02);
                    add(&mut sound, at, &beep, 1.0);
                }
                sound
            }
            Self::Insects => {
                let mut sound = swept(seconds, Band::Peak, 9.0, 59, |_| 4700.0);
                for (i, value) in sound.iter_mut().enumerate() {
                    let chirr = (dsp::time_of(i) * 2.0 * PI * 36.0).sin().max(0.0);
                    *value *= chirr * chirr;
                }
                sound
            }
            Self::IcyGust => bumped(swept(seconds, Band::Peak, 3.5, 61, move |t| {
                900.0 + 1700.0 * (PI * t / seconds).sin()
            })),
            Self::SandHiss => bumped(swept(seconds, Band::High, 0.8, 67, |_| 3400.0)),
            Self::CrystalTing => {
                let mut sound = struck(seconds, 2637.0, &[1.0, 2.76], 0.6);
                add(
                    &mut sound,
                    0.08,
                    &struck(seconds, 3136.0, &[1.0, 2.76], 0.5),
                    0.6,
                );
                sound
            }
            Self::Bubble => glide(0.1, 340.0, 920.0, 0.12),
            Self::RisingPop => {
                let mut sound = glide(0.1, 420.0, 1250.0, 0.12);
                sound.resize(samples(0.25), 0.0);
                add(&mut sound, 0.09, &glide(0.09, 620.0, 1760.0, 0.1), 0.7);
                sound
            }
            Self::StoneChime => {
                let mut sound = struck(seconds, 1175.0, &[1.0, 2.1, 3.3], 0.5);
                add(&mut sound, 0.0, &tap(0.02, 1800.0, 71), 0.5);
                sound
            }
            Self::BirdCall => {
                let mut sound = vec![0.0; samples(0.45)];
                for at in [0.0, 0.2] {
                    let mut note = oscillator(
                        0.14,
                        |t| 2900.0 - 4200.0 * t + 90.0 * (t * 190.0).sin(),
                        dsp::sine,
                    );
                    fade(&mut note, 0.02, 0.05);
                    add(&mut sound, at, &note, 1.0);
                }
                sound
            }
            Self::SensorBeep => pulses(&[1750.0, 1750.0, 2330.0], 0.05, 0.05),
            Self::Squelch => {
                let mut sound = swept(0.2, Band::Peak, 6.0, 73, |t| 1300.0 - 5200.0 * t);
                fade(&mut sound, 0.02, 0.06);
                add(&mut sound, 0.03, &glide(0.14, 300.0, 110.0, 0.2), 0.5);
                sound
            }
            Self::SoftSweep => bumped(swept(seconds, Band::Peak, 2.2, 79, move |t| {
                300.0 + 2800.0 * (t / seconds)
            })),
            Self::BuzzClick => {
                let mut sound = oscillator(0.14, |_| 120.0, dsp::square);
                filter(&mut sound, Band::High, 0.8, |_| 700.0);
                fade(&mut sound, 0.002, 0.01);
                add(&mut sound, 0.0, &tap(0.012, 3500.0, 83), 2.0);
                sound
            }
            Self::Tinkle => {
                let mut sound = vec![0.0; samples(seconds)];
                sprinkle(&mut sound, 20.0, 89, |rng| {
                    struck(0.18, rng.range(2400.0, 5200.0), &[1.0, 2.76], 0.16)
                });
                sound
            }
            Self::SoftThump => {
                let mut sound = glide(0.2, 120.0, 55.0, 0.22);
                add(&mut sound, 0.0, &tap(0.05, 350.0, 97), 0.6);
                sound
            }
            Self::Shimmer => shimmer(seconds, 101),
            Self::WaterDrop => {
                let mut sound = glide(0.08, 520.0, 1500.0, 0.1);
                sound.resize(samples(0.35), 0.0);
                add(&mut sound, 0.17, &glide(0.07, 640.0, 1750.0, 0.08), 0.35);
                sound
            }
            Self::ScopeWhine => bumped(oscillator(
                seconds,
                move |t| 2100.0 + 1500.0 * (t / seconds),
                dsp::sine,
            )),
            Self::RelayClick => {
                let mut sound = vec![0.0; samples(0.12)];
                add(&mut sound, 0.0, &tap(0.012, 2400.0, 103), 1.0);
                add(&mut sound, 0.065, &tap(0.01, 3200.0, 107), 0.7);
                sound
            }
            Self::Bloop => {
                let mut sound = glide(0.28, 250.0, 85.0, 0.5);
                fade(&mut sound, 0.05, 0.08);
                sound
            }
            Self::Clack => {
                let mut sound = tap(0.025, 2300.0, 109);
                sound.resize(samples(0.1), 0.0);
                add(&mut sound, 0.0, &glide(0.03, 340.0, 280.0, 0.03), 0.8);
                add(&mut sound, 0.05, &tap(0.02, 1900.0, 113), 0.6);
                sound
            }
            Self::PencilScratch => {
                let mut sound = swept(seconds, Band::Peak, 1.6, 127, |_| 2800.0);
                for (i, value) in sound.iter_mut().enumerate() {
                    let stroke = (dsp::time_of(i) * PI * 9.0).sin().abs();
                    *value *= stroke * stroke;
                }
                sound
            }
            Self::Ping => {
                let mut sound = struck(seconds, 1320.0, &[1.0], 0.7);
                add(
                    &mut sound,
                    0.18,
                    &struck(seconds, 1320.0, &[1.0], 0.5),
                    0.35,
                );
                sound
            }
            Self::BrushSwish => bumped(swept(seconds, Band::Low, 0.8, 131, move |t| {
                900.0 + 2000.0 * (PI * t / seconds).sin()
            })),
            Self::Keystrokes => {
                let mut sound = vec![0.0; samples(seconds)];
                sprinkle(&mut sound, 15.0, 137, |rng| {
                    tap(0.014, rng.range(1500.0, 3400.0), rng.index(1000) as u64)
                });
                sound
            }
            Self::SoftChime => {
                let mut sound = struck(seconds, 880.0, &[1.0, 1.5, 3.0], 0.8);
                fade(&mut sound, 0.03, 0.05);
                sound
            }
            Self::Stutter => {
                let mut rng = Rng::new(139);
                let mut sound = Vec::new();
                while sound.len() < samples(seconds) {
                    let hz = rng.range(90.0, 2400.0);
                    let mut piece = oscillator(rng.range(0.012, 0.05), |_| hz, dsp::square);
                    if rng.unit() < 0.3 {
                        piece.fill(0.0);
                    }
                    sound.extend(piece);
                }
                dsp::crush(&mut sound, 6, 6.0);
                sound
            }
            Self::LineZap => {
                let mut sound = oscillator(0.18, |t| 180.0 + 2600.0 * (-t * 26.0).exp(), dsp::saw);
                decay(&mut sound, 0.2);
                sound
            }
        }
    }
}

/// A cluster of very high tones, each trembling at its own rate, swelling and fading: light
/// gleaming.
pub fn shimmer(seconds: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut sound = vec![0.0; samples(seconds)];
    for note in [98.0, 101.0, 105.0, 108.0, 110.0, 113.0] {
        let hz = frequency(note + rng.range(-0.15, 0.15));
        let (rate, phase) = (rng.range(8.0, 17.0), rng.range(0.0, 2.0 * PI));
        let tone = oscillator(seconds, |_| hz, dsp::sine);
        for (i, (out, value)) in sound.iter_mut().zip(tone).enumerate() {
            let tremble = 0.5 + 0.5 * (dsp::time_of(i) * 2.0 * PI * rate + phase).sin();
            *out += value * tremble * 0.16;
        }
    }
    let air = swept(seconds, Band::High, 0.7, seed + 1, |_| 6500.0);
    add(&mut sound, 0.0, &air, 0.12);
    bumped(sound)
}

/// Adds `other`, scaled by `gain`, into `sound` from `at` seconds, as far as `sound` goes.
fn add(sound: &mut [f32], at: f32, other: &[f32], gain: f32) {
    for (out, value) in sound.iter_mut().skip(samples(at)).zip(other) {
        *out += value * gain;
    }
}

/// `sound` swelled from nothing and back.
fn bumped(mut sound: Vec<f32>) -> Vec<f32> {
    let last = sound.len().max(2) - 1;
    for (i, value) in sound.iter_mut().enumerate() {
        let rise = (PI * i as f32 / last as f32).sin();
        *value *= rise * rise;
    }
    sound
}

/// Noise through a filter whose cutoff may move.
fn swept(seconds: f32, band: Band, q: f32, seed: u64, cutoff: impl Fn(f32) -> f32) -> Vec<f32> {
    let mut sound = noise(seconds, seed);
    filter(&mut sound, band, q, cutoff);
    sound
}

/// A sine gliding from `from` to `to` hertz over `seconds`, dying away over `ring`.
fn glide(seconds: f32, from: f32, to: f32, ring: f32) -> Vec<f32> {
    let mut sound = oscillator(
        seconds,
        |t| from * (to / from).powf((t / seconds).min(1.0)),
        dsp::sine,
    );
    decay(&mut sound, ring);
    fade(&mut sound, 0.003, 0.008);
    sound
}

/// A struck object: sines at `ratios` of `hz`, each higher one quieter and shorter.
fn struck(seconds: f32, hz: f32, ratios: &[f32], ring: f32) -> Vec<f32> {
    let mut sound = vec![0.0; samples(seconds)];
    for (index, ratio) in ratios.iter().enumerate() {
        let order = (index + 1) as f32;
        let mut partial = oscillator(seconds, |_| hz * ratio, dsp::sine);
        decay(&mut partial, ring / order);
        add(&mut sound, 0.0, &partial, 1.0 / order);
    }
    fade(&mut sound, 0.002, 0.01);
    sound
}

/// A short knock of noise around `hz`.
fn tap(seconds: f32, hz: f32, seed: u64) -> Vec<f32> {
    let mut sound = swept(seconds, Band::Peak, 2.0, seed, |_| hz);
    decay(&mut sound, seconds);
    fade(&mut sound, 0.001, 0.002);
    sound
}

/// Adds about `rate` grains a second to `sound`, at uneven times and levels.
fn sprinkle(sound: &mut [f32], rate: f32, seed: u64, mut grain: impl FnMut(&mut Rng) -> Vec<f32>) {
    let mut rng = Rng::new(seed);
    let length = dsp::time_of(sound.len());
    let mut at = 0.0;
    while at < length {
        let piece = grain(&mut rng);
        add(sound, at, &piece, rng.range(0.4, 1.0));
        at += rng.range(0.4, 1.6) / rate;
    }
}

/// Short sine beeps one after another.
fn pulses(pitches: &[f32], on: f32, off: f32) -> Vec<f32> {
    let mut sound = Vec::new();
    for &hz in pitches {
        let mut tone = oscillator(on, |_| hz, dsp::sine);
        fade(&mut tone, 0.004, 0.006);
        sound.extend(tone);
        sound.extend(std::iter::repeat_n(0.0, samples(off)));
    }
    sound
}

/// A stream of bubbles: short sine chirps rising as each bubble does, over a low wash.
fn bubbles(seconds: f32) -> Vec<f32> {
    let mut sound = swept(seconds, Band::Low, 0.8, 23, |t| {
        260.0 + 80.0 * (t * 3.1).sin()
    });
    sound.iter_mut().for_each(|v| *v *= 0.5);
    let mut rng = Rng::new(7);
    let mut at = 0.0;
    while at < seconds {
        let (hz, length) = (rng.range(280.0, 1100.0), rng.range(0.03, 0.09));
        let mut bubble = oscillator(length, |t| hz * (1.0 + 1.4 * t / length), dsp::sine);
        decay(&mut bubble, length);
        fade(&mut bubble, 0.006, 0.006);
        add(&mut sound, at, &bubble, rng.range(0.2, 0.7));
        at += rng.range(0.015, 0.11);
    }
    sound
}
