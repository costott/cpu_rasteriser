//! The building blocks of the soundtrack: stereo tracks, oscillators, envelopes, filters,
//! effects and the limiter. Sounds are built as mono samples at [`RATE`] and placed on a
//! [`Track`] with a gain and a pan.

use std::f32::consts::{FRAC_PI_4, TAU};
use std::ops::Range;

use crate::themes::kit::random::Rng;

/// Samples per second, of everything in the soundtrack.
pub const RATE: u32 = 48_000;

/// How many samples `seconds` take.
pub fn samples(seconds: f32) -> usize {
    (seconds.max(0.0) * RATE as f32).round() as usize
}

/// The time of the sample at `index`, in seconds.
pub fn time_of(index: usize) -> f32 {
    index as f32 / RATE as f32
}

/// The frequency of a MIDI note number, which need not be whole: 69 is the A at 440 Hz.
pub fn frequency(note: f32) -> f32 {
    440.0 * ((note - 69.0) / 12.0).exp2()
}

/// A stereo stretch of sound.
#[derive(Clone)]
pub struct Track {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl Track {
    pub fn silent(seconds: f32) -> Self {
        let length = samples(seconds);
        Self {
            left: vec![0.0; length],
            right: vec![0.0; length],
        }
    }

    pub fn len(&self) -> usize {
        self.left.len()
    }

    /// Adds `sound` starting at `at` seconds, scaled by `gain` and panned from -1 (left) to 1
    /// (right) at constant power. Whatever falls outside the track is dropped.
    pub fn place(&mut self, at: f32, sound: &[f32], gain: f32, pan: f32) {
        let angle = (pan.clamp(-1.0, 1.0) + 1.0) * FRAC_PI_4;
        let (right, left) = angle.sin_cos();
        let start = (at * RATE as f32).round() as isize;
        let skip = (-start).max(0) as usize;
        let first = start.max(0) as usize;
        let count = sound
            .len()
            .saturating_sub(skip)
            .min(self.len().saturating_sub(first));
        for i in 0..count {
            let value = sound[skip + i] * gain;
            self.left[first + i] += value * left;
            self.right[first + i] += value * right;
        }
    }

    /// Adds `other`, scaled by `gain`.
    pub fn mix(&mut self, other: &Track, gain: f32) {
        for (own, theirs) in [
            (&mut self.left, &other.left),
            (&mut self.right, &other.right),
        ] {
            own.iter_mut().zip(theirs).for_each(|(a, b)| *a += b * gain);
        }
    }

    /// Scales every sample by `gain` at its time.
    pub fn automate(&mut self, gain: impl Fn(f32) -> f32) {
        for i in 0..self.len() {
            let level = gain(time_of(i));
            self.left[i] *= level;
            self.right[i] *= level;
        }
    }

    /// Replaces the stretch `span` (seconds) with what `process` makes of each of its channels,
    /// crossfading over `fade` seconds at both ends so the change doesn't click.
    pub fn treat(&mut self, span: Range<f32>, fade: f32, process: impl Fn(&mut [f32], usize)) {
        let (first, last) = (
            samples(span.start).min(self.len()),
            samples(span.end).min(self.len()),
        );
        let ramp = samples(fade).max(1);
        for (channel, data) in [&mut self.left, &mut self.right].into_iter().enumerate() {
            let mut treated = data[first..last].to_vec();
            process(&mut treated, channel);
            let length = treated.len();
            for (i, value) in treated.into_iter().enumerate() {
                let wet = (i.min(length - 1 - i) as f32 / ramp as f32).min(1.0);
                data[first + i] += (value - data[first + i]) * wet;
            }
        }
    }
}

/// `seconds` of a sound given by its value at each time.
pub fn make(seconds: f32, mut value: impl FnMut(f32) -> f32) -> Vec<f32> {
    (0..samples(seconds)).map(|i| value(time_of(i))).collect()
}

/// `seconds` of an oscillator whose frequency and shape follow the time: `wave` gives the sample
/// at a phase in turns, in [0, 1).
pub fn oscillator(
    seconds: f32,
    frequency: impl Fn(f32) -> f32,
    wave: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let mut phase = 0.0_f32;
    make(seconds, |t| {
        let value = wave(phase);
        phase = (phase + frequency(t) / RATE as f32).fract();
        value
    })
}

pub fn sine(phase: f32) -> f32 {
    (phase * TAU).sin()
}

pub fn square(phase: f32) -> f32 {
    if phase < 0.5 { 1.0 } else { -1.0 }
}

pub fn saw(phase: f32) -> f32 {
    2.0 * phase - 1.0
}

/// `seconds` of white noise; equal seeds give equal noise.
pub fn noise(seconds: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    make(seconds, |_| rng.range(-1.0, 1.0))
}

/// Fades `sound` in over `attack` seconds and out over its last `release` seconds.
pub fn fade(sound: &mut [f32], attack: f32, release: f32) {
    let length = sound.len();
    let (attack, release) = (samples(attack).max(1), samples(release).max(1));
    for (i, value) in sound.iter_mut().enumerate() {
        let rise = (i as f32 / attack as f32).min(1.0);
        let fall = ((length - 1 - i) as f32 / release as f32).min(1.0);
        *value *= rise * fall;
    }
}

/// Multiplies `sound` by an exponential decay that falls to a thousandth after `seconds`.
pub fn decay(sound: &mut [f32], seconds: f32) {
    let per_sample = 0.001_f32.powf(1.0 / samples(seconds).max(1) as f32);
    let mut level = 1.0;
    for value in sound {
        *value *= level;
        level *= per_sample;
    }
}

/// Which part of the spectrum a [`filter`] keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Low,
    High,
    /// Around the cutoff, with a gain of 1 at it.
    Peak,
}

/// Filters `sound` in place with a two-pole state variable filter (Zavalishin's trapezoidal
/// form), whose cutoff in hertz may move with the sample's time. `q` is the resonance: 0.7 is
/// flat, higher rings.
pub fn filter(sound: &mut [f32], band: Band, q: f32, cutoff: impl Fn(f32) -> f32) {
    let (mut first, mut second) = (0.0_f32, 0.0_f32);
    let damping = 1.0 / q;
    let mut g = 0.0;
    for (i, value) in sound.iter_mut().enumerate() {
        // The cutoff is only recomputed every 32 samples: it moves far more slowly than that.
        if i % 32 == 0 {
            let hz = cutoff(time_of(i)).clamp(10.0, RATE as f32 * 0.45);
            g = (std::f32::consts::PI * hz / RATE as f32).tan();
        }
        let high = (*value - (damping + g) * first - second) / (1.0 + damping * g + g * g);
        let middle = g * high + first;
        let low = g * middle + second;
        first = g * high + middle;
        second = g * middle + low;
        *value = match band {
            Band::Low => low,
            Band::High => high,
            Band::Peak => middle * damping,
        };
    }
}

/// Scales `sound` by `level` at each of its times, easing over a few milliseconds wherever the
/// level jumps, so that no jump clicks.
pub fn gate(sound: &mut [f32], level: impl Fn(f32) -> f32) {
    let ease = 1.0 / samples(0.003) as f32;
    let mut current = level(0.0);
    for (i, value) in sound.iter_mut().enumerate() {
        current += (level(time_of(i)) - current) * ease;
        *value *= current;
    }
}

/// Holds each value for `hold` samples and rounds it to `steps` levels either side of zero.
pub fn crush(sound: &mut [f32], hold: usize, steps: f32) {
    let mut held = 0.0;
    for (i, value) in sound.iter_mut().enumerate() {
        if i % hold.max(1) == 0 {
            held = (*value * steps).round() / steps;
        }
        *value = held;
    }
}

/// Plays `sound` ever more slowly, as a tape stopping would, until by its end it runs `by` (0 to
/// 1) slower and so lower in pitch.
pub fn slow(sound: &mut [f32], by: f32) {
    let original = sound.to_vec();
    let last = original.len().saturating_sub(1);
    let mut at = 0.0_f64;
    for (i, value) in sound.iter_mut().enumerate() {
        let (whole, part) = ((at as usize).min(last), at.fract() as f32);
        let next = (whole + 1).min(last);
        *value = original[whole] + (original[next] - original[whole]) * part;
        at += 1.0 - by as f64 * i as f64 / last.max(1) as f64;
    }
}

/// A reverb's tail for `input`, without the input itself: Schroeder's parallel combs, damped, into
/// series all-passes, with Freeverb's delay lengths. `seconds` is how long the tail takes to
/// fall to a thousandth. The two channels' delays differ, which widens it.
pub fn reverb(input: &Track, seconds: f32, damping: f32) -> Track {
    const COMBS: [usize; 8] = [1215, 1293, 1390, 1476, 1548, 1623, 1695, 1760];
    const ALL_PASSES: [usize; 4] = [605, 480, 371, 245];
    const SPREAD: usize = 25;
    let process = |dry: &[f32], spread: usize| {
        let mut wet = vec![0.0; dry.len()];
        for delay in COMBS {
            let delay = delay + spread;
            let feedback = 0.001_f32.powf(time_of(delay) / seconds);
            let (mut line, mut low) = (vec![0.0_f32; delay], 0.0);
            for (i, (value, out)) in dry.iter().zip(&mut wet).enumerate() {
                let echo = line[i % delay];
                low = echo * (1.0 - damping) + low * damping;
                line[i % delay] = value + low * feedback;
                *out += echo / COMBS.len() as f32;
            }
        }
        for delay in ALL_PASSES {
            let delay = delay + spread;
            let mut line = vec![0.0_f32; delay];
            for (i, out) in wet.iter_mut().enumerate() {
                let echo = line[i % delay];
                line[i % delay] = *out + echo * 0.5;
                *out = echo - *out;
            }
        }
        wet
    };
    Track {
        left: process(&input.left, 0),
        right: process(&input.right, SPREAD),
    }
}

/// Holds `track`'s peaks under `ceiling`, looking ahead so the gain is already down when a peak
/// arrives, and releasing gently after it.
pub fn limit(track: &mut Track, ceiling: f32) {
    let ahead = samples(0.005);
    let release = (-1.0 / (0.12 * RATE as f32)).exp();
    let needed: Vec<f32> = track
        .left
        .iter()
        .zip(&track.right)
        .map(|(l, r)| (ceiling / l.abs().max(r.abs()).max(1e-9)).min(1.0))
        .collect();
    // The lowest gain needed in the next `ahead` samples, by a sliding minimum.
    let mut window = std::collections::VecDeque::new();
    let mut next = 0;
    let mut gain = 1.0_f32;
    for i in 0..track.len() {
        while next < track.len() && next <= i + ahead {
            while window.back().is_some_and(|&j| needed[j] >= needed[next]) {
                window.pop_back();
            }
            window.push_back(next);
            next += 1;
        }
        while window.front().is_some_and(|&j| j < i) {
            window.pop_front();
        }
        let target = window.front().map_or(1.0, |&j| needed[j]);
        gain = if target < gain {
            // Reaches the target within the look-ahead.
            (gain - (gain - target) / (ahead as f32 / 4.0)).max(target)
        } else {
            target - (target - gain) * release
        };
        let applied = gain.min(needed[i]);
        track.left[i] *= applied;
        track.right[i] *= applied;
    }
}
