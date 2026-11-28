//! The sound effects, each timed by the same functions that time the picture: the tiles, the
//! chip's turns and gleams, its light streams, the captions, the teapot's triangles, the
//! projection's flicker, the eruption, the swells between scenes, each world's own sounds and the
//! final collapse. Levels are set by ear (provisional).

use std::f32::consts::PI;

use cpu_rasteriser::prelude::*;

use crate::director::{self, CAPTIONS, Phase, ramp};
use crate::effects::teapot::Triangle;
use crate::effects::{eruption, frustum, hologram};
use crate::shaders::composite;
use crate::sound::Buses;
use crate::sound::accents;
use crate::sound::dsp::{
    self, Band, decay, fade, filter, frequency, make, noise, oscillator, samples,
};
use crate::sound::recordings::Effects;
use crate::sound::world::{Ambience, Source, WorldSound};
use crate::themes::Schedule;
use crate::themes::kit::random::Rng;

/// The window the tiles' columns and the chip's place on screen are worked out for: the video's
/// shape. Larger 16:9 windows have the same tile grid.
const SCREEN: (f32, f32) = (1280.0, 720.0);

/// The D minor pentatonic scale's steps above D.
const PENTATONIC: [u8; 5] = [0, 3, 5, 7, 10];

/// How long before its world a background sound starts, and how long it runs on after, in
/// seconds, so that neighbouring worlds' sounds overlap.
const BED_LEAD: f32 = 0.15;
const BED_TAIL: f32 = 0.35;

/// The `index`th note of the pentatonic scale counting up from `base`, a D.
fn pentatonic(base: u8, index: usize) -> f32 {
    (base + PENTATONIC[index % 5] + 12 * (index / 5) as u8) as f32
}

/// A short tuned blip: a sine with a little of its third harmonic.
fn blip(note: f32, seconds: f32) -> Vec<f32> {
    let hz = frequency(note);
    let mut sound = make(seconds, |t| {
        let phase = t * hz;
        dsp::sine(phase.fract()) + 0.25 * dsp::sine((phase * 3.0).fract())
    });
    decay(&mut sound, seconds);
    fade(&mut sound, 0.002, 0.004);
    sound
}

/// A small soft bell, ringing for about `ring` seconds: a sine and a faint, quickly dying bell
/// overtone. `rng` varies its tuning, ring and level, so no two are quite alike.
fn twinkle(note: f32, ring: f32, rng: &mut Rng) -> Vec<f32> {
    let hz = frequency(note + rng.range(-0.05, 0.05));
    let ring = ring * rng.range(0.75, 1.3);
    let mut sound = make(ring, |t| dsp::sine((t * hz).fract()));
    let mut overtone = make(ring, |t| 0.16 * dsp::sine((t * hz * 2.76).fract()));
    decay(&mut overtone, ring * 0.35);
    sound.iter_mut().zip(overtone).for_each(|(a, b)| *a += b);
    decay(&mut sound, ring);
    fade(&mut sound, 0.008, 0.02);
    let level = rng.range(0.75, 1.15);
    sound.iter_mut().for_each(|v| *v *= level);
    sound
}

/// Filtered noise `seconds` long that dies away.
fn burst(seconds: f32, band: Band, hz: f32, seed: u64) -> Vec<f32> {
    let mut sound = noise(seconds, seed);
    filter(&mut sound, band, 1.2, |_| hz);
    decay(&mut sound, seconds);
    fade(&mut sound, 0.001, 0.003);
    sound
}

/// A stretch of a recording, from `from` seconds into it, faded at both ends and scaled so its
/// loudest sample is 1: the recordings' own levels differ widely.
fn excerpt(recording: &[f32], from: f32, seconds: f32, attack: f32, release: f32) -> Vec<f32> {
    let first = samples(from).min(recording.len());
    let last = (first + samples(seconds)).min(recording.len());
    let mut sound = recording[first..last].to_vec();
    let peak = sound.iter().fold(0.0_f32, |peak, v| peak.max(v.abs()));
    sound.iter_mut().for_each(|v| *v /= peak.max(1e-3));
    fade(&mut sound, attack, release);
    sound
}

/// A soft twinkle for each tile as its worker thread takes it, each of the sixteen threads on its
/// own note, with now and then a higher sparkle just after; and fainter ones as they switch off
/// at the end.
pub fn tiles(buses: &mut Buses) {
    let columns = (SCREEN.0 / composite::TILE).ceil() as usize;
    let count = columns * (SCREEN.1 / composite::TILE).ceil() as usize;
    let pan = |index: usize| ((index % columns) as f32 / (columns - 1) as f32) * 1.4 - 0.7;
    let mut rng = Rng::new(409);
    for index in 0..count {
        let thread = index % composite::THREADS;
        let at = director::POWER_ON_DELAY + composite::power_on_start_of(index);
        let note = pentatonic(74, thread);
        buses.effect(at, &twinkle(note, 0.3, &mut rng), 0.034, pan(index), 0.45);
        if index % 9 == 4 {
            let sparkle = twinkle(note + 12.0, 0.24, &mut rng);
            buses.effect(at + 0.05, &sparkle, 0.012, -pan(index), 0.65);
        }
    }
    let camera = crate::camera_at(director::SHUTDOWN_TILES, SCREEN.0 / SCREEN.1);
    let clip = camera.projection_matrix()
        * camera.view_matrix()
        * (director::chip_matrix(director::SHUTDOWN_TILES) * Vec4::new(0.0, 0.0, 0.0, 1.0));
    let centre = Vec2::new(clip.x / clip.w * 0.5 + 0.5, clip.y / clip.w * 0.5 + 0.5);
    for index in 0..count {
        let at = director::SHUTDOWN_TILES + composite::shutdown_dot_of(index, SCREEN, centre);
        let note = pentatonic(86, index % composite::THREADS);
        buses.effect(at, &twinkle(note, 0.13, &mut rng), 0.011, pan(index), 0.65);
    }
}

/// Whooshes that follow the chip's turning, low and soft, louder and a little higher the faster
/// it turns, and in the build-up's spin one pulse for each turn, with a brighter rush of air
/// layered over that spin and its winding down.
pub fn whooshes(buses: &mut Buses) {
    const CONTROL: f32 = 200.0;
    /// What the whooshes follow at each control step.
    #[derive(Clone, Copy)]
    struct Control {
        low: f32,
        low_cutoff: f32,
        bright: f32,
        bright_cutoff: f32,
        body_hz: f32,
        pan: f32,
    }
    let end = Phase::Shutdown.start();
    let count = (end * CONTROL) as usize + 1;
    let spin_start = Phase::BuildUp.start() + 1.0;
    let spin_end = Phase::Eruption.start() + director::COAST;
    let controls: Vec<Control> = (0..count)
        .map(|i| {
            let time = i as f32 / CONTROL;
            let rate = director::turn_rate(time);
            // Below 0.2 radians a second is only drift.
            let moving = ((rate - 0.2) / 1.3).clamp(0.0, 1.0);
            let spinning = ((rate - 4.0) / 25.0).clamp(0.0, 1.0);
            let facing = (director::chip_matrix(time) * Vec4::new(1.0, 0.0, 0.0, 0.0)).xyz();
            // The pulse: strongest as the chip's side sweeps towards the camera.
            let pulse = 1.0 - ((rate - 2.0) / 4.0).clamp(0.0, 1.0) * (0.5 - 0.5 * facing.z);
            // Frames after the eruption belong to the worlds: only the build-up's spin is loud.
            let quiet = if Phase::at(time) == Phase::Worlds {
                0.25
            } else {
                1.0
            };
            let level = (moving * moving * 0.55 + spinning * 0.6) * pulse * quiet;
            let layered = if time < spin_end {
                ramp(time, spin_start, spin_start + 2.0)
            } else {
                0.0
            };
            Control {
                low: level,
                low_cutoff: 90.0 + 300.0 * moving + 60.0 * spinning,
                bright: level * layered,
                bright_cutoff: 220.0 + 650.0 * moving + 1600.0 * spinning,
                body_hz: 48.0 + 30.0 * moving + 22.0 * spinning,
                pan: facing.x.clamp(-1.0, 1.0) * facing.z.signum() * 0.3,
            }
        })
        .collect();
    let at = |t: f32| controls[((t * CONTROL) as usize).min(count - 1)];

    // Twice through a two-pole low-pass: 24 dB an octave, which leaves no hiss.
    let mut low = noise(end, 71);
    for _ in 0..2 {
        filter(&mut low, Band::Low, 0.8, |t| at(t).low_cutoff);
    }
    let body = oscillator(end, |t| at(t).body_hz, dsp::sine);
    let mut bright = noise(end, 72);
    filter(&mut bright, Band::Peak, 1.4, |t| at(t).bright_cutoff);
    let (mut low_level, mut bright_level) = (0.0, 0.0);
    let mixed: Vec<f32> = low
        .into_iter()
        .zip(body)
        .zip(bright)
        .enumerate()
        .map(|(i, ((low, body), bright))| {
            let control = at(dsp::time_of(i));
            // Smoothed over a few milliseconds, so the control steps don't zip.
            low_level += (control.low - low_level) * 0.004;
            bright_level += (control.bright - bright_level) * 0.004;
            (low * 3.2 + body * 0.2) * low_level + bright * bright_level
        })
        .collect();
    // Placed in half-second pieces so the pan can follow the chip.
    for (i, piece) in mixed.chunks(samples(0.5)).enumerate() {
        let start = i as f32 * 0.5;
        buses.effect(start, piece, 0.9, at(start + 0.25).pan, 0.15);
    }
}

/// A faint airy shimmer as each gleam crosses the chip, moving across with it.
pub fn gleams(buses: &mut Buses) {
    for (i, start) in director::GLEAMS.into_iter().enumerate() {
        let sound = accents::shimmer(director::GLEAM_LENGTH, 211 + i as u64);
        // In short pieces that cross from left to right as the band does.
        let piece = samples(0.05);
        for (k, part) in sound.chunks(piece).enumerate() {
            let offset = dsp::time_of(k * piece);
            let across = offset / director::GLEAM_LENGTH;
            buses.effect(start + offset, part, 0.022, -0.6 + 1.2 * across, 0.5);
        }
    }
}

/// The light streams running across the die: a very quiet scatter of high glints, as strong as
/// the chip is powered, up to the worlds.
pub fn streams(buses: &mut Buses) {
    let mut rng = Rng::new(977);
    let mut time = 0.0;
    while time < Phase::Worlds.start() {
        let activity = director::frame(time).activity;
        if activity > 0.0 {
            let glint = twinkle(pentatonic(98, rng.index(10)), 0.16, &mut rng);
            buses.effect(time, &glint, 0.0035 * activity, rng.range(-0.8, 0.8), 0.7);
        }
        time += rng.range(0.06, 0.2);
    }
}

/// The tremor under the shudder, the build-up and the blast, as strong as the shake.
pub fn rumble(buses: &mut Buses) {
    let (start, end) = (
        Phase::Hologram.start(),
        Phase::Eruption.start() + director::COAST,
    );
    let mut sound = noise(end - start, 5);
    filter(&mut sound, Band::Low, 0.9, |_| 90.0);
    for (i, value) in sound.iter_mut().enumerate() {
        *value *= director::shake(start + dsp::time_of(i));
    }
    fade(&mut sound, 0.01, 0.2);
    buses.effect(start, &sound, 1.6, 0.0, 0.1);
}

/// A pen scratching while each caption draws itself in, and a snip as it cuts away.
pub fn captions(buses: &mut Buses, effects: &Effects) {
    let pen = effects.get("pen");
    for (i, caption) in CAPTIONS.iter().enumerate() {
        let drawing = caption.drawn() - caption.start;
        let sound = excerpt(pen, (i as f32 * 1.7) % 6.0, drawing, 0.02, 0.1);
        buses.effect(caption.start, &sound, 0.12, 0.0, 0.1);
        let snip = burst(0.035, Band::High, 3500.0, i as u64);
        buses.effect(caption.cuts_away(), &snip, 0.07, 0.0, 0.2);
    }
}

/// The teapot's triangles: a soft rush as the die first lets them go, a faint upward tick as each
/// leaves it, a glass tick as it lands, higher the higher it lands, and a softer one, backwards,
/// as it leaves again.
pub fn teapot(buses: &mut Buses, triangles: &[Triangle]) {
    release(buses);
    for (i, triangle) in triangles.iter().enumerate() {
        let note = pentatonic(86, (triangle.height * 11.0) as usize);
        let from = frequency(note);
        let mut launch = oscillator(0.03, |t| from * (1.0 + 16.0 * t), dsp::sine);
        decay(&mut launch, 0.035);
        fade(&mut launch, 0.002, 0.004);
        buses.effect(triangle.launches, &launch, 0.005, triangle.side * 0.5, 0.45);

        let hz = frequency(note + (i % 7) as f32 * 0.03);
        let mut tick = make(0.07, |t| {
            dsp::sine((t * hz).fract()) + 0.5 * dsp::sine((t * hz * 2.76).fract())
        });
        decay(&mut tick, 0.06);
        fade(&mut tick, 0.001, 0.004);
        buses.effect(triangle.lands, &tick, 0.024, triangle.side * 0.8, 0.35);
        tick.reverse();
        buses.effect(triangle.leaves, &tick, 0.012, triangle.side * 0.8, 0.35);
    }
}

/// The die letting its triangles go as the beams appear: a soft low thump, and a rush that rises
/// and thins away.
fn release(buses: &mut Buses) {
    let start = Phase::Teapot.start() + 0.2;
    let length = 1.5;
    let mut rush = noise(length, 223);
    filter(&mut rush, Band::Peak, 1.8, |t| {
        280.0 * 12.0_f32.powf(t / length)
    });
    for (i, value) in rush.iter_mut().enumerate() {
        let along = dsp::time_of(i) / length;
        *value *= (along / 0.22).min(1.0) * (1.0 - along).powi(2);
    }
    buses.effect(start, &rush, 0.2, 0.0, 0.4);
    let mut thump = oscillator(0.6, |t| 45.0 + 55.0 * (-t * 9.0).exp(), dsp::sine);
    decay(&mut thump, 0.55);
    fade(&mut thump, 0.02, 0.05);
    buses.effect(start, &thump, 0.2, 0.0, 0.2);
}

/// The projection: a charge as its rays rise, an electrical buzz that jolts, warps and fizzes on
/// the very frames the picture glitches, a sweep following the tear that rolls up it, and a warble
/// while the fractal is formed.
pub fn projection(buses: &mut Buses) {
    let (start, end) = (Phase::Hologram.start(), Phase::BuildUp.start() + 0.5);
    let length = end - start;

    let mut charge = oscillator(
        length,
        |t| 180.0 + 900.0 * frustum::rise(start + t.min(1.6)),
        dsp::saw,
    );
    filter(&mut charge, Band::Low, 1.5, |t| {
        400.0 + 2600.0 * frustum::rise(start + t.min(1.6))
    });
    for (i, value) in charge.iter_mut().enumerate() {
        let t = dsp::time_of(i);
        let rise = frustum::rise(start + t);
        *value *= 4.0 * rise * (1.0 - rise) + 0.15 * rise * (1.0 - ramp(t, 1.2, 2.5));
    }
    buses.effect(start, &charge, 0.1, 0.0, 0.4);

    buzz(buses, start, length);

    let mut tear = noise(length, 227);
    filter(&mut tear, Band::Peak, 4.0, |t| {
        600.0 + 3000.0 * frustum::tear(start + t).clamp(0.0, 1.0)
    });
    let mut warble = oscillator(
        length,
        |t| 440.0 + 90.0 * (t * 2.3).sin() + 40.0 * (t * 7.1).sin(),
        dsp::sine,
    );
    for (i, (tear, warble)) in tear.iter_mut().zip(&mut warble).enumerate() {
        let time = start + dsp::time_of(i);
        let flicker = frustum::flicker_level(time);
        // Heard only while the tear is on the projection, loudest half-way up it.
        let height = frustum::tear(time).clamp(0.0, 1.0);
        *tear *= frustum::volume(time) * (PI * height).sin().max(0.0);
        *warble *= hologram::formed(time) * (0.75 + 0.25 * flicker);
    }
    buses.effect(start, &tear, 0.09, -0.2, 0.35);
    buses.effect(start, &warble, 0.035, 0.15, 0.6);

    fizz(buses, start, length);
}

/// An electrical hum over the projection, as strong as it is, rich in harmonics from the mains'
/// hundred hertz up. On each frame the picture dips it jolts: its pitch leaps away and warps
/// back.
fn buzz(buses: &mut Buses, start: f32, length: f32) {
    // How fast a leap in pitch slides back, as a time constant in seconds.
    const SETTLE: f32 = 0.04;
    let slide = (-1.0 / (SETTLE * dsp::RATE as f32)).exp();
    let mut rng = Rng::new(229);
    // How far from its own pitch the hum is, in octaves, and its two voices' phases.
    let (mut bend, mut spikes, mut rough) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut dipping = false;
    let mut sound = make(length, |t| {
        let time = start + t;
        let dips = frustum::dips(time);
        if dips && !dipping {
            let leap = rng.range(0.7, 1.4);
            bend = if rng.unit() < 0.6 { leap } else { -leap * 0.6 };
        }
        dipping = dips;
        bend *= slide;
        // The slide wobbles as it goes.
        let warped = bend * (1.0 + 0.35 * (t * 2.0 * PI * 31.0).sin());
        let hz = 100.0 * warped.exp2();
        spikes = (spikes + hz / dsp::RATE as f32).fract();
        rough = (rough + hz * 1.006 / dsp::RATE as f32).fract();
        let pulse = if spikes < 0.16 { 1.0 } else { -0.19 };
        let tremble = 0.8 + 0.2 * (t * 2.0 * PI * 50.0).sin();
        (pulse * 0.6 + dsp::saw(rough) * 0.4) * tremble * frustum::volume(time)
    });
    filter(&mut sound, Band::High, 0.7, |_| 150.0);
    filter(&mut sound, Band::Low, 0.8, |_| 2900.0);
    buses.effect(start, &sound, 0.32, 0.0, 0.2);
}

/// The static that comes with each glitch: a grainy fizz that starts at once on the frame the
/// picture dips and dies away just after it.
fn fizz(buses: &mut Buses, start: f32, length: f32) {
    /// How many samples each grain lasts.
    const GRAIN: usize = 56;
    let mut rng = Rng::new(239);
    let mut grains = noise(length, 241);
    let (mut strength, mut grain) = (0.0_f32, 0.0_f32);
    for (i, value) in grains.iter_mut().enumerate() {
        let time = start + dsp::time_of(i);
        let target = if frustum::dips(time) {
            frustum::volume(time)
        } else {
            0.0
        };
        // Up within a millisecond, away over a dozen.
        let ease = if target > strength { 0.05 } else { 0.0017 };
        strength += (target - strength) * ease;
        // Most grains are faint and a few stand out, which is what makes it crackle.
        if i % GRAIN == 0 {
            grain = rng.unit().powi(3);
        }
        *value *= strength * grain;
    }
    filter(&mut grains, Band::High, 0.7, |_| 1500.0);
    filter(&mut grains, Band::Low, 0.7, |_| 6000.0);
    buses.effect(start, &grains, 0.2, 0.15, 0.2);
}

/// The burst: a recorded explosion over a falling sub bass, debris, and blips as the primitives
/// settle into their grid.
pub fn eruption(buses: &mut Buses, effects: &Effects) {
    let start = Phase::Eruption.start();
    for (name, gain) in [("explosion", 0.55), ("explosion-short", 0.35)] {
        let recording = effects.get(name);
        let sound = excerpt(recording, 0.0, 2.7, 0.002, 0.4);
        buses.effect(start, &sound, gain, 0.0, 0.25);
    }
    buses.effect(start, &sub_drop(2.2), 0.7, 0.0, 0.0);
    let mut rng = Rng::new(64);
    for i in 0..90 {
        let after = rng.unit().powi(2) * 1.4;
        let piece = burst(
            rng.range(0.02, 0.09),
            Band::Peak,
            rng.range(500.0, 5000.0),
            i,
        );
        let gain = 0.2 * (1.0 - after / 1.5);
        buses.effect(start + after, &piece, gain, rng.range(-0.9, 0.9), 0.35);
    }
    let settled = start + eruption::SETTLE + eruption::SETTLE_TIME;
    for _ in 0..110 {
        let at = settled - 0.25 + rng.unit() * (eruption::SETTLE_STAGGER + 0.25);
        let sound = blip(pentatonic(86, rng.index(15)), 0.05);
        buses.effect(at, &sound, 0.035, rng.range(-0.9, 0.9), 0.35);
    }
}

/// A sine falling from 90 to 30 hertz as it dies away.
fn sub_drop(seconds: f32) -> Vec<f32> {
    let mut sound = oscillator(seconds, |t| 30.0 + 60.0 * (-t * 3.0).exp(), dsp::sine);
    decay(&mut sound, seconds * 1.4);
    fade(&mut sound, 0.003, 0.2);
    sound
}

/// Swells of air that lead from one scene into the next: into each of the studio's later scenes,
/// and, shorter, out of each of the first `worlds` cuts that stays long enough to be led out of.
pub fn swells(buses: &mut Buses, schedule: &Schedule, worlds: usize) {
    let scenes = [Phase::Rise, Phase::Teapot, Phase::Hologram, Phase::BuildUp];
    for (i, phase) in scenes.into_iter().enumerate() {
        let sound = swell(1.2, 233 + i as u64);
        buses.effect(phase.start() - 1.2, &sound, 0.07, 0.0, 0.5);
    }
    for cut in 0..worlds.min(schedule.cut_count()) {
        let slot = schedule.slot(cut);
        if slot.end - slot.start >= 0.7 {
            let sound = swell(0.4, 251 + cut as u64);
            let pan = if cut % 2 == 0 { -0.25 } else { 0.25 };
            buses.effect(slot.end - 0.4, &sound, 0.06, pan, 0.4);
        }
    }
}

/// Bright noise that gathers and rises for `seconds`, then lets go at once.
fn swell(seconds: f32, seed: u64) -> Vec<f32> {
    let mut sound = noise(seconds, seed);
    filter(&mut sound, Band::Peak, 1.2, |t| {
        700.0 + 4300.0 * (t / seconds).powi(2)
    });
    for (i, value) in sound.iter_mut().enumerate() {
        *value *= (dsp::time_of(i) / seconds).powi(3);
    }
    fade(&mut sound, 0.01, 0.03);
    sound
}

/// Each world's own sounds.
pub fn worlds(buses: &mut Buses, effects: &Effects, schedule: &Schedule, worlds: &[WorldSound]) {
    // The first run through the worlds; the finale's flashes are too short for sounds.
    for index in 0..worlds.len().min(schedule.cut_count()) {
        let slot = schedule.slot(index);
        for ambience in &worlds[schedule.theme(index)].ambience {
            let (at, seconds, attack, release) = match ambience.lasting {
                None => {
                    let at = slot.start - BED_LEAD;
                    (at, slot.end + BED_TAIL - at, BED_LEAD, BED_TAIL)
                }
                Some(seconds) if ambience.at < slot.end - slot.start => {
                    (slot.start + ambience.at, seconds, 0.004, 0.06)
                }
                Some(_) => continue,
            };
            let mut sound = ambient(effects, ambience, seconds);
            fade(&mut sound, attack, release);
            buses.effect(at, &sound, ambience.gain, ambience.pan, 0.2);
        }
    }
}

/// `seconds` of what `ambience` sounds like, at full level.
fn ambient(effects: &Effects, ambience: &Ambience, seconds: f32) -> Vec<f32> {
    match ambience.kind {
        Source::Recording(name, from) => excerpt(effects.get(name), from, seconds, 0.0, 0.01),
        Source::Synth(accent) => accent.sound(seconds),
    }
}

/// The screen's collapse at the very end: a soft thunk and a falling whine.
pub fn collapse(buses: &mut Buses) {
    let (start, end) = director::COLLAPSE;
    let mut thunk = oscillator(0.35, |t| 46.0 + 40.0 * (-t * 30.0).exp(), dsp::sine);
    decay(&mut thunk, 0.3);
    fade(&mut thunk, 0.002, 0.03);
    buses.effect(start, &thunk, 0.35, 0.0, 0.2);
    let length = end - start;
    let mut whine = oscillator(
        length,
        |t| 90.0 + 5200.0 * (1.0 - t / length).powi(3),
        dsp::sine,
    );
    decay(&mut whine, length * 1.2);
    fade(&mut whine, 0.01, 0.15);
    buses.effect(start, &whine, 0.045, 0.0, 0.35);
}
