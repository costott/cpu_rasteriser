//! The instruments the music is played on. Most play recordings of single notes, moved to pitch
//! by playing them faster or slower; the rest are synthesised. Each is asked for the same thing,
//! a note, so a part can change instrument without changing what it plays.
//!
//! Recordings live in `assets/silicon_01/audio/notes/<instrument>/<midi note>.wav` (unpitched
//! drums under a nominal note), credited in the `CREDITS.md` beside them.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::sound::dsp::{
    self, Band, RATE, decay, fade, filter, frequency, make, noise, oscillator, samples,
};
use crate::sound::recordings::{self, LoadError};

/// An instrument the score can ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instrument {
    Piano,
    Harp,
    Violins,
    ViolinsShort,
    Cellos,
    CellosShort,
    Basses,
    /// The whole string section plucking.
    Pizzicato,
    Flute,
    Piccolo,
    Horn,
    Trumpet,
    Trombone,
    Tuba,
    /// Voices singing a held vowel.
    Choir,
    Timpani,
    BassDrum,
    Snare,
    Cymbal,
    /// A suspended cymbal swelling up to its crash.
    CymbalSwell,
    /// A pair of cymbals clashed together.
    Clash,
    Gong,
    /// Tubular bells.
    Bells,
    Glockenspiel,
    Marimba,
    Xylophone,
    SteelPan,
    /// Detuned saw waves behind a soft low-pass.
    Pad,
    /// A square wave, as an 8-bit console would play it.
    Chip,
    /// An organ's stops, added together from sine waves.
    Organ,
    /// A bright synthesiser lead whose tone closes as the note sounds.
    SawLead,
    /// A thumb piano's tine.
    Kalimba,
    /// A hand drum tuned two octaves under the note asked for.
    FrameDrum,
    /// Two sines multiplied together: a clangorous, hollow tone.
    Ringed,
    /// A square wave with its low end taken away.
    Thin,
    /// A tone that sinks in pitch as it sounds, like a tape slowing.
    Sagging,
}

impl Instrument {
    /// Every instrument that plays recordings.
    const RECORDED: [Instrument; 26] = [
        Self::Piano,
        Self::Harp,
        Self::Violins,
        Self::ViolinsShort,
        Self::Cellos,
        Self::CellosShort,
        Self::Basses,
        Self::Pizzicato,
        Self::Flute,
        Self::Piccolo,
        Self::Horn,
        Self::Trumpet,
        Self::Trombone,
        Self::Tuba,
        Self::Choir,
        Self::Timpani,
        Self::BassDrum,
        Self::Snare,
        Self::Cymbal,
        Self::CymbalSwell,
        Self::Clash,
        Self::Gong,
        Self::Bells,
        Self::Glockenspiel,
        Self::Marimba,
        Self::Xylophone,
    ];

    /// The folder of its recorded notes, or `None` if it is synthesised.
    fn folder(self) -> Option<&'static str> {
        Some(match self {
            Self::Piano => "piano",
            Self::Harp => "harp",
            Self::Violins => "violins",
            Self::ViolinsShort => "violins-short",
            Self::Cellos => "cellos",
            Self::CellosShort => "cellos-short",
            Self::Basses => "basses",
            Self::Pizzicato => "pizzicato",
            Self::Flute => "flute",
            Self::Piccolo => "piccolo",
            Self::Horn => "horn",
            Self::Trumpet => "trumpet",
            Self::Trombone => "trombone",
            Self::Tuba => "tuba",
            Self::Choir => "choir",
            Self::Timpani => "timpani",
            Self::BassDrum => "bass-drum",
            Self::Snare => "snare",
            Self::Cymbal => "cymbal",
            Self::CymbalSwell => "cymbal-swell",
            Self::Clash => "clash",
            Self::Gong => "gong",
            Self::Bells => "bells",
            Self::Glockenspiel => "glockenspiel",
            Self::Marimba => "marimba",
            Self::Xylophone => "xylophone",
            Self::SteelPan
            | Self::Pad
            | Self::Chip
            | Self::Organ
            | Self::SawLead
            | Self::Kalimba
            | Self::FrameDrum
            | Self::Ringed
            | Self::Thin
            | Self::Sagging => return None,
        })
    }

    /// Whether the instrument can hold a note for as long as it is asked, so its recordings loop.
    fn sustains(self) -> bool {
        matches!(
            self,
            Self::Violins
                | Self::Cellos
                | Self::Basses
                | Self::Flute
                | Self::Piccolo
                | Self::Horn
                | Self::Trumpet
                | Self::Trombone
                | Self::Tuba
                | Self::Choir
        )
    }

    /// How long a note takes to die away once released, in seconds (set by ear, provisional).
    fn release(self) -> f32 {
        match self {
            Self::Piano => 0.35,
            Self::Harp | Self::Glockenspiel => 0.6,
            Self::Violins | Self::Cellos | Self::Basses => 0.45,
            Self::Horn | Self::Trombone | Self::Trumpet => 0.25,
            Self::Flute | Self::Piccolo => 0.2,
            Self::Tuba => 0.3,
            Self::Choir => 0.9,
            Self::Pad => 0.8,
            Self::ViolinsShort | Self::CellosShort => 0.08,
            Self::Chip => 0.01,
            Self::Timpani | Self::BassDrum | Self::Snare => 0.3,
            Self::Cymbal | Self::CymbalSwell | Self::SteelPan => 0.4,
            Self::Clash => 1.0,
            Self::Gong => 2.5,
            Self::Bells => 1.5,
            Self::Marimba | Self::Pizzicato | Self::Kalimba => 0.3,
            Self::Xylophone | Self::FrameDrum => 0.2,
            Self::Organ => 0.3,
            Self::SawLead | Self::Thin | Self::Ringed => 0.12,
            Self::Sagging => 0.4,
        }
    }

    /// How long a note takes to reach full strength, in seconds.
    fn attack(self) -> f32 {
        match self {
            Self::Violins | Self::Cellos | Self::Basses => 0.12,
            Self::Choir => 0.18,
            Self::Tuba | Self::Organ => 0.04,
            Self::Flute | Self::Piccolo => 0.025,
            Self::Pad => 0.6,
            _ => 0.002,
        }
    }
}

/// One recorded note: the MIDI note it sounds, and its samples at [`RATE`].
struct Recorded {
    note: f32,
    sound: Vec<f32>,
}

/// Every instrument, with the recordings of those that have them.
pub struct Orchestra {
    recorded: HashMap<Instrument, Vec<Recorded>>,
}

impl Orchestra {
    /// Loads the recorded notes of every instrument that has them.
    ///
    /// # Errors
    ///
    /// Fails if an instrument's folder is missing or empty, or a file in it can't be read.
    pub fn load() -> Result<Self, LoadError> {
        let mut recorded = HashMap::new();
        for instrument in Instrument::RECORDED {
            let folder = instrument
                .folder()
                .expect("only recorded instruments are listed");
            let notes = recordings::notes(folder)?
                .into_iter()
                .map(|(note, sound)| Recorded {
                    note: note as f32,
                    sound,
                })
                .collect();
            recorded.insert(instrument, notes);
        }
        Ok(Self { recorded })
    }

    /// `instrument` playing the MIDI note `note`, held for `seconds`, then released. Unpitched
    /// drums take their recording's nominal note to sound as recorded.
    pub fn play(&self, instrument: Instrument, note: f32, seconds: f32) -> Vec<f32> {
        let length = seconds + instrument.release();
        let mut sound = match self.recorded.get(&instrument) {
            Some(notes) => {
                let nearest = notes
                    .iter()
                    .min_by(|a, b| (a.note - note).abs().total_cmp(&(b.note - note).abs()))
                    .expect("a loaded instrument has at least one note");
                let speed = ((note - nearest.note) / 12.0).exp2();
                if instrument.sustains() {
                    let held = looped(&nearest.sound, samples(length * speed) + 2);
                    resample(&held, speed, length)
                } else {
                    resample(&nearest.sound, speed, length)
                }
            }
            None => synthesise(instrument, note, length),
        };
        let end = dsp::time_of(sound.len());
        fade(
            &mut sound,
            instrument.attack().min(end * 0.5),
            instrument.release().min(end),
        );
        sound
    }
}

/// `sound` made at least `length` samples long by going round its later part again, each time
/// crossfading into it.
fn looped(sound: &[f32], length: usize) -> Cow<'_, [f32]> {
    // Clear of the recording's fade at its end.
    let end = sound.len().saturating_sub(samples(0.15));
    let from = end * 2 / 5;
    // A short recording has a shorter crossfade, so that it still goes round.
    let cross = samples(0.3).min((end - from) / 3);
    if sound.len() >= length || cross == 0 {
        return Cow::Borrowed(sound);
    }
    let mut held = sound[..end].to_vec();
    while held.len() < length {
        let tail = held.len() - cross;
        for i in 0..cross {
            let mix = i as f32 / cross as f32;
            held[tail + i] = held[tail + i] * (1.0 - mix) + sound[from + i] * mix;
        }
        held.extend_from_slice(&sound[from + cross..end]);
    }
    Cow::Owned(held)
}

/// Up to `seconds` of `sound` played `speed` times as fast, interpolating linearly.
pub fn resample(sound: &[f32], speed: f32, seconds: f32) -> Vec<f32> {
    let available = (sound.len().saturating_sub(1) as f32 / speed) as usize;
    (0..samples(seconds).min(available))
        .map(|i| {
            let at = i as f64 * speed as f64;
            let (whole, part) = (at as usize, at.fract() as f32);
            sound[whole] + (sound[whole + 1] - sound[whole]) * part
        })
        .collect()
}

fn synthesise(instrument: Instrument, note: f32, seconds: f32) -> Vec<f32> {
    let hz = frequency(note);
    match instrument {
        // A pan's note, its octave and a slightly sharp twelfth, with a soft mallet's click.
        Instrument::SteelPan => {
            let mut sound = make(seconds, |_| 0.0);
            for (ratio, level, ring) in [(1.0, 0.6, 1.1), (2.0, 0.3, 0.6), (3.01, 0.15, 0.3)] {
                let mut partial = oscillator(
                    seconds,
                    |t| hz * ratio * (1.0 + 0.012 * (-t * 40.0).exp()),
                    dsp::sine,
                );
                decay(&mut partial, ring);
                sound
                    .iter_mut()
                    .zip(partial)
                    .for_each(|(a, b)| *a += b * level);
            }
            let mut click = noise(0.015, 11);
            filter(&mut click, Band::Peak, 1.0, |_| hz * 6.0);
            decay(&mut click, 0.012);
            sound.iter_mut().zip(click).for_each(|(a, b)| *a += b * 0.2);
            sound
        }
        Instrument::Pad => {
            let mut sound = make(seconds, |_| 0.0);
            for (index, cents) in [-11.0_f32, -4.0, 3.0, 9.0].into_iter().enumerate() {
                let detuned = hz * (cents / 1200.0).exp2();
                let voice = oscillator(seconds, |_| detuned, dsp::saw);
                let offset = index * 137;
                sound
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, a)| *a += voice.get(i + offset).copied().unwrap_or(0.0) * 0.16);
            }
            filter(&mut sound, Band::Low, 0.8, |t| {
                500.0 + 700.0 * (t * 0.7).sin().abs()
            });
            sound
        }
        Instrument::Chip => {
            let mut sound = oscillator(seconds, |_| hz, dsp::square);
            sound.iter_mut().for_each(|v| *v *= 0.25);
            dsp::crush(&mut sound, (RATE / 22_050) as usize, 16.0);
            // A bare square wave's upper harmonics are harsh: they are rolled off, and it sits
            // 4 dB down (set by ear and from the measured treble, provisional).
            filter(&mut sound, Band::Low, 0.7, |_| 6000.0);
            sound.iter_mut().for_each(|v| *v *= 0.63);
            sound
        }
        Instrument::Organ => partials(
            seconds,
            hz,
            &[
                (1.0, 0.2, 0.0),
                (2.0, 0.12, 0.0),
                (3.0, 0.06, 0.0),
                (4.0, 0.07, 0.0),
                (6.0, 0.03, 0.0),
                (8.0, 0.02, 0.0),
                (1.004, 0.08, 0.0),
            ],
        ),
        Instrument::SawLead => {
            let mut sound = oscillator(seconds, |_| hz * 0.996, dsp::saw);
            let other = oscillator(seconds, |_| hz * 1.004, dsp::saw);
            sound
                .iter_mut()
                .zip(other)
                .for_each(|(a, b)| *a = (*a + b) * 0.17);
            filter(&mut sound, Band::Low, 1.6, |t| {
                1200.0 + 4000.0 * (-t * 5.0).exp()
            });
            sound
        }
        // A tine's note and its high, quickly dying overtone.
        Instrument::Kalimba => partials(seconds, hz, &[(1.0, 0.5, 0.9), (5.9, 0.18, 0.14)]),
        Instrument::FrameDrum => {
            let mut sound = oscillator(
                seconds,
                |t| hz * 0.25 * (1.0 + 0.35 * (-t * 35.0).exp()),
                dsp::sine,
            );
            decay(&mut sound, 0.45);
            let mut skin = noise(seconds.min(0.12), 29);
            filter(&mut skin, Band::Low, 0.8, |_| 1400.0);
            decay(&mut skin, 0.09);
            sound.iter_mut().for_each(|v| *v *= 0.8);
            sound.iter_mut().zip(skin).for_each(|(a, b)| *a += b * 0.5);
            sound
        }
        Instrument::Ringed => {
            let mut sound = oscillator(seconds, |_| hz, dsp::sine);
            let other = oscillator(seconds, |_| hz * 1.414, dsp::sine);
            sound
                .iter_mut()
                .zip(other)
                .for_each(|(a, b)| *a *= b * 0.55);
            decay(&mut sound, 1.2);
            sound
        }
        Instrument::Thin => {
            let mut sound = oscillator(seconds, |_| hz, dsp::square);
            filter(&mut sound, Band::High, 0.8, |_| hz * 3.0);
            sound.iter_mut().for_each(|v| *v *= 0.2);
            sound
        }
        Instrument::Sagging => {
            // Three semitones down over the note.
            let bend = move |t: f32| hz * (-0.25 * (t / seconds).min(1.0)).exp2();
            let mut sound = oscillator(seconds, bend, dsp::saw);
            let other = oscillator(seconds, |t| bend(t) * 1.007, dsp::saw);
            sound
                .iter_mut()
                .zip(other)
                .for_each(|(a, b)| *a = (*a + b) * 0.22);
            filter(&mut sound, Band::Low, 0.9, |_| 1300.0);
            sound
        }
        _ => unreachable!("every other instrument plays recordings"),
    }
}

/// A note built from sine waves: each `(ratio to the note, level, seconds to ring)`, ringing for
/// as long as the note when its seconds are 0.
fn partials(seconds: f32, hz: f32, of: &[(f32, f32, f32)]) -> Vec<f32> {
    let mut sound = make(seconds, |_| 0.0);
    for &(ratio, level, ring) in of {
        let mut partial = oscillator(seconds, |_| hz * ratio, dsp::sine);
        if ring > 0.0 {
            decay(&mut partial, ring);
        }
        sound
            .iter_mut()
            .zip(partial)
            .for_each(|(a, b)| *a += b * level);
    }
    sound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_looped_recording_lasts_as_long_as_asked_without_a_jump() {
        let recording = dsp::oscillator(1.0, |_| 110.0, dsp::sine);
        let held = looped(&recording, samples(5.0));
        assert!(held.len() >= samples(5.0));
        // A 110 Hz sine moves at most this far between samples.
        let step = std::f32::consts::TAU * 110.0 / RATE as f32;
        assert!(
            held.windows(2)
                .all(|pair| (pair[1] - pair[0]).abs() < 2.0 * step)
        );
    }
}
