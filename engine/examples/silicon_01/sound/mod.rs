//! The film's soundtrack: the music and the effects, mixed into one stereo loop.
//!
//! Nothing is played live. [`write`] renders the whole loop to a WAV file, which `media.sh`
//! puts on the video. Everything is timed by the functions that time the picture, so retiming
//! the film retimes its sound, and equal inputs give identical files.
//!
//! The sound is mixed in two parts: everything up to the climax, which is cut off dead just
//! after it, and what follows, which is left to ring, so the film turns suddenly soft. The
//! climax's low chord goes with the second, so it alone carries on, dying away under the piano.

use std::ops::Range;
use std::path::Path;

use crate::director::{FINALE_BLUR, LOOP_LENGTH, Phase, ramp};
use crate::effects::{frustum, teapot};
use crate::sound::dsp::{Band, RATE, Track, filter, limit, reverb};
use crate::sound::instrument::{Instrument, Orchestra};
use crate::sound::recordings::Effects;
use crate::sound::score::{BREATH, Shape, score};
use crate::sound::world::{Treatment, WorldSound};
use crate::themes::{Schedule, Theme};

pub mod accents;
pub mod dsp;
mod foley;
pub mod instrument;
mod recordings;
mod score;
pub mod world;

/// How long after the climax the loud part carries on, and how long it then takes to vanish, in
/// seconds (set by ear, provisional).
const CUT_HOLD: f32 = 0.14;
const CUT_TAIL: f32 = 0.7;
/// The whole mix's level, set so the loop measures about -16 LUFS, where most of the reference
/// films sit, with ffmpeg's `ebur128` (measured; provisional until the mix is final).
const MASTER_GAIN: f32 = 0.715;
/// How far the film's quiet stretches, before the build-up and after the climax, are lifted
/// above the loud part's level, in decibels (set from the measured loudness, provisional).
const QUIET_LIFT: f32 = 1.5;
/// The most a sample may reach: -1.5 dBFS, leaving the true peak under -1 dBTP.
const CEILING: f32 = 0.84;

/// Sound on its way into the mix: what is heard directly, and what is sent to the reverb.
struct Stem {
    dry: Track,
    send: Track,
}

impl Stem {
    fn new() -> Self {
        Self {
            dry: Track::silent(LOOP_LENGTH),
            send: Track::silent(LOOP_LENGTH),
        }
    }

    fn place(&mut self, at: f32, sound: &[f32], gain: f32, pan: f32, send: f32) {
        self.dry.place(at, sound, gain, pan);
        self.send.place(at, sound, gain * send, pan);
    }

    /// The dry sound with its reverb, whose tail lasts `seconds`.
    fn mixed(mut self, seconds: f32) -> Track {
        let tail = reverb(&self.send, seconds, 0.35);
        self.dry.mix(&tail, 0.5);
        self.dry
    }
}

/// The two parts of the mix. Sound goes to the one its start belongs to.
struct Buses {
    act: Stem,
    coda: Stem,
}

impl Buses {
    fn part(&mut self, at: f32) -> &mut Stem {
        if at < Phase::Shutdown.start() + 0.5 {
            &mut self.act
        } else {
            &mut self.coda
        }
    }

    /// Adds an effect starting at `at` seconds, with `send` of it to the reverb.
    fn effect(&mut self, at: f32, sound: &[f32], gain: f32, pan: f32, send: f32) {
        self.part(at).place(at, sound, gain, pan, send);
    }
}

/// How long a treatment of the music takes to come and go, in seconds.
const TREATMENT_FADE: f32 = 0.15;

/// How much of an instrument goes to the reverb.
fn room(instrument: Instrument) -> f32 {
    use Instrument::*;
    match instrument {
        Piano => 0.3,
        Harp | Glockenspiel | SteelPan | Marimba | Xylophone | Kalimba => 0.4,
        Violins | Cellos | Horn | Trombone | Trumpet | Flute | Piccolo | Pad => 0.4,
        Choir | Bells | Gong | Organ => 0.5,
        ViolinsShort | CellosShort | Basses | Pizzicato | Tuba => 0.25,
        Timpani | BassDrum | Snare | Cymbal | CymbalSwell | Clash | FrameDrum => 0.2,
        SawLead | Ringed | Thin | Sagging => 0.25,
        Chip => 0.1,
    }
}

/// The whole loop's soundtrack, for a film with these `themes`.
fn render(themes: &[Theme]) -> Result<Track, Box<dyn std::error::Error>> {
    let orchestra = Orchestra::load()?;
    let effects = Effects::load()?;
    let triangles = teapot::triangles()?;
    let schedule = Schedule::new(themes.len());
    let worlds: Vec<WorldSound> = themes.iter().map(|theme| theme.stage.sound()).collect();
    let leads: Vec<&[Instrument]> = worlds.iter().map(|world| world.leads).collect();

    let mut music = Buses {
        act: Stem::new(),
        coda: Stem::new(),
    };
    let hologram = Phase::Hologram.start()..Phase::BuildUp.start();
    for note in score(&schedule, &leads) {
        let mut sound = orchestra.play(note.instrument, note.pitch, note.length);
        match note.shape {
            Shape::Plain => {}
            Shape::Held => die_away(&mut sound),
            Shape::Swelling => {
                for (i, value) in sound.iter_mut().enumerate() {
                    *value *= swell(note.at + dsp::time_of(i));
                }
            }
        }
        // The pads flicker with the projection while the hologram is up.
        if note.instrument == Instrument::Pad && hologram.contains(&note.at) {
            dsp::gate(&mut sound, |t| {
                let level = frustum::flicker_level(note.at + t);
                level * level
            });
        }
        let send = room(note.instrument);
        // A held note goes to the part the climax doesn't cut off.
        let stem = if note.shape == Shape::Held {
            &mut music.coda
        } else {
            music.part(note.at)
        };
        stem.place(note.at, &sound, note.gain, note.pan, send);
    }
    // The rest of the music only dips, briefly and a little, as the picture glitches.
    music.act.dry.treat(hologram.clone(), 0.01, |channel, _| {
        dsp::gate(channel, |t| {
            if frustum::dips(hologram.start + t) {
                0.78
            } else {
                1.0
            }
        });
    });
    for (slot, world) in world_slots(&schedule, &worlds) {
        treat_music(&mut music.act, slot, world.treatment);
    }

    let mut buses = music;
    foley::tiles(&mut buses);
    foley::streams(&mut buses);
    foley::gleams(&mut buses);
    foley::whooshes(&mut buses);
    foley::rumble(&mut buses);
    foley::captions(&mut buses, &effects);
    foley::teapot(&mut buses, &triangles);
    foley::projection(&mut buses);
    foley::eruption(&mut buses, &effects);
    foley::swells(&mut buses, &schedule, worlds.len());
    foley::worlds(&mut buses, &effects, &schedule, &worlds);
    foley::collapse(&mut buses);

    let climax = Phase::Shutdown.start();
    let mut act = buses.act.mixed(2.2);
    for (slot, world) in world_slots(&schedule, &worlds) {
        if world.treatment == Treatment::EightBit {
            act.treat(slot, 0.004, |channel, _| eight_bit(channel));
        }
    }
    act.automate(|t| {
        dynamics(t) * (1.0 - ramp(t, climax + CUT_HOLD, climax + CUT_HOLD + CUT_TAIL))
    });
    let mut mix = buses.coda.mixed(3.2);
    mix.automate(|_| from_decibels(QUIET_LIFT));
    mix.mix(&act, 1.0);
    master(&mut mix);
    Ok(mix)
}

/// Crushes a channel of the whole mix as an eight-bit console would play it, then takes the edge
/// off: the crushing folds a lot of treble back in, so it is filtered and sits 1 dB down (set by
/// ear and from the measured treble, provisional).
fn eight_bit(channel: &mut [f32]) {
    dsp::crush(channel, 6, 12.0);
    filter(channel, Band::Low, 0.7, |_| 5000.0);
    channel.iter_mut().for_each(|v| *v *= from_decibels(-1.0));
}

/// Fades a held note away: full for its first moments, then falling ever more gently to nothing
/// at its end.
fn die_away(sound: &mut [f32]) {
    const FULL: f32 = 0.8;
    let end = dsp::time_of(sound.len());
    for (i, value) in sound.iter_mut().enumerate() {
        let gone = ((dsp::time_of(i) - FULL) / (end - FULL)).clamp(0.0, 1.0);
        *value *= (1.0 - gone) * (1.0 - gone);
    }
}

/// How much a swelling note is turned up or down at `time`: from well under its written level as
/// the finale starts to over it at the breath before the climax (set by ear, provisional).
fn swell(time: f32) -> f32 {
    let end = Phase::Worlds.end();
    let along = ramp(time, end - Schedule::FINALE, end - BREATH);
    from_decibels(-4.0 + 6.0 * along)
}

/// How much the loud part is turned up or down at `time`, over what its parts add up to: the
/// eruption held back so the drop into the worlds lands above it, the worlds' first bars held
/// back so the orchestra has somewhere to build to, and a breath before the climax (set from the
/// measured loudness curve, provisional).
fn dynamics(time: f32) -> f32 {
    let (eruption, worlds) = (Phase::Eruption.start(), Phase::Worlds.start());
    let climax = Phase::Shutdown.start();
    // Points of (time, decibels), straight between them.
    let build_up = Phase::BuildUp.start();
    let curve = [
        (0.0, QUIET_LIFT),
        (build_up, QUIET_LIFT),
        (build_up + 3.0, 0.0),
        (eruption - 0.2, 0.0),
        (eruption, -4.5),
        (worlds - 0.05, -3.5),
        (worlds, -0.5),
        (worlds + 3.0, 0.3),
        (worlds + 9.0, 1.8),
        (worlds + 15.0, 1.2),
        (worlds + 21.0, 2.4),
        (worlds + 27.0, 1.8),
        (climax - Schedule::FINALE, 1.6),
        // Up with the orchestra's second chord, where the limiter still has room; back a little
        // as the third fills the mix, so that the blur still has somewhere to rise to.
        (climax - 2.46, 2.4),
        (climax - 1.89, 1.5),
        (climax - FINALE_BLUR, 2.2),
        // A breath too short to hear as a gap, so the limiter lets go and the climax's hit
        // lands at full strength.
        (climax - BREATH - 0.02, 3.2),
        (climax - BREATH, -6.0),
        (climax - 0.005, -6.0),
        (climax, 3.4),
    ];
    let after = curve.partition_point(|&(t, _)| t <= time);
    let decibels = match (after.checked_sub(1).map(|i| curve[i]), curve.get(after)) {
        (Some((t0, a)), Some(&(t1, b))) => a + (b - a) * (time - t0) / (t1 - t0),
        (Some((_, last)), None) => last,
        _ => 0.0,
    };
    from_decibels(decibels)
}

/// The factor that changes a level by `decibels`.
fn from_decibels(decibels: f32) -> f32 {
    10.0_f32.powf(decibels / 20.0)
}

/// The slot and part of each world on the first run through them.
fn world_slots<'a>(
    schedule: &'a Schedule,
    worlds: &'a [WorldSound],
) -> impl Iterator<Item = (Range<f32>, &'a WorldSound)> {
    (0..worlds.len().min(schedule.cut_count()))
        .map(|i| (schedule.slot(i), &worlds[schedule.theme(i)]))
}

fn treat_music(music: &mut Stem, slot: Range<f32>, treatment: Treatment) {
    match treatment {
        // An eight-bit world crushes the whole mix, once it is all together.
        Treatment::Open | Treatment::EightBit => {}
        Treatment::Vast => {
            music.send.treat(slot, TREATMENT_FADE, |channel, _| {
                channel.iter_mut().for_each(|v| *v *= 2.8);
            });
        }
        Treatment::Underwater => reshape(music, slot, |channel, side| {
            filter(channel, Band::Low, 1.1, |t| {
                750.0 + 300.0 * (t * 9.0 + side as f32 * 1.3).sin()
            });
        }),
        Treatment::LoFi => reshape(music, slot, |channel, _| {
            filter(channel, Band::High, 0.8, |_| 420.0);
            filter(channel, Band::Low, 0.9, |_| 2800.0);
            channel.iter_mut().for_each(|v| *v *= 1.3);
        }),
        Treatment::Driven => reshape(music, slot, |channel, _| {
            channel
                .iter_mut()
                .for_each(|v| *v = (*v * 3.0).tanh() * 0.42);
        }),
        Treatment::Ringed => reshape(music, slot, |channel, _| {
            for (i, value) in channel.iter_mut().enumerate() {
                let ring = dsp::sine((dsp::time_of(i) * 190.0).fract());
                *value *= 0.45 + 0.8 * ring;
            }
        }),
        Treatment::Sagging => reshape(music, slot, |channel, _| dsp::slow(channel, 0.12)),
    }
}

/// Reshapes both the dry music and what it sends to the reverb over `slot`.
fn reshape(music: &mut Stem, slot: Range<f32>, process: impl Fn(&mut [f32], usize)) {
    for track in [&mut music.dry, &mut music.send] {
        track.treat(slot.clone(), TREATMENT_FADE, &process);
    }
}

/// Brings the mix to its level, holds its peaks, and silences the loop's seam.
fn master(mix: &mut Track) {
    for channel in [&mut mix.left, &mut mix.right] {
        // Below the lowest note's fundamental there is only rumble that takes up headroom.
        filter(channel, Band::High, 0.7, |_| 28.0);
        channel.iter_mut().for_each(|v| *v *= MASTER_GAIN);
    }
    limit(mix, CEILING);
    let end = dsp::time_of(mix.len());
    mix.automate(|t| ramp(t, 0.0, 0.02) * (1.0 - ramp(t, end - 0.35, end - 0.02)));
}

/// Renders the soundtrack of the stretch `span` of the loop, in seconds, to a 16-bit stereo WAV
/// file at `path`.
///
/// # Errors
///
/// Fails if a recording can't be loaded or the file can't be written.
pub fn write(
    path: &Path,
    themes: &[Theme],
    span: Range<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mix = render(themes)?;
    let first = dsp::samples(span.start).min(mix.len());
    let last = dsp::samples(span.end).clamp(first, mix.len());
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 2,
            sample_rate: RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for i in first..last {
        for value in [mix.left[i], mix.right[i]] {
            writer.write_sample((value.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mastering_holds_peaks_under_the_ceiling_and_silences_the_loops_seam() {
        let mut mix = Track::silent(2.0);
        let loud: Vec<f32> = dsp::noise(2.0, 3).iter().map(|v| v * 4.0).collect();
        mix.place(0.0, &loud, 1.0, 0.0);
        master(&mut mix);
        let channels = [&mix.left, &mix.right];
        assert!(
            channels
                .iter()
                .all(|c| c.iter().all(|v| v.abs() <= CEILING + 1e-4))
        );
        assert!(
            channels
                .iter()
                .all(|c| c[0] == 0.0 && c[c.len() - 1] == 0.0)
        );
        assert!(mix.left.iter().any(|v| v.abs() > 0.5));
    }
}
