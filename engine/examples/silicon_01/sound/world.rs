//! What a world brings to the soundtrack: the instruments that play the tune while it is on
//! screen, what it does to the music, and its own sounds. The tune itself is the same in every
//! world.

use crate::sound::accents::Accent;
use crate::sound::instrument::Instrument;

/// How a world changes the music while it is on screen.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Treatment {
    #[default]
    Open,
    /// Muffled, with a slow wobble.
    Underwater,
    /// A long, bright space around it.
    Vast,
    /// As through a small, tired speaker.
    LoFi,
    /// Pushed until it saturates.
    Driven,
    /// Sinking in pitch, like a tape slowing.
    Sagging,
    /// Multiplied by a low tone, thin and metallic.
    Ringed,
    /// The whole mix, effects included, as an 8-bit console would play it.
    EightBit,
}

/// A recorded or synthesised sound heard while a world is on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ambience {
    pub kind: Source,
    /// Seconds after the world appears that a one-off sound starts.
    pub at: f32,
    /// How long a one-off sound lasts, in seconds, or `None` for a background that lasts as long
    /// as the world does and overlaps its neighbours.
    pub lasting: Option<f32>,
    pub gain: f32,
    /// From -1 (left) to 1 (right).
    pub pan: f32,
}

impl Ambience {
    /// A background that fades in just before the world appears and out into the next one.
    pub fn bed(kind: Source, gain: f32, pan: f32) -> Self {
        Self {
            kind,
            at: 0.0,
            lasting: None,
            gain,
            pan,
        }
    }

    /// A sound `lasting` seconds, starting exactly `at` seconds after the world appears. It isn't
    /// heard if the world has gone by then.
    pub fn once(kind: Source, at: f32, lasting: f32, gain: f32, pan: f32) -> Self {
        Self {
            kind,
            at,
            lasting: Some(lasting),
            gain,
            pan,
        }
    }
}

/// Where an [`Ambience`]'s sound comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// A committed recording, by name, from this many seconds into it.
    Recording(&'static str, f32),
    Synth(Accent),
}

/// A world's part in the soundtrack.
#[derive(Clone, Debug)]
pub struct WorldSound {
    /// Play the tune's note together at the world's cut.
    pub leads: &'static [Instrument],
    pub treatment: Treatment,
    pub ambience: Vec<Ambience>,
}

impl Default for WorldSound {
    fn default() -> Self {
        Self {
            leads: &[Instrument::ViolinsShort],
            treatment: Treatment::Open,
            ambience: Vec::new(),
        }
    }
}
