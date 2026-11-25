//! The music: an original score as a list of notes on the film's timeline.
//!
//! One four-note tune in D minor runs through it. It starts on a solo piano; harp and strings
//! join at the teapot; short strings, brass and drums build to the eruption; in the worlds every
//! cut is a beat and the orchestra fills out section by section. Through the finale the tune never
//! stops: it carries on over the shrinking cuts, then the orchestra takes it up, ever faster and
//! on more instruments, over one broad statement of it which comes home to D at the climax. There
//! the low chord is held while the rest stops dead, and under it the piano is alone again,
//! resolving to D major on the closing title. It keeps 120 BPM up to the eruption, where every
//! phase starts on a beat. Levels and voicings are set by ear (provisional).

use std::ops::Range;

use crate::director::{CAPTIONS, CaptionKind, FINALE_BLUR, Phase, ramp};
use crate::sound::instrument::Instrument::{self, *};
use crate::themes::Schedule;

/// Seconds in a beat at 120 BPM, and in a bar of four.
const BEAT: f32 = 0.5;
const BAR: f32 = 4.0 * BEAT;

/// The tune, as MIDI notes and the beat of a bar each starts on: A, up to D and F, down to E.
const TUNE: [(u8, f32); 4] = [(69, 0.0), (74, 1.0), (77, 1.5), (76, 2.0)];

/// A chord: its bass note in the octave from C2, and three notes in the octave around middle C.
#[derive(Clone, Copy)]
struct Chord {
    bass: u8,
    notes: [u8; 3],
}

const D_MINOR: Chord = chord(38, [62, 65, 69]);
const B_FLAT: Chord = chord(34, [62, 65, 70]);
const F_MAJOR: Chord = chord(41, [60, 65, 69]);
const C_MAJOR: Chord = chord(36, [60, 64, 67]);
const G_MINOR: Chord = chord(43, [62, 67, 70]);
const A_MAJOR: Chord = chord(33, [61, 64, 69]);

/// The four chords the tune sits on, in turn.
const ROUND: [Chord; 4] = [D_MINOR, B_FLAT, F_MAJOR, C_MAJOR];

/// The gap the worlds' pulse aims for at the drop, and from the cut where it settles on: each
/// cut's slot is split into the whole number of even steps nearest the aim (set by ear,
/// provisional).
const PULSE_AIM: (f32, f32) = (0.25, 0.18);
const PULSE_SETTLES: usize = 19;

/// The cuts of the worlds' first run at which more of the orchestra joins the strings and drums,
/// each the start of a chord. None leaves before the climax.
const WOODWINDS: usize = 4;
const HORNS: usize = 8;
const SNARE_AND_TROMBONES: usize = 12;
const TRUMPETS_AND_TUBA: usize = 20;
const CHOIR_AND_BELLS: usize = 36;
/// From this cut the choir and strings reach an octave higher and the timpani drive.
const FULL: usize = 44;

/// The least time a world's note rings in the worlds' first run, in seconds.
const LEAD_RING: f32 = 0.7;

/// The chords the finale climbs through, each voiced higher than the last, to D minor's
/// dominant, so that the climax arrives as the way home. Each starts where the tune comes round
/// again over the finale's first cuts, and the last lasts to the climax.
const CLIMB: [Chord; 3] = [
    chord(34, [58, 62, 65]),
    chord(36, [60, 64, 67]),
    chord(45, [61, 64, 69]),
];
/// How strongly each of [`CLIMB`]'s chords is held under the finale, and the dominant once more
/// from the blur, as fractions of the full orchestra (set by ear, provisional).
const SUPPORT: [f32; 4] = [0.5, 0.7, 0.88, 1.05];
/// How strongly the broad tune's four notes are played (set by ear, provisional).
const BROAD: [f32; 4] = [0.5, 0.62, 0.78, 0.95];
/// The finale's tune takes two notes to a cut while that leaves them at least this far apart, in
/// seconds. Beyond that it runs free of the cuts, about [`FREE_GAP`] apart and closing to 20
/// notes a second: much faster and it would stop reading as notes (judgement, provisional).
const LOCKED_GAP: f32 = 0.06;
const FREE_GAP: f32 = 0.057;
/// How far ahead of its time a short violin note in the finale's run starts, in seconds: its
/// recordings take 30 to 80 ms to speak (measured).
const BOW_LEAD: f32 = 0.03;
/// The two notes of the chord under each of [`TUNE`]'s, in turn, when the trumpets play it in
/// chords over the dominant: A major under its A and E, D minor under its D and F.
const FANFARE: [[u8; 2]; 4] = [[61, 64], [65, 69], [69, 74], [69, 73]];

/// How long before the climax the finale stops for breath, in seconds, so its hit lands at full
/// strength.
pub const BREATH: f32 = 0.07;

/// How long the climax's low chord is held before it is released, in seconds: it has died away
/// by the closing title.
const HOLD: f32 = 2.9;

const fn chord(bass: u8, notes: [u8; 3]) -> Chord {
    Chord { bass, notes }
}

/// One note of the score.
#[derive(Clone, Copy, Debug)]
pub struct Note {
    /// When it starts, in seconds into the film.
    pub at: f32,
    pub instrument: Instrument,
    /// The MIDI note, which need not be whole.
    pub pitch: f32,
    /// Seconds it is held before it is released.
    pub length: f32,
    pub gain: f32,
    /// From -1 (left) to 1 (right).
    pub pan: f32,
    pub shape: Shape,
}

/// How the mix shapes a note's level over its time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shape {
    /// As its instrument plays it.
    #[default]
    Plain,
    /// It rings on, dying away, when the climax cuts the rest of the music off.
    Held,
    /// It grows louder with the whole finale.
    Swelling,
}

/// Collects notes, with each instrument's usual place in the stereo picture.
#[derive(Default)]
struct Writer(Vec<Note>);

impl Writer {
    fn note(&mut self, at: f32, instrument: Instrument, pitch: u8, length: f32, gain: f32) {
        let pan = match instrument {
            Violins | ViolinsShort => -0.35,
            Cellos | CellosShort => 0.3,
            Basses => 0.2,
            Pizzicato => -0.3,
            Flute => -0.15,
            Piccolo => 0.15,
            Horn => -0.2,
            Trumpet => 0.25,
            Trombone => 0.35,
            Tuba => 0.3,
            Harp => -0.25,
            Glockenspiel | SteelPan | Bells => 0.25,
            Timpani => 0.1,
            Snare => -0.1,
            Clash => 0.2,
            Gong => -0.2,
            _ => 0.0,
        };
        self.0.push(Note {
            at,
            instrument,
            pitch: pitch as f32,
            length,
            gain,
            pan,
            shape: Shape::Plain,
        });
    }

    /// A note that is held when the climax cuts the rest off.
    fn hold(&mut self, at: f32, instrument: Instrument, pitch: u8, length: f32, gain: f32) {
        self.note(at, instrument, pitch, length, gain);
        if let Some(note) = self.0.last_mut() {
            note.shape = Shape::Held;
        }
    }

    /// Whatever `write` adds swells with the finale.
    fn swelling(&mut self, write: impl FnOnce(&mut Self)) {
        let first = self.0.len();
        write(self);
        for note in &mut self.0[first..] {
            note.shape = Shape::Swelling;
        }
    }

    fn chord(&mut self, at: f32, instrument: Instrument, pitches: &[u8], length: f32, gain: f32) {
        for &pitch in pitches {
            self.note(
                at,
                instrument,
                pitch,
                length,
                gain / (pitches.len() as f32).sqrt(),
            );
        }
    }

    /// A drum, at its recording's own pitch (the nominal note its file is named after).
    fn drum(&mut self, at: f32, instrument: Instrument, gain: f32) {
        self.note(at, instrument, 60, 3.0, gain);
    }

    /// The tune from `at`, `stretch` times as slow as at 120 BPM.
    fn tune(&mut self, at: f32, instrument: Instrument, octave: i8, stretch: f32, gain: f32) {
        for (i, (pitch, beat)) in TUNE.into_iter().enumerate() {
            let length = if i == 3 { 2.0 } else { 0.9 } * BEAT * stretch;
            let accent = if i == 3 { 0.85 } else { 1.0 };
            self.note(
                at + beat * BEAT * stretch,
                instrument,
                (pitch as i8 + 12 * octave) as u8,
                length,
                gain * accent,
            );
        }
    }
}

/// Times from `span.start`, `step` apart, before `span.end`.
fn every(span: Range<f32>, step: f32) -> impl Iterator<Item = (usize, f32)> {
    (0..)
        .map(move |i| (i, span.start + i as f32 * step))
        .take_while(move |&(_, t)| t < span.end - 1e-3)
}

/// The whole score. `leads` names the instruments that play each world's note, by theme.
pub fn score(schedule: &Schedule, leads: &[&[Instrument]]) -> Vec<Note> {
    let mut w = Writer::default();
    opening(&mut w);
    reveal(&mut w);
    teapot(&mut w);
    hologram(&mut w);
    build_up(&mut w);
    eruption(&mut w);
    worlds(&mut w, schedule, leads);
    ending(&mut w);
    w.0
}

fn opening(w: &mut Writer) {
    let macro_start = Phase::Macro.start();
    // A low drone fades in under the tiles.
    w.chord(0.6, Pad, &[38, 45], macro_start - 0.4, 0.13);
    w.chord(
        macro_start,
        Pad,
        &[38, 45],
        Phase::Reveal.start() - macro_start,
        0.08,
    );
    w.tune(macro_start, Piano, 0, 1.0, 0.42);
    w.tune(macro_start + 2.0 * BAR, Piano, 0, 1.0, 0.36);
    // The title lands on a low D with a soft swell of the cymbal.
    let title = macro_start + 2.0 * BEAT;
    w.chord(title, Piano, &[38, 50], 3.5, 0.65);
    w.note(title, Glockenspiel, 86, 2.0, 0.12);
    w.chord(title + BAR, Piano, &[50, 57, 65], 3.0, 0.25);
}

/// The tune over slow chords and a heartbeat, leaving room around each line of narration.
fn reveal(w: &mut Writer) {
    let start = Phase::Reveal.start();
    for (i, t) in every(start..Phase::Rise.start() - BAR, 2.0 * BAR) {
        let chord = ROUND[i % 4];
        w.chord(t, Piano, &[chord.bass, chord.bass + 12], 3.6, 0.3);
        w.chord(t + BEAT, Piano, &chord.notes, 3.0, 0.16);
        w.tune(t + BAR, Piano, 0, 1.0, 0.3);
        w.chord(t, Pad, &[chord.bass + 12, chord.bass + 19], 2.0 * BAR, 0.07);
    }
    // A last chord leans into the rise.
    let turn = Phase::Rise.start() - BAR;
    w.chord(turn, Piano, &[C_MAJOR.bass, C_MAJOR.bass + 12], 2.1, 0.3);
    w.chord(turn + BEAT, Piano, &C_MAJOR.notes, 1.6, 0.18);
    w.chord(
        turn,
        Pad,
        &[C_MAJOR.bass + 12, C_MAJOR.bass + 19],
        BAR,
        0.08,
    );
    for (_, t) in every(start..Phase::Rise.start(), BAR) {
        w.note(t, Timpani, 45, 0.2, 0.07);
        w.note(t + 0.32, Timpani, 45, 0.2, 0.045);
    }
    for caption in CAPTIONS {
        if caption.kind == CaptionKind::Narration && Phase::at(caption.start) == Phase::Reveal {
            w.note(caption.drawn(), Glockenspiel, 81, 1.5, 0.09);
        }
    }
}

fn teapot(w: &mut Writer) {
    let (rise, teapot) = (Phase::Rise.start(), Phase::Teapot.start());
    // The harp runs up into the strings.
    for (i, pitch) in [50, 53, 57, 62, 65, 69, 74, 77, 81, 86]
        .into_iter()
        .enumerate()
    {
        w.note(
            rise + 1.6 + i as f32 * 0.14,
            Harp,
            pitch,
            1.0,
            0.2 + i as f32 * 0.015,
        );
    }
    w.chord(rise, Violins, &D_MINOR.notes, 3.2, 0.14);
    w.note(rise, Cellos, 50, 3.2, 0.16);
    let end = Phase::Hologram.start();
    for (i, t) in every(teapot..end, BAR) {
        let chord = [D_MINOR, B_FLAT, F_MAJOR, C_MAJOR, D_MINOR][i % 5];
        let swell = 0.2 + 0.14 * ramp(t, teapot, teapot + 8.0);
        w.chord(t, Violins, &chord.notes, BAR, swell);
        w.note(t, Cellos, chord.bass + 12, BAR, swell * 0.9);
        w.note(t, Basses, chord.bass, BAR, 0.2);
        let [a, b, c] = chord.notes;
        for (step, pitch) in [a, b, c, a + 12, b + 12, a + 12, c, b]
            .into_iter()
            .enumerate()
        {
            w.note(t + step as f32 * BEAT * 0.5, Harp, pitch, 0.6, 0.24);
        }
        if i % 2 == 0 {
            w.tune(t, Piano, 0, 1.0, 0.2);
        }
        // A plucked beat, which the hologram carries on.
        for (_, beat) in every(t..(t + BAR).min(end), BEAT) {
            w.note(beat, Pizzicato, chord.bass + 24, 0.3, 0.14);
        }
    }
}

/// The harmony at half the pace, over detuned pads that lean towards trouble, while the
/// teapot's beat carries on underneath in plucked eighths.
fn hologram(w: &mut Writer) {
    let (start, end) = (Phase::Hologram.start(), Phase::BuildUp.start());
    let chords = [D_MINOR, G_MINOR, B_FLAT, A_MAJOR];
    for (chord, (_, t)) in chords.into_iter().zip(every(start..end, 3.0)) {
        w.chord(t, Pad, &chord.notes, 3.0, 0.185);
        w.note(t, Pad, chord.bass + 12, 3.0, 0.118);
        w.note(t, Basses, chord.bass, 3.0, 0.32);
        w.note(t, Cellos, chord.bass + 12, 3.0, 0.26);
        w.chord(t, Violins, &chord.notes, 3.0, 0.185);
        w.tune(t + 0.5, Piano, 1, 2.0, 0.22);
    }
    for (i, t) in every(start..end, BEAT / 2.0) {
        let chord = chords[(((t - start) / 3.0) as usize).min(3)];
        let [a, b, c] = chord.notes;
        w.note(t, Harp, [a, c, b, c][i % 4] + 12, 0.5, 0.193);
        if i % 2 == 0 {
            w.note(t, Pizzicato, chord.notes[(i / 2) % 3], 0.3, 0.25);
        }
    }
}

/// Short strings from eighths to sixteenths, brass swelling up to the dominant, and drums that
/// close into a roll.
fn build_up(w: &mut Writer) {
    let (start, end) = (Phase::BuildUp.start(), Phase::Eruption.start());
    let loud = |t: f32| 0.5 + 0.5 * ((t - start) / (end - start)).powi(2);
    // The strings hold on under the first bar, so the build starts from where the hologram left.
    w.chord(start, Violins, &D_MINOR.notes, BAR, 0.3);
    w.note(start, Basses, D_MINOR.bass, BAR, 0.36);
    for (i, t) in every(start..start + BAR, BEAT / 2.0) {
        let [a, b, c] = D_MINOR.notes;
        w.note(t, Harp, [a, c, b, c][i % 4] + 12, 0.5, 0.22);
    }
    let double = end - 3.5;
    let steps = every(start..double, BEAT / 2.0).chain(every(double..end - 0.1, BEAT / 4.0));
    for (i, t) in steps {
        // The last two seconds sit on the dominant, wanting to come home.
        let chord = if t < end - 2.0 { D_MINOR } else { A_MAJOR };
        let root = chord.bass + if chord.bass < 36 { 24 } else { 12 };
        let figure = [0, 0, 7, 0, 12, 0, 7, 0][i % 8];
        w.note(t, CellosShort, root + figure, 0.2, 0.34 * loud(t) + 0.08);
        if t >= start + BAR {
            w.note(
                t,
                ViolinsShort,
                chord.notes[i % 3] + 12,
                0.18,
                0.26 * loud(t),
            );
        }
    }
    for (i, t) in every(start..end - 2.0, BAR).take(3) {
        let pitch = [50, 53, 57][i];
        w.note(t, Trombone, pitch, BAR, 0.14 + 0.07 * i as f32);
        w.note(
            t + BEAT,
            Horn,
            pitch + 12,
            BAR - BEAT,
            0.12 + 0.06 * i as f32,
        );
    }
    w.chord(end - 2.0, Trombone, &[45, 52, 57], 1.9, 0.45);
    w.chord(end - 2.0, Horn, &[57, 61, 64], 1.9, 0.45);
    for (_, t) in every(start + BAR..end - 2.0, BEAT) {
        w.note(t, Timpani, 50, 0.4, 0.34 * loud(t));
    }
    for (_, t) in every(end - 2.0..end - 0.05, BEAT / 4.0) {
        w.note(t, Timpani, 45, 0.2, 0.3 * loud(t));
    }
    for t in closing(end - 2.6, end, 0.125, 0.05) {
        w.note(t, Snare, 60, 0.1, 0.08 + 0.3 * ramp(t, end - 2.6, end));
    }
}

fn eruption(w: &mut Writer) {
    let (start, worlds) = (Phase::Eruption.start(), Phase::Worlds.start());
    impact(w, start, 0.8);
    // The grid of primitives climbs the chord into the white-out.
    let climb = [62, 65, 69, 74, 77, 81, 86, 89, 93];
    for (i, t) in every(start + BEAT..worlds - 0.05, BEAT / 4.0) {
        let loud = ramp(t, start, worlds);
        let pitch = climb[(i / 2 + [0, 2, 1, 3][i % 4]) % climb.len()];
        w.note(t, Harp, pitch, 0.3, 0.14 + 0.2 * loud);
        if i % 2 == 0 {
            w.note(t, Glockenspiel, pitch, 0.4, 0.05 + 0.1 * loud);
        }
    }
    w.chord(start, Violins, &[62, 65, 69, 74], worlds - start, 0.36);
    w.chord(worlds - 1.0, Horn, &A_MAJOR.notes, 1.0, 0.5);
    w.drum(worlds - 2.4, CymbalSwell, 0.4);
}

/// A hit on every drum, with the low D under it.
fn impact(w: &mut Writer, at: f32, gain: f32) {
    w.drum(at, BassDrum, gain);
    w.drum(at, Cymbal, gain * 0.6);
    w.note(at, Timpani, 50, 2.5, gain * 0.8);
    w.chord(at, Piano, &[26, 38, 50], 2.5, gain * 0.7);
    w.chord(at, Trombone, &[38, 45, 50], 1.2, gain * 0.6);
}

/// Every cut is a beat, the strings pulse between the beats, and from one chord to the next the
/// orchestra fills out until the finale has all of it.
fn worlds(w: &mut Writer, schedule: &Schedule, leads: &[&[Instrument]]) {
    let (start, end) = (Phase::Worlds.start(), Phase::Worlds.end());
    impact(w, start, 1.0);
    let slots: Vec<Range<f32>> = (0..schedule.cut_count())
        .map(|cut| schedule.slot(cut))
        .collect();
    // The first run shows each theme once; the finale's flurry follows it.
    let main = leads.len().min(slots.len());
    let finale = slots.get(main).map_or(end, |slot| slot.start);
    let chords = chord_starts(main);
    for (i, &first) in chords.iter().enumerate() {
        let next = chords.get(i + 1).copied().unwrap_or(main);
        let chord = ROUND[i % 4];
        let span = slots[first].start..slots.get(next).map_or(finale, |slot| slot.start);
        held_parts(
            w,
            chord,
            first,
            span.clone(),
            (span.start - start) / (end - start),
        );
        for cut in first..next {
            let along = (slots[cut].start - start) / (end - start);
            beat(
                w,
                chord,
                cut,
                &slots[cut],
                leads[schedule.theme(cut)],
                along,
                LEAD_RING,
            );
        }
    }
    let finale_leads: Vec<&[Instrument]> = (main..slots.len())
        .map(|cut| leads[schedule.theme(cut)])
        .collect();
    finale_rush(w, &slots[main..], &finale_leads, main, end);
}

/// The cuts of the first run, `count` long, that start a new chord: every fourth, and every eighth
/// once the cuts have settled at their shortest, so the harmony doesn't hurry with them.
fn chord_starts(count: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut cut = 0;
    while cut < count {
        starts.push(cut);
        cut += if cut < TRUMPETS_AND_TUBA { 4 } else { 8 };
    }
    starts
}

/// How many even steps the pulse takes through the slot of `cut`, `length` seconds long.
fn pulse_steps(cut: usize, length: f32) -> usize {
    let settled = (cut as f32 / PULSE_SETTLES as f32).min(1.0);
    let aim = PULSE_AIM.0 + (PULSE_AIM.1 - PULSE_AIM.0) * settled;
    ((length / aim).round() as usize).max(1)
}

/// One cut: the drum and the tune's next note on the world's own instrument, ringing on past the
/// cut for at least `ring` seconds in all, with the pulse and the woodwinds' answering line through
/// its slot.
fn beat(
    w: &mut Writer,
    chord: Chord,
    cut: usize,
    slot: &Range<f32>,
    leads: &[Instrument],
    along: f32,
    ring: f32,
) {
    let length = slot.end - slot.start;
    w.drum(slot.start, BassDrum, 0.3 + 0.3 * along);
    // The tune's notes in turn, climbing an octave for the last third.
    let (pitch, _) = TUNE[cut % TUNE.len()];
    let octave = if along > 0.66 { 12 } else { 0 };
    let ringing = (length * 1.35).max(ring);
    world_lead(w, slot.start, leads, pitch + octave, ringing);
    let steps = pulse_steps(cut, length);
    let gap = length / steps as f32;
    for step in 0..steps {
        let t = slot.start + step as f32 * gap;
        let root = chord.bass + 12 + if (cut + step) % 4 == 2 { 7 } else { 0 };
        w.note(t, CellosShort, root, 0.18, 0.24 + 0.14 * along);
        if step % 2 == 1 || steps == 1 {
            let pitch = chord.notes[(cut + step) % 3] + 12;
            w.note(t, ViolinsShort, pitch, 0.15, 0.12 + 0.14 * along);
        }
        if cut >= SNARE_AND_TROMBONES {
            let accent = if step == 0 { 0.07 } else { 0.03 };
            w.note(t, Snare, 60, 0.1, accent + 0.08 * along);
        }
    }
    // While a world stays long enough, two quick notes lead into the next.
    if length >= 0.7 {
        for (i, back) in [2.0, 1.0].into_iter().enumerate() {
            w.note(
                slot.end - back * gap,
                Harp,
                chord.notes[i + 1] + 12,
                0.4,
                0.14 + 0.05 * i as f32,
            );
        }
    }
    if cut >= WOODWINDS {
        let answer = chord.notes[(cut + 1) % 3] + 12;
        w.note(slot.start, Flute, answer, length * 1.1, 0.14 + 0.16 * along);
        if cut >= TRUMPETS_AND_TUBA {
            w.note(
                slot.start,
                Piccolo,
                answer + 12,
                length * 1.1,
                0.05 + 0.07 * along,
            );
        }
    }
}

/// A world's own instruments sounding the tune's note together.
fn world_lead(w: &mut Writer, at: f32, leads: &[Instrument], pitch: u8, length: f32) {
    let gain = 0.55 / (leads.len() as f32).sqrt();
    for &lead in leads {
        w.note(at, lead, pitch, length, gain);
    }
}

/// What is held under one chord's cuts, over `span`, from the cut `first`: the strings with the
/// tune drawn out across them, joined in turn by horns, trombones, trumpets over the tuba, and
/// the choir with the bells. Each note overlaps the next chord's, so the cuts run on into one
/// another.
fn held_parts(w: &mut Writer, chord: Chord, first: usize, span: Range<f32>, along: f32) {
    let (start, length) = (span.start, span.end - span.start);
    let held = length + 0.3;
    w.chord(start, Violins, &chord.notes, held, 0.18 + 0.2 * along);
    w.note(start, Basses, chord.bass, held, 0.26 + 0.18 * along);
    w.note(start, Cellos, chord.bass + 12, held, 0.16 + 0.14 * along);
    w.note(start, Timpani, chord.bass + 12, 0.8, 0.26 + 0.22 * along);
    let quarter = length / 4.0;
    for (i, (pitch, _)) in TUNE.into_iter().enumerate() {
        let at = start + i as f32 * quarter;
        w.note(at, Violins, pitch, quarter + 0.25, 0.18 + 0.2 * along);
        if first >= TRUMPETS_AND_TUBA {
            w.note(at, Trumpet, pitch, quarter + 0.12, 0.14 + 0.2 * along);
        }
    }
    if first >= HORNS {
        w.chord(start, Horn, &chord.notes, held, 0.18 + 0.3 * along);
    }
    if first >= SNARE_AND_TROMBONES {
        let low = [chord.bass + 12, chord.bass + 19];
        w.chord(start, Trombone, &low, held, 0.2 + 0.16 * along);
    }
    if first >= TRUMPETS_AND_TUBA {
        w.note(start, Tuba, chord.bass, held, 0.24 + 0.14 * along);
    }
    if first >= CHOIR_AND_BELLS {
        // Soft as it enters, full by the time it reaches higher.
        let swell = ramp(first as f32, CHOIR_AND_BELLS as f32 - 4.0, FULL as f32);
        let [a, b, c] = chord.notes;
        let voices = [chord.bass + 12, a, b, c];
        w.chord(start, Choir, &voices, held, 0.25 + 0.55 * swell);
        w.note(start, Bells, a, held, 0.16 + 0.08 * swell);
    }
    if first >= FULL {
        let high = chord.notes.map(|pitch| pitch + 12);
        w.note(start, Choir, high[0], held, 0.3);
        w.chord(start, Violins, &high, held, 0.22);
        w.drum(start, Clash, 0.16);
        for (_, t) in every(start + BEAT / 2.0..span.end, BEAT / 2.0) {
            w.note(t, Timpani, chord.bass + 12, 0.3, 0.14 + 0.2 * along);
        }
    }
}

/// The last cuts, numbered from `first`, shrink to a frame each. The song carries straight on
/// over the first of them, one note of the tune to each, on its world's own instrument (in
/// `leads`, a cut to each). Then the orchestra takes the tune up and runs with it (see
/// [`tune_times`]), joining a section at a time over chords that climb to the dominant, until by
/// the blur it covers everything else.
fn finale_rush(
    w: &mut Writer,
    cuts: &[Range<f32>],
    leads: &[&[Instrument]],
    first: usize,
    end: f32,
) {
    let Some(opening) = cuts.first() else {
        return;
    };
    let count = cuts.partition_point(|cut| cut.start < end - FINALE_BLUR);
    let (hits, blurred) = cuts.split_at(count);
    let blur = blurred.first().map_or(end, |cut| cut.start);
    // Where the second and third chords start, and with them the orchestra's next sections.
    let step = |i: usize| hits.get(i * TUNE.len()).map_or(blur, |cut| cut.start);
    let stages = [opening.start, step(1), step(2), blur, end];

    for (i, slot) in hits.iter().enumerate() {
        let along = (slot.start - Phase::Worlds.start()) / (end - Phase::Worlds.start());
        if i < TUNE.len() {
            // Each world's note rings less long than the last, so the notes stay apart.
            let ring = LEAD_RING * (1.0 - 0.15 * (i + 1) as f32);
            beat(w, CLIMB[0], first + i, slot, leads[i], along, ring);
        } else {
            let loud = i as f32 / count as f32;
            let chord = CLIMB[(i / TUNE.len()).min(CLIMB.len() - 1)];
            w.drum(slot.start, BassDrum, 0.6 + 0.3 * loud);
            w.note(slot.start, Snare, 60, 0.1, 0.12 + 0.12 * loud);
            w.note(slot.start, Timpani, chord.bass + 12, 0.15, 0.3 + 0.2 * loud);
        }
    }
    // The worlds' last bars' driving timpani, until every cut has its own.
    for (_, t) in every(opening.start + BEAT / 2.0..stages[1], BEAT / 2.0) {
        w.note(t, Timpani, CLIMB[0].bass + 12, 0.3, 0.3);
    }
    takeover(w, hits, leads, &stages);
    w.swelling(|w| {
        broad_tune(w, &stages);
        support(w, &stages);
    });
    last_violins(w, &hits[(2 * TUNE.len()).min(hits.len())..], blur..end);
    for (&at, gain) in stages.iter().zip([0.16, 0.25, 0.3]) {
        w.drum(at, Clash, gain);
    }
    final_roll(w, blur..end);
    w.drum(end - 2.4, CymbalSwell, 0.7);
}

/// When each note of the finale's tune starts, over the cuts before the blur and on to `end`: one
/// to a cut while the song carries on, then two to a cut while they stay [`LOCKED_GAP`] apart,
/// then free of the cuts and ever quicker. They never come further apart, and there is a whole
/// number of times through the tune, so it ends on the note that leads home.
fn tune_times(hits: &[Range<f32>], end: f32) -> Vec<f32> {
    let mut times: Vec<f32> = hits.iter().take(TUNE.len()).map(|cut| cut.start).collect();
    let mut free = end;
    for cut in hits.iter().skip(TUNE.len()) {
        let half = (cut.end - cut.start) / 2.0;
        if half < LOCKED_GAP {
            free = cut.start;
            break;
        }
        times.extend([cut.start, cut.start + half]);
    }
    let Some(&last) = times.last() else {
        return times;
    };
    let span = end - free;
    let nearest = times.len() + (span / FREE_GAP).round() as usize;
    let whole = (nearest + TUNE.len() / 2) / TUNE.len() * TUNE.len();
    let more = whole.saturating_sub(times.len());
    // The first gap of the free run is no wider than the one before it, and they close from there.
    let quicken = ((free - last) * more as f32 / span - 1.0).clamp(0.0, 0.2);
    times.extend((0..more).map(|i| {
        let x = i as f32 / more as f32;
        free + span * ((1.0 + quicken) * x - quicken * x * x)
    }));
    times
}

/// The tune running through the finale, taken up by more of the orchestra at each of `stages`: a
/// xylophone doubles it from the first note, coming up under the worlds' own instruments; strings
/// and woodwinds run with it in octaves from the second chord; and the glockenspiel and the
/// trumpets, in chords until the blur, join at the dominant.
fn takeover(w: &mut Writer, hits: &[Range<f32>], leads: &[&[Instrument]], stages: &[f32; 5]) {
    let [start, second, third, blur, end] = *stages;
    let last = end - BREATH;
    let times = tune_times(hits, last);
    for (i, &at) in times.iter().enumerate() {
        let gap = times.get(i + 1).copied().unwrap_or(last) - at;
        let (note, _) = TUNE[i % TUNE.len()];
        // The tune keeps the octave the worlds' last third took it to.
        let high = note + 12;
        let loud = ramp(at, start, end);
        w.note(at, Xylophone, high, gap, 0.26 + 0.14 * loud);
        if at < second - 1e-3 {
            continue;
        }
        // The worlds still play it wherever one of their cuts falls on a note.
        if let Some(cut) = hits.iter().position(|cut| (cut.start - at).abs() < 1e-4) {
            world_lead(w, at, leads[cut], high, gap * 1.35);
        }
        let dominant = at >= third - 1e-3;
        let strings = if dominant { 0.5 } else { 0.36 } + 0.1 * loud;
        for pitch in [note, high] {
            w.note(at - BOW_LEAD, ViolinsShort, pitch, gap, strings * 0.75);
        }
        w.note(at, CellosShort, note - 12, gap, strings);
        w.note(at, Flute, high, gap, if dominant { 0.32 } else { 0.22 });
        w.note(
            at,
            Piccolo,
            high + 12,
            gap,
            if dominant { 0.14 } else { 0.08 },
        );
        if dominant {
            w.note(at, Glockenspiel, high + 12, gap, 0.1);
            // At the top speed the brass can't speak: from the blur it holds the broad tune.
            if at < blur - 1e-3 {
                let [a, b] = FANFARE[i % TUNE.len()];
                w.chord(at, Trumpet, &[a, b, note], gap * 0.8, 0.65);
            }
        }
    }
}

/// The tune once more, drawn out so that each of `stages` has one note of it: horns, trombones,
/// trumpets, the choir and the low parts, in octaves.
fn broad_tune(w: &mut Writer, stages: &[f32; 5]) {
    for (i, ((note, _), span)) in TUNE.into_iter().zip(stages.windows(2)).enumerate() {
        let (at, length) = (span[0], span[1] - span[0] + 0.12);
        let level = BROAD[i];
        w.chord(at, Horn, &[note - 12, note], length, 0.8 * level);
        w.chord(at, Trombone, &[note - 24, note - 12], length, 0.65 * level);
        w.chord(at, Choir, &[note - 12, note], length, 0.7 * level);
        // From the second chord the low strings and tuba put its weight underneath.
        if i > 0 {
            w.note(at, Basses, note - 24, length, 0.6 * level);
            w.note(at, Tuba, note - 24, length, 0.55 * level);
        }
        // Over the third chord the trumpets are busy with the running tune.
        if i != 2 {
            w.note(at, Trumpet, note, length, 0.45 * level);
        }
    }
}

/// What is held under the finale's tune: the chords of [`CLIMB`] in the low parts and choir, each
/// stronger than the last, with the violins on the first two (the dominant's are
/// [`last_violins`]).
fn support(w: &mut Writer, stages: &[f32; 5]) {
    let [low, middle, dominant] = CLIMB;
    for (i, chord) in [low, middle, dominant, dominant].into_iter().enumerate() {
        // Each runs a little into the next, so the step up leaves no hole.
        let (at, length) = (stages[i], stages[i + 1] - stages[i] + 0.15);
        let level = SUPPORT[i];
        let root = [chord.bass + 12, chord.bass + 19];
        w.chord(at, Trombone, &root, length, 0.4 * level);
        w.note(at, Tuba, chord.bass, length, 0.65 * level);
        w.note(at, Basses, chord.bass, length, 0.75 * level);
        w.note(at, Cellos, chord.bass + 12, length, 0.45 * level);
        let [a, b, c] = chord.notes;
        let voices = [chord.bass + 12, a, b, c];
        w.chord(at, Choir, &voices, length, 0.9 * level);
        w.note(at, Bells, a, length, 0.2);
        if i < 2 {
            w.chord(at, Violins, &chord.notes, length, 0.36);
        }
    }
}

/// The violins on the dominant, over `cuts`, its cuts before the blur: its chord where it sits,
/// until the tune comes round again; then their last chords, high up, stronger from the blur with
/// a higher chord over them to the climax.
fn last_violins(w: &mut Writer, cuts: &[Range<f32>], blur: Range<f32>) {
    let dominant = CLIMB[2].notes;
    let high = dominant.map(|pitch| pitch + 12);
    let from = cuts.get(TUNE.len()).map_or(blur.start, |cut| cut.start);
    if let Some(first) = cuts.first() {
        w.chord(
            first.start,
            Violins,
            &dominant,
            from - first.start + 0.15,
            0.36,
        );
    }
    w.chord(from, Violins, &high, blur.start - from + 0.15, 0.56);
    let length = blur.end - blur.start;
    w.chord(blur.start, Violins, &high, length + 0.15, 0.7);
    w.chord(blur.start, Violins, &[85, 88, 93], length, 0.35);
}

/// The drums in a roll that closes over `span`, under the running tune.
fn final_roll(w: &mut Writer, span: Range<f32>) {
    let (start, end) = (span.start, span.end);
    for t in closing(start, end, 0.065, 0.035) {
        let loud = ramp(t, start, end);
        w.note(t, Timpani, 57, 0.2, 0.55 + 0.2 * loud);
        w.note(t, BassDrum, 60, 0.05, 0.5 + 0.1 * loud);
        let pitch = 60 + (12.0 * loud) as u8;
        w.note(t, Snare, pitch, 0.03, 0.16 + 0.14 * loud);
    }
    w.drum(start, Clash, 0.35);
    w.note(start, Bells, 69, end - start, 0.3);
}

/// The times of a roll from `start` to just before `end`, its strokes closing from `wide` seconds
/// apart to `narrow`.
fn closing(start: f32, end: f32, wide: f32, narrow: f32) -> impl Iterator<Item = f32> {
    std::iter::successors(Some(start), move |&t| {
        Some(t + wide - (wide - narrow) * ramp(t, start, end))
    })
    .take_while(move |&t| t < end - 0.03)
}

/// One last hit at the climax, which stops dead but for its low chord; and under that, suddenly
/// soft, the piano alone: the tune slower, coming home to D major on the closing title.
fn ending(w: &mut Writer) {
    let climax = Phase::Shutdown.start();
    impact(w, climax, 1.0);
    w.chord(climax, Horn, &[57, 62, 65, 69], 0.35, 0.8);
    w.chord(climax, Trumpet, &[69, 74, 77], 0.35, 0.65);
    w.chord(climax, Violins, &[74, 77, 81, 86], 0.35, 0.7);
    w.drum(climax, Clash, 0.5);
    low_hold(w, climax);

    let title = climax + 3.3;
    // The tune, slowing, its last note held back until the title.
    for (at, pitch) in [(0.95, 69), (1.75, 74), (2.15, 77), (2.65, 76)] {
        w.note(climax + at, Piano, pitch, 0.9, 0.2);
    }
    w.chord(climax + 0.95, Piano, &[50, 57], 2.2, 0.14);
    w.chord(title, Piano, &[38, 50, 57], 4.0, 0.27);
    w.chord(title + 0.12, Piano, &[66, 69, 74], 4.0, 0.22);
    w.note(title + 1.4, Piano, 78, 3.0, 0.13);
    w.note(title + 1.4, Glockenspiel, 90, 2.5, 0.05);
}

/// The climax's low chord, still in D minor so the title's D major comes as the way home: low
/// strings, tuba, choir, gong and a timpani roll, dying away.
fn low_hold(w: &mut Writer, at: f32) {
    w.hold(at, Basses, 38, HOLD, 0.6);
    w.hold(at, Cellos, 38, HOLD, 0.4);
    w.hold(at, Cellos, 45, HOLD, 0.3);
    w.hold(at, Tuba, 38, HOLD, 0.55);
    for pitch in [50, 57, 62, 65, 69] {
        w.hold(at, Choir, pitch, HOLD, 0.5);
    }
    w.hold(at, Piano, 26, HOLD, 0.45);
    w.hold(at, Piano, 38, HOLD, 0.45);
    w.hold(at, Gong, 60, HOLD, 0.6);
    w.hold(at, BassDrum, 60, HOLD, 0.8);
    for (_, t) in every(at..at + 2.2, 0.085) {
        let left = 1.0 - (t - at) / 2.2;
        w.hold(t, Timpani, 50, 0.2, 0.34 * left * left);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notes() -> (Schedule, Vec<Note>) {
        let schedule = Schedule::new(56);
        let notes = score(&schedule, &[&[ViolinsShort] as &[Instrument]; 56]);
        (schedule, notes)
    }

    fn sounds_at(notes: &[Note], instrument: Instrument, at: f32) -> bool {
        notes
            .iter()
            .any(|note| note.instrument == instrument && (note.at - at).abs() < 1e-3)
    }

    #[test]
    fn every_cut_of_the_worlds_lands_on_the_bass_drum_until_the_picture_blurs() {
        let (schedule, notes) = notes();
        let blur = Phase::Worlds.end() - FINALE_BLUR;
        for cut in 0..schedule.cut_count() {
            let start = schedule.slot(cut).start;
            if start < blur {
                assert!(sounds_at(&notes, BassDrum, start), "cut {cut}");
            }
        }
    }

    /// The finale's running tune, as the notes of the xylophone that leads it.
    fn running_tune(notes: &[Note]) -> Vec<Note> {
        let finale = Phase::Worlds.end() - Schedule::FINALE;
        let mut tune: Vec<Note> = notes
            .iter()
            .filter(|note| note.instrument == Xylophone && note.at > finale - 1e-3)
            .copied()
            .collect();
        tune.sort_by(|a, b| a.at.total_cmp(&b.at));
        tune
    }

    #[test]
    fn the_worlds_instruments_carry_the_tune_on_into_the_finale() {
        let schedule = Schedule::new(56);
        // An instrument the score gives nothing else to play.
        let notes = score(&schedule, &[&[Kalimba] as &[Instrument]; 56]);
        for cut in 52..60 {
            let start = schedule.slot(cut).start;
            let next = TUNE[cut % TUNE.len()].0 % 12;
            assert!(
                notes.iter().any(|note| {
                    note.instrument == Kalimba
                        && (note.at - start).abs() < 1e-3
                        && note.pitch as u8 % 12 == next
                }),
                "cut {cut}"
            );
        }
    }

    #[test]
    fn the_tune_runs_through_the_finale_in_order_never_slowing() {
        let (schedule, notes) = notes();
        let tune = running_tune(&notes);
        // It starts with the finale, and comes right through its last time before the breath.
        assert!((tune[0].at - schedule.slot(56).start).abs() < 1e-3);
        let end = Phase::Worlds.end() - BREATH;
        assert!(end - tune[tune.len() - 1].at < 0.1);
        assert_eq!(tune.len() % TUNE.len(), 0);
        for (i, note) in tune.iter().enumerate() {
            assert_eq!(note.pitch as u8 % 12, TUNE[i % TUNE.len()].0 % 12, "{i}");
        }
        let gaps: Vec<f32> = tune
            .windows(2)
            .map(|pair| pair[1].at - pair[0].at)
            .collect();
        assert!(
            gaps.windows(2).all(|pair| pair[1] <= pair[0] + 1e-3),
            "{gaps:?}"
        );
        assert!(gaps.iter().all(|&gap| gap > 0.05), "{gaps:?}");
    }

    #[test]
    fn more_of_the_orchestra_runs_with_the_tune_at_each_of_the_finales_chords() {
        let (schedule, notes) = notes();
        let blur = Phase::Worlds.end() - FINALE_BLUR;
        // The fewest instruments sounding any one of the tune's notes under each chord.
        let mut fewest = [usize::MAX; 3];
        for note in running_tune(&notes).iter().filter(|note| note.at < blur) {
            let chord = (1..3)
                .filter(|&i| note.at > schedule.slot(56 + i * TUNE.len()).start - 1e-3)
                .count();
            let mut with: Vec<Instrument> = notes
                .iter()
                .filter(|other| {
                    // A bow starts a little ahead of its note.
                    (-BOW_LEAD - 1e-3..1e-3).contains(&(other.at - note.at))
                        && other.pitch as u8 % 12 == note.pitch as u8 % 12
                })
                .map(|other| other.instrument)
                .collect();
            with.sort_by_key(|&instrument| instrument as u8);
            with.dedup();
            fewest[chord] = fewest[chord].min(with.len());
        }
        assert!(fewest[0] < fewest[1] && fewest[1] < fewest[2], "{fewest:?}");
    }

    #[test]
    fn the_finales_held_orchestra_only_gets_louder() {
        let (_, notes) = notes();
        let mut swelling: Vec<&Note> = notes
            .iter()
            .filter(|note| note.shape == Shape::Swelling)
            .collect();
        swelling.sort_by(|a, b| a.at.total_cmp(&b.at));
        let levels: Vec<f32> = swelling
            .chunk_by(|a, b| (a.at - b.at).abs() < 1e-3)
            .map(|together| together.iter().map(|note| note.gain).sum())
            .collect();
        assert!(levels.len() > 1);
        assert!(
            levels.windows(2).all(|pair| pair[1] > pair[0]),
            "{levels:?}"
        );
    }

    #[test]
    fn a_roll_closes_from_the_blur_into_the_climax() {
        let (_, notes) = notes();
        let (end, blur) = (Phase::Worlds.end(), Phase::Worlds.end() - FINALE_BLUR);
        let mut strokes: Vec<f32> = notes
            .iter()
            .filter(|note| note.instrument == Timpani && (blur..end - 1e-3).contains(&note.at))
            .map(|note| note.at)
            .collect();
        strokes.sort_by(f32::total_cmp);
        let gaps: Vec<f32> = strokes.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(strokes[0] - blur < 0.1 && end - strokes[strokes.len() - 1] < 0.1);
        assert!(
            gaps.windows(2).all(|pair| pair[1] <= pair[0] + 1e-3),
            "{gaps:?}"
        );
    }

    #[test]
    fn after_the_climax_only_the_held_chord_and_the_quiet_piano_play() {
        let (_, notes) = notes();
        let after: Vec<_> = notes
            .iter()
            .filter(|note| note.shape != Shape::Held && note.at > Phase::Shutdown.start() + 0.5)
            .collect();
        assert!(!after.is_empty());
        assert!(
            after
                .iter()
                .all(|note| { matches!(note.instrument, Piano | Glockenspiel) && note.gain < 0.3 })
        );
    }

    #[test]
    fn the_climaxs_low_chord_is_held_after_the_rest_stops() {
        let (_, notes) = notes();
        for instrument in [Basses, Tuba, Choir] {
            assert!(
                notes.iter().any(|note| {
                    note.shape == Shape::Held
                        && note.instrument == instrument
                        && note.at == Phase::Shutdown.start()
                        && note.length >= 2.5
                }),
                "{instrument:?}"
            );
        }
    }

    #[test]
    fn the_worlds_pulse_never_drags() {
        let (schedule, notes) = notes();
        let mut pulses: Vec<f32> = notes
            .iter()
            .filter(|note| note.instrument == CellosShort)
            .map(|note| note.at)
            .filter(|&at| at >= Phase::Worlds.start() && at < schedule.slot(56).start)
            .collect();
        pulses.sort_by(f32::total_cmp);
        assert!(pulses.windows(2).all(|pair| pair[1] - pair[0] <= 0.26));
        // Through the finale's hits the pulse only quickens.
        let blur = Phase::Worlds.end() - FINALE_BLUR;
        let mut rush: Vec<f32> = notes
            .iter()
            .filter(|note| note.instrument == CellosShort)
            .map(|note| note.at)
            .filter(|&at| at > schedule.slot(56).start - 1e-3 && at < blur)
            .collect();
        rush.sort_by(f32::total_cmp);
        rush.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        let gaps: Vec<f32> = rush.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(
            gaps.windows(2).all(|pair| pair[1] <= pair[0] + 1e-3),
            "{gaps:?}"
        );
        // Taken four cuts at a time, the pulses never come further apart than they did before.
        let spacing: Vec<f32> = (0..14)
            .map(|bar| {
                let span = schedule.slot(4 * bar).start..schedule.slot(4 * bar + 4).start;
                let count = pulses
                    .iter()
                    .filter(|&&at| at > span.start - 1e-3 && at < span.end - 1e-3)
                    .count();
                (span.end - span.start) / count as f32
            })
            .collect();
        assert!(
            spacing.windows(2).all(|pair| pair[1] <= pair[0] + 1e-3),
            "{spacing:?}"
        );
    }

    #[test]
    fn once_a_part_of_the_orchestra_joins_it_plays_until_the_climax() {
        let (schedule, notes) = notes();
        for (instrument, cut) in [
            (Violins, 0),
            (Flute, WOODWINDS),
            (Horn, HORNS),
            (Snare, SNARE_AND_TROMBONES),
            (Trombone, SNARE_AND_TROMBONES),
            (Trumpet, TRUMPETS_AND_TUBA),
            (Tuba, TRUMPETS_AND_TUBA),
            (Choir, CHOIR_AND_BELLS),
            (Bells, CHOIR_AND_BELLS),
        ] {
            let joins = schedule.slot(cut).start;
            for (_, t) in every(joins..Phase::Shutdown.start() - 0.25, 0.25) {
                assert!(
                    notes.iter().any(|note| {
                        note.instrument == instrument
                            && note.at <= t + 0.25
                            && note.at + note.length >= t
                    }),
                    "{instrument:?} drops out at {t}"
                );
            }
        }
    }
}
