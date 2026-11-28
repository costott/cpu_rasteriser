//! Command-line options: the window size, the render scale, and offline rendering of the demo to
//! PNG frames for turning into a GIF or a video.
//!
//! Flags (after `--` with `cargo run`):
//!
//! - `--window <width>x<height>`: open the window at this size (default 1280x720). Frames are
//!   saved at the size the window actually gets.
//! - `--scale <factor>`: render the scene at this fraction of the window's resolution (default
//!   0.6, for smooth real-time playback). Recordings look best at 2, which supersamples.
//! - `--from <seconds>`: start the demo at this time.
//! - `--record <dir>`: render every frame at a fixed time step and save it to `<dir>` as
//!   `frame_00000.png`, `frame_00001.png`, …, then exit. `--fps <n>` sets the step (default 30)
//!   and `--to <seconds>` where to stop (default the end of the loop). Frame time doesn't matter,
//!   so a recording is smooth however long each frame takes to render.
//! - `--shots <dir> --at <t1,t2,…>`: save one still at each listed time, then exit.
//! - `--theme <name>`: show this theme on its own whenever the worlds phase is on screen, for
//!   working on one theme (names are in `themes::all`).
//! - `--no-captions`: leave out every caption, for footage without text such as the README's GIF.
//! - `--soundtrack <file.wav>`: render the soundtrack to a WAV file and exit, without opening a
//!   window. `--from` and `--to` choose the stretch of the loop (default all of it).
//!
//! `media.sh`, next to this file, records the whole loop as a video with its soundtrack, or the
//! README's GIF, with these flags and encodes it with ffmpeg.

use std::ops::Range;
use std::path::PathBuf;

use cpu_rasteriser::prelude::*;

/// The scene's resolution relative to the window, unless `--scale` says otherwise. Chosen by
/// measuring real-time frame times at 1280×720 on a 16-thread desktop CPU.
const DEFAULT_SCALE: f32 = 0.6;

/// The window's size unless `--window` says otherwise.
const DEFAULT_WINDOW: (usize, usize) = (1280, 720);

pub struct Options {
    pub window: (usize, usize),
    pub scale: f32,
    pub capture: Capture,
    pub theme: Option<String>,
    pub captions: bool,
    /// Where to write the soundtrack, and the stretch of the loop to write, instead of playing.
    pub soundtrack: Option<(PathBuf, Range<f32>)>,
}

impl Options {
    /// Reads the flags from the command line.
    ///
    /// # Panics
    ///
    /// Panics with a usage message if a flag is missing its value or the value doesn't parse.
    pub fn from_args() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let scale = flag(&args, "--scale")
            .map(|v| {
                v.parse::<f32>()
                    .ok()
                    .filter(|scale| (0.1..=2.0).contains(scale))
                    .unwrap_or_else(|| panic!("--scale takes a factor from 0.1 to 2, not {v:?}"))
            })
            .unwrap_or(DEFAULT_SCALE);
        let window = flag(&args, "--window").map_or(DEFAULT_WINDOW, |v| {
            v.split_once('x')
                .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                .filter(|&(w, h)| w > 0 && h > 0)
                .unwrap_or_else(|| panic!("--window takes a size such as 1920x1080, not {v:?}"))
        });
        Self {
            window,
            scale,
            capture: Capture::from_args(&args),
            theme: flag(&args, "--theme").map(str::to_string),
            captions: !args.iter().any(|arg| arg == "--no-captions"),
            soundtrack: flag(&args, "--soundtrack").map(|path| {
                let from = seconds(&args, "--from").unwrap_or(0.0);
                let to = seconds(&args, "--to").unwrap_or(crate::director::LOOP_LENGTH);
                (path.into(), from..to)
            }),
        }
    }
}

/// The value after `name` on the command line, if it is there.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|arg| arg == name)?;
    Some(
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} needs a value"))
            .as_str(),
    )
}

/// The number after `name` on the command line, if it is there.
fn seconds(args: &[String], name: &str) -> Option<f32> {
    flag(args, name).map(|v| {
        v.parse::<f32>()
            .unwrap_or_else(|_| panic!("{name} takes a number of seconds, not {v:?}"))
    })
}

pub struct Capture {
    start: f32,
    mode: Mode,
    saved: usize,
}

enum Mode {
    Off,
    Record {
        dir: PathBuf,
        fps: f32,
        frames: usize,
    },
    Shots {
        dir: PathBuf,
        times: Vec<f32>,
    },
}

impl Capture {
    fn from_args(args: &[String]) -> Self {
        let value = |name: &str| flag(args, name);
        let seconds = |name: &str| seconds(args, name);

        let from = seconds("--from").unwrap_or(0.0);
        let mode = if let Some(dir) = value("--record") {
            let fps = seconds("--fps").unwrap_or(30.0);
            let to = seconds("--to").unwrap_or(crate::director::LOOP_LENGTH);
            Mode::Record {
                dir: dir.into(),
                fps,
                frames: ((to - from) * fps).round().max(1.0) as usize,
            }
        } else if let Some(dir) = value("--shots") {
            let times = value("--at")
                .expect("--shots needs --at <t1,t2,…>")
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("--at takes times in seconds, not {t:?}"))
                })
                .collect();
            Mode::Shots {
                dir: dir.into(),
                times,
            }
        } else {
            Mode::Off
        };
        let start = match &mode {
            Mode::Shots { times, .. } => times.first().copied().unwrap_or(0.0),
            _ => from,
        };
        Self {
            start,
            mode,
            saved: 0,
        }
    }

    pub fn start_time(&self) -> f32 {
        self.start
    }

    /// The demo time of the next frame to save, while capturing. A recording steps from its start
    /// before its first frame, and each frame's time comes from its index so that rounding never
    /// accumulates over a long recording.
    pub fn frame_time(&self) -> Option<f32> {
        match &self.mode {
            Mode::Off => None,
            Mode::Record { fps, .. } => Some(self.start + (self.saved + 1) as f32 / fps),
            Mode::Shots { times, .. } => times.get(self.saved).copied(),
        }
    }

    /// Saves the presented frame if capturing, and exits once the capture is complete.
    ///
    /// # Panics
    ///
    /// Panics if the output directory can't be created or a frame can't be written.
    pub fn save(&mut self, target: &RenderTarget, time: f32) {
        let (dir, name, done) = match &self.mode {
            Mode::Off => return,
            Mode::Record { dir, frames, .. } => (
                dir,
                format!("frame_{:05}.png", self.saved),
                self.saved + 1 >= *frames,
            ),
            Mode::Shots { dir, times } => (
                dir,
                format!("shot_{time:05.2}.png"),
                self.saved + 1 >= times.len(),
            ),
        };
        std::fs::create_dir_all(dir).expect("the capture directory should be creatable");
        let bytes: Vec<u8> = target
            .pixels()
            .iter()
            .flat_map(|colour| {
                let c = colour.clamp(0.0, 1.0);
                [c.r, c.g, c.b].map(|channel| (channel * 255.0).round() as u8)
            })
            .collect();
        image::save_buffer(
            dir.join(name),
            &bytes,
            target.width() as u32,
            target.height() as u32,
            image::ColorType::Rgb8,
        )
        .expect("the frame should be writable");
        self.saved += 1;
        if done {
            // The engine only takes exit requests from events, which a backend needn't send
            // every frame; the frames are all on disk, so ending here loses nothing.
            std::process::exit(0);
        }
    }
}
