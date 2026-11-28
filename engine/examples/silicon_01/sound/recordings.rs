//! Reading the committed recordings: mono 16-bit WAV files at [`RATE`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::sound::dsp::RATE;

/// Where the recordings are, from the workspace root.
const ROOT: &str = "assets/silicon_01/audio";

/// A recording that couldn't be loaded.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("couldn't read {path}: {source}")]
    Wav { path: PathBuf, source: hound::Error },
    #[error("couldn't list {path}: {source}")]
    List {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{0} isn't mono 16-bit at 48 kHz")]
    Format(PathBuf),
    #[error("{0} holds no recorded notes")]
    NoNotes(PathBuf),
    #[error("{0} isn't named after a MIDI note")]
    NoteName(PathBuf),
}

fn read(path: &Path) -> Result<Vec<f32>, LoadError> {
    let wav = |source| LoadError::Wav {
        path: path.to_path_buf(),
        source,
    };
    let reader = hound::WavReader::open(path).map_err(wav)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != RATE || spec.bits_per_sample != 16 {
        return Err(LoadError::Format(path.to_path_buf()));
    }
    reader
        .into_samples::<i16>()
        .map(|sample| Ok(sample.map_err(wav)? as f32 / 32768.0))
        .collect()
}

/// Every `.wav` file in `folder`, in name order.
fn files(folder: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let list = |source| LoadError::List {
        path: folder.to_path_buf(),
        source,
    };
    let mut found = Vec::new();
    for entry in std::fs::read_dir(folder).map_err(list)? {
        let path = entry.map_err(list)?.path();
        if path.extension().is_some_and(|e| e == "wav") {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

/// An instrument's recorded notes, each with the MIDI note its file is named after.
pub fn notes(instrument: &str) -> Result<Vec<(u8, Vec<f32>)>, LoadError> {
    let folder = Path::new(ROOT).join("notes").join(instrument);
    let notes = files(&folder)?
        .into_iter()
        .map(|path| {
            let note = path
                .file_stem()
                .and_then(|stem| stem.to_str()?.parse().ok())
                .ok_or_else(|| LoadError::NoteName(path.clone()))?;
            Ok((note, read(&path)?))
        })
        .collect::<Result<Vec<_>, LoadError>>()?;
    if notes.is_empty() {
        return Err(LoadError::NoNotes(folder));
    }
    Ok(notes)
}

/// The recorded effects, by file name without its extension.
pub struct Effects(HashMap<String, Vec<f32>>);

impl Effects {
    /// Loads every recording in the effects folder.
    ///
    /// # Errors
    ///
    /// Fails if the folder or a file in it can't be read.
    pub fn load() -> Result<Self, LoadError> {
        files(&Path::new(ROOT).join("effects"))?
            .into_iter()
            .map(|path| {
                let name = path
                    .file_stem()
                    .expect("a .wav file has a name")
                    .to_string_lossy()
                    .into_owned();
                Ok((name, read(&path)?))
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    /// The recording called `name`.
    ///
    /// # Panics
    ///
    /// Panics if there is none: the committed set holds every name the soundtrack asks for.
    pub fn get(&self, name: &str) -> &[f32] {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("the committed effects should include {name:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_committed_recording_is_credited() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(ROOT);
        let credits = std::fs::read_to_string(root.join("CREDITS.md"))
            .expect("the recordings are credited in CREDITS.md");
        let mut folders = vec![root.join("effects")];
        let notes = std::fs::read_dir(root.join("notes")).expect("there are recorded notes");
        folders.extend(notes.map(|entry| entry.expect("the folder is readable").path()));
        for folder in folders {
            for path in files(&folder).expect("the folder is readable") {
                let name = path
                    .strip_prefix(&root)
                    .expect("the file is under the root")
                    .to_string_lossy()
                    .replace('\\', "/");
                assert!(
                    credits.contains(&format!("`{name}`")),
                    "{name} has no credit"
                );
            }
        }
    }
}
