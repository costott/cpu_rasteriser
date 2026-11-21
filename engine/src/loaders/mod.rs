use std::num::ParseFloatError;

pub use mtl::*;
pub use obj::*;

pub mod mtl;
pub mod obj;

/// Why the values of a statement couldn't be parsed, before the loader adds the line and keyword.
enum ValuesError {
    Missing,
    InvalidNumber {
        value: String,
        source: ParseFloatError,
    },
}

/// Parses the first `N` values after a statement's keyword, ignoring any further values.
fn parse_floats<const N: usize>(parts: &[&str]) -> Result<[f32; N], ValuesError> {
    let values = parts.get(1..=N).ok_or(ValuesError::Missing)?;

    let mut floats = [0.0; N];
    for (float, value) in floats.iter_mut().zip(values) {
        *float = value.parse().map_err(|source| ValuesError::InvalidNumber {
            value: (*value).to_owned(),
            source,
        })?;
    }
    Ok(floats)
}

#[cfg(test)]
mod test_support {
    /// A file in its own temporary directory, removed with the directory on drop.
    pub(super) struct TempFile {
        dir: std::path::PathBuf,
        pub(super) path: std::path::PathBuf,
    }
    impl TempFile {
        /// Writes `contents` to `file_name` in a directory unique to `test_name` and this process.
        pub(super) fn new(test_name: &str, file_name: &str, contents: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("engine-loader-{test_name}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(file_name);
            std::fs::write(&path, contents).unwrap();
            Self { dir, path }
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}
