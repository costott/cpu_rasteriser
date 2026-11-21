use std::num::ParseFloatError;

/// Why the values of a statement couldn't be parsed, before the loader adds the line and keyword.
pub(super) enum ValuesError {
    Missing,
    InvalidNumber {
        value: String,
        source: ParseFloatError,
    },
}

/// Parses the first `N` values after a statement's keyword, ignoring any further values.
pub(super) fn parse_floats<const N: usize>(parts: &[&str]) -> Result<[f32; N], ValuesError> {
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
