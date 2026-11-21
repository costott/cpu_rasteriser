use std::path::{Path, PathBuf};

use cpu_rasteriser::prelude::*;

use crate::loaders::{ValuesError, parse_floats};
use crate::prelude::*;

fn load_texture(texture_path: &str, mtl_path: &Path) -> Result<TextureSampler, TextureError> {
    let texture_path = Path::new(texture_path);
    let texture_full_path = match mtl_path.parent() {
        Some(parent) => parent.join(texture_path),
        None => texture_path.to_path_buf(),
    };

    Ok(Texture::from_image(&texture_full_path)?.sampler(WrapMode::Repeat, FilterMode::Linear))
}

pub fn load_mtl(path: impl AsRef<std::path::Path>) -> Result<Vec<Material>, MtlError> {
    let path = path.as_ref();
    let mtl_data = std::fs::read_to_string(path).map_err(|source| MtlError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut materials: Vec<Material> = Vec::new();
    let mut current_material: Option<Material> = None;

    for (line_index, line) in mtl_data.lines().enumerate() {
        let line_number = line_index + 1;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let values_error = |error| MtlError::from_values(error, line_number, parts[0]);
        let colour = || {
            parse_floats(&parts)
                .map(|[r, g, b]| Colour::new(r, g, b, 1.0))
                .map_err(values_error)
        };
        let texture = || {
            let texture_path = parts.get(1).ok_or_else(|| MtlError::MissingValues {
                line: line_number,
                keyword: parts[0].to_owned(),
            })?;
            load_texture(texture_path, path).map_err(|source| MtlError::Texture {
                line: line_number,
                source,
            })
        };

        match parts[0] {
            "newmtl" => {
                if let Some(material) = current_material.take() {
                    materials.push(material);
                }
                let Some(name) = parts.get(1) else {
                    return Err(MtlError::MissingValues {
                        line: line_number,
                        keyword: parts[0].to_owned(),
                    });
                };
                current_material = Some(Material::named(name.to_string()));
            }
            "Kd" => {
                if let Some(material) = current_material.as_mut() {
                    material.diffuse = colour()?;
                }
            }
            "Ka" => {
                if let Some(material) = current_material.as_mut() {
                    material.ambient = colour()?;
                }
            }
            "Ks" => {
                if let Some(material) = current_material.as_mut() {
                    material.specular = colour()?;
                }
            }
            "Ns" => {
                if let Some(material) = current_material.as_mut() {
                    let [shininess] = parse_floats(&parts).map_err(values_error)?;
                    material.shininess = shininess;
                }
            }
            "map_Ka" => {
                if let Some(material) = current_material.as_mut() {
                    material.ambient_texture = Some(texture()?);
                }
            }
            "map_Kd" => {
                if let Some(material) = current_material.as_mut() {
                    material.diffuse_texture = Some(texture()?);
                }
            }
            "map_Ks" => {
                if let Some(material) = current_material.as_mut() {
                    material.specular_texture = Some(texture()?);
                }
            }
            "map_Bump" | "bump" | "norm" => {
                if let Some(material) = current_material.as_mut() {
                    material.normal_texture = Some(texture()?);
                }
            }
            _ => {}
        }
    }

    if let Some(material) = current_material.take() {
        materials.push(material);
    }

    Ok(materials)
}

/// An error loading an MTL material library with [`load_mtl`].
///
/// Line numbers count from 1.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MtlError {
    /// The file couldn't be read.
    #[error("failed to read `{}`", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// A statement has fewer values than its keyword requires.
    #[error("line {line}: `{keyword}` statement has too few values")]
    MissingValues { line: usize, keyword: String },
    /// A value that should be a number isn't one.
    #[error("line {line}: invalid number `{value}`")]
    InvalidNumber {
        line: usize,
        value: String,
        #[source]
        source: std::num::ParseFloatError,
    },
    /// A texture map named by the library couldn't be loaded.
    #[error("line {line}: failed to load the texture")]
    Texture {
        line: usize,
        #[source]
        source: TextureError,
    },
}
impl MtlError {
    fn from_values(error: ValuesError, line: usize, keyword: &str) -> Self {
        match error {
            ValuesError::Missing => MtlError::MissingValues {
                line,
                keyword: keyword.to_owned(),
            },
            ValuesError::InvalidNumber { value, source } => MtlError::InvalidNumber {
                line,
                value,
                source,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::loaders::test_support::TempFile;

    #[test]
    fn missing_texture_reports_its_line_and_chains_the_image_error() {
        let file = TempFile::new(
            "mtl-missing-texture",
            "materials.mtl",
            "newmtl a\nmap_Kd missing.png\n",
        );

        let error = load_mtl(&file.path).unwrap_err();

        let MtlError::Texture { line, source } = &error else {
            panic!("expected a texture error, got {error:?}");
        };
        assert_eq!(*line, 2);
        assert!(
            matches!(source, TextureError::Image { path, .. } if path.ends_with("missing.png"))
        );
        assert!(
            error
                .source()
                .and_then(Error::source)
                .is_some_and(|source| source.is::<image::ImageError>())
        );
    }

    #[test]
    fn colour_with_too_few_values_reports_its_line_and_keyword() {
        let file = TempFile::new("mtl-missing-values", "materials.mtl", "newmtl a\nKd 1 1\n");

        let error = load_mtl(&file.path).unwrap_err();

        assert!(
            matches!(&error, MtlError::MissingValues { line: 2, keyword } if keyword == "Kd"),
            "{error:?}"
        );
    }
}
