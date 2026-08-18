use crate::prelude::*;

use cpu_rasteriser::prelude::*;

use std::path::PathBuf;

fn parse_colour(parts: &[&str]) -> Result<Colour, MtlError> {
    if parts.len() < 4 {
        return Err(MtlError::ParseError(format!(
            "Invalid colour line: {}",
            parts.join(" ")
        )));
    }
    let r: f32 = parts[1]
        .parse()
        .map_err(|_| MtlError::ParseError(format!("Invalid colour r value: {}", parts[1])))?;
    let g: f32 = parts[2]
        .parse()
        .map_err(|_| MtlError::ParseError(format!("Invalid colour g value: {}", parts[2])))?;
    let b: f32 = parts[3]
        .parse()
        .map_err(|_| MtlError::ParseError(format!("Invalid colour b value: {}", parts[3])))?;
    Ok(Colour::from_f32(r, g, b, 1.0))
}

fn parse_texture(parts: &[&str], base_path: &PathBuf) -> Result<TextureSampler, MtlError> {
    if parts.len() < 2 {
        return Err(MtlError::ParseError(format!(
            "Invalid texture line: {}",
            parts.join(" ")
        )));
    }
    let tex_path = PathBuf::from(parts[1]);
    let tex_full_path = if let Some(parent) = base_path.parent() {
        parent.join(tex_path)
    } else {
        tex_path.to_path_buf()
    };

    Texture::from_image(&tex_full_path)
        .map_err(|e| MtlError::TextureError(e))
        .and_then(|tx| Ok(tx.sampler(WrapMode::Repeat, FilterMode::Linear)))
}

pub fn load_mtl(path: impl AsRef<std::path::Path>) -> Result<Vec<Material>, MtlError> {
    let mtl_data = std::fs::read_to_string(&path).map_err(|e| MtlError::IoError(e))?;

    let mut materials: Vec<Material> = Vec::new();
    let mut current_material: Option<Material> = None;

    for line in mtl_data.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "newmtl" => {
                if let Some(material) = current_material.take() {
                    materials.push(material);
                }
                if parts.len() < 2 {
                    return Err(MtlError::ParseError(format!(
                        "Invalid newmtl line: {}",
                        line
                    )));
                }
                current_material = Some(Material::default(parts[1].to_string()));
            }
            "Kd" => {
                if let Some(material) = current_material.as_mut() {
                    material.diffuse = parse_colour(&parts)?;
                }
            }
            "Ka" => {
                if let Some(material) = current_material.as_mut() {
                    material.ambient = parse_colour(&parts)?;
                }
            }
            "Ks" => {
                if let Some(material) = current_material.as_mut() {
                    material.specular = parse_colour(&parts)?;
                }
            }
            "Ns" => {
                if let Some(material) = current_material.as_mut() {
                    if parts.len() < 2 {
                        return Err(MtlError::ParseError(format!("Invalid Ns line: {}", line)));
                    }
                    material.shininess = parts[1].parse().map_err(|_| {
                        MtlError::ParseError(format!("Invalid shininess value: {}", parts[1]))
                    })?;
                }
            }
            "map_Ka" => {
                if let Some(material) = current_material.as_mut() {
                    if parts.len() < 2 {
                        return Err(MtlError::ParseError(format!(
                            "Invalid map_Ka line: {}",
                            line
                        )));
                    }

                    material.ambient_texture =
                        Some(parse_texture(&parts, &path.as_ref().to_path_buf())?);
                }
            }
            "map_Kd" => {
                if let Some(material) = current_material.as_mut() {
                    if parts.len() < 2 {
                        return Err(MtlError::ParseError(format!(
                            "Invalid map_Kd line: {}",
                            line
                        )));
                    }

                    material.diffuse_texture =
                        Some(parse_texture(&parts, &path.as_ref().to_path_buf())?);
                }
            }
            "map_Ks" => {
                if let Some(material) = current_material.as_mut() {
                    if parts.len() < 2 {
                        return Err(MtlError::ParseError(format!(
                            "Invalid map_Ks line: {}",
                            line
                        )));
                    }

                    material.specular_texture =
                        Some(parse_texture(&parts, &path.as_ref().to_path_buf())?);
                }
            }
            "map_Bump" | "bump" | "norm" => {
                if let Some(material) = current_material.as_mut() {
                    if parts.len() < 2 {
                        return Err(MtlError::ParseError(format!(
                            "Invalid {} line: {}",
                            parts[0], line
                        )));
                    }

                    material.normal_texture =
                        Some(parse_texture(&parts, &path.as_ref().to_path_buf())?);
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

#[derive(Debug)]
pub enum MtlError {
    IoError(std::io::Error),
    ParseError(String),
    TextureError(TextureError),
}
impl std::fmt::Display for MtlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MtlError::IoError(e) => write!(f, "I/O error: {}", e),
            MtlError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            MtlError::TextureError(e) => write!(f, "Texture error: {}", e),
        }
    }
}
impl std::error::Error for MtlError {}
