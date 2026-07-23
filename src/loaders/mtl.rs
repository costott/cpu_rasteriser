use crate::prelude::*;

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
    Ok(Colour::from_f32(r, g, b))
}

pub fn load_mtl(path: impl AsRef<std::path::Path>) -> Result<Vec<Material>, MtlError> {
    let mtl_data = std::fs::read_to_string(path).map_err(|e| MtlError::IoError(e))?;

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
                current_material = Some(Material::new(
                    parts[1].to_string(),
                    Colour::WHITE,
                    Colour::WHITE,
                    Colour::WHITE,
                    0.0,
                ));
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
}
impl std::fmt::Display for MtlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MtlError::IoError(e) => write!(f, "I/O error: {}", e),
            MtlError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}
impl std::error::Error for MtlError {}
