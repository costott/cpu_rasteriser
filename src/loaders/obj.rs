use std::{collections::HashMap, eprintln, num::FpCategory::Normal};

use super::mtl::load_mtl;
use crate::prelude::*;

pub fn load_obj(file_path: impl AsRef<std::path::Path>) -> Result<Model, ObjError> {
    let obj_data = std::fs::read_to_string(&file_path).map_err(|e| ObjError::IoError(e))?;

    // let mut vertices: Vec<Vertex3D> = Vec::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut texcoords: Vec<Vec2> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut materials: Vec<Material> = Vec::new();

    let mut obj_meshes: Vec<ObjMesh> = Vec::new();
    let mut current_mesh = ObjMesh::empty();

    let mut material_map: HashMap<String, usize> = HashMap::new();

    for line in obj_data.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" => {
                // Vertex position
                if parts.len() < 4 {
                    return Err(ObjError::ParseError(format!(
                        "Invalid vertex line: {}",
                        line
                    )));
                }
                let x: f32 = parts[1].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid vertex x value: {}", parts[1]))
                })?;
                let y: f32 = parts[2].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid vertex y value: {}", parts[2]))
                })?;
                let z: f32 = parts[3].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid vertex z value: {}", parts[3]))
                })?;
                positions.push(Vec3::new(x, y, z));
            }
            "vn" => {
                // Vertex normal
                let x: f32 = parts[1].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid normal x value: {}", parts[1]))
                })?;
                let y: f32 = parts[2].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid normal y value: {}", parts[2]))
                })?;
                let z: f32 = parts[3].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid normal z value: {}", parts[3]))
                })?;
                normals.push(Vec3::new(x, y, z));
            }
            "vt" => {
                // Texture coordinate
                if parts.len() < 3 {
                    return Err(ObjError::ParseError(format!(
                        "Invalid texture coordinate line: {}",
                        line
                    )));
                }
                let u: f32 = parts[1].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid texture u value: {}", parts[1]))
                })?;
                let v: f32 = parts[2].parse().map_err(|_| {
                    ObjError::ParseError(format!("Invalid texture v value: {}", parts[2]))
                })?;
                texcoords.push(Vec2::new(u, v));
            }
            "f" => {
                // Face indices
                if parts.len() < 4 {
                    return Err(ObjError::ParseError(format!("Invalid face line: {}", line)));
                }

                let mut face_indices = Vec::new();

                for i in 1..parts.len() {
                    let mut indices = parts[i].split('/');
                    let mut position_index: u32 = indices
                        .next()
                        .ok_or_else(|| {
                            ObjError::ParseError(format!("Invalid face index: {}", parts[i]))
                        })?
                        .parse()
                        .map_err(|_| {
                            ObjError::ParseError(format!("Invalid face index value: {}", parts[i]))
                        })?;
                    let mut texcoord_index: u32 = indices
                        .next()
                        .ok_or_else(|| {
                            ObjError::ParseError(format!("Invalid face index: {}", parts[i]))
                        })?
                        .parse()
                        .map_err(|_| {
                            ObjError::ParseError(format!("Invalid face index value: {}", parts[i]))
                        })?;
                    let mut normal_index: u32 = indices
                        .next()
                        .ok_or_else(|| {
                            ObjError::ParseError(format!("Invalid face index: {}", parts[i]))
                        })?
                        .parse()
                        .map_err(|_| {
                            ObjError::ParseError(format!("Invalid face index value: {}", parts[i]))
                        })?;

                    let vertex_key = (position_index, texcoord_index, normal_index);

                    let vertex_index =
                        if let Some(&index) = current_mesh.vertex_map.get(&vertex_key) {
                            index
                        } else {
                            let position = positions
                                .get((position_index - 1) as usize)
                                .ok_or_else(|| {
                                    ObjError::ParseError(format!(
                                        "Position index out of bounds: {}",
                                        position_index
                                    ))
                                })?;
                            let normal =
                                normals.get((normal_index - 1) as usize).ok_or_else(|| {
                                    ObjError::ParseError(format!(
                                        "Normal index out of bounds: {}",
                                        normal_index
                                    ))
                                })?;
                            let texcoord = texcoords
                                .get((texcoord_index - 1) as usize)
                                .ok_or_else(|| {
                                    ObjError::ParseError(format!(
                                        "Texcoord index out of bounds: {}",
                                        texcoord_index
                                    ))
                                })?;

                            let vertex =
                                Vertex3D::new_with_normal(*position, Colour::WHITE, *normal);
                            current_mesh.vertices.push(vertex);

                            let new_index = (current_mesh.vertices.len() - 1) as u32;
                            current_mesh.vertex_map.insert(vertex_key, new_index);

                            new_index
                        };
                    face_indices.push(vertex_index);
                }

                // Triangulate the face using a triangle fan
                for i in 1..(face_indices.len() - 1) {
                    current_mesh.indices.push(face_indices[0]);
                    current_mesh.indices.push(face_indices[i]);
                    current_mesh.indices.push(face_indices[i + 1]);
                }
            }
            "o" | "g" => {
                // Object / Group
                finish_current_mesh(&mut current_mesh, &mut obj_meshes);
            }
            "mtllib" => {
                // Material library
                if parts.len() < 2 {
                    return Err(ObjError::ParseError(format!(
                        "Invalid mtllib line: {}",
                        line
                    )));
                }
                let mtl_path = std::path::Path::new(parts[1]);
                let mtl_full_path = if let Some(parent) = file_path.as_ref().parent() {
                    parent.join(mtl_path)
                } else {
                    mtl_path.to_path_buf()
                };
                materials = load_mtl(&mtl_full_path).map_err(|e| {
                    ObjError::ParseError(format!("Failed to load material library: {}", e))
                })?;

                for (i, material) in materials.iter().enumerate() {
                    material_map.insert(material.name.clone(), i);
                }
            }
            "usemtl" => {
                // Use material
                if parts.len() < 2 {
                    return Err(ObjError::ParseError(format!(
                        "Invalid usemtl line: {}",
                        line
                    )));
                }

                finish_current_mesh(&mut current_mesh, &mut obj_meshes);

                let material_name = parts[1];
                if let Some(&material_index) = material_map.get(material_name) {
                    current_mesh.material_index = Some(material_index);
                } else {
                    eprintln!("Warning: Material '{}' not found", material_name);
                }
            }
            _ => {
                // Ignore other lines for now
                eprintln!("Warning: Ignoring unsupported line in OBJ file: {}", line);
            }
        }
    }

    finish_current_mesh(&mut current_mesh, &mut obj_meshes);

    let meshes = obj_meshes
        .into_iter()
        .map(|obj_mesh| Mesh::new(obj_mesh.vertices, obj_mesh.indices, obj_mesh.material_index))
        .collect();

    let model = Model::new(
        meshes,
        materials,
        ModelTransform::new(Vec3::ZERO, Vec3::ZERO, Vec3::ONE),
    )?;
    Ok(model)
}

#[derive(Clone)]
struct ObjMesh {
    vertices: Vec<Vertex3D>,
    indices: Vec<u32>,
    material_index: Option<usize>,

    vertex_map: HashMap<(u32, u32, u32), u32>, // (position_index, texcoord_index, normal_index) -> vertex_index
}
impl ObjMesh {
    fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            material_index: None,
            vertex_map: HashMap::new(),
        }
    }
}

fn finish_current_mesh(current_mesh: &mut ObjMesh, obj_meshes: &mut Vec<ObjMesh>) {
    if !current_mesh.vertices.is_empty() || !current_mesh.indices.is_empty() {
        obj_meshes.push(current_mesh.clone());
        *current_mesh = ObjMesh::empty();
    }
}

#[derive(Debug)]
pub enum ObjError {
    IoError(std::io::Error),
    ParseError(String),
    Model(ModelError),
}
impl std::fmt::Display for ObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjError::IoError(e) => write!(f, "IO error: {}", e),
            ObjError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            ObjError::Model(e) => write!(f, "Model error: {}", e),
        }
    }
}
impl std::error::Error for ObjError {}
impl From<ModelError> for ObjError {
    fn from(err: ModelError) -> Self {
        ObjError::Model(err)
    }
}
