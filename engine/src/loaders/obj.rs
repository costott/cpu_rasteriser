use std::collections::HashMap;
use std::path::PathBuf;

use cpu_rasteriser::prelude::*;

use crate::loaders::mtl::load_mtl;
use crate::loaders::values::{ValuesError, parse_floats};
use crate::prelude::*;

/// Load a 3D model from an OBJ file (should really be called load_model)
///
/// # Errors
///
/// Returns an [`ObjError`] if the file can't be read, a statement is malformed or refers to an
/// undefined attribute, a material library it names can't be loaded, or the result isn't a valid
/// [`Model`].
pub fn load_obj(file_path: impl AsRef<std::path::Path>) -> Result<Model<ObjVertex>, ObjError> {
    let obj_data = std::fs::read_to_string(&file_path).map_err(|source| ObjError::Io {
        path: file_path.as_ref().to_path_buf(),
        source,
    })?;

    let mut positions: Vec<Vec3> = Vec::new();
    let mut texcoords: Vec<Vec2> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut materials: Vec<Material> = Vec::new();

    let mut obj_meshes: Vec<ObjMesh> = Vec::new();
    let mut current_mesh = ObjMesh::empty();

    let mut material_map: HashMap<String, usize> = HashMap::new();

    for (line_index, line) in obj_data.lines().enumerate() {
        let line_number = line_index + 1;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let values_error = |error| ObjError::from_values(error, line_number, parts[0]);
        let missing_values = || ObjError::MissingValues {
            line: line_number,
            keyword: parts[0].to_owned(),
        };

        match parts[0] {
            "#" => {
                // Comment line, ignore
            }
            "v" => {
                let [x, y, z] = parse_floats(&parts).map_err(values_error)?;
                positions.push(Vec3::new(x, y, z));
            }
            "vn" => {
                let [x, y, z] = parse_floats(&parts).map_err(values_error)?;
                normals.push(Vec3::new(x, y, z));
            }
            "vt" => {
                let [u, v] = parse_floats(&parts).map_err(values_error)?;
                texcoords.push(Vec2::new(u, v));
            }
            "f" => {
                if parts.len() < 4 {
                    return Err(missing_values());
                }

                let mut face_indices = Vec::new();

                for part in &parts[1..] {
                    let invalid_face_vertex = || ObjError::InvalidFaceVertex {
                        line: line_number,
                        value: (*part).to_owned(),
                    };

                    let mut indices = part.split('/').map(|index| index.parse::<u32>());
                    let mut next_index = || {
                        indices
                            .next()
                            .and_then(Result::ok)
                            .ok_or_else(invalid_face_vertex)
                    };
                    let position_index = next_index()?;
                    let texcoord_index = next_index()?;
                    let normal_index = next_index()?;

                    let vertex_key = (position_index, texcoord_index, normal_index);

                    let vertex_index = if let Some(&index) =
                        current_mesh.vertex_map.get(&vertex_key)
                    {
                        index
                    } else {
                        let out_of_bounds = |attribute, index| ObjError::IndexOutOfBounds {
                            line: line_number,
                            attribute,
                            index,
                        };

                        let position = one_based(&positions, position_index)
                            .ok_or_else(|| out_of_bounds(ObjAttribute::Position, position_index))?;
                        let normal = one_based(&normals, normal_index)
                            .ok_or_else(|| out_of_bounds(ObjAttribute::Normal, normal_index))?;
                        let texcoord = one_based(&texcoords, texcoord_index).ok_or_else(|| {
                            out_of_bounds(ObjAttribute::TextureCoordinate, texcoord_index)
                        })?;

                        let vertex = ObjVertex::new(*position, Colour::WHITE, *normal, *texcoord);
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
                let Some(mtl_path) = parts.get(1) else {
                    return Err(missing_values());
                };
                let mtl_path = std::path::Path::new(mtl_path);
                let mtl_full_path = if let Some(parent) = file_path.as_ref().parent() {
                    parent.join(mtl_path)
                } else {
                    mtl_path.to_path_buf()
                };
                materials =
                    load_mtl(&mtl_full_path).map_err(|source| ObjError::MaterialLibrary {
                        line: line_number,
                        source,
                    })?;

                for (i, material) in materials.iter().enumerate() {
                    material_map.insert(material.name.clone(), i);
                }
            }
            "usemtl" => {
                let Some(&material_name) = parts.get(1) else {
                    return Err(missing_values());
                };

                finish_current_mesh(&mut current_mesh, &mut obj_meshes);

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

/// Looks up an OBJ index, which counts from 1.
fn one_based<T>(values: &[T], index: u32) -> Option<&T> {
    values.get(index.checked_sub(1)? as usize)
}

#[derive(Clone)]
struct ObjMesh {
    vertices: Vec<ObjVertex>,
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

/// The vertex attribute an OBJ face index refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ObjAttribute {
    /// A vertex position, defined by `v`.
    Position,
    /// A texture coordinate, defined by `vt`.
    TextureCoordinate,
    /// A vertex normal, defined by `vn`.
    Normal,
}
impl std::fmt::Display for ObjAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ObjAttribute::Position => "position",
            ObjAttribute::TextureCoordinate => "texture coordinate",
            ObjAttribute::Normal => "normal",
        })
    }
}

/// An error loading an OBJ file with [`load_obj`].
///
/// Line numbers count from 1.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ObjError {
    /// The file couldn't be read.
    #[error("failed to read `{}`", path.display())]
    Io {
        /// The path that was read.
        path: PathBuf,
        /// Why it couldn't be read.
        #[source]
        source: std::io::Error,
    },
    /// A statement has fewer values than its keyword requires.
    #[error("line {line}: `{keyword}` statement has too few values")]
    MissingValues {
        /// The statement's line.
        line: usize,
        /// The statement's keyword.
        keyword: String,
    },
    /// A value that should be a number isn't one.
    #[error("line {line}: invalid number `{value}`")]
    InvalidNumber {
        /// The statement's line.
        line: usize,
        /// The value as written.
        value: String,
        /// Why it isn't a number.
        #[source]
        source: std::num::ParseFloatError,
    },
    /// A face vertex isn't of the form `position/texture coordinate/normal`, with each index a
    /// non-negative integer.
    #[error("line {line}: invalid face vertex `{value}`")]
    InvalidFaceVertex {
        /// The face statement's line.
        line: usize,
        /// The face vertex as written.
        value: String,
    },
    /// A face index refers to an attribute the file hasn't defined.
    #[error("line {line}: {attribute} index {index} is out of bounds")]
    IndexOutOfBounds {
        /// The face statement's line.
        line: usize,
        /// The attribute the index refers to.
        attribute: ObjAttribute,
        /// The index as written, counting from 1.
        index: u32,
    },
    /// The material library named by an `mtllib` statement couldn't be loaded.
    #[error("line {line}: failed to load the material library")]
    MaterialLibrary {
        /// The `mtllib` statement's line.
        line: usize,
        /// Why the library couldn't be loaded.
        #[source]
        source: MtlError,
    },
    /// The loaded meshes and materials don't form a valid model.
    #[error("the loaded model is invalid")]
    Model(#[from] ModelError),
}
impl ObjError {
    fn from_values(error: ValuesError, line: usize, keyword: &str) -> Self {
        match error {
            ValuesError::Missing => ObjError::MissingValues {
                line,
                keyword: keyword.to_owned(),
            },
            ValuesError::InvalidNumber { value, source } => ObjError::InvalidNumber {
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

    fn load_error(contents: &str) -> ObjError {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.obj");
        std::fs::write(&path, contents).unwrap();
        load_obj(&path).err().expect("loading the OBJ should fail")
    }

    #[test]
    fn faces_split_into_triangles_sharing_vertices_with_the_same_indices() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.obj");
        std::fs::write(
            &path,
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 1\nvn 0 0 1\n\
             f 1/1/1 2/1/1 3/2/1 4/2/1\nf 1/1/1 3/1/1 4/2/1\n",
        )
        .unwrap();

        let model = load_obj(&path).unwrap();

        let [mesh] = model.meshes.as_slice() else {
            panic!("expected one mesh, got {}", model.meshes.len());
        };
        assert_eq!(mesh.indices, [0, 1, 2, 0, 2, 3, 0, 4, 3]);
        assert_eq!(mesh.material_index, None);
        let attributes: Vec<_> = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.position, vertex.uv, vertex.normal))
            .collect();
        let normal = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(
            attributes,
            [
                (Vec3::new(0.0, 0.0, 0.0), Vec2::new(0.0, 0.0), normal),
                (Vec3::new(1.0, 0.0, 0.0), Vec2::new(0.0, 0.0), normal),
                (Vec3::new(1.0, 1.0, 0.0), Vec2::new(1.0, 1.0), normal),
                (Vec3::new(0.0, 1.0, 0.0), Vec2::new(1.0, 1.0), normal),
                (Vec3::new(1.0, 1.0, 0.0), Vec2::new(0.0, 0.0), normal),
            ]
        );
    }

    #[test]
    fn usemtl_starts_a_mesh_using_the_named_material_from_the_library() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.obj");
        std::fs::write(
            &path,
            "mtllib materials.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n\
             usemtl blue\nf 1/1/1 2/1/1 3/1/1\nusemtl red\nf 1/1/1 2/1/1 3/1/1\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("materials.mtl"),
            "newmtl red\nKd 1 0 0\nnewmtl blue\nKd 0 0 1\n",
        )
        .unwrap();

        let model = load_obj(&path).unwrap();

        let names: Vec<_> = model.materials.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["red", "blue"]);
        let meshes: Vec<_> = model
            .meshes
            .iter()
            .map(|mesh| (mesh.material_index, mesh.indices.as_slice()))
            .collect();
        assert_eq!(
            meshes,
            [(Some(1), &[0, 1, 2][..]), (Some(0), &[0, 1, 2][..])]
        );
    }

    #[test]
    fn invalid_number_reports_its_line_and_chains_the_parse_error() {
        let error = load_error("v 1 2 3\nv 1 x 3\n");

        assert!(
            matches!(&error, ObjError::InvalidNumber { line: 2, value, .. } if value == "x"),
            "{error:?}"
        );
        assert!(
            error
                .source()
                .is_some_and(|source| source.is::<std::num::ParseFloatError>())
        );
    }

    #[test]
    fn statement_with_too_few_values_reports_its_keyword() {
        let error = load_error("v 1 2 3\nvn 0 1\n");

        assert!(
            matches!(&error, ObjError::MissingValues { line: 2, keyword } if keyword == "vn"),
            "{error:?}"
        );
    }

    #[test]
    fn face_vertex_needs_all_three_indices() {
        let error = load_error("v 0 0 0\nvn 0 0 1\nf 1//1 1//1 1//1\n");

        assert!(
            matches!(&error, ObjError::InvalidFaceVertex { line: 3, value } if value == "1//1"),
            "{error:?}"
        );
    }

    #[test]
    fn face_indices_count_from_one() {
        let error = load_error("v 0 0 0\nvt 0 0\nvn 0 0 1\nf 0/1/1 1/1/1 1/1/1\n");

        assert!(
            matches!(
                error,
                ObjError::IndexOutOfBounds {
                    line: 4,
                    attribute: ObjAttribute::Position,
                    index: 0,
                }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn failed_material_library_reports_its_line_and_chains_the_cause() {
        let error = load_error("# model\nmtllib missing.mtl\n");

        let ObjError::MaterialLibrary { line, source } = &error else {
            panic!("expected a material library error, got {error:?}");
        };
        assert_eq!(*line, 2);
        assert!(matches!(source, MtlError::Io { path, .. } if path.ends_with("missing.mtl")));

        let io_error = error
            .source()
            .and_then(Error::source)
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .expect("the chain ends in the I/O error");
        assert_eq!(io_error.kind(), std::io::ErrorKind::NotFound);
    }
}
