use crate::prelude::*;

#[derive(Debug, Clone, Copy)]
enum ClipPlane {
    Left,
    Right,
    Top,
    Bottom,
    Near,
    Far,
}

const PLANES: [ClipPlane; 6] = [
    ClipPlane::Left,
    ClipPlane::Right,
    ClipPlane::Top,
    ClipPlane::Bottom,
    ClipPlane::Near,
    ClipPlane::Far,
];

fn signed_distance<V: Interpolate>(v: &ClipVertex<V>, plane: ClipPlane) -> f32 {
    match plane {
        ClipPlane::Left => v.position.x + v.position.w,
        ClipPlane::Right => -v.position.x + v.position.w,
        ClipPlane::Top => -v.position.y + v.position.w,
        ClipPlane::Bottom => v.position.y + v.position.w,
        ClipPlane::Near => v.position.z + v.position.w,
        ClipPlane::Far => -v.position.z + v.position.w,
    }
}

fn intersection<V: Interpolate>(
    a: ClipVertex<V>,
    b: ClipVertex<V>,
    plane: ClipPlane,
) -> ClipVertex<V> {
    let da = signed_distance(&a, plane);
    let db = signed_distance(&b, plane);

    let t = da / (da - db);

    a.interpolate(&b, t)
}

fn clip_triangle_against_plane<V: Interpolate>(
    triangle: TriangleClip<V>,
    plane: ClipPlane,
) -> Vec<TriangleClip<V>> {
    let vertices = [triangle.a, triangle.b, triangle.c];

    let mut output = Vec::new();

    for i in 0..3 {
        let current = vertices[i].clone();
        let next = vertices[(i + 1) % 3].clone();

        let current_inside = signed_distance(&current, plane) >= 0.0;
        let next_inside = signed_distance(&next, plane) >= 0.0;

        match (current_inside, next_inside) {
            // Inside -> Inside
            (true, true) => {
                output.push(next);
            }

            // Inside -> Outside
            (true, false) => {
                output.push(intersection(current, next, plane));
            }

            // Outside -> Inside
            (false, true) => {
                output.push(intersection(current, next.clone(), plane));
                output.push(next);
            }

            // Outside -> Outside
            (false, false) => {}
        }
    }

    triangulate(output)
}

/// Sutherland-Hodgman polygon clipping algorithm for triangles against the view frustum defined by the clip planes.
pub fn clip_triangle<V: Interpolate>(triangle: TriangleClip<V>) -> Vec<TriangleClip<V>> {
    let mut triangles = vec![triangle];

    for plane in PLANES.iter() {
        triangles = triangles
            .into_iter()
            .flat_map(|triangle| clip_triangle_against_plane(triangle, *plane))
            .collect();
    }

    triangles
}

/// Triangulates a polygon represented by a list of vertices into triangles.
/// Clip result for a triangle can have either 0, 3, or 4 vertices. If it has 4 vertices, it will be split into 2 triangles.
fn triangulate<V: Interpolate>(vertices: Vec<ClipVertex<V>>) -> Vec<TriangleClip<V>> {
    match vertices.len() {
        3 => vec![TriangleClip::new(
            vertices[0].clone(),
            vertices[1].clone(),
            vertices[2].clone(),
        )],

        4 => vec![
            TriangleClip::new(
                vertices[0].clone(),
                vertices[1].clone(),
                vertices[2].clone(),
            ),
            TriangleClip::new(
                vertices[0].clone(),
                vertices[2].clone(),
                vertices[3].clone(),
            ),
        ],

        _ => vec![],
    }
}
