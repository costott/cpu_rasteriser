use crate::prelude::*;

/// Returns true if the vertex is inside the near plane (z + w >= 0)
fn inside_near(vertex: ClipVertex) -> bool {
    vertex.position.z + vertex.position.w >= 0.0
}

/// Returns the intersection point of the line segment between two vertices and the near plane.
/// Calculated as the point where the line segment intersects the plane defined by z + w = 0.
fn intersection_near(a: ClipVertex, b: ClipVertex) -> ClipVertex {
    let da = a.position.z + a.position.w;
    let db = b.position.z + b.position.w;

    let t = da / (da - db);

    a.lerp(&b, t)
}

/// Clips a triangle against the near plane and returns a list of resulting triangles.
/// Uses the Sutherland-Hodgman algorithm to clip the triangle against the near plane.
pub fn clip_triangle_near(triangle: TriangleClip) -> Vec<TriangleClip> {
    let vertices = [triangle.a, triangle.b, triangle.c];

    let mut output = Vec::new();

    for i in 0..3 {
        let current = vertices[i];
        let next = vertices[(i + 1) % 3];

        let current_inside = inside_near(current);
        let next_inside = inside_near(next);

        match (current_inside, next_inside) {
            // Inside -> Inside
            (true, true) => {
                output.push(next);
            }

            // Inside -> Outside
            (true, false) => {
                output.push(intersection_near(current, next));
            }

            // Outside -> Inside
            (false, true) => {
                output.push(intersection_near(current, next));
                output.push(next);
            }

            // Outside -> Outside
            (false, false) => {}
        }
    }

    triangulate(output)
}

/// Triangulates a polygon represented by a list of vertices into triangles.
/// Clip result for a triangle can have either 0, 3, or 4 vertices. If it has 4 vertices, it will be split into 2 triangles.
fn triangulate(vertices: Vec<ClipVertex>) -> Vec<TriangleClip> {
    match vertices.len() {
        3 => vec![TriangleClip::new(vertices[0], vertices[1], vertices[2])],

        4 => vec![
            TriangleClip::new(vertices[0], vertices[1], vertices[2]),
            TriangleClip::new(vertices[0], vertices[2], vertices[3]),
        ],

        _ => vec![],
    }
}
