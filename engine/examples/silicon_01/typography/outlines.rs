//! Traces glyph outlines in the typeface's distance field, as closed loops of straight segments
//! for letters to be drawn with.

use std::collections::BTreeMap;

use cpu_rasteriser::prelude::*;

/// The distance field's level on a glyph's outline.
const OUTLINE: f32 = 0.5;

/// A closed run of straight segments: each point joins the next, and the last joins the first.
pub type Loop = Vec<Vec2>;

/// An edge between neighbouring grid points: from `(x, y)` to the point right of it, or below it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    x: usize,
    y: usize,
    down: bool,
}

/// The outlines in a grid of distance-field levels, `width` points across and `height` down,
/// whose border lies outside every outline. Points are in grid units, y downwards. Each loop
/// strays from the field's outline by at most about `tolerance`, starts at its leftmost point and
/// runs clockwise as seen on screen.
pub fn trace(
    width: usize,
    height: usize,
    level: impl Fn(usize, usize) -> f32,
    tolerance: f32,
) -> Vec<Loop> {
    let mut neighbours = segments(width, height, &level);
    let crossing = |edge: Edge| {
        let (x, y) = if edge.down {
            (edge.x, edge.y + 1)
        } else {
            (edge.x + 1, edge.y)
        };
        let (from, to) = (level(edge.x, edge.y), level(x, y));
        let along = (OUTLINE - from) / (to - from);
        Vec2::new(
            edge.x as f32 + (x - edge.x) as f32 * along,
            edge.y as f32 + (y - edge.y) as f32 * along,
        )
    };

    let mut loops = Vec::new();
    while let Some((start, ends)) = neighbours.pop_first() {
        let mut points = vec![crossing(start)];
        let (mut previous, mut edge) = (start, ends[0]);
        while edge != start {
            points.push(crossing(edge));
            let Some(ends) = neighbours.remove(&edge) else {
                break;
            };
            let next = if ends[0] == previous {
                ends[1]
            } else {
                ends[0]
            };
            (previous, edge) = (edge, next);
        }
        let outline = simplify(&points, tolerance);
        if outline.len() >= 3 {
            loops.push(oriented(outline));
        }
    }
    loops
}

/// Marching squares: for every edge the outline crosses, the two edges it carries on to, one
/// through each cell the edge borders.
fn segments(
    width: usize,
    height: usize,
    level: &impl Fn(usize, usize) -> f32,
) -> BTreeMap<Edge, Vec<Edge>> {
    let mut neighbours: BTreeMap<Edge, Vec<Edge>> = BTreeMap::new();
    let mut join = |a: Edge, b: Edge| {
        neighbours.entry(a).or_default().push(b);
        neighbours.entry(b).or_default().push(a);
    };
    for y in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            // Corners clockwise from the top left, and the edges from each to the next.
            let corners =
                [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)].map(|(x, y)| level(x, y));
            let inside = corners.map(|corner| corner > OUTLINE);
            let edge = |x, y, down| Edge { x, y, down };
            let [top, right, bottom, left] = [
                edge(x, y, false),
                edge(x + 1, y, true),
                edge(x, y + 1, false),
                edge(x, y, true),
            ];
            let crossed: Vec<Edge> = [top, right, bottom, left]
                .into_iter()
                .enumerate()
                .filter(|&(i, _)| inside[i] != inside[(i + 1) % 4])
                .map(|(_, edge)| edge)
                .collect();
            match crossed[..] {
                [a, b] => join(a, b),
                [_, _, _, _] => {
                    // A saddle: the middle decides which pair of opposite corners connects.
                    let middle = corners.iter().sum::<f32>() * 0.25 > OUTLINE;
                    if middle == inside[0] {
                        join(top, right);
                        join(bottom, left);
                    } else {
                        join(top, left);
                        join(right, bottom);
                    }
                }
                _ => {}
            }
        }
    }
    neighbours
}

/// Douglas–Peucker on a closed loop, split at its first point and the point furthest from it.
fn simplify(points: &[Vec2], tolerance: f32) -> Loop {
    let Some(&first) = points.first() else {
        return Vec::new();
    };
    let far = (0..points.len())
        .max_by(|&a, &b| {
            (points[a] - first)
                .length()
                .total_cmp(&(points[b] - first).length())
        })
        .unwrap_or(0);
    let mut back = points[far..].to_vec();
    back.push(first);
    let mut kept = Vec::new();
    simplify_run(&points[..=far], tolerance, &mut kept);
    simplify_run(&back, tolerance, &mut kept);
    kept
}

/// Appends to `kept` the points of an open run that Douglas–Peucker keeps, without its last.
fn simplify_run(run: &[Vec2], tolerance: f32, kept: &mut Loop) {
    let (Some(&start), Some(&end)) = (run.first(), run.last()) else {
        return;
    };
    let furthest = (1..run.len().saturating_sub(1))
        .map(|i| (i, distance_to_segment(run[i], start, end)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    match furthest {
        Some((split, distance)) if distance > tolerance => {
            simplify_run(&run[..=split], tolerance, kept);
            simplify_run(&run[split..], tolerance, kept);
        }
        _ => kept.push(start),
    }
}

fn distance_to_segment(point: Vec2, start: Vec2, end: Vec2) -> f32 {
    let (along, offset) = (end - start, point - start);
    let length_squared = along.x * along.x + along.y * along.y;
    let nearest = if length_squared > 0.0 {
        ((offset.x * along.x + offset.y * along.y) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (offset - along * nearest).length()
}

/// The loop started from its leftmost point (the lowest of them) and wound clockwise on screen,
/// so that drawing it sets off upwards.
fn oriented(mut points: Loop) -> Loop {
    // Twice the signed area: positive for a clockwise loop when y runs downwards.
    let area: f32 = (0..points.len())
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            a.x * b.y - b.x * a.y
        })
        .sum();
    if area < 0.0 {
        points.reverse();
    }
    let leftmost = (0..points.len())
        .min_by(|&a, &b| {
            let (a, b) = (points[a], points[b]);
            a.x.total_cmp(&b.x).then(b.y.total_cmp(&a.y))
        })
        .unwrap_or(0);
    points.rotate_left(leftmost);
    points
}
