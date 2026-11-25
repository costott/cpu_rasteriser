use criterion::criterion_main;

mod binning;
mod blend;
mod buffers;
mod colour;
mod fixtures;
mod geometry;
mod interpolate;
mod maths;
mod output;
mod rasterise;

criterion_main!(
    binning::benches,
    blend::benches,
    buffers::benches,
    colour::benches,
    geometry::benches,
    interpolate::benches,
    maths::benches,
    output::benches,
    rasterise::benches,
);
