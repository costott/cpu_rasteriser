use std::sync::Arc;
use threadpool::ThreadPool;

use crate::prelude::*;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::camera::Camera;
use crate::graphics::fragment::Fragment;
use crate::graphics::fragment_shader::{FragmentShader, FragmentUniforms};
use crate::graphics::geometry_processing::GeometryProcessor;
use crate::graphics::lighting::DirectionalLight;
use crate::graphics::vertex_shader::{VertexShader, VertexUniforms};
use crate::viewport::Viewport;

pub struct Renderer {
    framebuffer: FrameBuffer,
    depthbuffer: DepthBuffer,

    vertex_shader: Box<dyn VertexShader>,
    fragment_shader: Arc<dyn FragmentShader + Send + Sync>,

    culling_mode: CullingMode,

    thread_pool: ThreadPool,
    tile_binner: TileBinner,
}
impl Renderer {
    const TILE_SIZE: i32 = 64;

    pub fn new(
        viewport: &Viewport,
        vertex_shader: Box<dyn VertexShader>,
        fragment_shader: Arc<dyn FragmentShader + Send + Sync>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            framebuffer: FrameBuffer::new(viewport.width, viewport.height),
            depthbuffer: DepthBuffer::new(viewport.width, viewport.height),
            vertex_shader,
            fragment_shader,
            culling_mode: CullingMode::None,
            thread_pool: ThreadPool::new(std::thread::available_parallelism()?.get()),
            tile_binner: TileBinner::new(viewport),
        })
    }

    pub fn culling_mode(&self) -> CullingMode {
        self.culling_mode
    }

    pub fn set_culling_mode(&mut self, culling_mode: CullingMode) {
        self.culling_mode = culling_mode;
    }

    /// Resizes the renderer's buffers to match the given viewport.
    pub fn resize(&mut self, viewport: &Viewport) {
        self.framebuffer.resize(viewport.width, viewport.height);
        self.depthbuffer.resize(viewport.width, viewport.height);

        self.tile_binner = TileBinner::new(viewport);
    }

    /// Sets the number of threads used for rendering.
    pub fn set_thread_pool_size(&mut self, size: usize) {
        self.thread_pool = ThreadPool::new(size);
    }

    /// Directly writes a fragment to the frame and depth buffer.
    ///
    /// # Safety / Warning
    /// This bypasses standard pipeline stages. Use `render_scene` instead.
    pub fn write_fragment(&mut self, p: Vec2, colour: Colour, depth: f32) {
        if depth < self.depthbuffer.get(p) {
            self.framebuffer.set_pixel(p, colour);
            self.depthbuffer.set_depth(p, depth);
        }
    }

    /// Returns a reference to the framebuffer's pixel data.
    pub fn pixels(&self) -> &[u32] {
        self.framebuffer.pixels()
    }

    /// Renders an entire scene.
    ///
    /// This performs a complete frame:
    /// - clears the frame and depth buffers
    /// - transforms and clips geometry
    /// - bins triangles into tiles
    /// - rasterises all tiles
    /// - writes the final image to the framebuffer
    ///
    /// This is the recommended entry point for rendering.
    pub fn draw_scene(&mut self, scene: &Scene, viewport: &Viewport) {
        self.begin_frame();

        for model in scene.models() {
            self.draw_model(model, scene, viewport);
        }

        self.submit_frame(scene);
    }

    /// Begins a new frame.
    ///
    /// This clears the framebuffer, depth buffer, and tile bins.
    ///
    /// Must be called before any draw calls.
    pub fn begin_frame(&mut self) {
        self.framebuffer.clear(Colour::BLACK);
        self.depthbuffer.clear();

        self.tile_binner.clear();
    }

    /// Queues a model for rendering.
    ///
    /// Geometry is transformed, clipped and binned into tiles.
    /// Rasterisation does not occur until `submit_frame` is called.
    ///
    /// Requires `begin_frame` to have been called.
    pub fn draw_model(&mut self, model: &Model, scene: &Scene, viewport: &Viewport) {
        for mesh in &model.meshes {
            let material = mesh.material_index.map(|index| &model.materials[index]);
            let draw_call = DrawCall::new(mesh, material, model.transform.model_matrix());
            self.draw_mesh(&draw_call, scene, viewport);
        }
    }

    /// Queues a mesh for rendering.
    ///
    /// Geometry is transformed, clipped and binned into tiles.
    /// Rasterisation does not occur until `submit_frame` is called.
    ///
    /// Requires `begin_frame` to have been called.
    pub fn draw_mesh(&mut self, draw_call: &DrawCall, scene: &Scene, viewport: &Viewport) {
        let vertex_uniforms = VertexUniforms {
            lights: scene.lights(),
        };

        for triangle in draw_call.mesh.triangles() {
            for triangle_2d in GeometryProcessor::process_triangle(
                triangle,
                &*self.vertex_shader,
                &vertex_uniforms,
                draw_call.model_matrix,
                scene.camera(),
                viewport,
                self.culling_mode(),
            ) {
                self.tile_binner
                    .bin_triangle(triangle_2d, draw_call.material);
            }
        }
    }

    /// Rasterises all queued geometry.
    ///
    /// This processes every populated tile and merges the results into the
    /// framebuffer.
    ///
    /// Must be called after all draw calls have completed.
    pub fn submit_frame(&mut self, scene: &Scene) {
        let (tx, rx) = std::sync::mpsc::channel();

        for tile in self.tile_binner.tiles.iter().cloned() {
            let tx = tx.clone();

            let camera = scene.camera.clone();
            let lights = scene.lights().to_vec();

            let fragment_shader = self.fragment_shader.clone();

            self.thread_pool.execute(move || {
                let result = render_tile(tile, &camera, &lights, &*fragment_shader);
                tx.send(result).unwrap();
            });
        }
        drop(tx);

        for result in rx {
            self.merge_tile(result);
        }

        self.thread_pool.join();
    }

    /// Merges the results of a tile render into the main framebuffer.
    fn merge_tile(&mut self, result: TileResult) {
        let tile_width = result.framebuffer.width();

        for y in 0..result.framebuffer.height() {
            for x in 0..tile_width {
                if let Some(colour) = result.framebuffer.get_pixel((x, y).into()) {
                    let screen_position = Vec2::new(
                        result.bounds.min_x as f32 + x as f32,
                        result.bounds.min_y as f32 + y as f32,
                    );

                    self.framebuffer.set_pixel(screen_position, colour);
                }
            }
        }
    }
}

struct TileBinner {
    tiles: Vec<Tile>,
    tiles_x: usize,
    tiles_y: usize,
}
impl TileBinner {
    fn new(viewport: &Viewport) -> Self {
        let tiles_x = viewport.width.div_ceil(Renderer::TILE_SIZE as usize);
        let tiles_y = viewport.height.div_ceil(Renderer::TILE_SIZE as usize);
        let mut tiles = Vec::with_capacity(tiles_x * tiles_y);

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                tiles.push(Tile {
                    bounds: Rect {
                        min_x: (tile_x * Renderer::TILE_SIZE as usize) as i32,
                        min_y: (tile_y * Renderer::TILE_SIZE as usize) as i32,
                        max_x: ((tile_x + 1) * Renderer::TILE_SIZE as usize) as i32,
                        max_y: ((tile_y + 1) * Renderer::TILE_SIZE as usize) as i32,
                    },
                    triangles: Vec::new(),
                });
            }
        }

        Self {
            tiles,
            tiles_x,
            tiles_y,
        }
    }

    fn clear(&mut self) {
        for tile in &mut self.tiles {
            tile.triangles.clear();
        }
    }

    fn bin_triangle(&mut self, triangle: Triangle2D, material: Option<&Material>) {
        // Determine which tiles the triangle overlaps and add it to those
        let (mins, maxs) = triangle.bounding_box();

        let min_tile_x = (mins.x as i32 / Renderer::TILE_SIZE).max(0);
        let min_tile_y = (mins.y as i32 / Renderer::TILE_SIZE).max(0);
        let max_tile_x = (maxs.x as i32 / Renderer::TILE_SIZE).min(self.tiles_x as i32 - 1);
        let max_tile_y = (maxs.y as i32 / Renderer::TILE_SIZE).min(self.tiles_y as i32 - 1);

        for tile_y in min_tile_y..=max_tile_y {
            for tile_x in min_tile_x..=max_tile_x {
                let index = tile_y as usize * self.tiles_x + tile_x as usize;

                if triangle.intersects_rect(self.tiles[index].bounds) {
                    self.tiles[index].triangles.push(TileTriangle {
                        triangle,
                        material: material.cloned(),
                    });
                }
            }
        }
    }
}

fn render_tile(
    tile: Tile,
    camera: &Camera,
    lights: &[DirectionalLight],
    fragment_shader: &dyn FragmentShader,
) -> TileResult {
    let width = (tile.bounds.max_x - tile.bounds.min_x) as usize;
    let height = (tile.bounds.max_y - tile.bounds.min_y) as usize;

    let mut framebuffer = FrameBuffer::new(width, height);
    let mut depthbuffer = DepthBuffer::new(width, height);

    for tile_triangle in tile.triangles {
        let uniforms = FragmentUniforms {
            camera,
            lights,
            material: tile_triangle.material.as_ref(),
        };

        tile_triangle
            .triangle
            .rasterise_segment(tile.bounds, |mut fragment| {
                // convert screen coordinates into tile coordinates
                fragment.position.x -= tile.bounds.min_x as f32;
                fragment.position.y -= tile.bounds.min_y as f32;

                if let Some(fragment) = fragment_shader.shade(fragment, &uniforms) {
                    if fragment.depth < depthbuffer.get(fragment.position) {
                        framebuffer.set_pixel(fragment.position, fragment.colour);

                        depthbuffer.set_depth(fragment.position, fragment.depth);
                    }
                }
            });
    }

    TileResult {
        bounds: tile.bounds,
        framebuffer,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullingMode {
    /// No culling is performed; all triangles are rendered.
    None,
    /// Triangles facing away from the camera are culled.
    BackFace,
}

pub struct DrawCall<'a> {
    pub mesh: &'a Mesh,
    pub material: Option<&'a Material>,
    pub model_matrix: Mat4,
}
impl<'a> DrawCall<'a> {
    pub fn new(mesh: &'a Mesh, material: Option<&'a Material>, model_matrix: Mat4) -> DrawCall<'a> {
        DrawCall {
            mesh,
            material,
            model_matrix,
        }
    }
}

struct TileResult {
    bounds: Rect,
    framebuffer: FrameBuffer,
}

#[derive(Debug, Clone)]
struct Tile {
    pub bounds: Rect,
    pub triangles: Vec<TileTriangle>,
}

#[derive(Debug, Clone)]
struct TileTriangle {
    triangle: Triangle2D,
    material: Option<Material>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}
