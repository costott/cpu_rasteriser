use std::sync::Arc;
use threadpool::ThreadPool;

use crate::graphics::camera::Camera;
use crate::prelude::*;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
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

    lights: Vec<DirectionalLight>,

    culling_mode: CullingMode,

    thread_pool: ThreadPool,
    tiles: Vec<Tile>,
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
            lights: vec![],
            culling_mode: CullingMode::None,
            thread_pool: ThreadPool::new(std::thread::available_parallelism()?.get()),
            tiles: vec![],
        })
    }

    pub fn culling_mode(&self) -> CullingMode {
        self.culling_mode
    }

    pub fn set_culling_mode(&mut self, culling_mode: CullingMode) {
        self.culling_mode = culling_mode;
    }

    pub fn set_thread_pool_size(&mut self, size: usize) {
        self.thread_pool = ThreadPool::new(size);
    }

    pub fn add_light(&mut self, light: DirectionalLight) {
        self.lights.push(light);
    }

    pub fn clear(&mut self, colour: Colour) {
        self.framebuffer.clear(colour);
        self.depthbuffer.clear();
    }

    pub fn shade(&mut self, fragment: Fragment, uniforms: &FragmentUniforms) {
        if let Some(fragment) = self.fragment_shader.shade(fragment, uniforms) {
            self.write_fragment(fragment.position, fragment.colour, fragment.depth);
        }
    }

    pub fn write_fragment(&mut self, p: Vec2, colour: Colour, depth: f32) {
        if depth < self.depthbuffer.get(p) {
            self.framebuffer.set_pixel(p, colour);
            self.depthbuffer.set_depth(p, depth);
        }
    }

    pub fn pixels(&self) -> &[u32] {
        self.framebuffer.pixels()
    }

    pub fn begin_frame(&mut self) {
        self.framebuffer.clear(Colour::BLACK);
        self.depthbuffer.clear();
    }

    pub fn draw_model(&mut self, model: &Model, scene: &Scene, viewport: &Viewport) {
        for mesh in &model.meshes {
            if let Some(material) = model.materials.get(mesh.material_index) {
                let draw_call = DrawCall::new(mesh, material, model.transform.model_matrix());
                self.draw_mesh(&draw_call, scene, viewport);
            }
        }
    }

    pub fn draw_mesh(&mut self, draw_call: &DrawCall, scene: &Scene, viewport: &Viewport) {
        let vertex_uniforms = VertexUniforms {
            lights: scene.lights(),
        };
        // let fragment_uniforms = FragmentUniforms {
        //     camera: scene.camera(),
        //     lights: scene.lights(),
        //     material: draw_call.material,
        // };

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
                // triangle_2d.rasterise(|fragment| {
                //     self.shade(fragment, &fragment_uniforms);
                // });
                self.bin_triangle(triangle_2d, *draw_call.material);
            }
        }
    }

    fn bin_triangle(&mut self, triangle: Triangle2D, material: Material) {
        // Determine which tiles the triangle overlaps and add it to those
        let (mins, maxs) = triangle.bounding_box();

        let min_tile_x = mins.x as i32 / Self::TILE_SIZE;
        let min_tile_y = mins.y as i32 / Self::TILE_SIZE;
        let max_tile_x = maxs.x as i32 / Self::TILE_SIZE;
        let max_tile_y = maxs.y as i32 / Self::TILE_SIZE;

        for tile_y in min_tile_y..=max_tile_y {
            for tile_x in min_tile_x..=max_tile_x {
                let tile_bounds = Rect {
                    min_x: tile_x * Self::TILE_SIZE,
                    min_y: tile_y * Self::TILE_SIZE,
                    max_x: (tile_x + 1) * Self::TILE_SIZE,
                    max_y: (tile_y + 1) * Self::TILE_SIZE,
                };

                if triangle.intersects_rect(tile_bounds) {
                    if let Some(tile) = self.tiles.iter_mut().find(|t| t.bounds == tile_bounds) {
                        tile.triangles.push(TileTriangle { triangle, material });
                    } else {
                        self.tiles.push(Tile {
                            bounds: tile_bounds,
                            triangles: vec![TileTriangle {
                                triangle,
                                material: material,
                            }],
                        });
                    }
                }
            }
        }
    }

    pub fn finish_frame(&mut self, scene: &Scene) {
        let (tx, rx) = std::sync::mpsc::channel();

        for tile in self.tiles.drain(..) {
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
            material: &tile_triangle.material,
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
    None,
    BackFace,
}

pub struct DrawCall<'a> {
    pub mesh: &'a Mesh,
    pub material: &'a Material,
    pub model_matrix: Mat4,
}
impl<'a> DrawCall<'a> {
    pub fn new(mesh: &'a Mesh, material: &'a Material, model_matrix: Mat4) -> DrawCall<'a> {
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

struct Tile {
    pub bounds: Rect,
    pub triangles: Vec<TileTriangle>,
}

#[derive(Clone)]
struct TileTriangle {
    triangle: Triangle2D,
    material: Material,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}
