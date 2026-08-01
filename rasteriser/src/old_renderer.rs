use std::sync::Arc;
use threadpool::ThreadPool;

use crate::prelude::*;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::fragment_shader::FragmentShader;
use crate::graphics::geometry_processing::GeometryProcessor;
use crate::graphics::vertex_shader::VertexShader;
use crate::viewport::Viewport;

/// A CPU renderer that executes a programmable graphics pipeline.
///
/// `Renderer` is responsible for transforming vertices, rasterising primitives,
/// executing fragment shaders, and writing the final image to the framebuffer.
///
/// Rendering is performed in three stages:
///
/// 1. Call [`Renderer::begin_frame`] to clear the framebuffer and prepare a new frame.
/// 2. Submit one or more [`DrawCall`]s using [`Renderer::submit_draw_call`].
/// 3. Call [`Renderer::submit_frame`] to rasterise all queued geometry.
///
/// Each draw call may provide different fragment shader uniforms, allowing
/// multiple objects with different materials or rendering parameters to be
/// rendered in a single frame.
///
/// The renderer internally uses tile-based rasterisation and multithreading
/// to improve CPU cache locality and rendering performance. These implementation
/// details are entirely transparent to users of the API.
///
/// # Type Parameters
///
/// - `VS` — The vertex shader used to transform input vertices.
/// - `FS` — The fragment shader used to shade rasterised fragments.
///
/// # Example
///
/// ```ignore
/// # use cpu_rasteriser::prelude::*;
/// #
/// let mut renderer = Renderer::new(
///     &viewport,
///     vertex_shader,
///     fragment_shader,
/// )?;
///
/// renderer.begin_frame();
///
/// renderer.submit_draw_call(
///     draw_call,
///     &vertex_uniforms,
///     &viewport,
/// );
///
/// renderer.submit_frame();
///
/// let pixels = renderer.pixels();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Renderer<VS, FS>
where
    VS: VertexShader,
    FS: FragmentShader<VS::Varyings>,
{
    framebuffer: FrameBuffer,
    depthbuffer: DepthBuffer,

    vertex_shader: VS,
    fragment_shader: Arc<FS>,

    culling_mode: CullingMode,

    thread_pool: ThreadPool,
    tile_binner: TileBinner<VS::Varyings, FS::Uniforms>,
}
impl<VS, FS> Renderer<VS, FS>
where
    VS: VertexShader,
    VS::Varyings: Interpolate + Send + Sync + 'static,
    FS: FragmentShader<VS::Varyings> + Send + Sync + 'static,
    FS::Uniforms: Send + Sync + 'static,
{
    pub fn new(
        viewport: &Viewport,
        vertex_shader: VS,
        fragment_shader: FS,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            framebuffer: FrameBuffer::new(viewport.width, viewport.height),
            depthbuffer: DepthBuffer::new(viewport.width, viewport.height),
            vertex_shader,
            fragment_shader: Arc::new(fragment_shader),
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

    /// Begins rendering a new frame.
    ///
    /// This clears the framebuffer, depth buffer, and any previously queued draw
    /// calls, preparing the renderer for a new frame.
    ///
    /// This method must be called before submitting any draw calls.
    ///
    /// # Rendering Order
    ///
    /// A typical frame is rendered as:
    ///
    /// ```text
    /// begin_frame()
    /// submit_draw_call(...)
    /// submit_draw_call(...)
    /// ...
    /// submit_frame()
    /// ```
    pub fn begin_frame(&mut self) {
        self.framebuffer.clear(Colour::BLACK);
        self.depthbuffer.clear();

        self.tile_binner.clear();
    }

    /// Queues a draw call for rendering.
    ///
    /// The primitive is transformed by the vertex shader, clipped against the view
    /// frustum, and binned into screen-space tiles. Rasterisation is deferred until
    /// [`Renderer::submit_frame`] is called.
    ///
    /// Multiple draw calls may be submitted between
    /// [`Renderer::begin_frame`] and [`Renderer::submit_frame`], allowing a frame
    /// to contain many independently shaded objects.
    ///
    /// Behaviour is unspecified if called before [`Renderer::begin_frame`].
    pub fn submit_draw_call(
        &mut self,
        draw_call: DrawCall<VS::Vertex, FS::Uniforms>,
        vertex_uniforms: &VS::Uniforms,
        viewport: &Viewport,
    ) {
        match draw_call.primitive_mode() {
            PrimitiveMode::TRIANGLES => {
                let fragment_uniforms = Arc::new(draw_call.fragment_uniforms);

                for triangle in draw_call.primitive.triangles() {
                    for triangle_2d in GeometryProcessor::process_triangle(
                        triangle,
                        &self.vertex_shader,
                        vertex_uniforms,
                        viewport,
                        self.culling_mode(),
                    ) {
                        self.tile_binner
                            .bin_triangle(triangle_2d, fragment_uniforms.clone());
                    }
                }
            }
        }
    }

    /// Rasterises all queued draw calls and produces the final image.
    ///
    /// This executes the fragment shader for every visible fragment, performs depth
    /// testing, and writes the resulting pixels into the framebuffer.
    ///
    /// After this method returns, the rendered image can be accessed through
    /// [`Renderer::pixels`].
    ///
    /// This method should be called once after all draw calls for the frame have
    /// been submitted.
    ///
    /// # Rendering Order
    ///
    /// ```text
    /// begin_frame()
    /// submit_draw_call(...)
    /// submit_draw_call(...)
    /// ...
    /// submit_frame()
    /// ```
    pub fn submit_frame(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();

        for tile in self.tile_binner.tiles.iter().cloned() {
            let tx = tx.clone();

            let fragment_shader = self.fragment_shader.clone();

            self.thread_pool.execute(move || {
                let result = render_tile(tile, fragment_shader);
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

/// Tile binning is a technique used to improve cache locality and parallelism in rasterisation.
///
/// It works by dividing the screen into a grid of tiles, and then determining which triangles overlap each tile.
/// Each tile can then be processed independently, allowing for better use of CPU caches and easier parallelisation
/// across multiple threads.
struct TileBinner<V, U>
where
    V: Interpolate,
{
    tiles: Vec<Tile<V, U>>,
    tiles_x: usize,
    tiles_y: usize,
}
impl<V: Interpolate, U> TileBinner<V, U> {
    const TILE_SIZE: i32 = 64;

    fn new(viewport: &Viewport) -> Self {
        let tiles_x = viewport.width.div_ceil(Self::TILE_SIZE as usize);
        let tiles_y = viewport.height.div_ceil(Self::TILE_SIZE as usize);
        let mut tiles = Vec::with_capacity(tiles_x * tiles_y);

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                tiles.push(Tile {
                    bounds: Rect {
                        min_x: (tile_x * Self::TILE_SIZE as usize) as i32,
                        min_y: (tile_y * Self::TILE_SIZE as usize) as i32,
                        max_x: ((tile_x + 1) * Self::TILE_SIZE as usize) as i32,
                        max_y: ((tile_y + 1) * Self::TILE_SIZE as usize) as i32,
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

    fn bin_triangle(&mut self, triangle: Triangle2D<V>, fragment_uniforms: Arc<U>) {
        // Determine which tiles the triangle overlaps and add it to those
        let (mins, maxs) = triangle.bounding_box();

        let min_tile_x = (mins.x as i32 / Self::TILE_SIZE).max(0);
        let min_tile_y = (mins.y as i32 / Self::TILE_SIZE).max(0);
        let max_tile_x = (maxs.x as i32 / Self::TILE_SIZE).min(self.tiles_x as i32 - 1);
        let max_tile_y = (maxs.y as i32 / Self::TILE_SIZE).min(self.tiles_y as i32 - 1);

        for tile_y in min_tile_y..=max_tile_y {
            for tile_x in min_tile_x..=max_tile_x {
                let index = tile_y as usize * self.tiles_x + tile_x as usize;

                if triangle.intersects_rect(self.tiles[index].bounds) {
                    self.tiles[index].triangles.push(TileTriangle {
                        triangle: triangle.clone(),
                        fragment_uniforms: fragment_uniforms.clone(),
                    });
                }
            }
        }
    }
}

/// Renders a single tile, returning the resulting framebuffer.
fn render_tile<V, FS>(tile: Tile<V, FS::Uniforms>, fragment_shader: Arc<FS>) -> TileResult
where
    V: Interpolate + Send + Sync + 'static,
    FS: FragmentShader<V> + Send + Sync + 'static,
    FS::Uniforms: Send + Sync + 'static,
{
    let width = (tile.bounds.max_x - tile.bounds.min_x) as usize;
    let height = (tile.bounds.max_y - tile.bounds.min_y) as usize;

    let mut framebuffer = FrameBuffer::new(width, height);
    let mut depthbuffer = DepthBuffer::new(width, height);

    for tile_triangle in tile.triangles {
        tile_triangle
            .triangle
            .rasterise_segment(tile.bounds, |mut fragment| {
                // convert screen coordinates into tile coordinates
                fragment.position.x -= tile.bounds.min_x as f32;
                fragment.position.y -= tile.bounds.min_y as f32;

                let frag_colour = fragment_shader
                    .shade(fragment.varyings, tile_triangle.fragment_uniforms.as_ref());

                if fragment.depth < depthbuffer.get(fragment.position) {
                    framebuffer.set_pixel(fragment.position, frag_colour);
                    depthbuffer.set_depth(fragment.position, fragment.depth);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveMode {
    TRIANGLES,
    // TODO: support other primitive modes (LINES, POINTS, etc.)
}

/// A primitive is a collection of vertices and indices that define a geometric shape.
pub struct Primitive<'a, V>
where
    V: Clone,
{
    pub vertices: &'a [V],
    pub indices: &'a [u32],
    pub primitive_mode: PrimitiveMode,
}
impl<'a, V> Primitive<'a, V>
where
    V: Clone,
{
    pub fn new(
        vertices: &'a [V],
        indices: &'a [u32],
        primitive_mode: PrimitiveMode,
    ) -> Primitive<'a, V> {
        Primitive {
            vertices,
            indices,
            primitive_mode,
        }
    }

    fn triangles(&self) -> impl Iterator<Item = Triangle3D<V>> {
        self.indices.chunks_exact(3).map(|indices| Triangle3D {
            a: self.vertices[indices[0] as usize].clone(),
            b: self.vertices[indices[1] as usize].clone(),
            c: self.vertices[indices[2] as usize].clone(),
        })
    }
}

/// A draw call is a request to render a primitive with specific uniforms.
pub struct DrawCall<'a, V, U>
where
    V: Clone,
{
    pub primitive: Primitive<'a, V>,
    pub fragment_uniforms: U,
}
impl<'a, V, U> DrawCall<'a, V, U>
where
    V: Clone,
{
    pub fn new(
        vertices: &'a [V],
        indices: &'a [u32],
        primitive_mode: PrimitiveMode,
        fragment_uniforms: U,
    ) -> DrawCall<'a, V, U> {
        DrawCall {
            primitive: Primitive::new(vertices, indices, primitive_mode),
            fragment_uniforms,
        }
    }

    fn primitive_mode(&self) -> PrimitiveMode {
        self.primitive.primitive_mode
    }
}

/// A tile result is the output of rendering a single tile, used for merging into the main framebuffer.
struct TileResult {
    bounds: Rect,
    framebuffer: FrameBuffer,
}

/// A tile is a rectangular region of the screen that contains a list of triangles to be rendered.
struct Tile<V, U>
where
    V: Interpolate,
{
    pub bounds: Rect,
    pub triangles: Vec<TileTriangle<V, U>>,
}
impl<V, U> Clone for Tile<V, U>
where
    V: Interpolate,
{
    fn clone(&self) -> Self {
        Self {
            bounds: self.bounds,
            triangles: self.triangles.clone(),
        }
    }
}

/// A tile triangle is a triangle that has been assigned to a specific tile for rendering.
struct TileTriangle<V, U>
where
    V: Interpolate,
{
    triangle: Triangle2D<V>,
    fragment_uniforms: Arc<U>,
}
impl<V, U> Clone for TileTriangle<V, U>
where
    V: Interpolate,
{
    fn clone(&self) -> Self {
        Self {
            triangle: self.triangle.clone(),
            fragment_uniforms: Arc::clone(&self.fragment_uniforms),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}
