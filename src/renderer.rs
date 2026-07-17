use crate::colour::Colour;
use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::maths::Vec2;
use crate::viewport::Viewport;

pub struct Renderer {
    framebuffer: FrameBuffer,
    depthbuffer: DepthBuffer,
}
impl Renderer {
    pub fn new(viewport: &Viewport) -> Self {
        Self {
            framebuffer: FrameBuffer::new(viewport.width, viewport.height),
            depthbuffer: DepthBuffer::new(viewport.width, viewport.height),
        }
    }

    pub fn clear(&mut self, colour: Colour) {
        self.framebuffer.clear(colour);
        self.depthbuffer.clear();
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
}
