use crate::colour::Colour;
use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::fragment::Fragment;
use crate::graphics::fragment_shader::FragmentShader;
use crate::maths::Vec2;
use crate::viewport::Viewport;

pub struct Renderer {
    framebuffer: FrameBuffer,
    depthbuffer: DepthBuffer,
    fragment_shader: Box<dyn FragmentShader>,
}
impl Renderer {
    pub fn new(viewport: &Viewport, fragment_shader: Box<dyn FragmentShader>) -> Self {
        Self {
            framebuffer: FrameBuffer::new(viewport.width, viewport.height),
            depthbuffer: DepthBuffer::new(viewport.width, viewport.height),
            fragment_shader,
        }
    }

    pub fn clear(&mut self, colour: Colour) {
        self.framebuffer.clear(colour);
        self.depthbuffer.clear();
    }

    pub fn shade(&mut self, fragment: Fragment) {
        if let Some(fragment) = self.fragment_shader.shade(fragment) {
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
}
