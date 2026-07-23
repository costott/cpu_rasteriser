use crate::graphics::geometry_processing::GeometryProcessor;
use crate::graphics::lighting::DirectionalLight;
use crate::graphics::vertex;
use crate::prelude::*;

use crate::depthbuffer::DepthBuffer;
use crate::framebuffer::FrameBuffer;
use crate::graphics::camera::Camera;
use crate::graphics::fragment::Fragment;
use crate::graphics::fragment_shader::{FragmentShader, FragmentUniforms};
use crate::graphics::vertex_shader::{VertexShader, VertexUniforms};
use crate::viewport::Viewport;

pub struct Renderer {
    framebuffer: FrameBuffer,
    depthbuffer: DepthBuffer,

    vertex_shader: Box<dyn VertexShader>,
    fragment_shader: Box<dyn FragmentShader>,

    lights: Vec<DirectionalLight>,

    culling_mode: CullingMode,
}
impl Renderer {
    pub fn new(
        viewport: &Viewport,
        vertex_shader: Box<dyn VertexShader>,
        fragment_shader: Box<dyn FragmentShader>,
    ) -> Self {
        Self {
            framebuffer: FrameBuffer::new(viewport.width, viewport.height),
            depthbuffer: DepthBuffer::new(viewport.width, viewport.height),
            vertex_shader,
            fragment_shader,
            lights: vec![],
            culling_mode: CullingMode::None,
        }
    }

    pub fn culling_mode(&self) -> CullingMode {
        self.culling_mode
    }

    pub fn set_culling_mode(&mut self, culling_mode: CullingMode) {
        self.culling_mode = culling_mode;
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

    pub fn draw_model(&mut self, model: &Model, scene: &Scene, viewport: &Viewport) {
        for mesh in &model.meshes {
            let material = mesh.material_index.map(|index| &model.materials[index]);
            let draw_call = DrawCall::new(mesh, material, model.transform.model_matrix());
            self.draw_mesh(&draw_call, scene, viewport);
        }
    }

    pub fn draw_mesh(&mut self, draw_call: &DrawCall, scene: &Scene, viewport: &Viewport) {
        let vertex_uniforms = VertexUniforms {
            lights: scene.lights(),
        };
        let fragment_uniforms = FragmentUniforms {
            camera: scene.camera(),
            lights: scene.lights(),
            material: draw_call.material,
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
                triangle_2d.rasterise(|fragment| {
                    self.shade(fragment, &fragment_uniforms);
                });
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullingMode {
    None,
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
