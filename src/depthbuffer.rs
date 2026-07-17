use crate::maths::Vec2;

pub struct DepthBuffer {
    width: usize,
    height: usize,
    buffer: Vec<f32>,
}
impl DepthBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![1.0; width * height],
        }
    }

    pub fn clear(&mut self) {
        self.buffer.fill(1.0);
    }

    pub fn set_depth(&mut self, p: Vec2, depth: f32) {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return;
        }
        let index = (y as usize) * self.width + (x as usize);
        self.buffer[index] = depth;
    }

    pub fn get(&self, p: Vec2) -> f32 {
        let x = p.x as i32;
        let y = p.y as i32;
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return 1.0;
        }
        let index = (y as usize) * self.width + (x as usize);
        self.buffer[index]
    }
}
