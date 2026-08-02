pub struct Viewport {
    pub width: usize,
    pub height: usize,
}
impl Viewport {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}
