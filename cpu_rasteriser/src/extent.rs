#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub width: usize,
    pub height: usize,
}
impl Extent {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}
