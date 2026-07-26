pub struct Timer {
    start_time: std::time::Instant,
    previous_time: std::time::Instant,
}
impl Timer {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            previous_time: std::time::Instant::now(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    pub fn delta(&mut self) -> std::time::Duration {
        let now = std::time::Instant::now();
        let delta = now - self.previous_time;
        self.previous_time = now;
        delta
    }
}
