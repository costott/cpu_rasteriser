//! Measuring time between frames.

/// A clock that measures the time since it was created and between successive calls to
/// [`delta`](Timer::delta).
///
/// It uses [`std::time::Instant`], so it is monotonic.
///
/// # Examples
///
/// ```
/// use engine::components::timer::Timer;
///
/// let mut timer = Timer::new();
///
/// for _frame in 0..3 {
///     let dt = timer.delta().as_secs_f32();
///     // ... advance the simulation by `dt` seconds ...
///     # let _ = dt;
/// }
///
/// // Each delta is part of the time since the timer started.
/// let dt = timer.delta();
/// assert!(timer.elapsed() >= dt);
/// ```
#[derive(Debug, Clone)]
pub struct Timer {
    start_time: std::time::Instant,
    previous_time: std::time::Instant,
}
impl Timer {
    /// Creates a timer that starts now.
    ///
    /// # Examples
    ///
    /// See [`Timer`].
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            previous_time: std::time::Instant::now(),
        }
    }

    /// Returns the time since the timer was created.
    ///
    /// # Examples
    ///
    /// See [`Timer`].
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Returns the time since the previous call, or since the timer was created on the first
    /// call, and restarts that interval.
    ///
    /// # Examples
    ///
    /// See [`Timer`].
    pub fn delta(&mut self) -> std::time::Duration {
        let now = std::time::Instant::now();
        let delta = now - self.previous_time;
        self.previous_time = now;
        delta
    }
}
impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
