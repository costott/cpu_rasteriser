use cpu_rasteriser::prelude::*;

use cpu_rasteriser::graphics::camera::Camera;

/// A simple camera controller that allows the user to orbit around a target point.
///
/// Orbit: WASD or arrow keys
/// Zoom: Mouse scroll wheel
///
/// # Example
/// ```
/// let mut controls = OrbitControls::new(&camera);
///
/// while window.is_open() && !window.is_key_down(Key::Escape) {
///     controls.update(&mut camera, &window, dt);
/// }
/// ```
pub struct OrbitControls {
    pub radius: f32,
    pub azimuth: f32,
    pub elevation: f32,
}
impl OrbitControls {
    const SPEED: f32 = 1.0;
    const ZOOM_SPEED: f32 = 0.5;

    /// Create a new `OrbitControls` instance with the given camera.  
    pub fn new(camera: &Camera) -> Self {
        Self {
            radius: camera.eye.length(),
            azimuth: camera.eye.z.atan2(camera.eye.x),
            elevation: (camera.eye.y / camera.eye.length()).asin(),
        }
    }

    /// Update the camera's position based on the controller's current state.
    pub fn update_camera(&self, camera: &mut Camera) {
        let x = self.radius * self.elevation.cos() * self.azimuth.sin();
        let y = self.radius * self.elevation.sin();
        let z = self.radius * self.elevation.cos() * self.azimuth.cos();

        camera.eye = Vec3::new(x, y, z);
    }

    /// Update the controller's and camera's state based on user input and the elapsed time.
    pub fn update(&mut self, camera: &mut Camera, window: &minifb::Window, dt: f32) {
        if window.is_key_down(minifb::Key::Left) || window.is_key_down(minifb::Key::A) {
            self.azimuth -= Self::SPEED * dt;
        }
        if window.is_key_down(minifb::Key::Right) || window.is_key_down(minifb::Key::D) {
            self.azimuth += Self::SPEED * dt;
        }
        if window.is_key_down(minifb::Key::Up) || window.is_key_down(minifb::Key::W) {
            self.elevation += Self::SPEED * dt;
        }
        if window.is_key_down(minifb::Key::Down) || window.is_key_down(minifb::Key::S) {
            self.elevation -= Self::SPEED * dt;
        }

        if let Some((_, y)) = window.get_scroll_wheel() {
            self.radius -= y * Self::ZOOM_SPEED * dt;
            if self.radius < 0.1 {
                self.radius = 0.1;
            }
        }

        self.update_camera(camera);
    }
}

/// A simple camera controller that allows the user to move the camera in a first-person style.
///
/// Move: WASD or arrow keys
/// Look: Mouse movement
/// Rise/Fall: Space/Shift
pub struct FirstPersonControls {
    pub yaw: f32,
    pub pitch: f32,

    last_mouse: Option<(f32, f32)>,
}
impl FirstPersonControls {
    const MOVEMENT_SPEED: f32 = 2.0;
    const MOUSE_SENSITIVITY: f32 = 0.01;

    pub fn new(camera: &Camera) -> Self {
        let forward = (camera.lookat - camera.eye).normalise();
        let yaw = forward.z.atan2(forward.x);
        let pitch = forward.y.asin();

        Self {
            yaw,
            pitch,
            last_mouse: None,
        }
    }

    pub fn update(&mut self, camera: &mut Camera, window: &minifb::Window, dt: f32) {
        self.update_mouse(camera, window);
        self.update_keyboard(camera, window, dt);
    }

    fn update_mouse(&mut self, camera: &mut Camera, window: &minifb::Window) {
        if let Some((x, y)) = window.get_mouse_pos(minifb::MouseMode::Discard) {
            if let Some((last_x, last_y)) = self.last_mouse {
                let dx = x - last_x;
                let dy = y - last_y;

                self.yaw += dx * Self::MOUSE_SENSITIVITY;
                self.pitch -= dy * Self::MOUSE_SENSITIVITY;

                let limit = std::f32::consts::FRAC_PI_2 - 0.01;
                self.pitch = self.pitch.clamp(-limit, limit);
            }

            self.last_mouse = Some((x, y));
        }

        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalise();

        camera.lookat = camera.eye + forward;
    }

    fn update_keyboard(&mut self, camera: &mut Camera, window: &minifb::Window, dt: f32) {
        let mut forward = (camera.lookat - camera.eye).normalise();
        forward.y = 0.0;
        forward = forward.normalise();

        let mut right = forward.cross(&camera.up).normalise();
        right.y = 0.0;
        right = right.normalise();

        let up = camera.up;

        let speed = Self::MOVEMENT_SPEED * dt;

        if window.is_key_down(minifb::Key::W) {
            camera.eye += forward * speed;
            camera.lookat += forward * speed;
        }
        if window.is_key_down(minifb::Key::S) {
            camera.eye -= forward * speed;
            camera.lookat -= forward * speed;
        }
        if window.is_key_down(minifb::Key::A) {
            camera.eye -= right * speed;
            camera.lookat -= right * speed;
        }
        if window.is_key_down(minifb::Key::D) {
            camera.eye += right * speed;
            camera.lookat += right * speed;
        }

        if window.is_key_down(minifb::Key::Space) {
            camera.eye += up * speed;
            camera.lookat += up * speed;
        }
        if window.is_key_down(minifb::Key::LeftShift) {
            camera.eye -= up * speed;
            camera.lookat -= up * speed;
        }

        camera.lookat = camera.eye
            + Vec3::new(
                self.yaw.cos() * self.pitch.cos(),
                self.pitch.sin(),
                self.yaw.sin() * self.pitch.cos(),
            );
    }
}
