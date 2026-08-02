use cpu_rasteriser::prelude::*;

#[derive(Debug, Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub lookat: Vec3,
    pub up: Vec3,
    pub projection: Projection,
}
impl Camera {
    pub fn new(eye: Vec3, lookat: Vec3, up: Vec3, projection: Projection) -> Self {
        Self {
            eye,
            lookat,
            up,
            projection,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let n = (self.eye - self.lookat).normalise();
        let u = self.up.cross(&n).normalise();
        let v = n.cross(&u);

        Mat4::new([
            [u.x, u.y, u.z, -u.dot(&self.eye)],
            [v.x, v.y, v.z, -v.dot(&self.eye)],
            [n.x, n.y, n.z, -n.dot(&self.eye)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn projection_matrix(&self) -> Mat4 {
        self.projection.matrix()
    }
}

#[derive(Debug, Clone)]
pub enum Projection {
    Perspective(PerspectiveProjection),
    Orthographic(OrthographicProjection),
}
impl Projection {
    pub fn matrix(&self) -> Mat4 {
        match self {
            Projection::Perspective(perspective) => perspective.matrix(),
            Projection::Orthographic(orthographic) => orthographic.matrix(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrthographicProjection {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
    pub near: f32,
    pub far: f32,
}
impl OrthographicProjection {
    pub fn new(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Self {
            left,
            right,
            bottom,
            top,
            near,
            far,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::orthographic(
            self.left,
            self.right,
            self.bottom,
            self.top,
            self.near,
            self.far,
        )
    }
}

#[derive(Debug, Clone)]
pub struct PerspectiveProjection {
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}
impl PerspectiveProjection {
    pub fn new(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        Self {
            fov,
            aspect_ratio,
            near,
            far,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::perspective(self.fov, self.aspect_ratio, self.near, self.far)
    }
}

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
        if let Some((x, y)) = window.get_mouse_pos(minifb::MouseMode::Pass) {
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
