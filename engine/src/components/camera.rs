use cpu_rasteriser::prelude::*;

use crate::app::AppEvent;
use crate::input::{CameraControlInput, CameraInputState, InputKey};

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
    pub fn update<I: CameraControlInput>(&mut self, camera: &mut Camera, input: &I, dt: f32) {
        if input.is_key_down(InputKey::Left) || input.is_key_down(InputKey::A) {
            self.azimuth -= Self::SPEED * dt;
        }
        if input.is_key_down(InputKey::Right) || input.is_key_down(InputKey::D) {
            self.azimuth += Self::SPEED * dt;
        }
        if input.is_key_down(InputKey::Up) || input.is_key_down(InputKey::W) {
            self.elevation += Self::SPEED * dt;
        }
        if input.is_key_down(InputKey::Down) || input.is_key_down(InputKey::S) {
            self.elevation -= Self::SPEED * dt;
        }

        let y = input.scroll_delta_y();
        if y != 0.0 {
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

    input_state: CameraInputState,
    cursor_grabbed: bool,
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
            input_state: CameraInputState::default(),
            cursor_grabbed: true,
            last_mouse: None,
        }
    }

    /// Consume a normalized application event and update internal controller input state.
    pub fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Key { key, state } => {
                self.input_state.set_key_state(key, state);
            }
            AppEvent::MouseButton { button, state } => {
                self.input_state.set_button_state(button, state);
            }
            AppEvent::MouseMoved { x, y } => {
                self.input_state.set_mouse_position(x, y);
            }
            AppEvent::MouseMotionDelta { dx, dy } => {
                self.input_state.add_mouse_delta(dx, dy);
            }
            AppEvent::MouseWheel { delta_y } => {
                self.input_state.add_scroll_delta_y(delta_y);
            }
            _ => {}
        }
    }

    /// Update camera by using controller-owned event state.
    pub fn update_from_events(&mut self, camera: &mut Camera, dt: f32) {
        if self.cursor_grabbed {
            let input = self.input_state.clone();
            self.update(camera, &input, dt);
        }
        self.input_state.clear_deltas();
    }

    pub fn cursor_grabbed(&self) -> bool {
        self.cursor_grabbed
    }

    /// Set whether this controller should own cursor grab mode.
    pub fn set_cursor_grabbed(&mut self, grabbed: bool) {
        self.cursor_grabbed = grabbed;

        if !grabbed {
            self.input_state.clear_deltas();
            self.last_mouse = None;
        }
    }

    pub fn toggle_cursor_grabbed(&mut self) {
        self.set_cursor_grabbed(!self.cursor_grabbed);
    }

    pub fn update<I: CameraControlInput>(&mut self, camera: &mut Camera, input: &I, dt: f32) {
        self.update_mouse(camera, input);
        self.update_keyboard(camera, input, dt);
    }

    fn update_mouse<I: CameraControlInput>(&mut self, camera: &mut Camera, input: &I) {
        let (dx, dy) = input.mouse_delta();
        if dx != 0.0 || dy != 0.0 {
            self.yaw += dx * Self::MOUSE_SENSITIVITY;
            self.pitch -= dy * Self::MOUSE_SENSITIVITY;

            let limit = std::f32::consts::FRAC_PI_2 - 0.01;
            self.pitch = self.pitch.clamp(-limit, limit);
        } else if let Some((x, y)) = input.mouse_position() {
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

    fn update_keyboard<I: CameraControlInput>(&mut self, camera: &mut Camera, input: &I, dt: f32) {
        let mut forward = (camera.lookat - camera.eye).normalise();
        forward.y = 0.0;
        forward = forward.normalise();

        let mut right = forward.cross(&camera.up).normalise();
        right.y = 0.0;
        right = right.normalise();

        let up = camera.up;

        let speed = Self::MOVEMENT_SPEED * dt;

        if input.is_key_down(InputKey::W) {
            camera.eye += forward * speed;
            camera.lookat += forward * speed;
        }
        if input.is_key_down(InputKey::S) {
            camera.eye -= forward * speed;
            camera.lookat -= forward * speed;
        }
        if input.is_key_down(InputKey::A) {
            camera.eye -= right * speed;
            camera.lookat -= right * speed;
        }
        if input.is_key_down(InputKey::D) {
            camera.eye += right * speed;
            camera.lookat += right * speed;
        }

        if input.is_key_down(InputKey::Space) {
            camera.eye += up * speed;
            camera.lookat += up * speed;
        }
        if input.is_key_down(InputKey::LeftShift) {
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
