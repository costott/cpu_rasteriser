//! Cameras, their projections, and controllers that move a camera from user input.
//!
//! A [`Camera`] gives the view and projection matrices a vertex shader needs to take world-space
//! positions to clip space. It builds them with [`Mat4::look_at`], [`Mat4::perspective`], and
//! [`Mat4::orthographic`], which define the view-space and projection conventions. Angles are in
//! radians.
//!
//! [`OrbitControls`] and [`FirstPersonControls`] update a camera from [`AppEvent`]s.

use cpu_rasteriser::prelude::*;

use crate::app::AppEvent;
use crate::input::{CameraControlInput, InputKey, InputState};

/// A camera: where it is, what it looks at, and how it projects the scene.
///
/// The view looks from [`eye`](Camera::eye) towards [`lookat`](Camera::lookat), with
/// [`up`](Camera::up) giving the vertical. `up` needn't be perpendicular to the view direction,
/// but it mustn't be parallel to it: the view matrix is then degenerate (no panic). See the
/// [module documentation](self) for the coordinate conventions.
///
/// # Examples
///
/// A vertex shader transforms world-space positions by the projection and view matrices. The
/// point the camera looks at lands in the centre of the screen:
///
/// ```
/// use std::f32::consts::FRAC_PI_3;
///
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let camera = Camera::new(
///     Vec3::new(0.0, 2.0, 5.0),
///     Vec3::ZERO,
///     Vec3::new(0.0, 1.0, 0.0),
///     Projection::Perspective(PerspectiveProjection::new(FRAC_PI_3, 16.0 / 9.0, 0.1, 100.0)),
/// );
///
/// let view_projection = camera.projection_matrix() * camera.view_matrix();
/// let ndc = (view_projection * camera.lookat.to_point4()).homogenize_to_vec3();
///
/// assert!(ndc.x.abs() < 1e-6 && ndc.y.abs() < 1e-6);
/// assert!((-1.0..=1.0).contains(&ndc.z));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// The camera's position in world space.
    pub eye: Vec3,
    /// The world-space point the camera looks at.
    pub lookat: Vec3,
    /// The world-space direction that appears upwards in the view.
    pub up: Vec3,
    /// How the view is projected to clip space.
    pub projection: Projection,
}
impl Camera {
    /// Creates a camera at `eye`, looking at `lookat`, with `up` as the vertical.
    ///
    /// # Examples
    ///
    /// See [`Camera`].
    pub fn new(eye: Vec3, lookat: Vec3, up: Vec3, projection: Projection) -> Self {
        Self {
            eye,
            lookat,
            up,
            projection,
        }
    }

    /// Returns the view matrix, which takes world space to the camera's view space.
    ///
    /// It is [`Mat4::look_at`] of the camera's `eye`, `lookat`, and `up`: the eye moves to the
    /// origin and the view direction to −z.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let projection = Projection::Perspective(PerspectiveProjection::new(1.0, 1.0, 0.1, 10.0));
    /// let camera = Camera::new(
    ///     Vec3::new(3.0, 0.0, 0.0),
    ///     Vec3::ZERO,
    ///     Vec3::new(0.0, 1.0, 0.0),
    ///     projection,
    /// );
    ///
    /// let target = camera.view_matrix() * camera.lookat.to_point4();
    /// assert!((target.xyz() - Vec3::new(0.0, 0.0, -3.0)).length() < 1e-6);
    /// ```
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(self.eye, self.lookat, self.up)
    }

    /// Returns the projection matrix, which takes view space to clip space.
    ///
    /// The same as [`Projection::matrix`] on the camera's [`projection`](Camera::projection).
    ///
    /// # Examples
    ///
    /// See [`Camera`].
    pub fn projection_matrix(&self) -> Mat4 {
        self.projection.matrix()
    }

    /// Fits the projection to a viewport whose width divided by its height is `aspect_ratio`.
    ///
    /// A perspective projection takes the new aspect ratio and keeps its vertical field of view.
    /// An orthographic projection keeps its width and vertical centre, and moves `bottom` and
    /// `top` so its height is the width divided by `aspect_ratio`.
    ///
    /// # Examples
    ///
    /// Keep the image undistorted when the window is resized, from
    /// [`Application::event`](crate::Application::event):
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let projection = Projection::Orthographic(OrthographicProjection::new(
    ///     -2.0, 2.0, -1.0, 1.0, 0.1, 10.0,
    /// ));
    /// let mut camera = Camera::new(
    ///     Vec3::new(0.0, 0.0, 5.0),
    ///     Vec3::ZERO,
    ///     Vec3::new(0.0, 1.0, 0.0),
    ///     projection,
    /// );
    ///
    /// let event = AppEvent::Resized { width: 800, height: 800 };
    /// if let AppEvent::Resized { width, height } = event {
    ///     camera.set_aspect_ratio(width as f32 / height as f32);
    /// }
    ///
    /// let Projection::Orthographic(orthographic) = camera.projection else { unreachable!() };
    /// assert_eq!((orthographic.left, orthographic.right), (-2.0, 2.0));
    /// assert_eq!((orthographic.bottom, orthographic.top), (-2.0, 2.0));
    /// ```
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        match &mut self.projection {
            Projection::Perspective(perspective) => perspective.aspect_ratio = aspect_ratio,
            Projection::Orthographic(orthographic) => {
                let width = orthographic.right - orthographic.left;
                let height = width / aspect_ratio;
                let center_y = (orthographic.top + orthographic.bottom) / 2.0;
                orthographic.bottom = center_y - height / 2.0;
                orthographic.top = center_y + height / 2.0;
            }
        }
    }
}

/// A camera projection, perspective or orthographic.
///
/// # Examples
///
/// See [`Camera`] and [`Camera::set_aspect_ratio`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    /// A perspective projection, in which distant objects look smaller.
    Perspective(PerspectiveProjection),
    /// An orthographic projection, in which size doesn't depend on distance.
    Orthographic(OrthographicProjection),
}
impl Projection {
    /// Returns the projection matrix, which takes view space to clip space.
    ///
    /// It is the matrix of whichever variant the projection is.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let perspective = PerspectiveProjection::new(1.0, 1.5, 1.0, 100.0);
    /// let orthographic = OrthographicProjection::new(-1.0, 1.0, -1.0, 1.0, 1.0, 100.0);
    ///
    /// assert_eq!(Projection::Perspective(perspective).matrix(), perspective.matrix());
    /// assert_eq!(Projection::Orthographic(orthographic).matrix(), orthographic.matrix());
    /// ```
    pub fn matrix(&self) -> Mat4 {
        match self {
            Projection::Perspective(perspective) => perspective.matrix(),
            Projection::Orthographic(orthographic) => orthographic.matrix(),
        }
    }
}

/// An orthographic projection of a box-shaped view volume.
///
/// The bounds are in view space: `left` to `right` along x and `bottom` to `top` along y, with
/// `near` and `far` the distances along −z. Nothing is checked: equal bounds give a matrix with
/// infinite or NaN entries.
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let orthographic = OrthographicProjection::new(-4.0, 4.0, -2.0, 2.0, 1.0, 11.0);
///
/// // The top-right corner of the far plane maps to the corner of the NDC cube.
/// let corner = orthographic.matrix() * Vec4::new(4.0, 2.0, -11.0, 1.0);
/// assert!((corner.xyz() - Vec3::new(1.0, 1.0, 1.0)).length() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthographicProjection {
    /// The view-space x of the volume's left side.
    pub left: f32,
    /// The view-space x of the volume's right side.
    pub right: f32,
    /// The view-space y of the volume's bottom.
    pub bottom: f32,
    /// The view-space y of the volume's top.
    pub top: f32,
    /// The distance from the camera to the near plane.
    pub near: f32,
    /// The distance from the camera to the far plane.
    pub far: f32,
}
impl OrthographicProjection {
    /// Creates an orthographic projection of the given view-space bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let orthographic = OrthographicProjection::new(-4.0, 4.0, -2.0, 2.0, 1.0, 11.0);
    ///
    /// assert_eq!((orthographic.left, orthographic.right), (-4.0, 4.0));
    /// assert_eq!((orthographic.bottom, orthographic.top), (-2.0, 2.0));
    /// assert_eq!((orthographic.near, orthographic.far), (1.0, 11.0));
    /// ```
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

    /// Returns the projection matrix, [`Mat4::orthographic`] of the bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let orthographic = OrthographicProjection::new(-4.0, 4.0, -2.0, 2.0, 1.0, 11.0);
    ///
    /// // The centre of the near plane maps to the centre of the NDC cube's near face.
    /// let centre = orthographic.matrix() * Vec4::new(0.0, 0.0, -1.0, 1.0);
    /// assert!((centre.xyz() - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
    /// ```
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

/// A perspective projection of a symmetric view frustum.
///
/// `fov` is the vertical field of view in radians, and `aspect_ratio` the viewport's width
/// divided by its height, which sets the horizontal field of view. `near` and `far` are the
/// distances to the clipping planes along −z, with `0 < near < far`. Nothing is checked: values
/// outside these ranges give a matrix with infinite, NaN, or inverted entries.
///
/// # Examples
///
/// ```
/// use std::f32::consts::FRAC_PI_2;
///
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let perspective = PerspectiveProjection::new(FRAC_PI_2, 2.0, 1.0, 100.0);
///
/// // With a 90° vertical field of view, the top edge of the near plane is 1 unit up.
/// let top = (perspective.matrix() * Vec4::new(0.0, 1.0, -1.0, 1.0)).homogenize_to_vec3();
/// assert!((top - Vec3::new(0.0, 1.0, -1.0)).length() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveProjection {
    /// The vertical field of view, in radians.
    pub fov: f32,
    /// The viewport's width divided by its height.
    pub aspect_ratio: f32,
    /// The distance from the camera to the near plane.
    pub near: f32,
    /// The distance from the camera to the far plane.
    pub far: f32,
}
impl PerspectiveProjection {
    /// Creates a perspective projection with the vertical field of view `fov`, in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f32::consts::FRAC_PI_3;
    ///
    /// use engine::prelude::*;
    ///
    /// let perspective = PerspectiveProjection::new(FRAC_PI_3, 16.0 / 9.0, 0.1, 100.0);
    ///
    /// assert_eq!(perspective.fov, FRAC_PI_3);
    /// assert_eq!(perspective.aspect_ratio, 16.0 / 9.0);
    /// assert_eq!((perspective.near, perspective.far), (0.1, 100.0));
    /// ```
    pub fn new(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        Self {
            fov,
            aspect_ratio,
            near,
            far,
        }
    }

    /// Returns the projection matrix, [`Mat4::perspective`] of the parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let perspective = PerspectiveProjection::new(1.0, 1.5, 1.0, 100.0);
    ///
    /// // Points on the near and far planes' axis map to NDC z = -1 and 1.
    /// let near = (perspective.matrix() * Vec4::new(0.0, 0.0, -1.0, 1.0)).homogenize_to_vec3();
    /// let far = (perspective.matrix() * Vec4::new(0.0, 0.0, -100.0, 1.0)).homogenize_to_vec3();
    /// assert!((near.z + 1.0).abs() < 1e-5);
    /// assert!((far.z - 1.0).abs() < 1e-5);
    /// ```
    pub fn matrix(&self) -> Mat4 {
        Mat4::perspective(self.fov, self.aspect_ratio, self.near, self.far)
    }
}

/// A camera controller that orbits the camera with the keyboard and zooms with the mouse wheel.
///
/// - Orbit: A and D or Left and Right change the azimuth, and W and S or Up and Down the
///   elevation, at 1 radian per second.
/// - Zoom: the mouse wheel changes the radius by half the scroll delta per second, down to a
///   minimum of 0.1.
///
/// The controller sets only the camera's [`eye`](Camera::eye), to
/// `radius * (cos(elevation) * sin(azimuth), sin(elevation), cos(elevation) * cos(azimuth))`
/// from the world origin. It leaves [`lookat`](Camera::lookat) alone.
///
/// # Examples
///
/// Pass the application's events to the controller and update the camera once per frame:
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let projection = Projection::Perspective(PerspectiveProjection::new(1.0, 1.0, 0.1, 100.0));
/// let mut camera = Camera::new(
///     Vec3::new(2.0, 1.0, 2.0),
///     Vec3::ZERO,
///     Vec3::new(0.0, 1.0, 0.0),
///     projection,
/// );
/// let mut controls = OrbitControls::new(&camera);
///
/// // In `Application::event`:
/// controls.handle_event(AppEvent::Input(InputEvent::Key {
///     key: InputKey::D,
///     state: ButtonState::Pressed,
/// }));
/// controls.handle_event(AppEvent::Input(InputEvent::MouseWheel { delta_y: 2.0 }));
///
/// // In `Application::update`:
/// let azimuth = controls.azimuth;
/// controls.update_from_events(&mut camera, 0.5);
///
/// assert_eq!(controls.azimuth, azimuth + 0.5);
/// assert_eq!(controls.radius, 2.5);
/// assert!((camera.eye.length() - 2.5).abs() < 1e-5);
///
/// // D is still held, but the scroll was consumed by the previous update.
/// controls.update_from_events(&mut camera, 0.5);
/// assert!((controls.azimuth - (azimuth + 1.0)).abs() < 1e-6);
/// assert_eq!(controls.radius, 2.5);
/// ```
///
/// # Known issues
///
/// - The controller doesn't keep the camera's view: unless the eye starts on the plane x = z, the
///   first update reflects it across that plane; it orbits the world origin, not the camera's
///   [`lookat`](Camera::lookat); and past an elevation of ±π/2 the view turns upside down, because
///   the elevation isn't clamped ([#52](https://github.com/costott/cpu_rasteriser/issues/52)).
#[derive(Debug, Clone)]
pub struct OrbitControls {
    /// The eye's distance from the origin.
    pub radius: f32,
    /// The eye's angle around the y axis, in radians, from +z towards +x.
    pub azimuth: f32,
    /// The eye's angle above the xz plane, in radians.
    pub elevation: f32,
    input_state: InputState,
}
impl OrbitControls {
    const SPEED: f32 = 1.0;
    const ZOOM_SPEED: f32 = 0.5;

    /// Creates a controller for the camera's current eye position, with no input held.
    ///
    /// The radius, azimuth, and elevation are read from the eye's position relative to the origin.
    /// An eye at the origin gives a NaN elevation.
    ///
    /// # Examples
    ///
    /// See [`OrbitControls`].
    ///
    /// # Known issues
    ///
    /// - The azimuth is read as `atan2(eye.z, eye.x)`, which doesn't match
    ///   [`update_camera`](OrbitControls::update_camera), so unless the eye starts on the plane
    ///   x = z, the first update reflects it across that plane
    ///   ([#52](https://github.com/costott/cpu_rasteriser/issues/52)).
    pub fn new(camera: &Camera) -> Self {
        Self {
            radius: camera.eye.length(),
            azimuth: camera.eye.z.atan2(camera.eye.x),
            elevation: (camera.eye.y / camera.eye.length()).asin(),
            input_state: InputState::default(),
        }
    }

    /// Records the input in an application event, for the next
    /// [`update_from_events`](OrbitControls::update_from_events).
    ///
    /// Events other than [`AppEvent::Input`] are ignored. Key and button presses last until
    /// their release; scroll deltas add up until the next update.
    ///
    /// # Examples
    ///
    /// See [`OrbitControls`].
    pub fn handle_event(&mut self, event: AppEvent) {
        if let AppEvent::Input(input_event) = event {
            self.input_state.handle_event(input_event);
        }
    }

    /// Updates the controller and the camera from the input recorded by
    /// [`handle_event`](OrbitControls::handle_event), over `dt` seconds.
    ///
    /// This is [`update`](OrbitControls::update) with the recorded input, after which the scroll
    /// delta is cleared so each scroll is applied once.
    ///
    /// # Examples
    ///
    /// See [`OrbitControls`].
    pub fn update_from_events(&mut self, camera: &mut Camera, dt: f32) {
        let input = self.input_state.clone();
        self.update(camera, &input, dt);
        self.input_state.clear_deltas();
    }

    /// Moves the camera's eye to the position given by the radius, azimuth, and elevation.
    ///
    /// Use this after setting the fields directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f32::consts::FRAC_PI_2;
    ///
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let projection = Projection::Perspective(PerspectiveProjection::new(1.0, 1.0, 0.1, 100.0));
    /// let mut camera = Camera::new(
    ///     Vec3::new(2.0, 1.0, 2.0),
    ///     Vec3::ZERO,
    ///     Vec3::new(0.0, 1.0, 0.0),
    ///     projection,
    /// );
    /// let mut controls = OrbitControls::new(&camera);
    ///
    /// controls.radius = 2.0;
    /// controls.azimuth = FRAC_PI_2;
    /// controls.elevation = 0.0;
    /// controls.update_camera(&mut camera);
    ///
    /// assert!((camera.eye - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-6);
    /// ```
    pub fn update_camera(&self, camera: &mut Camera) {
        let x = self.radius * self.elevation.cos() * self.azimuth.sin();
        let y = self.radius * self.elevation.sin();
        let z = self.radius * self.elevation.cos() * self.azimuth.cos();

        camera.eye = Vec3::new(x, y, z);
    }

    /// Updates the controller and the camera from `input`, over `dt` seconds.
    ///
    /// Use this instead of [`update_from_events`](OrbitControls::update_from_events) when the
    /// application tracks input itself. It applies the held keys and the scroll delta as
    /// [`OrbitControls`] describes, then calls [`update_camera`](OrbitControls::update_camera).
    /// It doesn't clear anything in `input`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let projection = Projection::Perspective(PerspectiveProjection::new(1.0, 1.0, 0.1, 100.0));
    /// let mut camera = Camera::new(
    ///     Vec3::new(2.0, 1.0, 2.0),
    ///     Vec3::ZERO,
    ///     Vec3::new(0.0, 1.0, 0.0),
    ///     projection,
    /// );
    /// let mut controls = OrbitControls::new(&camera);
    ///
    /// let mut input = InputState::default();
    /// input.set_key_state(InputKey::W, ButtonState::Pressed);
    /// let elevation = controls.elevation;
    /// controls.update(&mut camera, &input, 0.25);
    ///
    /// assert_eq!(controls.elevation, elevation + 0.25);
    /// assert!(camera.eye.y > 1.0);
    /// ```
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

/// A camera controller that moves and turns the camera in a first-person style.
///
/// - Move: W and S move forwards and backwards, and A and D sideways, in the horizontal plane,
///   at 2 units per second.
/// - Rise and fall: Space and LeftShift move up and down along the camera's
///   [`up`](Camera::up), at the same speed.
/// - Look: mouse movement turns the camera by 0.01 radians per pixel. Moving the mouse right
///   increases the yaw and moving it up increases the pitch, which is clamped to just inside
///   ±π/2. Relative motion ([`InputEvent::MouseMotion`](crate::input::InputEvent::MouseMotion))
///   is used when there is any; otherwise the change in cursor position since the previous
///   update.
///
/// The look direction is `(cos(yaw) * cos(pitch), sin(pitch), sin(yaw) * cos(pitch))`. Each update
/// sets the camera's [`lookat`](Camera::lookat) one unit along it from the eye.
///
/// While the cursor isn't grabbed ([`cursor_grabbed`](FirstPersonControls::cursor_grabbed)),
/// [`update_from_events`](FirstPersonControls::update_from_events) leaves the camera alone, so the
/// user can use the mouse for something else. The flag doesn't change the window: the application
/// applies it, typically through [`Application::window_state`](crate::Application::window_state).
///
/// # Examples
///
/// ```
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// let projection = Projection::Perspective(PerspectiveProjection::new(1.0, 1.0, 0.1, 100.0));
/// let mut camera = Camera::new(
///     Vec3::ZERO,
///     Vec3::new(0.0, 0.0, -1.0),
///     Vec3::new(0.0, 1.0, 0.0),
///     projection,
/// );
/// let mut controls = FirstPersonControls::new(&camera);
///
/// // In `Application::event`:
/// controls.handle_event(AppEvent::Input(InputEvent::Key {
///     key: InputKey::W,
///     state: ButtonState::Pressed,
/// }));
///
/// // In `Application::update`:
/// controls.update_from_events(&mut camera, 0.5);
/// assert!((camera.eye - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
///
/// // Escape releases the cursor, and the camera stops following the input.
/// controls.set_cursor_grabbed(false);
/// controls.update_from_events(&mut camera, 0.5);
/// assert!((camera.eye - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct FirstPersonControls {
    /// The look direction's angle around the y axis, in radians, from +x towards +z.
    pub yaw: f32,
    /// The look direction's angle above the horizontal, in radians.
    pub pitch: f32,

    input_state: InputState,
    cursor_grabbed: bool,
    last_mouse: Option<(f32, f32)>,
}
impl FirstPersonControls {
    const MOVEMENT_SPEED: f32 = 2.0;
    const MOUSE_SENSITIVITY: f32 = 0.01;

    /// Creates a controller facing the camera's current look direction, with the cursor grabbed
    /// and no input held.
    ///
    /// If the camera's `eye` and `lookat` are equal there is no look direction, and the
    /// controller starts with a yaw and pitch of 0 (facing +x).
    ///
    /// # Examples
    ///
    /// See [`FirstPersonControls`].
    pub fn new(camera: &Camera) -> Self {
        let forward = (camera.lookat - camera.eye).normalise();
        let yaw = forward.z.atan2(forward.x);
        let pitch = forward.y.asin();

        Self {
            yaw,
            pitch,
            input_state: InputState::default(),
            cursor_grabbed: true,
            last_mouse: None,
        }
    }

    /// Records the input in an application event, for the next
    /// [`update_from_events`](FirstPersonControls::update_from_events).
    ///
    /// Events other than [`AppEvent::Input`] are ignored. Key and button presses last until
    /// their release; mouse motion adds up until the next update.
    ///
    /// # Examples
    ///
    /// See [`FirstPersonControls`].
    pub fn handle_event(&mut self, event: AppEvent) {
        if let AppEvent::Input(input_event) = event {
            self.input_state.handle_event(input_event);
        }
    }

    /// Updates the controller and the camera from the input recorded by
    /// [`handle_event`](FirstPersonControls::handle_event), over `dt` seconds.
    ///
    /// While the cursor is grabbed this is [`update`](FirstPersonControls::update) with the
    /// recorded input; otherwise the camera is left alone. Either way the recorded mouse motion
    /// is then cleared, so each movement is applied once.
    ///
    /// # Examples
    ///
    /// See [`FirstPersonControls`].
    pub fn update_from_events(&mut self, camera: &mut Camera, dt: f32) {
        if self.cursor_grabbed {
            let input = self.input_state.clone();
            self.update(camera, &input, dt);
        }
        self.input_state.clear_deltas();
    }

    /// Returns whether the controller treats the cursor as grabbed, and so follows the input.
    ///
    /// # Examples
    ///
    /// See [`FirstPersonControls::toggle_cursor_grabbed`].
    pub fn cursor_grabbed(&self) -> bool {
        self.cursor_grabbed
    }

    /// Sets whether the controller treats the cursor as grabbed.
    ///
    /// Releasing the cursor discards any pending mouse movement, so the view doesn't jump when
    /// the cursor is grabbed again.
    ///
    /// # Examples
    ///
    /// See [`FirstPersonControls`].
    pub fn set_cursor_grabbed(&mut self, grabbed: bool) {
        self.cursor_grabbed = grabbed;

        if !grabbed {
            self.input_state.clear_deltas();
            self.last_mouse = None;
        }
    }

    /// Grabs the cursor if it is released, and releases it if it is grabbed.
    ///
    /// See [`set_cursor_grabbed`](FirstPersonControls::set_cursor_grabbed).
    ///
    /// # Examples
    ///
    /// Toggle with a key, and ask the window to lock the cursor while it is grabbed:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let projection = Projection::Perspective(PerspectiveProjection::new(1.0, 1.0, 0.1, 100.0));
    /// let camera = Camera::new(
    ///     Vec3::ZERO,
    ///     Vec3::new(0.0, 0.0, -1.0),
    ///     Vec3::new(0.0, 1.0, 0.0),
    ///     projection,
    /// );
    /// let mut controls = FirstPersonControls::new(&camera);
    ///
    /// let event = InputEvent::Key { key: InputKey::Escape, state: ButtonState::Pressed };
    /// if let InputEvent::Key { key: InputKey::Escape, state: ButtonState::Pressed } = event {
    ///     controls.toggle_cursor_grabbed();
    /// }
    /// assert!(!controls.cursor_grabbed());
    ///
    /// let grab = if controls.cursor_grabbed() { CursorGrab::Locked } else { CursorGrab::None };
    /// assert_eq!(grab, CursorGrab::None);
    /// ```
    pub fn toggle_cursor_grabbed(&mut self) {
        self.set_cursor_grabbed(!self.cursor_grabbed);
    }

    /// Updates the controller and the camera from `input`, over `dt` seconds.
    ///
    /// Use this instead of [`update_from_events`](FirstPersonControls::update_from_events) when
    /// the application tracks input itself. It turns, then moves, the camera as
    /// [`FirstPersonControls`] describes, whether or not the cursor is grabbed. It doesn't clear
    /// anything in `input`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// let projection = Projection::Perspective(PerspectiveProjection::new(1.0, 1.0, 0.1, 100.0));
    /// let mut camera = Camera::new(
    ///     Vec3::ZERO,
    ///     Vec3::new(0.0, 0.0, -1.0),
    ///     Vec3::new(0.0, 1.0, 0.0),
    ///     projection,
    /// );
    /// let mut controls = FirstPersonControls::new(&camera);
    ///
    /// // Moving the mouse 50π pixels to the right turns the camera a quarter turn, to face +x.
    /// let mut input = InputState::default();
    /// input.add_mouse_delta(50.0 * std::f32::consts::PI, 0.0);
    /// controls.update(&mut camera, &input, 0.0);
    ///
    /// assert!((camera.lookat - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    /// ```
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

        let mut right = forward.cross(camera.up).normalise();
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
