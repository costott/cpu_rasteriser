//! Keyboard and mouse input, the same on every backend.
//!
//! Each backend translates its window system's input into [`InputEvent`]s and delivers them to
//! [`Application::event`](crate::Application::event) inside
//! [`AppEvent::Input`](crate::AppEvent::Input). An app can react to the events directly, or feed
//! them to an [`InputState`] to ask which keys are held down at any moment.
//!
//! Only the keys in [`InputKey`] and the buttons in [`MouseButton`] are reported.

use std::collections::HashSet;

/// A keyboard key the engine reports. Other keys produce no events.
///
/// The winit backend identifies keys by their physical position on a US layout, so
/// [`InputKey::W`] is the key in the W position whatever the user's layout.
///
/// # Examples
///
/// See [`InputEvent`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputKey {
    /// The Escape key.
    Escape,
    /// The left arrow key.
    Left,
    /// The right arrow key.
    Right,
    /// The up arrow key.
    Up,
    /// The down arrow key.
    Down,
    /// The W key.
    W,
    /// The A key.
    A,
    /// The S key.
    S,
    /// The D key.
    D,
    /// The space bar.
    Space,
    /// The left Shift key.
    LeftShift,
}

/// A mouse button the engine reports. Other buttons produce no events.
///
/// # Examples
///
/// See [`InputState::is_button_down`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    /// The left (primary) button.
    Left,
    /// The right (secondary) button.
    Right,
    /// The middle button, usually the scroll wheel.
    Middle,
}

/// Whether a key or button event is a press or a release.
///
/// # Examples
///
/// Reacting to a key only when it goes down:
///
/// ```
/// use engine::prelude::*;
///
/// fn is_escape_press(event: InputEvent) -> bool {
///     matches!(event, InputEvent::Key { key: InputKey::Escape, state: ButtonState::Pressed })
/// }
///
/// let press = InputEvent::Key { key: InputKey::Escape, state: ButtonState::Pressed };
/// let release = InputEvent::Key { key: InputKey::Escape, state: ButtonState::Released };
///
/// assert!(is_escape_press(press));
/// assert!(!is_escape_press(release));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonState {
    /// The key or button went down.
    Pressed,
    /// The key or button came back up.
    Released,
}

/// A keyboard or mouse event, delivered inside [`AppEvent::Input`](crate::AppEvent::Input).
///
/// Cursor positions are in pixels, with the origin at the window's top-left corner and y
/// pointing down.
///
/// # Examples
///
/// React to a key press in [`Application::event`](crate::Application::event):
///
/// ```
/// use engine::prelude::*;
///
/// struct App {
///     paused: bool,
/// }
///
/// impl Application for App {
///     fn event(&mut self, event: AppEvent, _handle: &mut AppHandle) {
///         if let AppEvent::Input(InputEvent::Key {
///             key: InputKey::Space,
///             state: ButtonState::Pressed,
///         }) = event
///         {
///             self.paused = !self.paused;
///         }
///     }
/// }
///
/// let mut app = App { paused: false };
/// let space = InputEvent::Key {
///     key: InputKey::Space,
///     state: ButtonState::Pressed,
/// };
/// app.event(AppEvent::Input(space), &mut AppHandle::default());
/// assert!(app.paused);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputEvent {
    /// A key was pressed or released.
    Key {
        /// The key.
        key: InputKey,
        /// Whether it was pressed or released.
        state: ButtonState,
    },
    /// A mouse button was pressed or released.
    MouseButton {
        /// The button.
        button: MouseButton,
        /// Whether it was pressed or released.
        state: ButtonState,
    },
    /// The cursor's position.
    ///
    /// The winit backend sends it when the cursor moves over the window. The minifb backend
    /// sends it every frame while it knows the cursor's position, which may lie outside the
    /// window.
    MouseMoved {
        /// The horizontal position in pixels.
        x: f32,
        /// The vertical position in pixels.
        y: f32,
    },
    /// Raw relative motion of the mouse, which continues when the cursor is locked or at the
    /// edge of the screen. Only the winit backend sends it, and never with zero motion.
    MouseMotion {
        /// The horizontal motion, in the device's units rather than pixels.
        dx: f32,
        /// The vertical motion, in the device's units rather than pixels.
        dy: f32,
    },
    /// The scroll wheel or touchpad scrolled vertically. Zero scrolls aren't sent.
    MouseWheel {
        /// The vertical scroll amount. Its units depend on the backend and device: on winit,
        /// lines for a mouse wheel and pixels for a touchpad.
        delta_y: f32,
    },
}

/// A source of the input that camera controllers read.
///
/// [`OrbitControls::update`](crate::components::camera::OrbitControls::update) and
/// [`FirstPersonControls::update`](crate::components::camera::FirstPersonControls::update) take
/// any implementation, so a controller can be driven by something other than an
/// [`InputState`], such as recorded or scripted input.
///
/// # Examples
///
/// Scripted input that holds down one key:
///
/// ```
/// use engine::prelude::*;
///
/// struct HoldKey(InputKey);
///
/// impl CameraControlInput for HoldKey {
///     fn is_key_down(&self, key: InputKey) -> bool {
///         key == self.0
///     }
///
///     fn mouse_position(&self) -> Option<(f32, f32)> {
///         None
///     }
///
///     fn mouse_delta(&self) -> (f32, f32) {
///         (0.0, 0.0)
///     }
///
///     fn scroll_delta_y(&self) -> f32 {
///         0.0
///     }
/// }
///
/// let input = HoldKey(InputKey::Right);
/// assert!(input.is_key_down(InputKey::Right));
/// assert!(!input.is_key_down(InputKey::Left));
/// ```
pub trait CameraControlInput {
    /// Returns whether `key` is held down.
    fn is_key_down(&self, key: InputKey) -> bool;
    /// Returns the cursor's last known position in pixels from the window's top-left corner, or
    /// `None` if it isn't known.
    fn mouse_position(&self) -> Option<(f32, f32)>;
    /// Returns the raw mouse motion `(dx, dy)` accumulated since the input was last cleared.
    fn mouse_delta(&self) -> (f32, f32);
    /// Returns the vertical scroll accumulated since the input was last cleared.
    fn scroll_delta_y(&self) -> f32;
}

/// The current keyboard and mouse state, built up from [`InputEvent`]s.
///
/// Events update which keys and buttons are held down and record the cursor's position. Mouse
/// motion and scrolling accumulate until [`clear_deltas`](InputState::clear_deltas), which an
/// app typically calls once per frame after reading them. The default state has nothing held
/// down, no cursor position, and no accumulated motion.
///
/// # Examples
///
/// Hold the state in an app and read it once per frame:
///
/// ```
/// use engine::prelude::*;
///
/// #[derive(Default)]
/// struct App {
///     input: InputState,
///     position: f32,
/// }
///
/// impl Application for App {
///     fn event(&mut self, event: AppEvent, _handle: &mut AppHandle) {
///         if let AppEvent::Input(input_event) = event {
///             self.input.handle_event(input_event);
///         }
///     }
///
///     fn update(&mut self, dt: f32) {
///         if self.input.is_key_down(InputKey::D) {
///             self.position += 2.0 * dt;
///         }
///         self.input.clear_deltas();
///     }
/// }
///
/// let mut app = App::default();
/// let press_d = InputEvent::Key {
///     key: InputKey::D,
///     state: ButtonState::Pressed,
/// };
/// app.event(AppEvent::Input(press_d), &mut AppHandle::default());
/// app.update(0.5);
/// assert_eq!(app.position, 1.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct InputState {
    keys_down: HashSet<InputKey>,
    mouse_buttons_down: HashSet<MouseButton>,
    mouse_position: Option<(f32, f32)>,
    mouse_delta: (f32, f32),
    scroll_delta_y: f32,
}

impl InputState {
    /// Updates the state from `event`.
    ///
    /// Key and button events update what is held down, [`InputEvent::MouseMoved`] replaces the
    /// cursor position, and [`InputEvent::MouseMotion`] and [`InputEvent::MouseWheel`] add to
    /// the accumulated deltas.
    ///
    /// # Examples
    ///
    /// See [`InputState`].
    pub fn handle_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::Key { key, state } => self.set_key_state(key, state),
            InputEvent::MouseButton { button, state } => self.set_button_state(button, state),
            InputEvent::MouseMoved { x, y } => self.set_mouse_position(x, y),
            InputEvent::MouseMotion { dx, dy } => self.add_mouse_delta(dx, dy),
            InputEvent::MouseWheel { delta_y } => self.add_scroll_delta_y(delta_y),
        }
    }

    /// Records `key` as held down if `state` is [`ButtonState::Pressed`], and as up otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// input.set_key_state(InputKey::W, ButtonState::Pressed);
    /// assert!(input.is_key_down(InputKey::W));
    ///
    /// input.set_key_state(InputKey::W, ButtonState::Released);
    /// assert!(!input.is_key_down(InputKey::W));
    /// ```
    pub fn set_key_state(&mut self, key: InputKey, state: ButtonState) {
        match state {
            ButtonState::Pressed => {
                self.keys_down.insert(key);
            }
            ButtonState::Released => {
                self.keys_down.remove(&key);
            }
        }
    }

    /// Records `button` as held down if `state` is [`ButtonState::Pressed`], and as up
    /// otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// input.set_button_state(MouseButton::Right, ButtonState::Pressed);
    /// assert!(input.is_button_down(MouseButton::Right));
    ///
    /// input.set_button_state(MouseButton::Right, ButtonState::Released);
    /// assert!(!input.is_button_down(MouseButton::Right));
    /// ```
    pub fn set_button_state(&mut self, button: MouseButton, state: ButtonState) {
        match state {
            ButtonState::Pressed => {
                self.mouse_buttons_down.insert(button);
            }
            ButtonState::Released => {
                self.mouse_buttons_down.remove(&button);
            }
        }
    }

    /// Records the cursor's position, in pixels from the window's top-left corner.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// assert_eq!(input.mouse_position(), None);
    ///
    /// input.set_mouse_position(320.0, 180.0);
    /// assert_eq!(input.mouse_position(), Some((320.0, 180.0)));
    /// ```
    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_position = Some((x, y));
    }

    /// Adds `delta_y` to the vertical scroll accumulated since the last
    /// [`clear_deltas`](InputState::clear_deltas).
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// input.add_scroll_delta_y(1.0);
    /// input.add_scroll_delta_y(-3.0);
    /// assert_eq!(input.scroll_delta_y(), -2.0);
    /// ```
    pub fn add_scroll_delta_y(&mut self, delta_y: f32) {
        self.scroll_delta_y += delta_y;
    }

    /// Adds `(dx, dy)` to the mouse motion accumulated since the last
    /// [`clear_deltas`](InputState::clear_deltas).
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// input.add_mouse_delta(2.0, 1.0);
    /// input.add_mouse_delta(-5.0, 4.0);
    /// assert_eq!(input.mouse_delta(), (-3.0, 5.0));
    /// ```
    pub fn add_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Resets the accumulated mouse motion and scroll to zero.
    ///
    /// Held keys, held buttons, and the cursor position are kept. Call it once per frame after
    /// reading the deltas, so that each frame sees only the motion since the previous one.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// input.add_mouse_delta(3.0, -2.0);
    /// input.add_mouse_delta(1.0, 1.0);
    /// input.add_scroll_delta_y(5.0);
    /// assert_eq!(input.mouse_delta(), (4.0, -1.0));
    /// assert_eq!(input.scroll_delta_y(), 5.0);
    ///
    /// input.clear_deltas();
    /// assert_eq!(input.mouse_delta(), (0.0, 0.0));
    /// assert_eq!(input.scroll_delta_y(), 0.0);
    /// ```
    pub fn clear_deltas(&mut self) {
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta_y = 0.0;
    }

    /// Returns whether `key` is held down.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// assert!(!input.is_key_down(InputKey::Space));
    ///
    /// input.handle_event(InputEvent::Key {
    ///     key: InputKey::Space,
    ///     state: ButtonState::Pressed,
    /// });
    /// assert!(input.is_key_down(InputKey::Space));
    /// assert!(!input.is_key_down(InputKey::Escape));
    /// ```
    pub fn is_key_down(&self, key: InputKey) -> bool {
        self.keys_down.contains(&key)
    }

    /// Returns whether `button` is held down.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// input.handle_event(InputEvent::MouseButton {
    ///     button: MouseButton::Left,
    ///     state: ButtonState::Pressed,
    /// });
    /// assert!(input.is_button_down(MouseButton::Left));
    /// assert!(!input.is_button_down(MouseButton::Right));
    /// ```
    pub fn is_button_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons_down.contains(&button)
    }

    /// Returns the cursor's last recorded position in pixels from the window's top-left corner,
    /// or `None` if none has been recorded.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// assert_eq!(input.mouse_position(), None);
    ///
    /// input.handle_event(InputEvent::MouseMoved { x: 10.0, y: 20.0 });
    /// assert_eq!(input.mouse_position(), Some((10.0, 20.0)));
    /// ```
    pub fn mouse_position(&self) -> Option<(f32, f32)> {
        self.mouse_position
    }

    /// Returns the mouse motion `(dx, dy)` accumulated since the last
    /// [`clear_deltas`](InputState::clear_deltas).
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// assert_eq!(input.mouse_delta(), (0.0, 0.0));
    ///
    /// input.handle_event(InputEvent::MouseMotion { dx: 3.0, dy: -2.0 });
    /// assert_eq!(input.mouse_delta(), (3.0, -2.0));
    /// ```
    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    /// Returns the vertical scroll accumulated since the last
    /// [`clear_deltas`](InputState::clear_deltas).
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// let mut input = InputState::default();
    /// assert_eq!(input.scroll_delta_y(), 0.0);
    ///
    /// input.handle_event(InputEvent::MouseWheel { delta_y: 2.0 });
    /// assert_eq!(input.scroll_delta_y(), 2.0);
    /// ```
    pub fn scroll_delta_y(&self) -> f32 {
        self.scroll_delta_y
    }
}

impl CameraControlInput for InputState {
    fn is_key_down(&self, key: InputKey) -> bool {
        self.keys_down.contains(&key)
    }

    fn mouse_position(&self) -> Option<(f32, f32)> {
        self.mouse_position
    }

    fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    fn scroll_delta_y(&self) -> f32 {
        self.scroll_delta_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_state_tracks_pressed_and_released_events() {
        let mut state = InputState::default();

        state.handle_event(InputEvent::Key {
            key: InputKey::W,
            state: ButtonState::Pressed,
        });
        assert!(state.is_key_down(InputKey::W));

        state.handle_event(InputEvent::Key {
            key: InputKey::W,
            state: ButtonState::Released,
        });
        assert!(!state.is_key_down(InputKey::W));
    }

    #[test]
    fn input_state_accumulates_motion_and_scroll() {
        let mut state = InputState::default();

        state.handle_event(InputEvent::MouseMotion { dx: 3.0, dy: -2.0 });
        state.handle_event(InputEvent::MouseWheel { delta_y: 5.0 });

        assert_eq!(state.mouse_delta(), (3.0, -2.0));
        assert_eq!(state.scroll_delta_y(), 5.0);

        state.clear_deltas();
        assert_eq!(state.mouse_delta(), (0.0, 0.0));
        assert_eq!(state.scroll_delta_y(), 0.0);
    }
}
