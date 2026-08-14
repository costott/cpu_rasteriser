use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputKey {
    Escape,
    Left,
    Right,
    Up,
    Down,
    W,
    A,
    S,
    D,
    Space,
    LeftShift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonState {
    Pressed,
    Released,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputEvent {
    Key {
        key: InputKey,
        state: ButtonState,
    },
    MouseButton {
        button: MouseButton,
        state: ButtonState,
    },
    MouseMoved {
        x: f32,
        y: f32,
    },
    MouseMotion {
        dx: f32,
        dy: f32,
    },
    MouseWheel {
        delta_y: f32,
    },
}

pub trait CameraControlInput {
    fn is_key_down(&self, key: InputKey) -> bool;
    fn mouse_position(&self) -> Option<(f32, f32)>;
    fn mouse_delta(&self) -> (f32, f32);
    fn scroll_delta_y(&self) -> f32;
}

#[derive(Debug, Clone, Default)]
pub struct InputState {
    keys_down: HashSet<InputKey>,
    mouse_buttons_down: HashSet<MouseButton>,
    mouse_position: Option<(f32, f32)>,
    mouse_delta: (f32, f32),
    scroll_delta_y: f32,
}

impl InputState {
    pub fn handle_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::Key { key, state } => self.set_key_state(key, state),
            InputEvent::MouseButton { button, state } => self.set_button_state(button, state),
            InputEvent::MouseMoved { x, y } => self.set_mouse_position(x, y),
            InputEvent::MouseMotion { dx, dy } => self.add_mouse_delta(dx, dy),
            InputEvent::MouseWheel { delta_y } => self.add_scroll_delta_y(delta_y),
        }
    }

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

    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_position = Some((x, y));
    }

    pub fn add_scroll_delta_y(&mut self, delta_y: f32) {
        self.scroll_delta_y += delta_y;
    }

    pub fn add_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    pub fn clear_deltas(&mut self) {
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta_y = 0.0;
    }

    pub fn is_key_down(&self, key: InputKey) -> bool {
        self.keys_down.contains(&key)
    }

    pub fn is_button_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons_down.contains(&button)
    }

    pub fn mouse_position(&self) -> Option<(f32, f32)> {
        self.mouse_position
    }

    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

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
