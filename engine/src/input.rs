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
pub enum MouseState {
    Pressed,
    Released,
}

pub trait CameraControlInput {
    fn is_key_down(&self, key: InputKey) -> bool;
    fn mouse_position(&self) -> Option<(f32, f32)>;
    fn mouse_delta(&self) -> (f32, f32);
    fn scroll_delta_y(&self) -> f32;
}

#[derive(Debug, Clone, Default)]
pub struct CameraInputState {
    keys_down: HashSet<InputKey>,
    mouse_buttons_down: HashSet<MouseButton>,
    mouse_position: Option<(f32, f32)>,
    mouse_delta: (f32, f32),
    scroll_delta_y: f32,
}

impl CameraInputState {
    pub fn set_key_state(&mut self, key: InputKey, state: MouseState) {
        match state {
            MouseState::Pressed => {
                self.keys_down.insert(key);
            }
            MouseState::Released => {
                self.keys_down.remove(&key);
            }
        }
    }

    pub fn set_button_state(&mut self, button: MouseButton, state: MouseState) {
        match state {
            MouseState::Pressed => {
                self.mouse_buttons_down.insert(button);
            }
            MouseState::Released => {
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

    pub fn is_button_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons_down.contains(&button)
    }
}

impl CameraControlInput for CameraInputState {
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

impl CameraControlInput for minifb::Window {
    fn is_key_down(&self, key: InputKey) -> bool {
        match key {
            InputKey::Escape => self.is_key_down(minifb::Key::Escape),
            InputKey::Left => self.is_key_down(minifb::Key::Left),
            InputKey::Right => self.is_key_down(minifb::Key::Right),
            InputKey::Up => self.is_key_down(minifb::Key::Up),
            InputKey::Down => self.is_key_down(minifb::Key::Down),
            InputKey::W => self.is_key_down(minifb::Key::W),
            InputKey::A => self.is_key_down(minifb::Key::A),
            InputKey::S => self.is_key_down(minifb::Key::S),
            InputKey::D => self.is_key_down(minifb::Key::D),
            InputKey::Space => self.is_key_down(minifb::Key::Space),
            InputKey::LeftShift => self.is_key_down(minifb::Key::LeftShift),
        }
    }

    fn mouse_position(&self) -> Option<(f32, f32)> {
        self.get_mouse_pos(minifb::MouseMode::Pass)
    }

    fn mouse_delta(&self) -> (f32, f32) {
        (0.0, 0.0)
    }

    fn scroll_delta_y(&self) -> f32 {
        self.get_scroll_wheel().map(|(_, y)| y).unwrap_or(0.0)
    }
}

pub fn map_winit_key_code(key_code: winit::keyboard::KeyCode) -> Option<InputKey> {
    match key_code {
        winit::keyboard::KeyCode::Escape => Some(InputKey::Escape),
        winit::keyboard::KeyCode::ArrowLeft => Some(InputKey::Left),
        winit::keyboard::KeyCode::ArrowRight => Some(InputKey::Right),
        winit::keyboard::KeyCode::ArrowUp => Some(InputKey::Up),
        winit::keyboard::KeyCode::ArrowDown => Some(InputKey::Down),
        winit::keyboard::KeyCode::KeyW => Some(InputKey::W),
        winit::keyboard::KeyCode::KeyA => Some(InputKey::A),
        winit::keyboard::KeyCode::KeyS => Some(InputKey::S),
        winit::keyboard::KeyCode::KeyD => Some(InputKey::D),
        winit::keyboard::KeyCode::Space => Some(InputKey::Space),
        winit::keyboard::KeyCode::ShiftLeft => Some(InputKey::LeftShift),
        _ => None,
    }
}

pub fn map_winit_mouse_button(button: winit::event::MouseButton) -> Option<MouseButton> {
    match button {
        winit::event::MouseButton::Left => Some(MouseButton::Left),
        winit::event::MouseButton::Right => Some(MouseButton::Right),
        winit::event::MouseButton::Middle => Some(MouseButton::Middle),
        _ => None,
    }
}

pub fn map_winit_element_state(state: winit::event::ElementState) -> MouseState {
    match state {
        winit::event::ElementState::Pressed => MouseState::Pressed,
        winit::event::ElementState::Released => MouseState::Released,
    }
}

pub fn map_winit_scroll_delta_y(delta: winit::event::MouseScrollDelta) -> f32 {
    match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
    }
}
