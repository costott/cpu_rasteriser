//! The application framework: the [`Application`] trait an app implements, and the types the
//! engine passes to it.
//!
//! An [`EngineBackend`](crate::EngineBackend) opens a window and runs an [`Application`]. Each
//! frame it calls [`Application::update`] with the time since the previous frame, then
//! [`Application::render`] with a [`RenderContext`], and shows the presentation target in the
//! window. Window and input events reach [`Application::event`] as [`AppEvent`]s, together with
//! an [`AppHandle`] for asking the engine to exit. [`Application::window_state`] controls the
//! cursor.
//!
//! See [`Application`] for a complete app.

use cpu_rasteriser::prelude::*;

use crate::input::InputEvent;

/// How the window holds on to the mouse cursor.
///
/// Set through [`WindowCursorSettings::grab`]. A grabbed cursor suits first-person camera
/// controls, which read relative motion from [`InputEvent::MouseMotion`].
///
/// Only the winit backend ([`WinitEngine`](crate::WinitEngine)) applies a grab mode; the minifb
/// backend ignores it. The winit backend ignores a failure to apply the mode.
///
/// # Examples
///
/// See [`WindowState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorGrab {
    /// The cursor moves freely and can leave the window.
    None,
    /// The cursor can move but is kept inside the window.
    Confined,
    /// The cursor is held in place. The winit backend falls back to [`CursorGrab::Confined`] on
    /// platforms that can't lock it.
    Locked,
}

/// The window settings an [`Application`] asks for, returned from
/// [`Application::window_state`].
///
/// The backend applies it every frame, so an app changes the window by returning a different
/// value. The default is a visible, ungrabbed cursor.
///
/// # Examples
///
/// Hide and lock the cursor while a first-person camera is active, and release it otherwise:
///
/// ```
/// use engine::prelude::*;
///
/// struct Game {
///     looking_around: bool,
/// }
///
/// impl Application for Game {
///     fn window_state(&self) -> WindowState {
///         if self.looking_around {
///             WindowState {
///                 cursor: WindowCursorSettings {
///                     visible: false,
///                     grab: CursorGrab::Locked,
///                 },
///             }
///         } else {
///             WindowState::default()
///         }
///     }
/// }
///
/// let game = Game { looking_around: true };
/// assert_eq!(game.window_state().cursor.grab, CursorGrab::Locked);
/// assert!(WindowState::default().cursor.visible);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowState {
    /// The cursor's visibility and grab mode.
    pub cursor: WindowCursorSettings,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            cursor: WindowCursorSettings::default(),
        }
    }
}

/// The cursor settings in a [`WindowState`].
///
/// The default is a visible cursor with [`CursorGrab::None`].
///
/// # Examples
///
/// See [`WindowState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCursorSettings {
    /// Whether the cursor is shown over the window. Both backends apply it.
    pub visible: bool,
    /// How the window holds on to the cursor. Only the winit backend applies it.
    pub grab: CursorGrab,
}

impl Default for WindowCursorSettings {
    fn default() -> Self {
        Self {
            visible: true,
            grab: CursorGrab::None,
        }
    }
}

/// An app run by an [`EngineBackend`](crate::EngineBackend).
///
/// Implement the methods the app needs; each has a default that does nothing. Each frame, the
/// backend calls [`update`](Application::update) with the time since the previous frame, then
/// [`render`](Application::render), then shows the presentation target in the window. Between
/// frames it delivers window and input events to [`event`](Application::event), calls
/// [`resize`](Application::resize) when the window's size changes, and applies
/// [`window_state`](Application::window_state).
///
/// The app needn't be `Send`: it runs on the thread that called
/// [`EngineBackend::run`](crate::EngineBackend::run).
///
/// # Examples
///
/// An app that clears the window to a colour and exits when Escape is pressed:
///
/// ```no_run
/// use cpu_rasteriser::prelude::*;
/// use engine::prelude::*;
///
/// struct ClearApp {
///     colour: Colour,
/// }
///
/// impl Application for ClearApp {
///     fn render<'frame>(&mut self, context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
///         let extent = context.presentation_target().extent();
///         let pass = context.begin_presentation_pass(RenderPassDescriptor {
///             viewport: Viewport::full(&extent),
///             colour_load_op: LoadOp::Clear(self.colour),
///             depth_load_op: Some(LoadOp::Clear(1.0)),
///         });
///         // Record draw calls on `pass` here.
///         pass.finish()
///     }
///
///     fn event(&mut self, event: AppEvent, handle: &mut AppHandle) {
///         if let AppEvent::Input(InputEvent::Key {
///             key: InputKey::Escape,
///             state: ButtonState::Pressed,
///         }) = event
///         {
///             handle.request_exit();
///         }
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// MinifbEngine::new()
///     .with_title("Clear")
///     .run(ClearApp { colour: Colour::BLACK })?;
/// # Ok(())
/// # }
/// ```
pub trait Application {
    /// Advances the app's state by `dt` seconds, the time since the previous frame's update.
    ///
    /// Called once per frame, before [`render`](Application::render). Scale movement and
    /// animation by `dt` so that they don't depend on the frame rate. On the first frame, `dt` is
    /// the time since the backend started.
    ///
    /// The default does nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// struct Spinner {
    ///     angle: f32,
    /// }
    ///
    /// impl Application for Spinner {
    ///     fn update(&mut self, dt: f32) {
    ///         // Half a turn per second, whatever the frame rate.
    ///         self.angle += std::f32::consts::PI * dt;
    ///     }
    /// }
    ///
    /// let mut spinner = Spinner { angle: 0.0 };
    /// spinner.update(0.5);
    /// assert_eq!(spinner.angle, std::f32::consts::FRAC_PI_2);
    /// ```
    fn update(&mut self, _dt: f32) {}

    /// Renders a frame through `context` and returns the [`PresentedFrame`] that shows the
    /// window's contents were rendered.
    ///
    /// Called once per frame, after [`update`](Application::update). The app may render any
    /// number of offscreen passes with [`RenderContext::begin_render_pass`], then renders the
    /// window's contents with [`RenderContext::begin_presentation_pass`]. Only
    /// [`PresentationPass::finish`] creates the [`PresentedFrame`] this method returns, so an
    /// implementation can't forget to finish the presentation pass. When `render` returns, the
    /// backend shows the presentation target in the window.
    ///
    /// The default renders nothing and leaves the presentation target as it was.
    ///
    /// # Examples
    ///
    /// See [`Application`].
    fn render<'frame>(&mut self, _context: &'frame mut RenderContext<'frame>) -> PresentedFrame {
        PresentedFrame { _private: () }
    }

    /// Responds to the presentation target being resized to `width` × `height` pixels.
    ///
    /// The backend has already resized the presentation target, and never passes a zero size.
    /// Use it to resize offscreen render targets and update the camera's aspect ratio.
    ///
    /// The winit backend calls it when the window is created or resumed and whenever it is
    /// resized.
    ///
    /// The default does nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use engine::prelude::*;
    ///
    /// struct App {
    ///     aspect_ratio: f32,
    /// }
    ///
    /// impl Application for App {
    ///     fn resize(&mut self, width: u32, height: u32) {
    ///         self.aspect_ratio = width as f32 / height as f32;
    ///     }
    /// }
    ///
    /// let mut app = App { aspect_ratio: 1.0 };
    /// app.resize(640, 320);
    /// assert_eq!(app.aspect_ratio, 2.0);
    /// ```
    ///
    /// # Known issues
    ///
    /// - The minifb backend calls it every frame with the window's current size, whether or not it
    ///   has changed, so keep it cheap when the size is the same
    ///   ([#56](https://github.com/costott/cpu_rasteriser/issues/56)).
    fn resize(&mut self, _width: u32, _height: u32) {}

    /// Handles a window or input event.
    ///
    /// `handle` lets the app ask the engine to exit with [`AppHandle::request_exit`]. Input
    /// events can be recorded in an [`InputState`](crate::input::InputState). See [`AppEvent`]
    /// for which events each backend sends, and when.
    ///
    /// The default ignores every event.
    ///
    /// # Examples
    ///
    /// See [`Application`].
    fn event(&mut self, _event: AppEvent, _handle: &mut AppHandle) {}

    /// Returns the window settings the app currently wants.
    ///
    /// The backend calls it every frame and applies the result, so the settings can change at
    /// any time.
    ///
    /// The default returns [`WindowState::default`]: a visible, ungrabbed cursor.
    ///
    /// # Examples
    ///
    /// See [`WindowState`].
    fn window_state(&self) -> WindowState {
        WindowState::default()
    }
}

/// Access to the renderer and the presentation target during [`Application::render`].
///
/// The presentation target is the [`RenderTarget`] the window shows. It has a depth buffer, and
/// the backend keeps it the size of the window. Only the backend creates a `RenderContext`.
///
/// # Examples
///
/// See [`Application`] for a single presentation pass, and
/// [`begin_render_pass`](RenderContext::begin_render_pass) for rendering offscreen first.
#[derive(Debug)]
pub struct RenderContext<'a> {
    renderer: &'a mut Renderer,
    presentation_target: &'a mut RenderTarget,
}
impl<'a> RenderContext<'a> {
    pub(crate) fn new(
        renderer: &'a mut Renderer,
        presentation_target: &'a mut RenderTarget,
    ) -> Self {
        Self {
            renderer,
            presentation_target,
        }
    }

    /// Returns the presentation target, the [`RenderTarget`] the window shows.
    ///
    /// Its extent is the window's size in pixels, which a pass needs for its [`Viewport`].
    ///
    /// # Examples
    ///
    /// See [`Application`].
    pub fn presentation_target(&self) -> &RenderTarget {
        self.presentation_target
    }

    /// Begins a render pass that draws to the presentation target, the window's contents.
    ///
    /// Finishing the returned pass with [`PresentationPass::finish`] renders it and gives the
    /// [`PresentedFrame`] that [`Application::render`] returns. The presentation target has a
    /// depth buffer, so `descriptor` may set a depth load operation.
    ///
    /// # Examples
    ///
    /// Clear the window to grey and its depth to the far plane:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// struct GreyApp;
    ///
    /// impl Application for GreyApp {
    ///     fn render<'frame>(
    ///         &mut self,
    ///         context: &'frame mut RenderContext<'frame>,
    ///     ) -> PresentedFrame {
    ///         let extent = context.presentation_target().extent();
    ///         let pass = context.begin_presentation_pass(RenderPassDescriptor {
    ///             viewport: Viewport::full(&extent),
    ///             colour_load_op: LoadOp::Clear(Colour::new(0.5, 0.5, 0.5, 1.0)),
    ///             depth_load_op: Some(LoadOp::Clear(1.0)),
    ///         });
    ///         pass.finish()
    ///     }
    /// }
    /// ```
    pub fn begin_presentation_pass<'pass>(
        &'pass mut self,
        descriptor: RenderPassDescriptor,
    ) -> PresentationPass<'pass, 'pass> {
        PresentationPass::new(
            self.renderer
                .begin_render_pass(self.presentation_target, descriptor),
        )
    }

    /// Begins a render pass that draws to `target` instead of the window.
    ///
    /// Use it for multipass rendering: render a scene into an offscreen target, copy it into a
    /// [`Texture`](crate::components::texture::Texture), and sample that in the presentation
    /// pass. Finish the pass with [`RenderPass::finish`] before beginning the next one.
    ///
    /// # Panics
    ///
    /// Panics if `descriptor` sets a depth load operation and `target` has no depth buffer.
    ///
    /// # Examples
    ///
    /// Render into an offscreen target, then present a copy of it as a texture:
    ///
    /// ```
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// struct PostProcessApp {
    ///     scene_target: RenderTarget,
    ///     scene_texture: Texture,
    /// }
    ///
    /// impl Application for PostProcessApp {
    ///     fn render<'frame>(
    ///         &mut self,
    ///         context: &'frame mut RenderContext<'frame>,
    ///     ) -> PresentedFrame {
    ///         let extent = self.scene_target.extent();
    ///         let scene_pass = context.begin_render_pass(
    ///             &mut self.scene_target,
    ///             RenderPassDescriptor {
    ///                 viewport: Viewport::full(&extent),
    ///                 colour_load_op: LoadOp::Clear(Colour::BLACK),
    ///                 depth_load_op: Some(LoadOp::Clear(1.0)),
    ///             },
    ///         );
    ///         // Draw the scene on `scene_pass` here.
    ///         scene_pass.finish();
    ///
    ///         self.scene_texture.update_from(&self.scene_target);
    ///
    ///         let extent = context.presentation_target().extent();
    ///         let presentation_pass = context.begin_presentation_pass(RenderPassDescriptor {
    ///             viewport: Viewport::full(&extent),
    ///             colour_load_op: LoadOp::Clear(Colour::BLACK),
    ///             depth_load_op: None,
    ///         });
    ///         // Draw a fullscreen quad that samples `self.scene_texture` here.
    ///         presentation_pass.finish()
    ///     }
    /// }
    /// ```
    pub fn begin_render_pass<'pass>(
        &'pass mut self,
        target: &'pass mut RenderTarget,
        descriptor: RenderPassDescriptor,
    ) -> RenderPass<'pass, 'pass> {
        self.renderer.begin_render_pass(target, descriptor)
    }
}

/// A [`RenderPass`] that draws to the presentation target, the window's contents.
///
/// Begun with [`RenderContext::begin_presentation_pass`]. It dereferences to [`RenderPass`], so
/// draw calls are recorded on it directly, and a model can draw to it with
/// [`Model::draw_to_render_pass`](crate::components::model::Model::draw_to_render_pass). Unlike
/// [`RenderPass::finish`], [`finish`](PresentationPass::finish) returns the [`PresentedFrame`]
/// that [`Application::render`] must return.
///
/// # Examples
///
/// See [`Application`].
#[derive(Debug)]
pub struct PresentationPass<'a, 'b> {
    render_pass: RenderPass<'a, 'b>,
}
impl<'a, 'b> PresentationPass<'a, 'b> {
    pub(crate) fn new(render_pass: RenderPass<'a, 'b>) -> Self {
        Self { render_pass }
    }

    /// Returns the underlying render pass.
    ///
    /// The pass also dereferences to it, so this is only needed where a `&mut RenderPass` must be
    /// named explicitly.
    ///
    /// # Examples
    ///
    /// Handing the pass to a helper that draws into any [`RenderPass`]:
    ///
    /// ```no_run
    /// use cpu_rasteriser::prelude::*;
    /// use engine::prelude::*;
    ///
    /// fn draw_scene(_pass: &mut RenderPass<'_, '_>) {
    ///     // ... record draw calls ...
    /// }
    ///
    /// struct App;
    /// impl Application for App {
    ///     fn render<'frame>(
    ///         &mut self,
    ///         context: &'frame mut RenderContext<'frame>,
    ///     ) -> PresentedFrame {
    ///         let extent = context.presentation_target().extent();
    ///         let mut pass = context.begin_presentation_pass(RenderPassDescriptor {
    ///             viewport: Viewport::full(&extent),
    ///             colour_load_op: LoadOp::Clear(Colour::BLACK),
    ///             depth_load_op: Some(LoadOp::Clear(1.0)),
    ///         });
    ///         draw_scene(pass.render_pass_mut());
    ///         pass.finish()
    ///     }
    /// }
    /// ```
    pub fn render_pass_mut(&mut self) -> &mut RenderPass<'a, 'b> {
        &mut self.render_pass
    }

    /// Renders the pass's draw calls into the presentation target, like [`RenderPass::finish`],
    /// and returns the proof that the frame was presented.
    ///
    /// # Examples
    ///
    /// See [`Application`].
    pub fn finish(self) -> PresentedFrame {
        self.render_pass.finish();
        PresentedFrame { _private: () }
    }
}
impl<'a, 'b> std::ops::Deref for PresentationPass<'a, 'b> {
    type Target = RenderPass<'a, 'b>;

    fn deref(&self) -> &Self::Target {
        &self.render_pass
    }
}
impl<'a, 'b> std::ops::DerefMut for PresentationPass<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.render_pass
    }
}
/// The value [`Application::render`] returns to show that it finished a [`PresentationPass`].
///
/// Only [`PresentationPass::finish`] creates one, so the compiler checks that `render` rendered
/// the window's contents. It carries no data.
///
/// # Examples
///
/// See [`Application`].
#[derive(Debug)]
pub struct PresentedFrame {
    _private: (),
}

/// A window or input event delivered to [`Application::event`].
///
/// The two backends send different subsets; each variant says which backend sends it, and when.
///
/// # Examples
///
/// See [`Application`].
///
/// # Known issues
///
/// - The minifb backend sends [`AppEvent::Resized`] every frame with the window's current size,
///   whether or not it has changed ([#56](https://github.com/costott/cpu_rasteriser/issues/56)).
/// - The minifb backend sends [`AppEvent::CloseRequested`] once its loop has ended, however it
///   ended, including after [`AppHandle::request_exit`]
///   ([#57](https://github.com/costott/cpu_rasteriser/issues/57)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppEvent {
    /// The window is closing.
    ///
    /// The winit backend sends it when the user closes the window, after which the engine exits
    /// whatever the app does. The minifb backend sends it when the window is closed.
    CloseRequested,
    /// The window was resized to `width` × `height` pixels, after [`Application::resize`].
    ///
    /// Both backends send it when the window is resized.
    Resized {
        /// The new width in pixels.
        width: u32,
        /// The new height in pixels.
        height: u32,
    },
    /// A frame was rendered and presented. Only the winit backend sends it.
    RedrawRequested,
    /// The window's surface was released, for example when an app is sent to the background on
    /// a mobile platform. Only the winit backend sends it; no frames are rendered until
    /// [`AppEvent::Resumed`].
    Suspended,
    /// The window is ready to render. The winit backend sends it when the window's surface is
    /// created or recreated; the minifb backend sends it once, before the first frame.
    Resumed,
    /// A keyboard or mouse event.
    Input(InputEvent),
}

/// Lets [`Application::event`] ask the engine to act.
///
/// # Examples
///
/// ```
/// use engine::prelude::*;
///
/// let mut handle = AppHandle::default();
/// assert!(!handle.should_exit());
///
/// handle.request_exit();
/// assert!(handle.should_exit());
/// ```
#[derive(Debug)]
pub struct AppHandle {
    exit_requested: bool,
}
impl AppHandle {
    /// Asks the engine to exit.
    ///
    /// The backend stops running the app soon after, and
    /// [`EngineBackend::run`](crate::EngineBackend::run) returns `Ok(())`. The request can't be
    /// withdrawn.
    ///
    /// # Examples
    ///
    /// See [`Application`].
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    /// Returns whether [`request_exit`](AppHandle::request_exit) has been called.
    ///
    /// # Examples
    ///
    /// See [`AppHandle`].
    pub fn should_exit(&self) -> bool {
        self.exit_requested
    }
}
impl Default for AppHandle {
    fn default() -> Self {
        Self {
            exit_requested: false,
        }
    }
}
