use cpu_rasteriser::prelude::*;

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 360;
const HEIGHT: usize = 360;

#[derive(Clone, Copy, Debug)]
pub struct MandelbulbVertex {
    pub position: Vec3,
}

#[derive(Interpolate)]
pub struct MandelbulbVaryings {
    pub ray_point: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct MandelbulbVertexUniforms {
    pub camera_position: Vec3,
    pub aspect_ratio: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MandelbulbFragmentUniforms {
    pub camera_position: Vec3,
    pub light_position: Vec3,

    pub power: f32,
    pub iterations: u32,

    pub max_distance: f32,
    pub hit_threshold: f32,
    pub max_steps: u32,

    pub time: f32,
}

/// Vertex shader for the fullscreen Mandelbulb pass.
pub struct MandelbulbVertexShader;

impl VertexShader for MandelbulbVertexShader {
    type Vertex = MandelbulbVertex;
    type Uniforms = MandelbulbVertexUniforms;
    type Varyings = MandelbulbVaryings;

    fn shade(&self, vertex: Self::Vertex, uniforms: &Self::Uniforms) -> (Vec4, Self::Varyings) {
        let clip_position = Vec4::new(vertex.position.x, vertex.position.y, 0.0, 1.0);

        let ray_point = uniforms.camera_position
            + Vec3::new(
                vertex.position.x * uniforms.aspect_ratio,
                vertex.position.y,
                -1.0,
            );

        (clip_position, MandelbulbVaryings { ray_point })
    }
}

/// Fragment shader implementing a distance-estimated Mandelbulb.
pub struct MandelbulbFragmentShader;

impl MandelbulbFragmentShader {
    fn distance_estimator(position: Vec3, power: f32, iterations: u32, time: f32) -> f32 {
        let position = Self::rotate_y(position, time * 0.5);

        let mut z = position;
        let mut derivative = 1.0;
        let mut radius = 0.0;

        for _ in 0..iterations {
            radius = z.length();

            if radius > 2.0 {
                break;
            }

            let safe_radius = radius.max(1e-6);

            let theta = (z.z / safe_radius).clamp(-1.0, 1.0).acos();

            let phi = z.y.atan2(z.x);

            let radius_power = safe_radius.powf(power);

            derivative = power * safe_radius.powf(power - 1.0) * derivative + 1.0;

            let theta = theta * power;
            let phi = phi * power;

            let sin_theta = theta.sin();

            z = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos()) * radius_power
                + position;
        }

        if radius <= 1e-6 {
            return 0.0;
        }

        0.5 * radius.ln() * radius / derivative
    }
    fn ray_march(
        origin: Vec3,
        direction: Vec3,
        uniforms: &MandelbulbFragmentUniforms,
    ) -> Option<Vec3> {
        const BOUNDING_RADIUS: f32 = 2.0;

        // Intersect the ray with the bounding sphere first.
        //
        // This prevents us from evaluating the expensive Mandelbulb DE
        // for points that are definitely nowhere near the fractal.
        let b = origin.dot(&direction);
        let c = origin.dot(&origin) - BOUNDING_RADIUS * BOUNDING_RADIUS;

        let discriminant = b * b - c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();

        let mut distance = -b - sqrt_discriminant;
        let exit_distance = -b + sqrt_discriminant;

        if exit_distance < 0.0 {
            return None;
        }

        distance = distance.max(0.0);

        for _ in 0..uniforms.max_steps {
            if distance > exit_distance || distance > uniforms.max_distance {
                return None;
            }

            let position = origin + direction * distance;
            let de = Self::distance_estimator(
                position,
                uniforms.power,
                uniforms.iterations,
                uniforms.time,
            );

            if de.abs() < uniforms.hit_threshold {
                return Some(position);
            }

            distance += de.max(1e-4);
        }

        None
    }

    /// Estimate the surface normal from the gradient of the distance field.
    fn estimate_normal(position: Vec3, uniforms: &MandelbulbFragmentUniforms) -> Vec3 {
        let epsilon = 0.0005;

        let x = Self::distance_estimator(
            position + Vec3::new(epsilon, 0.0, 0.0),
            uniforms.power,
            uniforms.iterations,
            uniforms.time,
        ) - Self::distance_estimator(
            position - Vec3::new(epsilon, 0.0, 0.0),
            uniforms.power,
            uniforms.iterations,
            uniforms.time,
        );

        let y = Self::distance_estimator(
            position + Vec3::new(0.0, epsilon, 0.0),
            uniforms.power,
            uniforms.iterations,
            uniforms.time,
        ) - Self::distance_estimator(
            position - Vec3::new(0.0, epsilon, 0.0),
            uniforms.power,
            uniforms.iterations,
            uniforms.time,
        );

        let z = Self::distance_estimator(
            position + Vec3::new(0.0, 0.0, epsilon),
            uniforms.power,
            uniforms.iterations,
            uniforms.time,
        ) - Self::distance_estimator(
            position - Vec3::new(0.0, 0.0, epsilon),
            uniforms.power,
            uniforms.iterations,
            uniforms.time,
        );

        Vec3::new(x, y, z).normalise()
    }

    /// Simple diffuse + specular lighting.
    fn shade_surface(
        position: Vec3,
        ray_direction: Vec3,
        uniforms: &MandelbulbFragmentUniforms,
    ) -> Colour {
        let normal = Self::estimate_normal(position, uniforms);

        let light_direction = (uniforms.light_position - position).normalise();

        let view_direction = -ray_direction;

        let diffuse = normal.dot(&light_direction).max(0.0);

        let half_vector = (light_direction + view_direction).normalise();

        let specular = normal.dot(&half_vector).max(0.0).powf(32.0);

        // Fresnel/rim term: grazing angles (normal nearly perpendicular to
        // view) get brighter. Power controls how tight the rim is.
        let fresnel = (1.0 - normal.dot(&view_direction).max(0.0)).powf(2.5);

        let ambient = 0.06;

        let intensity = ambient + diffuse * 0.85 + specular * 0.2;

        let base_colour = Vec3::new(0.12, 0.55, 0.95);

        let glow_colour = Vec3::new(0.3, 0.7, 1.0);

        let colour = base_colour * intensity
            + Vec3::new(1.0, 1.0, 1.0) * specular * 0.5
            + glow_colour * fresnel * 0.9;

        colour.into()
    }

    /// Background colour for rays that miss the fractal.
    fn shade_background(ray_direction: Vec3) -> Colour {
        let t = 0.5 * (ray_direction.y + 1.0);

        let bottom: Vec4 = Colour::new(0.0, 0.0, 0.0, 1.0).into();

        let top: Vec4 = Colour::new(1.0, 1.0, 1.0, 1.0).into();

        let colour = bottom * (1.0 - t) + top * t;

        colour.into()
    }

    fn rotate_y(position: Vec3, angle: f32) -> Vec3 {
        let cos = angle.cos();
        let sin = angle.sin();

        Vec3::new(
            cos * position.x + sin * position.z,
            position.y,
            -sin * position.x + cos * position.z,
        )
    }
}

impl FragmentShader<MandelbulbVaryings> for MandelbulbFragmentShader {
    type Uniforms = MandelbulbFragmentUniforms;

    fn shade(&self, varyings: MandelbulbVaryings, uniforms: &Self::Uniforms) -> Colour {
        let ray_origin = uniforms.camera_position;

        let ray_direction = (varyings.ray_point - ray_origin).normalise();

        match Self::ray_march(ray_origin, ray_direction, uniforms) {
            Some(position) => Self::shade_surface(position, ray_direction, uniforms),

            None => Self::shade_background(ray_direction),
        }
    }
}

pub fn mandelbulb_vertices() -> [MandelbulbVertex; 6] {
    [
        MandelbulbVertex {
            position: Vec3::new(-1.0, -1.0, 0.0),
        },
        MandelbulbVertex {
            position: Vec3::new(1.0, -1.0, 0.0),
        },
        MandelbulbVertex {
            position: Vec3::new(1.0, 1.0, 0.0),
        },
        MandelbulbVertex {
            position: Vec3::new(-1.0, -1.0, 0.0),
        },
        MandelbulbVertex {
            position: Vec3::new(1.0, 1.0, 0.0),
        },
        MandelbulbVertex {
            position: Vec3::new(-1.0, 1.0, 0.0),
        },
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = Window::new(
        "Mandelbulb Demo - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )?;

    window.set_target_fps(60);

    let mut renderer = Renderer::new()?;

    let extent = Extent::new(WIDTH, HEIGHT);
    let mut screen_target = RenderTarget::new(extent);

    let mandelbulb_pipeline = Pipeline::new(MandelbulbVertexShader, MandelbulbFragmentShader);

    let camera_position = Vec3::new(0.0, 0.0, 1.7);

    let start_time = std::time::Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let vertex_uniforms = MandelbulbVertexUniforms {
            camera_position,
            aspect_ratio: WIDTH as f32 / HEIGHT as f32,
        };

        let fragment_uniforms = MandelbulbFragmentUniforms {
            camera_position,

            light_position: Vec3::new(-2.0, 2.0, 2.0),

            power: 8.0 + (4.0 * (start_time.elapsed().as_secs_f32() * 0.3).sin()),
            iterations: 16,

            max_distance: 10.0,
            hit_threshold: 0.001,
            max_steps: 128,

            time: start_time.elapsed().as_secs_f32(),
        };

        let vertices = mandelbulb_vertices();

        let mut frame = renderer.begin_render_pass(
            &mut screen_target,
            RenderPassDescriptor {
                viewport: Viewport::full(&extent),
                colour_load_op: LoadOp::Clear(Colour::BLACK),
                depth_load_op: None,
            },
        );

        frame.draw(
            &mandelbulb_pipeline,
            DrawCall::new(
                &vertices,
                &[0, 1, 2, 3, 4, 5],
                PrimitiveMode::TRIANGLES,
                fragment_uniforms,
            ),
            vertex_uniforms,
        );

        frame.finish();

        window.update_with_buffer(&screen_target.pixels_u32(), WIDTH, HEIGHT)?;
    }

    Ok(())
}
