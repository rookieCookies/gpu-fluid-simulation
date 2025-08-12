pub mod renderer;
pub mod buffer;
pub mod uniform;
pub mod shader;
pub mod egui_tools;

use std::{path::Path, time::Instant};

use egui_wgpu::wgpu;
use ffmpeg_next::{codec, decoder::Video, format::{self, context::{input::PacketIter, Input}, Pixel}, software::scaling::Context, Rational};
use glam::{UVec2, Vec2, Vec4};
use image::{GrayImage, ImageBuffer, RgbImage, RgbaImage};
use wgpu::{hal::CommandEncoder, CommandEncoderDescriptor};
use winit::{application::ApplicationHandler, dpi::LogicalSize, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::Window};
use ffmpeg_next as ffmpeg;

use crate::renderer::{Renderer, RENDER_DIMS};


struct App {
    renderer: Option<Renderer>,
    state: SimulationState,
    last_frame: Instant,
    time_since_last_sim: f32,

    packets: PacketIter<'static>,
    input: Input,
    decoder: Video,
    scaler: Context,

    frame_index: u32,
    max_frame_count: u32,
    start: Instant,
}



enum SimulationState {
    Running,
    Render,
    Step,
    Stopped,
}



impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(Window::default_attributes().with_inner_size(LogicalSize::new(960, 540))).unwrap();
        self.renderer = Some(pollster::block_on(Renderer::new(window, UVec2::new(self.decoder.width(), self.decoder.height()))));
    }


    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut()
        else { return };

        if renderer
            .egui
            .handle_input(renderer.window, &event)
            .consumed { return };



        match event {
            winit::event::WindowEvent::RedrawRequested => {
                let time = Instant::now();
                let elapsed = self.last_frame.elapsed();
                let dt = elapsed.as_secs_f32();
                self.last_frame = Instant::now();

                self.time_since_last_sim += dt;

                //renderer.staging_belt.recall();
                let mut encoder = renderer.device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("encoder"),
                });
                renderer.render(encoder);

                let mut encoder = renderer.device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("encoder"),
                });


                match self.state {
                    SimulationState::Running => {
                        if renderer.simulation_uniform.delta != 0.0 {
                            let packet = self.packets.next().unwrap();
                            self.decoder.send_packet(&packet.1);

                            let mut decoded = ffmpeg::frame::Video::empty();
                            while self.decoder.receive_frame(&mut decoded).is_ok() {
                                let mut gray_frame = ffmpeg::frame::Video::empty();
                                self.scaler.run(&decoded, &mut gray_frame).unwrap();

                                // Convert the FFMpeg frame to an `image::GrayImage`
                                let gray_image = GrayImage::from_raw(
                                    gray_frame.width(),
                                    gray_frame.height(),
                                    gray_frame.data(0).to_vec(),
                                )
                                .unwrap();

                                let smooth = generate_smooth_gradient_field(&gray_image);


                                renderer.queue.write_buffer(
                                    &renderer.particle_pipeline.texture,
                                    0,
                                    bytemuck::cast_slice(&smooth),
                                );



                                println!("frame {}", self.frame_index);
                                self.frame_index += 1;

                            }


                            let mut count = 0;
                            let time = Instant::now();

                            while self.time_since_last_sim > renderer.simulation_uniform.delta {

                                renderer.simulate(&mut encoder);

                                self.time_since_last_sim -= renderer.simulation_uniform.delta;
                                count += 1;
                                if count == 5 {
                                    println!("dropped frames {}", self.time_since_last_sim/renderer.simulation_uniform.delta);
                                    self.time_since_last_sim = 0.0;
                                }
                            }

                        }
                    },


                    SimulationState::Render => {
                        let packet = match self.packets.next() {
                            Some(v) => v,
                            None => {
                                println!("completed at {} frames", self.frame_index);
                                self.state = SimulationState::Stopped;
                                return;
                            },
                        };

                        self.decoder.send_packet(&packet.1);

                        let mut decoded = ffmpeg::frame::Video::empty();
                        while self.decoder.receive_frame(&mut decoded).is_ok() {
                            let mut gray_frame = ffmpeg::frame::Video::empty();
                            self.scaler.run(&decoded, &mut gray_frame).unwrap();

                            // Convert the FFMpeg frame to an `image::GrayImage`
                            let gray_image = GrayImage::from_raw(
                                gray_frame.width(),
                                gray_frame.height(),
                                gray_frame.data(0).to_vec(),
                            )
                            .unwrap();

                            let smooth = generate_smooth_gradient_field(&gray_image);

                            let path = format!("output/frame_gradient{:0>5}.png", self.frame_index);
                            gray_image.save(path).unwrap();



                            renderer.queue.write_buffer(
                                &renderer.particle_pipeline.texture,
                                0,
                                bytemuck::cast_slice(&smooth),
                            );



                            self.frame_index += 1;

                        }

                        for _ in 0..16 {
                            renderer.simulate(&mut encoder);
                        }

                        let image = renderer.render_for_video();
                        let buf : RgbaImage = ImageBuffer::from_vec(RENDER_DIMS.x, RENDER_DIMS.y, image).unwrap();
                        let path = format!("output/frame_{:0>5}.png", self.frame_index);
                        buf.save(path).unwrap();

                        let elapsed = self.start.elapsed();
                        let time = elapsed.div_f64(self.frame_index as f64).min(1.0);
                        let time = time.mul_f64(self.max_frame_count as f64);
                        let time = time - elapsed;
                        println!("saved frame {}/{}, elapsed: {:?}, estimate left: {:?}", self.frame_index, self.max_frame_count, elapsed, time);

                    }

                    SimulationState::Step => {
                        renderer.simulate(&mut encoder);
                        self.state = SimulationState::Stopped;
                    },

                    SimulationState::Stopped => (),
                };

                renderer.queue.submit(core::iter::once(encoder.finish()));

                renderer.window.request_redraw();
            },

            
            winit::event::WindowEvent::CloseRequested => {
                println!("closing");
                event_loop.exit();
            },



            winit::event::WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                if !event.state.is_pressed() {
                    return;
                }


                match event.physical_key {
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) => {
                        match self.state {
                            SimulationState::Stopped => {
                                self.time_since_last_sim = 0.0;
                                self.state = SimulationState::Running
                            },
                            _ => self.state = SimulationState::Stopped,
                        }
                    },
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyN) => {
                        self.state = SimulationState::Step;
                    },



                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Enter) => {
                        self.state = SimulationState::Render;
                        renderer.restart_simulation();
                        self.start = Instant::now();
                        self.decoder.send_eof();
                        self.decoder.flush();
                        self.packets = unsafe { core::mem::transmute(self.input.packets()) };
                        self.frame_index = 0;
                    }
                    _ => (),
                }
            }


            winit::event::WindowEvent::MouseInput { device_id, state, button } => {
                let renderer = self.renderer.as_mut().unwrap();
                match (state, button) {
                    (winit::event::ElementState::Pressed, winit::event::MouseButton::Left) => renderer.simulation_uniform.mouse_state = -1,
                    (winit::event::ElementState::Pressed, winit::event::MouseButton::Right) => renderer.simulation_uniform.mouse_state = 1,
                    (winit::event::ElementState::Released, winit::event::MouseButton::Left) => renderer.simulation_uniform.mouse_state = 0,
                    (winit::event::ElementState::Released, winit::event::MouseButton::Right) => renderer.simulation_uniform.mouse_state = 0,
                    _ => (),
                }
            }


            winit::event::WindowEvent::CursorMoved { device_id, position } => {
                let renderer = self.renderer.as_mut().unwrap();
                let position = Vec2::new(position.x as f32, position.y as f32);


                let window_size = renderer.window.inner_size();
                let window_size = Vec2::new(window_size.width as f32, window_size.height as f32);
                let ndc = (position / window_size) * 2.0 - 1.0;
                let clip_pos = Vec4::new(ndc.x, -ndc.y, 0.0, 1.0);
                let inv_proj = renderer.projection.inverse();

                let world_pos = inv_proj * clip_pos;
                let world_pos = world_pos.truncate() / world_pos.w;
                let world_pos = world_pos.truncate();

                renderer.simulation_uniform.mouse_pos = world_pos;
            }



            winit::event::WindowEvent::Resized(size) => {
                renderer.resize_surface(size.width, size.height);
            }



            _ => (),
        }
    }


}



fn main() {
    tracing_subscriber::fmt().init();

    let event_loop = EventLoop::builder().build().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);


    ffmpeg::init().unwrap();

    let input_path = "input.mp4";
    let mut ictx = ffmpeg::format::input(&Path::new(input_path)).unwrap();

    let input_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound).unwrap();

    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input_stream.parameters()).unwrap();
    let mut decoder = context_decoder.decoder().video().unwrap();

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::GRAY8, // We want grayscale frames
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    ).unwrap();


    let frame_count = input_stream.frames();


    let mut app = App {
        renderer: None,
        state: SimulationState::Stopped,
        last_frame: Instant::now(),
        time_since_last_sim: 0.0,
        packets: unsafe { core::mem::transmute(ictx.packets()) },
        decoder,
        scaler,
        frame_index: 0,
        input: ictx,
        max_frame_count: frame_count as u32,
        start: Instant::now(),
    };

    event_loop.run_app(&mut app).unwrap();
}



fn generate_smooth_gradient_field(img: &GrayImage) -> Vec<Vec2>{
    let width = img.width() as usize;
    let height = img.height() as usize;

    // Distance buffer, initialized to a large value
    let mut dist = vec![vec![f32::MAX; width]; height];
    // Nearest source pixel coords for each pixel
    let mut nearest = vec![vec![(0usize, 0usize); width]; height];

    // Step 1: Initialization
    let mut has_white = false;
    for y in 0..height {
        for x in 0..width {
            if img.get_pixel(x as u32, y as u32)[0] > 128 {
                dist[y][x] = 0.0;
                nearest[y][x] = (x, y);
                has_white = true;
            }

        }
    }

    if !has_white {
        for y in 0..height {
            for x in 0..width {
                if y == height-1 || y == 0 || x == width-1 || x == 0 {
                    dist[y][x] = 0.0;
                    nearest[y][x] = (x, y);
                }


            }
        }


    }


    // Helper to compute squared Euclidean distance
    let squared_dist = |x1, y1, x2, y2| -> f32 {
        let dx = x1 as f32 - x2 as f32;
        let dy = y1 as f32 - y2 as f32;
        dx*dx + dy*dy
    };

    // Step 2: Forward pass
    for y in 0..height {
        for x in 0..width {
            // Check neighbors: left, top-left, top, top-right
            for &(nx, ny) in &[
                (x.wrapping_sub(1), y), 
                (x.wrapping_sub(1), y.wrapping_sub(1)), 
                (x, y.wrapping_sub(1)), 
                (x + 1, y.wrapping_sub(1))
            ] {
                if nx < width && ny < height {
                    let candidate = nearest[ny][nx];
                    let candidate_dist = squared_dist(x, y, candidate.0, candidate.1);
                    if candidate_dist < dist[y][x] {
                        dist[y][x] = candidate_dist;
                        nearest[y][x] = candidate;
                    }
                }
            }
        }
    }

    // Step 3: Backward pass
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            // Check neighbors: right, bottom-right, bottom, bottom-left
            for &(nx, ny) in &[
                (x + 1, y), 
                (x + 1, y + 1), 
                (x, y + 1), 
                (x.wrapping_sub(1), y + 1)
            ] {
                if nx < width && ny < height {
                    let candidate = nearest[ny][nx];
                    let candidate_dist = squared_dist(x, y, candidate.0, candidate.1);
                    if candidate_dist < dist[y][x] {
                        dist[y][x] = candidate_dist;
                        nearest[y][x] = candidate;
                    }
                }
            }
        }
    }

    // Create RGB image for gradient visualization
    let mut rgb_img = vec![Vec2::ZERO; width*height];

    for y in 0..height {
        for x in 0..width {
            let (nx, ny) = nearest[y][x];
            let dir_x = nx as f32 - x as f32;
            let dir_y = ny as f32 - y as f32;

            // Normalize direction vector
            let length = (dir_x * dir_x + dir_y * dir_y).sqrt().max(1e-5);
            let nx = dir_x / length;
            let ny = dir_y / length;


            // Scale by distance with some smoothing, clamp max to 1.0
            let scale = (1.0 - (length / 100.0).clamp(0.5, 1.0));

            // Map from [-1, 1] to [0, 255]
            rgb_img[y*width+x] = Vec2::new(nx, ny) * scale;
        }
    }

    rgb_img
}


