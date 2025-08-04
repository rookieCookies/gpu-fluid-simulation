pub mod renderer;
pub mod buffer;
pub mod uniform;
pub mod shader;
pub mod egui_tools;

use std::time::Instant;

use egui_wgpu::wgpu;
use glam::{Vec2, Vec4};
use winit::{application::ApplicationHandler, dpi::LogicalSize, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::Window};

use crate::renderer::Renderer;


struct App {
    renderer: Option<Renderer>,
    state: SimulationState,
    last_frame: Instant,
    time_since_last_sim: f32,
}



enum SimulationState {
    Running,
    Step,
    Stopped,
}



impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(Window::default_attributes().with_inner_size(LogicalSize::new(960, 540))).unwrap();

        self.renderer = Some(pollster::block_on(Renderer::new(window)));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut()
        else { return };

        renderer
            .egui
            .handle_input(renderer.window, &event);



        match event {
            winit::event::WindowEvent::RedrawRequested => {
                let dt = self.last_frame.elapsed().as_secs_f32();
                self.last_frame = Instant::now();

                self.time_since_last_sim += dt;

                renderer.device.poll(wgpu::PollType::Poll).unwrap();

                renderer.render();

                match self.state {
                    SimulationState::Running => {
                        if renderer.simulation_uniform.delta != 0.0 {
                            while self.time_since_last_sim > renderer.simulation_uniform.delta {
                                renderer.simulate();
                                self.time_since_last_sim -= renderer.simulation_uniform.delta;
                            }

                        }
                    },

                    SimulationState::Step => {
                        renderer.simulate();
                        self.state = SimulationState::Stopped;
                    },

                    SimulationState::Stopped => (),
                };
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

    let mut app = App {
        renderer: None,
        state: SimulationState::Stopped,
        last_frame: Instant::now(),
        time_since_last_sim: 0.0,
    };

    event_loop.run_app(&mut app).unwrap();
}
