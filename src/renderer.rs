use std::num::NonZero;

use bytemuck::{offset_of, Pod, Zeroable};
use glam::{IVec2, IVec3, IVec4, Mat4, Vec2, Vec3, Vec4};
use rand::Rng;
use wgpu::{util::{DeviceExt, StagingBelt}, BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BindingType, Buffer, BufferUsages, ComputePipeline, PrimitiveState, PrimitiveTopology, RenderPassDepthStencilAttachment, ShaderStages, TextureUsages, TextureViewDescriptor, VertexAttribute, VertexBufferLayout};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{buffer::ResizableBuffer, shader::create_shader_module, uniform::Uniform};


const MSAA_SAMPLE_COUNT : u32 = 1;
const PARTICLE_SPACING : f32 = 0.4;
const PARTICLE_COUNT : u64 = 8000;
const SIZE : Vec2 = Vec2::new(53.0, 30.0);


pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub window: &'static Window,

    pub framebuffer: wgpu::TextureView,
    pub staging_belt: StagingBelt,

    pub particle_pipeline: ParticlePipeline,
    pub simulation_pipeline: SimulationPipeline,

    pub frame_time: u32,

    pub mouse_pos: Vec2,
    pub mouse_state: i32,


    pub projection: Mat4,
}


#[derive(Clone, Copy, Zeroable, Pod, Default, Debug)]
#[repr(C)]
struct ParticleInstance {
    position: Vec2,
    predicted_position: Vec2,
    velocity: Vec2,
    density: f32,
    pad00: f32,
}


#[derive(Clone, Copy, Zeroable, Pod, Debug)]
#[repr(C)]
struct ParticleUniform {
    projection: Mat4,
    pad00: Vec3,
    scale: f32,
    colour0: Vec4,
    colour1: Vec4,
    colour2: Vec4,
    colour3: Vec4,
}



#[derive(Clone, Copy, Zeroable, Pod, Debug)]
#[repr(C)]
struct SimulationUniform {
    delta: f32,
    particle_count: u32,
    gravity: Vec2,
    bounds: Vec2,
    sqr_radius: f32,
    frame_time: u32,
    smoothing_radius: f32,
    particle_mass: f32,
    pressure_constant: f32,
    rest_density: f32,
    damping_factor: f32,
    viscosity_coefficient: f32,
    surface_tension_treshold: f32,
    surface_tension_coefficient: f32,

    mouse_pos: Vec2,
    mouse_state: i32,

    mouse_force_radius: f32,
    mouse_force_power: f32,
    pad: f32,
}


const QUAD_VERTICES : &[ParticleVertex] = &[
    ParticleVertex { pos: Vec2::new(1.0, 1.0) },
    ParticleVertex { pos: Vec2::new(1.0, 0.0) },
    ParticleVertex { pos: Vec2::new(0.0, 0.0) },
    ParticleVertex { pos: Vec2::new(0.0, 0.0) },
    ParticleVertex { pos: Vec2::new(0.0, 1.0) },
    ParticleVertex { pos: Vec2::new(1.0, 1.0) },

];


pub struct ParticlePipeline {
    uniform: Uniform<ParticleUniform>,
    render_pipeline: wgpu::RenderPipeline,
    debug_render_pipeline: wgpu::RenderPipeline,
    vertices: Buffer,
    instances: Buffer,
    side_buffer: Buffer,
}


pub struct SimulationPipeline {
    compute_pipeline: Box<[ComputePipeline]>,
    uniform: Uniform<SimulationUniform>,
    main_bg: BindGroup,
    swapped_bg: BindGroup,
}



impl Renderer {
    pub async fn new(window: Window) -> Self {
        let window = Box::leak(Box::new(window));

        let size = PhysicalSize::new(960, 540);
        let _ = window.request_inner_size(size);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            ..Default::default()
        });

        dbg!(&window);
        let surface = instance.create_surface(&*window).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::POLYGON_MODE_LINE
                                    | wgpu::Features::TEXTURE_BINDING_ARRAY
                                    | wgpu::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                                    | wgpu::Features::MULTI_DRAW_INDIRECT
                                    | wgpu::Features::INDIRECT_FIRST_INSTANCE
                                    | wgpu::Features::TIMESTAMP_QUERY,
                required_limits: {
                    let mut limits = wgpu::Limits::downlevel_defaults();
                    limits.max_buffer_size = adapter.limits().max_buffer_size;
                    limits.max_storage_buffer_binding_size = 512 << 20;
                    limits.max_texture_dimension_2d = 8192;
                    limits
                },
                label: Some("main device"),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ).await.unwrap();

        let surface_capabilities = surface.get_capabilities(&adapter);

        let surface_format = surface_capabilities.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_capabilities.formats[0]);


        let config = wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let framebuffer = create_multisampled_framebuffer(&device, &config);


        
        let particles = {
            let shader = create_shader_module(&device, wgpu::ShaderModuleDescriptor {
                label: Some("particle-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../particle_shader.wgsl").into()),
            });
            let debug_shader = create_shader_module(&device, wgpu::ShaderModuleDescriptor {
                label: Some("debug-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../debug.wgsl").into()),
            });



            let uniform = Uniform::new("particle-uniform", &device, 0, wgpu::ShaderStages::VERTEX_FRAGMENT);


            let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle-render-pipeline-layout"),
                bind_group_layouts: &[&uniform.bind_group_layout()],
                push_constant_ranges: &[],
            });


            let targets = [Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })];




            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particle-render-pipeline"),
                layout: Some(&rpl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[ParticleVertex::desc(), ParticleInstance::desc()],
                },


                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &targets,
                }),


                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },


                depth_stencil: None,


                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },


                multiview: None,
                cache: None,
            });


            let debug_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particle-debug-render-pipeline"),
                layout: Some(&rpl),
                vertex: wgpu::VertexState {
                    module: &debug_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[ParticleVertex::desc(), ParticleInstance::desc()],
                },


                fragment: Some(wgpu::FragmentState {
                    module: &debug_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &targets,
                }),


                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },


                depth_stencil: None,


                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },


                multiview: None,
                cache: None,
            });


            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle-shader-quad-vertices"),
                contents: bytemuck::cast_slice(QUAD_VERTICES),
                usage: BufferUsages::VERTEX,
            });


            let instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances-buffer"),
                size: PARTICLE_COUNT * size_of::<ParticleInstance>() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });



            let side_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances-buffer"),
                size: PARTICLE_COUNT * size_of::<ParticleInstance>() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });



            ParticlePipeline {
                render_pipeline: pipeline,
                debug_render_pipeline: debug_pipeline,
                vertices: vertex_buffer,
                side_buffer,
                instances,
                uniform,
            }
        };



        let compute = {
            let shader = create_shader_module(&device, wgpu::ShaderModuleDescriptor {
                label: Some("compute-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../compute.wgsl").into()),
            });



            let uniform = Uniform::new("compute-uniform", &device, 0, ShaderStages::COMPUTE);


            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("compute-bind-group-descriptor")),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None,
                    },
                ],
            });




            let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particles.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particles.side_buffer.as_entire_binding(),
                    }
                ],
            });


            let side_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particles.side_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particles.instances.as_entire_binding(),
                    }
                ],
            });



            let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle-render-pipeline-layout"),
                bind_group_layouts: &[&uniform.bind_group_layout(), &bind_group_layout],
                push_constant_ranges: &[],
            });



            let compute_pipeline = Box::new([
                
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute-pipeline"),
                    layout: Some(&rpl),
                    module: &shader,
                    entry_point: Some("predict_next_position"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }),
                
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute-pipeline"),
                    layout: Some(&rpl),
                    module: &shader,
                    entry_point: Some("calculate_density"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }),
                
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute-pipeline"),
                    layout: Some(&rpl),
                    module: &shader,
                    entry_point: Some("move_particle"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }),
            ]);


            SimulationPipeline {
                compute_pipeline,
                main_bg: main_bind_group,
                swapped_bg: side_bind_group,
                uniform,
            }
        };


        let staging_belt = StagingBelt::new(128 << 20);




        let mut this = Self {
            surface,
            device,
            queue,
            config,
            window,
            framebuffer,
            staging_belt,
            particle_pipeline: particles,
            simulation_pipeline: compute,
            frame_time: 0,

            mouse_pos: Vec2::new(0.0, 0.0),
            mouse_state: 0,
            projection: Mat4::IDENTITY,
        };



        let mut encoder = this.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("command-encoder"),
        });


        let mut view = this.staging_belt.write_buffer(
            &mut encoder,
            &this.particle_pipeline.instances,
            0,
            NonZero::new(PARTICLE_COUNT * size_of::<ParticleInstance>() as u64).unwrap(),
            &this.device
        );

        let particles_per_row = (PARTICLE_COUNT as f32).sqrt();
        let particles_per_column = (PARTICLE_COUNT as f32 - 1.0) / particles_per_row + 1.0;

        let buf = Vec::from_iter((0..PARTICLE_COUNT)
            .map(|i| {
                let x = i as usize % particles_per_row as usize;
                let x = (x as f32  - particles_per_row * 0.5 + 0.5) * PARTICLE_SPACING;
                let y = ((i as f32 / particles_per_row).floor() - particles_per_column * 0.5 + 0.5) * PARTICLE_SPACING;

                ParticleInstance {
                    position: Vec2::new(x, y),
                    predicted_position: Vec2::new(x, y),
                    ..Default::default()
                }

            })
        );


            /*
|_| ParticleInstance {
                position: Vec2::new(
                              (rand::rng().random::<f32>() * 2.0 - 1.0) * SIZE.x * 0.3,
                              (rand::rng().random::<f32>() * 2.0 - 1.0) * SIZE.y * 0.3,
                          ),
                velocity: Vec2::ZERO,
                predicted_position: Vec2::ZERO,
                density: 0.0,
                pad00: 0.0,
            })
        );*/


        view.copy_from_slice(bytemuck::cast_slice(&buf));

        drop(view);

        this.staging_belt.finish();
        this.queue.submit(core::iter::once(encoder.finish()));

        this
    }



    pub fn simulate(&mut self) {
        self.frame_time += 1;

        println!("simulation");
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute-command-encoder"),
        });


        let smoothing_radius = PARTICLE_SPACING*2.0;
        self.simulation_pipeline.uniform.update(&self.queue,

            &SimulationUniform {
                delta: 1.0/60.0,
                gravity: Vec2::new(0.0, 5.0),
                bounds: SIZE * 0.9,

                particle_count: PARTICLE_COUNT as _,
                frame_time: self.frame_time,
                sqr_radius: smoothing_radius*smoothing_radius,
                smoothing_radius: smoothing_radius,
                particle_mass: 1.0,
                pressure_constant: 34.38,
                rest_density: 2.86,
                damping_factor: 0.8,
                viscosity_coefficient: 0.57,
                surface_tension_treshold: 0.1,
                surface_tension_coefficient: 5.0,


                mouse_pos: self.mouse_pos,
                mouse_state: self.mouse_state,

                mouse_force_radius: 5.0,
                mouse_force_power: 1.5,
                pad: 0.0,
            }

        );


        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for pipeline in &self.simulation_pipeline.compute_pipeline {
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &self.simulation_pipeline.uniform.bind_group, &[]);
            cpass.set_bind_group(1, &self.simulation_pipeline.main_bg, &[]);
            cpass.insert_debug_marker("simulation");
            cpass.dispatch_workgroups(PARTICLE_COUNT as _, 1, 1);


            core::mem::swap(
                &mut self.simulation_pipeline.main_bg,
                &mut self.simulation_pipeline.swapped_bg
            );
        }


        drop(cpass);

        self.queue.submit(core::iter::once(encoder.finish()));
    }



    pub fn render(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-command-encoder"),
        });


        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&TextureViewDescriptor::default());


        self.projection = glam::Mat4::orthographic_rh(
            -SIZE.x * 0.5, SIZE.x as f32 * 0.5,
            SIZE.y as f32 * 0.5, -SIZE.y * 0.5,
            -1.0, 0.0);

        fn srgb_channel_to_linear(x: f32) -> f32 {
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        }

        fn conv(x: Vec4) -> Vec4 {
            Vec4::new(
                srgb_channel_to_linear(x.x),
                srgb_channel_to_linear(x.y),
                srgb_channel_to_linear(x.z),
                srgb_channel_to_linear(x.w),
            )
        }

        self.particle_pipeline.uniform.update(&self.queue, &ParticleUniform {
            projection: self.projection,
            scale: PARTICLE_SPACING,
            pad00: Vec3::NAN,
            colour0: conv(Vec4::new(54.0, 112.0, 255.0, 255.0) / Vec4::splat(255.0)),
            colour1: conv(Vec4::new(0.0, 225.0, 163.0, 255.0) / Vec4::splat(255.0)),
            colour2: conv(Vec4::new(205.0, 220.0, 25.0, 255.0) / Vec4::splat(255.0)),
            colour3: conv(Vec4::new(255.0, 55.0, 60.0, 255.0) / Vec4::splat(255.0)),

        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("atlas-render-pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })
            ],

            depth_stencil_attachment: None,

            ..Default::default()
        });


        pass.set_pipeline(&self.particle_pipeline.render_pipeline);
        self.particle_pipeline.uniform.use_uniform(&mut pass);

        pass.set_vertex_buffer(0, self.particle_pipeline.vertices.slice(..));
        pass.set_vertex_buffer(1, self.particle_pipeline.instances.slice(..));

        pass.draw(0..(QUAD_VERTICES.len() as _), 0..(PARTICLE_COUNT as _));


        pass.set_pipeline(&self.particle_pipeline.debug_render_pipeline);
        pass.set_vertex_buffer(0, self.particle_pipeline.vertices.slice(..));
        pass.draw(0..(QUAD_VERTICES.len() as _), 0..1);

        drop(pass);

        self.staging_belt.finish();
        self.queue.submit(core::iter::once(encoder.finish()));

        output.present();


    }



    pub fn window_size(&self) -> Vec2 {
        let (w, h) = (self.config.width, self.config.height);
        Vec2::new(w as _, h as _)
    }


}




pub fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };


    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_view(&wgpu::TextureViewDescriptor::default())
}



#[derive(Clone, Copy, Pod, Zeroable, Debug)]
#[repr(C)]
struct ParticleVertex {
    pos: Vec2,
}


impl ParticleVertex {
    fn desc() -> VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                }
            ],
        }
    }
}

impl ParticleInstance {
    fn desc() -> VertexBufferLayout<'static> {
        const ATTRS : &[VertexAttribute] = &[
            VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(ParticleInstance, position) as _,
                shader_location: 1,
            },
            VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(ParticleInstance, predicted_position) as _,
                shader_location: 2,
            },
            VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(ParticleInstance, velocity) as _,
                shader_location: 3,
            },
            VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: core::mem::offset_of!(ParticleInstance, density) as _,
                shader_location: 4,
            },
            VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: core::mem::offset_of!(ParticleInstance, pad00) as _,
                shader_location: 5,
            },
        ];

        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as _,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}
