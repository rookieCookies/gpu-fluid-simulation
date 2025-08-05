use std::{num::NonZero, sync::mpsc};

use bytemuck::{offset_of, Pod, Zeroable};
use egui_wgpu::{wgpu, ScreenDescriptor};
use glam::{IVec2, IVec3, IVec4, Mat4, Vec2, Vec3, Vec4};
use rand::Rng;
use tracing::span;
use wgpu::{util::{DeviceExt, StagingBelt}, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlasTriangleGeometry, Buffer, BufferDescriptor, BufferUsages, ComputePipeline, PollType, PrimitiveState, PrimitiveTopology, RenderPassDepthStencilAttachment, ShaderStages, TextureUsages, TextureViewDescriptor, VertexAttribute, VertexBufferLayout};
use winit::{dpi::PhysicalSize, window::{self, Window}};

use crate::{buffer::{ResizableBuffer, SSBO}, egui_tools::EguiRenderer, shader::create_shader_module, uniform::Uniform};


const MSAA_SAMPLE_COUNT : u32 = 1;
const PARTICLE_SPACING : f32 = 0.125;
const PARTICLE_COUNT : u32 = 30000;
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
    pub sort_pipeline: SortPipeline,
    pub egui: EguiRenderer,

    pub projection: Mat4,


    pub simulation_uniform: SimulationUniform,
    pub spawn_settings: SpawnSettings,
    pub current_settings: SpawnSettings,
}


#[derive(Clone, Copy, Zeroable, Pod, Default, Debug)]
#[repr(C)]
struct ParticleInstance {
    position: Vec2,
    predicted_position: Vec2,
    velocity: Vec2,
    density: f32,
    grid: u32,
}


#[derive(Clone, Copy, Zeroable, Pod, Debug)]
#[repr(C)]
struct ParticleUniform {
    projection: Mat4,
    pad00: f32,
    scale: f32,
    grid_size: u32,
    grid_w: u32,
    colour0: Vec4,
    colour1: Vec4,
    colour2: Vec4,
    colour3: Vec4,
}


#[derive(Clone, Copy, Zeroable, Pod, Debug)]
#[repr(C)]
#[repr(align(256))]
struct SortUniform {
    group_width: u32,
    group_height: u32,
    step_index: u32,
    num_values: u32,

    pad: [u8; 240],
}


#[derive(Clone, Copy, Zeroable, Pod, Debug)]
#[repr(C)]
struct SpatialLookupCell {
    particle: u32,
    grid: u32,
}



#[derive(Clone, Copy, Zeroable, Pod, Debug)]
#[repr(C)]
pub struct SimulationUniform {
    pub delta: f32,
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

    pub mouse_pos: Vec2,
    pub mouse_state: i32,

    mouse_force_radius: f32,
    mouse_force_power: f32,


    grid_w: u32,
    grid_h: u32,
    pad: f32,
}


#[derive(Clone)]
pub struct SpawnSettings {
    particle_count: u32,
    particle_spacing: f32,
    smoothing_radius: f32,
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
    spatial_lookup: Buffer,
    start_indices: Buffer,


    spatial_lookup_bg: BindGroup,
    spatial_lookup_bgl: BindGroupLayout,
}


pub struct SimulationPipeline {
    compute_pipeline: Box<[ComputePipeline]>,
    uniform: Uniform<SimulationUniform>,
    main_bgl: BindGroupLayout,
    main_bg: BindGroup,


    predict_next_positions: ComputePipeline,
    create_spatial_lookup: ComputePipeline,
    compute_start_indices: ComputePipeline,
    calculate_density: ComputePipeline,
    move_particle: ComputePipeline,

}


pub struct SortPipeline {
    uniform: ResizableBuffer<SortUniform>,
    bgl: BindGroupLayout,
    bg: BindGroup,


    values: SSBO<u32>,
    pipeline: ComputePipeline,
}



impl Renderer {
    pub async fn new(window: Window) -> Self {
        let window = Box::leak(Box::new(window));

        let size = window.inner_size();

        dbg!(wgpu::Instance::enabled_backend_features());

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
                required_features: wgpu::Features::empty(),
                required_limits: {
                    let mut limits = wgpu::Limits::downlevel_defaults();
                    limits.max_buffer_size = adapter.limits().max_buffer_size;
                    limits.max_compute_workgroups_per_dimension = adapter.limits().max_compute_workgroups_per_dimension;
                    dbg!(limits.max_compute_workgroups_per_dimension);
                    limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
                    limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
                    limits
                },
                label: Some("main device"),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
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
            present_mode: wgpu::PresentMode::Fifo,
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
                size: 1,
                usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });


            let spatial_lookup = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("spatial-lookup-buffer"),
                size: 1,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });


            let spatial_lookup_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("spatial-lookup-bgl"),
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
                ]
            });


            let spatial_lookup_bg = device.create_bind_group(&BindGroupDescriptor {
                label: Some("spatial-lookup-bg"),
                layout: &spatial_lookup_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: spatial_lookup.as_entire_binding(),
                    },
                ]
            });


            ParticlePipeline {
                render_pipeline: pipeline,
                debug_render_pipeline: debug_pipeline,
                vertices: vertex_buffer,
                instances,
                uniform,
                start_indices: spatial_lookup.clone(),
                spatial_lookup,
                spatial_lookup_bg,
                spatial_lookup_bgl,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
                        resource: particles.spatial_lookup.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particles.spatial_lookup.as_entire_binding(),
                    },
                ],
            });


            let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle-render-pipeline-layout"),
                bind_group_layouts: &[&uniform.bind_group_layout(), &bind_group_layout],
                push_constant_ranges: &[],
            });



            let create_compute_pipeline = |name: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&rpl),
                    module: &shader,
                    entry_point: Some(name),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };
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
                    entry_point: Some("create_spatial_lookup"),
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
                uniform,
                main_bgl: bind_group_layout,


                predict_next_positions: create_compute_pipeline("predict_next_position"),
                create_spatial_lookup: create_compute_pipeline("create_spatial_lookup"),
                compute_start_indices: create_compute_pipeline("compute_start_indices"),
                calculate_density: create_compute_pipeline("calculate_density"),
                move_particle: create_compute_pipeline("move_particle"),



            }
        };





        let sort = {
            let shader = create_shader_module(&device, wgpu::ShaderModuleDescriptor {
                label: Some("sort-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../sort.wgsl").into()),
            });


            let uniforms = ResizableBuffer::new("sort-uniform", &device, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, 1);
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("sort-uniform-bind-group-layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: true,
                            min_binding_size: Some(NonZero::new(size_of::<SortUniform>() as u64).unwrap())
                        },

                        count: None,
                    }
                ]
            });


            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("sort-uniform-bind-group"),
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniforms.buffer,
                            offset: 0,
                            size: Some(NonZero::new(size_of::<SortUniform>() as u64).unwrap()),
                        }),
                    }
                ]
            });


            let ssbo = SSBO::new("sort-values-buffer", &device, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, ShaderStages::COMPUTE, 1);

            let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sort-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout, &ssbo.layout()],
                push_constant_ranges: &[],
            });


            let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sort-compute-pipeline"),
                layout: Some(&rpl),
                module: &shader,
                entry_point: Some("sort"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });


            SortPipeline {
                uniform: uniforms,
                values: ssbo,
                pipeline: compute_pipeline,
                bgl: bind_group_layout,
                bg: bind_group,
            }
        };


        let staging_belt = StagingBelt::new(1024 * 1024);


        let egui = EguiRenderer::new(
            &device,
            config.format,
            None,
            MSAA_SAMPLE_COUNT,
            window
        );

        let smoothing_radius = PARTICLE_SPACING*2.0;

        let settings = SpawnSettings {
            particle_count: PARTICLE_COUNT,
            particle_spacing: PARTICLE_SPACING,
            smoothing_radius,
        };

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
            sort_pipeline: sort,
            egui,
            projection: Mat4::IDENTITY,

            spawn_settings: settings.clone(),
            current_settings: settings,

            simulation_uniform: SimulationUniform {
                delta: 1.0/120.0,
                gravity: Vec2::new(0.0, 0.0),
                bounds: SIZE * 0.9,

                particle_count: PARTICLE_COUNT as _,
                sqr_radius: smoothing_radius*smoothing_radius,
                smoothing_radius: smoothing_radius,
                particle_mass: 1.0,
                pressure_constant: 60.0,
                rest_density: 0.0,
                damping_factor: 0.8,
                viscosity_coefficient: 0.05,
                surface_tension_treshold: 0.1,
                surface_tension_coefficient: 100.0,
                mouse_force_radius: 5.0,
                mouse_force_power: 1.5,



                // state
                mouse_pos: Vec2::ZERO,
                mouse_state: 0,
                frame_time: 0,
                pad: 0.0,
                grid_w: 0,
                grid_h: 0,
            },
        };


        this.restart_simulation();


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

        this
    }


    pub fn restart_simulation(&mut self) {
        let grid_w = (SIZE.x / self.spawn_settings.smoothing_radius).ceil() as usize + 2;
        let grid_h = (SIZE.y / self.spawn_settings.smoothing_radius).ceil() as usize + 2;



        let instances = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances-buffer"),
            size: self.spawn_settings.particle_count as u64 * size_of::<ParticleInstance>() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let spatial_lookup = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatial-lookup-buffer"),
            size: self.spawn_settings.particle_count as u64 * size_of::<SpatialLookupCell>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });


        let start_indices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("start-indices-buffer"),
            size: (grid_w * grid_h * size_of::<u32>()) as _,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });


    

        let spatial_lookup_bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("spatial-lookup-bg"),
            layout: &self.particle_pipeline.spatial_lookup_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spatial_lookup.as_entire_binding(),
                },
            ]
        });



        self.particle_pipeline.instances.destroy();
        self.particle_pipeline.instances = instances;

        self.particle_pipeline.spatial_lookup.destroy();
        self.particle_pipeline.spatial_lookup = spatial_lookup;
        self.particle_pipeline.spatial_lookup_bg = spatial_lookup_bg;

        self.particle_pipeline.start_indices.destroy();
        self.particle_pipeline.start_indices= start_indices;

        let main_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.simulation_pipeline.main_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.particle_pipeline.instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.particle_pipeline.spatial_lookup.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.particle_pipeline.start_indices.as_entire_binding(),
                },
            ],
        });

        self.simulation_pipeline.main_bg = main_bind_group;
        self.simulation_uniform.smoothing_radius = self.spawn_settings.smoothing_radius;
        self.simulation_uniform.grid_w = grid_w as _;
        self.simulation_uniform.grid_h = grid_h as _;
        self.current_settings = self.spawn_settings.clone();


        let particle_count = self.spawn_settings.particle_count;
        let particle_spacing = self.spawn_settings.particle_spacing;
        
        let particles_per_row = (particle_count as f32).sqrt();
        let particles_per_column = (particle_count as f32 - 1.0) / particles_per_row + 1.0;

        let buf = Vec::from_iter((0..particle_count)
            .map(|i| {
                let x = i as usize % particles_per_row as usize;
                let x = (x as f32  - particles_per_row * 0.5 + 0.5) * particle_spacing;
                let y = ((i as f32 / particles_per_row).floor() - particles_per_column * 0.5 + 0.5) * particle_spacing;

                ParticleInstance {
                    position: Vec2::new(x, y),
                    predicted_position: Vec2::new(x, y),
                    ..Default::default()
                }

            })
        );

        self.queue.write_buffer(&self.particle_pipeline.instances, 0, bytemuck::cast_slice(&buf));




        // prepare sort buffers
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prepare-sort-buffers") });

        let num_pairs = self.current_settings.particle_count.next_power_of_two() / 2;
        let num_stages = (num_pairs * 2).ilog2();

        let mut count = 0;
        for stage_index in 0..num_stages {
            for _ in 0..=stage_index {
                count += 1;
            }
        }


        let resized = self.sort_pipeline.uniform.resize(&self.device, &mut encoder, count);
        if resized {
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("sort-uniform-bind-group"),
                layout: &self.sort_pipeline.bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.sort_pipeline.uniform.buffer,
                            offset: 0,
                            size: Some(NonZero::new(size_of::<SortUniform>() as u64).unwrap()),
                        }),
                    }
                ]
            });




            self.sort_pipeline.bg = bind_group;
        }





        self.queue.submit(core::iter::once(encoder.finish()));

    }



    pub fn simulate(&mut self, encoder: &mut wgpu::CommandEncoder) {
        println!("simulataaaee");
        self.simulation_uniform.frame_time += 1;

        const WORKGROUP_SIZE : u32 = 256;
        let num_workgroups = (self.current_settings.particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        dbg!(num_workgroups);

        self.simulation_uniform.sqr_radius = self.simulation_uniform.smoothing_radius;
        self.simulation_uniform.particle_count = self.current_settings.particle_count;
        println!("update uniform");
        self.simulation_pipeline.uniform.update(
            &self.queue,
            &self.simulation_uniform
        );

        println!("updated uniform");

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });

        cpass.set_bind_group(0, &self.simulation_pipeline.uniform.bind_group, &[]);
        cpass.set_bind_group(1, &self.simulation_pipeline.main_bg, &[]);


        cpass.set_pipeline(&self.simulation_pipeline.predict_next_positions);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);

        cpass.set_pipeline(&self.simulation_pipeline.create_spatial_lookup);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);

        self.sort_spatial_lookup(&mut cpass);

        cpass.set_bind_group(0, &self.simulation_pipeline.uniform.bind_group, &[]);
        cpass.set_bind_group(1, &self.simulation_pipeline.main_bg, &[]);

        cpass.set_pipeline(&self.simulation_pipeline.compute_start_indices);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);


        cpass.set_pipeline(&self.simulation_pipeline.calculate_density);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);


        cpass.set_pipeline(&self.simulation_pipeline.move_particle);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);


        drop(cpass);
    }



    pub fn sort_spatial_lookup(&mut self, cpass: &mut wgpu::ComputePass) {
        println!("start sorting");
        let num_pairs = self.current_settings.particle_count.next_power_of_two() / 2;
        let num_stages = (num_pairs * 2).ilog2();
        let dispatch = num_pairs as u32 / 128;


        let mut buf = vec![];
        for stage_index in 0..num_stages {
            for step_index in 0..=stage_index {
                let group_width = 1 << (stage_index - step_index);
                let group_height = 2 * group_width - 1;

                if dispatch == 0 {
                    println!("skipping because dispatch == 0");
                }

                let settings = SortUniform {
                    group_width,
                    group_height,
                    step_index,
                    num_values: self.current_settings.particle_count,
                    pad: [0; _],
                };

                buf.push(settings);

            }
        }


        self.queue.write_buffer(&self.sort_pipeline.uniform.buffer, 0, bytemuck::cast_slice(&buf));


        cpass.set_pipeline(&self.sort_pipeline.pipeline);
        cpass.set_bind_group(1, &self.particle_pipeline.spatial_lookup_bg, &[]);

        for i in 0..buf.len() as u32 {
            cpass.set_bind_group(0, &self.sort_pipeline.bg, &[i * size_of::<SortUniform>() as u32]);
            cpass.dispatch_workgroups(dispatch, 1, 1);
        }
    }



    pub fn render(&mut self, mut encoder: wgpu::CommandEncoder) {
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


        let grid_w = (SIZE.x / self.current_settings.smoothing_radius).ceil() as usize + 2;
        let grid_h = (SIZE.y / self.current_settings.smoothing_radius).ceil() as usize + 2;


        self.particle_pipeline.uniform.update(&self.queue, &ParticleUniform {
            projection: self.projection,
            scale: self.current_settings.particle_spacing,
            pad00: 0.0,
            colour0: conv(Vec4::new(54.0, 112.0, 255.0, 255.0) / Vec4::splat(255.0)),
            colour1: conv(Vec4::new(0.0, 225.0, 163.0, 255.0) / Vec4::splat(255.0)),
            colour2: conv(Vec4::new(205.0, 220.0, 25.0, 255.0) / Vec4::splat(255.0)),
            colour3: conv(Vec4::new(255.0, 55.0, 60.0, 255.0) / Vec4::splat(255.0)),
            grid_size: (grid_w * grid_h) as _,
            grid_w: grid_w as _,

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

        pass.draw(0..(QUAD_VERTICES.len() as _), 0..(self.current_settings.particle_count as _));


        pass.set_pipeline(&self.particle_pipeline.debug_render_pipeline);
        pass.set_vertex_buffer(0, self.particle_pipeline.vertices.slice(..));
        pass.draw(0..(QUAD_VERTICES.len() as _), 0..1);

        drop(pass);


        let mut restart_sim = false;
        {

            let screen_descriptor = ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            };

            self.egui.begin_frame(self.window);



            egui::Window::new("spawn settings")
                .resizable(true)
                .vscroll(true)
                .default_open(true)
                .show(self.egui.context(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("particle count");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.spawn_settings.particle_count)
                                .range(0..=u32::MAX)
                                .speed(10),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("particle spacing");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.spawn_settings.particle_spacing)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });


                    ui.horizontal(|ui| {
                        ui.label("smoothing radius");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.spawn_settings.smoothing_radius)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });


                    if ui.button("restart simulation").clicked() {
                        restart_sim = true;
                    }
                });

            egui::Window::new("simulation settings")
                .resizable(true)
                .vscroll(true)
                .default_open(true)
                .show(self.egui.context(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("delta");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.delta)
                                .range(0.0..=1.0)
                                .speed(0.001),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("gravity");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.gravity.x)
                                .range(0.0..=f32::MAX)
                                .speed(0.1),
                        );
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.gravity.y)
                                .range(0.0..=f32::MAX)
                                .speed(0.1),
                        );
                    });



                    ui.horizontal(|ui| {
                        ui.label("particle mass");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.particle_mass)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("pressure constant");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.pressure_constant)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("rest density");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.rest_density)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("damping factor");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.damping_factor)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("viscosity coefficient");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.viscosity_coefficient)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("surface tension treshold");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.surface_tension_treshold)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("surface tension coefficient");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.surface_tension_coefficient)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("mouse force radius");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.mouse_force_radius)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("mouse force power");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.simulation_uniform.mouse_force_power)
                                .range(0.0..=f32::MAX)
                                .speed(0.025),
                        );
                    });
                });

            
            self.egui.end_frame_and_draw(
                &self.device,
                &self.queue,
                &mut encoder,
                self.window,
                &view,
                screen_descriptor,
            );
        }



        self.staging_belt.finish();
        self.queue.submit(core::iter::once(encoder.finish()));

        output.present();

        if restart_sim {
            self.restart_simulation();
        }


    }



    pub fn window_size(&self) -> Vec2 {
        let (w, h) = (self.config.width, self.config.height);
        Vec2::new(w as _, h as _)
    }




    pub fn resize_surface(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
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
                format: wgpu::VertexFormat::Uint32,
                offset: core::mem::offset_of!(ParticleInstance, grid) as _,
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
