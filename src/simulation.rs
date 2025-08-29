use std::{f32::consts::PI, num::NonZero};

use bytemuck::{Pod, Zeroable};
use glam::{UVec2, Vec2};
use egui_wgpu::wgpu as wgpu;
use wgpu::{util::{BufferInitDescriptor, DeviceExt}, BufferUsages, ComputePipeline, ShaderStages};

use crate::{buffer::{ResizableBuffer, SSBO}, shader::create_shader_module, uniform::Uniform};

pub struct FluidSimulation {
    settings: SimulationSettings,
    pub tick: u32,

    particles_buffer: wgpu::Buffer,
    start_indices: wgpu::Buffer,
    sort_uniforms: wgpu::Buffer,
    texture: wgpu::Buffer,

    spatial_lookup: wgpu::BindGroup,
    simulation_bg: wgpu::BindGroup,
    sort_bg: wgpu::BindGroup,

    simulation_bgl: wgpu::BindGroupLayout,

    sort_buf_len: u32,
    sort_dispatch: u32,

    simulation_uniform: Uniform<SimulationUniform>,

    predict_next_positions: ComputePipeline,
    create_spatial_lookup: ComputePipeline,
    compute_start_indices: ComputePipeline,
    calculate_density: ComputePipeline,
    move_particle: ComputePipeline,
    sort_pipeline: ComputePipeline,
    
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
pub struct SimulationUniform {
    delta: f32,
    particle_count: u32,
    sqr_radius: f32,
    frame_time: u32,

    gravity: Vec2,
    bounds: Vec2,
    mouse_pos: Vec2,

    smoothing_radius: f32,
    particle_mass: f32,
    pressure_constant: f32,
    rest_density: f32,

    damping_factor: f32,
    viscosity_coefficient: f32,
    surface_tension_treshold: f32,
    surface_tension_coefficient: f32,
    poly6_kernel_volume: f32,
    poly6_kernel_derivative: f32,
    poly6_kernel_laplacian: f32,
    spiky_kernel_derivative: f32,
    viscosity_kernel: f32,

    mouse_state: i32,

    mouse_force_radius: f32,
    mouse_force_power: f32,

    grid_w: u32,
    grid_h: u32,

    texture_size: Vec2,

}




#[derive(Clone, Copy)]
pub struct SimulationSettings {
    pub particle_count: u32,
    pub particle_spacing: f32,
    pub smoothing_radius: f32,

    pub size: Vec2,

    pub texture_size: UVec2,
}


#[derive(Clone, Copy)]
pub struct TickSettings {
    pub delta: f32,
    pub gravity: Vec2,
    pub mass: f32,
    pub pressure_constant: f32,
    pub rest_density: f32,
    pub damping_factor: f32,
    pub viscosity_coefficient: f32,
    pub surface_tension_treshold: f32,
    pub surface_tension_coefficient: f32,
    pub mouse_force_radius: f32,
    pub mouse_force_power: f32,
    pub mouse_pos: Vec2,
    pub mouse_state: i32,
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


impl FluidSimulation {
    pub fn new(device: &wgpu::Device, settings: SimulationSettings) -> Self {
        let grid_w = (settings.size.x / settings.smoothing_radius).ceil() as usize + 2;
        let grid_h = (settings.size.y / settings.smoothing_radius).ceil() as usize + 2;


        let particle_count = settings.particle_count;
        let particle_spacing = settings.particle_spacing;
        
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




        let instances = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("particle-instances"),
            contents: bytemuck::cast_slice(&buf),
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE,
        });


        let spatial_lookup_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial-lookup-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                }
            ],
        });


        let spatial_lookup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort-uniform-bind-group"),
            layout: &spatial_lookup_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances.as_entire_binding(),
                }
            ],
        });


        let start_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("start-indices-buffer"),
            size: (grid_w * grid_h * size_of::<u32>()) as _,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });



        let force_field_texture = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("force-field-texture"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            size: (settings.texture_size.x * settings.texture_size.y * size_of::<Vec2>() as u32) as _,
            mapped_at_creation: false,
        });





        
        let compute_shader = create_shader_module(&device, wgpu::ShaderModuleDescriptor {
            label: Some("compute-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../compute.wgsl").into()),
        });



        let simulation_settings_uniform = Uniform::new(
            "simulation-settings-uniform",
            &device,
            0,
            wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT
        );


        let simulation_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("compute-bind-group-descriptor")),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
            ],
        });


        let simulation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &simulation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: start_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: force_field_texture.as_entire_binding(),
                },
            ],
        });


        let compute_rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute-render-pipeline-layout"),
            bind_group_layouts: &[&simulation_settings_uniform.bind_group_layout(), &simulation_bind_group_layout],
            push_constant_ranges: &[],
        });


        let create_compute_pipeline = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&compute_rpl),
                module: &compute_shader,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let predict_next_positions = create_compute_pipeline("predict_next_position");
        let create_spatial_lookup = create_compute_pipeline("create_spatial_lookup");
        let compute_start_indices = create_compute_pipeline("compute_start_indices");
        let calculate_density = create_compute_pipeline("calculate_density");
        let move_particle = create_compute_pipeline("move_particle");



        let num_pairs = particle_count.next_power_of_two() / 2;
        let num_stages = (num_pairs * 2).ilog2();
        let sort_dispatch_count = (num_pairs as u32).div_ceil(128);

        let mut buf = vec![];
        for stage_index in 0..num_stages {
            for step_index in 0..=stage_index {
                let group_width = 1 << (stage_index - step_index);
                let group_height = 2 * group_width - 1;

                if sort_dispatch_count == 0 {
                    continue;
                }

                let settings = SortUniform {
                    group_width,
                    group_height,
                    step_index,
                    num_values: particle_count,
                    pad: [0; _],
                };

                buf.push(settings);
            }
        }


        let sort_uniforms = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("sort-uniform-buffer"),
            contents: bytemuck::cast_slice(&buf),
            usage: wgpu::BufferUsages::STORAGE,
        });


        let sort_buf_len = buf.len() as u32;


        let sort_shader = create_shader_module(&device, wgpu::ShaderModuleDescriptor {
            label: Some("sort-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../sort.wgsl").into()),
        });


        let ssbo : SSBO<ParticleInstance> = SSBO::new("sort-values-buffer", &device, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, ShaderStages::COMPUTE, 1);


        let sort_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort-uniform-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: Some(NonZero::new(size_of::<SortUniform>() as u64).unwrap())
                    },

                    count: None,
                }
            ]
        });


        let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort-uniform-bind-group"),
            layout: &sort_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sort_uniforms,
                        offset: 0,
                        size: Some(NonZero::new(size_of::<SortUniform>() as u64).unwrap()),
                    }),
                }
            ]
        });



        let sort_rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sort-pipeline-layout"),
            bind_group_layouts: &[&sort_bind_group_layout, &ssbo.layout()],
            push_constant_ranges: &[],
        });


        let sort_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sort-compute-pipeline"),
            layout: Some(&sort_rpl),
            module: &sort_shader,
            entry_point: Some("sort"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });



        
        let jf_shader = create_shader_module(&device, wgpu::ShaderModuleDescriptor {
            label: Some("jump-flooding-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/jump_flood.wgsl").into()),
        });





        Self {
            settings,
            tick: 0,
            particles_buffer: instances,
            start_indices,
            texture: force_field_texture,
            spatial_lookup,
            simulation_uniform: simulation_settings_uniform,
            sort_uniforms,
            predict_next_positions,
            create_spatial_lookup,
            compute_start_indices,
            calculate_density,
            move_particle,
            sort_pipeline,
            simulation_bg: simulation_bind_group,
            sort_bg: sort_bind_group,

            simulation_bgl: simulation_bind_group_layout,

            sort_buf_len,
            sort_dispatch: sort_dispatch_count,
        }
    }



    pub fn tick(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, settings: TickSettings) {
        self.tick += 1;


        const WORKGROUP_SIZE : u32 = 256;
        let num_workgroups = (self.settings.particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let grid_w = (self.settings.size.x / self.settings.smoothing_radius).ceil() as usize + 2;
        let grid_h = (self.settings.size.y / self.settings.smoothing_radius).ceil() as usize + 2;


        let uniform = SimulationUniform {
            delta: settings.delta,
            particle_count: self.settings.particle_count,
            sqr_radius: self.settings.smoothing_radius*self.settings.smoothing_radius,
            frame_time: self.tick,
            gravity: settings.gravity,
            bounds: self.settings.size,
            mouse_pos: settings.mouse_pos,
            smoothing_radius: self.settings.smoothing_radius,
            particle_mass: settings.mass,
            pressure_constant: settings.pressure_constant,
            rest_density: settings.rest_density,
            damping_factor: settings.damping_factor,
            viscosity_coefficient: settings.viscosity_coefficient,
            surface_tension_treshold: settings.surface_tension_treshold,
            surface_tension_coefficient: settings.surface_tension_coefficient,
            poly6_kernel_volume: 4.0 / (PI * self.settings.smoothing_radius.powi(8)),
            poly6_kernel_derivative: 24.0 / (PI * self.settings.smoothing_radius.powi(8)),   // used in gradient
            poly6_kernel_laplacian: 8.0 / (PI * self.settings.smoothing_radius.powi(8)),
            spiky_kernel_derivative: 12.0 / (self.settings.smoothing_radius.powi(4) * PI),
            viscosity_kernel: 15.0 / (2.0 * PI * self.settings.smoothing_radius.powi(3)),
            mouse_state: settings.mouse_state,
            mouse_force_radius: settings.mouse_force_radius,
            mouse_force_power: settings.mouse_force_power,
            grid_w: grid_w as u32,
            grid_h: grid_h as u32,
            texture_size: self.settings.texture_size.as_vec2(),
        };

        self.simulation_uniform.update(queue, &uniform);


        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("simulation-pass-tick{}", self.tick)),
            timestamp_writes: None,
        });


        pass.set_bind_group(0, &self.simulation_uniform.bind_group, &[]);
        pass.set_bind_group(1, &self.simulation_bg, &[]);


        pass.set_pipeline(&self.predict_next_positions);
        pass.dispatch_workgroups(num_workgroups, 1, 1);

        pass.set_pipeline(&self.create_spatial_lookup);
        pass.dispatch_workgroups(num_workgroups, 1, 1);

        pass.set_pipeline(&self.sort_pipeline);
        pass.set_bind_group(1, &self.spatial_lookup, &[]);

        for i in 0..self.sort_buf_len {
            pass.set_bind_group(0, &self.sort_bg, &[i * size_of::<SortUniform>() as u32]);
            pass.dispatch_workgroups(self.sort_dispatch, 1, 1);
        }


        pass.set_bind_group(0, &self.simulation_uniform.bind_group, &[]);
        pass.set_bind_group(1, &self.simulation_bg, &[]);

        pass.set_pipeline(&self.compute_start_indices);
        pass.dispatch_workgroups(num_workgroups, 1, 1);

        pass.set_pipeline(&self.calculate_density);
        pass.dispatch_workgroups(num_workgroups, 1, 1);

        pass.set_pipeline(&self.move_particle);
        pass.dispatch_workgroups(num_workgroups, 1, 1);

    }


    pub fn simulation_settings_bgl(&self) -> &wgpu::BindGroupLayout {
        self.simulation_uniform.bind_group_layout()
    }


    pub fn simulation_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.simulation_bgl
    }


    pub fn simulation_settings_bg(&self) -> &wgpu::BindGroup {
        &self.simulation_uniform.bind_group
    }


    pub fn simulation_bg(&self) -> &wgpu::BindGroup {
        &self.simulation_bg
    }


    pub fn force_field_texture(&self) -> &wgpu::Buffer {
        &self.texture
    }
}



impl ParticleInstance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS : &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(ParticleInstance, position) as _,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(ParticleInstance, predicted_position) as _,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(ParticleInstance, velocity) as _,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: core::mem::offset_of!(ParticleInstance, density) as _,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
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
