struct ParticleInstance {
    position: vec2<f32>,
    predicted_position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    grid: u32,
}


struct SpatialLookupCell {
    particle: u32,
    grid: u32,
}


struct Uniforms {
    delta: f32,
    particle_count: u32,
    sqr_radius: f32,
    frame_time: u32,

    gravity: vec2<f32>,
    bounds: vec2<f32>,
    mouse_pos: vec2<f32>,

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
}


const PI: f32 = 3.14159265359;
const EPSILON: f32 = 1.19209290e-07;


@group(0) @binding(0)
var<uniform> u: Uniforms;

@group(1) @binding(0)
var<storage, read_write> in_particles : array<ParticleInstance>;


@group(1) @binding(2)
var<storage, read_write> start_indices: array<u32>;


fn poly6_kernel_gradient(h: f32, r: vec2<f32>) -> vec2<f32> {
    return poly6_kernel_derivative(h, length(r)) * normalize(r);
}


fn poly6_kernel_derivative(h: f32, dst: f32) -> f32 {
    if dst >= h { return 0.0; }

    let a = 24.0*(2.0*h-2.0*dst);
    let b = u.poly6_kernel_derivative;
    return a / b;
}


fn poly6_kernel_laplacian(h: f32, r: f32) -> f32 {
    if r <= h {
        let constant = u.poly6_kernel_laplacian;
        return constant * (h*h-r*r)*(3.0*h*h-7.0*r*r);
    } else {
        return 0.0;
    }
}


fn poly6_kernel(h: f32, r2: f32) -> f32 {
    let h2 = u.smoothing_radius * u.smoothing_radius;
    if r2 > h2 {
        return 0.0;
    }

    let volume = u.poly6_kernel_volume;
    let diff = h2 - r2;
    return diff*diff*diff / volume;
}



fn spiky_kernel_derivative(h: f32, r: f32) -> f32 {
    if r <= h {
        let v = h - r;
        let constant = u.spiky_kernel_derivative;
        return -v * constant;
    } else {
        return 0.0;
    }
}


fn viscosity_kernel(h: f32, r: f32) -> f32 {
    if r <= h {
        let constant = u.viscosity_kernel;
        if r == 0.0 {
            return constant;
        }

        return constant * ((-(r*r*r)/(2.0*h*h*h))+((r*r)/(h*h))+(h/(2.0*r))-1.0);
    } else {
        return 0.0;
    }
}




// Xorshift32 PRNG: mutates state and returns new u32 random number
fn xorshift32(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

// Convert u32 random number to float in [0,1)
fn u32_to_uniform01(x: u32) -> f32 {
    // Divide by 2^32 as float (approximate)
    return f32(x) / 4294967296.0;
}


fn rand_f32(seed: ptr<function, u32>) -> f32 {
    let rand_u32 = xorshift32(seed);
    let rand_float = u32_to_uniform01(rand_u32);
    return rand_float;
}


fn calculate_pressure(density: f32) -> f32 {
    return u.pressure_constant * (density - u.rest_density);
}


fn calculate_density_at_point(point: vec2<f32>) -> f32 {
    var density = 0.0;

    for (var i = 0u; i < u.particle_count; i = i + 1u) {

        let neighbour = in_particles[i];
        let neighbour_pos = neighbour.predicted_position;

        let offset_to_neighbour = neighbour_pos - point;
        let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

        let dst = sqr_dst_to_neighbour;

        let kernel = poly6_kernel(u.smoothing_radius, dst);
        density += u.particle_mass * kernel;
    }

    return max(density, EPSILON);
}


fn cell_of_point(point: vec2<f32>) -> u32 {
    let grid = xy_of_point(point);
    return grid_pos_to_id(vec2<u32>(grid));
}


fn xy_of_point(point: vec2<f32>) -> vec2<u32> {
    return vec2<u32>(floor((point + u.bounds*0.5) / u.smoothing_radius)) + vec2<u32>(1);
}

fn grid_pos_to_id(point: vec2<u32>) -> u32 {
    return point.y * u.grid_w + point.x;
}
