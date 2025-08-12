#include funcs.wgsl


@group(1) @binding(2)
var<storage> texture : array<vec2<f32>>;


@compute @workgroup_size(256)
fn predict_next_position(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if id.x >= u.particle_count { return; }

    var particle = in_particles[id.x];

    particle.predicted_position = particle.position + particle.velocity * u.delta;

    let bounds_size = u.bounds * 0.5;
    if abs(particle.predicted_position.x) > bounds_size.x {
        particle.predicted_position.x = bounds_size.x * sign(particle.predicted_position.x);
    }


    if abs(particle.predicted_position.y) > bounds_size.y {
        particle.predicted_position.y = bounds_size.y * sign(particle.predicted_position.y);
    }


    in_particles[id.x] = particle;
}


@compute @workgroup_size(256)
fn create_spatial_lookup(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if id.x >= u.particle_count { return; }

    let predicted_position = in_particles[id.x].predicted_position;
    let grid = cell_of_point(predicted_position);
    in_particles[id.x].grid = grid;
}


@compute @workgroup_size(256)
fn compute_start_indices(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if id.x >= u.particle_count { return; }
    if id.x == 0 { return; };

    let particle = in_particles[id.x];
    if particle.grid != in_particles[id.x-1].grid {
        start_indices[particle.grid] = id.x;
    }
}


@compute @workgroup_size(256)
fn calculate_density(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {

    if gid.x >= u.particle_count { return; }

    let id = gid.x;
    var particle = in_particles[id];

    let point = particle.predicted_position;
    let density = max(calculate_density_at_point(point), 0.1);
    particle.density = density;

    in_particles[id] = particle;
}




@compute @workgroup_size(256)
fn move_particle(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if gid.x >= u.particle_count { return; }

    let id = gid.x;
    var particle = in_particles[id];

    

    let pressure = calculate_pressure_force(id);
    let viscosity = calculate_viscosity_force(id);
    let surface_tension = calculate_surface_tension(particle.predicted_position);
    let acceleration = pressure + viscosity + surface_tension;
   
    particle.velocity += (acceleration / particle.density) * u.delta;
    particle.velocity += u.gravity * u.delta;


    if u.mouse_state != 0 {
        let diff = u.mouse_pos - particle.predicted_position;
        let dist = length(diff);

        if dist <= u.mouse_force_radius {
            let dir = diff / dist / dist;
            let ratio = dist / u.mouse_force_radius;
            particle.velocity += dir * u.mouse_force_power * f32(u.mouse_state) * ratio;
        }
    }


    // apply the texture gradient
    let uv = (particle.predicted_position / u.bounds*1.0) + 0.5;

    let pos = vec2<u32>(uv * u.texture_size);
    if distance(uv, vec2<f32>(0)) > 0.3 {
        let force = texture[u32(pos.y * u32(u.texture_size.x) + pos.x)];
        particle.velocity += force * u.mouse_force_power * u.delta;
    }


    if (!all(particle.velocity == particle.velocity)) {
        // reset to something sane so it can recover
        particle.velocity = vec2(0.0, 0.0);
    }

    let max_speed = 500.0;
    let speed = length(particle.velocity);
    if (speed > max_speed) {
        particle.velocity = (particle.velocity / speed) * max_speed;
    }


    particle.position += particle.velocity * u.delta;


    let bounds_size = u.bounds * 0.5;
    if abs(particle.position.x) > bounds_size.x {
        particle.position.x = bounds_size.x * sign(particle.position.x);
        particle.velocity.x *= -1.0 * u.damping_factor;
    }


    if abs(particle.position.y) > bounds_size.y {
        particle.position.y = bounds_size.y * sign(particle.position.y);
        particle.velocity.y *= -1.0 * u.damping_factor;
    }

    in_particles[id] = particle;

}


fn calculate_pressure_force(particle_id: u32) -> vec2<f32> {
    var rand_seed = particle_id * 12 + u.frame_time * 69;

    let particle = in_particles[particle_id];

    let pressure = calculate_pressure(particle.density);
    let position = particle.predicted_position;

    var pressure_force = vec2<f32>(0.0);


    // loop neighbours
    let cell = vec2<i32>(xy_of_point(position));
    for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
        for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
            let x = u32(cell.x + offset_x);
            let y = u32(cell.y + offset_y);

            let id = grid_pos_to_id(vec2<u32>(x, y));
            var start_index = start_indices[id];


            while true {
                if start_index >= u.particle_count { break; }

                let neighbour = in_particles[start_index];

                if neighbour.grid != id { break; }

                let i = start_index;
                start_index += 1;

                // func start
                

                if i == particle_id { continue; }

                let neighbour_pos = neighbour.predicted_position;

                let offset_to_neighbour = neighbour_pos - position;
                let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

                if sqr_dst_to_neighbour > u.sqr_radius {
                    continue;
                }


                let dst = sqrt(sqr_dst_to_neighbour);

                var dir_to_neighbour = vec2<f32>(0.0);

                if dst == 0.0 {
                    dir_to_neighbour = normalize(vec2<f32>(rand_f32(&rand_seed), rand_f32(&rand_seed)));
                } else {
                    dir_to_neighbour = offset_to_neighbour / dst;
                }

                let neighbour_density = neighbour.density;
                let neighbour_pressure = calculate_pressure(neighbour.density);

                let kernel = spiky_kernel_derivative(u.smoothing_radius, dst);
                let shared_pressure = (pressure + neighbour_pressure) * 0.5;
                
                pressure_force += dir_to_neighbour * kernel * shared_pressure / neighbour_density;


                // func end


            }

        }
    }

/*
    for (var i = 0u; i < u.particle_count; i = i + 1u) {
        if i == particle_id { continue; }

        let neighbour = in_particles[i];
        let neighbour_pos = neighbour.predicted_position;

        let offset_to_neighbour = neighbour_pos - position;
        let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

        if sqr_dst_to_neighbour > u.sqr_radius {
            continue;
        }


        let dst = sqrt(sqr_dst_to_neighbour);

        var dir_to_neighbour = vec2<f32>(0.0);

        if dst == 0.0 {
            dir_to_neighbour = normalize(vec2<f32>(rand_f32(&rand_seed), rand_f32(&rand_seed)));
        } else {
            dir_to_neighbour = offset_to_neighbour / dst;
        }

        let neighbour_density = neighbour.density;
        let neighbour_pressure = calculate_pressure(neighbour.density);

        let kernel = spiky_kernel_derivative(u.smoothing_radius, dst);
        let shared_pressure = (pressure + neighbour_pressure) * 0.5;
        
        pressure_force += dir_to_neighbour * kernel * shared_pressure / neighbour_density;
    }*/

    return pressure_force;


}


fn calculate_viscosity_force(particle_id: u32) -> vec2<f32> {
    let particle = in_particles[particle_id];

    let position = particle.predicted_position;

    var pressure_force = vec2<f32>(0.0);



    // loop neighbours
    let cell = vec2<i32>(xy_of_point(position));
    for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
        for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
            let x = u32(cell.x + offset_x);
            let y = u32(cell.y + offset_y);

            let id = grid_pos_to_id(vec2<u32>(x, y));
            var start_index = start_indices[id];


            while true {
                if start_index >= u.particle_count { break; }


                let neighbour = in_particles[start_index];

                if neighbour.grid != id { break; }

                let i = start_index;
                start_index += 1;

                // func start

                if i == particle_id { continue; }

                let neighbour_pos = neighbour.predicted_position;

                let offset_to_neighbour = neighbour_pos - position;
                let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

                if sqr_dst_to_neighbour > u.sqr_radius {
                    continue;
                }


                let dst = sqrt(sqr_dst_to_neighbour);
                let neighbour_density = neighbour.density;

                let kernel = viscosity_kernel(u.smoothing_radius, dst);
                
                pressure_force += (neighbour.velocity - particle.velocity) / neighbour_density * kernel;
                       
                // func end


            }

        }
    }


/*

    for (var i = 0u; i < u.particle_count; i = i + 1u) {
        if i == particle_id { continue; }

        let neighbour = in_particles[i];
        let neighbour_pos = neighbour.predicted_position;

        let offset_to_neighbour = neighbour_pos - position;
        let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

        if sqr_dst_to_neighbour > u.sqr_radius {
            continue;
        }


        let dst = sqrt(sqr_dst_to_neighbour);
        let neighbour_density = neighbour.density;

        let kernel = viscosity_kernel(u.smoothing_radius, dst);
        
        pressure_force += (neighbour.velocity - particle.velocity) / neighbour_density * kernel;
    }
*/

    return pressure_force * u.viscosity_coefficient;


}



fn calculate_surface_tension(point: vec2<f32>) -> vec2<f32> {
    let n = calculate_colour_field_gradient(point);
    let n_len = length(n);

    if n_len > u.surface_tension_treshold {
        let k = (-calculate_colour_field_laplacian(point)) / (n_len + 1e-6);
        let f = -u.surface_tension_coefficient * k * (n / n_len);

        return f;
    } else {
        return vec2<f32>(0.0);
    }
}



fn calculate_colour_field_laplacian(point: vec2<f32>) -> f32 {
    var sum = 0.0;


    // loop neighbours
    let cell = vec2<i32>(xy_of_point(point));
    for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
        for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
            let x = u32(cell.x + offset_x);
            let y = u32(cell.y + offset_y);

            let id = grid_pos_to_id(vec2<u32>(x, y));
            var start_index = start_indices[id];


            while true {
                if start_index >= u.particle_count { break; }

                let neighbour = in_particles[start_index];

                if neighbour.grid != id { break; }

                let i = start_index;
                start_index += 1;

                // func start
                let neighbour_pos = neighbour.predicted_position;

                let offset_to_neighbour = neighbour_pos - point;
                let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

                if sqr_dst_to_neighbour > u.sqr_radius {
                    continue;
                }


                let dst = sqrt(sqr_dst_to_neighbour);

                var dir_to_neighbour = vec2<f32>(0.0);
                let neighbour_density = neighbour.density;

                let kernel = poly6_kernel_laplacian(u.smoothing_radius, dst);

                sum += (u.particle_mass / neighbour_density) * kernel;

                // func end


            }

        }
    }



/*
    for (var i = 0u; i < u.particle_count; i = i + 1u) {
        let neighbour = in_particles[i];
        let neighbour_pos = neighbour.predicted_position;

        let offset_to_neighbour = neighbour_pos - point;
        let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

        if sqr_dst_to_neighbour > u.sqr_radius {
            continue;
        }


        let dst = sqrt(sqr_dst_to_neighbour);

        var dir_to_neighbour = vec2<f32>(0.0);
        let neighbour_density = neighbour.density;

        let kernel = poly6_kernel_laplacian(u.smoothing_radius, dst);

        sum += (u.particle_mass / neighbour_density) * kernel;


    }
*/

    return sum;
}



fn calculate_colour_field_gradient(point: vec2<f32>) -> vec2<f32> {
    var rand_seed = u32(point.x) * 324 + u.frame_time * 5632;
    var sum = vec2<f32>(0.0);


    // loop neighbours
    let cell = vec2<i32>(xy_of_point(point));
    for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
        for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
            let x = u32(cell.x + offset_x);
            let y = u32(cell.y + offset_y);

            let id = grid_pos_to_id(vec2<u32>(x, y));
            var start_index = start_indices[id];


            while true {
                if start_index >= u.particle_count { break; }

                let neighbour = in_particles[start_index];

                if neighbour.grid != id { break; }

                let i = start_index;
                start_index += 1;

                // func start

                let neighbour_pos = neighbour.predicted_position;

                let offset_to_neighbour = neighbour_pos - point;
                let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

                if sqr_dst_to_neighbour > u.sqr_radius {
                    continue;
                }


                let dst = sqrt(sqr_dst_to_neighbour);

                var dir_to_neighbour = vec2<f32>(0.0);
                if dst == 0.0 {
                    dir_to_neighbour = normalize(vec2<f32>(rand_f32(&rand_seed), rand_f32(&rand_seed)));
                } else {
                    dir_to_neighbour = offset_to_neighbour / dst;
                }

                let neighbour_density = neighbour.density;
                let kernel = poly6_kernel_gradient(u.smoothing_radius, dir_to_neighbour);
                
                sum += (u.particle_mass / neighbour_density) * kernel;

                // func end


            }

        }
    }



/*
    for (var i = 0u; i < u.particle_count; i = i + 1u) {
        let neighbour = in_particles[i];
        let neighbour_pos = neighbour.predicted_position;

        let offset_to_neighbour = neighbour_pos - point;
        let sqr_dst_to_neighbour = dot(offset_to_neighbour, offset_to_neighbour);

        if sqr_dst_to_neighbour > u.sqr_radius {
            continue;
        }


        let dst = sqrt(sqr_dst_to_neighbour);

        var dir_to_neighbour = vec2<f32>(0.0);
        if dst == 0.0 {
            dir_to_neighbour = normalize(vec2<f32>(rand_f32(&rand_seed), rand_f32(&rand_seed)));
        } else {
            dir_to_neighbour = offset_to_neighbour / dst;
        }

        let neighbour_density = neighbour.density;
        let kernel = poly6_kernel_gradient(u.smoothing_radius, dir_to_neighbour);
        
        sum += (u.particle_mass / neighbour_density) * kernel;
    }
*/

    return sum;

}
