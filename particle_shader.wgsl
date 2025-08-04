struct Vertex {
    @location(0) position: vec2<f32>,
}



struct Instance {
    @location(1) position: vec2<f32>,
    @location(2) predicted_position: vec2<f32>,
    @location(3) velocity: vec2<f32>,
    @location(4) density: f32,
    @location(5) grid: u32,
}


struct Fragment {
    @builtin(position) position: vec4<f32>,
    @location(0) modulate: vec4<f32>,
}



struct Uniforms {
    projection: mat4x4<f32>,
    pad00: f32,
    scale: f32,
    grid_size: u32,
    grid_w: u32,
    colour0: vec4<f32>,
    colour1: vec4<f32>,
    colour2: vec4<f32>,
    colour3: vec4<f32>,
}


@group(0) @binding(0) var<uniform> u : Uniforms;



@vertex
fn vs_main(vertex: Vertex, instance: Instance) -> Fragment {
    var output : Fragment;
    let pos = (vertex.position * u.scale) + instance.position - u.scale * 0.5;
    output.position = u.projection * vec4<f32>(pos.x, pos.y, 0.0, 1.0);

    let step = length(instance.velocity) * 0.05;
    let grid_x = instance.grid % u.grid_w;
    let grid_y = instance.grid / u.grid_w;
    let grid_h = u.grid_size / u.grid_w;

    var colour = vec4<f32>(0.0);
    if step < 0.4 {
        colour = mix(u.colour0, u.colour1, step/0.4);
    } else if step < 0.85 {
        colour = mix(u.colour1, u.colour2, (step-0.4)/0.45);
    } else {
        colour = mix(u.colour2, u.colour3, (step-0.85)/0.15);
    }

    output.modulate = colour;

    return output;
}

@fragment
fn fs_main(fragment: Fragment) -> @location(0) vec4<f32> {
    return fragment.modulate;
}
