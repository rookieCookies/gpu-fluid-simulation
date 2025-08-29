struct Settings {
    group_width: u32,
    group_height: u32,
    step_index: u32,
    num_values: u32,
}


struct ParticleInstance {
    position: vec2<f32>,
    predicted_position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    grid: u32,
}


@group(0) @binding(0)
var<storage> settings : Settings;


@group(1) @binding(0)
var<storage, read_write> values : array<ParticleInstance>;



@compute @workgroup_size(128)
fn sort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let h = i & (settings.group_width - 1u);
    let index_low = h + (settings.group_height + 1u) * (i / settings.group_width);
    let index_high = index_low + select(
        (settings.group_height + 1u) / 2u,
        settings.group_height - 2u * h,
        settings.step_index == 0u
    );


    if index_high >= settings.num_values {
        return;
    }


    let value_low = values[index_low];
    let value_high = values[index_high];

    if value_low.grid > value_high.grid {
        values[index_low] = value_high;
        values[index_high] = value_low;
    }
}
