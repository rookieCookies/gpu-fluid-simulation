struct Vertex {
    @location(0) position: vec2<f32>,
}


struct FluidObject {
    pos: vec2<f32>,
    kind: u32,
    pad: u32,
    pad2: vec4<u32>,
}


struct Uniforms {
    inv_proj: mat4x4<f32>,
    pad: vec3<u32>,
    ssbo_len: u32,
}


@group(0) @binding(0) var<uniform> uniform : Uniforms ;
@group(1) @binding(0) var<storage, read_write> ssbo : array<FluidObject>;


struct Fragment {
    @builtin(position) screen_position: vec4<f32>,
    @location(0) clip_position: vec4<f32>,
}



@vertex
fn vs_main(vertex: Vertex) -> Fragment {
    var output : Fragment;
    output.screen_position = vec4((vertex.position-vec2(0.5)) * 2.0, 0.0, 1.0);
    output.clip_position = output.screen_position;
    return output;
}


@fragment
fn fs_main(input: Fragment) -> @location(0) vec4<u32> {
    let clip_pos = input.clip_position;
    let inv_pos = uniform.inv_proj * clip_pos;
    let world_pos = (inv_pos.xyz / inv_pos.w).xy;

    for (var i = 0u; i < uniform.ssbo_len; i = i + 1) {
        let obj = ssbo[i];
        
        if obj.kind == 0 {
            let dst = distance(world_pos, obj.pos);
            let radius = bitcast<f32>(obj.pad);
            if dst < radius {
                return vec4<u32>(0);
            }
        } else if obj.kind == 1 {
            let rot = bitcast<f32>(obj.pad);
            let extents = bitcast<vec2<f32>>(obj.pad2.xy);
            if point_in_rect(world_pos, obj.pos, extents, rot) {
                return vec4<u32>(0);
            }
        }

    }

    return vec4<u32>(255);
}


fn point_in_rect(point: vec2<f32>, rect_center: vec2<f32>, size: vec2<f32>, rot: f32) -> bool {
    // translate point to rect local space
    let local = point - rect_center;

    // rotate point by -rot to align with rectangle axes
    let c = cos(-rot);
    let s = sin(-rot);
    let rotated = vec2<f32>(
        local.x * c - local.y * s,
        local.x * s + local.y * c
    );

    // check if inside axis-aligned rect
    let half_size = size * 0.5;
    return all(rotated >= -half_size) && all(rotated <= half_size);
}
