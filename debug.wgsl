#include funcs.wgsl


struct Vertex {
    @location(0) position: vec2<f32>,
}

struct Fragment {
    @builtin(position) position: vec4<f32>,
}


@vertex
fn vs_main(vertex: Vertex) -> Fragment {
    var output : Fragment;
    output.position = vec4(vertex.position * 100.0, 0.0, 0.0);

    return output;
}

@fragment
fn fs_main(fragment: Fragment) -> @location(0) vec4<f32> {
    return vec4(1.0);
}
