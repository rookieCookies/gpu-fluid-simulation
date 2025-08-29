struct Settings {
    width: u32,
    height: u32,
    jump: u32,
}


@group(0) @binding(0)
var<storage> settings : Settings;


@group(1) @binding(0)
var in_values : texture_storage_2d<rg32uint, read>;
@group(1) @binding(1)
var out_values : texture_storage_2d<rg32uint, write>;

@compute @workgroup_size(128, 128, 1)
fn floodfill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.xy;

    if (pos.x >= settings.width || pos.y >= settings.height) {
        return;
    }

    let x = i32(pos.x);
    let y = i32(pos.y);

    let pixel = textureLoad(in_values, pos);
    var best = pixel.xy;
    var best_d = sq_dist(x, y, i32(best.x), i32(best.y));


    let j = i32(settings.jump);
    // Unrolled 8 offsets at distance j
    let offs = array<vec2<i32>, 8>(
        vec2<i32>(-j, -j), vec2<i32>( 0, -j), vec2<i32>( j, -j),
        vec2<i32>(-j,  0),                     vec2<i32>( j,  0),
        vec2<i32>(-j,  j), vec2<i32>( 0,  j), vec2<i32>( j,  j)
    );


    for (var i = 0u; i < 8u; i = i + 1u) {
        let nx = x + offs[i].x;
        let ny = y + offs[i].y;
        if (nx < 0 || ny < 0 || nx >= i32(settings.width) || ny >= i32(settings.height)) { continue; }

        let n = textureLoad(in_values, vec2<i32>(nx, ny));
        if (n.x > settings.width || n.y > settings.width) { continue; } // neighbor has no valid seed

        let d = sq_dist(x, y, i32(n.x), i32(n.y));
        if (d < best_d) {
            best = n.xy; // keep pixel xy, update seed zw
            best_d = d;
        }
    }

    textureStore(out_values, pos, vec4(best, 0, 0));
}


fn sq_dist(ax: i32, ay: i32, bx: i32, by: i32) -> i32 {
    let dx = ax - bx;
    let dy = ay - by;
    return dx*dx + dy*dy;
}


/*
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let here = textureLoad(seeds_in, vec2<i32>(x, y), 0);
    // here.zw = current best seed coords (or -1, -1)
    var best = here;
    var best_d = select( // if invalid, start with "infinite" (large int)
        0x3fffffff,                                  // ~big
        sq_dist(x, y, best.z, best.w),
        best.z < 0 || best.w < 0
    );

    let j = i32(params.jump);
    // Unrolled 8 offsets at distance j
    let offs = array<vec2<i32>, 8>(
        vec2<i32>(-j, -j), vec2<i32>( 0, -j), vec2<i32>( j, -j),
        vec2<i32>(-j,  0),                     vec2<i32>( j,  0),
        vec2<i32>(-j,  j), vec2<i32>( 0,  j), vec2<i32>( j,  j)
    );

    for (var i = 0u; i < 8u; i = i + 1u) {
        let nx = x + offs[i].x;
        let ny = y + offs[i].y;
        if (nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height)) { continue; }

        let n = textureLoad(seeds_in, vec2<i32>(nx, ny), 0);
        if (n.z < 0 || n.w < 0) { continue; } // neighbor has no valid seed

        let d = sq_dist(x, y, n.z, n.w);
        if (d < best_d) {
            best = vec4<i32>(here.x, here.y, n.z, n.w); // keep pixel xy, update seed zw
            best_d = d;
        }
    }

    textureStore(seeds_out, vec2<i32>(x, y), best);
}

*/
