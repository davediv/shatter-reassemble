#version 410 core

// Every shard in the pile, in one draw call.
//
// Per-shard transforms live in a small RGBA32F texture rather than in
// per-instance attributes. Instancing wants identical geometry per
// instance, and Voronoi cells are anything but -- they range from three
// to thirteen vertices. Padding them all to a fixed topology would waste
// a fifth of the vertex budget on degenerate triangles. So the geometry
// is one static batch tagged with a shard index, the transforms are a
// 3-row texture rewritten each frame (12.8KB, measured at 0.01ms), and
// the whole pile is a single glDrawArrays with no per-shard CPU work.

in vec2 in_local;     // position on the OUTER cell boundary
in vec2 in_inset;     // offset that insets it for the bevel
in float in_edge;     // bevel: 0 outer, 1 inner. wall: 0 near, 1 extruded
in float in_part;     // 0 wall, 1 bevel, 2 face
in float in_shard;
in vec2 in_normal;    // outward edge normal, shard-local

out vec2 v_uv;        // into the frozen frame, from the REST pose
out vec2 v_screen;    // canvas px, for sampling what is behind
out vec2 v_normal;    // rotated into world
out float v_edge;
out float v_part;
out float v_depth;
out float v_alpha;
out float v_flash;

uniform sampler2D u_shards;
uniform vec2 u_canvas_size;
uniform vec4 u_uv;            // uv = a * rest_px + b, mirror baked in
uniform float u_bevel;        // 1 normally, 0 when fully reassembled
uniform float u_thickness;
uniform float u_perspective;
uniform vec2 u_shadow_offset;   // zero for the lit pass

void main() {
    int sid = int(in_shard + 0.5);
    vec4 xf = texelFetch(u_shards, ivec2(sid, 0), 0);   // x, y, cos, sin
    vec4 st = texelFetch(u_shards, ivec2(sid, 1), 0);   // rest x, y, depth, -
    vec4 ex = texelFetch(u_shards, ivec2(sid, 2), 0);   // alpha, flash, scale, -

    // The bevel is animated, not baked. Driving it to zero returns every
    // vertex to the true cell boundary, so neighbouring shards close up
    // exactly and reassembly lands without a hairline crack.
    vec2 local = in_local + in_inset * u_bevel;

    // UVs come from the rest pose, so each shard carries its own slice of
    // the frozen frame wherever it ends up on the floor. Taken before the
    // wall extrusion so a side wall samples its owning edge.
    vec2 rest = local + st.xy;
    v_uv = vec2(rest.x * u_uv.x + u_uv.y, rest.y * u_uv.z + u_uv.w);

    vec2 centre = u_canvas_size * 0.5;
    vec2 offset = (xf.xy - centre) / centre;

    // Thickness is faked as parallax: the side walls extrude away from the
    // centre of frame, by more for shards nearer the viewer. Shards in the
    // middle show no side, shards at the edges show a lot -- which is what
    // real perspective does, for one multiply.
    vec2 extrude = offset * u_thickness * (0.45 + st.z);
    vec2 world_normal = vec2(xf.z * in_normal.x - xf.w * in_normal.y,
                             xf.w * in_normal.x + xf.z * in_normal.y);
    if (in_part < 0.5) {
        // Walls on edges facing away are hidden by the face anyway; drop
        // them off-screen rather than paying to rasterise and depth-test.
        if (dot(world_normal, extrude) <= 0.0) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            return;
        }
    }

    vec2 rotated = vec2(xf.z * local.x - xf.w * local.y,
                        xf.w * local.x + xf.z * local.y) * ex.z;
    vec2 world = rotated + xf.xy;
    if (in_part < 0.5) {
        world += extrude * in_edge;
    }

    // Fake perspective: near shards swell slightly, far ones shrink.
    world = centre + (world - centre) * (1.0 + (st.z - 0.5) * u_perspective);
    // The shadow pass is the same geometry, displaced. Scaling the
    // offset by depth makes near shards throw their shadow further,
    // which is the cheapest possible cue that the pile has depth.
    world += u_shadow_offset * (0.5 + st.z);

    v_screen = world;
    v_normal = world_normal;
    v_edge = in_edge;
    v_part = in_part;
    v_depth = st.z;
    v_alpha = ex.x;
    v_flash = ex.y;

    // Depth buffer resolves 800 overlapping shards without any CPU sort.
    // Nearer shards (higher depth) get smaller z. The per-part nudge keeps
    // a shard's own face in front of its bevel and wall.
    float z = (1.0 - st.z) * 0.9 + 0.05 - in_part * 0.0006;
    gl_Position = vec4(world.x / u_canvas_size.x * 2.0 - 1.0,
                       1.0 - world.y / u_canvas_size.y * 2.0,
                       z * 2.0 - 1.0, 1.0);
}
