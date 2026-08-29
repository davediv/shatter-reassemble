#version 410 core

// Glass, not coloured triangles.
//
// Three things separate a shard from a flat polygon: a lit bevel that
// catches a highlight as it tumbles, a dark side wall that gives it
// thickness, and refraction of whatever lies behind it. The first two are
// nearly free. The third samples the already-rendered background -- the
// black void with the person's silhouette in it -- displaced along the
// edge normal, so the void visibly bends through the glass.

in vec2 v_uv;
in vec2 v_screen;
in vec2 v_normal;
in float v_edge;
in float v_part;
in float v_depth;
in float v_alpha;
in float v_flash;

out vec4 f_color;

uniform sampler2D u_frozen;
uniform sampler2D u_scene;
uniform vec2 u_canvas_size;
uniform vec2 u_light;
uniform float u_refraction;    // 0 disables it (ladder rung 3)
uniform float u_bevel_shade;   // 0 disables bevel shading (ladder rung 4)
uniform float u_shadow;        // 1 = shadow pass: flat, dark, translucent
uniform float u_shadow_alpha;

void main() {
    if (u_shadow > 0.5) {
        // Flat and dark; the shape is the whole point of a shadow.
        f_color = vec4(0.0, 0.0, 0.0, u_shadow_alpha * v_alpha);
        return;
    }
    vec3 base = texture(u_frozen, v_uv).bgr;

    // Depth shading, so the pile reads as a pile and not as a decal.
    float shade = 0.72 + v_depth * 0.42;

    vec3 color;
    if (v_part < 0.5) {
        // Side wall: the glass seen edge-on. Dark, and tinted by the face
        // it belongs to so it never looks like a black outline.
        color = base * 0.16 + vec3(0.02, 0.03, 0.05);
        color *= 0.75 + v_edge * 0.25;
    } else if (v_part < 1.5 && u_bevel_shade > 0.5) {
        // Bevel rim. The lambert term makes edges flare as shards tumble,
        // which is what sells them as glass rather than paper.
        vec2 n = normalize(v_normal + vec2(1e-6));
        float lambert = max(dot(n, u_light), 0.0);
        float rim = pow(lambert, 3.0);

        vec3 lit = base * (0.55 + 0.35 * lambert) + vec3(rim * 0.55);
        if (u_refraction > 0.0) {
            vec2 suv = (v_screen + n * u_refraction * (1.0 - v_edge))
                       / u_canvas_size;
            vec3 behind = texture(u_scene, clamp(suv, 0.0, 1.0)).rgb;
            lit = mix(lit, lit + behind * 0.85, 0.45 * (1.0 - v_edge));
        }
        color = lit;
    } else {
        color = base;
        if (u_refraction > 0.0) {
            // A whisper of the void bleeding through the body of the
            // shard. Subtle on purpose: the face has to stay a legible
            // piece of the frozen room.
            vec2 suv = (v_screen + v_normal * u_refraction * 0.25) / u_canvas_size;
            color += texture(u_scene, clamp(suv, 0.0, 1.0)).rgb * 0.10;
        }
    }

    color *= shade;
    color += vec3(v_flash);
    f_color = vec4(color, v_alpha);
}
