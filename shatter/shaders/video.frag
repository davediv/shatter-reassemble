#version 410 core

// The live camera feed, or the frozen frame, filling the canvas.
//
// The camera texture holds BGR bytes exactly as OpenCV delivered them --
// converting on the CPU would cost a full-frame pass per frame for
// something a swizzle does for free.

in vec2 v_canvas;
out vec4 f_color;

uniform sampler2D u_video;
uniform sampler2D u_frozen;
uniform vec4 u_uv;          // (ax, bx, ay, by): uv = a * canvas_px + b
uniform float u_freeze;     // 0 = live, 1 = frozen
uniform float u_exposure;
uniform float u_alpha;

void main() {
    vec2 uv = vec2(v_canvas.x * u_uv.x + u_uv.y,
                   v_canvas.y * u_uv.z + u_uv.w);
    vec3 color;
    if (u_freeze <= 0.0) {
        color = texture(u_video, uv).bgr;
    } else if (u_freeze >= 1.0) {
        color = texture(u_frozen, uv).bgr;
    } else {
        vec3 live = texture(u_video, uv).bgr;
        vec3 frozen = texture(u_frozen, uv).bgr;
        color = mix(live, frozen, u_freeze);
    }
    f_color = vec4(color * u_exposure, u_alpha);
}
