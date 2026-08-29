#version 410 core

// The live camera feed filling the canvas.
//
// The camera texture holds BGR bytes exactly as OpenCV delivered them --
// converting on the CPU would cost a full-frame pass per frame for
// something a swizzle does for free.

in vec2 v_canvas;
out vec4 f_color;

uniform sampler2D u_video;
uniform vec4 u_uv;          // (ax, bx, ay, by): uv = a * canvas_px + b
uniform float u_exposure;
uniform float u_alpha;

void main() {
    vec2 uv = vec2(v_canvas.x * u_uv.x + u_uv.y,
                   v_canvas.y * u_uv.z + u_uv.w);
    vec3 live = texture(u_video, uv).bgr;
    f_color = vec4(live * u_exposure, u_alpha);
}
