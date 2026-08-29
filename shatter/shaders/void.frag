#version 410 core

// What is left when the room is on the floor: black, and the outline of
// the person still moving in it.
//
// The outline is a gradient of the segmentation mask rather than a
// threshold. A hard threshold gives a jagged edge that crawls frame to
// frame as the mask flickers; the gradient of a soft confidence value
// gives a band that thickens and thins smoothly, which reads as a glow
// instead of a jitter. The mask is only 256x144, and that is fine --
// bilinear sampling turns its coarseness into exactly the soft edge we
// want anyway.

in vec2 v_canvas;
out vec4 f_color;

uniform sampler2D u_mask;
uniform vec4 u_uv;              // uv = a * canvas_px + b, mirror baked in
uniform vec2 u_mask_texel;      // 1 / mask size
uniform float u_outline;        // outline brightness
uniform float u_fill;           // interior brightness
uniform vec3 u_tint;

void main() {
    vec2 uv = vec2(v_canvas.x * u_uv.x + u_uv.y,
                   v_canvas.y * u_uv.z + u_uv.w);

    float centre = texture(u_mask, uv).r;
    float left   = texture(u_mask, uv - vec2(u_mask_texel.x, 0.0)).r;
    float right  = texture(u_mask, uv + vec2(u_mask_texel.x, 0.0)).r;
    float up     = texture(u_mask, uv - vec2(0.0, u_mask_texel.y)).r;
    float down   = texture(u_mask, uv + vec2(0.0, u_mask_texel.y)).r;

    float gradient = length(vec2(right - left, down - up)) * 2.2;
    float edge = clamp(gradient, 0.0, 1.0);
    edge = pow(edge, 0.65);

    vec3 color = u_tint * (edge * u_outline + centre * u_fill);
    f_color = vec4(color, 1.0);
}
