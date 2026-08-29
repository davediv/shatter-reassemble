#version 410 core

// The atlas is a single-channel coverage map. Sampling it as alpha keeps
// the glyphs tintable and lets one atlas serve every colour on the HUD.

in vec2 v_uv;
in vec4 v_color;
out vec4 f_color;

uniform sampler2D u_atlas;

void main() {
    float coverage = texture(u_atlas, v_uv).r;
    if (coverage < 0.02) discard;
    f_color = vec4(v_color.rgb, v_color.a * coverage);
}
