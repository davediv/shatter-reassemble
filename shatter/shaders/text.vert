#version 410 core

in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;

out vec2 v_uv;
out vec4 v_color;

uniform vec2 u_canvas_size;

void main() {
    vec2 ndc = vec2(in_pos.x / u_canvas_size.x * 2.0 - 1.0,
                    1.0 - in_pos.y / u_canvas_size.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}
