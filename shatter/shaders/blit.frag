#version 410 core

// Canvas -> window. The canvas is a fixed 1920x1080 regardless of window
// size, so this contains it in the window and blacks out the remainder
// rather than stretching a shattered room out of shape.

in vec2 v_canvas;
out vec4 f_color;

uniform sampler2D u_canvas;
uniform vec4 u_rect;        // (offset_x, offset_y, scale_x, scale_y)

void main() {
    vec2 uv = (v_canvas - u_rect.xy) * u_rect.zw;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    f_color = vec4(texture(u_canvas, uv).rgb, 1.0);
}
