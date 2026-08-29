#version 410 core

// Fullscreen coverage from gl_VertexID alone -- no vertex buffer, no VAO
// binding, three vertices. Emits canvas-pixel coordinates with y pointing
// down, matching the convention physics and fracture use so that nothing
// in this codebase ever has to flip y.

out vec2 v_canvas;

uniform vec2 u_canvas_size;

void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    v_canvas = vec2(p.x, 1.0 - p.y) * u_canvas_size;
}
