#version 330

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 frag_normal;
out vec2 frag_uv;

void main() {
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    frag_normal = mat3(model) * in_normal;
    frag_uv = in_uv;
}
