#version 330

in vec3 frag_normal;
in vec2 frag_uv;

uniform sampler2D texture0;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform float light_intensity;

out vec4 final_color;

void main() {
    vec3 N = normalize(frag_normal);
    float diff = max(dot(N, normalize(-light_direction)), 0.0);
    vec3 ambient = vec3(0.2);
    vec3 diffuse = diff * light_color * light_intensity;
    vec4 tex_color = texture(texture0, frag_uv);
    vec3 result = tex_color.rgb * (ambient + diffuse);
    final_color = vec4(result, 1.0);
}
