//

#version 430

in vec3 DrawPosition;
in vec3 DrawNormal;
in vec2 DrawTexcoord;

layout(binding = 0) uniform sampler2D InColor;

layout(binding = 4, std430) buffer UBO_SYNC
{
  vec4 TimeDeltaTimeWidthHeight;
};


layout(binding = 2, std430) buffer UBO_MTL
{
  vec4 KaPh;
  vec4 KdTr;
  vec4 Ks4;
  vec4 Empty;
};

out vec4 FragColor;

void main() {
    // --- old experimental version (kept for reference) ---
    // vec3 v = DrawPosition * DrawNormal * vec3(DrawTexcoord, 1) * 0.001;
    // FragColor = vec4(DrawTexcoord.xyx * 0 + KaPh.xyz + v, 1);
    // --- active version ---
    vec3 v = DrawPosition * DrawNormal * vec3(DrawTexcoord, 1) * 0.001;
    float t = TimeDeltaTimeWidthHeight.x;
    vec3 c = texture(InColor, DrawTexcoord).rgb;
    /*FragColor = vec4(vec3((sin(TimeDeltaTimeWidthHeight.x) + 1)/2, gl_FragCoord.x / TimeDeltaTimeWidthHeight.z,
      gl_FragCoord.y / TimeDeltaTimeWidthHeight.w), 1);*/
    
    FragColor = vec4(DrawNormal, 1);

}
