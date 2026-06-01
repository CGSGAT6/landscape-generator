//
#version 430

layout(location=0) in vec3 InPosition;
layout(location=1) in vec3 InNormal;
layout(location=2) in vec2 InTexcoord;

out vec3 DrawPosition;
out vec3 DrawNormal;
out vec2 DrawTexcoord;

layout(binding = 4, std430) buffer UBO_SYNC
{
  vec4 TimeDeltaTimeWidthHeight;
};

layout(binding = 1, std430) buffer UBO_PRIM
{
  mat4 MatrW;
  mat4 MatrWInv;
  mat4 MatrWVP;
};

layout(binding = 3, std430) buffer UBO_CAMERA
{
  mat4 MatrView;
  mat4 MatrProj;
  mat4 MatrVP;
};

/*
mat4 MatrVP;
  mat4 MatrV;
  mat4 MatrP;
  mat4 MatrShadow;
  vec4 CamLoc4;
  vec4 CamRight4;
  vec4 CamUp4;
  vec4 CamDir4;
  vec4 FrameWHProjSizeDist;
  */

void main() {
    // --- old experimental versions (kept for reference) ---
    //    gl_Position = vec4(InPosition, 1);//MatrWVP * vec4(InPosition, 1);
    //    gl_Position = MatrWVP * vec4(InPosition, 1);
    //    gl_Position = vec4(InPosition, 1);
        gl_Position = MatrProj * MatrView * MatrW * vec4(InPosition/*+vec3(sin(TimeDeltaTimeWidthHeight.x))*/, 1);
    //    //gl_Position = MatrVP * MatrW * vec4(InPosition, 1);
    // --- active version ---
    //gl_Position = MatrWVP * vec4(InPosition, 1);
    DrawNormal = mat3(MatrWInv) * InNormal;
    DrawPosition = vec3(MatrW * vec4(InPosition, 1));
    DrawTexcoord = InTexcoord;
}
