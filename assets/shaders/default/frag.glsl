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

layout(binding = 3, std430) buffer UBO_CAMERA
{
/*
  mat4 MatrView;
  mat4 MatrProj;
  mat4 MatrVP;
*/
  mat4 MatrVP;
  mat4 MatrView;
  mat4 MatrProj;
  //mat4 MatrShadow;
  vec4 CamLoc4;
  vec4 CamDir4;
};


layout(binding = 2, std430) buffer UBO_MTL
{
  vec4 KaPh;
  vec4 KdTr;
  vec4 Ks4;
  vec4 Empty;
  ivec4 TexMask0;
  ivec4 TexMask1;
};
#define Ka KaPh.rgb
#define Kd KdTr.rgb
#define Ph KaPh.a
#define Tr KdTr.a
#define Ks Ks4.rgb


out vec4 FragColor;

#define CamLoc CamLoc4.xyz
#define Time TimeDeltaTimeWidthHeight.x

vec3 Shade( vec3 P, vec3 N, vec3 InKa, vec3 InKd, vec3 InKs, float InPh )
{
  vec3 V = normalize(P - CamLoc);
  //vec3 L = (vec3(sin(Time) * 0.30 * 5 * 5, 8, 1));
  //vec3 LDir = vec3(0, -1, 0);
  //L = CamLoc;
  vec3 LightPos = vec3(1, 1.5, 1) * 15;

  vec3 L = LightPos - P;
 
  float d = length(L);
  L /= d;
 
  float
    cc = 1,
    cl = 0.01,
    cq = 0.0001,
    att = min(1, 1 / (cc + cl * d + cq * d * d));
  /*
  float ld = dot(-L, LDir);
  float cut = cos(radians(47.0));
  if (ld < cut)
    att = 0;
  else
    att *= pow(1 - cut / ld, 0.47);
 
  att = 1;
  */
  vec3 color = min(vec3(0.1), InKa);
 
  N = faceforward(N, V, N);
 
  color += InKd * max(0.0, dot(N, L));
  //Specular
  vec3 R;
  color += InKs * max(0, pow(dot((R = reflect(V, N)), L), InPh));
 
  color *= att;
  /*
  // Add sky sphere color
  vec2
    ts = vec2(atan(R.x, R.z) / (2 * acos(-1)),
              -acos(R.y) / acos(-1));
  vec4
    skyc = texture(SkyTex, ts);
  color += 0.47 * MtlKs * skyc.rgb;
  */
  /*
  vec2 tv = pow(abs(sin(DrawPosition.xz * 0.30 + Time)), vec2(111));
  float t = tv.x * tv.y;
  color = mix(vec3(0, 1, 1), color, 1 - t);
  tv = pow(abs(sin(DrawPosition.xz * 0.30 + Time)), vec2(122115));
  t = tv.x + tv.y;
  color = mix(vec3(0, 1, 0), color, 1 - t);
  */
 
  return color;
}


void main() {
    // --- old experimental version (kept for reference) ---
    // vec3 v = DrawPosition * DrawNormal * vec3(DrawTexcoord, 1) * 0.001;
    // FragColor = vec4(DrawTexcoord.xyx * 0 + KaPh.xyz + v, 1);
    // --- active version ---
    float t = TimeDeltaTimeWidthHeight.x;
    vec3 c = texture(InColor, DrawTexcoord).rgb;
    /*FragColor = vec4(vec3((sin(TimeDeltaTimeWidthHeight.x) + 1)/2, gl_FragCoord.x / TimeDeltaTimeWidthHeight.z,
      gl_FragCoord.y / TimeDeltaTimeWidthHeight.w), 1);*/
    vec3 v = (CamLoc4.xyz - DrawPosition);
    vec3 n = DrawNormal;

    //int b = dot(n, v) > 0 ? 1 : 0;
    //n = n * (b) -n * (1 - b);

    n = normalize(n);

    /*vec3 LightPos = vec3(1, 1.5, 1) * 15;

    vec3 l = LightPos - DrawPosition;

    vec3 C = vec3(0.47, 0.30, 0.18) * dot(l, n) / (0.8 +length(l)); 
    */

    int ht = TexMask0.x;
    
    vec3 col = Kd * (1-ht) + ht * texture(InColor, DrawTexcoord).rgb;
    


    vec3 C = Shade(DrawPosition, n, col, col, col, 30 * 3);

    FragColor = vec4(C, 1);

}
