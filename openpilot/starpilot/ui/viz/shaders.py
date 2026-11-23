"""GLSL for the scene views.

One full-layer pass draws sky and road for every pixel by intersecting its view ray with the road plane, so the cost is
independent of how much road is visible. The camera variant samples the road camera at that road point (inverse
perspective mapping), so chase and top views reuse the live frame without any CPU work. On comma hardware the frame is
an EGL external image (zero copy, like CameraView); elsewhere it arrives as separate Y and UV textures.
"""
import platform

_DARWIN = platform.system() == "Darwin"
VERSION = "#version 330 core\n" if _DARWIN else "#version 300 es\nprecision highp float;\n"
EGL_VERSION = "#version 300 es\n#extension GL_OES_EGL_image_external_essl3 : enable\nprecision highp float;\n"

VERTEX_SHADER = VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec4 vertexColor;
uniform mat4 mvp;
out vec2 fragTexCoord;
out vec4 fragColor;
void main() {
  fragTexCoord = vertexTexCoord;
  fragColor = vertexColor;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

ROAD_SAMPLES = 16  # road edge samples passed to the ground shaders, ROAD_STEP metres apart from the camera
ROAD_STEP = 6.0

# Shared by both ground shaders. Ray directions and positions are in the calib frame (x forward, y right, z down).
_GROUND_COMMON = f"""
#define ROAD_SAMPLES {ROAD_SAMPLES}
""" + """
in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;

uniform vec3 uvToRay[3];    // columns: (u, v, 1) over the layer -> view ray
uniform vec3 camPos;        // virtual camera position
uniform float groundZ;      // road plane
uniform float odometer;     // metres driven, scrolls the grid
uniform float gridSize;
uniform float fogDistance;
uniform vec4 skyTop;
uniform vec4 skyHorizon;
uniform vec4 groundColor;
uniform vec4 gridColor;     // rgb, a = strength
uniform vec4 roadColor;     // surface between the model's road edges; a = 0 when there are none
uniform float roadLeft[ROAD_SAMPLES];
uniform float roadRight[ROAD_SAMPLES];
uniform float roadStep;

vec3 skyColor(vec3 dir) {
  return mix(skyHorizon.rgb, skyTop.rgb, smoothstep(0.0, 0.55, clamp(-dir.z, 0.0, 1.0)));
}

vec2 roadSpan(float x) {
  float s = clamp(x / roadStep, 0.0, float(ROAD_SAMPLES - 1) - 0.001);
  int i = int(s);
  float f = s - float(i);
  return vec2(mix(roadLeft[i], roadLeft[i + 1], f), mix(roadRight[i], roadRight[i + 1], f));
}

// 1 on the road (between the edges widened by `grow`), fading over `blur` metres
float roadMask(vec2 p, float grow, float blur) {
  if (roadColor.a <= 0.0) {
    return 0.0;
  }
  vec2 span = roadSpan(p.x) + vec2(-grow, grow);
  return smoothstep(span.x - blur, span.x + blur, p.y) * (1.0 - smoothstep(span.y - blur, span.y + blur, p.y));
}

vec3 syntheticRoad(vec2 p) {
  vec2 g = vec2((p.x + odometer) / gridSize, p.y / gridSize);
  vec2 fw = max(fwidth(g), vec2(1e-5));
  vec2 a = abs(fract(g - 0.5) - 0.5) / fw;
  float line = 1.0 - min(min(a.x, a.y), 1.0);
  line *= 1.0 - smoothstep(0.12, 0.45, max(fw.x, fw.y));  // drop the grid where cells get only a few pixels
  float road = roadMask(p, 0.0, 0.12) * roadColor.a;
  vec3 base = mix(groundColor.rgb, roadColor.rgb, road);
  return mix(base, gridColor.rgb, line * gridColor.a * (1.0 - 0.5 * road));
}

float fogAmount(float distance) {
  return 1.0 - exp(-distance / fogDistance);
}

bool groundHit(out vec3 dir, out vec3 p, out float t) {
  dir = normalize(mat3(uvToRay[0], uvToRay[1], uvToRay[2]) * vec3(fragTexCoord, 1.0));
  if (dir.z < 1e-4) {
    return false;
  }
  t = (groundZ - camPos.z) / dir.z;
  p = camPos + dir * t;
  return t > 0.0;
}
"""

SYNTHETIC_GROUND_SHADER = VERSION + _GROUND_COMMON + """
void main() {
  vec3 dir, p;
  float t;
  if (!groundHit(dir, p, t)) {
    finalColor = vec4(skyColor(dir), 1.0);
    return;
  }
  finalColor = vec4(mix(syntheticRoad(p.xy), skyHorizon.rgb, fogAmount(t)), 1.0);
}
"""

_CAMERA_GROUND = _GROUND_COMMON + """
uniform vec3 groundToTex[3];  // columns: (X, Y, 1) on the road -> homogeneous texture uv of the real camera
uniform vec4 camLimits;       // x: nearest usable road distance, y/z: distance fade start/end, w: lowest usable v (hood)
uniform vec4 camOptics;       // xy: principal point, zw: focal length, both in texture uv units
uniform float maxTan2;        // widest usable ray (squared tangent) for the fisheye; 0 = no limit
uniform int mode;             // 1: camera colour, 2: hybrid (synthetic road + markings found in the camera)
uniform float colorStrength;
uniform vec4 paintWhite;
uniform vec4 paintYellow;

ROAD_COLOR

vec2 texAt(vec2 g) {
  vec3 c = mat3(groundToTex[0], groundToTex[1], groundToTex[2]) * vec3(g, 1.0);
  return c.xy / max(c.z, 1e-6);
}

float cameraWeight(vec2 g, out vec2 uv) {
  vec3 c = mat3(groundToTex[0], groundToTex[1], groundToTex[2]) * vec3(g, 1.0);
  uv = c.xy / max(c.z, 1e-6);
  if (c.z <= 1e-4 || g.x < camLimits.x) {
    return 0.0;
  }
  vec2 n = (uv - camOptics.xy) / camOptics.zw;
  // smoothstep edges must increase (reversed edges are undefined in GLSL), so falling edges use 1 - smoothstep
  float w = smoothstep(0.0, 0.015, uv.x) * (1.0 - smoothstep(0.985, 1.0, uv.x)) *
            smoothstep(0.0, 0.015, uv.y) * (1.0 - smoothstep(camLimits.w - 0.02, camLimits.w, uv.y));
  if (maxTan2 > 0.0) {
    w *= 1.0 - smoothstep(maxTan2 * 0.85, maxTan2, dot(n, n));
  }
  w *= smoothstep(camLimits.x, camLimits.x + 1.5, g.x);
  return w * (1.0 - smoothstep(camLimits.y, camLimits.z, length(g)));
}

float luma(vec3 c) {
  return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
  vec3 dir, p;
  float t;
  if (!groundHit(dir, p, t)) {
    finalColor = vec4(skyColor(dir), 1.0);
    return;
  }
  vec3 color = syntheticRoad(p.xy);
  vec2 uv;
  float w = cameraWeight(p.xy, uv);
  if (w > 0.0) {
    vec3 cam = cameraRgb(uv);
    if (mode == 1) {
      color = mix(color, road_color(cam, colorStrength), w);
    } else {
      // Paint is brighter than the road on BOTH sides of it: across for lane lines, along for stop lines and crosswalks.
      float across = max(luma(cameraRgb(texAt(p.xy + vec2(0.0, 0.35)))), luma(cameraRgb(texAt(p.xy - vec2(0.0, 0.35)))));
      float along = max(luma(cameraRgb(texAt(p.xy + vec2(0.6, 0.0)))), luma(cameraRgb(texAt(p.xy - vec2(0.6, 0.0)))));
      float l0 = luma(cam);
      float background = min(across, along);
      float contrast = l0 - background;
      float paint = smoothstep(0.05, 0.12, contrast) * smoothstep(0.15, 0.35, contrast / max(background, 0.05));
      // Only near-white or yellow counts, which rejects grass, foliage and brake lights.
      float high = max(cam.r, max(cam.g, cam.b));
      float saturation = (high - min(cam.r, min(cam.g, cam.b))) / max(high, 1e-3);
      float redGreen = cam.r / max(cam.g, 1e-3);
      float yellow = smoothstep(0.82, 0.95, redGreen) * (1.0 - smoothstep(1.35, 1.6, redGreen)) * smoothstep(0.25, 0.45, saturation);
      float white = 1.0 - smoothstep(0.16, 0.30, saturation);
      paint *= max(white, yellow) * roadMask(p.xy, 0.4, 0.4);
      color = mix(color, mix(paintWhite.rgb, paintYellow.rgb, yellow), paint * w);
    }
  }
  finalColor = vec4(mix(color, skyHorizon.rgb, fogAmount(t)), 1.0);
}
"""

_EGL_SAMPLER = """
uniform samplerExternalOES texture0;
vec3 cameraRgb(vec2 uv) {
  return texture(texture0, uv).rgb;
}
"""

_YUV_SAMPLER = """
uniform sampler2D texture0;  // Y
uniform sampler2D texture1;  // interleaved UV as luminance-alpha
vec3 cameraRgb(vec2 uv) {
  float y = texture(texture0, uv).r;
  vec2 c = texture(texture1, uv).ra - 0.5;
  return vec3(y + 1.402 * c.y, y - 0.344 * c.x - 0.714 * c.y, y + 1.772 * c.x);
}
"""


def camera_ground_shader(egl: bool, road_color: str) -> str:
  """Fragment shader for camera/hybrid modes. ``road_color`` supplies ``vec3 road_color(vec3, float)``."""
  header = EGL_VERSION if egl else VERSION
  sampler = _EGL_SAMPLER if egl else _YUV_SAMPLER
  return header + _CAMERA_GROUND.replace("ROAD_COLOR", sampler + road_color)


# Lit sphere drawn on a screen-aligned square (an impostor): cheaper and smoother than a tessellated mesh.
SPHERE_SHADER = VERSION + """
in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;
uniform vec3 lightDir;  // x right, y up, z towards the viewer
void main() {
  vec2 p = fragTexCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  float aa = max(fwidth(r2), 1e-4);
  float coverage = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r2);
  if (coverage <= 0.0) {
    discard;
  }
  vec3 n = vec3(p.x, -p.y, sqrt(max(1.0 - r2, 0.0)));
  float diffuse = max(dot(n, lightDir), 0.0);
  float spec = pow(max(dot(n, normalize(lightDir + vec3(0.0, 0.0, 1.0))), 0.0), 36.0);
  float rim = pow(1.0 - n.z, 2.5);
  vec3 c = fragColor.rgb * (0.28 + 0.78 * diffuse) + vec3(0.55 * spec) + fragColor.rgb * rim * 0.4;
  finalColor = vec4(c, fragColor.a * coverage);
}
"""
