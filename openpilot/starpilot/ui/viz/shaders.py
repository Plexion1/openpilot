"""GLSL for the scene views.

Ground pass: one full-layer quad. Every pixel intersects its view ray with the road plane and is shaded there, so the
cost does not depend on how much road is visible and the CPU only uploads a few uniforms per frame. Everything that
lies on the road is drawn in this pass from small sampled arrays: road surface, lane lines (dashed lines scroll with
the distance driven), road edges, the planned path, the lane change highlight, the stop line, vehicle shadows,
headlight beams and brake light glow. The camera variant samples the road camera at that road point (inverse
perspective mapping); on comma hardware the frame is an EGL external image (zero copy, like CameraView), elsewhere it
arrives as separate Y and UV textures.

Vehicle pass: one small quad per vehicle, ray marched against a signed distance model of a car, pickup truck or
sphere, with paint, glass, tyres and working lights. Edges are antialiased from the closest approach of the ray.
"""
import platform

_DARWIN = platform.system() == "Darwin"
VERSION = "#version 330 core\n" if _DARWIN else "#version 300 es\nprecision highp float;\nprecision highp int;\n"
EGL_VERSION = "#version 300 es\n#extension GL_OES_EGL_image_external_essl3 : enable\nprecision highp float;\nprecision highp int;\n"

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

SAMPLES = 24       # must match scene.SAMPLES
MAX_BODIES = 4     # shadows / glows in the ground pass (our car + up to three others)

# Shared by both ground shaders. Ray directions and positions are in the calib frame (x forward, y right, z down).
_GROUND_COMMON = f"""
#define SAMPLES {SAMPLES}
#define MAX_BODIES {MAX_BODIES}
""" + """
in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;

uniform vec3 uvToRay[3];    // columns: (u, v, 1) over the layer -> view ray
uniform vec3 camPos;        // virtual camera position
uniform float groundZ;      // road plane
uniform float odometer;     // metres driven, scrolls the dashes and the grid
uniform float time;         // seconds, for pulses
uniform float fogDistance;
uniform vec4 skyTop;
uniform vec4 skyHorizon;
uniform vec4 groundColor;
uniform vec4 gridColor;     // rgb, a = strength
uniform vec4 roadColor;     // surface between the road edges; a = 0 when there are none

// Road geometry sampled at x_i = sampleMaxX * (i / (SAMPLES - 1))^2
uniform vec4 laneY[SAMPLES];  // y of the four lane lines
uniform vec4 roadY[SAMPLES];  // left edge y, right edge y, path y, planned acceleration
uniform float sampleMaxX;

uniform vec4 laneAlpha;     // per lane line (model probability x style)
uniform vec4 laneDash;      // 1 dashed, 0 solid
uniform vec4 laneInner;     // colour of the lines beside our lane, a = strength
uniform vec4 laneOuter;     // colour of the outer lines
uniform vec4 edgeStyle;     // xy: alpha of the left/right road edge, z: half width, w: lane line half width
uniform vec4 pathRange;     // x: start, y: end (m), z: alpha, w: colour mode (0 throttle, 1 solid, 2 acceleration, 3 rainbow)
uniform vec4 pathNear;
uniform vec4 pathFar;
uniform float pathHalfWidth;
uniform vec4 laneChange;    // x: direction (-1 left, 1 right), y: strength, z: blocked, w: unused
uniform vec4 stopLine;      // x: distance, y: strength
uniform vec4 lighting;      // x: darkness, y: headlights, z: our front bumper x, w: unused
uniform vec4 bodyPose[MAX_BODIES];   // shadow/glow centre x, y, cos(yaw), sin(yaw)
uniform vec4 bodySize[MAX_BODIES];   // half length, half width, shadow strength, tail glow
uniform vec4 statusGlow;    // our car's engagement colour under it, a = strength

const vec3 HEADLIGHT = vec3(1.0, 0.93, 0.78);
const vec3 TAIL = vec3(1.0, 0.12, 0.08);

vec3 skyColor(vec3 dir) {
  vec3 top = mix(skyTop.rgb, skyTop.rgb * 0.35, lighting.x);
  vec3 horizon = mix(skyHorizon.rgb, skyHorizon.rgb * 0.45, lighting.x);
  return mix(horizon, top, smoothstep(0.0, 0.55, clamp(-dir.z, 0.0, 1.0)));
}

float sampleX(int i) {
  float s = float(i) / float(SAMPLES - 1);
  return sampleMaxX * s * s;
}

// Lane lines and road at distance x; slopes give the true (perpendicular) distance to a curved line.
void roadAt(float x, out vec4 lanes, out vec4 road, out vec4 laneSlope, out float pathSlope) {
  float s = sqrt(clamp(x / sampleMaxX, 0.0, 1.0)) * float(SAMPLES - 1);
  int i = min(int(s), SAMPLES - 2);
  float f = clamp(s - float(i), 0.0, 1.0);
  vec4 l0 = laneY[i], l1 = laneY[i + 1];
  vec4 r0 = roadY[i], r1 = roadY[i + 1];
  lanes = mix(l0, l1, f);
  road = mix(r0, r1, f);
  float inside = step(0.0, x) * step(x, sampleMaxX);
  float dx = max(sampleX(i + 1) - sampleX(i), 1e-3);
  laneSlope = (l1 - l0) / dx * inside;
  pathSlope = (r1.z - r0.z) / dx * inside;
}

// Antialiased coverage of lines of half width w at signed lateral offsets delta; thinner than a pixel keeps its weight.
vec4 lineCoverage(vec4 delta, vec4 slope, float w) {
  vec4 d = abs(delta) * inversesqrt(1.0 + slope * slope);
  vec4 fw = max(fwidth(delta), vec4(1e-4)) * 0.7;
  vec4 width = max(vec4(w), fw * 0.5);
  return (1.0 - smoothstep(width - fw, width + fw, d)) * min(vec4(w) / width, vec4(1.0));
}

float band(float v, float lo, float hi, float soft) {
  return smoothstep(lo - soft, lo + soft, v) * (1.0 - smoothstep(hi - soft, hi + soft, v));
}

// 3 m dashes every 12 m (US highway spacing), scrolling as we drive; averaged out where they would shimmer.
float dashes(float x) {
  float u = (x + odometer) / 12.0;
  float fu = max(fwidth(u), 1e-4);
  float ph = fract(u);
  float m = smoothstep(0.0, fu, ph) * (1.0 - smoothstep(0.25 - fu, 0.25 + fu, ph));
  return mix(m, 0.25, smoothstep(0.08, 0.3, fu));
}

vec3 hsl(float h, float s, float l) {
  vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
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

// Ground colour under synthetic lighting: grid off the road, plain surface on it.
vec3 syntheticGround(vec2 p, vec4 road) {
  vec2 g = vec2((p.x + odometer) / 4.0, p.y / 4.0);
  vec2 fw = max(fwidth(g), vec2(1e-5));
  vec2 a = abs(fract(g - 0.5) - 0.5) / fw;
  float grid = (1.0 - min(min(a.x, a.y), 1.0)) * (1.0 - smoothstep(0.12, 0.45, max(fw.x, fw.y)));
  float onRoad = roadColor.a <= 0.0 ? 0.0 : band(p.y, road.x, road.y, 0.12) * roadColor.a;
  vec3 base = mix(groundColor.rgb, roadColor.rgb, onRoad);
  return mix(base, gridColor.rgb, grid * gridColor.a * (1.0 - onRoad));
}

// Everything painted onto the road, applied over the base colour (synthetic surface or camera image).
vec3 roadDecals(vec3 color, vec2 p, vec4 lanes, vec4 road, vec4 laneSlope, float pathSlope, bool synthetic) {
  float dark = lighting.x;
  // headlight beams: two soft cones from our bumper, brighter close in
  float beam = 0.0;
  if (lighting.y > 0.0) {
    float ahead = p.x - lighting.z;
    if (ahead > 0.0) {
      for (int s = -1; s <= 1; s += 2) {
        float off = abs(p.y - float(s) * 0.7) / (ahead + 1.5);
        beam += (1.0 - smoothstep(0.1, 0.42, off)) * smoothstep(0.5, 6.0, ahead) / (1.0 + ahead * ahead / 1600.0);
      }
    }
    beam = min(beam, 1.2) * lighting.y;
  }
  if (synthetic) {
    color *= mix(1.0, 0.45, dark);
    color += color * HEADLIGHT * beam * 1.6 + HEADLIGHT * beam * 0.035;
  } else {
    color += HEADLIGHT * beam * 0.06;
  }

  // road edges
  float edges = dot(lineCoverage(vec4(p.y - road.x, p.y - road.y, 0.0, 0.0), vec4(0.0), edgeStyle.z).xy, edgeStyle.xy);
  color = mix(color, vec3(1.0, 0.16, 0.16), clamp(edges, 0.0, 1.0));

  // lane change: tint the lane we are moving into (red while the blind spot is occupied)
  if (laneChange.y > 0.0) {
    float a = laneChange.x < 0.0 ? lanes.x : lanes.z;
    float b = laneChange.x < 0.0 ? lanes.y : lanes.w;
    float lane = band(p.y, min(a, b), max(a, b), 0.25) * band(p.x, 3.0, 55.0, 4.0);
    vec3 tint = laneChange.z > 0.5 ? vec3(1.0, 0.22, 0.18) : vec3(0.28, 0.72, 1.0);
    float flow = 0.75 + 0.25 * sin((p.x - time * 18.0) * 0.35);
    color = mix(color, tint, lane * laneChange.y * 0.32 * flow);
  }

  // lane lines: solid beside a road edge, dashed between lanes; retroreflective paint lights up in the beams
  vec4 cover = lineCoverage(vec4(p.y) - lanes, laneSlope, edgeStyle.w) * laneAlpha;
  float dash = dashes(p.x);
  cover *= mix(vec4(1.0), vec4(dash), laneDash);
  float glow = 1.0 + beam * 1.5;
  color = mix(color, laneOuter.rgb * glow, clamp(cover.x + cover.w, 0.0, 1.0) * laneOuter.a);
  color = mix(color, laneInner.rgb * glow, clamp(cover.y + cover.z, 0.0, 1.0) * laneInner.a);

  // planned path
  if (pathRange.z > 0.0 && pathRange.y > pathRange.x) {
    float d = (p.y - road.z) * inversesqrt(1.0 + pathSlope * pathSlope);
    float fw = max(fwidth(p.y - road.z), 1e-4);
    float side = 1.0 - smoothstep(pathHalfWidth - fw, pathHalfWidth + fw, abs(d));
    float along = smoothstep(pathRange.x, pathRange.x + 1.0, p.x) * (1.0 - smoothstep(pathRange.y - 1.5, pathRange.y, p.x));
    float t = clamp((p.x - pathRange.x) / max(pathRange.y - pathRange.x, 1.0), 0.0, 1.0);
    vec4 c = mix(pathNear, pathFar, t);
    if (pathRange.w > 2.5) {
      c = vec4(hsl(fract(p.x / 45.0 - time * 0.15), 0.9, 0.6), mix(0.55, 0.1, t));
    } else if (pathRange.w > 1.5) {
      float accel = road.w;
      float sat = min(abs(accel * 1.5), 1.0);
      c = vec4(hsl(clamp(60.0 + accel * 35.0, 0.0, 120.0) / 360.0, sat, mix(0.95, 0.62, sat)), mix(0.5, 0.0, t));
    }
    // a soft brighter rim reads better than a flat fill from above
    float rim = smoothstep(pathHalfWidth * 0.6, pathHalfWidth * 0.97, abs(d));
    color = mix(color, c.rgb, clamp(side * along * pathRange.z * c.a * (0.7 + 0.9 * rim), 0.0, 1.0));
  }

  // stop line: glowing red bar across our lane where the model wants to stop
  if (stopLine.y > 0.0) {
    float lo = min(lanes.y, lanes.z), hi = max(lanes.y, lanes.z);
    if (hi - lo < 2.0 || hi - lo > 6.0) {
      lo = road.z - 1.8;
      hi = road.z + 1.8;
    }
    float across = band(p.y, lo - 0.15, hi + 0.15, 0.15);
    float dx = p.x - stopLine.x;
    float fw = max(fwidth(dx), 1e-4);
    float bar = 1.0 - smoothstep(0.22 - fw, 0.22 + fw, abs(dx));
    float halo = exp(-abs(dx) * 1.1) * (0.8 + 0.2 * sin(time * 4.5));
    vec3 red = vec3(1.0, 0.13, 0.1);
    color = mix(color, red, bar * across * stopLine.y);
    color += red * halo * across * stopLine.y * 0.45;
  }

  // shadows under vehicles, their tail light glow at night or when braking, and our engagement colour
  for (int i = 0; i < MAX_BODIES; i++) {
    vec4 pose = bodyPose[i];
    vec4 size = bodySize[i];
    if (size.x <= 0.0) {
      continue;
    }
    vec2 d = p - pose.xy;
    vec2 l = vec2(d.x * pose.z + d.y * pose.w, -d.x * pose.w + d.y * pose.z);
    vec2 q = abs(l) - size.xy;
    float box = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
    color *= 1.0 - (1.0 - smoothstep(-0.25, 0.45, box)) * size.z;
    if (size.w > 0.0) {
      vec2 tail = l - vec2(-size.x - 1.2, 0.0);
      color += TAIL * size.w * exp(-dot(tail, tail * vec2(0.9, 1.6))) * 0.45;
    }
    if (i == 0 && statusGlow.a > 0.0) {
      color += statusGlow.rgb * statusGlow.a * exp(-max(box, 0.0) * 3.0) * smoothstep(-0.4, 0.05, box);
    }
  }
  return color;
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
  vec4 lanes, road, laneSlope;
  float pathSlope;
  roadAt(p.x, lanes, road, laneSlope, pathSlope);
  vec3 color = roadDecals(syntheticGround(p.xy, road), p.xy, lanes, road, laneSlope, pathSlope, true);
  finalColor = vec4(mix(color, skyColor(vec3(1.0, 0.0, 0.0)), fogAmount(t)), 1.0);
}
"""

_CAMERA_GROUND = _GROUND_COMMON + """
uniform vec3 groundToTex[3];  // columns: (X, Y, 1) on the road -> homogeneous texture uv of the real camera
uniform vec4 camLimits;       // x: nearest usable road distance, y/z: distance fade start/end, w: lowest usable v (hood)
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
  // smoothstep edges must increase (reversed edges are undefined in GLSL), so falling edges use 1 - smoothstep
  float w = smoothstep(0.0, 0.1, uv.x) * (1.0 - smoothstep(0.9, 1.0, uv.x)) *
            smoothstep(0.0, 0.06, uv.y) * (1.0 - smoothstep(camLimits.w - 0.04, camLimits.w, uv.y));
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
  vec4 lanes, road, laneSlope;
  float pathSlope;
  roadAt(p.x, lanes, road, laneSlope, pathSlope);
  vec3 color = syntheticGround(p.xy, road);
  vec2 uv;
  float w = cameraWeight(p.xy, uv);
  bool synthetic = true;
  if (w > 0.0) {
    vec3 cam = cameraRgb(uv);
    if (mode == 1) {
      color = mix(color, road_color(cam, colorStrength), w);
      synthetic = w < 0.5;
    } else {
      // Paint is brighter than the road on BOTH sides of it: across for lane lines, along for stop lines and crosswalks.
      float across = max(luma(cameraRgb(texAt(p.xy + vec2(0.0, 0.35)))), luma(cameraRgb(texAt(p.xy - vec2(0.0, 0.35)))));
      float along = max(luma(cameraRgb(texAt(p.xy + vec2(0.6, 0.0)))), luma(cameraRgb(texAt(p.xy - vec2(0.6, 0.0)))));
      float l0 = luma(cam);
      float background = min(across, along);
      float contrast = l0 - background;
      float paint = smoothstep(0.04, 0.10, contrast) * smoothstep(0.12, 0.30, contrast / max(background, 0.05));
      // Only near-white or yellow counts, which rejects grass, foliage and brake lights.
      float high = max(cam.r, max(cam.g, cam.b));
      float saturation = (high - min(cam.r, min(cam.g, cam.b))) / max(high, 1e-3);
      float redGreen = cam.r / max(cam.g, 1e-3);
      float yellow = smoothstep(0.80, 0.95, redGreen) * (1.0 - smoothstep(1.45, 1.75, redGreen)) * smoothstep(0.18, 0.38, saturation);
      float white = 1.0 - smoothstep(0.16, 0.30, saturation);
      float onRoad = band(p.y, road.x - 0.4, road.y + 0.4, 0.4);
      paint *= max(white, yellow) * onRoad;
      color = mix(color, mix(paintWhite.rgb, paintYellow.rgb, yellow), paint * w);
    }
  }
  color = roadDecals(color, p.xy, lanes, road, laneSlope, pathSlope, synthetic);
  finalColor = vec4(mix(color, skyColor(vec3(1.0, 0.0, 0.0)), fogAmount(t)), 1.0);
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


# ---- vehicles

# Model ids (renderer.VEHICLE_MODELS): 0 car, 1 pickup truck, 2 sphere, 3 stop sign, 4 stop sign lying flat (a map marker
# for the top view). Local frame: x forward, y right, z UP, metres, origin at the centre of the footprint on the road
# (the foot of the post for the sign).
VEHICLE_SHADER = VERSION + """
in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;

uniform vec3 uvToRay[3];
uniform vec3 camPos;
uniform vec4 quad;          // this quad within the layer, in layer uv: x, y, w, h
uniform vec4 pose;          // x, y, road z (calib), yaw
uniform int model;
uniform vec4 paint;         // rgb, a = gloss
uniform vec4 lights;        // x: headlights, y: tail (0 off, ~0.4 tail lamps, 1 brake), z/w: left/right indicator;
                            // for the stop sign x is its opacity
uniform vec4 extra;         // x: reversing lights, y: darkness, z: radians per pixel, w: wheel turn (rad)
uniform vec4 sky;           // ambient sky colour for reflections

const vec3 GLASS = vec3(0.035, 0.045, 0.06);
const vec3 TYRE = vec3(0.035, 0.035, 0.04);
const vec3 RIM = vec3(0.55, 0.57, 0.6);
const vec3 LIGHT_DIR = vec3(0.45, -0.35, 0.82);  // from front left, above

float smin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}

float smax(float a, float b, float k) {
  return -smin(-a, -b, k);
}

float roundBox(vec3 p, vec3 b, float r) {
  vec3 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Wheels mirrored to all four corners; returns the distance to the tyres.
float wheels(vec3 p, float base, float track, float radius, float width) {
  vec3 w = vec3(abs(p.x) - base, abs(p.y) - track, p.z - radius);
  vec2 d = vec2(length(w.xz) - radius, abs(w.y) - width);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - 0.03;
}

float wells(vec3 p, float base, float radius, float inner) {
  vec3 w = vec3(abs(p.x) - base, abs(p.y), p.z - radius);
  return max(length(w.xz) - radius - 0.07, inner - w.y);
}

// Sedan, 4.6 x 1.84 x 1.45 m.
float carBody(vec3 p) {
  float body = roundBox(p - vec3(0.0, 0.0, 0.6), vec3(2.3, 0.92, 0.31), 0.22);
  body = smax(body, dot(p - vec3(2.3, 0.0, 0.74), normalize(vec3(0.32, 0.0, 1.0))), 0.18);   // hood drops to the nose
  body = smax(body, dot(p - vec3(-2.3, 0.0, 0.84), normalize(vec3(-0.2, 0.0, 1.0))), 0.14);  // trunk lid
  body = smax(body, dot(vec2(abs(p.y), p.z) - vec2(0.92, 0.5), normalize(vec2(1.0, -0.25))), 0.1);  // tucked lower sides
  body = max(body, -wells(p, 1.38, 0.34, 0.66));
  vec3 c = p - vec3(-0.35, 0.0, 0.0);
  float cabin = smax(c.z - 1.42, 0.78 - c.z, 0.05);
  cabin = smax(cabin, dot(c - vec3(0.72, 0.0, 0.84), normalize(vec3(0.78, 0.0, 0.55))), 0.25);   // raked windshield
  cabin = smax(cabin, dot(c - vec3(-1.15, 0.0, 0.84), normalize(vec3(-0.72, 0.0, 0.58))), 0.25); // fastback rear window
  cabin = smax(cabin, dot(vec2(abs(c.y), c.z) - vec2(0.84, 0.84), normalize(vec2(1.0, 0.24))), 0.12);  // tumblehome
  return smin(body, cabin, 0.07);
}

// Pickup, 5.6 x 2.0 x 1.9 m: hood, tall cab, open bed.
float truckBody(vec3 p) {
  float body = roundBox(p - vec3(0.0, 0.0, 0.82), vec3(2.8, 1.0, 0.36), 0.14);
  body = smax(body, dot(p - vec3(2.8, 0.0, 1.06), normalize(vec3(0.12, 0.0, 1.0))), 0.08);
  body = max(body, -wells(p, 1.8, 0.4, 0.7));
  float bed = roundBox(p - vec3(-1.75, 0.0, 1.0), vec3(1.0, 0.88, 0.4), 0.05);
  body = max(body, -bed);
  vec3 c = p - vec3(0.25, 0.0, 0.0);
  float cab = roundBox(c - vec3(0.0, 0.0, 1.45), vec3(0.85, 0.92, 0.45), 0.12);
  cab = smax(cab, dot(c - vec3(0.78, 0.0, 1.5), normalize(vec3(0.75, 0.0, 0.35))), 0.15);
  cab = smax(cab, dot(vec2(abs(c.y), c.z) - vec2(0.9, 1.2), normalize(vec2(1.0, 0.12))), 0.1);
  return smin(body, cab, 0.06);
}

float octagon(vec2 p, float r) {
  const vec3 k = vec3(-0.9238795325, 0.3826834323, 0.4142135623);
  p = abs(p);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= 2.0 * min(dot(vec2(-k.x, k.y), p), 0.0) * vec2(-k.x, k.y);
  p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
  return length(p) * sign(p.y);
}

// 30 inch (0.76 m) stop sign facing oncoming traffic (-x), centre 2.3 m up, on a post just behind it.
float stopSign(vec3 p) {
  vec3 q = p - vec3(0.0, 0.0, 2.3);
  vec2 d = vec2(octagon(q.yz, 0.381), abs(q.x) - 0.015);
  float plate = min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - 0.004;
  float post = max(length(p.xy - vec2(0.05, 0.0)) - 0.035, max(-p.z, p.z - 2.25));
  return min(plate, post);
}

// Flat marker: the same octagon lying on the road, top edge pointing ahead.
float stopMarker(vec3 p) {
  vec2 d = vec2(octagon(vec2(p.y, p.x), 0.9), abs(p.z - 0.06) - 0.04);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - 0.01;
}

float shape(vec3 p) {
  if (model == 4) {
    return stopMarker(p);
  }
  if (model == 3) {
    return stopSign(p);
  }
  if (model == 2) {
    return length(p - vec3(0.0, 0.0, 0.9)) - 0.9;
  }
  if (model == 1) {
    return min(truckBody(p), wheels(p, 1.8, 0.86, 0.4, 0.14));
  }
  return min(carBody(p), wheels(p, 1.38, 0.8, 0.34, 0.12));
}

vec3 normalAt(vec3 p, float eps) {
  vec2 k = vec2(1.0, -1.0);
  return normalize(k.xyy * shape(p + k.xyy * eps) + k.yyx * shape(p + k.yyx * eps) +
                   k.yxy * shape(p + k.yxy * eps) + k.xxx * shape(p + k.xxx * eps));
}

// Half extents of the model's bounds around (0, 0, h) and that centre height.
vec4 bounds() {
  if (model == 4) return vec4(0.95, 0.95, 0.12, 0.06);
  if (model == 3) return vec4(0.12, 0.44, 1.37, 1.37);
  if (model == 2) return vec4(0.92, 0.92, 0.92, 0.9);
  if (model == 1) return vec4(2.86, 1.08, 0.98, 0.96);
  return vec4(2.36, 1.0, 0.75, 0.73);
}

vec2 boxHit(vec3 ro, vec3 rd, vec3 half_) {
  vec3 inv = 1.0 / mix(rd, vec3(1e-6), step(abs(rd), vec3(1e-6)));
  vec3 t0 = (-half_ - ro) * inv, t1 = (half_ - ro) * inv;
  vec3 lo = min(t0, t1), hi = max(t0, t1);
  return vec2(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

struct Lamp {
  vec3 pos;      // local
  vec3 facing;   // outward normal
  vec3 color;
  float level;
};

float lampLevel(int i) {
  if (model >= 3) {
    return 0.0;
  }
  if (model == 2) {
    return i < 2 ? max(lights.y, 0.0) : 0.0;
  }
  if (i < 2) return lights.y;                                 // tail / brake
  if (i < 4) return (i == 2 ? lights.z : lights.w);           // rear indicators
  if (i < 6) return (i == 4 ? lights.z : lights.w);           // front indicators
  if (i < 8) return lights.x;                                 // headlights
  return extra.x;                                              // reversing
}

Lamp lamp(int i) {
  float side = (i % 2 == 0) ? -1.0 : 1.0;  // left lamps first
  bool truck = model == 1;
  float rear = truck ? -2.8 : -2.3, front = truck ? 2.8 : 2.3, y = truck ? 0.86 : 0.72;
  Lamp l;
  l.facing = vec3(-1.0, 0.0, 0.0);
  l.color = vec3(1.0, 0.08, 0.05);
  if (model == 2) {
    l.pos = vec3(-0.78, side * 0.34, 1.05);
  } else if (i < 2) {
    l.pos = vec3(rear, side * y, truck ? 1.0 : 0.76);
  } else if (i < 4) {
    l.pos = vec3(rear, (i == 2 ? -1.0 : 1.0) * (y + 0.1), truck ? 0.82 : 0.62);
    l.color = vec3(1.0, 0.55, 0.05);
  } else if (i < 6) {
    l.pos = vec3(front, (i == 4 ? -1.0 : 1.0) * (y + 0.12), truck ? 0.92 : 0.68);
    l.facing = vec3(1.0, 0.0, 0.0);
    l.color = vec3(1.0, 0.55, 0.05);
  } else if (i < 8) {
    l.pos = vec3(front, side * (y - 0.06), truck ? 1.0 : 0.74);
    l.facing = vec3(1.0, 0.0, 0.0);
    l.color = vec3(1.0, 0.96, 0.86);
  } else {
    l.pos = vec3(rear, side * (y - 0.32), truck ? 0.9 : 0.7);
    l.color = vec3(1.0, 1.0, 0.95);
  }
  l.level = lampLevel(i);
  return l;
}

// Surface colour and emission at a hit. n is the local normal.
vec3 surface(vec3 p, vec3 n, vec3 rd, out float gloss) {
  gloss = paint.a;
  vec3 albedo = paint.rgb;
  bool truck = model == 1;
  if (model >= 3) {
    gloss = 0.25;
    // face coordinates as the driver sees them (x right, y up) and the octagon's half size
    bool lying = model == 4;
    vec2 f = lying ? vec2(p.y, p.x) : vec2(p.y, p.z - 2.3);
    float r = lying ? 0.9 : 0.381;
    float face = octagon(f, r);
    bool front = lying ? n.z > 0.5 : n.x < -0.3;
    if (face > 0.01 || !front) {
      return vec3(0.5, 0.52, 0.55);  // post, edges and the back of the plate
    }
    float glow = 1.0 + 0.7 * extra.y;  // retroreflective: brighter at night, as if in our headlights
    vec2 g = f / r;
    if (face > -0.09 * r || (abs(g.y) < 0.22 && abs(g.x) < 0.7 && fract((g.x + 0.7) / 0.35) < 0.78)) {
      return vec3(0.93) * glow;  // white border and four blocks standing in for STOP
    }
    return vec3(0.76, 0.05, 0.06) * glow;
  }
  if (model == 2) {
    return albedo;
  }
  float base = truck ? 1.8 : 1.38, track = truck ? 0.86 : 0.8, radius = truck ? 0.4 : 0.34;
  vec3 w = vec3(abs(p.x) - base, abs(p.y) - track, p.z - radius);
  if (length(w.xz) < radius + 0.02 && abs(w.y) < (truck ? 0.16 : 0.14)) {
    gloss = 0.15;
    // rims on the outer face, with five spokes that turn as we drive
    float r = length(w.xz) / radius;
    float a = atan(w.z, w.x * sign(p.x) + 1e-5) + extra.w * sign(p.y);
    float spokes = smoothstep(0.55, 0.75, abs(cos(a * 2.5)));
    float rim = (1.0 - smoothstep(0.62, 0.68, r)) * step(0.0, w.y) * mix(0.55, 1.0, spokes);
    return mix(TYRE, RIM, rim);
  }
  float beltline = truck ? 1.12 : 0.88;
  // glass: the cabin sides, windshield and rear window above the belt line
  float cabinTop = truck ? 1.83 : 1.38;
  if (p.z > beltline + 0.04 && p.z < cabinTop && abs(n.z) < 0.8) {
    gloss = 1.0;
    return GLASS;
  }
  // lamps
  for (int i = 0; i < 10; i++) {
    Lamp l = lamp(i);
    if (dot(n, l.facing) < 0.5) {
      continue;
    }
    vec2 d = abs(p.yz - l.pos.yz);
    float lens = (i < 2 || i >= 8) ? 0.16 : (i < 6 ? 0.1 : 0.13);
    if (d.x < lens && d.y < 0.06 + (truck && i < 2 ? 0.12 : 0.0)) {
      gloss = 0.8;
      vec3 off = l.color * (i >= 8 ? 0.16 : i >= 6 ? 0.5 : 0.32);
      return mix(off, l.color * 2.2, clamp(l.level, 0.0, 1.0));
    }
  }
  // dark grille and bumper insert at the nose
  if (n.x > 0.6 && abs(p.y) < (truck ? 0.6 : 0.45) && p.z > 0.38 && p.z < (truck ? 0.95 : 0.6)) {
    return albedo * 0.18;
  }
  // black trim along the bottom
  if (p.z < (truck ? 0.5 : 0.34)) {
    gloss = 0.1;
    return vec3(0.05);
  }
  return albedo;
}

void main() {
  vec2 uv = quad.xy + fragTexCoord * quad.zw;
  vec3 dir = normalize(mat3(uvToRay[0], uvToRay[1], uvToRay[2]) * vec3(uv, 1.0));
  // into the vehicle frame: x forward, y right, z up
  float c = cos(pose.w), s = sin(pose.w);
  vec3 o = camPos - pose.xyz;
  vec3 ro = vec3(o.x * c + o.y * s, -o.x * s + o.y * c, -o.z);
  vec3 rd = vec3(dir.x * c + dir.y * s, -dir.x * s + dir.y * c, -dir.z);

  vec4 b = bounds();
  vec3 center = vec3(0.0, 0.0, b.w);
  vec2 span = boxHit(ro - center, rd, b.xyz);
  float pixel = extra.z;
  vec4 result = vec4(0.0);

  if (span.y > max(span.x, 0.0)) {
    float t = max(span.x, 0.0);
    float best = 1e9, bestT = t;
    bool hit = false;
    for (int i = 0; i < 72; i++) {
      float d = shape(ro + rd * t);
      float ratio = d / (pixel * t);
      if (ratio < best) {
        best = ratio;
        bestT = t;
      }
      if (d < 0.0008 * t) {
        hit = true;
        break;
      }
      t += d * 0.9;
      if (t > span.y) {
        break;
      }
    }
    float coverage = hit ? 1.0 : clamp(1.0 - best, 0.0, 1.0);
    if (coverage > 0.0) {
      vec3 p = ro + rd * (hit ? t : bestT);
      vec3 n = normalAt(p, clamp(0.0015 * bestT, 0.002, 0.03));
      float gloss;
      vec3 albedo = surface(p, n, rd, gloss);
      float dark = extra.y;
      vec3 l = normalize(LIGHT_DIR);
      float diffuse = max(dot(n, l), 0.0);
      float ao = clamp(0.35 + 0.65 * shape(p + n * 0.18) / 0.18, 0.0, 1.0) * clamp(0.5 + p.z * 1.2, 0.0, 1.0);
      float ambient = 0.5 + 0.5 * n.z;
      float sun = mix(1.0, 0.22, dark), fill = mix(1.0, 0.45, dark);
      vec3 color = albedo * (vec3(0.30, 0.33, 0.38) * ambient * ao * fill + vec3(1.0, 0.97, 0.92) * diffuse * 0.95 * sun);
      vec3 h = normalize(l - rd);
      float spec = pow(max(dot(n, h), 0.0), 60.0) * gloss * sun;
      float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 4.0);
      vec3 reflected = mix(sky.rgb * 0.6, sky.rgb * 1.3, clamp(reflect(rd, n).z * 2.0, 0.0, 1.0)) * fill;
      color += vec3(spec) * 0.7 + reflected * (0.08 + 0.55 * fresnel) * gloss * ao;
      // lamps already carry their emission in the albedo; keep them bright regardless of lighting
      if (max(albedo.r, max(albedo.g, albedo.b)) > 1.0) {
        color = albedo;
      }
      result = vec4(color, coverage);
    }
  }

  // glow around lit lamps so they read at a distance (at least a few pixels wide)
  for (int i = 0; i < 10; i++) {
    Lamp l = lamp(i);
    if (l.level <= 0.02 || dot(rd, l.facing) > -0.05) {
      continue;
    }
    vec3 rel = l.pos - ro;
    float along = dot(rel, rd);
    if (along <= 0.0) {
      continue;
    }
    float miss = length(rel - rd * along);
    float radius = max(0.22, pixel * along * 4.0) * (i >= 6 ? 1.6 : (i >= 2 ? 1.5 : 1.0));
    float g = exp(-miss * miss / (radius * radius)) * l.level * (0.55 + 0.45 * extra.y);
    result.rgb = mix(result.rgb, l.color, g * (1.0 - result.a * 0.5));
    result.a = max(result.a, g);
  }
  if (model >= 3) {
    result.a *= lights.x;
  }
  if (result.a <= 0.003) {
    discard;
  }
  finalColor = result;
}
"""
