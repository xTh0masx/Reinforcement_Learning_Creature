#version 330
#pragma include "common.glsl"
uniform float wind_time;
uniform sampler2D sky_tex;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_view;
in vec4 v_color;
out vec4 o_color;

vec2 equirect(vec3 d) {
    return vec2(atan(d.y, d.x) / 6.2831853 + 0.5, asin(clamp(d.z, -1.0, 1.0)) / 3.1415927 + 0.5);
}

void main() {
    vec2 p = v_world.xy;
    float t = wind_time;
    vec2 g = vec2(0.0);
    g += 0.06 * cos(dot(p, vec2(1.3, 0.7)) * 2.1 + t * 1.6) * vec2(1.3, 0.7);
    g += 0.04 * cos(dot(p, vec2(-0.6, 1.1)) * 3.3 + t * 2.1) * vec2(-0.6, 1.1);
    g += 0.025 * cos(dot(p, vec2(0.9, -1.4)) * 5.7 + t * 2.9) * vec2(0.9, -1.4);
    vec3 n = normalize(vec3(-g, 1.0));
    vec3 v = normalize(cam_pos - v_world);
    vec3 r = reflect(-v, n);
    r.z = abs(r.z);
    vec3 sky = to_linear(texture(sky_tex, equirect(r)).rgb);
    float fres = 0.03 + 0.5 * pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 5.0);
    float edge = v_color.r;                         // 1 at the centre, 0 at the shore
    vec3 deep = to_linear(vec3(0.05, 0.16, 0.17)) * (sky_ambient + sun_color * 0.4);
    float sh = shadow_at(v_view, n);
    vec3 c = mix(deep, sky * 0.75, fres);
    vec3 h = normalize(v + sun_dir);
    c += sun_color * pow(clamp(dot(n, h), 0.0, 1.0), 220.0) * 3.0 * sh;
    float a = mix(0.35, 0.92, smoothstep(0.0, 0.5, edge)) * mix(0.85, 1.0, fres);
    o_color = vec4(finish(c, v_world), a);
}
