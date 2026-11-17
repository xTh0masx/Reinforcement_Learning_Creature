#version 330
#pragma include "common.glsl"
uniform sampler2D grass_tex;
uniform sampler2D dirt_tex;
uniform sampler2D sand_tex;
uniform sampler2D rock_tex;
uniform sampler2D macro_tex;     // large-scale brightness variation (breaks the tiling)
in vec3 v_world;
in vec3 v_normal;
in vec4 v_view;
in vec4 v_splat;
out vec4 o_color;

vec3 two_scale(sampler2D t, vec2 p) {
    return mix(texture(t, p / 3.0).rgb, texture(t, p / 11.0).rgb, 0.35);
}

void main() {
    vec3 n = normalize(v_normal);
    vec2 p = v_world.xy;
    vec4 w = v_splat;
    // rock: tri-planar (the cliffs are steep)
    vec3 bw = pow(abs(n), vec3(4.0));
    bw /= (bw.x + bw.y + bw.z);
    vec3 rock = texture(rock_tex, v_world.yz / 6.0).rgb * bw.x + texture(rock_tex, v_world.xz / 6.0).rgb * bw.y
              + texture(rock_tex, v_world.xy / 6.0).rgb * bw.z;
    // sharper transitions where one layer's texture is brighter (height-blend look)
    vec3 g = two_scale(grass_tex, p) * vec3(0.78, 1.0, 0.62), d = two_scale(dirt_tex, p), s = two_scale(sand_tex, p) * vec3(0.86, 0.72, 0.56);
    vec4 hb = w * (vec4(dot(g, vec3(0.33)), dot(d, vec3(0.33)), dot(s, vec3(0.33)), dot(rock, vec3(0.33))) + 0.25);
    float mx = max(max(hb.r, hb.g), max(hb.b, hb.a));
    hb = max(hb - (mx - 0.18), 0.0);
    hb /= max(hb.r + hb.g + hb.b + hb.a, 1e-4);
    vec3 albedo = g * hb.r + d * hb.g + s * hb.b + rock * hb.a;
    float macro = texture(macro_tex, p / 90.0).r;
    albedo *= 0.8 + 0.4 * macro;
    vec3 c = light(to_linear(albedo), n, v_world, v_view, 0.9, 0.0);
    o_color = vec4(finish(c, v_world), 1.0);
}
