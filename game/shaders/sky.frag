#version 330
uniform sampler2D sky_tex;
uniform vec3 fog_color;
in vec3 v_dir;
out vec4 o_color;

void main() {
    vec3 d = normalize(v_dir);
    vec2 uv = vec2(atan(d.y, d.x) / 6.2831853 + 0.5, asin(clamp(d.z, -1.0, 1.0)) / 3.1415927 + 0.5);
    vec3 c = textureLod(sky_tex, uv, 0.0).rgb;
    float haze = 1.0 - smoothstep(0.0, 0.18, d.z);     // the horizon fades into the fog colour
    o_color = vec4(mix(c, fog_color, haze * 0.85), 1.0);
}
