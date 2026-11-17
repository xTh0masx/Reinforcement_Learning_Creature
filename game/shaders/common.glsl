// Shared lighting for every lit surface: sun with soft shadow (Panda's shadow map), sky / ground ambient,
// distance fog, ACES tone mapping. Colours are lit in linear space.
uniform struct p3d_LightSourceParameters {
    vec4 color;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];

uniform vec3 sun_dir;        // world, towards the sun
uniform vec3 sun_color;
uniform vec3 sky_ambient;
uniform vec3 ground_ambient;
uniform vec3 fog_color;
uniform float fog_density;
uniform vec3 cam_pos;

vec3 to_linear(vec3 c) { return pow(c, vec3(2.2)); }

float shadow_at(vec4 view_pos, vec3 n) {
    vec4 sc = p3d_LightSource[0].shadowViewMatrix * view_pos;
    if (sc.w <= 0.0) return 1.0;
    vec3 p = sc.xyz / sc.w;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 1.0;
    float bias = 0.0006 + 0.0012 * (1.0 - clamp(dot(n, sun_dir), 0.0, 1.0));
    float s = 0.0;
    float texel = 1.0 / 4096.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            s += texture(p3d_LightSource[0].shadowMap, vec3(p.xy + vec2(i, j) * texel * 1.2, p.z - bias));
    return s / 9.0;
}

vec3 light(vec3 albedo_lin, vec3 n, vec3 world_pos, vec4 view_pos, float rough, float wrap) {
    float ndl = dot(n, sun_dir);
    float diff = clamp((ndl + wrap) / (1.0 + wrap), 0.0, 1.0);
    float sh = shadow_at(view_pos, n);
    vec3 v = normalize(cam_pos - world_pos);
    vec3 h = normalize(v + sun_dir);
    float spec = pow(clamp(dot(n, h), 0.0, 1.0), mix(80.0, 6.0, rough)) * (1.0 - rough) * 0.6;
    vec3 amb = mix(ground_ambient, sky_ambient, n.z * 0.5 + 0.5);
    return albedo_lin * (amb + sun_color * diff * sh) + sun_color * spec * sh * step(0.0, ndl);
}

vec3 finish(vec3 lin, vec3 world_pos) {
    // ACES (Narkowicz), then fog in display space so far terrain melts exactly into the sky's horizon colour
    vec3 x = lin * 1.1;
    vec3 c = pow(clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0), vec3(1.0 / 2.2));
    float d = length(world_pos - cam_pos);
    float f = 1.0 - exp(-pow(d * fog_density, 1.6));
    f *= clamp(1.2 - (world_pos.z - 10.0) / 120.0, 0.35, 1.0);       // thinner fog up on the mountains
    return mix(c, fog_color, f);
}
