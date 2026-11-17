#version 330
#pragma include "common.glsl"
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform struct p3d_MaterialParameters { vec4 baseColor; float roughness; } p3d_Material;
uniform int mode;            // 0 solid, 1 foliage (alpha cut), 2 creature (back / belly colouring), 3 grass cards
uniform vec3 back_color;
uniform vec3 belly_color;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_view;
in vec4 v_color;
out vec4 o_color;

void main() {
    vec4 tex = texture(p3d_Texture0, v_uv);
    vec4 base = tex * p3d_Material.baseColor * p3d_ColorScale;
    if (mode == 3) base = vec4(tex.rgb * v_color.rgb, tex.a);
    if ((mode == 1 || mode == 3) && base.a < 0.45) discard;
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing && mode != 3) n = -n;
    vec3 albedo = to_linear(base.rgb);
    float rough = clamp(p3d_Material.roughness, 0.25, 1.0);
    float wrap = mode == 1 || mode == 3 ? 0.5 : 0.1;
    if (mode == 2) {
        float up = smoothstep(-0.5, 0.7, n.z);
        albedo = to_linear(mix(belly_color, back_color, up)) * (0.55 + 0.45 * base.rgb / max(p3d_Material.baseColor.rgb, vec3(0.05)));
        rough = 0.45;
        wrap = 0.25;
    }
    vec3 c = light(albedo, n, v_world, v_view, rough, wrap);
    o_color = vec4(finish(c, v_world), 1.0);
}
