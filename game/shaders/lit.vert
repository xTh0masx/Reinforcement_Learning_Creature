#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform float wind_time;
uniform float wind_strength;     // 0 = solid, >0 foliage sways (top of the model more)
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_view;
out vec4 v_color;

void main() {
    vec4 wp = p3d_ModelMatrix * p3d_Vertex;
    if (wind_strength > 0.0) {
        float h = max(p3d_Vertex.z, 0.0);
        float ph = wind_time * 1.7 + wp.x * 0.35 + wp.y * 0.27;
        wp.xy += wind_strength * h * h * vec2(sin(ph), cos(ph * 0.83)) * 0.04;
    }
    vec4 lp = inverse(p3d_ModelMatrix) * wp;
    gl_Position = p3d_ModelViewProjectionMatrix * lp;
    v_view = p3d_ModelViewMatrix * lp;
    v_world = wp.xyz;
    v_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    v_uv = p3d_MultiTexCoord0;
    v_color = p3d_Color;
}
