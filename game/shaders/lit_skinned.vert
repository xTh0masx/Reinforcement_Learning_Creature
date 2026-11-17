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
uniform mat4 p3d_TransformTable[64];
in vec4 transform_weight;
in uvec4 transform_index;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_view;
out vec4 v_color;

void main() {
    mat4 skin = p3d_TransformTable[transform_index.x] * transform_weight.x
              + p3d_TransformTable[transform_index.y] * transform_weight.y
              + p3d_TransformTable[transform_index.z] * transform_weight.z
              + p3d_TransformTable[transform_index.w] * transform_weight.w;
    vec4 vtx = skin * p3d_Vertex;
    vec3 nrm = mat3(skin) * p3d_Normal;
    vec4 wp = p3d_ModelMatrix * vtx;
    // hardware skinning (the creatures): Panda fills the bone matrices
    gl_Position = p3d_ModelViewProjectionMatrix * vtx;
    v_view = p3d_ModelViewMatrix * vtx;
    v_world = wp.xyz;
    v_normal = normalize(mat3(p3d_ModelMatrix) * nrm);
    v_uv = p3d_MultiTexCoord0;
    v_color = p3d_Color;
}
