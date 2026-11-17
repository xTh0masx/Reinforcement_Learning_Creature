#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
out vec3 v_world;
out vec3 v_normal;
out vec4 v_view;
out vec4 v_splat;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_view = p3d_ModelViewMatrix * p3d_Vertex;
    v_world = p3d_Vertex.xyz;        // the terrain sits at the world origin
    v_normal = p3d_Normal;
    v_splat = p3d_Color;             // weights: r grass, g dirt, b sand, a rock
}
