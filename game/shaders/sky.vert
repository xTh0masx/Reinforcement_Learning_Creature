#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
out vec3 v_dir;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_Position.z = gl_Position.w * 0.99999;        // always at the far plane
    v_dir = p3d_Vertex.xyz;                          // the dome is centred on the camera
}
