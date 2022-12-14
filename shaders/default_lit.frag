#version 450

// Shader input
layout (location = 0) in vec3 inColor;

// Output write
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform  SceneData{
    vec4 fogColor; // w is for exponent
	vec4 fogDistances; //x for min, y for max, zw unused.
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
} sceneData;

void main()
{
    // Return color
    outFragColor = vec4(inColor + sceneData.ambientColor.xzy, 1.0f);
}