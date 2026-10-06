#version 430 core

out vec4 FragColor;

void main()
{
    vec3 furColor =
        vec3(
            0.96,
            0.91,
            0.80);

    FragColor =
        vec4(
            furColor,
            0.78);
}