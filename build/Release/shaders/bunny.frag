#version 430 core

in vec3 vNormal;

out vec4 FragColor;

void main()
{
    vec3 normal =
        normalize(vNormal);

    vec3 lightDirection =
        normalize(
            vec3(
                -0.4,
                0.8,
                0.6));

    float diffuse =
        max(
            dot(
                normal,
                lightDirection),
            0.0);

    vec3 creamWhite =
        vec3(
            0.93,
            0.88,
            0.76);

    float lighting =
        0.35 +
        diffuse * 0.65;

    FragColor =
        vec4(
            creamWhite *
                lighting,
            1.0);
}