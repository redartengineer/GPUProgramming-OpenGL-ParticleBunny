#version 430 core

layout(location = 0) in vec2 aPosition;

uniform vec2 uScreenSize;

void main()
{
    float x =
        (aPosition.x /
         uScreenSize.x) *
        2.0 -
        1.0;

    float y =
        1.0 -
        (aPosition.y /
         uScreenSize.y) *
        2.0;

    gl_Position =
        vec4(
            x,
            y,
            0.0,
            1.0);
}